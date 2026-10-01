import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyrealsense2 as rs
from pymodbus.client import ModbusTcpClient
import time
import threading
import os
import sys

# --- Configuration ---
# Default to Right hand
SIDE = "right"
HEADLESS = False
BOTH = False

for arg in sys.argv[1:]:
    if arg.lower() == "left":
        SIDE = "left"
    elif arg.lower() == "--headless":
        HEADLESS = True
    elif arg.lower() == "--both":
        BOTH = True

# IP Mappings (Physical: left is .211, right is .210)
IP_LEFT_HAND = "192.168.123.211"
IP_RIGHT_HAND = "192.168.123.210"

if BOTH:
    print(f"Configured for BOTH hands.")
    print(f"  MediaPipe 'Left'  (Physical Right) -> {IP_RIGHT_HAND}")
    print(f"  MediaPipe 'Right' (Physical Left)  -> {IP_LEFT_HAND}")
elif SIDE == "left":
    IP = IP_LEFT_HAND
    TARGET_HAND_LABEL = "Right" # Selfie view: Physical Left is often labeled "Right"
    print(f"Configured for LEFT hand at {IP}. Search for MediaPipe '{TARGET_HAND_LABEL}'")
else:
    IP = IP_RIGHT_HAND
    TARGET_HAND_LABEL = "Left" # Selfie view: Physical Right is often labeled "Left"
    print(f"Configured for RIGHT hand at {IP}. Search for MediaPipe '{TARGET_HAND_LABEL}'")

PORT = 6000
SLAVE_ID = 1
ANGLE_SET_BASE = 1486 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Shared state for both hands
current_cmds = {
    "Left": [0, 0, 0, 0, 200, 1000],
    "Right": [0, 0, 0, 0, 200, 1000]
}
running = True

def modbus_worker(ip, hand_label):
    global current_cmds, running
    # Added timeout to prevent hanging if a hand is offline
    client = ModbusTcpClient(ip, port=PORT, timeout=1.0)
    if not client.connect():
        print(f"Modbus connection failed to {ip} ({hand_label})")
        return
    
    print(f"Modbus thread started for {hand_label} hand at {ip}.")
    last_cmd = None
    
    while running:
        try:
            cmd = current_cmds[hand_label]
            if cmd != last_cmd:
                # Try multiple common keyword names for the slave/unit ID
                try:
                    client.write_registers(ANGLE_SET_BASE, cmd, slave=SLAVE_ID)
                except TypeError:
                    try:
                        client.write_registers(ANGLE_SET_BASE, cmd, unit=SLAVE_ID)
                    except TypeError:
                        client.write_registers(ANGLE_SET_BASE, cmd)
                last_cmd = list(cmd)
            time.sleep(0.04) # ~25Hz
        except Exception as e:
            print(f"Modbus Error ({hand_label}): {e}")
            break
    
    # Reset to neutral on exit
    try:
        neutral = [0, 0, 0, 0, 200, 1000]
        try:
            client.write_registers(ANGLE_SET_BASE, neutral, slave=SLAVE_ID)
        except TypeError:
            try:
                client.write_registers(ANGLE_SET_BASE, neutral, unit=SLAVE_ID)
            except TypeError:
                client.write_registers(ANGLE_SET_BASE, neutral)
    except:
        pass
    client.close()
    print(f"Modbus thread stopped for {hand_label} hand.")

def get_3d_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_angle(p1, p2, p3):
    """Calculate angle at p2 between p1-p2 and p3-p2"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 180.0
    v1 /= v1_norm
    v2 /= v2_norm
    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    return angle

def main():
    global current_cmds, running
    
    # Initialize MediaPipe Landmarker
    print("Initializing MediaPipe...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2 if BOTH else 1)
    detector = vision.HandLandmarker.create_from_options(options)
    print("MediaPipe initialized.")

    # Initialize RealSense
    print("Initializing RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        profile = pipeline.start(config)
        print("RealSense pipeline started.")
    except Exception as e:
        print(f"Failed to start RealSense pipeline: {e}")
        running = False
        return

    # Initialize Modbus Threads AFTER RealSense to avoid long timeouts blocking startup
    print("Connecting to Modbus...")
    if BOTH:
        # MP 'Left' (Physical Right) -> IP_RIGHT_HAND (.210)
        # MP 'Right' (Physical Left) -> IP_LEFT_HAND (.211)
        t_left_mp = threading.Thread(target=modbus_worker, args=(IP_RIGHT_HAND, "Left"), daemon=True)
        t_right_mp = threading.Thread(target=modbus_worker, args=(IP_LEFT_HAND, "Right"), daemon=True)
        t_left_mp.start()
        t_right_mp.start()
    else:
        # Single hand mode: MP label for this thread will be 'Left' or 'Right'
        # but the main loop will map the detected hand to this key.
        label = "Left" if SIDE == "left" else "Right"
        t = threading.Thread(target=modbus_worker, args=(IP, label), daemon=True)
        t.start()

    # Post-processing filters for better depth quality
    # We can tune these for lower latency
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.holes_fill, 2)
    
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    align_to = rs.stream.color
    align = rs.align(align_to)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    alpha = 0.5 
    smoothed_vals = {
        "Left": np.array([0, 0, 0, 0, 200, 1000], dtype=float),
        "Right": np.array([0, 0, 0, 0, 200, 1000], dtype=float)
    }

    print("RealSense Mimic started. Press 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # Apply filters to depth frame
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)
            
            # Cast to depth_frame to access get_distance
            depth_frame = depth_frame.as_depth_frame()
            if not depth_frame:
                continue
            
            # Optimization: Convert depth to numpy array for faster sampling
            depth_data = np.asanyarray(depth_frame.get_data())
            
            image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            detection_result = detector.detect(mp_image)

            if detection_result.hand_landmarks:
                for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    detected_label = detection_result.handedness[i][0].category_name
                    score = detection_result.handedness[i][0].score
                    
                    if score < 0.7: continue
                    
                    # Map the detected hand to the correct target
                    target_label = detected_label
                    if not BOTH:
                        # In single hand mode, we force the detected hand to control the selected side
                        # and we are permissive about which hand label we see
                        target_label = "Left" if SIDE == "left" else "Right"
                        # Only use the hand if it matches the target label for that side
                        if detected_label != TARGET_HAND_LABEL:
                            continue

                    lm = hand_landmarks
                    
                    # 1. Convert 2D landmarks to 3D points using Depth
                    points_3d = []
                    for j in range(21):
                        px = int(lm[j].x * image.shape[1])
                        py = int(lm[j].y * image.shape[0])
                        
                        # Clamp and handle edge cases
                        px = max(0, min(image.shape[1]-1, px))
                        py = max(0, min(image.shape[0]-1, py))
                        
                        # Sample depth with a small window for robustness
                        depth = 0
                        count = 0
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                sx, sy = px + dx, py + dy
                                # Ensure sampling coordinates are within the 640x480 frame
                                if 0 <= sx < 640 and 0 <= sy < 480:
                                    # Use optimized numpy access
                                    d_raw = depth_data[sy, sx]
                                    if d_raw > 0:
                                        depth += d_raw * depth_scale
                                        count += 1
                        if count > 0:
                            depth /= count
                        
                        # Deproject to 3D point (meters)
                        pt = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth)
                        points_3d.append(pt)

                    # 2. Calculate finger flexion using 3D distances
                    raw_vals = [0, 0, 0, 0, 200, 1000]
                    def calc_flex_3d(tip_idx, mcp_idx, wrist_idx, min_r=0.6, max_r=1.1):
                        d_wrist_tip = get_3d_distance(points_3d[wrist_idx], points_3d[tip_idx])
                        d_wrist_mcp = get_3d_distance(points_3d[wrist_idx], points_3d[mcp_idx])
                        ratio = d_wrist_tip / (d_wrist_mcp + 1e-6)
                        val = int((ratio - min_r) * (1000 / (max_r - min_r)))
                        return max(0, min(1000, val))

                    raw_vals[0] = calc_flex_3d(20, 17, 0, min_r=0.6, max_r=1.0) 
                    raw_vals[1] = calc_flex_3d(16, 13, 0, min_r=0.6, max_r=1.1) 
                    raw_vals[2] = calc_flex_3d(12, 9, 0,  min_r=0.6, max_r=1.2)
                    raw_vals[3] = calc_flex_3d(8, 5, 0,   min_r=0.6, max_r=1.2)
                    
                    thumb_flex_angle = get_angle(points_3d[4], points_3d[3], points_3d[2])
                    raw_vals[4] = int(np.interp(thumb_flex_angle, [140.0, 175.0], [0, 1000]))
                    angle_rot = get_angle(points_3d[2], points_3d[0], points_3d[5])
                    raw_vals[5] = int(np.interp(angle_rot, [25.0, 55.0], [0, 1000]))

                    # Smoothing and Update for this specific hand
                    smoothed_vals[target_label] = alpha * np.array(raw_vals) + (1 - alpha) * smoothed_vals[target_label]
                    current_cmds[target_label] = [int(v) for v in smoothed_vals[target_label]]

                    # Draw landmarks for visualization
                    if not HEADLESS:
                        connections = [
                            (0,1), (1,2), (2,3), (3,4), # Thumb
                            (0,5), (5,6), (6,7), (7,8), # Index
                            (9,10), (10,11), (11,12),   # Middle
                            (13,14), (14,15), (15,16), # Ring
                            (0,17), (17,18), (18,19), (19,20), # Pinky
                            (5,9), (9,13), (13,17) # Palm
                        ]
                        pixels = []
                        for pt_3d in points_3d:
                            pixel = rs.rs2_project_point_to_pixel(intrinsics, pt_3d)
                            px_x, px_y = int(pixel[0]), int(pixel[1])
                            # Clamp and check for invalid coordinates
                            px_x = max(0, min(image.shape[1]-1, px_x))
                            px_y = max(0, min(image.shape[0]-1, px_y))
                            pixels.append((px_x, px_y))
                            cv2.circle(image, pixels[-1], 3, (0, 255, 0), -1)
                        for start, end in connections:
                            cv2.line(image, pixels[start], pixels[end], (255, 0, 0), 1)
                        cv2.putText(image, f"{target_label}", pixels[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display
            if not HEADLESS:
                cv2.putText(image, f"L: {current_cmds['Left']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(image, f"R: {current_cmds['Right']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('RealSense Hand Mimic', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if int(time.time() * 10) % 50 == 0:
                    print(f"L: {current_cmds['Left']} | R: {current_cmds['Right']}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        running = False
        pipeline.stop()
        if not HEADLESS:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
