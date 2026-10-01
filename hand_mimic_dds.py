import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyrealsense2 as rs
import time
import threading
import os
import sys

# Unitree SDK2 / DDS Imports
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from inspire_sdkpy import inspire_hand_defaut, inspire_dds

# --- Configuration ---
SIDE = "right"
HEADLESS = False
BOTH = False
NETWORK_INTERFACE = None

# Parse arguments
for arg in sys.argv[1:]:
    if arg.lower() == "left":
        SIDE = "left"
    elif arg.lower() == "--headless":
        HEADLESS = True
    elif arg.lower() == "--both":
        BOTH = True
    elif not arg.startswith("-"):
        NETWORK_INTERFACE = arg

# Initialize DDS
if NETWORK_INTERFACE is not None:
    print(f"Initializing DDS on interface: {NETWORK_INTERFACE}")
    ChannelFactoryInitialize(0, NETWORK_INTERFACE)
else:
    ChannelFactoryInitialize(0)

# Shared state for both hands
current_cmds = {
    "Left": [0, 0, 0, 0, 200, 1000],
    "Right": [0, 0, 0, 0, 200, 1000]
}
running = True

def dds_worker(hand_mp_label, pub):
    """
    hand_mp_label: "Left" or "Right" (The label MediaPipe gives the hand)
    pub: Initialized ChannelPublisher object
    """
    global current_cmds, running
    
    # Map MediaPipe label to DDS topic suffix (for logging)
    topic_suffix = "r" if hand_mp_label == "Left" else "l"
    topic = f"rt/inspire_hand/ctrl/{topic_suffix}"
    
    print(f"DDS Worker active: MediaPipe '{hand_mp_label}' -> Topic '{topic}'")
    last_cmd_list = None
    
    while running:
        try:
            cmd_list = current_cmds[hand_mp_label]
            if cmd_list != last_cmd_list:
                msg = inspire_hand_defaut.get_inspire_hand_ctrl()
                msg.angle_set = [int(v) for v in cmd_list]
                msg.mode = 0b0001 # Angle mode
                if pub.Write(msg):
                    # print(f"DDS Write success for {hand_mp_label}")
                    pass
                else:
                    print(f"DDS Write FAILED for {hand_mp_label}")
                last_cmd_list = list(cmd_list)
            time.sleep(0.04) # ~25Hz
        except Exception as e:
            print(f"DDS Error ({hand_mp_label}): {e}")
            break
            
    # Reset to neutral on exit
    try:
        msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        msg.angle_set = [0, 0, 0, 0, 200, 1000]
        msg.mode = 0b0001
        pub.Write(msg)
    except:
        pass
    print(f"DDS Publisher stopped for {hand_mp_label} hand.")

def get_3d_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_angle(p1, p2, p3):
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
    
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

    # Initialize MediaPipe Landmarker
    print("Initializing MediaPipe...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2 if BOTH else 1)
    detector = vision.HandLandmarker.create_from_options(options)

    # Initialize RealSense
    print("Initializing RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"Failed to start RealSense pipeline: {e}")
        return

    # Post-processing filters
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

    # Initialize DDS Workers
    print("Starting DDS workers...")
    time.sleep(1.0) # Give DDS time to settle
    if BOTH:
        # Initialize publishers sequentially in the main thread to avoid race conditions in CycloneDDS
        print("Initializing Right Hand Publisher (Topic: rt/inspire_hand/ctrl/r)...")
        pub_r = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_dds.inspire_hand_ctrl)
        pub_r.Init()
        
        print("Initializing Left Hand Publisher (Topic: rt/inspire_hand/ctrl/l)...")
        pub_l = ChannelPublisher("rt/inspire_hand/ctrl/l", inspire_dds.inspire_hand_ctrl)
        pub_l.Init()
        
        threading.Thread(target=dds_worker, args=("Left", pub_r), daemon=True).start()
        threading.Thread(target=dds_worker, args=("Right", pub_l), daemon=True).start()
    else:
        # Single hand mode
        topic_suffix = "r" if SIDE == "right" else "l"
        topic = f"rt/inspire_hand/ctrl/{topic_suffix}"
        label = "Left" if SIDE == "right" else "Right"
        
        print(f"Initializing {SIDE.capitalize()} Hand Publisher (Topic: {topic})...")
        pub = ChannelPublisher(topic, inspire_dds.inspire_hand_ctrl)
        pub.Init()
        
        threading.Thread(target=dds_worker, args=(label, pub), daemon=True).start()

    alpha = 0.5 
    smoothed_vals = {
        "Left": np.array([0, 0, 0, 0, 200, 1000], dtype=float),
        "Right": np.array([0, 0, 0, 0, 200, 1000], dtype=float)
    }

    print("DDS Mimic started. Press 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame: continue
            
            depth_frame = hole_filling.process(temporal.process(spatial.process(depth_frame.as_depth_frame())))
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
                    
                    target_label = detected_label
                    if not BOTH:
                        target_side_label = "Left" if SIDE == "right" else "Right"
                        if detected_label != target_side_label: continue
                        target_label = target_side_label

                    points_3d = []
                    for j in range(21):
                        px = int(hand_landmarks[j].x * image.shape[1])
                        py = int(hand_landmarks[j].y * image.shape[0])
                        px = max(0, min(image.shape[1]-1, px))
                        py = max(0, min(image.shape[0]-1, py))
                        
                        depth, count = 0, 0
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                sx, sy = px + dx, py + dy
                                if 0 <= sx < 640 and 0 <= sy < 480:
                                    d_raw = depth_data[sy, sx]
                                    if d_raw > 0:
                                        depth += d_raw * depth_scale
                                        count += 1
                        if count > 0: depth /= count
                        points_3d.append(rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth))

                    def calc_flex_3d(tip_idx, mcp_idx, wrist_idx, min_r=0.6, max_r=1.1):
                        d_wrist_tip = get_3d_distance(points_3d[wrist_idx], points_3d[tip_idx])
                        d_wrist_mcp = get_3d_distance(points_3d[wrist_idx], points_3d[mcp_idx])
                        ratio = d_wrist_tip / (d_wrist_mcp + 1e-6)
                        return max(0, min(1000, int((ratio - min_r) * (1000 / (max_r - min_r)))))

                    raw_vals = [0, 0, 0, 0, 200, 1000]
                    raw_vals[0] = calc_flex_3d(20, 17, 0, min_r=0.6, max_r=1.0) 
                    raw_vals[1] = calc_flex_3d(16, 13, 0, min_r=0.6, max_r=1.1) 
                    raw_vals[2] = calc_flex_3d(12, 9, 0,  min_r=0.6, max_r=1.2)
                    raw_vals[3] = calc_flex_3d(8, 5, 0,   min_r=0.6, max_r=1.2)
                    raw_vals[4] = int(np.interp(get_angle(points_3d[4], points_3d[3], points_3d[2]), [140.0, 175.0], [0, 1000]))
                    raw_vals[5] = int(np.interp(get_angle(points_3d[2], points_3d[0], points_3d[5]), [25.0, 55.0], [0, 1000]))

                    smoothed_vals[target_label] = alpha * np.array(raw_vals) + (1 - alpha) * smoothed_vals[target_label]
                    current_cmds[target_label] = [int(v) for v in smoothed_vals[target_label]]

                    if not HEADLESS:
                        for pt_3d in points_3d:
                            pixel = rs.rs2_project_point_to_pixel(intrinsics, pt_3d)
                            cv2.circle(image, (int(pixel[0]), int(pixel[1])), 3, (0, 255, 0), -1)
                        cv2.putText(image, f"{target_label}", (int(hand_landmarks[0].x * 640), int(hand_landmarks[0].y * 480)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if not HEADLESS:
                cv2.imshow('DDS Hand Mimic', image)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            elif int(time.time() * 10) % 50 == 0:
                print(f"L: {current_cmds['Left']} | R: {current_cmds['Right']}")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        running = False
        pipeline.stop()
        if not HEADLESS: cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
