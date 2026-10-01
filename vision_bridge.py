import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyrealsense2 as rs
import time
import json
import os
import sys

# Unitree SDK2 / DDS Imports
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
TOPIC = "rt/vision/hand_landmarks"

def main():
    # Initialize DDS
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    pub = ChannelPublisher(TOPIC, String_)
    pub.Init()

    # Initialize MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # Initialize RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"Failed to start RealSense: {e}")
        return

    align = rs.align(rs.stream.color)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    print(f"Vision Bridge started. Publishing 3D landmarks to {TOPIC}")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            detection_result = detector.detect(mp_image)

            if detection_result.hand_landmarks:
                landmarks_data = []
                for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    side = detection_result.handedness[i][0].category_name
                    score = detection_result.handedness[i][0].score
                    
                    if score < 0.7: continue
                    
                    points_3d = []
                    for lm in hand_landmarks:
                        px, py = int(lm.x * 640), int(lm.y * 480)
                        px = max(0, min(639, px))
                        py = max(0, min(479, py))
                        
                        d = depth_frame.get_distance(px, py)
                        if d == 0: # Try window sampling if direct point is zero
                            d_list = []
                            for dx in range(-2, 3):
                                for dy in range(-2, 3):
                                    nx, ny = px + dx, py + dy
                                    if 0 <= nx < 640 and 0 <= ny < 480:
                                        dd = depth_frame.get_distance(nx, ny)
                                        if dd > 0: d_list.append(dd)
                            if d_list: d = sum(d_list) / len(d_list)

                        pt = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], d)
                        points_3d.append(pt) # [x, y, z] in meters
                    
                    landmarks_data.append({
                        "side": side,
                        "points_3d": points_3d
                    })
                
                if landmarks_data:
                    print(f"Publishing landmarks for {len(landmarks_data)} hands")
                    msg = String_(data=json.dumps(landmarks_data))
                    pub.Write(msg)

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
