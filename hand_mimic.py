import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pymodbus.client import ModbusTcpClient
import time
import threading

# --- Configuration ---
IP = "192.168.123.210"  # Right hand
PORT = 6000
SLAVE_ID = 1
ANGLE_SET_BASE = 1486 
MODEL_PATH = "hand_landmarker.task"
TARGET_HAND_LABEL = "Left" # In selfie view, physical Right is labeled "Left"

# Shared state
current_cmd = [0, 0, 0, 0, 200, 1000]
running = True
tracking_locked = False # Stick to the hand once identified

def modbus_worker():
    global current_cmd, running
    client = ModbusTcpClient(IP, port=PORT)
    if not client.connect():
        print(f"Modbus connection failed to {IP}")
        return
    
    print("Modbus thread started.")
    last_cmd = None
    
    while running:
        try:
            if current_cmd != last_cmd:
                client.write_registers(ANGLE_SET_BASE, current_cmd, slave=SLAVE_ID)
                last_cmd = list(current_cmd)
            time.sleep(0.04) # ~25Hz
        except Exception as e:
            print(f"Modbus Error: {e}")
            break
    
    # Reset to neutral on exit
    try:
        client.write_registers(ANGLE_SET_BASE, [0, 0, 0, 0, 200, 1000], slave=SLAVE_ID)
    except:
        pass
    client.close()
    print("Modbus thread stopped.")

def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    for hand_landmarks in hand_landmarks_list:
        for lm in hand_landmarks:
            x = int(lm.x * annotated_image.shape[1])
            y = int(lm.y * annotated_image.shape[0])
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
            
        connections = [
            (0,1), (1,2), (2,3), (3,4), # Thumb
            (0,5), (5,6), (6,7), (7,8), # Index
            (9,10), (10,11), (11,12),   # Middle
            (13,14), (14,15), (15,16), # Ring
            (0,17), (17,18), (18,19), (19,20), # Pinky
            (5,9), (9,13), (13,17) # Palm
        ]
        for start, end in connections:
            p1 = hand_landmarks[start]
            p2 = hand_landmarks[end]
            cv2.line(annotated_image, 
                     (int(p1.x * annotated_image.shape[1]), int(p1.y * annotated_image.shape[0])),
                     (int(p2.x * annotated_image.shape[1]), int(p2.y * annotated_image.shape[0])),
                     (255, 0, 0), 2)
                     
    return annotated_image

def main():
    global current_cmd, running, tracking_locked
    
    # Initialize Modbus Thread
    t = threading.Thread(target=modbus_worker, daemon=True)
    t.start()

    # Initialize MediaPipe Landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    alpha = 0.3 
    smoothed_vals = np.array([0, 0, 0, 0, 200, 1000], dtype=float)

    print("Mimic started. Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        detection_result = detector.detect(mp_image)
        raw_vals = [0, 0, 0, 0, 200, 1000]

        if detection_result.hand_landmarks:
            detected_label = detection_result.handedness[0][0].category_name
            score = detection_result.handedness[0][0].score
            
            # Lock logic: Establish lock if target hand is found with high confidence
            if not tracking_locked:
                if detected_label == TARGET_HAND_LABEL and score > 0.7:
                    tracking_locked = True
                    print("Hand Tracked & Locked.")
            
            # Use data if locked (ignore label fluctuations)
            if tracking_locked:
                cv2.putText(image, f"LOCKED: {detected_label}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                lm = detection_result.hand_landmarks[0]
                
                def calc_flex(tip_idx, mcp_idx, wrist_idx, min_r=1.1, max_r=1.9):
                    d_wrist_tip = get_distance(lm[wrist_idx], lm[tip_idx])
                    d_wrist_mcp = get_distance(lm[wrist_idx], lm[mcp_idx])
                    ratio = d_wrist_tip / (d_wrist_mcp + 1e-6)
                    val = int((ratio - min_r) * (1000 / (max_r - min_r)))
                    return max(0, min(1000, val))

                raw_vals[0] = calc_flex(20, 17, 0, min_r=1.1, max_r=1.55) 
                raw_vals[1] = calc_flex(16, 13, 0, min_r=1.1, max_r=1.85) 
                raw_vals[2] = calc_flex(12, 9, 0,  min_r=1.1, max_r=1.9)
                raw_vals[3] = calc_flex(8, 5, 0,   min_r=1.1, max_r=1.9)

                # --- DECOUPLED THUMB LOGIC ---
                
                # 1. Thumb Flexion (DOF 4) - Angle of the "first three joints" (4-3-2)
                # Calculating angle at IP joint (3)
                v1 = np.array([lm[4].x - lm[3].x, lm[4].y - lm[3].y]) # 3 -> 4
                v2 = np.array([lm[2].x - lm[3].x, lm[2].y - lm[3].y]) # 3 -> 2
                v1 /= np.linalg.norm(v1) + 1e-6
                v2 /= np.linalg.norm(v2) + 1e-6
                angle_flex = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
                
                # "Much more based on those three tip joints":
                # High Sensitivity Range: 145 (Bent) to 175 (Straight)
                # Any bend below 145 results in FULL CLOSE.
                raw_vals[4] = int(np.interp(angle_flex, [145.0, 175.0], [0, 1000]))
                
                # Debug angle to terminal
                print(f"Thumb Tip Angle: {angle_flex:.1f} -> Cmd: {raw_vals[4]}")


                # 2. Thumb Rotation (DOF 5) - Track spread of the thumb base
                # Angle between Wrist->ThumbBase(2) and Wrist->IndexBase(5)
                v_thumb = np.array([lm[2].x - lm[0].x, lm[2].y - lm[0].y])
                v_index = np.array([lm[5].x - lm[0].x, lm[5].y - lm[0].y])
                v_thumb /= np.linalg.norm(v_thumb) + 1e-6
                v_index /= np.linalg.norm(v_index) + 1e-6
                angle_rot = np.degrees(np.arccos(np.clip(np.dot(v_thumb, v_index), -1.0, 1.0)))
                
                # Map [25 (In) to 55 (Out)] -> [0 (In) to 1000 (Out)]
                raw_vals[5] = int(np.interp(angle_rot, [25.0, 55.0], [0, 1000]))


                image = draw_landmarks_on_image(image, detection_result)
            else:
                cv2.putText(image, "Searching for Right Hand...", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Unlock when no hand is seen
            if tracking_locked:
                print("Lost Hand. Unlocking.")
            tracking_locked = False

        smoothed_vals = alpha * np.array(raw_vals) + (1 - alpha) * smoothed_vals
        current_cmd = [int(v) for v in smoothed_vals]

        cv2.putText(image, f"CMD: {current_cmd}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Inspire Hand Mimic', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    running = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
