import sys
import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def _cv2_gui_available():
    try:
        cv2.namedWindow("__cv2_gui_test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__cv2_gui_test__")
        return True
    except cv2.error:
        return False


class XRTeleopRobustViewer:
    def __init__(self):
        # 1. RealSense Setup
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        # 2. MediaPipe Hands & Pose (Robust Head)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        # Using Pose for the head anchor is much more stable than FaceDetection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        self.mp_draw = mp.solutions.drawing_utils
        self._use_gui = '--no-display' not in sys.argv and _cv2_gui_available()
        self._writer = None

    def get_3d_pos(self, landmark, depth_frame, intrinsics):
        x_px, y_px = int(landmark.x * 640), int(landmark.y * 480)
        x_px, y_px = np.clip(x_px, 0, 639), np.clip(y_px, 0, 479)
        dist = depth_frame.get_distance(x_px, y_px)
        if dist > 0:
            return np.array(rs.rs2_deproject_pixel_to_point(intrinsics, [x_px, y_px], dist))
        return None

    def get_wrist_rot(self, landmarks):
        # Landmarks: Wrist(0), Index_MCP(5), Pinky_MCP(17)
        w = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        i = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
        p = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
        x_axis = (i - w) / np.linalg.norm(i - w)
        temp_y = p - w
        z_axis = np.cross(x_axis, temp_y)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        return np.stack([x_axis, y_axis, z_axis], axis=1)

    def run(self):
        print("🚀 Robust Head Anchor Enabled (using Pose model)")
        if not self._use_gui:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter('pose_viewer_output.mp4', fourcc, 30.0, (640, 480))
            if self._writer.isOpened():
                print("OpenCV GUI unavailable; writing pose_viewer_output.mp4 (Ctrl+C to stop).")
            else:
                self._writer.release()
                self._writer = None
                print("OpenCV GUI unavailable; running without display or video file.")
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_f, depth_f = aligned.get_color_frame(), aligned.get_depth_frame()
                if not color_f or not depth_f: continue

                intrinsics = color_f.profile.as_video_stream_profile().get_intrinsics()
                img = np.asanyarray(color_f.get_data())
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Detect Pose (Head Anchor) and Hands
                pose_results = self.pose.process(img_rgb)
                hand_results = self.hands.process(img_rgb)

                head_pos_3d = None

                # 1. Calculate Head Anchor from Pose (Nose is landmark 0)
                if pose_results.pose_landmarks:
                    nose_lm = pose_results.pose_landmarks.landmark[0]
                    head_pos_3d = self.get_3d_pos(nose_lm, depth_f, intrinsics)
                    
                    if head_pos_3d is not None:
                        hx, hy = int(nose_lm.x * 640), int(nose_lm.y * 480)
                        cv2.circle(img, (hx, hy), 10, (255, 0, 255), -1) # Purple dot on nose
                        cv2.putText(img, "HEAD ANCHOR", (hx + 15, hy), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # 2. Process Hands Relative to Head
                if hand_results.multi_hand_landmarks:
                    for i, hand_lms in enumerate(hand_results.multi_hand_landmarks):
                        mp_label = hand_results.multi_handedness[i].classification[0].label
                        # Perspective Correction
                        lbl = "Right" if mp_label == "Left" else "Left"
                        
                        wrist_raw_3d = self.get_3d_pos(hand_lms.landmark[0], depth_f, intrinsics)
                        rot_mat = self.get_wrist_rot(hand_lms.landmark)
                        
                        rel_pos_3d = None
                        if wrist_raw_3d is not None and head_pos_3d is not None:
                            rel_pos_3d = wrist_raw_3d - head_pos_3d

                        # Orientation Logic
                        z_norm = rot_mat[2, 2]
                        side = "Palm" if (lbl == "Right" and z_norm < 0) or (lbl == "Left" and z_norm > 0) else "Back"

                        # Draw Skeleton
                        self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Display Data Block UNDER hand
                        wx, wy = int(hand_lms.landmark[0].x * 640), int(hand_lms.landmark[0].y * 480)
                        start_y = wy + 30
                        
                        # Semi-transparent background for data
                        cv2.rectangle(img, (wx - 5, start_y - 15), (wx + 210, start_y + 40), (0, 0, 0), -1)
                        
                        color = (0, 255, 0) if side == "Palm" else (0, 165, 255)
                        cv2.putText(img, f"{lbl}: {side}", (wx, start_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        if rel_pos_3d is not None:
                            coord_str = f"X:{rel_pos_3d[0]:.2f} Y:{rel_pos_3d[1]:.2f} Z:{rel_pos_3d[2]:.2f}m"
                            cv2.putText(img, coord_str, (wx, start_y + 22), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        else:
                            cv2.putText(img, "HEAD NOT FOUND", (wx, start_y + 22), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                if self._use_gui:
                    cv2.imshow('Robust Head-Relative Tracking', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                elif self._writer is not None:
                    self._writer.write(img)
        finally:
            self.pipeline.stop()
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

if __name__ == "__main__":
    XRTeleopRobustViewer().run()