import numpy as np
import cv2
import pyrealsense2 as rs
import mediapipe as mp


FINGER_TIPS = (
    ("thumb", 4),
    ("index", 8),
    ("middle", 12),
    ("ring", 16),
    ("pinky", 20),
)


def _cv2_has_gui():
    try:
        cv2.imshow("_rs_gui_test", np.zeros((1, 1, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow("_rs_gui_test")
        return True
    except cv2.error:
        try:
            cv2.destroyWindow("_rs_gui_test")
        except cv2.error:
            pass
        return False


def depth_m_at(depth_frame, x: int, y: int, w: int, h: int) -> float:
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    d = depth_frame.get_distance(x, y)
    return float(d) if d > 0 else float("nan")


def main():
    use_gui = _cv2_has_gui()
    writer = None
    out_path = None
    if not use_gui:
        out_path = "realsense_hand_demo_out.mp4"
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (640, 480),
        )
        if not writer.isOpened():
            writer = None
            print(
                "OpenCV has no GUI and VideoWriter failed. "
                "Try: pip uninstall opencv-python-headless -y && pip install opencv-python"
            )
            return
        print(
            "OpenCV has no HighGUI (both opencv-python and opencv-python-headless "
            "often leave you with no window). Try: pip uninstall opencv-python-headless -y. "
            f"Recording to {out_path}; Ctrl+C to stop."
        )
    else:
        print("RealSense D435i hand + finger demo. Press q to exit.")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    mp_draw = mp.solutions.drawing_utils

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
            w, h = intr.width, intr.height
            bgr = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for idx, lm in enumerate(result.multi_hand_landmarks):
                    handed = result.multi_handedness[idx].classification[0]
                    label = handed.label
                    score = handed.score

                    mp_draw.draw_landmarks(
                        bgr,
                        lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1),
                    )

                    wrist = lm.landmark[0]
                    wx, wy = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(
                        bgr,
                        f"{label} {score:.2f}",
                        (wx, max(20, wy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    col_x = 10 + idx * 260
                    tip_line_y = 25
                    for name, li in FINGER_TIPS:
                        lm_pt = lm.landmark[li]
                        px, py = int(lm_pt.x * w), int(lm_pt.y * h)
                        dist_m = depth_m_at(depth_frame, px, py, w, h)
                        if np.isfinite(dist_m):
                            p3 = rs.rs2_deproject_pixel_to_point(
                                intr, [float(px), float(py)], dist_m
                            )
                            cv2.circle(bgr, (px, py), 5, (0, 128, 255), -1)
                            txt = f"{name}: {dist_m * 100:.1f} cm  z={p3[2]:.3f}m"
                        else:
                            txt = f"{name}: depth n/a"
                        cv2.putText(
                            bgr,
                            txt,
                            (col_x, tip_line_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (200, 230, 200),
                            1,
                            cv2.LINE_AA,
                        )
                        tip_line_y += 18

            if use_gui:
                cv2.imshow("D435i hand + finger tracking", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if writer is not None and writer.isOpened():
                    writer.write(bgr)
    finally:
        hands.close()
        pipeline.stop()
        if writer is not None:
            writer.release()
            if out_path:
                print(f"Saved: {out_path}")
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
