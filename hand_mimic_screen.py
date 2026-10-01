import argparse
import os
import sys
import threading
import time
import tkinter as tk

import cv2
import mediapipe as mp
import mss
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pymodbus.client import ModbusTcpClient

IP_LEFT_HAND = "192.168.123.211"
IP_RIGHT_HAND = "192.168.123.210"
PORT = 6000
SLAVE_ID = 1
ANGLE_SET_BASE = 1486
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
NEUTRAL = [0, 0, 0, 0, 200, 1000]

current_cmds = {
    "Left": list(NEUTRAL),
    "Right": list(NEUTRAL),
}
running = True


def modbus_worker(ip, hand_label, dry_run=False):
    global current_cmds, running
    if dry_run:
        print(f"[dry-run] skip Modbus for {hand_label} @ {ip}")
        return

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
                try:
                    client.write_registers(ANGLE_SET_BASE, cmd, slave=SLAVE_ID)
                except TypeError:
                    try:
                        client.write_registers(ANGLE_SET_BASE, cmd, unit=SLAVE_ID)
                    except TypeError:
                        client.write_registers(ANGLE_SET_BASE, cmd)
                last_cmd = list(cmd)
            time.sleep(0.04)
        except Exception as e:
            print(f"Modbus Error ({hand_label}): {e}")
            break

    try:
        try:
            client.write_registers(ANGLE_SET_BASE, NEUTRAL, slave=SLAVE_ID)
        except TypeError:
            try:
                client.write_registers(ANGLE_SET_BASE, NEUTRAL, unit=SLAVE_ID)
            except TypeError:
                client.write_registers(ANGLE_SET_BASE, NEUTRAL)
    except Exception:
        pass
    client.close()
    print(f"Modbus thread stopped for {hand_label} hand.")


def select_screen_region():
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-alpha", 0.30)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    root.title("Draw Teams speaker region")

    canvas = tk.Canvas(root, cursor="cross", bg="black", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    canvas.create_text(
        root.winfo_screenwidth() // 2,
        40,
        text="Drag a rectangle over the pinned speaker video  |  Enter=confirm  Esc=cancel",
        fill="white",
        font=("Segoe UI", 16, "bold"),
    )

    state = {"x0": None, "y0": None, "rect": None, "done": False, "region": None}

    def on_press(event):
        state["x0"], state["y0"] = event.x_root, event.y_root
        if state["rect"] is not None:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#00ff88", width=3
        )

    def on_drag(event):
        if state["rect"] is None or state["x0"] is None:
            return
        x0 = state["x0"] - root.winfo_rootx()
        y0 = state["y0"] - root.winfo_rooty()
        canvas.coords(state["rect"], x0, y0, event.x, event.y)

    def finish(ok):
        if ok and state["x0"] is not None and state["rect"] is not None:
            x1, y1, x2, y2 = canvas.coords(state["rect"])
            left = int(min(x1, x2) + root.winfo_rootx())
            top = int(min(y1, y2) + root.winfo_rooty())
            right = int(max(x1, x2) + root.winfo_rootx())
            bottom = int(max(y1, y2) + root.winfo_rooty())
            if right - left >= 40 and bottom - top >= 40:
                state["region"] = {
                    "left": left,
                    "top": top,
                    "width": right - left,
                    "height": bottom - top,
                }
        state["done"] = True
        root.quit()

    def on_release(_event):
        pass

    def on_key(event):
        if event.keysym in ("Return", "KP_Enter"):
            finish(True)
        elif event.keysym == "Escape":
            finish(False)

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Key>", on_key)
    root.focus_force()
    root.mainloop()
    root.destroy()
    return state["region"]


def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def landmarks_to_cmd(lm):
    raw_vals = list(NEUTRAL)

    def calc_flex(tip_idx, mcp_idx, wrist_idx, min_r=1.1, max_r=1.9):
        d_wrist_tip = get_distance(lm[wrist_idx], lm[tip_idx])
        d_wrist_mcp = get_distance(lm[wrist_idx], lm[mcp_idx])
        ratio = d_wrist_tip / (d_wrist_mcp + 1e-6)
        val = int((ratio - min_r) * (1000 / (max_r - min_r)))
        return max(0, min(1000, val))

    raw_vals[0] = calc_flex(20, 17, 0, min_r=1.1, max_r=1.55)
    raw_vals[1] = calc_flex(16, 13, 0, min_r=1.1, max_r=1.85)
    raw_vals[2] = calc_flex(12, 9, 0, min_r=1.1, max_r=1.9)
    raw_vals[3] = calc_flex(8, 5, 0, min_r=1.1, max_r=1.9)

    v1 = np.array([lm[4].x - lm[3].x, lm[4].y - lm[3].y])
    v2 = np.array([lm[2].x - lm[3].x, lm[2].y - lm[3].y])
    v1 /= np.linalg.norm(v1) + 1e-6
    v2 /= np.linalg.norm(v2) + 1e-6
    angle_flex = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    raw_vals[4] = int(np.interp(angle_flex, [145.0, 175.0], [0, 1000]))

    v_thumb = np.array([lm[2].x - lm[0].x, lm[2].y - lm[0].y])
    v_index = np.array([lm[5].x - lm[0].x, lm[5].y - lm[0].y])
    v_thumb /= np.linalg.norm(v_thumb) + 1e-6
    v_index /= np.linalg.norm(v_index) + 1e-6
    angle_rot = np.degrees(np.arccos(np.clip(np.dot(v_thumb, v_index), -1.0, 1.0)))
    raw_vals[5] = int(np.interp(angle_rot, [25.0, 55.0], [0, 1000]))
    return raw_vals


def draw_hand(bgr, lm, label):
    h, w = bgr.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    for a, b in connections:
        cv2.line(bgr, pts[a], pts[b], (255, 80, 0), 2)
    for p in pts:
        cv2.circle(bgr, p, 3, (0, 255, 0), -1)
    cv2.putText(bgr, label, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def parse_args():
    p = argparse.ArgumentParser(description="Screen-region hand mimic for Teams speaker tile")
    p.add_argument("side", nargs="?", default="right", choices=["left", "right"], help="hand side")
    p.add_argument("--both", action="store_true", help="drive both hands")
    p.add_argument("--dry-run", action="store_true", help="track only, no Modbus")
    p.add_argument("--no-gui", action="store_true", help="no preview window")
    return p.parse_args()


def main():
    global running, current_cmds

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    args = parse_args()
    both = args.both
    dry_run = args.dry_run
    side = args.side

    if both:
        print("Mode: BOTH hands")
        print(f"  MediaPipe Left  -> physical Right @ {IP_RIGHT_HAND}")
        print(f"  MediaPipe Right -> physical Left  @ {IP_LEFT_HAND}")
    elif side == "left":
        print(f"Mode: LEFT hand @ {IP_LEFT_HAND}")
    else:
        print(f"Mode: RIGHT hand @ {IP_RIGHT_HAND}")

    print("Select the Teams speaker video region...")
    region = select_screen_region()
    if not region:
        print("No region selected. Exiting.")
        return
    print(f"Tracking region: {region}")

    if not os.path.isfile(MODEL_PATH):
        print(f"Missing model: {MODEL_PATH}")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2 if both else 1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    threads = []
    if both:
        threads = [
            threading.Thread(target=modbus_worker, args=(IP_RIGHT_HAND, "Left", dry_run), daemon=True),
            threading.Thread(target=modbus_worker, args=(IP_LEFT_HAND, "Right", dry_run), daemon=True),
        ]
    elif side == "left":
        threads = [threading.Thread(target=modbus_worker, args=(IP_LEFT_HAND, "Left", dry_run), daemon=True)]
    else:
        threads = [threading.Thread(target=modbus_worker, args=(IP_RIGHT_HAND, "Right", dry_run), daemon=True)]
    for t in threads:
        t.start()

    alpha = 0.35
    smoothed = {
        "Left": np.array(NEUTRAL, dtype=float),
        "Right": np.array(NEUTRAL, dtype=float),
    }
    paused = False

    print("Tracking. Keys: q=quit  r=reselect region  space=pause")
    sct = mss.mss()

    try:
        while running:
            if not paused:
                shot = np.asarray(sct.grab(region))
                bgr = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)

                if result.hand_landmarks:
                    for i, lm in enumerate(result.hand_landmarks):
                        score = result.handedness[i][0].score
                        mp_label = result.handedness[i][0].category_name
                        if score < 0.5:
                            continue

                        if both:
                            target = mp_label
                        else:
                            target = "Left" if side == "left" else "Right"

                        raw = landmarks_to_cmd(lm)
                        smoothed[target] = alpha * np.array(raw) + (1 - alpha) * smoothed[target]
                        current_cmds[target] = [int(v) for v in smoothed[target]]
                        draw_hand(bgr, lm, f"{mp_label}->{target}")

                if not args.no_gui:
                    status = f"L:{current_cmds['Left']}  R:{current_cmds['Right']}"
                    cv2.putText(bgr, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    cv2.putText(
                        bgr,
                        "q quit | r reselect | space pause",
                        (8, bgr.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (200, 200, 200),
                        1,
                    )
                    cv2.imshow("Teams region -> Inspire hand", bgr)

            key = 0
            if not args.no_gui:
                key = cv2.waitKey(1) & 0xFF
            else:
                time.sleep(0.01)

            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
                print("Paused" if paused else "Resumed")
            if key == ord("r"):
                print("Reselect region...")
                new_region = select_screen_region()
                if new_region:
                    region = new_region
                    print(f"Tracking region: {region}")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        running = False
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        sct.close()


if __name__ == "__main__":
    main()
