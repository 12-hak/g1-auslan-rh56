import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from auslan_features import extract_features
from auslan_recognize import train_classifier

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
SAMPLES_DIR = os.path.join(HERE, "auslan_data", "samples")
SEQ_DIR = os.path.join(HERE, "auslan_data", "sequences")
CLF_PATH = os.path.join(HERE, "auslan_data", "classifier.joblib")


def ensure_dirs():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(SEQ_DIR, exist_ok=True)


def save_static(label, feat):
    folder = os.path.join(SAMPLES_DIR, label.upper())
    os.makedirs(folder, exist_ok=True)
    name = f"{int(time.time() * 1000)}_{np.random.randint(0, 9999):04d}.npy"
    path = os.path.join(folder, name)
    np.save(path, feat.astype(np.float32))
    return path


def save_sequence(label, frames):
    folder = os.path.join(SEQ_DIR, label.upper())
    os.makedirs(folder, exist_ok=True)
    name = f"{int(time.time() * 1000)}_{np.random.randint(0, 9999):04d}.npy"
    path = os.path.join(folder, name)
    np.save(path, np.stack(frames).astype(np.float32))
    return path


def draw_hands(bgr, result):
    if not result.hand_landmarks:
        return
    h, w = bgr.shape[:2]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    for lm in result.hand_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in connections:
            cv2.line(bgr, pts[a], pts[b], (255, 80, 0), 2)
        for p in pts:
            cv2.circle(bgr, p, 3, (0, 255, 0), -1)


def prompt_custom_label():
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        name = simpledialog.askstring("New sign", "Label name (e.g. COFFEE):", parent=root)
        root.destroy()
        if name:
            return "".join(ch for ch in name.upper() if ch.isalnum() or ch == "_")
    except Exception:
        pass
    name = input("New sign label: ").strip().upper()
    return "".join(ch for ch in name if ch.isalnum() or ch == "_") or None


def collect_loop(camera, frames_per_key, seq_mode, seq_len):
    if not os.path.isfile(MODEL_PATH):
        print(f"Missing {MODEL_PATH}")
        return 1

    ensure_dirs()
    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(base_options=base, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Cannot open camera {camera}")
        return 1

    active = None
    remaining = 0
    saved = 0
    seq_buf = []
    word_keys = {
        ord("1"): "HELLO",
        ord("2"): "YES",
        ord("3"): "NO",
        ord("4"): "THANKS",
        ord("5"): "PLEASE",
        ord("6"): "GOOD",
        ord("7"): "BYE",
    }

    mode = "SEQ" if seq_mode else "STATIC"
    print(f"Collect [{mode}]: A-Z letters, 1-7 words, n=new custom label")
    print("t=train RF | T=train TF (static+seq) | q=quit")

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        bgr = cv2.flip(bgr, 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        draw_hands(bgr, result)

        feat = None
        if result.hand_landmarks:
            feat = extract_features(result.hand_landmarks, result.handedness)

        if active and feat is not None and remaining > 0:
            if seq_mode:
                seq_buf.append(feat)
                remaining -= 1
                if remaining <= 0:
                    # pad/trim
                    while len(seq_buf) < seq_len:
                        seq_buf.insert(0, np.zeros_like(feat))
                    save_sequence(active, seq_buf[-seq_len:])
                    saved += 1
                    print(f"Saved sequence for {active} ({seq_len} frames)")
                    active = None
                    seq_buf = []
            else:
                save_static(active, feat)
                remaining -= 1
                saved += 1
                if remaining <= 0:
                    print(f"Saved batch for {active}")
                    active = None

        status = f"[{mode}] label={active or '-'} left={remaining} total={saved}"
        cv2.putText(bgr, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(
            bgr,
            "A-Z / 1-7 / n custom | s toggle seq | t RF | T TF | q",
            (10, bgr.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
        )
        cv2.imshow("Auslan collect", bgr)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            seq_mode = not seq_mode
            mode = "SEQ" if seq_mode else "STATIC"
            print(f"Mode -> {mode}")
        if key == ord("t"):
            try:
                n, labels, path = train_classifier(SAMPLES_DIR, CLF_PATH)
                print(f"RF trained on {n} -> {path} labels={labels}")
            except Exception as e:
                print(f"RF train failed: {e}")
        if key == ord("T"):
            try:
                from auslan_train import train_static_keras, train_seq_keras, STATIC_KERAS, SEQ_KERAS

                try:
                    n, labels, path = train_static_keras(SAMPLES_DIR, STATIC_KERAS)
                    print(f"TF-static {n} -> {path} {labels}")
                except Exception as e:
                    print(f"TF-static: {e}")
                try:
                    n, labels, path = train_seq_keras(SEQ_DIR, SEQ_KERAS, seq_len=seq_len)
                    print(f"TF-seq {n} -> {path} {labels}")
                except Exception as e:
                    print(f"TF-seq: {e}")
            except Exception as e:
                print(f"TF train failed: {e}")
        if key == ord("n"):
            custom = prompt_custom_label()
            if custom:
                active = custom
                remaining = seq_len if seq_mode else frames_per_key
                seq_buf = []
                print(f"Capturing custom {active} x{remaining}")
        if ord("a") <= key <= ord("z"):
            active = chr(key).upper()
            remaining = seq_len if seq_mode else frames_per_key
            seq_buf = []
            print(f"Capturing {active} x{remaining}")
        if key in word_keys:
            active = word_keys[key]
            remaining = seq_len if seq_mode else frames_per_key
            seq_buf = []
            print(f"Capturing {active} x{remaining}")

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    return 0


def main():
    p = argparse.ArgumentParser(description="Collect Auslan samples (static or sequences) / train")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--frames", type=int, default=25, help="static frames per keypress")
    p.add_argument("--seq", action="store_true", help="start in sequence capture mode")
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--train", action="store_true", help="train RF from static samples and exit")
    p.add_argument(
        "--train-tf",
        action="store_true",
        help="train TensorFlow static+seq (if data exists) and exit",
    )
    p.add_argument("--label", default=None, help="optional: print path hint for a new label folder")
    args = p.parse_args()

    ensure_dirs()
    if args.label:
        lab = args.label.upper()
        os.makedirs(os.path.join(SAMPLES_DIR, lab), exist_ok=True)
        os.makedirs(os.path.join(SEQ_DIR, lab), exist_ok=True)
        print(f"Ready folders for {lab}")

    if args.train:
        n, labels, path = train_classifier(SAMPLES_DIR, CLF_PATH)
        print(f"Trained RF on {n} -> {path}")
        print("Labels:", labels)
        return 0

    if args.train_tf:
        rc = os.system(f'"{sys.executable}" "{os.path.join(HERE, "auslan_train.py")}" --backend all')
        return 0 if rc == 0 else 1

    return collect_loop(args.camera, args.frames, args.seq, args.seq_len)


if __name__ == "__main__":
    sys.exit(main() or 0)
