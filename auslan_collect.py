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


def save_static(label, feat, root=None):
    folder = os.path.join(root or SAMPLES_DIR, label.upper())
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


LIVE_DIR = os.path.join(HERE, "auslan_data", "live_samples")
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# Auslan C is formed with one hand. The other letters in this set use both.
ONE_HAND_LETTERS = {"C"}
# These letters were one frozen pose, so a small change flipped them.
# I / O / U are a point on middle, ring, and little finger. J and H move.
CUES = {
    "I": "Point at the middle finger. Hold still.",
    "J": "Hold the end of J, after the hook. Not the start.",
    "T": "Hold the finished T. Both hands still.",
    "S": "Hold the finished S. Both hands still.",
    "F": "Hold the finished F. Both hands still.",
    "O": "Point at the ring finger. Hold still.",
    "H": "Hold the finished H, not the middle of the movement.",
    "U": "Point at the little finger. Hold still.",
}


def hands_needed(letter):
    return 1 if str(letter).upper() in ONE_HAND_LETTERS else 2


def _replace_letter_samples(letter):
    folder = os.path.join(LIVE_DIR, letter.upper())
    if not os.path.isdir(folder):
        return
    for name in os.listdir(folder):
        if name.endswith(".npy"):
            os.remove(os.path.join(folder, name))


def guided_collect(camera, realsense, frames=15, letters=None):
    """Prompt each letter. Space starts a 3s countdown, then records it."""
    from auslan_app import draw_hands, open_capture, poll_keys

    if not os.path.isfile(MODEL_PATH):
        print(f"Missing {MODEL_PATH}")
        return 1
    os.makedirs(LIVE_DIR, exist_ok=True)

    class _Args:
        pass

    args = _Args()
    args.camera = camera
    args.realsense = realsense
    cap = open_capture(args)
    if cap is None:
        return 1

    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(base_options=base, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(opts)

    queue = [ch.upper() for ch in (letters or LETTERS)]
    index = 0
    phase = "ready"
    started = 0.0
    grabbed = []
    saved_any = False
    replaced = set()
    notice = "Press space when you are holding the letter"
    print("Guided capture. Click the window. Space = 3s countdown, then record. q = quit, b = back.")

    try:
        while index < len(queue):
            ok, bgr = cap.read()
            if not ok or bgr is None:
                print("Camera frame stopped")
                return 1
            bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            draw_hands(bgr, result)
            letter = queue[index]
            need = hands_needed(letter)
            n_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
            feat = None
            if n_hands:
                feat = extract_features(result.hand_landmarks, result.handedness)

            now = time.time()
            if phase == "countdown":
                left = 3.0 - (now - started)
                if left <= 0:
                    phase = "record"
                    grabbed = []
                    started = now
                else:
                    cv2.putText(
                        bgr,
                        str(max(1, int(np.ceil(left)))),
                        (bgr.shape[1] // 2 - 30, bgr.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.0,
                        (0, 220, 255),
                        6,
                    )
            elif phase == "record":
                hand_ok = n_hands == 1 if need == 1 else n_hands >= 2
                if feat is not None and hand_ok:
                    grabbed.append(feat)
                cv2.putText(
                    bgr,
                    f"Recording {letter}  {len(grabbed)}/{frames}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                if len(grabbed) >= frames:
                    if letter not in replaced:
                        _replace_letter_samples(letter)
                        replaced.add(letter)
                    for sample in grabbed:
                        save_static(letter, sample, root=LIVE_DIR)
                    print(f"Saved {len(grabbed)} one-hand frames for {letter}" if need == 1 else f"Saved {len(grabbed)} frames for {letter}")
                    print(f"  -> {LIVE_DIR}")
                    saved_any = True
                    phase = "ready"
                    notice = f"Saved {letter}. Space adds another take. N goes to the next letter."
                    grabbed = []
                elif now - started > 2.5:
                    phase = "ready"
                    notice = (
                        "Use one hand only. Press space to try again."
                        if need == 1
                        else "Need both hands in frame. Press space to try again."
                    )
                    grabbed = []

            if index >= len(queue):
                break

            letter = queue[index]
            need = hands_needed(letter)
            cv2.putText(
                bgr,
                f"Sign {letter} ({'one hand' if need == 1 else 'both hands'})   {index + 1}/{len(queue)}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3,
            )
            cue = CUES.get(letter, "")
            if cue:
                cv2.putText(
                    bgr,
                    cue,
                    (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 255, 180),
                    2,
                )
            cv2.putText(
                bgr,
                f"hands={n_hands}  {notice}",
                (10, 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                2,
            )
            cv2.putText(
                bgr,
                "space = 3s then capture | n next | b back | q quit",
                (10, bgr.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
            )
            cv2.imshow("Auslan collect", bgr)
            keys = poll_keys()
            if "q" in keys:
                break
            if "b" in keys and phase == "ready" and index > 0:
                index -= 1
                notice = f"Back to {queue[index]}. Press space when ready."
            if "n" in keys and phase == "ready" and index < len(queue) - 1:
                index += 1
                notice = f"Next is {queue[index]}. Press space when ready."
            if " " in keys and phase == "ready":
                phase = "countdown"
                started = time.time()
                notice = "Hold the letter"
        else:
            print(f"Finished A–Z in {LIVE_DIR}")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    print(f"Live samples are in {LIVE_DIR}")
    if saved_any:
        from auslan_train import RF_JOBLIB, train_random_forest

        count, labels, path = train_random_forest(LIVE_DIR, RF_JOBLIB)
        print(f"Retrained on {count} live frames -> {path}")
        print(f"Labels: {labels}")
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
    p.add_argument("--realsense", action="store_true", help="capture from the Intel RealSense color camera")
    p.add_argument("--letters", default="", help="only these letters, e.g. C")
    p.add_argument(
        "--guided",
        action="store_true",
        help="prompt each letter; space waits 3s, then records",
    )
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

    if args.guided or args.realsense:
        chosen = [ch.upper() for ch in args.letters if ch.isalpha()]
        return guided_collect(
            args.camera,
            args.realsense,
            frames=min(args.frames, 20),
            letters=chosen or None,
        )

    return collect_loop(args.camera, args.frames, args.seq, args.seq_len)


if __name__ == "__main__":
    sys.exit(main() or 0)
