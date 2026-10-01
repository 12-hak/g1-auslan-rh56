import argparse
import json
import os
import queue
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pymodbus.client import ModbusTcpClient

from auslan_features import extract_features
from auslan_recognize import SignClassifier, TemporalCommitter, load_lexicon

HERE = os.path.dirname(os.path.abspath(__file__))
LANDMARKER_PATH = os.path.join(HERE, "hand_landmarker.task")
LEXICON_PATH = os.path.join(HERE, "auslan_lexicon.json")
POSES_PATH = os.path.join(HERE, "auslan_poses.json")
CLF_PATH = os.path.join(HERE, "auslan_data", "classifier.joblib")
STATIC_KERAS = os.path.join(HERE, "auslan_data", "sign_static.keras")
SEQ_KERAS = os.path.join(HERE, "auslan_data", "sign_seq.keras")

# Stage-1 Modbus IPs (G1 placement is stage 2)
IP_LEFT_HAND = "192.168.123.210"
IP_RIGHT_HAND = "192.168.123.211"
PORT = 6000
SLAVE_ID = 1
ANGLE_SET_BASE = 1486
NEUTRAL = [0, 0, 0, 0, 200, 1000]

current_cmds = {
    "Left": list(NEUTRAL),
    "Right": list(NEUTRAL),
}
running = True
modbus_muted = False


class TtsWorker:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.q = queue.Queue()
        self._last = ""
        self._last_t = 0.0
        self._thread = None
        if enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def speak(self, text):
        if not self.enabled or not text:
            return
        now = time.time()
        if text == self._last and (now - self._last_t) < 0.35:
            return
        self._last = text
        self._last_t = now
        self.q.put(str(text))

    def _loop(self):
        try:
            import pyttsx3
        except ImportError:
            print("pyttsx3 not installed; TTS disabled")
            self.enabled = False
            return
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        while True:
            text = self.q.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")

    def stop(self):
        if self.enabled:
            self.q.put(None)


def modbus_worker(ip, hand_label, dry_run=False):
    global current_cmds, running, modbus_muted
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
            if modbus_muted:
                time.sleep(0.05)
                continue
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


def load_poses(path):
    if not os.path.isfile(path):
        return {"NEUTRAL": {"Left": list(NEUTRAL), "Right": list(NEUTRAL)}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def apply_pose(poses, label, smoothed, alpha=0.4):
    global current_cmds
    key = str(label).upper()
    pose = poses.get(key) or poses.get("NEUTRAL")
    if not pose:
        return
    for side in ("Left", "Right"):
        target = np.array(pose.get(side, NEUTRAL), dtype=float)
        smoothed[side] = alpha * target + (1.0 - alpha) * smoothed[side]
        current_cmds[side] = [int(np.clip(v, 0, 1000)) for v in smoothed[side]]


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


def parse_args():
    p = argparse.ArgumentParser(description="Auslan webcam sign -> TTS + Inspire hands")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="no Modbus writes")
    p.add_argument("--no-tts", action="store_true")
    p.add_argument(
        "--model",
        default=None,
        help="optional .keras / .joblib path (default: auto pick seq>static>rf)",
    )
    p.add_argument("--conf", type=float, default=0.55)
    p.add_argument("--seq-len", type=int, default=30)
    return p.parse_args()


def main():
    global running, modbus_muted, current_cmds

    args = parse_args()
    if not os.path.isfile(LANDMARKER_PATH):
        print(f"Missing MediaPipe model: {LANDMARKER_PATH}")
        print("Place hand_landmarker.task next to this script.")
        return 1

    lexicon = load_lexicon(LEXICON_PATH)
    poses = load_poses(POSES_PATH)
    model_path = args.model
    if model_path is None:
        for cand in (SEQ_KERAS, STATIC_KERAS, CLF_PATH):
            if os.path.isfile(cand):
                model_path = cand
                break
    clf = SignClassifier(model_path, seq_len=args.seq_len)
    committer = TemporalCommitter(lexicon, conf_thresh=args.conf)
    tts = TtsWorker(enabled=not args.no_tts)

    threads = [
        threading.Thread(
            target=modbus_worker, args=(IP_LEFT_HAND, "Left", args.dry_run), daemon=True
        ),
        threading.Thread(
            target=modbus_worker, args=(IP_RIGHT_HAND, "Right", args.dry_run), daemon=True
        ),
    ]
    for t in threads:
        t.start()

    base = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        running = False
        tts.stop()
        return 1

    smoothed = {
        "Left": np.array(NEUTRAL, dtype=float),
        "Right": np.array(NEUTRAL, dtype=float),
    }
    live_label = "-"
    live_conf = 0.0

    print("Auslan app. Keys: q quit | c clear | space commit word | m mute Modbus")
    print(f"Left={IP_LEFT_HAND} Right={IP_RIGHT_HAND} dry_run={args.dry_run}")
    print("Stage 2 later: mount poses on Unitree G1.")

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            draw_hands(bgr, result)

            has_hands = bool(result.hand_landmarks)
            if has_hands:
                feat = extract_features(result.hand_landmarks, result.handedness)
                live_label, live_conf = clf.predict(
                    feat, result.hand_landmarks, result.handedness
                )
                if live_label:
                    apply_pose(poses, live_label, smoothed)
            else:
                live_label, live_conf = None, 0.0

            events = committer.update(live_label, live_conf, has_hands)
            for kind, value in events:
                if kind == "letter":
                    print(f"letter: {value}  buf={''.join(committer.buffer)}")
                    tts.speak(value)
                    apply_pose(poses, value, smoothed)
                elif kind == "word":
                    print(f"word: {value}")
                    tts.speak(value)
                    apply_pose(poses, value, smoothed)

            buf = "".join(committer.buffer) or "-"
            cv2.putText(
                bgr,
                f"pred={live_label or '-'} conf={live_conf:.2f} [{clf.backend or 'geo'}]",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                bgr,
                f"buf={buf}  word={committer.last_word or '-'}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            mute_s = "MUTE" if modbus_muted else ("DRY" if args.dry_run else "LIVE")
            cv2.putText(
                bgr,
                f"L{current_cmds['Left']} R{current_cmds['Right']} [{mute_s}]",
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                bgr,
                "q quit | c clear | space word | m mute",
                (10, bgr.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
            )
            cv2.imshow("Auslan -> Inspire", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                committer.clear()
                print("buffer cleared")
            if key == ord(" "):
                word = committer.force_word()
                if word:
                    print(f"word(force): {word}")
                    tts.speak(word)
                    apply_pose(poses, word, smoothed)
            if key == ord("m"):
                modbus_muted = not modbus_muted
                print("Modbus muted" if modbus_muted else "Modbus unmuted")
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        running = False
        cap.release()
        tts.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
