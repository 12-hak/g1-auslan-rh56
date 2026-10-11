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


def _dist(a, b):
    return float(np.hypot(a.x - b.x, a.y - b.y))


def finger_angles(lm):
    """Inspire angles from one hand: pinky, ring, middle, index, thumb bend, thumb rotation.

    0 is closed, 1000 is open. Same mapping as the hand-mimic demos.
    """

    def flex(tip, mcp, min_r, max_r):
        ratio = _dist(lm[0], lm[tip]) / (_dist(lm[0], lm[mcp]) + 1e-6)
        val = int((ratio - min_r) * (1000 / (max_r - min_r)))
        return max(0, min(1000, val))

    vals = [
        flex(20, 17, 1.1, 1.55),
        flex(16, 13, 1.1, 1.85),
        flex(12, 9, 1.1, 1.9),
        flex(8, 5, 1.1, 1.9),
        200,
        1000,
    ]
    v1 = np.array([lm[4].x - lm[3].x, lm[4].y - lm[3].y], dtype=float)
    v2 = np.array([lm[2].x - lm[3].x, lm[2].y - lm[3].y], dtype=float)
    v1 /= np.linalg.norm(v1) + 1e-6
    v2 /= np.linalg.norm(v2) + 1e-6
    bend = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    vals[4] = int(np.interp(bend, [145.0, 175.0], [0, 1000]))

    thumb = np.array([lm[2].x - lm[0].x, lm[2].y - lm[0].y], dtype=float)
    index = np.array([lm[5].x - lm[0].x, lm[5].y - lm[0].y], dtype=float)
    thumb /= np.linalg.norm(thumb) + 1e-6
    index /= np.linalg.norm(index) + 1e-6
    rot = np.degrees(np.arccos(np.clip(np.dot(thumb, index), -1.0, 1.0)))
    vals[5] = int(np.interp(rot, [25.0, 55.0], [0, 1000]))
    return vals


def live_hand_angles(result):
    """Map detected hands to the left and right side of the preview."""
    found = {}
    if not result.hand_landmarks:
        return found
    hands = sorted(result.hand_landmarks, key=lambda lm: lm[0].x)
    if len(hands) == 1:
        side = "Left" if hands[0][0].x < 0.5 else "Right"
        found[side] = finger_angles(hands[0])
    else:
        found["Left"] = finger_angles(hands[0])
        found["Right"] = finger_angles(hands[1])
    return found


def _auslan_window_focused():
    if os.name != "nt":
        return False
    import ctypes

    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    return buf.value.startswith("Auslan")


_held_keys = set()


def poll_keys():
    """Keys while the preview is focused. OpenCV waitKey misses keys on some Windows sessions."""
    edges = set()
    key = cv2.waitKey(1) & 0xFF
    if key not in (255, 0):
        edges.add(" " if key == ord(" ") else chr(key).lower())
    if os.name == "nt" and _auslan_window_focused():
        import ctypes

        user32 = ctypes.windll.user32
        down = set()
        for name, vk in (("q", 0x51), ("c", 0x43), ("m", 0x4D), ("b", 0x42), ("n", 0x4E), (" ", 0x20)):
            if user32.GetAsyncKeyState(vk) & 0x8000:
                down.add(name)
        edges |= down - _held_keys
        _held_keys.clear()
        _held_keys.update(down)
    else:
        _held_keys.clear()
    return edges


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


def _draw_bone(img, p0, p1, color, thickness):
    a = (int(p0[0]), int(p0[1]))
    b = (int(p1[0]), int(p1[1]))
    cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
    cv2.circle(img, b, max(2, thickness), color, -1, cv2.LINE_AA)


def draw_inspire_hand(img, cx, cy, angles, thumb_side, color, active):
    """Cartoon of one Inspire hand. thumb_side +1 puts the thumb on the right."""
    if not active:
        color = tuple(int(c * 0.35) for c in color)
    palm_w, palm_h = 48, 62
    top = int(cy - 6)
    cv2.rectangle(
        img,
        (int(cx - palm_w / 2), top),
        (int(cx + palm_w / 2), top + palm_h),
        color,
        2,
        cv2.LINE_AA,
    )
    # Pinky is on the opposite side from the thumb.
    knuckle_x = [cx - thumb_side * rel for rel in (18, 6, -6, -18)]
    for i in range(4):
        openness = float(angles[i]) / 1000.0
        curl = (1.0 - openness) * np.pi * 0.85
        inward = np.sign(cx - knuckle_x[i])
        dx = inward * np.sin(curl) * 0.35
        dy = -np.cos(curl)
        knuckle = (knuckle_x[i], top)
        mid = (knuckle[0] + dx * 28, knuckle[1] + dy * 28)
        curl2 = min(np.pi * 0.95, curl * 1.25)
        dx2 = inward * np.sin(curl2) * 0.35
        dy2 = -np.cos(curl2)
        tip = (mid[0] + dx2 * 20, mid[1] + dy2 * 20)
        _draw_bone(img, knuckle, mid, color, 2)
        _draw_bone(img, mid, tip, color, 2)

    flex = float(angles[4]) / 1000.0
    rot = float(angles[5]) / 1000.0
    spread = 0.45 + rot * 1.15
    base = (cx + thumb_side * 22, cy + 22)
    d1 = np.array([thumb_side * np.sin(spread), -np.cos(spread * 0.45)], dtype=float)
    d1 /= np.linalg.norm(d1) + 1e-6
    mid = (base[0] + d1[0] * 22, base[1] + d1[1] * 22)
    d2 = np.array(
        [thumb_side * np.sin(spread * flex), -flex * 0.85 + (1.0 - flex) * 0.55],
        dtype=float,
    )
    d2 /= np.linalg.norm(d2) + 1e-6
    tip = (mid[0] + d2[0] * 16, mid[1] + d2[1] * 16)
    _draw_bone(img, base, mid, color, 3)
    _draw_bone(img, mid, tip, color, 3)


def robot_preview(height, cmds, seen):
    """Side panel of the finger commands the physical hands would receive."""
    width = 260
    panel = np.full((height, width, 3), 28, np.uint8)
    cv2.putText(
        panel, "Robot stand-in", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA
    )
    cv2.putText(
        panel,
        "not used for training",
        (12, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    slots = (
        ("L", "Left", 70, (80, 180, 255), 1),
        ("R", "Right", 190, (80, 220, 120), -1),
    )
    cy = max(150, height // 2)
    for name, side, cx, color, thumb_side in slots:
        active = side in seen
        draw_inspire_hand(panel, cx, cy, cmds[side], thumb_side, color, active)
        label = name if active else f"{name} none"
        cv2.putText(
            panel,
            label,
            (cx - 28, cy + 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color if active else (90, 90, 90),
            1,
            cv2.LINE_AA,
        )
    return panel


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
    colors = ((255, 80, 0), (255, 220, 0))
    for i, lm in enumerate(result.hand_landmarks):
        color = colors[i % 2]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in connections:
            cv2.line(bgr, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(bgr, p, 3, (0, 255, 0), -1)


def _wrist_sep(lm, other):
    return ((lm[0].x - other[0].x) ** 2 + (lm[0].y - other[0].y) ** 2) ** 0.5


def _unique_hands(result):
    """Drop a second skeleton that is really the same hand seen twice."""
    kept_lm, kept_h, kept_w = [], [], []
    handed = result.handedness or []
    world = result.hand_world_landmarks or []
    for i, lm in enumerate(result.hand_landmarks or []):
        if any(_wrist_sep(lm, prev) < 0.05 for prev in kept_lm):
            continue
        kept_lm.append(lm)
        kept_h.append(handed[i] if i < len(handed) else [])
        kept_w.append(world[i] if i < len(world) else [])
    return kept_lm, kept_h, kept_w


class TwoHandTracker:
    """Track both hands, including while they touch.

    A fresh detection often keeps only one palm once the hands overlap.
    Video tracking holds a hand that was already seen, and a second pass
    looks again after covering the palm that was found.
    """

    def __init__(self, model_path):
        self.video = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
        self.still = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.25,
                min_hand_presence_confidence=0.25,
            )
        )
        self._t0 = time.perf_counter()
        self._last_ms = 0
        self._held = None
        self._held_until = 0.0

    def detect(self, rgb):
        now = time.perf_counter()
        ms = max(1, int((now - self._t0) * 1000))
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.video.detect_for_video(image, ms)
        found = self._as_result(*_unique_hands(result))
        if len(found.hand_landmarks) < 2:
            found = self._search_again(rgb, found)
        if len(found.hand_landmarks) >= 2:
            self._held = found
            self._held_until = now + 0.5
            return found
        if len(found.hand_landmarks) == 1 and self._held is not None and now < self._held_until:
            glued = self._keep_partner(found, self._held)
            if glued is not None:
                return glued
        return found

    def _search_again(self, rgb, found):
        if not found.hand_landmarks:
            return found
        masked = rgb.copy()
        h, w = masked.shape[:2]
        for lm in found.hand_landmarks:
            pts = np.array(
                [[lm[i].x * w, lm[i].y * h] for i in (0, 5, 9, 13, 17)],
                dtype=np.float32,
            )
            center = pts.mean(axis=0)
            span = float(np.linalg.norm(pts[0] - pts[2]))
            radius = int(max(12, span * 0.55))
            cv2.circle(masked, (int(center[0]), int(center[1])), radius, (0, 0, 0), -1)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=masked)
        extra = self.still.detect(image)
        lm, handed, world = _unique_hands(found)
        extra_lm, extra_h, extra_w = _unique_hands(extra)
        for hand, label, world_lm in zip(extra_lm, extra_h, extra_w):
            if len(lm) >= 2:
                break
            if any(_wrist_sep(hand, prev) < 0.06 for prev in lm):
                continue
            lm.append(hand)
            handed.append(label)
            world.append(world_lm)
        return self._as_result(lm, handed, world)

    def _keep_partner(self, current, held):
        live = current.hand_landmarks[0]
        nearest = min(range(len(held.hand_landmarks)), key=lambda i: _wrist_sep(live, held.hand_landmarks[i]))
        if _wrist_sep(live, held.hand_landmarks[nearest]) > 0.18:
            return None
        if len(held.hand_landmarks) < 2:
            return None
        other = 1 - nearest if nearest < 2 else None
        if other is None or other >= len(held.hand_landmarks):
            return None
        return self._as_result(
            [live, held.hand_landmarks[other]],
            [
                (current.handedness or [[]])[0],
                (held.handedness or [[], []])[other] if other < len(held.handedness or []) else [],
            ],
            [
                (current.hand_world_landmarks or [[]])[0],
                (held.hand_world_landmarks or [[], []])[other]
                if other < len(held.hand_world_landmarks or [])
                else [],
            ],
        )

    @staticmethod
    def _as_result(lm, handed, world):
        return vision.HandLandmarkerResult(
            handedness=handed,
            hand_landmarks=lm,
            hand_world_landmarks=world,
        )


class RealSenseColor:
    """Color frames from an Intel RealSense. OpenCV VideoCapture does not deliver D435i frames."""

    def __init__(self):
        import pyrealsense2 as rs

        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._pipe.start(cfg)

    def isOpened(self):
        return True

    def read(self):
        try:
            frames = self._pipe.wait_for_frames(2000)
        except RuntimeError as e:
            print(f"RealSense frame timeout: {e}")
            return False, None
        color = frames.get_color_frame()
        if not color:
            return False, None
        return True, np.asanyarray(color.get_data())

    def release(self):
        try:
            self._pipe.stop()
        except RuntimeError:
            pass


def open_capture(args):
    if args.realsense:
        try:
            cap = RealSenseColor()
        except ImportError:
            print("pyrealsense2 is not installed. pip install pyrealsense2")
            return None
        except RuntimeError as e:
            print(f"Cannot open RealSense: {e}")
            return None
        print("Camera: RealSense color 640x480")
        return cap

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return None
    return cap


def parse_args():
    p = argparse.ArgumentParser(description="Auslan webcam sign -> TTS + Inspire hands")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument(
        "--realsense",
        action="store_true",
        help="use Intel RealSense color (D435i) instead of OpenCV camera index",
    )
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

    detector = TwoHandTracker(LANDMARKER_PATH)

    cap = open_capture(args)
    if cap is None:
        running = False
        tts.stop()
        return 1

    smoothed = {
        "Left": np.array(NEUTRAL, dtype=float),
        "Right": np.array(NEUTRAL, dtype=float),
    }
    live_label = "-"
    live_conf = 0.0
    status = "click this window, then c / space / m"
    status_until = time.time() + 4.0

    print("Auslan app. Click the preview, then: q quit | c clear | space commit word | m mute")
    print("L/R numbers are finger angles for the hand on that side of the picture.")
    print("Order: pinky, ring, middle, index, thumb bend, thumb rotation. 0 closed, 1000 open.")
    print(f"Left={IP_LEFT_HAND} Right={IP_RIGHT_HAND} dry_run={args.dry_run}")
    print("Stage 2 later: mount poses on Unitree G1.")

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = detector.detect(rgb)
            draw_hands(bgr, result)

            has_hands = bool(result.hand_landmarks)
            angles = live_hand_angles(result)
            for side, vals in angles.items():
                target = np.array(vals, dtype=float)
                smoothed[side] = 0.45 * target + 0.55 * smoothed[side]
                current_cmds[side] = [int(v) for v in smoothed[side]]
            if has_hands:
                feat = extract_features(result.hand_landmarks, result.handedness)
                live_label, live_conf = clf.predict(
                    feat, result.hand_landmarks, result.handedness
                )
            else:
                live_label, live_conf = None, 0.0

            events = committer.update(live_label, live_conf, has_hands)
            for kind, value in events:
                if kind == "letter":
                    print(f"letter: {value}  buf={''.join(committer.buffer)}")
                    tts.speak(value)
                    status = f"letter {value}"
                    status_until = time.time() + 1.5
                    if not args.dry_run:
                        apply_pose(poses, value, smoothed)
                elif kind == "word":
                    print(f"word: {value}")
                    tts.speak(value)
                    status = f"word {value}"
                    status_until = time.time() + 1.5
                    if not args.dry_run:
                        apply_pose(poses, value, smoothed)

            buf = "".join(committer.buffer) or "-"
            n_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
            cv2.putText(
                bgr,
                f"pred={live_label or '-'} conf={live_conf:.2f}  hands={n_hands}",
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

            def _fmt(side):
                if side not in angles:
                    return "none"
                return str(current_cmds[side])

            cv2.putText(
                bgr,
                f"L {_fmt('Left')}",
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                bgr,
                f"R {_fmt('Right')}  [{mute_s}]",
                (10, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                bgr,
                "pinky ring mid index thumb rot | 0 closed 1000 open",
                (10, 128),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (160, 160, 160),
                1,
            )
            if time.time() < status_until and status:
                cv2.putText(
                    bgr,
                    status,
                    (10, 156),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
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
            preview = robot_preview(bgr.shape[0], current_cmds, angles)
            cv2.imshow("Auslan -> Inspire", np.hstack([bgr, preview]))
            keys = poll_keys()
            if "q" in keys:
                break
            if "c" in keys:
                committer.clear()
                committer.last_word = ""
                status = "buffer cleared"
                status_until = time.time() + 1.5
                print("buffer cleared")
            if " " in keys:
                word = committer.force_word()
                if word:
                    print(f"word(force): {word}")
                    tts.speak(word)
                    status = f"word {word}"
                    status_until = time.time() + 1.5
                    if not args.dry_run:
                        apply_pose(poses, word, smoothed)
                else:
                    status = "buffer empty"
                    status_until = time.time() + 1.5
                    print("buffer empty")
            if "m" in keys:
                modbus_muted = not modbus_muted
                status = "Modbus muted" if modbus_muted else "Modbus unmuted"
                status_until = time.time() + 1.5
                print(status)
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
