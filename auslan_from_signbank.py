"""Build Auslan letter samples from Signbank's two-handed alphabet and train.

Images are the public stills (and the short H/J sequences) from
https://auslan.org.au/spell/twohanded.html
licensed CC BY-NC-ND 4.0. They are cached under auslan_data/signbank/ and
are not part of this repo. Only derived landmark samples and the classifier
are written for local recognition.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from auslan_collect import SAMPLES_DIR, save_static
from auslan_features import extract_features
from auslan_train import RF_JOBLIB, train_random_forest

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "auslan_data", "signbank")
BASE_URL = "https://auslan.org.au/static/images/twohanded/"
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")

LETTERS = {}
for _ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    if _ch == "H":
        LETTERS[_ch] = [f"th_h{i}.jpg" for i in range(5)]
    elif _ch == "J":
        LETTERS[_ch] = [f"th_j{i}.jpg" for i in range(3)]
    else:
        LETTERS[_ch] = [f"th_{_ch.lower()}.jpg"]


class _Lm:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _Cat:
    def __init__(self, name):
        self.category_name = name


def _download():
    os.makedirs(CACHE_DIR, exist_ok=True)
    names = [name for frames in LETTERS.values() for name in frames]
    for name in names:
        path = os.path.join(CACHE_DIR, name)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        urllib.request.urlretrieve(BASE_URL + name, path)
        print(f"downloaded {name}")


def _prepare(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    bg = img[0, 0].tolist()
    img = cv2.copyMakeBorder(img, 48, 48, 48, 48, cv2.BORDER_CONSTANT, value=bg)
    h, w = img.shape[:2]
    scale = 720 / float(w)
    return cv2.resize(img, (720, int(round(h * scale))), interpolation=cv2.INTER_CUBIC)


def _detect(detector, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.hand_landmarks:
        return None, None
    hands = []
    handed = []
    for lm, hd in zip(result.hand_landmarks, result.handedness):
        hands.append([_Lm(p.x, p.y, p.z) for p in lm])
        handed.append([_Cat(hd[0].category_name)])
    return hands, handed


def _jitter(hands, rng, sigma):
    out = []
    for hand in hands:
        out.append(
            [
                _Lm(
                    p.x + rng.normal(0, sigma),
                    p.y + rng.normal(0, sigma),
                    p.z + rng.normal(0, sigma * 0.5),
                )
                for p in hand
            ]
        )
    return out


def build_samples(jitter_per_view=24, sigma=0.006, seed=42):
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"Missing {MODEL_PATH}")
    _download()
    rng = np.random.default_rng(seed)
    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    detector = vision.HandLandmarker.create_from_options(opts)

    saved = 0
    weak = []
    for letter, frames in LETTERS.items():
        views = []
        for name in frames:
            img = _prepare(os.path.join(CACHE_DIR, name))
            for view in (img, cv2.flip(img, 1)):
                hands, handed = _detect(detector, view)
                if hands:
                    views.append((hands, handed))
        if not views:
            weak.append(f"{letter}: no hands")
            continue
        n_hands = max(len(h) for h, _ in views)
        if n_hands < 2:
            weak.append(f"{letter}: only {n_hands} hand detected")
        for hands, handed in views:
            feat = extract_features(hands, handed)
            save_static(letter, feat)
            saved += 1
            for _ in range(jitter_per_view):
                save_static(letter, extract_features(_jitter(hands, rng, sigma), handed))
                saved += 1
        print(f"{letter}: {len(views)} view(s), max hands {n_hands}")
    print(f"saved {saved} static samples")
    if weak:
        print("weak detections:")
        for line in weak:
            print(f"  {line}")
    return saved


def main():
    n = build_samples()
    if n < 10:
        raise SystemExit("Not enough Signbank samples to train")
    count, labels, path = train_random_forest(SAMPLES_DIR, RF_JOBLIB)
    print(f"trained {count} samples, {len(labels)} letters -> {path}")
    print("Source: Auslan Signbank two-handed fingerspelling (CC BY-NC-ND 4.0)")
    print("https://auslan.org.au/spell/twohanded.html")


if __name__ == "__main__":
    main()
