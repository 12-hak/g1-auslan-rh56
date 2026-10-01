import json
import os
import time
from collections import deque

import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

from auslan_features import FEATURE_DIM, finger_curl_ratios

WORD_LABELS = {"HELLO", "YES", "NO", "THANKS", "PLEASE", "GOOD", "BYE"}
LETTER_LABELS = {chr(c) for c in range(ord("A"), ord("Z") + 1)}

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "auslan_data")
DEFAULT_LABELS = os.path.join(DATA_DIR, "labels.json")
DEFAULT_STATIC_KERAS = os.path.join(DATA_DIR, "sign_static.keras")
DEFAULT_SEQ_KERAS = os.path.join(DATA_DIR, "sign_seq.keras")
DEFAULT_RF = os.path.join(DATA_DIR, "classifier.joblib")


def geometric_guess(hand_landmarks_list, handedness_list):
    if not hand_landmarks_list:
        return None, 0.0

    best = None
    best_score = 0.0
    for lm in hand_landmarks_list:
        curls = finger_curl_ratios(lm)
        openish = sum(1 for k in ("index", "middle", "ring", "pinky") if curls[k] > 1.4)
        closedish = sum(1 for k in ("index", "middle", "ring", "pinky") if curls[k] < 1.15)
        thumb_out = curls["thumb"] > 1.2

        if openish >= 4 and thumb_out:
            label, score = "B", 0.35
        elif closedish >= 4 and not thumb_out:
            label, score = "A", 0.35
        elif curls["index"] > 1.4 and closedish >= 3:
            label, score = "D", 0.3
        elif openish >= 3 and curls["index"] > 1.35 and curls["middle"] > 1.35:
            label, score = "V", 0.3
        else:
            label, score = None, 0.0

        if score > best_score:
            best_score = score
            best = label

    if best is None:
        return None, 0.0
    return best, best_score


def _load_labels(path):
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [str(x).upper() for x in data.get("labels", [])]


class FeatureRing:
    def __init__(self, seq_len=30, feat_dim=FEATURE_DIM):
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.buf = deque(maxlen=seq_len)

    def clear(self):
        self.buf.clear()

    def push(self, feat):
        v = np.asarray(feat, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.feat_dim:
            out = np.zeros(self.feat_dim, dtype=np.float32)
            n = min(self.feat_dim, v.shape[0])
            out[:n] = v[:n]
            v = out
        self.buf.append(v)

    def ready(self):
        return len(self.buf) >= max(8, self.seq_len // 3)

    def as_array(self):
        if not self.buf:
            return np.zeros((self.seq_len, self.feat_dim), dtype=np.float32)
        arr = np.stack(list(self.buf)).astype(np.float32)
        if arr.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - arr.shape[0], self.feat_dim), dtype=np.float32)
            arr = np.concatenate([pad, arr], axis=0)
        elif arr.shape[0] > self.seq_len:
            arr = arr[-self.seq_len :]
        return arr


class SignClassifier:
    """
    Backends (auto-picked if model_path is a directory / default):
      1) keras sequence LSTM  auslan_data/sign_seq.keras
      2) keras static MLP     auslan_data/sign_static.keras
      3) sklearn RF           auslan_data/classifier.joblib
      4) geometric fallback
    """

    def __init__(self, model_path=None, labels_path=None, seq_len=30):
        self.backend = None
        self.model = None
        self.labels = []
        self.seq_len = seq_len
        self.ring = FeatureRing(seq_len=seq_len)
        self.load(model_path, labels_path)

    def load(self, model_path=None, labels_path=None):
        self.backend = None
        self.model = None
        self.labels = []
        labels_path = labels_path or DEFAULT_LABELS

        candidates = []
        if model_path and os.path.isfile(model_path):
            candidates.append(model_path)
        else:
            for p in (DEFAULT_SEQ_KERAS, DEFAULT_STATIC_KERAS, DEFAULT_RF):
                if os.path.isfile(p):
                    candidates.append(p)

        for path in candidates:
            if path.endswith(".keras") or path.endswith(".h5"):
                if self._load_keras(path, labels_path):
                    return True
            elif path.endswith(".joblib") or path.endswith(".pkl"):
                if self._load_joblib(path):
                    return True
        print("No ML model loaded; using geometric fallback. Train with auslan_train.py")
        return False

    def _load_joblib(self, path):
        if joblib is None:
            return False
        try:
            payload = joblib.load(path)
            self.model = payload["model"]
            self.labels = [str(x).upper() for x in payload.get("labels", [])]
            self.backend = "sklearn_rf"
            print(f"Loaded RF ({len(self.labels)} labels): {path}")
            return True
        except Exception as e:
            print(f"RF load failed: {e}")
            return False

    def _load_keras(self, path, labels_path):
        try:
            import tensorflow as tf
        except ImportError:
            print("tensorflow not installed; skip keras model")
            return False
        try:
            self.model = tf.keras.models.load_model(path)
            self.labels = _load_labels(labels_path)
            shape = self.model.input_shape
            if isinstance(shape, list):
                shape = shape[0]
            if len(shape) == 3:
                self.backend = "keras_seq"
                self.seq_len = int(shape[1] or self.seq_len)
                self.ring = FeatureRing(seq_len=self.seq_len)
            else:
                self.backend = "keras_static"
            if not self.labels:
                n = int(self.model.output_shape[-1])
                self.labels = [f"CLS{i}" for i in range(n)]
            print(f"Loaded {self.backend} ({len(self.labels)} labels): {path}")
            return True
        except Exception as e:
            print(f"Keras load failed: {e}")
            return False

    def push_features(self, features):
        self.ring.push(features)

    def predict(self, features, hand_landmarks_list=None, handedness_list=None):
        feat = np.asarray(features, dtype=np.float32).reshape(-1)
        self.push_features(feat)

        if self.backend == "keras_seq" and self.model is not None:
            if not self.ring.ready():
                return None, 0.0
            x = self.ring.as_array()[None, ...]
            proba = self.model.predict(x, verbose=0)[0]
            idx = int(np.argmax(proba))
            label = self.labels[idx] if idx < len(self.labels) else str(idx)
            return str(label).upper(), float(proba[idx])

        if self.backend == "keras_static" and self.model is not None:
            x = feat.reshape(1, -1)
            proba = self.model.predict(x, verbose=0)[0]
            idx = int(np.argmax(proba))
            label = self.labels[idx] if idx < len(self.labels) else str(idx)
            return str(label).upper(), float(proba[idx])

        if self.backend == "sklearn_rf" and self.model is not None:
            x = feat.reshape(1, -1)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(x)[0]
                idx = int(np.argmax(proba))
                label = self.model.classes_[idx]
                return str(label).upper(), float(proba[idx])
            label = self.model.predict(x)[0]
            return str(label).upper(), 0.6

        return geometric_guess(hand_landmarks_list, handedness_list)


def is_letter_label(label):
    return label is not None and len(str(label)) == 1 and str(label).upper() in LETTER_LABELS


def is_word_sign_label(label, lexicon):
    if not label:
        return False
    u = str(label).upper()
    if is_letter_label(u):
        return False
    return True


class TemporalCommitter:
    def __init__(
        self,
        lexicon,
        letter_hold_frames=10,
        word_hold_frames=18,
        conf_thresh=0.55,
        gap_s=0.8,
    ):
        self.lexicon = set(w.upper() for w in lexicon)
        self.letter_hold_frames = letter_hold_frames
        self.word_hold_frames = word_hold_frames
        self.conf_thresh = conf_thresh
        self.gap_s = gap_s

        self._cand = None
        self._cand_count = 0
        self._last_hand_t = time.time()
        self.buffer = []
        self.last_word = ""
        self.last_letter = ""
        self._recent = deque(maxlen=5)

    def clear(self):
        self.buffer = []
        self._cand = None
        self._cand_count = 0

    def update(self, label, confidence, has_hands):
        events = []
        now = time.time()

        if has_hands:
            self._last_hand_t = now
        else:
            if self.buffer and (now - self._last_hand_t) >= self.gap_s:
                word = "".join(self.buffer)
                events.append(("word", word))
                self.last_word = word
                self.buffer = []
            self._cand = None
            self._cand_count = 0
            return events

        if not label or confidence < self.conf_thresh:
            self._cand = None
            self._cand_count = 0
            return events

        label = str(label).upper()
        if label == self._cand:
            self._cand_count += 1
        else:
            self._cand = label
            self._cand_count = 1

        need = (
            self.word_hold_frames
            if is_word_sign_label(label, self.lexicon)
            else self.letter_hold_frames
        )
        if self._cand_count < need:
            return events

        if self._recent and self._recent[-1] == label:
            self._cand_count = 0
            return events

        if is_word_sign_label(label, self.lexicon):
            events.append(("word", label))
            self.last_word = label
            self.buffer = []
            self._recent.append(label)
            self._cand_count = 0
            return events

        if is_letter_label(label):
            self.buffer.append(label)
            self.last_letter = label
            events.append(("letter", label))
            self._recent.append(label)
            self._cand_count = 0

            joined = "".join(self.buffer)
            if joined in self.lexicon:
                events.append(("word", joined))
                self.last_word = joined
                self.buffer = []
            return events

        return events

    def force_word(self):
        if not self.buffer:
            return None
        word = "".join(self.buffer)
        self.last_word = word
        self.buffer = []
        self._cand = None
        self._cand_count = 0
        return word


def load_lexicon(path):
    words = sorted(WORD_LABELS)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        words = [str(w).upper() for w in data.get("words", words)]
    # Merge trained labels that look like words
    trained = _load_labels(DEFAULT_LABELS)
    for lab in trained:
        if not is_letter_label(lab) and lab not in words:
            words.append(lab)
    return words


def train_classifier(samples_dir, out_path):
    from auslan_train import train_random_forest

    return train_random_forest(samples_dir, out_path)
