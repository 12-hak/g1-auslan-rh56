import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "auslan_data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
SEQ_DIR = os.path.join(DATA_DIR, "sequences")
STATIC_KERAS = os.path.join(DATA_DIR, "sign_static.keras")
SEQ_KERAS = os.path.join(DATA_DIR, "sign_seq.keras")
LABELS_JSON = os.path.join(DATA_DIR, "labels.json")
RF_JOBLIB = os.path.join(DATA_DIR, "classifier.joblib")


def _load_static(samples_dir):
    xs, ys = [], []
    if not os.path.isdir(samples_dir):
        return xs, ys
    for label in sorted(os.listdir(samples_dir)):
        folder = os.path.join(samples_dir, label)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if not name.endswith(".npy"):
                continue
            feat = np.load(os.path.join(folder, name)).astype(np.float32).reshape(-1)
            xs.append(feat)
            ys.append(label.upper())
    return xs, ys


def _load_sequences(seq_dir, seq_len):
    xs, ys = [], []
    if not os.path.isdir(seq_dir):
        return xs, ys
    for label in sorted(os.listdir(seq_dir)):
        folder = os.path.join(seq_dir, label)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if not name.endswith(".npy"):
                continue
            arr = np.load(os.path.join(folder, name)).astype(np.float32)
            if arr.ndim == 1:
                continue
            if arr.shape[0] >= seq_len:
                arr = arr[-seq_len:]
            else:
                pad = np.zeros((seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([pad, arr], axis=0)
            xs.append(arr)
            ys.append(label.upper())
    return xs, ys


def _encode_labels(ys):
    labels = sorted(set(ys))
    idx = {l: i for i, l in enumerate(labels)}
    y = np.array([idx[v] for v in ys], dtype=np.int32)
    return y, labels


def _save_labels(path, labels, meta=None):
    payload = {"labels": labels}
    if meta:
        payload["meta"] = meta
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train_random_forest(samples_dir, out_path):
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    xs, ys = _load_static(samples_dir)
    if len(xs) < 10:
        raise RuntimeError(f"Need >=10 static samples (got {len(xs)})")
    x = np.stack(xs)
    y = np.array(ys)
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=24,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(x, y)
    labels = sorted(set(ys))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump({"model": clf, "labels": labels, "backend": "sklearn_rf"}, out_path)
    _save_labels(LABELS_JSON, labels, {"backend": "sklearn_rf", "n_samples": len(xs)})
    return len(xs), labels, out_path


def train_static_keras(samples_dir, out_path, epochs=40, batch_size=32):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    xs, ys = _load_static(samples_dir)
    if len(xs) < 20:
        raise RuntimeError(f"Need >=20 static samples for TF (got {len(xs)})")
    x = np.stack(xs).astype(np.float32)
    y, labels = _encode_labels(ys)
    n_feat = x.shape[1]
    n_class = len(labels)

    x_tr, x_va, y_tr, y_va = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y if len(labels) > 1 else None
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_feat,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n_class, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        )
    ]
    hist = model.fit(
        x_tr,
        y_tr,
        validation_data=(x_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save(out_path)
    _save_labels(
        LABELS_JSON,
        labels,
        {
            "backend": "keras_static",
            "model": os.path.basename(out_path),
            "n_samples": len(xs),
            "val_acc": float(max(hist.history.get("val_accuracy", [0]))),
        },
    )
    return len(xs), labels, out_path


def train_seq_keras(seq_dir, out_path, seq_len=30, epochs=50, batch_size=16):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    xs, ys = _load_sequences(seq_dir, seq_len)
    if len(xs) < 20:
        raise RuntimeError(f"Need >=20 sequence samples for TF LSTM (got {len(xs)})")
    x = np.stack(xs).astype(np.float32)
    y, labels = _encode_labels(ys)
    n_feat = x.shape[2]
    n_class = len(labels)

    x_tr, x_va, y_tr, y_va = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y if len(labels) > 1 else None
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(seq_len, n_feat)),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(n_class, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        )
    ]
    hist = model.fit(
        x_tr,
        y_tr,
        validation_data=(x_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save(out_path)
    _save_labels(
        LABELS_JSON,
        labels,
        {
            "backend": "keras_seq",
            "model": os.path.basename(out_path),
            "seq_len": seq_len,
            "n_samples": len(xs),
            "val_acc": float(max(hist.history.get("val_accuracy", [0]))),
        },
    )
    return len(xs), labels, out_path


def main():
    p = argparse.ArgumentParser(description="Train Auslan sign classifiers (RF / TF static / TF LSTM)")
    p.add_argument(
        "--backend",
        choices=["rf", "tf-static", "tf-seq", "all"],
        default="tf-static",
        help="rf=sklearn, tf-static=MLP, tf-seq=LSTM on sequences, all=try each",
    )
    p.add_argument("--samples", default=SAMPLES_DIR)
    p.add_argument("--sequences", default=SEQ_DIR)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    backends = ["rf", "tf-static", "tf-seq"] if args.backend == "all" else [args.backend]
    for b in backends:
        try:
            if b == "rf":
                n, labels, path = train_random_forest(args.samples, RF_JOBLIB)
            elif b == "tf-static":
                n, labels, path = train_static_keras(
                    args.samples, STATIC_KERAS, epochs=args.epochs, batch_size=args.batch_size
                )
            else:
                n, labels, path = train_seq_keras(
                    args.sequences,
                    SEQ_KERAS,
                    seq_len=args.seq_len,
                    epochs=max(args.epochs, 50),
                    batch_size=min(args.batch_size, 16),
                )
            print(f"[{b}] trained on {n} samples -> {path}")
            print(f"[{b}] labels ({len(labels)}): {labels}")
        except Exception as e:
            print(f"[{b}] skipped/failed: {e}")
            if args.backend != "all":
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
