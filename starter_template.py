"""
Starter template — BCI Neurofeedback Hackathon
===============================================

This file connects to the brain emulator, accumulates a sliding window of
labeled samples, and gives you clean numpy arrays to work with.

Run the emulator first (in a separate terminal):
    python -m emulator

Then run this file:
    python starter_template.py

The emulator publishes one JSON sample every 100 ms (10 Hz).
Each sample looks like:
{
    "timestamp":        1713000000.123,
    "sample_idx":       42,
    "data":             [0.31, -1.22, ...],   # 256 floats — raw neural signal
    "label":            0,                    # int 0-3, or null for rest
    "label_name":       "left_hand",
    "n_dims":           256,
    "sample_rate":      10,
    "difficulty":       "d1",
    "class_scale":      0.74,                 # signal strength 0-1
    "strategy_quality": 0.92                  # how well operator is doing 0-1
}

Label mapping
-------------
    0 → left_hand
    1 → right_hand
    2 → left_leg
    3 → right_leg
    None → rest (no active intention)
"""

import collections
import json
import time

import numpy as np
import zmq

# ---------------------------------------------------------------------------
# Connection settings — change port if you launched the emulator with --port
# ---------------------------------------------------------------------------

HOST = "localhost"
PORT = 5555

# ---------------------------------------------------------------------------
# Buffer settings
# ---------------------------------------------------------------------------

WINDOW_SIZE = 200    # number of recent samples to keep (200 samples = 20 s at 10 Hz)
MIN_SAMPLES = 30     # don't try to fit a model until we have at least this many

# ---------------------------------------------------------------------------
# Connect to the emulator
# ---------------------------------------------------------------------------

ctx    = zmq.Context()
socket = ctx.socket(zmq.SUB)
socket.connect(f"tcp://{HOST}:{PORT}")
socket.setsockopt_string(zmq.SUBSCRIBE, "")   # subscribe to all topics

print(f"Connected to tcp://{HOST}:{PORT}")
print(f"Waiting for data … (press Ctrl-C to stop)\n")

# ---------------------------------------------------------------------------
# Sliding window buffers
# ---------------------------------------------------------------------------

# All samples (data + label), up to WINDOW_SIZE
window: collections.deque = collections.deque(maxlen=WINDOW_SIZE)

# Per-class buffers — keep the last 50 labeled samples per class separately.
# Useful so that all classes are always represented in your fit data even if
# the operator has only been pressing one key recently.
PER_CLASS_BUF = 50
class_buffers: dict[int, collections.deque] = {
    c: collections.deque(maxlen=PER_CLASS_BUF) for c in range(4)
}

# ---------------------------------------------------------------------------
# Helper: extract numpy arrays from the window
# ---------------------------------------------------------------------------

def get_window_arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : np.ndarray, shape (N, n_dims)
        Raw neural signal — one row per sample, all in the window.
    y : np.ndarray, shape (N,), dtype int
        Class labels. -1 means rest / unlabeled.
    """
    samples = list(window)
    X = np.array([s["data"]  for s in samples], dtype=float)
    y = np.array([s["label"] if s["label"] is not None else -1
                  for s in samples], dtype=int)
    return X, y


def get_labeled_arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    Same as get_window_arrays() but only returns rows where label != -1.
    Handy for fitting classifiers / projections.
    """
    X, y = get_window_arrays()
    mask = y >= 0
    return X[mask], y[mask]


def get_per_class_arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns labeled data drawn from per-class buffers so all four classes
    are always represented (even if the operator hasn't pressed a key recently).

    Returns X (N, n_dims) and y (N,) — both labeled only.
    """
    Xs, ys = [], []
    for cls, buf in class_buffers.items():
        if len(buf) > 0:
            Xs.append(np.array([s["data"] for s in buf], dtype=float))
            ys.extend([cls] * len(buf))
    if not Xs:
        return np.empty((0, 0)), np.empty(0, dtype=int)
    return np.vstack(Xs), np.array(ys, dtype=int)


# ---------------------------------------------------------------------------
# TODO: define your model / projection here
# ---------------------------------------------------------------------------
# Example skeleton — replace with your own:
#
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
# projection = PCA(n_components=2)
# fitted = False
#
# def fit_projection(X: np.ndarray, y: np.ndarray) -> None:
#     global fitted
#     projection.fit(X)          # or projection.fit(X, y) for LDA
#     fitted = True
#
# def project(X: np.ndarray) -> np.ndarray:
#     """X: (N, n_dims)  →  coords: (N, 2)"""
#     return projection.transform(X)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

sample_count = 0
last_fit_idx = 0
FIT_EVERY    = 10   # refit your projection every N new samples (~1 s)

try:
    while True:
        # ----------------------------------------------------------------
        # 1. Receive the next sample
        # ----------------------------------------------------------------
        raw = socket.recv_string()
        msg = json.loads(raw)

        data        = np.array(msg["data"], dtype=float)   # shape: (n_dims,)
        label       = msg["label"]                          # int 0-3, or None
        label_name  = msg["label_name"]
        sample_idx  = msg["sample_idx"]
        class_scale = msg.get("class_scale", None)
        strategy_q  = msg.get("strategy_quality", None)
        difficulty  = msg["difficulty"]

        # ----------------------------------------------------------------
        # 2. Accumulate into buffers
        # ----------------------------------------------------------------
        entry = {"data": data, "label": label, "idx": sample_idx}
        window.append(entry)
        if label is not None:
            class_buffers[label].append(entry)

        sample_count += 1

        # ----------------------------------------------------------------
        # 3. Skip until we have enough data
        # ----------------------------------------------------------------
        if sample_count < MIN_SAMPLES:
            continue

        # ----------------------------------------------------------------
        # 4. Get labeled data for fitting
        # ----------------------------------------------------------------
        X_all, y_all      = get_window_arrays()          # full window
        X_labeled, y_labeled = get_labeled_arrays()      # only labeled rows
        X_cls, y_cls      = get_per_class_arrays()       # balanced per-class

        n_classes_seen = len(np.unique(y_labeled[y_labeled >= 0])) \
                         if len(y_labeled) > 0 else 0

        # ----------------------------------------------------------------
        # 5. TODO: fit / update your projection
        # ----------------------------------------------------------------
        # Refit every FIT_EVERY samples once at least 2 classes are present:
        #
        # if (sample_count - last_fit_idx >= FIT_EVERY
        #         and n_classes_seen >= 2
        #         and len(X_cls) >= 8):
        #     fit_projection(X_cls, y_cls)
        #     last_fit_idx = sample_count

        # ----------------------------------------------------------------
        # 6. TODO: project the window to 2D / 3D
        # ----------------------------------------------------------------
        # coords = project(X_all)   # shape: (N, 2)

        # ----------------------------------------------------------------
        # 7. TODO: update your visualisation / feedback display
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # 8. Print a heartbeat so you can see data is flowing
        # ----------------------------------------------------------------
        if sample_count % 10 == 0:   # ~once per second
            scale_str = f"  signal={class_scale:.2f}" if class_scale is not None else ""
            qual_str  = f"  quality={strategy_q:.2f}"  if strategy_q  is not None else ""
            print(
                f"[{sample_idx:6d}]  {label_name:<12}  "
                f"classes seen: {n_classes_seen}"
                f"{scale_str}{qual_str}  "
                f"window: {len(window)} samples"
            )

except KeyboardInterrupt:
    print(f"\nStopped after {sample_count} samples.")
finally:
    socket.close()
    ctx.term()
