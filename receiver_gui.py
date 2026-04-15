"""
Starter visualization GUI for the Brain Emulator.

Connects to the emulator via ZMQ and shows:
  Left panel  — 2D projection of brain state over time.
                Points fade out with age so you can follow the trajectory.
                Projection axes are recomputed every few seconds and
                sign-aligned with the previous axes so the plot stays stable.
  Right panel — Raw signal (first 8 channels) scrolling over time.

Run the emulator first:
    python -m emulator -d easy

Then in a separate terminal:
    python receiver_gui.py
"""

import json
import threading
import collections
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import zmq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOST              = "localhost"
PORT              = 5555

HISTORY_LEN       = 300    # samples kept for display (30 s at 10 Hz)
FIT_WINDOW        = 50     # samples used to FIT the projection (most recent ~5 s)
RAW_DISPLAY_LEN   = 100    # samples shown on raw signal panel
TRAIL_LEN         = 40     # length of the bright trajectory line
REPROJECT_EVERY   = 10     # recompute projection every N new samples (~1 s)
UPDATE_MS         = 150    # plot refresh interval in ms
MIN_FIT_SAMPLES   = 20     # need at least this many samples before projecting

CLASS_COLORS = {
    0:    "#6495ed",   # cornflower blue  — left_hand
    1:    "#ffa500",   # orange           — right_hand
    2:    "#32cd32",   # lime green       — left_leg
    3:    "#dc143c",   # crimson          — right_leg
    None: "#787878",
}
CLASS_NAMES = {0: "left_hand", 1: "right_hand", 2: "left_leg", 3: "right_leg", None: "rest"}

# ---------------------------------------------------------------------------
# ZMQ receiver thread
# ---------------------------------------------------------------------------

# Each entry: {"data": np.array (n_dims,), "label": int|None, "t": float}
_buffer  : collections.deque = collections.deque(maxlen=HISTORY_LEN)
_lock    = threading.Lock()
_meta    : dict = {}
_running = True


def _receiver_thread():
    ctx    = zmq.Context()
    sock   = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{HOST}:{PORT}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 500)
    print(f"Receiver connected to tcp://{HOST}:{PORT}")
    while _running:
        try:
            msg = json.loads(sock.recv_string())
            with _lock:
                _buffer.append({
                    "data":  np.array(msg["data"], dtype=float),
                    "label": msg["label"],
                    "idx":   msg["sample_idx"],
                })
                _meta.update({k: msg[k] for k in
                               ("sample_idx", "difficulty", "sample_rate",
                                "class_scale", "strategy_quality")
                               if k in msg})
        except zmq.Again:
            pass
    sock.close(); ctx.term()


threading.Thread(target=_receiver_thread, daemon=True).start()

# ---------------------------------------------------------------------------
# Stable PCA projection
# ---------------------------------------------------------------------------

class StableProjection:
    """
    Fits PCA axes on the most recent FIT_WINDOW samples only.
    This means the projection reflects *current* strategy quality — not the
    full history which may be dominated by earlier bad-strategy periods.

    All history samples are then projected onto these recent axes for display,
    so you see the full trail but through a lens tuned to current conditions.

    Signs are aligned with the previous axes on each refit to prevent flipping.
    """
    def __init__(self):
        self.components: np.ndarray | None = None   # shape (2, n_dims)
        self._mean:      np.ndarray | None = None
        self._since_update = 0
        self.fit_window_size = 0   # exposed for the title

    def update(self, X_fit: np.ndarray, X_all: np.ndarray) -> np.ndarray:
        """
        X_fit: recent samples used to fit axes  (N_fit, n_dims)
        X_all: all history samples to project   (N_all, n_dims)
        Returns projected coordinates           (N_all, 2)
        """
        self._since_update += 1
        n_fit = len(X_fit)
        self.fit_window_size = n_fit

        if n_fit < MIN_FIT_SAMPLES:
            return np.zeros((len(X_all), 2))

        if self.components is None or self._since_update >= REPROJECT_EVERY:
            self._refit(X_fit)
            self._since_update = 0

        return (X_all - self._mean) @ self.components.T

    def _refit(self, X: np.ndarray):
        mean     = X.mean(axis=0)
        Xc       = X - mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        new_comp = Vt[:2]

        if self.components is not None:
            for i in range(2):
                if np.dot(new_comp[i], self.components[i]) < 0:
                    new_comp[i] *= -1

        self.components = new_comp
        self._mean      = mean


_proj = StableProjection()

# ---------------------------------------------------------------------------
# Fisher separability score
# ---------------------------------------------------------------------------

def fisher_score(X2: np.ndarray, labels: np.ndarray) -> float:
    classes = [c for c in np.unique(labels) if c >= 0]
    if len(classes) < 2:
        return 0.0
    mu  = X2.mean(axis=0)
    between = sum((labels == c).sum() * np.linalg.norm(X2[labels==c].mean(axis=0) - mu)**2
                  for c in classes)
    within  = sum(np.sum((X2[labels==c] - X2[labels==c].mean(axis=0))**2)
                  for c in classes)
    return float(between / (within + 1e-9))

# ---------------------------------------------------------------------------
# Plot setup
# ---------------------------------------------------------------------------

fig, (ax_proj, ax_raw) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0e0e1a")

for ax in (ax_proj, ax_raw):
    ax.set_facecolor("#141422")
    ax.tick_params(colors="#888", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

ax_proj.set_title("PCA projection", color="#aaa", fontsize=11)
ax_proj.set_xlabel("PC 1", color="#777", fontsize=9)
ax_proj.set_ylabel("PC 2", color="#777", fontsize=9)

ax_raw.set_title("Raw signal — first 8 channels", color="#aaa", fontsize=11)
ax_raw.set_xlabel("sample index", color="#777", fontsize=9)
ax_raw.set_ylabel("amplitude (offset per channel)", color="#777", fontsize=9)

legend_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=CLASS_COLORS[i],
           markersize=8, label=CLASS_NAMES[i], linewidth=0)
    for i in range(4)
] + [Line2D([0],[0], marker='o', color='w', markerfacecolor=CLASS_COLORS[None],
            markersize=6, label="rest", linewidth=0)]
ax_proj.legend(handles=legend_handles, loc="upper right",
               facecolor="#1e1e2e", edgecolor="#444", labelcolor="#ccc", fontsize=8)

fig.tight_layout(pad=2.5)

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

_raw_palette = plt.cm.plasma(np.linspace(0.2, 0.9, 8))

def update(_frame):
    with _lock:
        if len(_buffer) < 2:
            return
        snapshot = list(_buffer)

    data   = np.array([s["data"]  for s in snapshot])
    labels = np.array([s["label"] if s["label"] is not None else -1
                       for s in snapshot])
    idxs   = np.array([s["idx"]   for s in snapshot])
    N      = len(snapshot)

    # Fit projection on recent window only; display all history
    n_fit   = min(FIT_WINDOW, N)
    X_fit   = data[-n_fit:]
    coords  = _proj.update(X_fit, data)   # axes from recent, coords for all

    # ----------------------------------------------------------------
    # Left: stable PCA scatter with fade trail
    # ----------------------------------------------------------------

    ax_proj.cla()
    ax_proj.set_facecolor("#141422")
    for sp in ax_proj.spines.values():
        sp.set_edgecolor("#333355")

    # Age-based alpha: newest sample = 1.0, oldest = 0.05
    ages   = np.linspace(0.05, 1.0, N)

    # Draw points per class, alpha encodes age
    for cls_key, color in CLASS_COLORS.items():
        if cls_key is None:
            mask = labels == -1
            size, zorder = 10, 1
        else:
            mask = labels == cls_key
            size, zorder = 18, 2
        if not mask.any():
            continue
        rgba = np.zeros((mask.sum(), 4))
        c    = plt.matplotlib.colors.to_rgb(color)
        rgba[:, :3] = c
        rgba[:, 3]  = ages[mask]
        ax_proj.scatter(coords[mask, 0], coords[mask, 1],
                        c=rgba, s=size, linewidths=0, zorder=zorder)

    # Bright trajectory line for the most recent TRAIL_LEN points
    if N >= 2:
        trail_n = min(TRAIL_LEN, N)
        tx, ty  = coords[-trail_n:, 0], coords[-trail_n:, 1]
        ax_proj.plot(tx, ty, color="#ffffff", linewidth=0.8, alpha=0.35, zorder=3)

    # Current point: large white dot
    ax_proj.scatter(coords[-1, 0], coords[-1, 1],
                    c="white", s=70, zorder=5, linewidths=0)

    # Separability score (computed on fit window only — reflects current strategy)
    fit_labels   = labels[-n_fit:]
    fit_coords   = coords[-n_fit:]
    labeled_mask = fit_labels >= 0
    score = fisher_score(fit_coords[labeled_mask], fit_labels[labeled_mask]) \
            if labeled_mask.sum() > 8 else 0.0

    diff  = _meta.get("difficulty", "?")
    scale = _meta.get("class_scale", None)
    scale_str = f"   signal={scale:.2f}" if scale is not None else ""
    ax_proj.set_title(
        f"PCA (fit={n_fit} recent)   sep={score:.2f}{scale_str}   [{diff}]",
        color="#aaa", fontsize=10,
    )
    ax_proj.set_xlabel("PC 1", color="#777", fontsize=9)
    ax_proj.set_ylabel("PC 2", color="#777", fontsize=9)
    ax_proj.tick_params(colors="#888", labelsize=9)
    ax_proj.legend(handles=legend_handles, loc="upper right",
                   facecolor="#1e1e2e", edgecolor="#444", labelcolor="#ccc", fontsize=8)

    # ----------------------------------------------------------------
    # Right: raw signal scrolling
    # ----------------------------------------------------------------
    ax_raw.cla()
    ax_raw.set_facecolor("#141422")
    for sp in ax_raw.spines.values():
        sp.set_edgecolor("#333355")

    n_raw  = min(RAW_DISPLAY_LEN, N)
    raw8   = data[-n_raw:, :8]
    ridxs  = idxs[-n_raw:]
    rlabels = labels[-n_raw:]

    # Background shading by label
    for i in range(len(ridxs) - 1):
        lbl = rlabels[i]
        if lbl >= 0:
            ax_raw.axvspan(ridxs[i], ridxs[i+1],
                           color=CLASS_COLORS[lbl], alpha=0.08, linewidth=0)

    for ch in range(8):
        ax_raw.plot(ridxs, raw8[:, ch] + ch * 3,
                    color=_raw_palette[ch], linewidth=0.9, alpha=0.85)

    ax_raw.set_title("Raw signal — first 8 channels", color="#aaa", fontsize=11)
    ax_raw.set_xlabel("sample index", color="#777", fontsize=9)
    ax_raw.set_ylabel("ch offset +3 each", color="#777", fontsize=9)
    ax_raw.tick_params(colors="#888", labelsize=9)

    fig.tight_layout(pad=2.5)


ani = animation.FuncAnimation(fig, update, interval=UPDATE_MS, cache_frame_data=False)

print(f"Visualization running.  History: {HISTORY_LEN} samples  |  Refresh: {UPDATE_MS} ms")
print("Close the window to quit.\n")

try:
    plt.show()
finally:
    _running = False
