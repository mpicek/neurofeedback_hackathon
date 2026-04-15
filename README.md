# Brain Emulator — BCI Neurofeedback Hackathon

A synthetic neural data streamer that emulates what a brain implant would broadcast
during a spinal cord injury rehabilitation session.  Students use it as the data source
for building a real-time neurofeedback GUI.

---

## Background: what is this simulating?

In real BCI sessions with spinal cord injury patients, a multi-electrode array implanted
in motor cortex records the activity of hundreds of neurons simultaneously.  The goal is
to **decode the patient's movement intentions** from this high-dimensional signal so the
patient can receive feedback and learn to produce more discriminable brain states.

The core clinical challenge:

> Some movement intentions (e.g. left hand vs right hand) produce neural signals that
> **overlap heavily** in the raw recording space.  The patient needs neurofeedback —
> a real-time visualisation of their own brain state — to discover mental strategies
> that make the signals separable.  Without feedback, patients cannot find these
> strategies on their own.

This emulator reproduces that challenge without requiring a real patient or implant.

---

## How the emulator works

### The latent state (what the "brain" is doing)

The brain state is represented as an **8-dimensional latent vector** `z`:

```
z = [ z_class    (dims 0-2) ]   — encodes which movement is intended
    [ z_strategy (dims 3-4) ]   — controlled by the arrow keys
    [ z_noise    (dims 5-7) ]   — slow random walk / physiological drift
```

**z_class** is pulled toward one of four class centroids depending on which number key
is held:

| Key | Intention    | Centroid in z_class space     |
|-----|-------------|-------------------------------|
| `1` | Left hand   | `[ 2.0,  0.9,  0.0]`          |
| `2` | Right hand  | `[ 2.0, -0.9,  0.0]`          |
| `3` | Left leg    | `[-2.0,  0.9,  0.0]`          |
| `4` | Right leg   | `[-2.0, -0.9,  0.0]`          |
| `0` | Rest        | decays to origin               |

This structure is intentional: **hand vs leg is easy** (large separation on dim 0),
**left vs right is hard** (small separation on dim 1) — mirroring real clinical data.

### Per-class strategy targets

Each class has its own hidden optimal strategy position in the 2D strategy space:

| Class | Optimal strategy position |
|---|---|
| left_hand  | `[+0.62, -0.41]` |
| right_hand | `[-0.55, +0.38]` |
| left_leg   | `[+0.24, +0.70]` |
| right_leg  | `[-0.37, -0.63]` |

Finding the right strategy for left hand does **not** help with right leg.
When the operator switches intention (number key), they must also navigate to a
completely different region of strategy space.  Using the wrong strategy for a class
suppresses its signal to zero regardless of which projection the students apply.

**z_strategy** is navigated continuously with the arrow keys.  It represents the
patient's *mental strategy* — how they are thinking about the movement, not just
what movement they intend.

### Intention vs strategy — why they are separate

This distinction is the core of the simulation:

- **Intention** (number key) = *what* movement the patient is trying to perform.
  This sets the class label attached to each data sample.
- **Strategy** (arrow keys) = *how* the patient is trying to perform it.
  This determines whether the neural signal actually reflects the intention.

In a real session the clinician announces the intention ("now: left hand") — that is
the label.  Whether the patient's brain signal *looks like* left hand depends entirely
on the mental strategy they use.  The neurofeedback teaches them to find better
strategies.

Crucially, using the arrow keys to fight disturbances does **not** corrupt the label,
because the label comes only from the number key, not from how many arrows are pressed.

### The observation model (latent → 256 dims)

The observed 256-dimensional signal `x` is generated as:

```
x = A  @  R(z_strategy)  @  diag(scale)  @  z  +  noise
```

- **A** — a fixed random 256×8 mixing matrix (unknown to students).
  Represents the electrode array geometry scrambling the underlying signal.

- **R(z_strategy)** — a strategy-dependent rotation matrix built from three
  [Givens rotations](https://en.wikipedia.org/wiki/Givens_rotation) in planes
  `(0,5)`, `(1,6)`, `(2,7)`.  When the player is far from the hidden good-strategy
  position, this rotation mixes the class signal into the noise dimensions.
  Different strategy positions require different projection directions to see the
  classes — which is why the students' projection must be *recomputed* continuously
  as strategy shifts.

- **scale** — suppresses the class signal when strategy is poor.  Implemented as a
  **leaky integrator** with time constant ~3 s: arriving at the right strategy position
  is not enough — the signal builds up gradually and requires the strategy to be *held*.
  Leaving too soon causes decay.  This creates genuine temporal dynamics without a
  prescribed sequence.

- **noise** — Gaussian observation noise added on top.

The result: finding the right 2D projection is not a one-time exercise.  As the
strategy shifts, the subspace in which classes are visible *rotates*, so the
projection must be continuously adapted.  This is exactly what happens in real BCI
co-adaptation.

### Temporal dynamics of the class signal

`class_scale` is a leaky integrator rather than an instantaneous function:

```
scale(t+1) = scale(t) + (dt / τ) * (strategy_quality³ - scale(t))
```

where τ = 3 seconds.  The consequences:

- **Build-up**: arriving at the correct strategy gives ~16% scale after 0.5 s,
  ~50% after 2 s, ~82% after 5 s.
- **Decay**: moving to the wrong position causes scale to fall back toward zero
  (~58% after 1 s away, ~21% after 4 s away).
- **Switching classes**: scale resets because the new class has a different target.
  The operator must navigate to the new target AND hold it before the signal
  for that class builds up.

This means the challenge is not just *where* to put the arrows but *when* and
for *how long* — temporal strategy matters.

### Why classes are hard to separate by default

At the default strategy position `(0, 0)` (arrows untouched):

| Metric | Value |
|---|---|
| Strategy quality | ~0.16 |
| Class signal scale | ~0.004 |
| PCA-2D accuracy | ~28% ≈ chance |

At the hidden good-strategy position:

| Metric | Value |
|---|---|
| Strategy quality | 1.00 |
| Class signal scale | 1.00 |
| PCA-2D accuracy | ~70% |

Students must guide the operator (arrow keys) toward that hidden position by watching
how well their projection separates the labeled classes.

### Disturbances (medium / hard)

The emulator adds structured disturbances to `z_class` every step:

- **Sinusoidal kicks** — three independent oscillations at a fixed frequency,
  each pushing in a random direction.  The operator must counteract these with
  continuous arrow input; holding still is not enough.
- **Random spikes** — sudden large kicks in a random direction.
- **Non-stationarity (hard only)** — the hidden good-strategy target drifts slowly
  in a circle over time.  The projection that worked 2 minutes ago stops working;
  students must detect and adapt to this drift.

### Difficulty levels

| | Easy | Medium | Hard |
|---|---|---|---|
| Observation noise std | 0.3 | 0.8 | 1.8 |
| Latent noise std | 0.03 | 0.12 | 0.28 |
| Sinusoidal disturbances | off | on | on (stronger) |
| Random spikes | off | rare | frequent |
| Non-stationarity | off | off | on |

---

## Installation

Requires Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pyzmq`, `pygame`.

---

## Running the emulator

```bash
# default: medium difficulty, 256 dimensions, ZMQ port 5555
python -m emulator

# easy mode
python -m emulator --difficulty easy

# hard mode, 128 dimensions, custom port
python -m emulator --difficulty hard --dims 128 --port 5556
```

### Controls

| Key | Action |
|-----|--------|
| `1` | Set intention → left hand |
| `2` | Set intention → right hand |
| `3` | Set intention → left leg |
| `4` | Set intention → right leg |
| `0` | Rest (no intention) |
| `←` `→` `↑` `↓` | Navigate strategy space (hold for continuous movement) |
| `ESC` / `Q` | Quit |

The **strategy quality** bar in the GUI shows how close the current arrow position
is to the hidden good-strategy target.  This is visible to the operator but would
not be shown to a real patient — it is there so you can verify the emulator is
behaving correctly.

---

## Receiving data in your GUI

The emulator publishes one JSON message per sample over a **ZMQ PUB** socket.

### Message format

```json
{
    "timestamp":   1713000000.123,
    "sample_idx":  42,
    "data":        [0.31, -1.22, 0.07, ...],
    "label":       0,
    "label_name":  "left_hand",
    "n_dims":      256,
    "sample_rate": 10,
    "difficulty":  "medium"
}
```

| Field | Type | Description |
|---|---|---|
| `timestamp` | float | Unix time of sample generation |
| `sample_idx` | int | Monotonically increasing counter |
| `data` | list[float] | The 256-dim neural signal — this is your input |
| `label` | int \| null | Intended class (0–3) or null for rest |
| `label_name` | str | `"left_hand"`, `"right_hand"`, `"left_leg"`, `"right_leg"`, `"rest"` |
| `n_dims` | int | Length of `data` |
| `sample_rate` | int | Samples per second (10 Hz) |
| `difficulty` | str | Active difficulty level |

### Minimal Python receiver

```python
import json
import numpy as np
import zmq

ctx    = zmq.Context()
socket = ctx.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")   # subscribe to all messages

while True:
    msg   = json.loads(socket.recv_string())
    data  = np.array(msg["data"])      # shape: (256,)
    label = msg["label"]               # int 0-3, or None
    name  = msg["label_name"]          # e.g. "left_hand"
    # your processing here
```

A runnable example with console output is in `receiver_example.py`.

### Receiving from other languages

Any language with a ZMQ SUB binding can connect.  The message is plain JSON so no
special deserialiser is needed.

**JavaScript (Node.js)**
```js
const zmq = require("zeromq");
const sock = new zmq.Subscriber();
sock.connect("tcp://localhost:5555");
sock.subscribe("");
for await (const [msg] of sock) {
    const { data, label, label_name } = JSON.parse(msg.toString());
}
```

**Julia**
```julia
using ZMQ, JSON
ctx  = Context()
sock = Socket(ctx, SUB)
connect(sock, "tcp://localhost:5555")
subscribe(sock, "")
while true
    msg = JSON.parse(String(recv(sock)))
    data, label = msg["data"], msg["label"]
end
```

---

## What students should build

The task is to build a GUI that:

1. **Receives** the data stream and accumulates a sliding window of labeled samples.
2. **Computes a projection** (e.g. PCA, LDA, UMAP) from the 256-dim signal to 2D/3D.
3. **Displays** the projected brain state in real time, coloured by label.
4. **Recomputes** the projection regularly — as the operator changes strategy, the
   subspace in which classes are visible rotates, so a static projection will go stale.
5. **Provides feedback** that helps the operator navigate toward a better strategy:
   e.g. a separability score, a confidence bar, or a cursor the operator tries to
   move to a target region.

The key insight students should discover: *there is no single projection that works
forever*.  The projection and the strategy must co-adapt — exactly what happens in
real neurofeedback.
