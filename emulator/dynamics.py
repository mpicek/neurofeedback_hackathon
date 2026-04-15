"""
Latent space dynamics for the brain emulator.

State vector z (8-dimensional):
  dims 0-2  →  z_class    : class signal (pulled toward the active class centroid)
  dims 3-4  →  z_strategy : controlled by arrow keys
  dims 5-7  →  z_noise    : slow random walk

Key design decisions
--------------------
1. Each class has its own hidden optimal strategy position.
   Finding the right strategy for left_hand does not help with right_leg.
   The operator must navigate to a different region when switching classes.

2. class_scale is a leaky integrator (time constant ~3 s).
   Arriving at the right strategy position is not enough — you must *hold* it
   for a few seconds before the class signal fully develops.  Leaving too soon
   causes the scale to decay even if you return.  This creates genuine temporal
   structure without prescribing an explicit sequence.
"""

import numpy as np
from .config import DifficultyConfig


# ---------------------------------------------------------------------------
# Class centroids in the 3-dim z_class subspace
# Dim 0: hand vs leg  (large gap → easy to separate)
# Dim 1: left vs right (small gap → hard to separate)
# ---------------------------------------------------------------------------
CLASS_CENTROIDS = np.array(
    [
        [ 2.0,  0.9,  0.0],  # 0 — left_hand
        [ 2.0, -0.9,  0.0],  # 1 — right_hand
        [-2.0,  0.9,  0.0],  # 2 — left_leg
        [-2.0, -0.9,  0.0],  # 3 — right_leg
    ],
    dtype=float,
)

# ---------------------------------------------------------------------------
# Per-class hidden optimal strategy positions
# Spread across different quadrants so no single arrow position works for all.
# These are unknown to students — they must be discovered through feedback.
# ---------------------------------------------------------------------------
GOOD_STRATEGIES = np.array(
    [
        [ 0.62, -0.41],   # 0 — left_hand
        [-0.55,  0.38],   # 1 — right_hand
        [ 0.24,  0.70],   # 2 — left_leg
        [-0.37, -0.63],   # 3 — right_leg
    ],
    dtype=float,
)

# Integration time constant for class_scale (seconds).
# The signal needs ~SCALE_TAU seconds at the right strategy before fully developing.
SCALE_TAU = 3.0

N_CLASS_DIMS    = 3
N_STRATEGY_DIMS = 2
N_NOISE_DIMS    = 3
N_LATENT        = N_CLASS_DIMS + N_STRATEGY_DIMS + N_NOISE_DIMS  # 8


def _givens(n: int, i: int, j: int, theta: float) -> np.ndarray:
    """Return an n×n Givens rotation matrix rotating in the (i, j) plane by theta."""
    R = np.eye(n)
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] =  c;  R[i, j] = -s
    R[j, i] =  s;  R[j, j] =  c
    return R


class LatentDynamics:
    """
    Simulates the patient's latent brain state.

    Public interface
    ----------------
    set_class(k)           — set the current movement intention (0-3) or None for rest
    update_strategy(delta) — arrow-key input: delta is a 2-vector in [-1, 1]
    step() → dict          — advance one time step; returns a snapshot of the state
    get_rotation()         — build the 8×8 strategy-dependent rotation matrix
    strategy_quality       — float in [0, 1] for the current class
    class_scale            — integrated scale (0-1), builds up over ~SCALE_TAU seconds
    z_full                 — full 8-dim latent vector
    """

    def __init__(self, config: DifficultyConfig, sample_rate: float = 10.0, seed: int = 42):
        self.cfg         = config
        self.dt          = 1.0 / sample_rate
        self.t           = 0.0

        rng = np.random.default_rng(seed)

        # --- Latent state ---
        self.z_class    = np.zeros(N_CLASS_DIMS)
        self.z_strategy = np.zeros(N_STRATEGY_DIMS)
        self.z_noise    = np.zeros(N_NOISE_DIMS)

        # --- Current intention ---
        self.current_class: int | None = None

        # --- Integrated class scale (leaky integrator) ---
        # Rises toward strategy_quality³ at the current class's target,
        # decays toward 0 when strategy is wrong or class is None.
        self._scale_integrated: float = 0.0

        # --- Structured disturbances ---
        self._dist_phases = rng.uniform(0, 2 * np.pi, 3)
        raw_dirs          = rng.standard_normal((3, N_CLASS_DIMS))
        self._dist_dirs   = raw_dirs / np.linalg.norm(raw_dirs, axis=1, keepdims=True)

        # --- Per-class good strategy targets (with non-stationarity offset) ---
        self._good_strategies = GOOD_STRATEGIES.copy()   # shape (4, 2)
        self._ns_angle        = 0.0

    # ------------------------------------------------------------------
    # Public controls
    # ------------------------------------------------------------------

    def set_class(self, class_idx: int | None) -> None:
        """Set the intended movement class (0-3) or None for rest."""
        self.current_class = class_idx

    def update_strategy(self, delta: np.ndarray) -> None:
        """Move z_strategy by delta (arrow keys). Clamped to [-1, 1]²."""
        self.z_strategy = np.clip(
            self.z_strategy + np.asarray(delta, float) * self.cfg.strategy_speed,
            -1.0, 1.0,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> dict:
        """Advance one time step and return a state snapshot."""
        cfg = self.cfg

        # 1. Non-stationarity: drift ALL class targets by the same circular offset
        self._ns_angle += cfg.nonstationarity_speed * self.dt
        if cfg.nonstationarity_amplitude > 0:
            offset = cfg.nonstationarity_amplitude * np.array(
                [np.cos(self._ns_angle), np.sin(self._ns_angle)]
            )
            self._good_strategies = np.clip(GOOD_STRATEGIES + offset, -1.0, 1.0)

        # 2. Pull z_class toward the active class centroid (or decay to rest)
        if self.current_class is not None:
            target = CLASS_CENTROIDS[self.current_class]
            self.z_class += cfg.class_pull_strength * (target - self.z_class)
        else:
            self.z_class *= 1.0 - cfg.class_pull_strength * 0.4

        # 3. Sinusoidal disturbances
        for i in range(3):
            phase = self._dist_phases[i] + 2 * np.pi * cfg.disturbance_frequency * self.t
            self.z_class += cfg.disturbance_amplitude * np.sin(phase) * self._dist_dirs[i]

        # 4. Random spikes
        if cfg.spike_probability > 0 and np.random.random() < cfg.spike_probability:
            direction = np.random.standard_normal(N_CLASS_DIMS)
            direction /= np.linalg.norm(direction)
            self.z_class += cfg.spike_amplitude * direction

        # 5. Latent Gaussian noise
        self.z_class += np.random.normal(0, cfg.latent_noise_std, N_CLASS_DIMS)

        # 6. Noise dims: slow AR(1) random walk
        self.z_noise = 0.85 * self.z_noise + np.random.normal(0, 0.25, N_NOISE_DIMS)

        # 7. Leaky integrator for class_scale
        #    Target = strategy_quality³ for current class (0 when no class active)
        target_scale = self.strategy_quality ** 3
        alpha        = self.dt / SCALE_TAU
        self._scale_integrated += alpha * (target_scale - self._scale_integrated)

        # 8. Advance time
        self.t += self.dt

        return {
            "z_class":           self.z_class.copy(),
            "z_strategy":        self.z_strategy.copy(),
            "z_noise":           self.z_noise.copy(),
            "current_class":     self.current_class,
            "strategy_quality":  self.strategy_quality,
            "class_scale":       self.class_scale,
            "t":                 self.t,
        }

    # ------------------------------------------------------------------
    # Rotation matrix
    # ------------------------------------------------------------------

    def get_rotation(self) -> np.ndarray:
        """
        Build an 8×8 rotation matrix based on how far the current strategy
        is from the active class's optimal position.

        At good strategy for the current class: rotation ~ identity
          → class signal projects cleanly
        Far from it: class dims rotate into noise dims
          → class signal buried regardless of projection

        When no class is active uses a large rotation (everything is buried).
        """
        if self.current_class is not None:
            target = self._good_strategies[self.current_class]
        else:
            target = np.zeros(2)   # rest: always bad strategy → full rotation

        err     = self.z_strategy - target
        scale   = self.cfg.strategy_sensitivity
        half_pi = np.pi / 2

        theta1 = half_pi * np.tanh(scale * err[0])
        theta2 = half_pi * np.tanh(scale * err[1])
        theta3 = half_pi * np.tanh(scale * (err[0] + err[1]) / 2)

        R = (
            _givens(N_LATENT, 0, 5, theta1)
            @ _givens(N_LATENT, 1, 6, theta2)
            @ _givens(N_LATENT, 2, 7, theta3)
        )
        return R

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def z_full(self) -> np.ndarray:
        return np.concatenate([self.z_class, self.z_strategy, self.z_noise])

    @property
    def strategy_quality(self) -> float:
        """
        How close the current strategy is to the optimal for the active class.
        Returns 0 when no class is active (rest).
        """
        if self.current_class is None:
            return 0.0
        target = self._good_strategies[self.current_class]
        error  = np.linalg.norm(self.z_strategy - target)
        return float(np.exp(-2.5 * error))

    @property
    def class_scale(self) -> float:
        """
        Integrated class signal scale (0–1).
        Builds up over ~SCALE_TAU seconds when strategy_quality is high,
        decays when strategy drifts away or class switches.
        """
        return float(self._scale_integrated)
