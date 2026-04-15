from dataclasses import dataclass


@dataclass
class DifficultyConfig:
    name: str

    # --- Observation noise (added to the final 256-dim signal) ---
    gaussian_noise_std: float

    # --- Latent noise (added to z_class each step) ---
    latent_noise_std: float

    # --- Structured disturbances (sinusoidal kicks to z_class) ---
    disturbance_amplitude: float   # magnitude of sinusoidal push
    disturbance_frequency: float   # Hz — how fast the oscillation cycles

    # --- Random spikes (sudden large kicks) ---
    spike_probability: float       # per-step probability
    spike_amplitude: float         # magnitude when a spike fires

    # --- Class dynamics ---
    class_pull_strength: float     # α: how fast z_class moves to target centroid (0–1)

    # --- Strategy ---
    strategy_speed: float          # how far arrows move z_strategy per frame
    strategy_sensitivity: float    # how sharply rotation reacts to strategy error

    # --- Non-stationarity (hard mode: good_strategy target drifts over time) ---
    nonstationarity_speed: float   # rad/s of circular drift
    nonstationarity_amplitude: float  # radius of drift circle

    # --- Classes ---
    n_classes: int = 4


DIFFICULTIES: dict[str, DifficultyConfig] = {
    "easy": DifficultyConfig(
        name="easy",
        gaussian_noise_std=0.3,
        latent_noise_std=0.03,
        disturbance_amplitude=0.0,
        disturbance_frequency=0.1,
        spike_probability=0.0,
        spike_amplitude=0.0,
        class_pull_strength=0.35,
        strategy_speed=0.06,
        strategy_sensitivity=1.5,
        nonstationarity_speed=0.0,
        nonstationarity_amplitude=0.0,
    ),
    "medium": DifficultyConfig(
        name="medium",
        gaussian_noise_std=0.8,
        latent_noise_std=0.12,
        disturbance_amplitude=0.25,
        disturbance_frequency=0.15,
        spike_probability=0.02,
        spike_amplitude=0.8,
        class_pull_strength=0.22,
        strategy_speed=0.04,
        strategy_sensitivity=2.5,
        nonstationarity_speed=0.0,
        nonstationarity_amplitude=0.0,
    ),
    "hard": DifficultyConfig(
        name="hard",
        gaussian_noise_std=1.8,
        latent_noise_std=0.28,
        disturbance_amplitude=0.55,
        disturbance_frequency=0.22,
        spike_probability=0.05,
        spike_amplitude=1.8,
        class_pull_strength=0.14,
        strategy_speed=0.03,
        strategy_sensitivity=4.0,
        nonstationarity_speed=0.04,
        nonstationarity_amplitude=0.38,
    ),
}

# 4 movement classes
# Structured so that:
#   - hand vs leg is easy (large separation on dim 0)
#   - left vs right is hard (small separation on dim 1)
CLASS_NAMES: dict[int, str] = {
    0: "left_hand",
    1: "right_hand",
    2: "left_leg",
    3: "right_leg",
}

CLASS_COLORS: dict[int | None, tuple[int, int, int]] = {
    0: (100, 149, 237),   # cornflower blue  — left hand
    1: (255, 165,   0),   # orange           — right hand
    2: ( 50, 205,  50),   # lime green       — left leg
    3: (220,  20,  60),   # crimson          — right leg
    None: (120, 120, 120),
}
