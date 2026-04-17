"""
Difficulty configurations for the Brain Emulator — Version 2.

z_strategy is always attracted back to (0, 0) at constant speed (not exponential).
spring_rate is in units/second; at 10 Hz each step pulls by spring_rate * dt.
strategy_speed must exceed spring_rate * dt for the player to hold any position.

With strategy_speed=0.15 and dt=0.1:
    d1 (spring=0.6): return speed 0.06/step — easy to hold, ~1.5 s to return from corner
    d5 (spring=1.4): return speed 0.14/step — hard to hold, ~0.65 s to return from corner
"""

from dataclasses import dataclass


@dataclass
class DifficultyConfig:
    name: str

    # Spring rate: z_strategy moves toward (0,0) at spring_rate units/second.
    # Higher → faster return to center → harder to hold optimal corner.
    spring_rate: float = 1.0

    # Observation noise (added to final 256-dim signal)
    gaussian_noise_std: float = 0.5

    # Latent noise on z_class (background neural noise)
    latent_noise_std: float = 0.05

    # How fast z_class is pulled toward the active class centroid
    class_pull_strength: float = 0.28

    # How fast arrow keys move z_strategy
    strategy_speed: float = 0.15

    # How steeply rotation reacts to z_strategy deviation from optimal
    strategy_sensitivity: float = 2.0

    n_classes: int = 4


# ---------------------------------------------------------------------------
# Five difficulty levels
# ---------------------------------------------------------------------------

DIFFICULTIES: dict[str, DifficultyConfig] = {
    "d1": DifficultyConfig(
        name="d1",
        spring_rate=0.6,
        gaussian_noise_std=0.35,
        latent_noise_std=0.04,
        class_pull_strength=0.30,
        strategy_speed=0.15,
        strategy_sensitivity=1.8,
    ),
    "d2": DifficultyConfig(
        name="d2",
        spring_rate=0.8,
        gaussian_noise_std=0.50,
        latent_noise_std=0.06,
        class_pull_strength=0.26,
        strategy_speed=0.15,
        strategy_sensitivity=2.0,
    ),
    "d3": DifficultyConfig(
        name="d3",
        spring_rate=1.0,
        gaussian_noise_std=0.65,
        latent_noise_std=0.08,
        class_pull_strength=0.24,
        strategy_speed=0.15,
        strategy_sensitivity=2.2,
    ),
    "d4": DifficultyConfig(
        name="d4",
        spring_rate=1.2,
        gaussian_noise_std=0.85,
        latent_noise_std=0.10,
        class_pull_strength=0.22,
        strategy_speed=0.15,
        strategy_sensitivity=2.5,
    ),
    "d5": DifficultyConfig(
        name="d5",
        spring_rate=1.4,
        gaussian_noise_std=1.10,
        latent_noise_std=0.12,
        class_pull_strength=0.20,
        strategy_speed=0.15,
        strategy_sensitivity=2.8,
    ),
}

# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

CLASS_NAMES: dict[int, str] = {
    0: "left_hand",
    1: "right_hand",
    2: "left_leg",
    3: "right_leg",
}

CLASS_COLORS: dict[int | None, tuple[int, int, int]] = {
    0:    (100, 149, 237),   # cornflower blue  — left_hand
    1:    (255, 165,   0),   # orange           — right_hand
    2:    ( 50, 205,  50),   # lime green       — left_leg
    3:    (220,  20,  60),   # crimson          — right_leg
    None: (120, 120, 120),
}
