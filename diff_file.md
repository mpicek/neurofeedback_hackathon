# Changes: Version 3 → Version 4

## emulator/dynamics.py

### Latent space structure
- **v4**: Flat class centroids — all four classes share the same 3-dim `z_class` subspace. `z_class` dims were `[coarse, fine, 0]`, with left/right separated by dim 1 and hand/leg by dim 0.
- **v4**: Hierarchical clusters — Cluster A (`left_hand`, `right_hand`) and Cluster B (`left_leg`, `right_leg`) are separated by `z_coarse` (dim 0, value ±2.0). Within each cluster, a dedicated fine dim separates the pair (`fine_A` for cluster A, `fine_B` for cluster B). Dim 0 is **never** rotated, so the coarse separation is always visible regardless of strategy.

### Optimal strategies — v4 initial (single per class)
- **v4**: Stationary — optimal is always `(0, 0)` for every class. Difficulty was achieved by pushing `z_strategy` away from origin via per-class temporal disturbance functions. The target ring in the GUI was always drawn at center.
- **v4 initial**: Per-class single corners. Each class had exactly one optimal position in `[-1,1]²`. `OPTIMAL_STRATEGIES` was a 4×2 numpy array. `strategy_quality` measured distance to the one class-specific corner. Disturbance functions removed.

### Optimal strategies — v4 extended (multiple per class, merge mechanic)

Each class now has **multiple** optimal strategy positions, each paired with a specific `z_class` centroid target. The structures changed from a single array to per-class dicts.

| Class | Index | # Strategies | Positions | Notes |
|-------|-------|--------------|-----------|-------|
| left_hand  | 0 | 2 | `(+0.7,+0.7)` primary, `(+0.7,−0.7)` merge | merge targets `[0,0,0]` |
| right_hand | 1 | 1 | `(−0.7,+0.7)` primary | |
| left_leg   | 2 | 1 | `(−0.7,−0.7)` primary | |
| right_leg  | 3 | 3 | `(−0.5,−0.5)` primary, `(−0.7,+0.7)` merge, `(+0.5,0.0)` alt | merge targets `[0,0,0]` |

**Centroid blending**: at each step, the system finds the nearest optimal for the active class and blends between the cluster centroid (fine dims = 0, classes within cluster indistinguishable) and the strategy-specific centroid, weighted by `strategy_quality`. This ensures classes are only separable when near a valid optimal position.

**Merge mechanic**: the merge strategies of class 0 and class 3 both target the shared centroid `_MERGE = [0, 0, 0]`. When either class uses its merge strategy, `z_class` converges to `[0,0,0]` — the same point for both. In the LDA/PCA projection on the receiver, classes 0 and 3 become visually indistinguishable (same cluster, no fine separation).

**Step order changed**: centroid quality is now evaluated *before* the spring fires. This ensures that holding `z_strategy` exactly at an optimal position gives `strategy_quality = 1.0`, guaranteeing exact convergence to the intended centroid (including the shared merge centroid).

**New data structures**:
- `OPTIMAL_STRATEGIES: dict[int, np.ndarray]` — per-class array of shape `(K, 2)`
- `STRATEGY_CENTROIDS: dict[int, np.ndarray]` — per-class per-strategy z_class targets, shape `(K, 3)`
- `CLUSTER_CENTROIDS: np.ndarray` — shape `(4, 3)`, fine dims = 0, used as fallback when off-strategy
- `_MERGE = [0, 0, 0]` — shared centroid for merge strategies

**New/changed properties**:
- `optimal_strategies` → `(K, 2)` all positions for the active class
- `optimal_strategy` → `(2,)` nearest position (kept for GUI backward compat)
- `_nearest_strategy_idx` stored on instance so `get_rotation()` reuses it without recomputing

### Spring mechanics
- **v4**: No spring; difficulty came from disturbance forces that actively pushed `z_strategy` around.
- **v4**: Spring toward `(0, 0)` with configurable `spring_rate`. `z_strategy` passively decays to center when arrow keys are released. The patient must actively navigate to and hold the correct corner.

### Rotation matrix (`get_rotation`)
- **v4 initial**: Error computed against the single class-specific optimal. Dim 0 never rotated.
- **v4 extended**: Error computed against `OPTIMAL_STRATEGIES[cls][_nearest_strategy_idx]` — the same nearest optimal used for centroid pulling. Otherwise identical Givens rotation logic.

---

## emulator/gui.py

### Target ring (optimal strategy circle)
- **v4**: Single ring always drawn at `(0, 0)`.
- **v4 initial**: Single ring drawn at the active class's per-class optimal corner.
- **v4 extended**: **One ring drawn per optimal strategy** for the active class. Rings for merge strategies (centroid == `_MERGE`) are drawn in **white**; all other rings use the class colour. This lets the player see all available targets and distinguish merge strategies visually.
- **Toggle button added**: a "Show optimal" checkbox in the bottom-left corner. When unchecked, all rings are hidden and the **Strategy quality** bar in the stats panel is also hidden (only Signal strength remains). State tracked via `show_optimal` (`tk.BooleanVar`, default `True`).

### Strategy bookmarks (Space key)
- **v4 (new)**: Pressing **Space** saves the current `z_strategy` position as a bookmark, tagged with the active class. Bookmarks are stored in `saved_marks: dict[int|None, list[np.ndarray]]` and persist for the session.
- Each bookmark is drawn on the strategy pad as a **diamond** in the active class's colour (white outline), rendered above the trail and below the current dot so it is always visible.
- Only marks for the **currently active class** are shown — switching class hides the previous class's marks and reveals any marks saved under the new class.

---

## receiver_gui.py

### CLI / key handling
- **v4**: Added `argparse` with `--local-keys` flag. Default behavior uses `pynput` for system-wide key capture (L/P work without the window needing focus). Falls back to matplotlib-local key events if `pynput` is not installed or `--local-keys` is passed.
