# GPU Fire Engine — Architecture, Upgrades, Bugs Fixed, and Open Questions

*Written: 2026-05-04 | Status: active development*

---

## What it is

`angrybird/fire_engines/gpu_fire_engine.py` is the primary ensemble fire spread engine. It replaces `SimpleFire` (wispsim) as a drop-in via `FireEngineProtocol`:

```python
result = engine.run(terrain, gp_prior, fire_state, n_members, horizon_min, rng)
# → EnsembleResult: arrival times (min), burn probability, fire types
```

It runs **N members in parallel** as a batch dimension on any PyTorch device (CPU, CUDA, MPS). Terrain is loaded to device once at construction; each `run()` call handles a full ensemble.

---

## How it works

### 1. Physics: Rothermel (1972) surface fire spread

For each ensemble member, the engine computes a **head-fire rate of spread** (m/s) at every grid cell using the Rothermel 1972 equations:

- **Fuel load** `w0` (kg/m²), **surface-area-to-volume ratio** `σ` (1/m), **depth** `δ` (m): from the fuel model lookup table
- **Packing ratio** `β = w0 / (δ ρ_p)` and its optimal `β_op = 3.348 σ^{-0.8189}`
- **Reaction intensity** `I_R` (kJ/m²·min): combines combustion efficiency (`γ`), moisture damping (`η_m`), mineral damping (`η_s`)
- **Wind factor** `φ_w` and **slope factor** `φ_s`: amplify spread in the wind / upslope direction
- **Wind + slope are resolved as 2-D vectors** before summation, giving the correct resultant head-fire direction `θ_max`

The result is a head-fire ROS `R` and direction `θ_max`. An **Anderson (1983) elliptical spread** model then projects this onto the fire front's local outward normal:

```
R(θ) = R_head · (1−e) / (1−e·cos θ)
```

where `e` is the ellipse eccentricity from midflame wind speed.

### 2. Crown fire: Van Wagner (1977)

When surface fire intensity exceeds a critical threshold `I_crit = (0.01 · CBH · (460 + 2590))^1.5`, crown fire is triggered and ROS switches to the Van Wagner crown ROS.

### 3. Level-set propagation

Fire is tracked as a **signed distance function** `φ(x, y)`:
- `φ < 0`: burned
- `φ > 0`: unburned  
- `φ = 0`: the fire front

At each timestep:
```
φ_new = φ - dt · R_dir · |∇φ|_Godunov
```

The **Godunov upwind scheme** ensures numerically correct front propagation (no spurious wave speeds). The gradient magnitude `|∇φ|` and the central-difference components (for the outward normal direction) are computed in a single **fused pass** (4 roll+clone operations instead of 8).

**Redistancing** (Sussman-Smereka-Osher 1994) restores `|∇φ| = 1` every 20 steps to prevent SDF drift. Only the exterior narrowband (`φ > 0`, within 5 cells of the front) is updated — applying the scheme to the interior `(φ < 0)` was using the wrong characteristic direction and pushing burned cells back across the front (see bugs below).

### 4. Ensemble perturbation

For each of the N members, FMC, wind speed, and wind direction are sampled from the GP posterior:

```
fmc_i = μ_fmc + ε_i · σ_fmc,   ε_i ~ N(0,1)
```

All Rothermel quantities (`ros_s`, `θ_max`, eccentricity, crown ROS, `I_crit`) are **precomputed once** before the loop since FMC/wind are fixed per member. The loop only runs the gradient + level-set advance.

### 5. CFL adaptive timestep (fixed dt optimisation)

The maximum ROS across all ensemble members is computed **once** before the loop (a single GPU→CPU sync). `dt = CFL · dx / max_ros`. This replaces the per-step `max().item()` sync that caused ~500 serialisation points per run.

---

## SB40 Upgrade

Previously the engine used **Anderson-13** fuel models (codes 1–13). LANDFIRE delivers **Scott & Burgan 40** (FBFM40, codes 91–204). This created a silent correctness bug: SB40 codes were being clipped to 13 (the maximum Anderson code), so every cell mapped to FM13 (heavy slash, ~0.322 kg/m² and steep slopes), causing extreme ROS.

### What changed

**`config.py`** — Added `SB40_FUEL_PARAMS` with all 40 fuel models (5 NB + 35 burnable) using the full per-class parameter schema from Scott & Burgan (2005):

```python
SB40_FUEL_PARAMS[183] = _fm(
    0.50,  2.20, 2.80,  # load_1h, load_10h, load_100h  (tons/acre)
    0.00,  0.00,        # load_live_herb, load_live_woody
    2000,  9999, 9999,  # sav_1h, sav_live_herb, sav_live_woody (1/ft)
    0.3,   20,   8000,  # depth (ft), mx (%), h (BTU/lb)
    False,              # static (not dynamic)
)  # → TL3 (timber litter)
```

**`gpu_fire_engine.py`** — `_build_fuel_table()` now builds a **(205, 8)** table covering both Anderson-13 and SB40. SB40 entries are aggregated using the **Rothermel (1972) area-weighted characteristic SAV**:

```
σ̄ = Σ(w_i · σ_i²) / Σ(w_i · σ_i)    [dead classes: 1h, 10h, 100h]
w₀ = Σ w_i                              [total dead load]
```

This correctly weights fine fuels (1h, σ~6500 1/m) much more heavily than coarse fuels (100h, σ~98 1/m) when computing the effective SAV that drives spread rate.

**`fuel_idx` clamped to 0–204** (was 0–13): unknown codes map to row 0 (zero load, no fire) instead of FM13.

---

## Bugs Found and Fixed

### Bug 1: NaN propagation from non-burnable cells

**Symptom:** 0% burn probability on every LANDFIRE run. NaN count in `φ_new` grew from 14 → 56 → 112 → 234 over 5 steps.

**Root cause:** Non-burnable cells (NB codes 91, 98, 99) have `σ = 0`, `w₀ = 0`. This causes:
- `β_ratio = β / β_op → 0^{−E} = ∞` in `φ_w` (wind factor)
- `β^{−0.3} = 0^{-0.3} = ∞` in `φ_s` (slope factor)  
- When `sin(θ) = 0` at certain wind directions: `∞ × 0 = NaN`
- NaN in `θ_max` propagates through `_directional_ros` even when `ros_head = 0`

**Fix:** Zero out both multipliers for NB cells using `torch.nan_to_num(..., posinf=0.0)`:
```python
phi_w = torch.nan_to_num(C * ws_midflame.pow(B) * beta_ratio.pow(-E), nan=0.0, posinf=0.0)
phi_s = torch.nan_to_num(5.275 * beta.pow(-0.3) * tan_slope_sq, nan=0.0, posinf=0.0)
```
This makes `rx = ry = 0` for NB cells → `θ_max = 0°` (finite) → `ros_dir = 0 × finite = 0`.

### Bug 2: Redistancing pushback on burned cells

**Symptom:** Fire front froze or shrank after redistancing, eventually collapsing back to the ignition point.

**Root cause:** The Godunov upwind scheme in `_phi_gradient` uses the **exterior characteristic direction** (appropriate for `φ > 0`). Applying Sussman redistancing to `φ < 0` cells uses the wrong characteristics, effectively pushing burned pixels back toward zero, undoing level-set crossings.

**Fix:** Restrict redistancing to exterior narrowband only:
```python
in_band = (phi > 0) & (phi < (n_iters * dx + dx))
```

### Bug 3: Anderson-13 codes clipped from SB40 (silent correctness bug)

**Symptom:** Realistic ROS on synthetic terrain but nonsense physics — all fuels acting like FM13 (heavy slash).

**Root cause:** `np.clip(fuel_model, 0, 13)` in the original `__init__` clipped SB40 codes (91–204) to 13. Corrected by building a 205-row table and clamping to `[0, 204]`.

### Bug 4: WAF lookup missing SB40 codes

**Symptom:** Wind adjustment factor defaulted to 0.6 everywhere on LANDFIRE terrain (open terrain WAF), regardless of canopy cover.

**Root cause:** The WAF lookup dict only had keys 1–13. SB40 codes returned 0 → defaulted to 0.6.

**Fix:** Prefer `terrain.canopy_cover` directly (available from both LANDFIRE and `synthetic_terrain`):
```python
cc_np = np.asarray(terrain.canopy_cover, dtype=np.float32)
waf_np = np.where(cc_np > 0.5, 0.4, np.where(cc_np > 0.1, 0.5, 0.6))
```

---

## Open Questions / Still Unsure About

### 1. Live fuel in the simplified Rothermel

The SB40 aggregation currently drops live herb and live woody loads from the spread calculation. For dynamic grass models (GR1–GR9, GS1–GS4, SH1, SH9), live fuels can dominate the fuelbed and significantly affect ROS, especially mid-season before curing. The current implementation is **conservative (under-predicts ROS for dynamic grass models)**. A proper implementation would apply the curing fraction to live herb load.

### 2. Multi-class Rothermel vs. simplified single-class

The aggregation `σ̄ = Σ(w_i σ_i²) / Σ(w_i σ_i)` is correct for the characteristic SAV, but the reaction intensity `I_R` in the full multi-class Rothermel is computed per-class and summed. In the simplified 1-class version, using total dead load `w₀` with `σ̄` overestimates `β` (packing ratio) when fine fuels dominate SAV but coarse fuels dominate load. This can push `β/β_op >> 1` (over-packed), reducing `γ` and under-predicting `I_R`. **TL3 shows this issue** (w₀=1.23 kg/m², β~0.026, β_opt~0.003). For timber litter models (TL1–TL9) the spread rate may be understated.

### 3. Performance at full LANDFIRE resolution

The 249×322 LANDFIRE grid at 100m resolution has ~80K cells. With N=50 members and a 60-min horizon at the CFL-limited dt (~2–5 s), that's roughly 720–1800 loop iterations, each doing 4 roll+clone ops on an (N, R, C) tensor. This should be manageable on CPU in a few minutes and near-interactive on MPS/CUDA. Not benchmarked yet.

### 4. Redistancing convergence

The 5-iteration Sussman redistancing reduces `|∇φ|` deviation from ~15% to ~7% in a single call. Over many calls (every 20 CFL steps), it should accumulate to near-unit gradient, but no convergence analysis has been done. The `_REDISTANCE_EVERY = 20` constant was chosen empirically.

### 5. Sub-cell arrival time accuracy

Arrival times use linear interpolation at the zero crossing: `t_cross = t + dt · φ / (φ − φ_new)`. This is first-order accurate. For cells near the ignition point where `|∇φ|` may be poorly resolved due to the redistancing being exterior-only, arrival times could have O(dx) error (50–100m at LANDFIRE resolution).

---

## Test Suite

`angrybird/tests/test_gpu_fire_engine.py` — 43 tests, all passing.

| Class | Tests | What's covered |
|---|---|---|
| `TestFuelTable` | 5 | Table shape, Anderson-13 populated, SAV units |
| `TestRothermelROS` | 6 | Positive ROS, wind/slope monotonicity, vector direction |
| `TestGradientKernels` | 3 | Godunov vs. central-diff consistency |
| `TestRedistancing` | 2 | `|∇φ|` convergence (ring SDF, exterior only), zero-crossing preservation |
| `TestDirectionalROS` | 4 | Head/backing/flank/isotropic |
| `TestEngineConstruction` | 3 | CPU build, CUDA fallback, fuel param shape |
| `TestEngineRun` | 9 | Output shapes, sentinels, bounded arrival times |
| `TestCircularSpread` | 1 | Symmetric spread (no wind, no slope) |
| `TestDirectionalSpread` | 1 | Downwind faster than upwind |
| `TestInitialPhi` | 3 | Per-member SDF seeding |
| `TestReproducibility` | 2 | Same/different seed behaviour |
| `TestFullStack` | 3 | Ensemble variance, fire types, horizon sensitivity |
