# Cinder — Changelog

Agent name: **Cinder**
Model: Claude Sonnet 4.6
First session: 2026-05-04

---

## Session 1 — 2026-05-04

### Phase 7: RAWS integration, GP temporal decay, wind-direction sensitivity, multi-ignition
**Commit:** `928f21f` → rebased onto `b45e404`, landed as `928f21f` on main

#### New files
- `angrybird/raws.py` — `RAWSStation`, `RAWSDataProvider` protocol, `RAWSObserver`, `place_raws_stations`. Drop-in swap between simulated and production telemetry providers.
- `docs/GP Temporal Decay Specification.md` — design document for the temporal decay subsystem.

#### `angrybird/config.py`
- RAWS noise constants: `RAWS_FMC_SIGMA` (0.01), `RAWS_WIND_SPEED_SIGMA` (0.5), `RAWS_WIND_DIR_SIGMA` (5.0).
- Temporal decay: `TAU` dict, `TAU_FMC_S` (3600 s), `TAU_WIND_SPEED_S` (7200 s), `TAU_WIND_DIR_S` (3600 s), `GP_OBS_DECAY_DROP_FACTOR` (10).
- Noise floors: `AGGREGATION_SIGMA_FLOOR` (0.015), `PROCESS_NOISE_FLOOR` (0.01).

#### `angrybird/gp.py`
- Separated RAWS vs. drone observation stores. RAWS obs replaced on each `add_raws()` call (no decay); drone obs accumulate and decay.
- `_prune_and_decay()`: effective sigma = `orig * exp(age / tau)`; obs pruned when sigma_eff > 10× original.
- Per-observation `alpha` array in `GaussianProcessRegressor` — decay is gradual downweighting, not a hard prune.
- `update_time(t)`: advances GP clock, marks dirty.
- `set_wind_prior_mean()`: mirrors `set_nelson_mean` for wind — GP fits residuals, predict adds prior back.
- `normalize_y=False` in locked-kernel refit path.
- Removed `_build_gp()` helper; inlined kernel construction so per-obs `alpha` arrays can be passed.

#### `angrybird/types.py`
- `EnsembleResult`: added `member_wind_dir_fields: Optional[np.ndarray]`.
- `StrategyEvaluation`: added `gp_var_before`, `gp_var_after`, `gp_var_reduction`.

#### `angrybird/information.py`
- Wind direction sensitivity channel with circular wrap to `[-180, +180]` (prevents 360° spike).
- `compute_observability()` returns `wind_dir` channel alongside `fmc` and `wind_speed`.
- `compute_information_field()` propagates the `wind_dir` channel through sensitivity, observability, burned-cell masking, priority overlay, and exclusion mask.

#### `angrybird/assimilation.py`
- Timestamps threaded through to `gp.add_observations()`.
- `AGGREGATION_SIGMA_FLOOR` replaces hardcoded `OBS_FMC_SIGMA / 2.0`.

#### `angrybird/orchestrator.py`
- `_neutral_ensemble()` populates `member_wind_dir_fields` (uniform 270°).

#### `angrybird/simulation/gpu_fire_engine.py`
- Wind direction perturbation computed and stored in `EnsembleResult.member_wind_dir_fields`.

#### `angrybird/simulation/fire_oracle.py` + `ground_truth.py`
- `GroundTruthFire` and `generate_ground_truth` accept a list of ignition cells for multi-ignition scenarios.
- `GroundTruth.ignition_cells` field added (used by `place_raws_stations` for exclusion).

#### `angrybird/simulation/runner.py`
- `SimulationRunner.__init__()`: places RAWS stations, seeds GP via `add_raws()`, snapshots RAWS-only GP variance for renderer baseline, sets uninformed wind prior (5 m/s / 270°).
- `SimulationRunner._run_ignis_cycle()`: advances GP clock, tracks `_prev_gp_var_sum` (monotonically decreasing), computes `gp_var_reduction` per cycle.
- `SimulationConfig`: added `n_raws`, `raws_locations` fields.

#### `angrybird/simulation/renderer.py`
- RAWS station markers (yellow diamonds labelled W1, W2…) on all four map panels.
- Convergence strip replaced by GP variance reduction plot: cumulative drone reduction vs. RAWS-only horizontal baseline.
- Per-run video naming (`out_dir.name.mp4`) replaces fixed `simulation.mp4`.

#### `angrybird/simulation/scenarios.py`
- `hilly_heterogeneous`: added 30° wind event at hour 2.
- `dual_ignition`: new scenario — two simultaneous fires, 45° wind shift at t=30 min, 1-hour duration, 5 drones.

#### `angrybird/simulation/__init__.py`
- Exports `dual_ignition`.

#### `scripts/run_sim.py`
- Auto-discovers scenario factories from `scenarios.py` via `inspect`.
- Default scenario changed to `"hilly_heterogeneous"`.
- Video path fixed to match per-run naming.

---

### Fix: `angrybird/tif_getter.py` — LANDFIRE terrain loader
**Commit:** `aa8a008`

Full rewrite. Original was entirely non-functional:

| Bug | Fix |
|-----|-----|
| Wrong API (`landfire.gov/lfdata/` does not exist) | Real LFPS endpoint at `lfps.usgs.gov` with async job workflow |
| Invented layer names | Real codes: `US_220DEM`, `US_220FBFM13`, `US_220SLPD`, `US_220ASP`, `US_220CC`, `US_220CBH`, `US_220CBD` |
| `BBOX` defined but never used | Passed as Esri JSON envelope to `Area_of_Interest` in `submitJob` |
| Naive HTML scraping | Submit async job → poll `jobs/{id}` → fetch `results/Output_File` → download zip |
| No `TerrainData` construction | `_build_terrain_data()`: reproject all layers to shared UTM grid, nodata masking, unit conversions (CBD ÷100, CC ÷100), slope/aspect derived from DEM as fallback, non-burnable FM codes (91–99) → 0 |
| Unused `import math, urlencode` | Removed |

Added `argparse` CLI for standalone use. Installed missing deps: `rasterio`, `pyproj`.
Note: `landfire-python` in `requirements.txt` does not exist on PyPI — should be removed.

---

### `CycleReport`: add `gp_prior`, `selection_result`, `start_time`
**Commit:** `971081a`

#### `angrybird/types.py`
- Three new optional fields on `CycleReport`:
  - `gp_prior: Optional[GPPrior] = None` — GP posterior used for the cycle's ensemble.
  - `selection_result: Optional[SelectionResult] = None` — primary strategy selection output.
  - `start_time: float = 0.0` — simulation clock at cycle start (seconds).

#### `angrybird/orchestrator.py`
- `run_cycle()` gains `start_time: float = 0.0` parameter.
- `CycleReport(...)` now passes all three new fields from in-function locals.
- Added `SelectionResult` to imports.

#### `angrybird/simulation/runner.py`
- `SimulationRunner._run_ignis_cycle()`: passes `start_time=self.current_time` to `orchestrator.run_cycle()`.
- `CycleRunner.run_cycle()`: passes `gp_prior` (local `predict()` result) and `selection_result=selection_results.get(primary_strategy)`.
