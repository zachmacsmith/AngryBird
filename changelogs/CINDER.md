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

## Session 2 — 2026-05-04

### Phase 8: mode-based continuous flight path planner
**Commit:** `b063f00`

Implements `docs/Drone Path.md`: replaces per-cycle station-to-station sorties with
persistent `DroneFlightState` that carries across cycle boundaries within a single sortie.

#### New types (`angrybird/types.py`)
- `DroneMode(str, Enum)` — NORMAL / RETURN / EMERGENCY, one-way FSM
- `DroneFlightState(@dataclass)` — position_m, remaining_range_m, mode, target_gs_idx,
  visited_domains, sortie_distance_m, returned
- `DronePlan` — new fields: `plan_distance_m`, `drone_mode`
- `PathSelectionResult` — new optional field: `updated_drone_states`

#### New config constants (`angrybird/config.py`)
`DRONE_CYCLE_DURATION_S` (1200), `DRONE_CYCLE_DISTANCE_M` (18 km), `DRONE_RETURN_THRESHOLD`
(0.35), `DRONE_SAFETY_FRACTION` (0.10), `DRONE_SAFETY_MARGIN_M` (2 km),
`DRONE_HOVER_POWER_RATIO` (0.85), `DRONE_MIN_USEFUL_INFO`, `DRONE_REVISIT_PERCENTILE` (95).

#### `angrybird/selectors/correlation_path.py` — full rewrite of select()
- `_check_mode_transitions()`: reserve_on_arrival FSM; GS lock override (physically unreachable);
  RETURN→EMERGENCY when r ≤ d_return + d_safety
- `_plan_greedy_path()`: unified greedy for NORMAL and RETURN (parameterised by budget and
  return_costs); RETURN uses target-GS distances + d_safety already baked into budget
- `_dijkstra_path()`: shortest-path Dijkstra for EMERGENCY direct routing to GS
- `_apply_revisit_threshold()`: re-admits visited domains where w_i > 95th percentile
- `_compute_all_gs_distances()`: Dijkstra from each GS domain; `_min_gs_return_costs()` aggregates
- `select()`: dispatches per drone by mode; builds `updated_drone_states`; loiter detection

#### `angrybird/orchestrator.py`
- `ground_stations` constructor parameter (list of extra GS grid coords)
- `_ground_stations_m` computed at init
- `_drone_states: Optional[list[DroneFlightState]]` — initialised lazily by selector on first call
- `registry.run()` passes `drone_states` and `ground_stations_m`; result's `updated_drone_states`
  fed into `_advance_drone_states()`
- `_advance_drone_states()`: drones with `returned=True` reset to full battery + NORMAL mode

#### Tests (`angrybird/tests/test_drone_path.py`) — 18 tests
- Mode transitions (7 tests): NORMAL→RETURN, RETURN→EMERGENCY, emergency terminal, GS lock
- Budget calculation (2 tests): NORMAL = d_cycle, RETURN = min(d_cycle, r − d_safety)
- Reachability invariant (1 test): endpoint within R_feasible of target GS
- EMERGENCY mode (2 tests): direct path + returned=True
- Visited-domain tracking (4 tests): accumulation, revisit threshold, cross-drone deconfliction
- Full sortie simulation (2 tests): distance bound, mode sequence one-way guarantee

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

---

### Restructure: simulation/ top-level package, demo scripts archived
**Commit:** `a45e287`

#### `simulation/` — promoted from `angrybird/simulation/`
- All 11 source files moved via `git mv` (history preserved).
- All `from ..X import` replaced with `from angrybird.X import` — the package now treats `angrybird` as an external dependency rather than a sibling subpackage.
- `simulation/__init__.py` docstring updated; relative imports within the package unchanged.
- **New:** `simulation/simple_fire.py` — `SimpleFire` + `_run_fire_member` extracted from `demo_sim.py` so `run_sim.py` no longer depends on the deprecated script.

#### `archived/` — deprecated files moved here
- `scripts/demo_sim.py`: was already marked DEPRECATED in its own docstring; agents kept running it instead of `run_sim.py`.
- `scripts/demo_phase2.py`: cycle-based phase-2 runner, superseded by `SimulationRunner`.
- `angrybird/_visualization_old.py`: backward-compat shim, only consumed by the two archived demo scripts.

#### `scripts/run_sim.py`
- Removed `from demo_sim import SimpleFire, make_gp`.
- `SimpleFire` imported from `simulation`; `make_gp` inlined (15 lines).
- All simulation imports updated to `from simulation import ...`.

#### `scripts/test_subsystems.py`
- `from angrybird.simulation.X` → `from simulation.X` (3 import lines).

#### `angrybird/visualization/core.py` + `__init__.py`
- Removed `_visualization_old` backward-compat shim (only consumer was `demo_phase2.py`, now archived).

---

## Session 3 — 2026-05-05

### Phase 9: SimulatedEnvironmentalSource, oracle removal, unified runner

#### `angrybird/prior/sources.py` — `SimulatedEnvironmentalSource`
- New class delivering frequency-gated, noisy sensor measurements to `DynamicPrior`:
  - NWP weather channel (1 hr): temperature + RH from ground truth with Gaussian noise.
  - NWP wind channel (5 min): speed + direction with spatially correlated noise.
  - Satellite FMC channel (daily): sparse footprint observations with elevated sigma.
- `collect_obs_store_inputs(timestamp, fire_state)` generates fire detection observations for three satellite tiers:
  - GOES fire (5 min, 40-cell pixel footprint, TP=0.92/FP=0.02).
  - VIIRS fire (12 hr, 7-cell pixel footprint, TP=0.95/FP=0.01).
  - Satellite FMC obs (daily, 10-cell stride, sigma=0.04).
- `latlon_to_cell()` helper for geographic coordinate conversion.

#### `angrybird/prior/__init__.py`
- Added `SimulatedEnvironmentalSource` to public exports.

#### `wispsim/drone_sim.py` — `collect_fire_observation()`
- One `FireDetectionObservation` per drone per timestep from thermal camera.
- TP rate=0.90, FP rate=0.05; confidence set to instrument accuracy (not filtered by ensemble).
- Reports both is_fire=True and is_fire=False so particle filter can reward and punish members.
- Returns `None` for idle drones.

#### `angrybird/orchestrator.py` — oracle fire state removal
- `run_cycle()` signature: `observations` is now the first positional argument; `fire_state` is keyword-optional (default `None`).
- Without `fire_state`: bootstraps from `obs_store.get_fire_detections()` → `reconstruct_arrival_time` → `initialize_from_reconstruction`.
- Warning + `_neutral_ensemble` fallback if no fire obs are available and no oracle state is passed.

#### `wispsim/runner.py`
- Imports `collect_fire_observation` from `drone_sim`; one call per drone per timestep.
- Fire detections added directly to `obs_store` (no oracle path).
- `_run_ignis_cycle()`: `orchestrator.run_cycle(observations=observations, start_time=...)` — `fire_state` arg removed.
- Burn probability for renderer sourced from `_last_ensemble_result.burn_probability`.
- `SimulationRunner.__init__()`: removed hardcoded ignition seed; now warns if `obs_store` is empty (seeding responsibility transferred to caller).

#### `angrybird/tests/test_dynamic_prior.py`
- Fixed 5 tests that used old positional call form `run_cycle(fire_state, [], ...)` → `run_cycle([], fire_state=fire_state, ...)`.

#### `scripts/run.py` — unified simulation entry point (NEW FILE)
Replaces `scripts/run_landfire_wispsim.py` and `scripts/run_sim.py`.

**Terrain modes:**
- LANDFIRE mode (default): loads GeoTIFF cache via `load_from_directory`.
- Synthetic scenario mode (`--scenario`): `hilly_heterogeneous`, `wind_shift`, `flat_homogeneous`, `dual_ignition`, `crown_fire_risk`.

**Shared helpers (no duplication):**
- `find_burnable_cell()` — spiral search from grid centre for first non-NB cell.
- `latlon_to_cell()` — flat-earth WGS-84 → (row, col) conversion.
- `cells_within_radius()` — all burnable cells within a report uncertainty radius.
- `make_gp()` — matched GP + ObservationStore pair.
- `make_registry()` — SelectorRegistry built with correct resolution and drone params.
- `seed_fire_report()` — adds `FireDetectionObservation` entries to obs_store before runner construction.

**Key args:**
- `--fire-lat/lon/radius-m/confidence` — uncertain fire report (not oracle).
- `--wind-speed/direction/temperature/humidity/base-fmc` — weather priors.
- `--wind-event-*` — single optional wind shift event.
- `--hours/cycle-min/horizon-min/seed` — simulation timing.
- `--drones/targets/drone-speed-ms/drone-endurance-s/n-raws/mesh-network` — fleet.
- `--members/selector/device` — ensemble and GPU.
- `--out` — output directory.
