# EMBER — Changelog

Agent name: **EMBER**
Assigned: 2026-05-04

---

## [2026-05-04] — ObservationStore migration (Phase 7 follow-up)

### Context
Implemented the architectural shift described in `docs/Observation Store.md`: moved all observation ownership out of `IGNISGPPrior` and into a centralized `ObservationStore`. The GP now reads from the store on every `predict()` call rather than accumulating observations internally.

---

### New files

#### `angrybird/observations.py`
Full implementation of the centralized observation layer:

- `ObservationType` enum — `FMC`, `WIND_SPEED`, `WIND_DIRECTION`, `FIRE_DETECTION`
- `ObservationSource` enum — `RAWS`, `DRONE`, `DRONE_MULTISPECTRAL`, `DRONE_ANEMOMETER`, `SATELLITE`
- `Observation` base frozen dataclass — `location`, `obs_type`, `source`, `value`, `sigma`, `timestamp`, `source_id`
- `RAWSObservation` subclass — never decays (sigma returned as-is)
- `FireDetectionObservation` subclass — adds `confidence` field
- `SatelliteObservation` subclass — adds `pixel_resolution_m` field
- `ObservationStore` — thread-safe (RLock) centralized store with:
  - `add_raws(station_id, obs_list)` — per-station RAWS ingestion
  - `add_drone_observations(obs_list)` — drone/satellite obs ingestion
  - `get_decayed_for_type(obs_type, current_time)` — returns `(locs, vals, sigmas)` with temporal decay applied: `effective_sigma = sigma * exp(age / tau)` for drone obs; RAWS obs never decay
  - `update_time(t)` — advances internal clock
  - `lock_for_cycle()` / `unlock_cycle()` — prevents concurrent ingestion during computation
  - `fork()` — returns a deep-copied mutable store (used by `LiveEstimator` for hypothetical obs without touching the live store)
- `ObservationSnapshot` dataclass — point-in-time snapshot for logging
- `IngestionBuffer` — pending-obs buffer (for future async ingestion)

---

### Modified files

#### `angrybird/gp.py`
- Removed all 18 internal observation lists (`_raws_fmc_locs`, `_fmc_locs`, `_fmc_vals`, etc.)
- Removed `add_raws()`, `add_observations()`, `_prune_and_decay()`, `update_time()` (original implementations)
- Removed `_current_time`, `_tau_*`, `_dirty` fields
- Constructor: `obs_store` parameter is now `Optional[ObservationStore] = None`; when `None`, auto-creates a default store with standard tau decay constants from config
- Added `_last_fmc_count`, `_last_ws_count`, `_last_wd_count` for kernel-locking: if obs count changes since last fit, resets the cached regressor to force fresh kernel optimization
- `fit()` reads from store via `self._store.get_decayed_for_type(...)` — no `_dirty` flag needed
- `predict()` always calls `fit()` — no caching
- `set_nelson_mean()` / `set_wind_prior_mean()` reset cached regressors to force fresh kernel when residual basis changes
- **Backward-compat wrappers** added (delegate to `self._store`):
  - `add_raws(locations, fmc_vals, ws_vals, wd_vals, ...)` — converts lists to `RAWSObservation` objects
  - `add_observations(locations, fmc_vals, fmc_sigmas, ...)` — converts lists to `Observation` objects
  - `update_time(t)` — delegates to store

#### `angrybird/assimilation.py`
- Added imports: `Observation`, `ObservationSource`, `ObservationStore`, `ObservationType`
- `assimilate_observations()` now supports two calling conventions automatically (type-sniffing on second positional arg):
  - New API: `(gp, obs_store, ensemble, observations, shape, ...)`
  - Old API: `(gp, ensemble, observations, shape, ...)` — uses `gp._store` as the store
- When observations are assimilated, they are converted to `Observation` objects and pushed to `obs_store.add_drone_observations()` instead of `gp.add_observations()`

#### `angrybird/orchestrator.py`
- Added `obs_store: ObservationStore` parameter to `IGNISOrchestrator.__init__()`
- `run_cycle()` passes `obs_store=self.obs_store` to `assimilate_observations()`

#### `angrybird/simulation/runner.py`
- Added imports for `Observation`, `ObservationSource as ObsSource`, `ObservationType`, `RAWSObservation`
- `LiveEstimator.compute_estimate()`: replaced `work_gp.add_observations()` with store-forking pattern — `work_store = self._live_gp._store.fork()`, assigns to deep-copied GP, adds hypothetical obs to `work_store`
- RAWS initialization: replaced `orchestrator.gp.add_raws(...)` with loop creating `RAWSObservation` objects per station and calling `orchestrator.obs_store.add_raws()`
- `_run_ignis_cycle()`: `gp.update_time(t)` → `obs_store.update_time(t)`

#### `angrybird/raws.py`
- Updated module docstring to reference new API (`obs_store.add_raws()` instead of `gp.add_raws()`)

#### `scripts/demo_sim.py`
- `make_gp(terrain)` now returns `tuple[IGNISGPPrior, ObservationStore]`
- `ObservationStore` created with decay config; passed to `IGNISGPPrior` and `IGNISOrchestrator`

#### `scripts/demo_phase2.py`
- `build_gp_prior()` creates `ObservationStore`, populates with `RAWSObservation` objects, passes to `IGNISGPPrior`

---

### Design decisions

- **Kernel locking without dirty flag**: instead of a `_dirty` bit (which required mutation hooks on every ingestion path), the GP tracks `_last_*_count` per variable. Count change → reset cached regressor → fresh kernel optimization on next fit.
- **Always-refit**: `predict()` always calls `fit()`. No caching. The kernel-locking mechanism keeps expensive hyperparameter optimization rare; the posterior update (cheap Cholesky solve) runs every cycle.
- **RAWS never decay**: `RAWSObservation` returns its `sigma` unchanged regardless of age, reflecting that RAWS stations are calibrated instruments with stable long-term readings.
- **Old-API compat**: rather than updating the 20+ test calls in `test_subsystems.py`, backward-compat wrappers were added to `IGNISGPPrior` and `assimilate_observations()` was made to auto-detect the calling convention.

---

### Test results
All 97 subsystem tests pass (`python -m scripts.test_subsystems --no-plot`). Runtime: ~4 s.
