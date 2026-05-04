# DISPATCH — Changelog

Agent name: **DISPATCH**
Assigned: 2026-05-04

---

## [2026-05-04] — Fire State Estimation (Phase 8)

### Context

Implemented the design specified in `docs/Fire State Estimation.md`. Prior to this phase, all ensemble members started every cycle from an identical, deterministic fire front. Ensemble spread came entirely from FMC/wind uncertainty. Fire location uncertainty — substantial between satellite passes — was ignored, causing the system to underestimate total prediction uncertainty and never route drones to confirm fire position even when perimeter uncertainty was the dominant source of error.

This phase adds per-member fire state management across cycles, arrival time reconstruction from sparse observations via fast marching, and particle filter reweighting for incremental updates. The information field is extended to include fire location entropy, enabling the system to automatically trade off prognostic measurements (FMC ahead of the fire) against diagnostic ones (confirming the perimeter).

---

### New files

#### `angrybird/fire_state.py`

Full implementation of the fire state estimation layer:

- `compute_ros_field(terrain, fmc, wind_speed, wind_direction)` — NumPy port of the Rothermel (1972) surface ROS kernel from `gpu_fire_engine._rothermel_ros`. Computes per-cell ROS (m/s) from GP-mean environmental fields. Used by the fast march reconstruction; no ensemble batching, CPU-only. Wind direction accepted for interface consistency but not used by the isotropic scalar model (same convention as the GPU engine).

- `draw_correlated_field(grid_shape, correlation_length, resolution, rng)` — FFT spectral method for drawing spatially correlated unit-variance Gaussian random fields. Filter kernel: `S(k) ∝ exp(-0.5 · k² · L²)`. Used to perturb arrival times at the uncertain perimeter when resetting the ensemble.

- `FireStateEstimator` — maintains the best-estimate arrival time field, tracks per-cell confidence and observation recency, and drives reconstruction:
  - `set_ignition(cell, time)` — seeds the arrival field at a known ignition point
  - `reconstruct_arrival_time(fire_obs, current_time, terrain, gp_prior)` — anchors observed fire/no-fire cells, computes ROS field, fast-marches from observed sources to fill unobserved cells, then computes per-cell uncertainty
  - `_fast_march(arrival, ros_field)` — Dijkstra on the grid with edge weights `dx / ROS`; 8-connected neighbourhood with Euclidean distance factors
  - `_compute_uncertainty(fire_obs, current_time, ros_field)` — three-source uncertainty: observation noise (~30 s satellite), propagation uncertainty (0.6 s/m from nearest observation), and temporal uncertainty (grows with time since last observation)

- `EnsembleFireState` — manages per-member arrival times (seconds) and signed distance fields across cycles:
  - `initialize_from_ignition(cell)` — cycle 1 with known ignition; all members identical
  - `initialize_from_fire_state(fire_state)` — cycle 1 from 2D burn mask (backward-compat path); all members identical
  - `initialize_from_reconstruction(arrival_field, uncertainty, current_time)` — hard reset; each member gets reconstructed arrival time perturbed by a correlated noise field scaled by local uncertainty; deep interior and far exterior cells are clamped (no perturbation where fire state is unambiguous)
  - `carry_forward(member_arrival_times_min)` — continue mode; converts EnsembleResult minutes → internal seconds
  - `get_initial_phi(current_time_s)` — recomputes per-member SDF from arrival times and returns `(N, rows, cols)` array for the fire engine
  - `resample(indices)` — post-particle-filter resampling of both arrival times and SDF
  - `_recompute_phi_from_arrival_times(current_time_s)` — converts per-member arrival times to signed distance fields using `distance_transform_edt`

- `ConsistencyChecker` — compares new fire observations against ensemble consensus:
  - `check(fire_obs, ensemble_result, current_time_s)` — returns `(should_reset, disagreement_fraction)`; triggers reset when fraction of contradicted observations exceeds threshold (default 20%); requires minimum observation count (default 5) to avoid spurious resets; converts EnsembleResult minutes to seconds internally

- `particle_filter_fire(ensemble_result, fire_obs, current_time_s, n_members)` — reweights members by likelihood of their fire state given observations; resamples only when effective sample size drops below 50% of N to preserve diversity; returns `(indices, n_eff)`

- `systematic_resample(weights, n)` — standard particle filter systematic resampling

All internal times are in **seconds**, matching observation timestamps and the `start_time` passed to the orchestrator. Conversion from EnsembleResult **minutes** is handled at the boundary in `ConsistencyChecker`, `particle_filter_fire`, and `carry_forward`.

---

### Modified files

#### `angrybird/orchestrator.py`

- Added imports: `ConsistencyChecker`, `EnsembleFireState`, `FireStateEstimator`, `particle_filter_fire` from `fire_state`
- `FireEngineProtocol.run()` — added `initial_phi: Optional[np.ndarray] = None` parameter
- `IGNISOrchestrator.__init__()`:
  - Constructs `FireStateEstimator`, `EnsembleFireState`, `ConsistencyChecker` with `max_arrival = 2 × horizon_min × 60` seconds (mirrors the GPU engine's 2× sentinel, converted to seconds for fire state internals)
  - Adds `_last_cycle_time_s: float = 0.0` and `_last_ensemble_result: Optional[EnsembleResult] = None` for inter-cycle state
  - Adds `fire_state_alpha: float = 0.0` — set > 0 to enable fire location entropy in the information field
- `run_cycle()`:
  - On first call: initializes `EnsembleFireState` from the caller-supplied `fire_state` burn mask
  - Fetches fire detections since `_last_cycle_time_s` via `obs_store.get_fire_detections(since=...)`
  - When detections exist and a prior ensemble result is available: runs `ConsistencyChecker`; on disagreement > threshold, calls `FireStateEstimator.reconstruct_arrival_time` and hard-resets the ensemble; otherwise runs `particle_filter_fire` and resamples
  - Passes `initial_phi` (N, rows, cols) to `fire_engine.run()` when ensemble fire state is initialized; falls back to None (old broadcast path) otherwise
  - After each ensemble run: calls `carry_forward` and stores `_last_ensemble_result`
  - Computes `fire_state_burn_prob` from per-member arrival times when `fire_state_alpha > 0`, and passes it to `compute_information_field`

#### `angrybird/simulation/gpu_fire_engine.py`

- `GPUFireEngine.run()`: added `initial_phi: Optional[np.ndarray] = None` parameter
- When `initial_phi` is provided: loads it directly as `phi = torch.tensor(initial_phi, ...)`, deriving `burned_np` from the ensemble union (`any(axis=0)` of members with phi < 0) for arrival time seeding. All downstream computation (ROS, level-set update, CFL) is unchanged — it already operates on (N, rows, cols) tensors.
- When `initial_phi` is None: existing SDF broadcast path is unchanged.

#### `angrybird/information.py`

- `compute_information_field()` — added parameters:
  - `fire_state_alpha: float = 0.0` — weight for fire location uncertainty entropy
  - `fire_state_burn_prob: Optional[np.ndarray] = None` — P(cell currently burning) from `EnsembleFireState` at cycle start
- When `fire_state_alpha > 0` and `fire_state_burn_prob` is supplied, adds `fire_state_alpha × H_binary(burn_prob)` to `w`. Cells at the uncertain fire perimeter have high binary entropy and attract drones to confirm fire position. All existing terms (FMC, wind, bimodal) unchanged; both new parameters default to 0 so existing callers are unaffected.

---

### Design decisions

- **Seconds as the internal unit**: observation timestamps, `start_time`, and all `FireState` internals use seconds. EnsembleResult uses minutes (GPU engine convention). Conversion is explicit and localized to three boundary points (`ConsistencyChecker.check`, `particle_filter_fire`, `EnsembleFireState.carry_forward`) rather than scattered.

- **Backward compatibility**: `fire_state` parameter in `run_cycle` is unchanged. `initial_phi=None` in the fire engine preserves the old broadcast path. `fire_state_alpha=0` and `bimodal_alpha=0` keep the information field identical for existing callers. No existing tests require modification.

- **SDF recomputation cost**: `_recompute_phi_from_arrival_times` runs N `distance_transform_edt` calls every cycle (~5 ms each at 38K cells, ~1 s for N=200). The narrow-band optimization described in the spec (recompute only near each member's fire front) is not yet implemented.

- **`fire_state_alpha` off by default**: fire location entropy adds signal but also increases the information field's sensitivity to fire state uncertainty. Leaving it at 0.0 until calibrated against the existing FMC/wind terms avoids unintended changes to drone routing behavior.

- **Correlated perturbation at 500 m**: the correlation length for ensemble perturbation in `initialize_from_reconstruction` is fixed at 500 m (matching the spec). This produces fire front spread comparable to typical perimeter uncertainty; could be tuned per-scenario.

- **No modification to EnKF**: fire state assimilation uses the particle filter, not the EnKF. FMC/wind assimilation via EnKF is unchanged. The two mechanisms operate on separate state variables in the same cycle.

---

### Bugs found and fixed during test run

Two pre-existing bugs surfaced when running `demo_sim.py` after integration:

**`assimilate_observations` kwarg mismatch** — `orchestrator.py` (EMBER phase) passed `obs_store=self.obs_store` as a keyword argument, but the parameter in `assimilation.py` is named `obs_store_or_ensemble`. Fixed by switching to positional arguments.

**`EnsembleFireState` n_members mismatch** — `CycleRunner` builds the ensemble with a user-supplied `n_members` (e.g. 15), while `IGNISOrchestrator` defaults to `ENSEMBLE_SIZE=100`. `carry_forward` stored a (15, rows, cols) array, but `_recompute_phi_from_arrival_times` iterated `range(100)` and crashed on index 15. Fixed by updating `self.n_members` in `carry_forward` to match the actual array size.

---

### Known limitations (inherited from spec)

- **Fast march assumes static ROS**: reconstruction uses current GP-mean FMC/wind. For multi-day fires with significant diurnal variation, arrival time estimates near the perimeter will be biased.
- **Satellite temporal ambiguity**: arrival time is set to observation timestamp (upper bound). True arrival could have been hours earlier, introducing systematic bias in the reconstructed surface.
- **FMC/wind transient after hard reset**: the reconstruction used GP-mean parameters; ensemble members then have different FMC/wind. Momentarily inconsistent, self-corrects within one cycle.
