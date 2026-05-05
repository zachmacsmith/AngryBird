"""
Core orchestrator: the operational IGNIS cycle.

Phase 4a — ships with the production system.

The orchestrator sequences:
  GP prior → ensemble → information field → select → plan → assimilate → repeat

Observation-source agnostic: receives DroneObservations from the caller and never
generates them. In production, observations come from real UAV telemetry. In the
simulation harness (Phase 4b), SimulationRunner provides synthetic observations.

Seed management (PotentialBugs1 §8): a fresh Generator keyed on base_seed +
cycle_id is created each cycle so all strategies see an identical ensemble.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .assimilation import assimilate_observations
from .prior import DynamicPrior, EnvironmentalDataSource, StaticDataSource
from .config import (
    DRONE_RANGE_M,
    ENSEMBLE_SIZE,
    GP_DEFAULT_FMC_MEAN,
    GP_DEFAULT_WIND_DIR_MEAN,
    GP_DEFAULT_WIND_SPEED_MEAN,
    GRID_RESOLUTION_M,
    N_DRONES,
    SIMULATION_HORIZON_MIN,
)
from .fire_state import (
    ConsistencyChecker,
    EnsembleFireState,
    FireStateEstimator,
    particle_filter_fire,
)
from .gp import IGNISGPPrior
from .information import compute_information_field
from .observations import ObservationStore
from .path_planner import plan_paths, selections_to_mission_queue
from .selectors import registry as _default_registry
from .selectors.base import SelectorRegistry
from .types import (
    CycleReport,
    DronePlan,
    DroneFlightState,
    DroneMode,
    DroneObservation,
    EnsembleResult,
    GPPrior,
    InformationField,
    MissionRequest,
    SelectionResult,
    TerrainData,
)
from .utils import jaccard

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fire engine protocol (PotentialBugs1 §4)
# ---------------------------------------------------------------------------

@runtime_checkable
class FireEngineProtocol(Protocol):
    """
    Interface the orchestrator expects from the fire engine.
    The NumPy CPU engine and any future PyTorch GPU engine both satisfy this.
    The return type is always numpy-backed EnsembleResult regardless of backend.
    """

    def run(
        self,
        terrain: TerrainData,
        gp_prior: GPPrior,
        fire_state: np.ndarray,
        n_members: int,
        horizon_min: int,
        rng: Optional[np.random.Generator] = None,
        initial_phi: Optional[np.ndarray] = None,
    ) -> EnsembleResult: ...


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class IGNISOrchestrator:
    """
    Sequences the full IGNIS pipeline for one or many cycles.

    obs_store owns all observations. Call obs_store.add_raws() / add() /
    add_batch() externally before each cycle. run_cycle() passes the store
    to assimilation so drone observations collected this cycle are added to
    the store and the GP refits from it.

    Args:
        terrain:           static TerrainData (loaded once, never mutated)
        gp:                IGNISGPPrior backed by obs_store
        obs_store:         centralized observation store (shared with gp)
        fire_engine:       implements FireEngineProtocol
        selector_name:     which strategy drives the live mission queue
        selector_registry: plug-and-play registry of all strategies
        n_drones:          number of drones
        horizon_min:       fire-spread simulation horizon in minutes
        n_members:         ensemble size
        staging_area:      (row, col) drone launch/return point
        resolution_m:      grid cell size in metres
        base_seed:         base random seed; cycle_id is added each cycle
        ground_stations:   additional (row, col) GS positions; staging_area always included
    """

    def __init__(
        self,
        terrain: TerrainData,
        gp: IGNISGPPrior,
        obs_store: ObservationStore,
        fire_engine: FireEngineProtocol,
        selector_name: str = "greedy",
        selector_registry: SelectorRegistry = _default_registry,
        n_drones: int = N_DRONES,
        n_targets: Optional[int] = None,
        horizon_min: int = SIMULATION_HORIZON_MIN,
        n_members: int = ENSEMBLE_SIZE,
        staging_area: tuple[int, int] = (0, 0),
        resolution_m: float = GRID_RESOLUTION_M,
        base_seed: int = 0,
        bimodal_alpha: float = 0.5,
        bimodal_beta: float = 0.3,
        ground_stations: Optional[list[tuple[int, int]]] = None,
        data_source: Optional[EnvironmentalDataSource] = None,
    ) -> None:
        self.terrain          = terrain
        self.gp               = gp
        self.obs_store        = obs_store
        self.fire_engine      = fire_engine
        self.selector_name    = selector_name
        self.registry         = selector_registry
        self.n_drones         = n_drones
        # n_targets: how many locations the selector picks each cycle.
        # Defaults to n_drones so each drone gets 1 target; set higher for
        # multi-waypoint paths when n_drones=1.
        self.n_targets        = n_targets if n_targets is not None else n_drones
        self.horizon_min      = horizon_min
        self.n_members        = n_members
        self.staging_area     = staging_area
        self.resolution_m     = resolution_m
        self.base_seed        = base_seed
        self.bimodal_alpha    = bimodal_alpha
        self.bimodal_beta     = bimodal_beta
        self.fire_state_alpha = 0.0   # set > 0 to enable fire location entropy term
        self.cycle_count      = 0
        self._previous_selections: list[tuple[int, int]] = []
        self._drone_plans: list[DronePlan] = []

        # ── Ground stations (always include staging_area) ─────────────────
        _all_gs = [staging_area] + (ground_stations or [])
        # Deduplicate while preserving order (staging_area first)
        seen: set[tuple[int, int]] = set()
        self._ground_stations: list[tuple[int, int]] = []
        for gs in _all_gs:
            if gs not in seen:
                seen.add(gs)
                self._ground_stations.append(gs)
        self._ground_stations_m: list[np.ndarray] = [
            np.array([r * resolution_m, c * resolution_m], dtype=np.float64)
            for r, c in self._ground_stations
        ]

        # ── Persistent drone flight states (mode-based planner) ──────────
        # Initialised lazily on first cycle (need graph to be built).
        self._drone_states: Optional[list[DroneFlightState]] = None

        # ── Dynamic prior + data source ───────────────────────────────
        self.dynamic_prior = DynamicPrior(
            grid_shape=terrain.shape,
            resolution_m=resolution_m,
        )
        self.data_source: Optional[EnvironmentalDataSource] = data_source

        # ── Fire state estimation components ──────────────────────────────
        # max_arrival sentinel: 2× horizon in seconds (mirrors GPU engine's
        # 2× horizon_min sentinel, converted to seconds for FireState internals).
        _max_arrival_s = 2.0 * float(horizon_min) * 60.0
        self.fire_state_estimator = FireStateEstimator(
            grid_shape=terrain.shape,
            dx=resolution_m,
            max_arrival=_max_arrival_s,
        )
        self.ensemble_fire_state = EnsembleFireState(
            n_members=n_members,
            grid_shape=terrain.shape,
            dx=resolution_m,
            max_arrival=_max_arrival_s,
        )
        self.consistency_checker = ConsistencyChecker(
            disagreement_threshold=0.2,
            min_observations=5,
        )
        # Last simulation time (seconds) at which run_cycle was called.
        self._last_cycle_time_s: float = 0.0
        # Last ensemble result — used by ConsistencyChecker next cycle.
        self._last_ensemble_result: Optional[EnsembleResult] = None

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        observations: list[DroneObservation],
        fire_state: Optional[np.ndarray] = None,
        ensemble: Optional[EnsembleResult] = None,
        priority_weight_field: Optional[np.ndarray] = None,
        exclusion_mask: Optional[np.ndarray] = None,
        start_time: float = 0.0,
        weather_source: Optional[dict] = None,
    ) -> tuple[list[MissionRequest], CycleReport]:
        """
        Execute one full IGNIS cycle.

        Args:
            observations:          DroneObservations from the previous cycle
            fire_state:            DEPRECATED — oracle burn mask; pass None
                                   (or omit) to use obs_store fire detections.
                                   Still accepted for backward compatibility with
                                   CycleRunner; ignored when initial_phi is set.
            ensemble:              pre-computed EnsembleResult (SimulationRunner
                                   passes one to avoid double computation)
            priority_weight_field: operator priority overlay (>1 in priority regions)
            exclusion_mask:        operator exclusion zones (True = excluded)

        Returns:
            (list[MissionRequest], CycleReport) — requests for the UTM layer + diagnostics.
            CycleReport.evaluations is empty here; SimulationRunner fills it.
        """
        self.cycle_count += 1
        rng   = np.random.default_rng(self.base_seed + self.cycle_count)
        shape = self.terrain.shape
        _t = {}  # phase timing: name → wall seconds

        # 0. Update dynamic prior, then push derived fields into the GP as prior means.
        #    Prefer self.data_source (new API); fall back to weather_source dict (legacy).
        _t0 = time.perf_counter()
        active_source = self.data_source
        if active_source is None and weather_source is not None:
            # Legacy dict path: keep existing callers (e.g. tests) working
            self.dynamic_prior.update_cycle(
                current_time=start_time,
                terrain=self.terrain,
                weather_source=weather_source,
                ensemble_result=self._last_ensemble_result,
            )
        elif active_source is not None:
            self.dynamic_prior.compute_cycle(
                source=active_source,
                terrain=self.terrain,
                current_time=start_time,
                ensemble_result=self._last_ensemble_result,
            )
        if self.dynamic_prior.is_initialized():
            means = self.dynamic_prior.get_gp_prior_means()
            if means["fmc"] is not None:
                self.gp.set_nelson_mean(means["fmc"])
            if means["wind_speed"] is not None and means["wind_direction"] is not None:
                self.gp.set_wind_prior_mean(means["wind_speed"], means["wind_direction"])
        _t["dynamic_prior"] = time.perf_counter() - _t0

        # 1. Snapshot GP prior before assimilation (used for replan-flag comparison)
        _t0 = time.perf_counter()
        self.gp.fit(start_time)
        gp_prior_before = self.gp.predict(shape)
        _t["gp_fit_pre"] = time.perf_counter() - _t0

        # 2. Assimilate previous cycle's observations
        #    If ensemble not yet available use a neutral placeholder so the EnKF
        #    still works (thinned obs will be empty on cycle 1 anyway).
        _t0 = time.perf_counter()
        assim_ensemble = ensemble if ensemble is not None else _neutral_ensemble(shape, self.n_members)
        assim = assimilate_observations(
            self.gp,
            self.obs_store,      # obs_store_or_ensemble (new API)
            assim_ensemble,
            observations,
            shape=shape,
            resolution_m=self.resolution_m,
            gp_prior=gp_prior_before,
            rng=rng,
        )
        _t["assimilation"] = time.perf_counter() - _t0

        # 3. Updated GP prior (post-assimilation; used for ensemble + info field)
        _t0 = time.perf_counter()
        self.gp.fit(start_time)
        gp_prior = self.gp.predict(shape)
        _t["gp_fit_post"] = time.perf_counter() - _t0

        # ── Fire state management ─────────────────────────────────────────
        # Primary path: obs_store fire detections drive all fire state
        # estimation.  fire_state (oracle burn mask) is accepted only for
        # backward compatibility with CycleRunner and ignored when initial_phi
        # is available from the ensemble fire state.
        _t0 = time.perf_counter()
        initial_phi_for_engine: Optional[np.ndarray] = None

        if not self.ensemble_fire_state.initialized:
            if fire_state is not None:
                # Legacy / CycleRunner path: oracle burn mask available.
                self.ensemble_fire_state.initialize_from_fire_state(fire_state)
            else:
                # Primary path: reconstruct from all accumulated fire detections.
                seed_obs = self.obs_store.get_fire_detections()
                if seed_obs:
                    arrival_field = self.fire_state_estimator.reconstruct_arrival_time(
                        seed_obs, start_time, self.terrain, gp_prior)
                    self.ensemble_fire_state.initialize_from_reconstruction(
                        arrival_field,
                        self.fire_state_estimator.arrival_uncertainty,
                        current_time=start_time,
                    )
                    logger.info(
                        "Cycle %d | fire state INITIALIZED from %d observations",
                        self.cycle_count, len(seed_obs),
                    )
                else:
                    logger.warning(
                        "Cycle %d | no fire detections in store yet — "
                        "fire state deferred; ensemble will use null initial state",
                        self.cycle_count,
                    )

        # Incremental update: particle filter or hard reset from new obs
        if self.ensemble_fire_state.initialized and self._last_ensemble_result is not None:
            fire_obs = self.obs_store.get_fire_detections(since=self._last_cycle_time_s)
            if fire_obs:
                should_reset, disagreement = self.consistency_checker.check(
                    fire_obs, self._last_ensemble_result, start_time)

                if should_reset:
                    arrival_field = self.fire_state_estimator.reconstruct_arrival_time(
                        fire_obs, start_time, self.terrain, gp_prior)
                    self.ensemble_fire_state.initialize_from_reconstruction(
                        arrival_field,
                        self.fire_state_estimator.arrival_uncertainty,
                        current_time=start_time,
                    )
                    logger.info(
                        "Cycle %d | fire state HARD RESET | disagreement=%.0f%% | "
                        "n_obs=%d",
                        self.cycle_count, disagreement * 100, len(fire_obs),
                    )
                else:
                    indices, n_eff = particle_filter_fire(
                        self._last_ensemble_result, fire_obs,
                        start_time, self.n_members)
                    self.ensemble_fire_state.resample(indices)
                    logger.info(
                        "Cycle %d | fire state particle filter | "
                        "disagreement=%.0f%% | N_eff=%.0f",
                        self.cycle_count, disagreement * 100, n_eff,
                    )

        if self.ensemble_fire_state.initialized:
            initial_phi_for_engine = self.ensemble_fire_state.get_initial_phi(start_time)
        _t["fire_state"] = time.perf_counter() - _t0

        # 4. Ensemble — use caller-provided or run fire engine
        _t0 = time.perf_counter()
        if ensemble is None:
            if initial_phi_for_engine is None and fire_state is None:
                # No fire state available at all — cannot run ensemble.
                # This should only occur on cycle 1 if the obs_store has no
                # fire detections (e.g. ignition seed was not injected).
                logger.warning(
                    "Cycle %d | skipping ensemble: no fire state available",
                    self.cycle_count,
                )
                ensemble = _neutral_ensemble(shape, self.n_members)
            else:
                ensemble = self.fire_engine.run(
                    self.terrain, gp_prior,
                    fire_state,          # None OK: engine ignores when initial_phi set
                    self.n_members, self.horizon_min, rng,
                    initial_phi=initial_phi_for_engine,
                )
        _t["ensemble"] = time.perf_counter() - _t0

        # Carry forward per-member fire state for next cycle.
        self.ensemble_fire_state.carry_forward(ensemble.member_arrival_times)
        self._last_ensemble_result = ensemble
        self._last_cycle_time_s    = start_time

        # 5. Information field
        _t0 = time.perf_counter()
        fire_state_burn_prob: Optional[np.ndarray] = None
        if self.fire_state_alpha > 0.0 and self.ensemble_fire_state.initialized:
            mat = self.ensemble_fire_state.member_arrival_times
            if mat is not None:
                fire_state_burn_prob = (mat < start_time).mean(axis=0).astype(np.float32)

        info_field = compute_information_field(
            ensemble, gp_prior,
            resolution_m=self.resolution_m,
            horizon_minutes=self.horizon_min,
            priority_weight_field=priority_weight_field,
            exclusion_mask=exclusion_mask,
            bimodal_alpha=self.bimodal_alpha,
            bimodal_beta=self.bimodal_beta,
            fire_state_alpha=self.fire_state_alpha,
            fire_state_burn_prob=fire_state_burn_prob,
        )
        _t["info_field"] = time.perf_counter() - _t0

        # 6. Primary strategy selection
        _t0 = time.perf_counter()
        primary_result = self.registry.run(
            self.selector_name, info_field, self.gp, ensemble, k=self.n_targets,
            terrain=self.terrain,
            staging_area=self.staging_area,
            resolution_m=self.resolution_m,
            drone_states=self._drone_states,
            ground_stations_m=self._ground_stations_m,
        )
        _t["selection"] = time.perf_counter() - _t0

        # 7. Plan paths + build mission queue
        # Path selectors produce DronePlans directly; point selectors need the extra step.
        _t0 = time.perf_counter()
        if primary_result.kind == "paths":
            drone_plans = primary_result.drone_plans
            selected_locations = [
                p.waypoints[1] for p in drone_plans if len(p.waypoints) > 1
            ]
            if primary_result.updated_drone_states is not None:
                self._drone_states = self._advance_drone_states(
                    primary_result.updated_drone_states
                )
        else:
            drone_plans = plan_paths(
                primary_result.selected_locations,
                staging_area=self.staging_area,
                n_drones=self.n_drones,
                shape=shape,
                resolution_m=self.resolution_m,
            )
            selected_locations = primary_result.selected_locations

        mission_queue = selections_to_mission_queue(
            drone_plans,
            info_field, self.terrain, self.resolution_m,
        )
        _t["path_plan"] = time.perf_counter() - _t0

        total = sum(_t.values())
        logger.info(
            "Cycle %d timing (total=%.2fs): %s",
            self.cycle_count,
            total,
            "  ".join(f"{k}={v:.2f}s" for k, v in _t.items() if v > 0.001),
        )

        # 8. Placement stability metric
        stability = jaccard(selected_locations, self._previous_selections)
        self._previous_selections = selected_locations
        self._drone_plans = drone_plans

        ensemble_summary = {
            "mean_burn_probability":      float(ensemble.burn_probability.mean()),
            "max_burn_probability":       float(ensemble.burn_probability.max()),
            "arrival_time_variance_mean": float(np.nanmean(ensemble.arrival_time_variance)),
            "n_obs_assimilated":          assim["n_obs_used"],
            "replan_flags":               assim["replan_flags"],
        }

        cycle_report = CycleReport(
            cycle_id=self.cycle_count,
            info_field=info_field,
            evaluations={},          # filled by SimulationRunner (Phase 4b)
            ensemble_summary=ensemble_summary,
            placement_stability=stability,
            gp_prior=gp_prior,
            selection_result=primary_result,
            start_time=start_time,
        )

        logger.info(
            "Cycle %d | strategy=%s | selected=%d | stability=%.2f | obs=%d",
            self.cycle_count, self.selector_name,
            len(selected_locations),
            stability, assim["n_obs_used"],
        )
        return mission_queue, cycle_report

    # ------------------------------------------------------------------
    # Drone state lifecycle
    # ------------------------------------------------------------------

    def _advance_drone_states(
        self,
        updated_states: list[DroneFlightState],
    ) -> list[DroneFlightState]:
        """
        Process updated drone states after a cycle:
        - Drones that landed (returned=True) are reset to full battery + NORMAL.
        - Others carry forward their state unchanged.
        """
        staging_m = self._ground_stations_m[0]
        result = []
        for state in updated_states:
            if state.returned:
                # Drone has landed — reset for next sortie
                landed_gs_m = (
                    self._ground_stations_m[state.target_gs_idx]
                    if 0 <= state.target_gs_idx < len(self._ground_stations_m)
                    else staging_m
                )
                logger.info(
                    "Drone %d: landed at GS%d — sortie %.0f m | resetting",
                    state.drone_id, state.target_gs_idx, state.sortie_distance_m,
                )
                result.append(DroneFlightState(
                    drone_id=state.drone_id,
                    position_m=landed_gs_m.copy(),
                    remaining_range_m=DRONE_RANGE_M,
                    mode=DroneMode.NORMAL,
                    target_gs_idx=-1,
                    sortie_distance_m=0.0,
                    returned=False,
                ))
            else:
                result.append(state)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _neutral_ensemble(shape: tuple[int, int], n_members: int) -> EnsembleResult:
    """
    Zero-spread placeholder used on cycle 1 when no prior ensemble exists.
    All members agree → zero variance → EnKF Kalman gain ≈ 0 → no spurious update.
    """
    rows, cols = shape
    zero = np.zeros((rows, cols), dtype=np.float32)
    return EnsembleResult(
        member_arrival_times=np.full((n_members, rows, cols), np.nan, dtype=np.float32),
        member_fmc_fields=np.full((n_members, rows, cols), GP_DEFAULT_FMC_MEAN,        dtype=np.float32),
        member_wind_fields=np.full((n_members, rows, cols), GP_DEFAULT_WIND_SPEED_MEAN, dtype=np.float32),
        member_wind_dir_fields=np.full((n_members, rows, cols), GP_DEFAULT_WIND_DIR_MEAN, dtype=np.float32),
        burn_probability=zero,
        mean_arrival_time=zero,
        arrival_time_variance=zero,
        n_members=n_members,
    )
