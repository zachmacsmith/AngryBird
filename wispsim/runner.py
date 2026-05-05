"""
Simulation runners for WISP.

CycleRunner (Phase 4b — strategy comparison)
    Cycle-based harness for multi-strategy PERR comparison.  All strategies
    run on the same ensemble each cycle; only the primary strategy's
    observations update the real GP/EnKF state.  No clock, no drone movement.

SimulationRunner (Phase 4b — clock-based demo)
    Clock-based harness implementing the full simulation loop from the
    architecture spec.  Drones move continuously, collect observations, and
    WISP cycles trigger at fixed intervals.  Produces an MP4 video via
    FrameRenderer.

LiveEstimator
    Lightweight between-cycle estimator.  Snapshots the GP state at each
    WISP cycle boundary, then accumulates raw observations as they arrive
    and computes an updated (FMC mean, wind mean, estimated arrival time)
    at each render frame — using only a GP predict() call + one fire member.
    This avoids the 30-member ensemble and runs in seconds, allowing the
    operator display to reflect new observations immediately rather than
    waiting 20 minutes for the next WISP cycle.

SimulationConfig
    Shared configuration dataclass for SimulationRunner (and scenarios.py).
"""

from __future__ import annotations

import copy
import logging
from concurrent.futures import Future
from dataclasses import dataclass, field, replace as dc_replace
from typing import Optional

import numpy as np

from angrybird.config import (
    ENSEMBLE_SIZE,
    GP_DEFAULT_WIND_DIR_MEAN,
    GP_DEFAULT_WIND_SPEED_MEAN,
    GRID_RESOLUTION_M,
    IGNIS_SELECTOR,
    N_DRONES,
    NELSON_DEFAULT_RH,
    NELSON_DEFAULT_T_C,
    OBSERVATION_THINNING_SPACING_M,
    SIMULATION_HORIZON_MIN,
)
from angrybird.information import compute_information_field
from angrybird.nelson import nelson_fmc_field
from angrybird.orchestrator import IGNISOrchestrator, FireEngineProtocol
from angrybird.path_planner import plan_paths
from angrybird.selectors import registry as _default_registry
from angrybird.selectors.base import SelectorRegistry
from angrybird.types import (
    CycleReport,
    DronePlan,
    DroneObservation,
    EnsembleResult,
    GPPrior,
    InformationField,
    SelectionResult,
    StrategyEvaluation,
)
from .drone_sim import (
    DroneState,
    NoiseConfig,
    assign_waypoints,
    cell_to_pos_m,
    collect_fire_observation,
    collect_observations,
    move_drone,
)
from .evaluator import CounterfactualEvaluator, compute_arrival_accuracy
from .ground_truth import GroundTruth, compute_wind_field
from angrybird.hardware import HardwareBackend
from angrybird.prior import SimulatedEnvironmentalSource
from .observation_buffer import ObservationBuffer, thin_observations
from angrybird.assimilation import aggregate_drone_observations
from angrybird.config import (
    CAMERA_FOOTPRINT_M,
    CYCLE_INTERVAL_S,
    DRONE_ENDURANCE_S,
    DRONE_SPEED_MS,
    GP_CORRELATION_LENGTH_FMC_M,
    GP_DEFAULT_FMC_VARIANCE,
    GP_DEFAULT_WIND_DIR_MEAN,
    GP_DEFAULT_WIND_SPEED_MEAN,
    GP_DEFAULT_WIND_SPEED_VARIANCE,
    N_DRONES,
    NELSON_DEFAULT_RH,
    NELSON_DEFAULT_T_C,
    OBSERVATION_THINNING_SPACING_M,
    RAWS_FMC_SIGMA,
    RAWS_WIND_DIR_SIGMA,
    RAWS_WIND_SPEED_SIGMA,
    SIM_TOTAL_TIME_S,
)
from .observer import ObservationSource, SimulatedObserver
from .renderer import FrameRenderer
from angrybird.observations import (
    DroneObservation as DroneObs,
    ObservationStore,
    RAWSObservation,
    VariableType,
)
from angrybird.raws import RAWSObserver, RAWSStation, place_raws_stations
from .network import (
    MeshNetworkConfig,
    PingMeshNetwork,
    assign_packet_priority,
    make_pams_like_mesh_config,
    make_improved_mesh_config,
)

import csv
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimulationConfig
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Configuration for the clock-based SimulationRunner."""
    dt: float = 10.0                                   # simulation timestep (seconds)
    total_time_s: float = SIM_TOTAL_TIME_S             # total simulation duration
    ignis_cycle_interval_s: float = CYCLE_INTERVAL_S   # WISP cycle interval
    n_drones: int = N_DRONES
    drone_speed_ms: float = DRONE_SPEED_MS             # m/s cruise speed
    drone_endurance_s: float = DRONE_ENDURANCE_S       # flight endurance per sortie
    camera_footprint_m: float = CAMERA_FOOTPRINT_M     # FMC observation footprint radius
    base_cell: tuple[int, int] = (195, 100) # (row, col) drone home location
    frame_interval: int = 6                 # render one frame every N sim steps
    fps: int = 10                           # video frames per second
    live_fire_update_interval_s: float = 300.0  # how often to rerun the 1-member live fire estimate (s)
    live_fire_horizon_h: float = 1.0            # horizon for the between-cycle 1-member fire estimate (h)
    output_path: str = "out/simulation"
    scenario_name: str = "simulation"
    # RAWS stations — 1 randomly placed by default; pass raws_locations to fix positions
    n_raws: int = 1
    raws_locations: list = field(default_factory=list)
    # Mesh network simulation.
    # These defaults represent the improved product mode, not the degraded PAMS baseline.
    enable_mesh_network: bool = True

    # Realistic small-UAS wildfire mesh assumption:
    # enough range for multi-hop links, but not so high that every drone is always connected.
    mesh_range_m: float = 1800.0

    # A link should not disappear instantly after one missed ping.
    # Wildfire comms can have intermittent losses, so keep links alive briefly.
    mesh_max_link_age_s: float = 45.0

    # Allow weak-but-usable links, but reject very poor links.
    mesh_min_link_quality: float = 0.12

    # Wisp uses dt = 10s, so pinging every timestep is fine.
    mesh_ping_interval_s: float = 10.0

    # More than 3 prevents old packets from clogging the queue,
    # but not so high that the network becomes unrealistically perfect.
    mesh_packets_per_tick: int = 8

    # Small background loss, because path loss is already modeled by link quality.
    mesh_packet_loss_probability: float = 0.015
    selector_name: str = IGNIS_SELECTOR
    # Per-drone spawn times (seconds).  drone_spawn_times_s[i] is the simulation
    # time at which drone i activates and leaves the ground station.  Drones with
    # no entry (or entry=0) activate immediately at t=0.  Shorter list than
    # n_drones → remaining drones default to 0.
    drone_spawn_times_s: list = field(default_factory=list)

# ---------------------------------------------------------------------------
# LiveEstimator
# ---------------------------------------------------------------------------

class LiveEstimator:
    """
    Provides a continuously-updated "best estimate" of FMC, wind, and fire
    arrival times between WISP planning cycles.

    Why this exists
    ---------------
    WISP cycles run every ~20 minutes and require a full 30-member ensemble.
    But the operator display only needs:
      • GP posterior mean for FMC and wind   ←  cheap: GP predict() ~ ms
      • Estimated arrival time per cell      ←  cheap: 1 fire member ~ seconds

    LiveEstimator snapshots the orchestrator's GP at each cycle boundary, then
    incrementally adds raw observations as drones collect them.  At each render
    frame it calls gp.predict() (auto-re-fits if dirty) and runs one fire
    member with the mean-field conditions, yielding arrival times in hours.

    The orchestrator's own GP is not touched between cycles; LiveEstimator
    operates on its own deep-copied GP instance so cycle logic is unaffected.

    Usage (inside SimulationRunner)
    --------------------------------
        # After each WISP cycle completes:
        self._live_est.snapshot_from_cycle()

        # After each drone observation step:
        self._live_est.add_observations(new_obs)

        # At each render frame:
        prior, arrival_h = self._live_est.compute_estimate(
            shape, fire_state, rng
        )
    """

    def __init__(
        self,
        orchestrator: IGNISOrchestrator,
        terrain,
        horizon_h: float = 1.0,
        fire_update_interval_s: float = 300.0,
        backend=None,  # HardwareBackend | None — injected by SimulationRunner
    ) -> None:
        self._orchestrator  = orchestrator
        self._terrain       = terrain
        self.horizon_h      = horizon_h
        # How often (sim-seconds) to rerun the 1-member fire estimate.
        # Default 300 s (5 min) — configurable via SimulationConfig.live_fire_update_interval_s.
        self._fire_update_interval_s = fire_update_interval_s
        self._last_fire_est_time: float = -fire_update_interval_s  # trigger immediately on first call
        # Hardware backend — drives async vs blocking dispatch.
        # None → always blocking (safe default for CycleRunner / tests).
        self._backend = backend
        # In-flight future for async CUDA dispatch; None when idle.
        self._pending_future: Optional[Future] = None
        # Snapshot of the orchestrator GP at the last cycle boundary.
        # Never modified between cycles.
        self._live_gp = copy.deepcopy(orchestrator.gp)
        self._obs_buffer = ObservationBuffer(
            min_spacing_m=OBSERVATION_THINNING_SPACING_M,
            resolution_m=terrain.resolution_m,
        )
        self._has_obs = False
        # GP prior: updated at cycle snapshots; inter-cycle just uses snapshot predict.
        # The expensive fork+fit-on-drone-obs was costing ~2-3s per render frame
        # (sklearn re-fits on 300+ training points each call).  Displaying the
        # cycle-snapshot posterior is accurate enough between 5-min fire updates.
        self._cached_prior: Optional[GPPrior] = None
        # Cached 1-member fire arrival time map — reused until update interval elapses.
        self._cached_arrival_times_h: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def snapshot_from_cycle(self) -> None:
        """
        Sync live GP to orchestrator GP state immediately after an WISP cycle.

        Called once per cycle boundary.  Deep-copy is ~ms (sklearn GP objects
        are small once fitted) and happens only every 20 sim-minutes.
        The obs buffer is cleared so the next inter-cycle estimate starts fresh
        from the updated posterior rather than re-adding old observations.

        Any in-flight async fire estimate is cancelled: its GP snapshot is now
        stale.  On CUDA, Future.cancel() is a best-effort signal — if the
        worker has already started running, the result is simply discarded
        when the next collect attempt finds a fresh estimate has been queued.

        NOTE: _cached_prior and _cached_arrival_times_h are intentionally
        preserved here.  The renderer will show the previous cycle's estimate
        for the one frame that coincides with the cycle boundary, rather than
        flashing blank while the new async fire run completes.  The timer
        reset below forces a fresh run on the very next compute_estimate call
        so the display is updated within one render frame.
        """
        # Discard any in-flight run — it was seeded from the old GP snapshot.
        if self._pending_future is not None and not self._pending_future.done():
            self._pending_future.cancel()
        self._pending_future = None

        self._live_gp     = copy.deepcopy(self._orchestrator.gp)
        self._obs_buffer  = ObservationBuffer(
            min_spacing_m=OBSERVATION_THINNING_SPACING_M,
            resolution_m=self._terrain.resolution_m,
        )
        self._has_obs = False
        # Invalidate the cached GP prior so the new posterior is used on the
        # next predict() call, but keep _cached_arrival_times_h alive so the
        # renderer has something to show on the cycle-boundary frame.
        self._cached_prior = None
        # Force a fire re-run on the next compute_estimate call after a cycle,
        # but do NOT null _cached_arrival_times_h — the old estimate is still
        # better than a blank frame for one render step.
        self._last_fire_est_time = -self._fire_update_interval_s

    def add_observations(self, obs: list[DroneObservation]) -> None:
        """
        Buffer incoming drone observations.

        The ObservationBuffer handles duplicate suppression — passing raw
        footprint obs (which can arrive at 50–100 m spacing) is fine; the
        buffer uses the same 200 m coarse-binning thinning as the main
        observation pipeline.
        """
        if obs:
            self._obs_buffer.add(obs)
            self._has_obs = True

    def compute_estimate(
        self,
        shape: tuple[int, int],
        fire_state: np.ndarray,
        rng: np.random.Generator,
        current_time: float = 0.0,
    ) -> tuple[GPPrior, Optional[np.ndarray]]:
        """
        Compute the live best estimate.

        GP prior
        --------
        Returns the cycle-snapshot posterior (cheap: GP predict ~1 ms).
        The expensive fork+fit-on-drone-obs path was removed: sklearn re-fits
        on 300+ training points costs 2-3 s per call, and displaying the
        snapshot posterior between cycles is visually accurate enough.

        Fire arrival times
        ------------------
        Re-runs the 1-member fire simulation at most once per
        `fire_update_interval_s` (configurable, default 300 s / 5 min).
        Between updates the cached result is returned unchanged.  A WISP cycle
        snapshot resets the timer so the first post-cycle frame always gets a
        fresh fire estimate.

        Returns
        -------
        prior : GPPrior
        arrival_times_h : float32[rows, cols] or None
        """
        self._has_obs = False   # drain flag; observations already in _obs_buffer

        prior = self._cached_prior if self._cached_prior is not None \
            else self._live_gp.predict(shape)

        fire_engine = getattr(self._orchestrator, "fire_engine", None)

        # ── Collect: pick up a finished async run ─────────────────────────
        # On CUDA, the background thread may have finished since the last
        # render frame.  On MPS/CPU the future is always already resolved
        # (blocking inline path), so this check is effectively free.
        if self._pending_future is not None and self._pending_future.done():
            try:
                result = self._pending_future.result()
                self._cached_arrival_times_h = (
                    (result.member_arrival_times[0].astype(np.float32) / 60.0) + (current_time / 3600.0)
                )
                self._last_fire_est_time = current_time
            except Exception as exc:
                logger.warning("LiveEstimator async fire run failed: %s", exc)
            finally:
                self._pending_future = None

        # ── Submit: launch a new run when due and nothing is in flight ────
        # On CUDA this returns immediately; the background thread does the
        # work while the main loop continues to the next render frame.
        # On MPS/CPU HardwareBackend.submit_fire_estimate() runs inline so
        # the pending_future is already resolved by the collect step above on
        # the same call.
        due = (current_time - self._last_fire_est_time) >= self._fire_update_interval_s
        if (due or self._cached_arrival_times_h is None) \
                and self._pending_future is None \
                and fire_engine is not None:

            # Capture a snapshot of mutable arguments so the background
            # thread never races with the main loop.
            _terrain     = self._terrain
            _prior       = prior
            _fire_state  = fire_state.copy() if hasattr(fire_state, "copy") else fire_state
            _rng         = rng
            _horizon_min = int(self.horizon_h * 60)

            def _run_fire_est():
                return fire_engine.run(
                    _terrain, _prior, _fire_state,
                    n_members=1,
                    horizon_min=_horizon_min,
                    rng=_rng,
                )

            if self._backend is not None:
                self._pending_future = self._backend.submit_fire_estimate(
                    _run_fire_est
                )
                # If we have absolutely no cached estimate (e.g. very first frame),
                # block and wait for it so we don't render a blank frame.
                if self._cached_arrival_times_h is None:
                    try:
                        result = self._pending_future.result()
                        self._cached_arrival_times_h = (
                            (result.member_arrival_times[0].astype(np.float32) / 60.0) + (current_time / 3600.0)
                        )
                        self._last_fire_est_time = current_time
                    except Exception as exc:
                        logger.warning("LiveEstimator first async fire run failed: %s", exc)
                    finally:
                        self._pending_future = None

            else:
                # No backend supplied (e.g. CycleRunner / unit tests) — run
                # blocking and store result directly.
                try:
                    result = _run_fire_est()
                    self._cached_arrival_times_h = (
                        (result.member_arrival_times[0].astype(np.float32) / 60.0) + (current_time / 3600.0)
                    )
                    self._last_fire_est_time = current_time
                except Exception as exc:
                    logger.warning("LiveEstimator fire run failed: %s", exc)

        return prior, self._cached_arrival_times_h


# ---------------------------------------------------------------------------
# CycleRunner  (renamed from SimulationRunner — cycle-based, no clock)
# ---------------------------------------------------------------------------

class CycleRunner:
    """
    Cycle-based harness for multi-strategy comparison.

    All strategies run on the SAME ensemble each cycle (same seed → same data).
    Only the PRIMARY strategy's observations update the real GP/EnKF state.
    Other strategies are evaluated counterfactually.

    Args:
        orchestrator:       IGNISOrchestrator — holds GP + terrain + primary selector
        ground_truth:       GroundTruth — hidden true state (FMC, wind fields)
        fire_engine:        implements FireEngineProtocol
        strategies:         names of strategies to compare (must be in registry)
        primary_strategy:   which strategy's observations update real state
        selector_registry:  registry used to run all strategies
        n_drones:           number of drones per cycle
        horizon_min:        fire-spread simulation horizon in minutes
        n_members:          ensemble size
        resolution_m:       grid cell size in metres
        obs_source:         ObservationSource — defaults to SimulatedObserver(ground_truth)
    """

    def __init__(
        self,
        orchestrator: IGNISOrchestrator,
        ground_truth: GroundTruth,
        fire_engine: FireEngineProtocol,
        strategies: Optional[list[str]] = None,
        primary_strategy: Optional[str] = None,
        selector_registry: SelectorRegistry = _default_registry,
        n_drones: int = N_DRONES,
        horizon_min: int = SIMULATION_HORIZON_MIN,
        n_members: int = ENSEMBLE_SIZE,
        resolution_m: float = GRID_RESOLUTION_M,
        obs_source: Optional[ObservationSource] = None,
        bimodal_alpha: float = 0.5,
        bimodal_beta: float = 0.0,   # disabled — see orchestrator.py comment
        nelson_T_C: float = NELSON_DEFAULT_T_C,
        nelson_RH: float = NELSON_DEFAULT_RH,
        nelson_hour: float = 14.0,
    ) -> None:
        self.orchestrator     = orchestrator
        self.ground_truth     = ground_truth
        self.fire_engine      = fire_engine
        self.strategies       = strategies or list(orchestrator.registry.names())
        self.primary_strategy = primary_strategy or orchestrator.selector_name
        self.registry         = selector_registry
        self.n_drones         = n_drones
        self.horizon_min      = horizon_min
        self.n_members        = n_members
        self.resolution_m     = resolution_m
        self.obs_source       = obs_source or SimulatedObserver(ground_truth)
        self.evaluator        = CounterfactualEvaluator(self.obs_source, orchestrator.gp)
        self.bimodal_alpha    = bimodal_alpha
        self.bimodal_beta     = bimodal_beta
        self.nelson_T_C       = nelson_T_C
        self.nelson_RH        = nelson_RH
        self.nelson_hour      = nelson_hour

        self._pending_observations: list[DroneObservation] = []

        self.first_ensemble = None
        self.first_gp_prior = None
        self.last_ensemble  = None
        self.last_gp_prior  = None

    # ------------------------------------------------------------------
    # Single cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        fire_state: np.ndarray,
        cycle_seed: int = 0,
    ) -> CycleReport:
        """
        Run one full comparison cycle.

        1. Build shared ensemble (same seed → all strategies see identical data)
        2. Compute information field
        3. Run all strategies
        4. Plan paths for each strategy
        5. Evaluate each strategy counterfactually
        6. Collect primary strategy's observations
        7. Call orchestrator.run_cycle with pre-built ensemble + pending observations
        8. Assemble full CycleReport
        """
        shape = self.orchestrator.terrain.shape
        rng   = np.random.default_rng(cycle_seed)

        # Push Nelson FMC prior mean before GP prediction so the GP fits
        # residuals from the physics model rather than raw values.
        if self.orchestrator.terrain.origin_latlon is not None:
            lat = float(self.orchestrator.terrain.origin_latlon[0])
        else:
            lat = 37.5
        nelson_field = nelson_fmc_field(
            self.orchestrator.terrain,
            T_C=self.nelson_T_C,
            RH=self.nelson_RH,
            hour_of_day=self.nelson_hour,
            latitude_deg=lat,
        )
        self.orchestrator.gp.set_nelson_mean(nelson_field)

        gp_prior = self.orchestrator.gp.predict(shape)

        ensemble = self.fire_engine.run(
            self.orchestrator.terrain, gp_prior, fire_state,
            self.n_members, self.horizon_min, rng,
        )

        if self.first_ensemble is None:
            self.first_ensemble = ensemble
            self.first_gp_prior = gp_prior
        self.last_ensemble = ensemble
        self.last_gp_prior = gp_prior

        info_field = compute_information_field(
            ensemble, gp_prior,
            resolution_m=self.resolution_m,
            horizon_minutes=self.horizon_min,
            bimodal_alpha=self.bimodal_alpha,
            bimodal_beta=self.bimodal_beta,
        )

        selection_results: dict[str, SelectionResult] = {}
        for name in self.strategies:
            try:
                selection_results[name] = self.registry.run(
                    name, info_field, self.orchestrator.gp, ensemble, k=self.n_drones,
                )
            except Exception as exc:
                logger.warning("Strategy '%s' raised an exception: %s", name, exc)

        drone_plans_by_strategy = {
            name: plan_paths(
                sel.selected_locations,
                staging_area=self.orchestrator.staging_area,
                n_drones=self.n_drones,
                shape=shape,
                resolution_m=self.resolution_m,
            )
            for name, sel in selection_results.items()
        }

        evaluations = {
            name: self.evaluator.evaluate(
                strategy_name=name,
                selection_result=sel,
                drone_plans=drone_plans_by_strategy[name],
                info_field=info_field,
                gp_prior=gp_prior,
                n_drones=self.n_drones,
            )
            for name, sel in selection_results.items()
        }

        # Enrich the primary strategy's evaluation with ground-truth accuracy metrics.
        # The ensemble was built from the GP posterior that already incorporates all
        # observations assimilated in previous cycles (pending obs from the prior cycle
        # were fed to orchestrator.run_cycle() which updates the GP state).
        if self.primary_strategy in evaluations:
            horizon_s = self.horizon_min * 60.0
            crps_min, rmse_min, spread_skill = compute_arrival_accuracy(
                ensemble,
                self.ground_truth.fire.arrival_times,
                horizon_s=horizon_s,
            )
            if crps_min > 0.0 or rmse_min > 0.0:
                evaluations[self.primary_strategy] = dc_replace(
                    evaluations[self.primary_strategy],
                    crps_minutes=crps_min,
                    rmse_minutes=rmse_min,
                    spread_skill_ratio=spread_skill,
                )

        if self.primary_strategy in drone_plans_by_strategy:
            primary_cells: set[tuple[int, int]] = set()
            for plan in drone_plans_by_strategy[self.primary_strategy]:
                primary_cells.update(plan.cells_observed)
            new_observations = self.obs_source.observe(list(primary_cells))
        else:
            logger.warning(
                "Primary strategy '%s' not in results; no observations collected.",
                self.primary_strategy,
            )
            new_observations = []

        _, cycle_report = self.orchestrator.run_cycle(
            fire_state=fire_state,
            observations=self._pending_observations,
            ensemble=ensemble,
        )

        self._pending_observations = new_observations

        full_report = CycleReport(
            cycle_id=cycle_report.cycle_id,
            info_field=info_field,
            evaluations=evaluations,
            ensemble_summary=cycle_report.ensemble_summary,
            placement_stability=cycle_report.placement_stability,
            gp_prior=gp_prior,
            selection_result=selection_results.get(self.primary_strategy),
        )

        logger.info(
            "Cycle %d | PERR: %s",
            full_report.cycle_id,
            {n: f"{e.perr:.4f}" for n, e in evaluations.items()},
        )
        return full_report

    # ------------------------------------------------------------------
    # Multi-cycle comparison
    # ------------------------------------------------------------------

    def run_comparison(
        self,
        fire_states: list[np.ndarray],
        base_seed: int = 0,
    ) -> list[CycleReport]:
        """Run a multi-cycle comparison across a sequence of fire states."""
        reports: list[CycleReport] = []
        for cycle_idx, fire_state in enumerate(fire_states):
            report = self.run_cycle(
                fire_state=fire_state,
                cycle_seed=base_seed + cycle_idx,
            )
            reports.append(report)

        if reports:
            for name in self.strategies:
                perrs = [
                    r.evaluations[name].perr
                    for r in reports
                    if name in r.evaluations
                ]
                if perrs:
                    logger.info(
                        "Strategy '%s' — mean PERR=%.4f over %d cycles",
                        name, float(np.mean(perrs)), len(perrs),
                    )

        return reports


# ---------------------------------------------------------------------------
# SimulationRunner  (clock-based, full simulation loop)
# ---------------------------------------------------------------------------

class SimulationRunner:
    """
    Clock-based simulation harness implementing the full architecture spec loop.

    At each timestep (default dt=10s):
      1. Wind field updated via compute_wind_field()
      2. Ground truth fire stepped forward
      3. All drones moved and observations collected
      4. New observations fed to LiveEstimator (cheap: GP dirty + 1 fire member)
      5. Observations buffered; flushed + thinned at each WISP cycle boundary
      6. WISP cycle runs (full ensemble → info field → selection → assimilation)
      7. LiveEstimator syncs to updated GP state after each cycle
      8. Frame rendered (every frame_interval steps) with live estimate

    The live estimate panels (FMC+wind, arrival time) update every render
    frame — reflecting new observations immediately rather than waiting for
    the next 20-minute WISP cycle.  The information field and mission
    targets still update only at WISP cycle boundaries (require full ensemble).

    Args:
        config:        SimulationConfig
        terrain:       TerrainData (static)
        ground_truth:  GroundTruth (mutable — wind + fire updated in-place)
        orchestrator:  IGNISOrchestrator (runs WISP cycles)
    """

    def __init__(
        self,
        config: SimulationConfig,
        terrain,
        ground_truth: GroundTruth,
        orchestrator: IGNISOrchestrator,
    ) -> None:
        self.config       = config
        self.terrain      = terrain
        self.truth        = ground_truth
        self.orchestrator = orchestrator

        horizon_h = getattr(orchestrator, "horizon_min", SIMULATION_HORIZON_MIN) / 60.0

        self.obs_buffer = ObservationBuffer(
            min_spacing_m=OBSERVATION_THINNING_SPACING_M,
            resolution_m=terrain.resolution_m,
        )
        self.renderer = FrameRenderer(
            terrain=terrain,
            out_dir=config.output_path,
            frame_interval=config.frame_interval,
            fps=config.fps,
            horizon_h=horizon_h,
            raws_locations=None,   # filled after RAWS placement below
        )
        self.noise = NoiseConfig(camera_footprint_m=config.camera_footprint_m)

        # Tracks info_field.w.sum() from the previous cycle (mission-value metric).
        # Can increase as fire spreads into new high-sensitivity cells, so is NOT
        # a pure information-gain metric — use _prev_gp_var_sum for that.
        self._prev_info_entropy: Optional[float] = None

        # Pure GP variance baseline: sum of fmc_variance + wind_speed_variance over the grid.
        # Initialized from the no-observations default so cycle 1 captures the RAWS
        # seeding contribution. Monotonically decreasing (more obs → lower variance).
        _n = terrain.shape[0] * terrain.shape[1]
        self._prev_gp_var_sum: float = (GP_DEFAULT_FMC_VARIANCE + GP_DEFAULT_WIND_SPEED_VARIANCE) * _n

        base_pos = cell_to_pos_m(config.base_cell, terrain.resolution_m)
        _spawn_times = list(config.drone_spawn_times_s) + [0.0] * config.n_drones
        self.drones = [
            DroneState(
                drone_id=f"drone_{i}",
                position=base_pos.copy(),
                speed=config.drone_speed_ms,
                status="pending" if _spawn_times[i] > 0.0 else "idle",
                waypoint_queue=[],
                current_target=None,
                path_history=[base_pos.copy()],
                endurance_remaining_s=config.drone_endurance_s,
                base_position=base_pos.copy(),
                spawn_time_s=_spawn_times[i],
            )
            for i in range(config.n_drones)
        ]

        drone_ids = []

        for i in range(len(self.drones)):
            drone_ids.append(self.drones[i].drone_id)

        mesh_config = make_improved_mesh_config()

        self.network = PingMeshNetwork(
            ground_station_position=base_pos.copy(),
            drone_ids=drone_ids,
            config=mesh_config,
            rng=np.random.default_rng(777),
        )

        # Build RAWS stations and seed the GP from their initial observations.
        # The provider is a SimulatedObserver with lower noise than drone sensors
        # (fixed ground station quality).  In production, swap the provider for a
        # real telemetry client — the GP.add_raws() call below is identical.
        if config.raws_locations:
            raws_stations = [RAWSStation(loc) for loc in config.raws_locations]
        else:
            raws_stations = place_raws_stations(
                shape=terrain.shape,
                n=config.n_raws,
                ignition_cells=ground_truth.ignition_cells,
                base_cell=config.base_cell,
                rng=np.random.default_rng(config.n_raws * 37),
            )
        self.raws_observer = RAWSObserver(
            raws_stations,
            SimulatedObserver(
                ground_truth,
                fmc_sigma=RAWS_FMC_SIGMA,
                wind_speed_sigma=RAWS_WIND_SPEED_SIGMA,
                wind_dir_sigma=RAWS_WIND_DIR_SIGMA,
                rng=np.random.default_rng(55),
            ),
        )
        self.renderer._raws_locations = self.raws_observer.locations

        raws_obs = self.raws_observer.observe_all()
        for station, obs in zip(self.raws_observer.stations, raws_obs):
            orchestrator.obs_store.add_raws(RAWSObservation(
                _source_id           = station.station_id,
                _timestamp           = 0.0,
                location             = obs.location,
                fmc                  = obs.fmc,
                fmc_sigma            = obs.fmc_sigma,
                wind_speed           = obs.wind_speed,
                wind_speed_sigma     = obs.wind_speed_sigma,
                wind_direction       = obs.wind_dir,
                wind_direction_sigma = obs.wind_dir_sigma,
            ))

        # Snapshot RAWS-only GP variance (before any drone observations).
        # Deep-copy so the main GP's fitted state is not disturbed; the predict()
        # call triggers a fit on the copy with only RAWS obs.  This is the baseline
        # the renderer uses to draw the RAWS-only horizontal reference line.
        _raws_only_gp = copy.deepcopy(orchestrator.gp)
        _raws_prior = _raws_only_gp.predict(terrain.shape)
        self.renderer._raws_only_gp_var_sum = float(
            _raws_prior.fmc_variance.sum() + _raws_prior.wind_speed_variance.sum()
        )
        self.renderer._initial_gp_var_sum = self._prev_gp_var_sum

        # Wire the simulated sensor source as the orchestrator's environmental data source.
        # Produces frequency-gated, noisy NWP weather/wind and satellite FMC prior fields.
        # Also provides collect_obs_store_inputs() for fire-detection + FMC GP observations.
        orchestrator.data_source = SimulatedEnvironmentalSource(
            ground_truth, rng=np.random.default_rng(99)
        )

        # Fire report seeding is the caller's responsibility — add
        # FireDetectionObservation(s) to orchestrator.obs_store before
        # constructing SimulationRunner.  The runner script (run.py) does
        # this with configurable confidence and radius from the CLI fire-report
        # args.  Warn loudly if nothing is in the store so the missing seed
        # is caught early rather than silently deferring fire state init.
        if not orchestrator.obs_store.get_fire_detections():
            logger.warning(
                "SimulationRunner: obs_store contains no fire detections. "
                "Add at least one FireDetectionObservation to obs_store before "
                "constructing SimulationRunner, or fire state init will be deferred "
                "until the first satellite pass (~5 min)."
            )

        # Set a constant uninformed wind prior (5 m/s / 270° westerly) to:
        #   1. Activate circular arithmetic in fit() for wind direction.
        #   2. Ensure non-zero GP residuals (RAWS obs - prior ≠ 0) so the kernel
        #      amplitude optimizer can fit a sensible C rather than collapsing to
        #      the lower bound.
        #   3. Give cells far from any training data a neutral fallback (prior).
        # A prior equal to the RAWS observation gives residual = 0 → C → 1e-3
        # → GP variance collapses and wind observations get no credit in the
        # information field.  In production this prior would come from a NWP
        # model (HRRR/GFS); here we use a fixed uninformed background.
        orchestrator.gp.set_wind_prior_mean(
            np.full(terrain.shape, GP_DEFAULT_WIND_SPEED_MEAN, dtype=np.float32),
            np.full(terrain.shape, GP_DEFAULT_WIND_DIR_MEAN,   dtype=np.float32),
        )

        self.current_time: float = 0.0
        # Trigger first WISP cycle immediately at t=0
        self._last_cycle_time: float = -config.ignis_cycle_interval_s
        self.cycle_reports: list[CycleReport] = []

        # Cached WISP outputs for the renderer (update per cycle)
        self._gp_prior: Optional[GPPrior] = None
        self._burn_probability: Optional[np.ndarray] = None
        self._mission_targets: list[tuple[int, int]] = []
        self._drone_plans: list[DronePlan] = []
        # Last ensemble result — used to recompute info field live at render cadence.
        self._last_ensemble: Optional[EnsembleResult] = None
        self._live_info_field: Optional[InformationField] = None
        # Dirty flag: True after each WISP cycle updates the ensemble/prior.
        # compute_information_field is only recomputed when dirty — the result
        # is identical between cycles since neither ensemble nor prior changes.
        self._live_info_field_dirty: bool = False

        # Hardware backend — resolves device capabilities once and passes them
        # down to components that need async dispatch or explicit synchronisation.
        # Inferred from the fire engine's device; falls back to "cpu" if the
        # orchestrator has no fire engine attached yet.
        _fe = getattr(orchestrator, "fire_engine", None)
        _dev = str(getattr(_fe, "device", "cpu"))  # torch.device or str both work
        self._hw_backend = HardwareBackend(_dev)

        # Live estimator — updates between cycles with raw per-step observations.
        # Uses a shorter horizon than the full ensemble (default 1 h) so each
        # 1-member fire run is fast enough to run every few minutes of sim time.
        # On CUDA the backend enables async dispatch so the fire sim overlaps
        # with the main loop; on MPS/CPU it falls back to blocking.
        self._live_est = LiveEstimator(
            orchestrator=orchestrator,
            terrain=terrain,
            horizon_h=config.live_fire_horizon_h,
            fire_update_interval_s=config.live_fire_update_interval_s,
            backend=self._hw_backend,
        )
        self._live_gp_prior: Optional[GPPrior] = None
        self._live_arrival_times_h: Optional[np.ndarray] = None
        # Oracle arrival times: fire-engine projection using true FMC/wind, same
        # format as live_arrival_times_h so both panels are directly comparable.
        self._truth_arrival_times_h: Optional[np.ndarray] = None
        self._last_truth_est_time: float = -config.live_fire_update_interval_s
        # Per-cycle metrics CSV: rows accumulated in _run_ignis_cycle, written at end.
        self._cycle_metrics_rows: list[dict] = []
        self._cumulative_obs: int = 0   # drone obs delivered to GP (excluding RAWS)
        # Per-cycle accuracy trace for the renderer (WispSim-only; not stored in CycleReport).
        self._accuracy_trace: list[dict] = []
        self.network_log_rows = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> list[CycleReport]:
        """
        Execute the full simulation.

        Returns list[CycleReport] — one per WISP cycle.
        """
        n_steps = int(self.config.total_time_s / self.config.dt)
        rng     = np.random.default_rng(0)

        logger.info(
            "SimulationRunner starting: %s | %.0f h | dt=%.0f s | %d drones | %d steps",
            self.config.scenario_name,
            self.config.total_time_s / 3600.0,
            self.config.dt,
            self.config.n_drones,
            n_steps,
        )

        import time as _time
        _step_buckets: dict[str, float] = {
            "wind": 0.0, "fire_ca": 0.0, "drones": 0.0,
            "network": 0.0, "wisp_cycle": 0.0, "live_est": 0.0, "render": 0.0,
        }

        for step in range(n_steps):
            self.current_time = step * self.config.dt

            # 1. Update ground truth wind
            _t0 = _time.perf_counter()
            ws, wd = compute_wind_field(
                self.truth.base_wind_speed,
                self.truth.base_wind_direction,
                self.terrain,
                self.current_time,
                self.truth.wind_events,
                rng=rng,
            )
            self.truth.wind_speed     = ws
            self.truth.wind_direction = wd
            _step_buckets["wind"] += _time.perf_counter() - _t0

            # 2. Advance ground truth fire
            _t0 = _time.perf_counter()
            self.truth.fire.step(
                self.config.dt,
                self.truth.wind_speed,
                self.truth.wind_direction,
                self.truth.fmc,
            )
            _step_buckets["fire_ca"] += _time.perf_counter() - _t0

            # 3. Move drones + collect observations
            _t0 = _time.perf_counter()
            for drone in self.drones:
                # Activate pending drones whose spawn time has arrived.
                if drone.status == "pending" and self.current_time >= drone.spawn_time_s:
                    drone.status = "idle"
                    drone.position = drone.base_position.copy()
                    logger.info(
                        "Drone %s activated at t=%.0fs (spawn_time=%.0fs)",
                        drone.drone_id, self.current_time, drone.spawn_time_s,
                    )

                if drone.status == "pending":
                    continue  # not yet active — skip movement and observations

                move_drone(drone, self.config.dt)
                self._refuel_if_home(drone)

                if drone.status in ("transit", "returning"):
                    obs = collect_observations(
                        drone=drone,
                        fmc_field=self.truth.fmc,
                        wind_speed_field=self.truth.wind_speed,
                        wind_direction_field=self.truth.wind_direction,
                        terrain_shape=self.terrain.shape,
                        resolution_m=self.terrain.resolution_m,
                        noise=self.noise,
                        current_time=self.current_time,
                        fire_arrival_times=self.truth.fire.arrival_times,
                        rng=rng,
                    )

                    if self.config.enable_mesh_network:
                        self.network.buffer_observations(
                            drone_id=drone.drone_id,
                            current_time=self.current_time,
                            observations=obs,
                            priority=assign_packet_priority(obs),
                        )
                    else:
                        self.obs_buffer.add(obs)
                        self._live_est.add_observations(obs)

                    # Thermal camera: one fire detection per drone per timestep
                    # from the nadir cell.  Bypasses the obs_buffer (which is for
                    # FMC/wind thinning) and goes directly to obs_store so the
                    # particle filter can read it at the next WISP cycle.
                    fire_det = collect_fire_observation(
                        drone=drone,
                        fire_arrival_times=self.truth.fire.arrival_times,
                        terrain_shape=self.terrain.shape,
                        resolution_m=self.terrain.resolution_m,
                        current_time=self.current_time,
                        rng=rng,
                    )
                    if fire_det is not None:
                        self.orchestrator.obs_store.add(fire_det)
            _step_buckets["drones"] += _time.perf_counter() - _t0

            # 4. Network step
            _t0 = _time.perf_counter()
            if self.config.enable_mesh_network:
                drone_positions = {}

                for i in range(len(self.drones)):
                    drone = self.drones[i]
                    drone_positions[drone.drone_id] = drone.position

                received_obs_this_step = self.network.step(
                    drone_positions=drone_positions,
                    current_time=self.current_time,
                )
                metrics = self.network.get_metrics()
                buffer_sizes = self.network.get_buffer_sizes()
                paths = self.network.get_last_paths()
                connected = self.network.get_connected_drones()

                self.network_log_rows.append({
                    "time_s": self.current_time,
                    "packets_created": metrics["packets_created"],
                    "packets_delivered": metrics["packets_delivered"],
                    "packets_failed": metrics["packets_failed"],
                    "packet_delivery_rate": metrics["packet_delivery_rate"],
                    "observations_created": metrics["observations_created"],
                    "observations_delivered": metrics["observations_delivered"],
                    "observation_delivery_rate": metrics["observation_delivery_rate"],
                    "mean_delivery_delay_s": metrics["mean_delivery_delay_s"],
                    "connected_drones": len(connected),
                    "buffered_packets_total": sum(buffer_sizes.values()),
                    "paths": str(paths),
                })

                if received_obs_this_step:
                    self.obs_buffer.add(received_obs_this_step)
                    self._live_est.add_observations(received_obs_this_step)
            _step_buckets["network"] += _time.perf_counter() - _t0

            # 5. WISP cycle when due
            _t0 = _time.perf_counter()
            if (self.current_time - self._last_cycle_time
                    >= self.config.ignis_cycle_interval_s):
                self._run_ignis_cycle()
                self._last_cycle_time = self.current_time
            _step_buckets["wisp_cycle"] += _time.perf_counter() - _t0

            # 6. Update live estimate at render cadence
            #    Predict (refit if dirty) + 1 fire member — cheap enough to run
            #    every render frame (every frame_interval*dt seconds of sim time).
            _t0 = _time.perf_counter()
            if step % self.config.frame_interval == 0:
                fire_state = self.truth.fire.fire_state
                self._live_gp_prior, self._live_arrival_times_h = (
                    self._live_est.compute_estimate(
                        shape=self.terrain.shape,
                        fire_state=fire_state,
                        rng=np.random.default_rng(step),
                        current_time=self.current_time,
                    )
                )
                # Oracle arrival time: same fire-engine projection as the estimate
                # but seeded with the true FMC and wind fields instead of the GP
                # posterior.  Uses the same update interval as the live estimate.
                fire_engine = getattr(self.orchestrator, "fire_engine", None)
                if (fire_engine is not None and (
                    self._truth_arrival_times_h is None or
                    (self.current_time - self._last_truth_est_time)
                        >= self.config.live_fire_update_interval_s
                )):
                    try:
                        _zero = np.zeros(self.terrain.shape, dtype=np.float32)
                        _truth_prior = GPPrior(
                            fmc_mean=self.truth.fmc.astype(np.float32),
                            fmc_variance=_zero,
                            wind_speed_mean=self.truth.wind_speed.astype(np.float32),
                            wind_speed_variance=_zero,
                            wind_dir_mean=self.truth.wind_direction.astype(np.float32),
                            wind_dir_variance=_zero,
                        )
                        _truth_result = fire_engine.run(
                            self.terrain, _truth_prior,
                            fire_state.copy(),
                            n_members=1,
                            horizon_min=int(self._live_est.horizon_h * 60),
                            rng=np.random.default_rng(step + 1),
                        )
                        self._truth_arrival_times_h = (
                            _truth_result.member_arrival_times[0].astype(np.float32)
                            / 60.0
                        ) + (self.current_time / 3600.0)
                        self._last_truth_est_time = self.current_time
                    except Exception as _exc:
                        logger.warning("Oracle fire estimate failed: %s", _exc)
                # Recompute the info field only when the ensemble or GP prior has
                # changed (i.e., just after a WISP cycle).  Between cycles the
                # ensemble and prior are both constant, so the result is identical
                # every frame — no need to recompute 60× per hour.
                if self._last_ensemble is not None and self._live_info_field_dirty:
                    live_prior = (
                        self._live_gp_prior
                        if self._live_gp_prior is not None
                        else self._gp_prior
                    )
                    if live_prior is not None:
                        try:
                            self._live_info_field = compute_information_field(
                                self._last_ensemble,
                                live_prior,
                                resolution_m=self.terrain.resolution_m,
                                horizon_minutes=self.orchestrator.horizon_min,
                                bimodal_alpha=self.orchestrator.bimodal_alpha,
                                bimodal_beta=self.orchestrator.bimodal_beta,
                            )
                            self._live_info_field_dirty = False
                        except Exception:
                            pass  # fall back to last cycle's info field
            _step_buckets["live_est"] += _time.perf_counter() - _t0

            # 7. Render frame
            _t0 = _time.perf_counter()
            self.renderer.render_frame(
                step=step,
                time_s=self.current_time,
                ground_truth=self.truth,
                drones=self.drones,
                gp_prior=self._gp_prior,
                burn_probability=self._burn_probability,
                info_field=(self._live_info_field
                            if self._live_info_field is not None
                            else (self.cycle_reports[-1].info_field
                                  if self.cycle_reports else None)),
                mission_targets=self._mission_targets,
                drone_plans=self._drone_plans or None,
                cycle_reports=self.cycle_reports,
                live_gp_prior=self._live_gp_prior,
                live_arrival_times_h=self._live_arrival_times_h,
                truth_arrival_times_h=self._truth_arrival_times_h,
                accuracy_trace=self._accuracy_trace,
            )
            _step_buckets["render"] += _time.perf_counter() - _t0

        self.renderer.finalize()
        logger.info(
            "SimulationRunner finished: %d WISP cycles | %d frames",
            len(self.cycle_reports),
            self.renderer._frame_count,
        )

        # ── Step-level timing summary ─────────────────────────────────────
        total_wall = sum(_step_buckets.values())
        logger.info(
            "Step timing (total=%.1fs across %d steps):\n%s",
            total_wall,
            n_steps,
            "\n".join(
                f"  {k:12s}  {v:7.2f}s  ({100*v/max(total_wall,1e-9):5.1f}%)"
                for k, v in sorted(_step_buckets.items(), key=lambda x: -x[1])
            ),
        )

        if self.config.enable_mesh_network:
            logger.info("Mesh network metrics: %s", self.network.get_metrics())
            logger.info("Final drone buffer sizes: %s", self.network.get_buffer_sizes())

        # Write per-cycle metrics CSV (used by demo/compare_runs.py).
        if self._cycle_metrics_rows:
            metrics_csv = self.renderer.out_dir.parent / f"{self.config.scenario_name}_metrics.csv"
            with open(metrics_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._cycle_metrics_rows[0].keys())
                writer.writeheader()
                writer.writerows(self._cycle_metrics_rows)
            logger.info("Cycle metrics saved to %s", metrics_csv)

        if hasattr(self, "network_log_rows"):
            output_dir = Path("out")
            output_dir.mkdir(exist_ok=True)

            network_csv = output_dir / "network_metrics.csv"

            if len(self.network_log_rows) > 0:
                with open(network_csv, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=self.network_log_rows[0].keys(),
                    )
                    writer.writeheader()
                    writer.writerows(self.network_log_rows)

                logger.info("Network metrics saved to %s", network_csv)

        # Ensure the background fire-estimate executor is stopped cleanly.
        # On CUDA this waits for any in-flight run to finish before returning.
        # On MPS/CPU the executor is None so this is a no-op.
        self._hw_backend.shutdown(wait=True)

        return self.cycle_reports

    # ------------------------------------------------------------------
    # WISP cycle
    # ------------------------------------------------------------------

    def _run_ignis_cycle(self) -> None:
        """Flush observation buffer, run orchestrator, dispatch drones."""
        observations = self.obs_buffer.flush_thinned()
        self._cumulative_obs += len(observations)

        # Prune expired observations; temporal decay is query-time-parameterized
        # (passed to get_data_points at fit time) so no update_time() needed.
        self.orchestrator.obs_store.prune(self.current_time)

        # Collect satellite/fire-detection observations from the simulated source and
        # add them to the GP store before the cycle so the ensemble sees them.
        # Ground truth fire state is used only here (for satellite pixel sampling),
        # NOT passed to the orchestrator — fire state estimation is fully obs-driven.
        data_source = self.orchestrator.data_source
        if hasattr(data_source, "collect_obs_store_inputs"):
            sat_obs = data_source.collect_obs_store_inputs(
                self.current_time, self.truth.fire.fire_state
            )
            if sat_obs:
                self.orchestrator.obs_store.add_batch(sat_obs)

        # Sync physical drone positions into the orchestrator's DroneFlightState so
        # that the path planner starts each drone's route from its actual location,
        # not from the predicted end-of-last-cycle domain centroid.
        # Pending (not-yet-spawned) drones are pinned to the base position so the
        # path planner correctly treats them as starting from the ground station.
        if self.orchestrator._drone_states is not None:
            from angrybird.types import DroneFlightState as _DFS
            synced = []
            for phys, flight in zip(self.drones, self.orchestrator._drone_states):
                actual_pos = (
                    phys.base_position.copy()
                    if phys.status == "pending"
                    else phys.position.copy()
                )
                synced.append(_DFS(
                    drone_id      = flight.drone_id,
                    position_m    = actual_pos,
                    remaining_range_m = flight.remaining_range_m,
                    mode          = flight.mode,
                    target_gs_idx = flight.target_gs_idx,
                    sortie_distance_m = flight.sortie_distance_m,
                    returned      = flight.returned,
                ))
            self.orchestrator._drone_states = synced

        # No fire_state argument — orchestrator derives fire state from obs_store
        # (drone thermal detections + GOES/VIIRS satellite passes).
        mission_queue, cycle_report = self.orchestrator.run_cycle(
            observations=observations,
            start_time=self.current_time,
        )

        # Cache cycle-level outputs for the renderer (info field, targets)
        shape = self.terrain.shape
        self._gp_prior = self.orchestrator.gp.predict(shape)
        # Use last ensemble result for burn probability if available.
        cycle_ensemble = getattr(self.orchestrator, "_last_ensemble_result", None)
        if cycle_ensemble is not None:
            self._burn_probability = cycle_ensemble.burn_probability
            self._last_ensemble    = cycle_ensemble
        else:
            self._burn_probability = None

        targets = self.orchestrator._previous_selections
        self._mission_targets = list(targets)
        self._drone_plans = list(self.orchestrator._drone_plans)

        # Arrival time accuracy against oracle ground truth.
        # Always uses the ensemble from THIS cycle (post-assimilation GP),
        # including cycle 0 where the GP is seeded by RAWS only.
        # compute_arrival_accuracy returns (0, 0, 0) if no cells will burn in horizon.
        horizon_s = self.orchestrator.horizon_min * 60.0
        crps_min = rmse_min = spread_skill = 0.0
        if self._last_ensemble is not None:
            crps_min, rmse_min, spread_skill = compute_arrival_accuracy(
                self._last_ensemble,
                self.truth.fire.arrival_times,
                horizon_s=horizon_s,
                current_time_s=self.current_time,
            )

        # Per-burning-cell CRPS: divide by the number of cells that will burn
        # within the planning horizon from now.  Removes the confound where raw
        # CRPS grows simply because the fire is larger; ideal value is 0.
        _active_mask = (
            (self.truth.fire.arrival_times >= self.current_time) &
            ((self.truth.fire.arrival_times - self.current_time) < horizon_s) &
            np.isfinite(self.truth.fire.arrival_times)
        )
        n_active_cells = int(_active_mask.sum())
        crps_per_cell_min = crps_min / max(n_active_cells, 1)

        # Oracle CRPS: 1-member fire run seeded with true FMC + wind (zero GP
        # uncertainty).  Gives the irreducible model-error floor — the best CRPS
        # achievable even with perfect observations.
        oracle_crps_min = 0.0
        _fire_engine = getattr(self.orchestrator, "fire_engine", None)
        if _fire_engine is not None:
            try:
                _zero = np.zeros(self.terrain.shape, dtype=np.float32)
                _oracle_prior = GPPrior(
                    fmc_mean=self.truth.fmc.astype(np.float32),
                    fmc_variance=_zero,
                    wind_speed_mean=self.truth.wind_speed.astype(np.float32),
                    wind_speed_variance=_zero,
                    wind_dir_mean=self.truth.wind_direction.astype(np.float32),
                    wind_dir_variance=_zero,
                )
                _oracle_result = _fire_engine.run(
                    self.terrain, _oracle_prior,
                    self.truth.fire.fire_state.copy(),
                    n_members=1,
                    horizon_min=self.orchestrator.horizon_min,
                    rng=np.random.default_rng(0),
                )
                oracle_crps_min, _, _ = compute_arrival_accuracy(
                    _oracle_result,
                    self.truth.fire.arrival_times,
                    horizon_s=horizon_s,
                    current_time_s=self.current_time,
                )
            except Exception as _exc:
                logger.warning("Oracle CRPS computation failed: %s", _exc)

        evaluations: dict[str, StrategyEvaluation] = {}

        # Pure GP variance metric: sum of fmc_variance + wind_speed_variance.
        # Monotonically decreasing; baseline is the pre-observation default so cycle 1
        # shows the RAWS seeding contribution and later cycles show drone contributions.
        curr_gp_var = float(
            self._gp_prior.fmc_variance.sum() + self._gp_prior.wind_speed_variance.sum()
        ) if self._gp_prior is not None else self._prev_gp_var_sum
        gp_var_red = max(0.0, self._prev_gp_var_sum - curr_gp_var)
        self._prev_gp_var_sum = curr_gp_var

        if cycle_report.info_field is not None:
            curr_entropy = float(cycle_report.info_field.w.sum())
            prev_entropy = (self._prev_info_entropy
                            if self._prev_info_entropy is not None
                            else curr_entropy)
            reduction = max(0.0, prev_entropy - curr_entropy)
            self._prev_info_entropy = curr_entropy
            evaluations["greedy"] = StrategyEvaluation(
                strategy_name="greedy",
                selected_locations=list(targets),
                entropy_before=prev_entropy,
                entropy_after=curr_entropy,
                entropy_reduction=reduction,
                perr=reduction / max(self.config.n_drones, 1),
                cells_observed=[],
                gp_var_before=self._prev_gp_var_sum + gp_var_red,  # prev value
                gp_var_after=curr_gp_var,
                gp_var_reduction=gp_var_red,
                crps_minutes=crps_min,
                rmse_minutes=rmse_min,
                spread_skill_ratio=spread_skill,
                oracle_crps_minutes=oracle_crps_min,
            )

        # CycleReport is frozen — use dataclasses.replace to attach evaluations.
        cycle_report = dc_replace(cycle_report, evaluations=evaluations)
        self.cycle_reports.append(cycle_report)

        # Accumulate per-cycle metrics row for CSV export.
        _n_cells = self.terrain.shape[0] * self.terrain.shape[1]
        _n_burned = int((self.truth.fire.arrival_times < self.current_time).sum())
        self._accuracy_trace.append({
            "time_min":              round(self.current_time / 60.0, 3),
            "crps_per_cell_minutes": round(crps_per_cell_min, 6),
            "n_active_cells":        n_active_cells,
        })
        self._cycle_metrics_rows.append({
            "scenario_name":         self.config.scenario_name,
            "n_drones":              self.config.n_drones,
            "cycle":                 len(self.cycle_reports),
            "time_min":              round(self.current_time / 60.0, 3),
            "n_obs_cumulative":      self._cumulative_obs,
            "crps_minutes":          round(crps_min, 4),
            "crps_per_cell_minutes": round(crps_per_cell_min, 6),
            "n_active_cells":        n_active_cells,
            "oracle_crps_minutes":   round(oracle_crps_min, 4),
            "spread_skill":          round(spread_skill, 4),
            "gp_var_fmc_mean":       round(float(self._gp_prior.fmc_variance.mean()), 6)
                                     if self._gp_prior is not None else 0.0,
            "gp_var_wind_mean":      round(float(self._gp_prior.wind_speed_variance.mean()), 6)
                                     if self._gp_prior is not None else 0.0,
            "n_burned_cells":        _n_burned,
            "burn_fraction":         round(_n_burned / max(_n_cells, 1), 6),
        })

        # Sync live estimator to the updated orchestrator GP state so the next
        # inter-cycle live estimate starts from the freshest posterior.
        self._live_est.snapshot_from_cycle()
        # Mark info field stale — will be recomputed on the next render frame
        # after the live estimator produces a fresh fire estimate.
        self._live_info_field_dirty = True

        self._assign_drone_waypoints(self.orchestrator._drone_plans)

        logger.info(
            "WISP cycle %d | t=%.0fs | obs=%d | info_w=%.2f | "
            "mission_val_red=%.4f | gp_var=%.1f | gp_var_red=%.1f | "
            "crps=%.2fmin | rmse=%.2fmin | spread_skill=%.2f",
            len(self.cycle_reports),
            self.current_time,
            len(observations),
            self._prev_info_entropy or 0.0,
            evaluations.get("greedy", StrategyEvaluation("greedy", [], 0., 0., 0., 0., [])).entropy_reduction,
            curr_gp_var,
            gp_var_red,
            crps_min,
            rmse_min,
            spread_skill,
        )

    # ------------------------------------------------------------------
    # Drone helpers
    # ------------------------------------------------------------------

    def _assign_drone_waypoints(
        self, drone_plans: list[DronePlan]
    ) -> None:
        """Override every drone's path with the new plan immediately.

        New WISP cycle plans preempt whatever the drone is currently flying —
        the drone drops its remaining waypoints and starts the new route from
        its current position.  Pending (not-yet-spawned) drones are skipped.
        """
        for drone, plan in zip(self.drones, drone_plans):
            if drone.status == "pending":
                continue
            path = plan.waypoints[1:] if len(plan.waypoints) > 1 else []
            if path:
                assign_waypoints(drone, path, self.terrain.resolution_m)

    def _refuel_if_home(self, drone: DroneState) -> None:
        """Reset endurance when a drone returns to its base cell."""
        from .drone_sim import pos_m_to_cell
        cell = pos_m_to_cell(
            drone.position, self.terrain.resolution_m, self.terrain.shape
        )
        if (drone.status in ("idle", "returning")
                and cell == self.config.base_cell):
            drone.endurance_remaining_s = self.config.drone_endurance_s
            drone.status = "idle"
