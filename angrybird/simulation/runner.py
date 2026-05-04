"""
Simulation runners for IGNIS.

CycleRunner (Phase 4b — strategy comparison)
    Cycle-based harness for multi-strategy PERR comparison.  All strategies
    run on the same ensemble each cycle; only the primary strategy's
    observations update the real GP/EnKF state.  No clock, no drone movement.

SimulationRunner (Phase 4b — clock-based demo)
    Clock-based harness implementing the full simulation loop from the
    architecture spec.  Drones move continuously, collect observations, and
    IGNIS cycles trigger at fixed intervals.  Produces an MP4 video via
    FrameRenderer.

LiveEstimator
    Lightweight between-cycle estimator.  Snapshots the GP state at each
    IGNIS cycle boundary, then accumulates raw observations as they arrive
    and computes an updated (FMC mean, wind mean, estimated arrival time)
    at each render frame — using only a GP predict() call + one fire member.
    This avoids the 30-member ensemble and runs in seconds, allowing the
    operator display to reflect new observations immediately rather than
    waiting 20 minutes for the next IGNIS cycle.

SimulationConfig
    Shared configuration dataclass for SimulationRunner (and scenarios.py).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, replace as dc_replace
from typing import Optional

import numpy as np

from ..config import (
    ENSEMBLE_SIZE,
    GRID_RESOLUTION_M,
    N_DRONES,
    SIMULATION_HORIZON_MIN,
)
from ..information import compute_information_field
from ..nelson import nelson_fmc_field
from ..orchestrator import IGNISOrchestrator, FireEngineProtocol
from ..path_planner import plan_paths
from ..selectors import registry as _default_registry
from ..selectors.base import SelectorRegistry
from ..types import (
    CycleReport,
    DroneObservation,
    GPPrior,
    SelectionResult,
    StrategyEvaluation,
)
from .drone_sim import (
    DroneState,
    NoiseConfig,
    assign_waypoints,
    cell_to_pos_m,
    collect_observations,
    move_drone,
)
from .evaluator import CounterfactualEvaluator
from .ground_truth import GroundTruth, compute_wind_field
from .observation_buffer import ObservationBuffer, thin_observations
from ..assimilation import aggregate_drone_observations
from ..config import (
    GP_CORRELATION_LENGTH_FMC_M,
    RAWS_FMC_SIGMA,
    RAWS_WIND_SPEED_SIGMA,
    RAWS_WIND_DIR_SIGMA,
)
from .observer import ObservationSource, SimulatedObserver
from .renderer import FrameRenderer
from ..raws import RAWSObserver, RAWSStation, place_raws_stations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimulationConfig
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Configuration for the clock-based SimulationRunner."""
    dt: float = 10.0                        # simulation timestep (seconds)
    total_time_s: float = 21600.0           # 6 hours
    ignis_cycle_interval_s: float = 1200.0  # IGNIS runs every 20 minutes
    n_drones: int = 5
    drone_speed_ms: float = 15.0            # m/s cruise speed
    drone_endurance_s: float = 1800.0       # 30 minutes per sortie
    camera_footprint_m: float = 100.0       # FMC observation footprint radius
    base_cell: tuple[int, int] = (195, 100) # (row, col) drone home location
    frame_interval: int = 6                 # render one frame every N sim steps
    fps: int = 10                           # video frames per second
    output_path: str = "out/simulation"
    scenario_name: str = "simulation"
    # RAWS stations — 1 randomly placed by default; pass raws_locations to fix positions
    n_raws: int = 1
    raws_locations: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# LiveEstimator
# ---------------------------------------------------------------------------

class LiveEstimator:
    """
    Provides a continuously-updated "best estimate" of FMC, wind, and fire
    arrival times between IGNIS planning cycles.

    Why this exists
    ---------------
    IGNIS cycles run every ~20 minutes and require a full 30-member ensemble.
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
        # After each IGNIS cycle completes:
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
    ) -> None:
        self._orchestrator  = orchestrator
        self._terrain       = terrain
        self.horizon_h      = horizon_h
        # Snapshot of the orchestrator GP at the last cycle boundary.
        # Never modified between cycles — all inter-cycle obs go into _obs_buffer.
        self._live_gp       = copy.deepcopy(orchestrator.gp)
        # ObservationBuffer applies the same 200 m spatial thinning the main
        # runner uses before IGNIS cycles.  Raw obs are added here; thinned()
        # view is computed in compute_estimate without consuming the buffer.
        self._obs_buffer = ObservationBuffer(
            min_spacing_m=200.0,
            resolution_m=terrain.resolution_m,
        )
        self._has_obs       = False   # True when buffer has new unseen obs
        self._cached_prior: Optional[GPPrior] = None  # reused when buffer unchanged

    # ------------------------------------------------------------------

    def snapshot_from_cycle(self) -> None:
        """
        Sync live GP to orchestrator GP state immediately after an IGNIS cycle.

        Called once per cycle boundary.  Deep-copy is ~ms (sklearn GP objects
        are small once fitted) and happens only every 20 sim-minutes.
        The obs buffer is cleared so the next inter-cycle estimate starts fresh
        from the updated posterior rather than re-adding old observations.
        """
        self._live_gp     = copy.deepcopy(self._orchestrator.gp)
        self._obs_buffer  = ObservationBuffer(
            min_spacing_m=200.0,
            resolution_m=self._terrain.resolution_m,
        )
        self._has_obs     = False
        self._cached_prior = None

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
    ) -> tuple[GPPrior, Optional[np.ndarray]]:
        """
        Compute the live best estimate using thinned drone observations.

        When new observations have arrived since the last call, thin the
        buffer (200 m min spacing — identical to the IGNIS cycle pipeline) and
        temporarily condition a copy of the snapshot GP on FMC + wind speed +
        wind direction from those thinned obs.  The snapshot GP itself is never
        mutated between cycles.

        Returns
        -------
        prior : GPPrior
            GP posterior conditioned on all thinned observations collected since
            the last IGNIS cycle (FMC mean, wind speed mean, wind dir mean,
            and their variances).
        arrival_times_h : ndarray or None
            Float32 (rows, cols) estimated ignition time in hours from
            simulation start.  NaN for cells not reached within the horizon.
            None if no fire engine is available.
        """
        if self._has_obs:
            # Aggregate the buffer's raw obs at the full GP correlation length.
            # This mirrors the IGNIS cycle assimilation path and prevents the
            # live estimate from showing spiky artifacts between cycle boundaries
            # (the old thin_observations used only 200 m spacing, dumping dozens
            # of barely-separated obs into the already-fitted GP copy).
            aggregated = aggregate_drone_observations(
                self._obs_buffer._buffer,
                spacing_m=GP_CORRELATION_LENGTH_FMC_M,
                resolution_m=self._obs_buffer._resolution_m,
            )
            if aggregated:
                work_gp    = copy.deepcopy(self._live_gp)
                locs       = [o.location  for o in aggregated]
                fmc_vals   = [o.fmc       for o in aggregated]
                fmc_sigmas = [o.fmc_sigma for o in aggregated]

                # Wind speed — nadir cells only (off-centre footprint obs have NaN)
                ws_obs     = [o for o in aggregated
                              if o.wind_speed is not None
                              and np.isfinite(o.wind_speed)
                              and o.wind_speed_sigma is not None
                              and np.isfinite(o.wind_speed_sigma)]
                ws_locs    = [o.location         for o in ws_obs]
                ws_vals    = [o.wind_speed        for o in ws_obs]
                ws_sigmas  = [o.wind_speed_sigma  for o in ws_obs]

                # Wind direction — same nadir cells (also carry wind_dir)
                wd_obs     = [o for o in ws_obs
                              if o.wind_dir is not None
                              and np.isfinite(o.wind_dir)]
                wd_locs    = [o.location   for o in wd_obs]
                wd_vals    = [o.wind_dir   for o in wd_obs]
                wd_sigmas  = [o.wind_dir_sigma
                              if o.wind_dir_sigma is not None else 10.0
                              for o in wd_obs]

                obs_times = [o.timestamp if o.timestamp is not None else 0.0 for o in aggregated]
                ws_times  = [o.timestamp if o.timestamp is not None else 0.0 for o in ws_obs]
                wd_times  = [o.timestamp if o.timestamp is not None else 0.0 for o in wd_obs]
                work_gp.add_observations(
                    locs, fmc_vals, fmc_sigmas,
                    obs_times=obs_times,
                    ws_vals=ws_vals or None,
                    ws_sigmas=ws_sigmas or None,
                    ws_locs=ws_locs or None,
                    ws_times=ws_times or None,
                    wd_vals=wd_vals or None,
                    wd_sigmas=wd_sigmas or None,
                    wd_locs=wd_locs or None,
                    wd_times=wd_times or None,
                )
                self._cached_prior = work_gp.predict(shape)
            # Always reset so we don't redo the expensive deepcopy/predict unless
            # add_observations() brings in new data between render frames.
            self._has_obs = False

        prior = self._cached_prior if self._cached_prior is not None \
            else self._live_gp.predict(shape)

        arrival_times_h: Optional[np.ndarray] = None
        fire_engine = getattr(self._orchestrator, "fire_engine", None)
        if fire_engine is not None:
            try:
                result = fire_engine.run(
                    self._terrain,
                    prior,
                    fire_state,
                    n_members=1,
                    horizon_min=int(self.horizon_h * 60),
                    rng=rng,
                )
                # member_arrival_times: (n_members, rows, cols) in hours;
                # NaN for cells not reached within the horizon.
                arrival_times_h = result.member_arrival_times[0].astype(np.float32)
            except Exception as exc:
                logger.warning("LiveEstimator fire run failed: %s", exc)

        return prior, arrival_times_h


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
        bimodal_beta: float = 0.3,
        nelson_T_C: float = 28.0,
        nelson_RH: float = 0.20,
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
        lat = float(self.orchestrator.terrain.origin[0])
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
      5. Observations buffered; flushed + thinned at each IGNIS cycle boundary
      6. IGNIS cycle runs (full ensemble → info field → selection → assimilation)
      7. LiveEstimator syncs to updated GP state after each cycle
      8. Frame rendered (every frame_interval steps) with live estimate

    The live estimate panels (FMC+wind, arrival time) update every render
    frame — reflecting new observations immediately rather than waiting for
    the next 20-minute IGNIS cycle.  The information field and mission
    targets still update only at IGNIS cycle boundaries (require full ensemble).

    Args:
        config:        SimulationConfig
        terrain:       TerrainData (static)
        ground_truth:  GroundTruth (mutable — wind + fire updated in-place)
        orchestrator:  IGNISOrchestrator (runs IGNIS cycles)
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
            min_spacing_m=200.0,
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
        self._prev_gp_var_sum: float = (0.04 + 4.0) * _n  # FMC + wind-speed defaults

        base_pos = cell_to_pos_m(config.base_cell, terrain.resolution_m)
        self.drones = [
            DroneState(
                drone_id=f"drone_{i}",
                position=base_pos.copy(),
                speed=config.drone_speed_ms,
                status="idle",
                waypoint_queue=[],
                current_target=None,
                path_history=[base_pos.copy()],
                endurance_remaining_s=config.drone_endurance_s,
                base_position=base_pos.copy(),
            )
            for i in range(config.n_drones)
        ]

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
        orchestrator.gp.add_raws(
            locations=[o.location for o in raws_obs],
            fmc_vals=[o.fmc for o in raws_obs],
            ws_vals=[o.wind_speed for o in raws_obs],
            wd_vals=[o.wind_dir for o in raws_obs],
            fmc_sigmas=[o.fmc_sigma for o in raws_obs],
            ws_sigmas=[o.wind_speed_sigma for o in raws_obs],
            wd_sigmas=[o.wind_dir_sigma for o in raws_obs],
        )

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
            np.full(terrain.shape, 5.0,   dtype=np.float32),
            np.full(terrain.shape, 270.0, dtype=np.float32),
        )

        self.current_time: float = 0.0
        # Trigger first IGNIS cycle immediately at t=0
        self._last_cycle_time: float = -config.ignis_cycle_interval_s
        self.cycle_reports: list[CycleReport] = []

        # Cached IGNIS outputs for the renderer (update per cycle)
        self._gp_prior: Optional[GPPrior] = None
        self._burn_probability: Optional[np.ndarray] = None
        self._mission_targets: list[tuple[int, int]] = []

        # Live estimator — updates between cycles with raw per-step observations
        self._live_est = LiveEstimator(
            orchestrator=orchestrator,
            terrain=terrain,
            horizon_h=horizon_h,
        )
        self._live_gp_prior: Optional[GPPrior] = None
        self._live_arrival_times_h: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> list[CycleReport]:
        """
        Execute the full simulation.

        Returns list[CycleReport] — one per IGNIS cycle.
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

        for step in range(n_steps):
            self.current_time = step * self.config.dt

            # 1. Update ground truth wind
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

            # 2. Advance ground truth fire
            self.truth.fire.step(
                self.config.dt,
                self.truth.wind_speed,
                self.truth.wind_direction,
                self.truth.fmc,
            )

            # 3. Move drones + collect observations
            new_obs_this_step: list[DroneObservation] = []
            for drone in self.drones:
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
                    self.obs_buffer.add(obs)
                    new_obs_this_step.extend(obs)

            # 4. Feed new observations to the live estimator
            #    (GP dirty-flag means refit is deferred to next predict() call)
            if new_obs_this_step:
                self._live_est.add_observations(new_obs_this_step)

            # 5. IGNIS cycle when due
            if (self.current_time - self._last_cycle_time
                    >= self.config.ignis_cycle_interval_s):
                self._run_ignis_cycle()
                self._last_cycle_time = self.current_time

            # 6. Update live estimate at render cadence
            #    Predict (refit if dirty) + 1 fire member — cheap enough to run
            #    every render frame (every frame_interval*dt seconds of sim time).
            if step % self.config.frame_interval == 0:
                fire_state = self.truth.fire.fire_state
                self._live_gp_prior, self._live_arrival_times_h = (
                    self._live_est.compute_estimate(
                        shape=self.terrain.shape,
                        fire_state=fire_state,
                        rng=np.random.default_rng(step),
                    )
                )

            # 7. Render frame
            self.renderer.render_frame(
                step=step,
                time_s=self.current_time,
                ground_truth=self.truth,
                drones=self.drones,
                gp_prior=self._gp_prior,
                burn_probability=self._burn_probability,
                info_field=(self.cycle_reports[-1].info_field
                            if self.cycle_reports else None),
                mission_targets=self._mission_targets,
                cycle_reports=self.cycle_reports,
                live_gp_prior=self._live_gp_prior,
                live_arrival_times_h=self._live_arrival_times_h,
            )

        self.renderer.finalize()
        logger.info(
            "SimulationRunner finished: %d IGNIS cycles | %d frames",
            len(self.cycle_reports),
            self.renderer._frame_count,
        )
        return self.cycle_reports

    # ------------------------------------------------------------------
    # IGNIS cycle
    # ------------------------------------------------------------------

    def _run_ignis_cycle(self) -> None:
        """Flush observation buffer, run orchestrator, dispatch drones."""
        observations = self.obs_buffer.flush_thinned()

        # Advance the GP clock so temporal decay and stale-observation pruning
        # reflect the current simulation time before the cycle fits new data.
        self.orchestrator.gp.update_time(self.current_time)

        fire_state = self.truth.fire.fire_state  # current burn mask (float32)

        # Update Nelson FMC prior mean for current time of day.
        # Simulation clock starts at 06:00 local solar time by convention.
        hour_of_day = (6.0 + self.current_time / 3600.0) % 24.0
        lat = float(self.terrain.origin[0])
        nelson_field = nelson_fmc_field(
            self.terrain,
            T_C=28.0,
            RH=0.20,
            hour_of_day=hour_of_day,
            latitude_deg=lat,
        )
        self.orchestrator.gp.set_nelson_mean(nelson_field)

        mission_queue, cycle_report = self.orchestrator.run_cycle(
            fire_state=fire_state,
            observations=observations,
        )

        # Cache cycle-level outputs for the renderer (info field, targets)
        shape = self.terrain.shape
        self._gp_prior = self.orchestrator.gp.predict(shape)
        self._burn_probability = (
            self.orchestrator.fire_engine.run(
                self.terrain, self._gp_prior, fire_state,
                self.orchestrator.n_members,
                self.orchestrator.horizon_min,
                np.random.default_rng(len(self.cycle_reports)),
            ).burn_probability
            if hasattr(self.orchestrator, "fire_engine")
            else None
        )

        targets = self.orchestrator._previous_selections
        self._mission_targets = list(targets)

        # Mission-value metric: diff of info_field.w.sum() between cycles.
        # w = gp_variance × |sensitivity| × observability — can INCREASE as fire
        # spreads into new high-sensitivity cells, so is NOT a pure information-gain
        # signal.  Use gp_var_reduction below for a monotonically-decreasing metric.
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
            )

        # CycleReport is frozen — use dataclasses.replace to attach evaluations.
        cycle_report = dc_replace(cycle_report, evaluations=evaluations)
        self.cycle_reports.append(cycle_report)

        # Sync live estimator to the updated orchestrator GP state so the next
        # inter-cycle live estimate starts from the freshest posterior.
        self._live_est.snapshot_from_cycle()

        self._assign_drone_waypoints(targets)

        logger.info(
            "IGNIS cycle %d | t=%.0fs | obs=%d | info_w=%.2f | "
            "mission_val_red=%.4f | gp_var=%.1f | gp_var_red=%.1f",
            len(self.cycle_reports),
            self.current_time,
            len(observations),
            self._prev_info_entropy or 0.0,
            evaluations.get("greedy", StrategyEvaluation("greedy", [], 0., 0., 0., 0., [])).entropy_reduction,
            curr_gp_var,
            gp_var_red,
        )

    # ------------------------------------------------------------------
    # Drone helpers
    # ------------------------------------------------------------------

    def _assign_drone_waypoints(
        self, targets: list[tuple[int, int]]
    ) -> None:
        """Assign mission targets to idle or recently-returned drones."""
        available = [d for d in self.drones if d.status == "idle"]
        for drone, target_cell in zip(available, targets):
            assign_waypoints(drone, target_cell, self.terrain.resolution_m)

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
