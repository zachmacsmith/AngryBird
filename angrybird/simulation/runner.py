"""
Simulation runner: wraps the core orchestrator with simulated observers and
full four-way strategy comparison.

Phase 4b — simulation harness only. None of this ships with production code.
(Build order test: "would you ship it with real drones?" → No → simulation/)

Design:
  - All strategies run on the SAME ensemble each cycle (same seed → same data).
    This is critical for fair comparison: differences in PERR are purely due
    to the selection algorithm, not to different stochastic fire runs.
  - The SimulationRunner computes the ensemble externally and passes it to
    orchestrator.run_cycle(ensemble=...) to avoid redundant computation.
  - Only the PRIMARY strategy's observations update the real GP/EnKF state.
    Other strategies are evaluated counterfactually — "what would have happened."
  - Pending observations (primary strategy's cells from cycle N) are collected
    at the end of cycle N and passed to the orchestrator at the start of cycle N+1.

Usage:
    from angrybird.simulation import SimulationRunner, generate_ground_truth

    truth  = generate_ground_truth(shape, gp_prior, seed=42)
    runner = SimulationRunner(
        orchestrator, truth, fire_engine,
        strategies=["greedy", "qubo", "uniform", "fire_front"],
    )
    reports = runner.run_comparison(fire_states)   # list[CycleReport]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import (
    ENSEMBLE_SIZE,
    GRID_RESOLUTION_M,
    N_DRONES,
    SIMULATION_HORIZON_MIN,
)
from ..information import compute_information_field
from ..orchestrator import FireEngineProtocol, IGNISOrchestrator
from ..path_planner import plan_paths
from ..selectors import registry as _default_registry
from ..selectors.base import SelectorRegistry
from ..types import (
    CycleReport,
    DroneObservation,
    SelectionResult,
)
from .evaluator import CounterfactualEvaluator
from .ground_truth import GroundTruth
from .observer import ObservationSource, SimulatedObserver

logger = logging.getLogger(__name__)


class SimulationRunner:
    """
    Simulation harness for multi-cycle, multi-strategy comparison.

    Args:
        orchestrator:       IGNISOrchestrator — holds GP + terrain + primary selector
        ground_truth:       GroundTruth — hidden true state (FMC, wind fields)
        fire_engine:        implements FireEngineProtocol
        strategies:         names of strategies to compare (must be in registry)
        primary_strategy:   which strategy's observations update real state
                            (defaults to orchestrator.selector_name)
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

        # Observations collected at end of each cycle for the next cycle's assimilation
        self._pending_observations: list[DroneObservation] = []

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
          6. Collect primary strategy's observations (fed to orchestrator as assimilation input)
          7. Call orchestrator.run_cycle with pre-built ensemble + pending observations
          8. Assemble full CycleReport with evaluations from all strategies
        """
        shape = self.orchestrator.terrain.shape
        rng   = np.random.default_rng(cycle_seed)

        # Current GP prior — reflects all observations assimilated so far
        gp_prior = self.orchestrator.gp.predict(shape)

        # 1. Shared ensemble — all strategies evaluate against this
        #    (PotentialBugs1 §8: identical seed → identical ensemble → fair comparison)
        ensemble = self.fire_engine.run(
            self.orchestrator.terrain, gp_prior, fire_state,
            self.n_members, self.horizon_min, rng,
        )

        # 2. Information field
        info_field = compute_information_field(
            ensemble, gp_prior,
            resolution_m=self.resolution_m,
            horizon_minutes=self.horizon_min,
        )

        # 3. Run all strategies
        selection_results: dict[str, SelectionResult] = {}
        for name in self.strategies:
            try:
                selection_results[name] = self.registry.run(
                    name, info_field, self.orchestrator.gp, ensemble, k=self.n_drones,
                )
            except Exception as exc:
                logger.warning("Strategy '%s' raised an exception: %s", name, exc)

        # 4. Plan paths for every strategy
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

        # 5. Counterfactual evaluation — read-only, no state mutation
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

        # 6. Primary strategy drives actual state: collect its observations
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

        # 7. Orchestrator cycle — passes pre-built ensemble and pending observations
        #    so neither the ensemble nor the GP are computed twice.
        _, cycle_report = self.orchestrator.run_cycle(
            fire_state=fire_state,
            observations=self._pending_observations,
            ensemble=ensemble,
        )

        # Stage observations for the next cycle's assimilation
        self._pending_observations = new_observations

        # 8. Attach full evaluations to the report
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
        """
        Run a multi-cycle comparison across a sequence of fire states.

        Args:
            fire_states: one burn mask (bool/float32[rows, cols]) per cycle.
                         len(fire_states) determines the number of cycles.
            base_seed:   base seed; cycle index is added so each cycle is
                         independently reproducible (PotentialBugs1 §8).

        Returns:
            list[CycleReport] with full per-strategy evaluations for every cycle.
            Use evaluations[strategy_name].perr for the headline PERR metric.
        """
        reports: list[CycleReport] = []
        for cycle_idx, fire_state in enumerate(fire_states):
            report = self.run_cycle(
                fire_state=fire_state,
                cycle_seed=base_seed + cycle_idx,
            )
            reports.append(report)

        # Summary log
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
