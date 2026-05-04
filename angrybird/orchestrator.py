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
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .assimilation import assimilate_observations
from .config import (
    ENSEMBLE_SIZE,
    GRID_RESOLUTION_M,
    N_DRONES,
    SIMULATION_HORIZON_MIN,
)
from .gp import IGNISGPPrior
from .information import compute_information_field
from .path_planner import plan_paths, selections_to_mission_queue
from .selectors import registry as _default_registry
from .selectors.base import SelectorRegistry
from .types import (
    CycleReport,
    DroneObservation,
    EnsembleResult,
    GPPrior,
    InformationField,
    MissionQueue,
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
    ) -> EnsembleResult: ...


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class IGNISOrchestrator:
    """
    Sequences the full IGNIS pipeline for one or many cycles.

    Args:
        terrain:           static TerrainData (loaded once, never mutated)
        gp:                IGNISGPPrior — mutated by add_observations each cycle
        fire_engine:       implements FireEngineProtocol
        selector_name:     which strategy drives the live mission queue
        selector_registry: plug-and-play registry of all strategies
        n_drones:          number of drones
        horizon_min:       fire-spread simulation horizon in minutes
        n_members:         ensemble size
        staging_area:      (row, col) drone launch/return point
        resolution_m:      grid cell size in metres
        base_seed:         base random seed; cycle_id is added each cycle
    """

    def __init__(
        self,
        terrain: TerrainData,
        gp: IGNISGPPrior,
        fire_engine: FireEngineProtocol,
        selector_name: str = "greedy",
        selector_registry: SelectorRegistry = _default_registry,
        n_drones: int = N_DRONES,
        horizon_min: int = SIMULATION_HORIZON_MIN,
        n_members: int = ENSEMBLE_SIZE,
        staging_area: tuple[int, int] = (0, 0),
        resolution_m: float = GRID_RESOLUTION_M,
        base_seed: int = 0,
        bimodal_alpha: float = 0.5,
        bimodal_beta: float = 0.3,
    ) -> None:
        self.terrain          = terrain
        self.gp               = gp
        self.fire_engine      = fire_engine
        self.selector_name    = selector_name
        self.registry         = selector_registry
        self.n_drones         = n_drones
        self.horizon_min      = horizon_min
        self.n_members        = n_members
        self.staging_area     = staging_area
        self.resolution_m     = resolution_m
        self.base_seed        = base_seed
        self.bimodal_alpha    = bimodal_alpha
        self.bimodal_beta     = bimodal_beta
        self.cycle_count      = 0
        self._previous_selections: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        fire_state: np.ndarray,
        observations: list[DroneObservation],
        ensemble: Optional[EnsembleResult] = None,
        priority_weight_field: Optional[np.ndarray] = None,
        exclusion_mask: Optional[np.ndarray] = None,
    ) -> tuple[MissionQueue, CycleReport]:
        """
        Execute one full IGNIS cycle.

        Args:
            fire_state:            bool/float32[rows, cols] current burn mask
            observations:          DroneObservations from the previous cycle
            ensemble:              pre-computed EnsembleResult (SimulationRunner
                                   passes one to avoid double computation)
            priority_weight_field: operator priority overlay (>1 in priority regions)
            exclusion_mask:        operator exclusion zones (True = excluded)

        Returns:
            (MissionQueue, CycleReport) — queue for the UTM layer + diagnostics.
            CycleReport.evaluations is empty here; SimulationRunner fills it.
        """
        self.cycle_count += 1
        rng   = np.random.default_rng(self.base_seed + self.cycle_count)
        shape = self.terrain.shape

        # 1. Snapshot GP prior before assimilation (used for replan-flag comparison)
        gp_prior_before = self.gp.predict(shape)

        # 2. Assimilate previous cycle's observations
        #    If ensemble not yet available use a neutral placeholder so the EnKF
        #    still works (thinned obs will be empty on cycle 1 anyway).
        assim_ensemble = ensemble if ensemble is not None else _neutral_ensemble(shape, self.n_members)
        assim = assimilate_observations(
            gp=self.gp,
            ensemble=assim_ensemble,
            observations=observations,
            shape=shape,
            resolution_m=self.resolution_m,
            gp_prior=gp_prior_before,
            rng=rng,
        )

        # 3. Updated GP prior (post-assimilation; used for ensemble + info field)
        gp_prior = self.gp.predict(shape)

        # 4. Ensemble — use caller-provided or run fire engine
        if ensemble is None:
            ensemble = self.fire_engine.run(
                self.terrain, gp_prior, fire_state,
                self.n_members, self.horizon_min, rng,
            )

        # 5. Information field
        info_field = compute_information_field(
            ensemble, gp_prior,
            resolution_m=self.resolution_m,
            horizon_minutes=self.horizon_min,
            priority_weight_field=priority_weight_field,
            exclusion_mask=exclusion_mask,
            bimodal_alpha=self.bimodal_alpha,
            bimodal_beta=self.bimodal_beta,
        )

        # 6. Primary strategy selection
        primary_result = self.registry.run(
            self.selector_name, info_field, self.gp, ensemble, k=self.n_drones,
        )

        # 7. Plan paths + build mission queue
        drone_plans = plan_paths(
            primary_result.selected_locations,
            staging_area=self.staging_area,
            n_drones=self.n_drones,
            shape=shape,
            resolution_m=self.resolution_m,
        )
        mission_queue = selections_to_mission_queue(
            primary_result.selected_locations,
            info_field, self.terrain, self.resolution_m,
        )

        # 8. Placement stability metric
        stability = jaccard(primary_result.selected_locations, self._previous_selections)
        self._previous_selections = primary_result.selected_locations

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
        )

        logger.info(
            "Cycle %d | strategy=%s | selected=%d | stability=%.2f | obs=%d",
            self.cycle_count, self.selector_name,
            len(primary_result.selected_locations),
            stability, assim["n_obs_used"],
        )
        return mission_queue, cycle_report


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
        member_fmc_fields=np.full((n_members, rows, cols), 0.10,  dtype=np.float32),
        member_wind_fields=np.full((n_members, rows, cols), 5.0,   dtype=np.float32),
        member_wind_dir_fields=np.full((n_members, rows, cols), 270.0, dtype=np.float32),
        burn_probability=zero,
        mean_arrival_time=zero,
        arrival_time_variance=zero,
        n_members=n_members,
    )
