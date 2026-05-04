"""Shared data types for all IGNIS components. Agree on these before writing component code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class TerrainData:
    # Spatial layers — all float32[rows, cols] unless noted
    elevation:           np.ndarray  # meters
    slope:               np.ndarray  # degrees
    aspect:              np.ndarray  # degrees from north
    fuel_model:          np.ndarray  # int16[rows, cols], SB40 codes (91-204)
    canopy_cover:        np.ndarray  # fraction 0-1
    canopy_height:       np.ndarray  # meters
    canopy_base_height:  np.ndarray  # meters
    canopy_bulk_density: np.ndarray  # kg/m³
    # Grid metadata
    shape:        Tuple[int, int]                          # (rows, cols)
    resolution_m: float                                    # meters per cell
    origin_latlon: Optional[Tuple[float, float]] = None   # (lat, lon) of NW corner

    @property
    def origin(self):
        if self.origin_latlon is not None:
            return self.origin_latlon

        return (37.5, -119.5)

@dataclass(frozen=True)
class GPPrior:
    fmc_mean: np.ndarray              # float32[rows, cols]
    fmc_variance: np.ndarray          # float32[rows, cols]
    wind_speed_mean: np.ndarray       # float32[rows, cols], m/s
    wind_speed_variance: np.ndarray   # float32[rows, cols]
    wind_dir_mean: np.ndarray         # float32[rows, cols], degrees from north
    wind_dir_variance: np.ndarray     # float32[rows, cols]


@dataclass(frozen=True)
class EnsembleResult:
    member_arrival_times: np.ndarray   # float32[N, rows, cols], sentinel=2×horizon for unburned
    member_fmc_fields: np.ndarray      # float32[N, rows, cols], perturbed FMC used
    member_wind_fields: np.ndarray     # float32[N, rows, cols], perturbed wind speed used
    burn_probability: np.ndarray       # float32[rows, cols], fraction of members that burned
    mean_arrival_time: np.ndarray      # float32[rows, cols]
    arrival_time_variance: np.ndarray  # float32[rows, cols]
    n_members: int
    member_fire_types: Optional[np.ndarray] = field(default=None)      # int8[N, rows, cols]: 1=surface 2=crown
    member_wind_dir_fields: Optional[np.ndarray] = field(default=None) # float32[N, rows, cols], perturbed wind direction used


@dataclass(frozen=True)
class InformationField:
    w: np.ndarray                        # float32[rows, cols], total information value
    w_by_variable: dict[str, np.ndarray] # per-variable breakdown ("fmc", "wind_speed", "wind_dir")
    sensitivity: dict[str, np.ndarray]   # per-variable sensitivity S_v
    gp_variance: dict[str, np.ndarray]   # per-variable GP variance σ²_v


@dataclass(frozen=True)
class SelectionResult:
    selected_locations: list[tuple[int, int]]  # grid (row, col) indices
    marginal_gains: list[float]                # w_i at time of each selection
    cumulative_gain: list[float]               # running total
    strategy_name: str
    compute_time_s: float
    kind: str = "points"
    solver_metadata: Optional[dict] = None    # QUBO-specific: energy, chain breaks, solver name


@dataclass(frozen=True)
class PathSelectionResult:
    """Returned by path selectors that skip the point→path step entirely."""
    kind: str                         # always "paths"
    drone_plans: list["DronePlan"]
    strategy_name: str
    compute_time_s: float
    total_info: float
    marginal_gains: list[float]       # per-drone info contribution


@dataclass(frozen=True)
class DroneObservation:
    location: tuple[int, int]   # grid (row, col)
    fmc: float                  # measured fuel moisture content (fraction)
    fmc_sigma: float            # measurement noise std dev
    wind_speed: float           # m/s
    wind_speed_sigma: float     # measurement noise std dev
    wind_dir: Optional[float] = None        # degrees from north
    wind_dir_sigma: Optional[float] = None
    timestamp: Optional[float] = None      # simulation time (seconds)
    drone_id: Optional[str] = None


@dataclass(frozen=True)
class MissionRequest:
    drone_id: int                              # which drone executes this path
    path: list[tuple[float, float]]            # ordered (lat, lon) waypoints
    information_value: float                   # total w across all waypoints
    dominant_variable: str                     # most common dominant variable along path
    expiry_minutes: float                      # after this, re-solve needed


@dataclass(frozen=True)
class MissionQueue:
    requests: list[MissionRequest]  # one per drone, sorted by information_value descending


@dataclass(frozen=True)
class DronePlan:
    drone_id: int
    waypoints: list[tuple[int, int]]   # ordered (row, col) grid indices
    cells_observed: list[tuple[int, int]]  # all cells overflown (including path)


@dataclass(frozen=True)
class StrategyEvaluation:
    strategy_name: str
    selected_locations: list[tuple[int, int]]
    entropy_before: float
    entropy_after: float
    entropy_reduction: float
    perr: float                  # per-drone entropy reduction
    cells_observed: list[tuple[int, int]]
    # Pure GP variance sum (fmc_variance + wind_speed_variance, summed over grid).
    # Unlike entropy_reduction, this is monotonically decreasing and unconfounded
    # by fire-spread changes in sensitivity / observability.
    gp_var_before: float = 0.0
    gp_var_after: float = 0.0
    gp_var_reduction: float = 0.0


@dataclass(frozen=True)
class CycleReport:
    cycle_id: int
    info_field: InformationField
    evaluations: dict[str, StrategyEvaluation]
    ensemble_summary: dict
    placement_stability: float        # Jaccard similarity with previous cycle's primary selections
    gp_prior: Optional[GPPrior] = None           # GP posterior used for this cycle's ensemble
    selection_result: Optional["SelectionResult | PathSelectionResult"] = None
    start_time: float = 0.0           # simulation clock at cycle start (seconds)
