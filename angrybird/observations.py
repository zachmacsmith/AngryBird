"""
Centralized observation types and storage for IGNIS.

Design principles:
  - Store physical observations, serve mathematical DataPoints.
  - Time-parameterized queries: callers pass query_time; sigmas are decayed
    at query time. The store is append-only (plus pruning). No mutable state.
  - Decay logic lives in each observation class via to_data_points().
  - The GP calls get_data_points() and sees only (location, value, sigma) tuples.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Variable and source enumerations
# ---------------------------------------------------------------------------

class VariableType(Enum):
    FMC            = "fmc"
    WIND_SPEED     = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    TEMPERATURE    = "temperature"
    HUMIDITY       = "humidity"
    FIRE_DETECTION = "fire_detection"


# Backward-compat alias
ObservationType = VariableType


# ---------------------------------------------------------------------------
# DataPoint — the mathematical representation consumed by GP / EnKF
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataPoint:
    """
    A single (location, value, sigma) tuple consumed by the GP/EnKF.
    Mathematical consumers never see the Observation object that produced it.
    sigma is the effective (post-decay) measurement noise.
    """
    location:  tuple[int, int]
    variable:  VariableType
    value:     float
    sigma:     float      # effective sigma after decay
    timestamp: float      # when the original measurement was taken


# ---------------------------------------------------------------------------
# Decay constants and expiry threshold
# ---------------------------------------------------------------------------

# Tau values (seconds) per variable type, used by decaying observations.
TAU: dict[VariableType, float] = {
    VariableType.FMC:            3_600.0,   # 1 hr
    VariableType.WIND_SPEED:     7_200.0,   # 2 hr
    VariableType.WIND_DIRECTION: 3_600.0,   # 1 hr
    VariableType.TEMPERATURE:    3_600.0,
    VariableType.HUMIDITY:       3_600.0,
}

EXPIRY_FACTOR: float = 10.0  # observation expires when sigma > 10× original


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Observation(ABC):
    """
    Interface for all observation types.

    An observation represents a physical measurement event. It may contain
    multiple variable types (a drone measures FMC + wind simultaneously).
    It knows how to decay itself and how to decide when it has expired.

    Subclasses MUST implement:
      to_data_points(query_time) -> list[DataPoint]
      is_expired(query_time) -> bool
      source_id (property)
      timestamp (property)
      variables (property)
    """

    @abstractmethod
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        """
        Convert this observation to mathematical data points at query_time.
        Each returned DataPoint has sigma adjusted for temporal decay.
        May return multiple DataPoints (one per variable measured).
        """
        ...

    @abstractmethod
    def is_expired(self, query_time: float) -> bool:
        """
        Should this observation be pruned at query_time?
        RAWS observations return False (never expire — replaced instead).
        Fire detections return False (permanent).
        Drone observations return True when sigma has inflated > EXPIRY_FACTOR × original.
        """
        ...

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for the source (station ID, drone ID, satellite pass)."""
        ...

    @property
    @abstractmethod
    def timestamp(self) -> float:
        """When this observation was recorded (simulation seconds)."""
        ...

    @property
    @abstractmethod
    def variables(self) -> list[VariableType]:
        """Which variable types this observation contains."""
        ...


# ---------------------------------------------------------------------------
# RAWS observation — fixed ground station, never decays
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RAWSObservation(Observation):
    """
    Fixed ground station reading. Contains FMC + wind speed + wind direction.
    Never decays. Replaced entirely when a new reading arrives from the
    same station (store is keyed on source_id).
    """
    _source_id:          str
    _timestamp:          float
    location:            tuple[int, int]
    fmc:                 float
    fmc_sigma:           float
    wind_speed:          float
    wind_speed_sigma:    float
    wind_direction:      float
    wind_direction_sigma: float

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def variables(self) -> list[VariableType]:
        return [VariableType.FMC, VariableType.WIND_SPEED, VariableType.WIND_DIRECTION]

    def to_data_points(self, query_time: float) -> list[DataPoint]:
        # No decay — sigma is always original
        return [
            DataPoint(self.location, VariableType.FMC,
                      self.fmc, self.fmc_sigma, self._timestamp),
            DataPoint(self.location, VariableType.WIND_SPEED,
                      self.wind_speed, self.wind_speed_sigma, self._timestamp),
            DataPoint(self.location, VariableType.WIND_DIRECTION,
                      self.wind_direction, self.wind_direction_sigma, self._timestamp),
        ]

    def is_expired(self, query_time: float) -> bool:
        return False  # never expires — only replaced


# ---------------------------------------------------------------------------
# Drone observation — decays over time
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DroneObservation(Observation):
    """
    Single measurement from a drone at one grid cell.
    Contains whichever variables the drone's sensors captured.
    Decays exponentially — sigma inflates with age.
    """
    _source_id:           str
    _timestamp:           float
    location:             tuple[int, int]

    # Optional per variable — None means this drone didn't measure it here
    fmc:                  Optional[float] = None
    fmc_sigma:            Optional[float] = None
    wind_speed:           Optional[float] = None
    wind_speed_sigma:     Optional[float] = None
    wind_direction:       Optional[float] = None
    wind_direction_sigma: Optional[float] = None

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def variables(self) -> list[VariableType]:
        v = []
        if self.fmc is not None:
            v.append(VariableType.FMC)
        if self.wind_speed is not None:
            v.append(VariableType.WIND_SPEED)
        if self.wind_direction is not None:
            v.append(VariableType.WIND_DIRECTION)
        return v

    def _effective_sigma(self, original_sigma: float, tau: float,
                          query_time: float) -> float:
        age = max(0.0, query_time - self._timestamp)
        return original_sigma * np.exp(age / tau)

    def to_data_points(self, query_time: float) -> list[DataPoint]:
        points = []
        if self.fmc is not None and self.fmc_sigma is not None:
            sigma = self._effective_sigma(
                self.fmc_sigma, TAU[VariableType.FMC], query_time)
            points.append(DataPoint(
                self.location, VariableType.FMC,
                self.fmc, sigma, self._timestamp))

        if self.wind_speed is not None and self.wind_speed_sigma is not None:
            sigma = self._effective_sigma(
                self.wind_speed_sigma, TAU[VariableType.WIND_SPEED], query_time)
            points.append(DataPoint(
                self.location, VariableType.WIND_SPEED,
                self.wind_speed, sigma, self._timestamp))

        if self.wind_direction is not None and self.wind_direction_sigma is not None:
            sigma = self._effective_sigma(
                self.wind_direction_sigma, TAU[VariableType.WIND_DIRECTION], query_time)
            points.append(DataPoint(
                self.location, VariableType.WIND_DIRECTION,
                self.wind_direction, sigma, self._timestamp))

        return points

    def is_expired(self, query_time: float) -> bool:
        # Expired only if ALL contained variables have decayed beyond threshold
        vars_ = self.variables
        if not vars_:
            return True
        for var in vars_:
            sigma_orig = getattr(self, f"{var.value}_sigma")
            tau = TAU[var]
            sigma_eff = self._effective_sigma(sigma_orig, tau, query_time)
            if sigma_eff <= EXPIRY_FACTOR * sigma_orig:
                return False  # at least one variable is still informative
        return True


# ---------------------------------------------------------------------------
# Fire detection observation — permanent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FireDetectionObservation(Observation):
    """
    Binary fire/no-fire observation from thermal camera or satellite.
    Never decays — a cell that was burning at time T was burning at time T.
    Consumed by the particle filter, not the GP.
    """
    _source_id:  str
    _timestamp:  float
    location:    tuple[int, int]
    is_fire:     bool
    confidence:  float  # P(observation is correct), typically 0.8–0.99

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def variables(self) -> list[VariableType]:
        return [VariableType.FIRE_DETECTION]

    def to_data_points(self, query_time: float) -> list[DataPoint]:
        # value = 1.0 (fire) or 0.0 (no fire); sigma = 1 - confidence
        return [DataPoint(
            self.location, VariableType.FIRE_DETECTION,
            1.0 if self.is_fire else 0.0,
            1.0 - self.confidence,
            self._timestamp,
        )]

    def is_expired(self, query_time: float) -> bool:
        return False  # permanent


# ---------------------------------------------------------------------------
# Satellite FMC observation — coarse multi-cell footprint, decays like drone
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SatelliteFMCObservation(Observation):
    """
    Coarse FMC estimate from satellite (MODIS/VIIRS).
    Covers a footprint of multiple cells. Decays like drone observations.
    """
    _source_id:     str
    _timestamp:     float
    center_location: tuple[int, int]
    footprint_cells: tuple[tuple[int, int], ...]
    fmc:            float
    fmc_sigma:      float  # higher than drone — coarser measurement

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def location(self) -> tuple[int, int]:
        return self.center_location

    @property
    def variables(self) -> list[VariableType]:
        return [VariableType.FMC]

    def to_data_points(self, query_time: float) -> list[DataPoint]:
        age = max(0.0, query_time - self._timestamp)
        sigma = self.fmc_sigma * np.exp(age / TAU[VariableType.FMC])
        return [
            DataPoint(cell, VariableType.FMC, self.fmc, sigma, self._timestamp)
            for cell in self.footprint_cells
        ]

    def is_expired(self, query_time: float) -> bool:
        age = max(0.0, query_time - self._timestamp)
        return (self.fmc_sigma * np.exp(age / TAU[VariableType.FMC])
                > EXPIRY_FACTOR * self.fmc_sigma)


# ---------------------------------------------------------------------------
# Fire retrospect observation — synthetic, non-decaying, replaced each cycle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FireRetrospectObservation(Observation):
    """
    Synthetic FMC + wind observation implied by particle filter weights at the
    active fire front.

    Generated each cycle from weighted ensemble statistics over the fire-active
    correlation domains.  Non-decaying — replaced wholesale each cycle via
    ObservationStore.remove_by_source_prefix("fire_retrospect").
    """
    _source_id:           str      # "fire_retrospect_{domain_id}"
    _timestamp:           float
    location:             tuple[int, int]
    fmc:                  float
    fmc_sigma:            float
    wind_speed:           float
    wind_speed_sigma:     float
    wind_direction:       Optional[float] = None
    wind_direction_sigma: Optional[float] = None

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def variables(self) -> list[VariableType]:
        v = [VariableType.FMC, VariableType.WIND_SPEED]
        if self.wind_direction is not None:
            v.append(VariableType.WIND_DIRECTION)
        return v

    def to_data_points(self, query_time: float) -> list[DataPoint]:
        pts = [
            DataPoint(self.location, VariableType.FMC,
                      self.fmc, self.fmc_sigma, self._timestamp),
            DataPoint(self.location, VariableType.WIND_SPEED,
                      self.wind_speed, self.wind_speed_sigma, self._timestamp),
        ]
        if self.wind_direction is not None and self.wind_direction_sigma is not None:
            pts.append(DataPoint(self.location, VariableType.WIND_DIRECTION,
                                 self.wind_direction, self.wind_direction_sigma,
                                 self._timestamp))
        return pts

    def is_expired(self, query_time: float) -> bool:
        return False  # replaced wholesale, never expires individually


# ---------------------------------------------------------------------------
# Observation store
# ---------------------------------------------------------------------------

class ObservationStore:
    """
    Central repository for all observations.

    Append-only (plus pruning). Queries are time-parameterized — callers
    pass a query_time and receive DataPoints with appropriately decayed sigmas.
    The store never mutates observation objects.

    Thread-safe for concurrent reads. Lock during cycle computation to
    prevent observation ingestion mid-cycle.
    """

    def __init__(self) -> None:
        # RAWS: keyed by source_id. New readings replace old.
        self._raws: dict[str, RAWSObservation] = {}

        # All other observations: append-only list.
        # Pruning removes expired entries periodically.
        self._observations: list[Observation] = []

        self._lock   = threading.RLock()
        self._locked = False

        self._last_prune_time: float = -1.0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_raws(self, obs: RAWSObservation) -> None:
        """Add or replace a RAWS reading. Keyed by source_id."""
        with self._lock:
            self._check_not_locked()
            self._raws[obs.source_id] = obs

    def add(self, obs: Observation) -> None:
        """Add any non-RAWS observation."""
        with self._lock:
            self._check_not_locked()
            self._observations.append(obs)

    def add_batch(self, observations: list[Observation]) -> None:
        """Add multiple observations atomically."""
        with self._lock:
            self._check_not_locked()
            self._observations.extend(observations)

    # ------------------------------------------------------------------
    # Queries — time-parameterized, non-mutating
    # ------------------------------------------------------------------

    def get_data_points(
        self,
        query_time: float,
        variable: Optional[VariableType] = None,
        min_spacing_cells: Optional[int] = None,
    ) -> list[DataPoint]:
        """
        Primary query method. Returns processed DataPoints with decayed sigmas
        appropriate for query_time.

        All observation types are flattened into DataPoints.
        Optionally filter by variable type.
        Optionally thin to one point per min_spacing_cells neighborhood.

        This is what the GP calls.
        """
        points: list[DataPoint] = []

        for raws in self._raws.values():
            points.extend(raws.to_data_points(query_time))

        for obs in self._observations:
            if obs.is_expired(query_time):
                continue
            points.extend(obs.to_data_points(query_time))

        if variable is not None:
            points = [p for p in points if p.variable == variable]

        if min_spacing_cells is not None:
            points = self._thin(points, min_spacing_cells)

        return points

    def get_fire_detections(
        self, since: Optional[float] = None
    ) -> list[FireDetectionObservation]:
        """
        Return fire detection observations for the particle filter.
        Returned as full observation objects (not DataPoints) because the
        particle filter needs is_fire and confidence fields.
        """
        detections = [o for o in self._observations
                      if isinstance(o, FireDetectionObservation)]
        if since is not None:
            detections = [d for d in detections if d.timestamp >= since]
        return detections

    def get_raw_observations(
        self, variable: Optional[VariableType] = None
    ) -> list[Observation]:
        """
        Return raw observation objects (undecayed).
        For diagnostics, visualization, and debugging.
        """
        result: list[Observation] = list(self._raws.values())
        result.extend(self._observations)
        if variable is not None:
            result = [o for o in result if variable in o.variables]
        return result

    # ------------------------------------------------------------------
    # Thinning — operates on DataPoints (post-decay)
    # ------------------------------------------------------------------

    def _thin(self, points: list[DataPoint], min_spacing: int) -> list[DataPoint]:
        """Keep lowest-sigma point per spatial neighbourhood."""
        sorted_pts = sorted(points, key=lambda p: p.sigma)
        thinned: list[DataPoint] = []
        for pt in sorted_pts:
            if all(
                abs(pt.location[0] - t.location[0]) > min_spacing or
                abs(pt.location[1] - t.location[1]) > min_spacing
                for t in thinned
            ):
                thinned.append(pt)
        return thinned

    # ------------------------------------------------------------------
    # Pruning — removes expired observations for efficiency
    # ------------------------------------------------------------------

    def remove_by_source_prefix(self, prefix: str) -> int:
        """Remove all non-RAWS observations whose source_id starts with prefix."""
        before = len(self._observations)
        self._observations = [
            o for o in self._observations
            if not o.source_id.startswith(prefix)
        ]
        return before - len(self._observations)

    def prune(self, current_time: float) -> int:
        """
        Remove observations expired as of current_time.
        Returns number of observations pruned.
        RAWS and FireDetectionObservation are never pruned.
        """
        if current_time <= self._last_prune_time:
            return 0

        before = len(self._observations)
        self._observations = [
            obs for obs in self._observations
            if not obs.is_expired(current_time)
        ]
        pruned = before - len(self._observations)
        self._last_prune_time = current_time
        return pruned

    # ------------------------------------------------------------------
    # Cycle locking
    # ------------------------------------------------------------------

    def lock(self) -> None:
        """Lock during cycle computation. Prevents ingestion."""
        self._lock.acquire()
        self._locked = True

    def unlock(self) -> None:
        """Unlock after cycle. Allows ingestion."""
        self._locked = False
        self._lock.release()

    def _check_not_locked(self) -> None:
        if self._locked:
            raise RuntimeError(
                "Cannot add observations during cycle computation. "
                "Buffer externally and add after unlock().")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def count(self) -> dict:
        """Observation counts by type."""
        counts: dict = {"raws": len(self._raws), "other": 0}
        by_type: dict[str, int] = {}
        for obs in self._observations:
            name = type(obs).__name__
            by_type[name] = by_type.get(name, 0) + 1
        counts.update(by_type)
        counts["total"] = counts["raws"] + len(self._observations)
        return counts

    # ------------------------------------------------------------------
    # Fork — for LiveEstimator hypothetical queries
    # ------------------------------------------------------------------

    def fork(self) -> ObservationStore:
        """
        Create a mutable deep copy of this store.
        Used by LiveEstimator to build a temporary working copy without
        affecting the live store.
        """
        new = ObservationStore()
        new._raws         = dict(self._raws)
        new._observations = list(self._observations)
        new._last_prune_time = self._last_prune_time
        return new

    def __deepcopy__(self, memo: dict) -> ObservationStore:
        """Custom deepcopy: creates a fresh RLock (threading.RLock is not copyable)."""
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_lock":
                setattr(result, k, threading.RLock())
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


# ---------------------------------------------------------------------------
# Ingestion buffer — buffers while store is locked
# ---------------------------------------------------------------------------

class IngestionBuffer:
    """
    Buffers observations while the store is locked during cycle computation.
    External sources (drone telemetry, satellite passes) push here.
    Flushed to store after each cycle via flush().
    """

    def __init__(self, store: ObservationStore) -> None:
        self._store        = store
        self._pending:      list[Observation]     = []
        self._pending_raws: list[RAWSObservation] = []

    def add(self, obs: Observation) -> None:
        if isinstance(obs, RAWSObservation):
            self._pending_raws.append(obs)
        else:
            self._pending.append(obs)

    def flush(self) -> None:
        """Push all buffered observations to store. Call after unlock()."""
        for raws in self._pending_raws:
            self._store.add_raws(raws)
        if self._pending:
            self._store.add_batch(self._pending)
        self._pending.clear()
        self._pending_raws.clear()
