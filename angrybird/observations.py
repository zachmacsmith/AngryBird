"""
Centralized observation types and storage for IGNIS.

All observation ingestion, decay, and pruning is managed here.
GP, EnKF, and visualization read from ObservationStore; they don't store obs.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ObservationType(Enum):
    FMC            = "fmc"
    WIND_SPEED     = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    FIRE_DETECTION = "fire_detection"
    TEMPERATURE    = "temperature"
    HUMIDITY       = "humidity"


class ObservationSource(Enum):
    RAWS                = "raws"
    DRONE_MULTISPECTRAL = "drone_multispectral"
    DRONE_ANEMOMETER    = "drone_anemometer"
    DRONE_WEATHER       = "drone_weather"
    DRONE_THERMAL       = "drone_thermal"
    SATELLITE_VIIRS     = "satellite_viirs"
    SATELLITE_GOES      = "satellite_goes"
    SATELLITE_MODIS     = "satellite_modis"


# ---------------------------------------------------------------------------
# Base observation class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """Immutable observation. Subclass for non-standard decay behaviour."""
    location:  tuple[int, int]      # (row, col) in grid coordinates
    obs_type:  ObservationType
    source:    ObservationSource
    value:     float                # measured value in SI units
    sigma:     float                # measurement noise (original, never inflated)
    timestamp: float                # simulation time in seconds
    source_id: str                  # e.g. "RAWS_CEDU", "drone_03", "VIIRS_pass_12"

    def effective_sigma(self, current_time: float, tau: float) -> float:
        """Sigma inflated by temporal decay."""
        age = max(0.0, current_time - self.timestamp)
        return self.sigma * np.exp(age / tau)

    def is_expired(self, current_time: float, tau: float,
                   drop_factor: float = 10.0) -> bool:
        return self.effective_sigma(current_time, tau) > drop_factor * self.sigma


# ---------------------------------------------------------------------------
# Specialised subclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RAWSObservation(Observation):
    """
    RAWS readings don't decay — they're replaced by fresh readings each cycle.
    Sigma is fixed at the original measurement noise regardless of age.
    """

    def effective_sigma(self, current_time: float, tau: float) -> float:
        return self.sigma

    def is_expired(self, current_time: float, tau: float,
                   drop_factor: float = 10.0) -> bool:
        return False


@dataclass(frozen=True)
class FireDetectionObservation(Observation):
    """
    Binary fire/no-fire detection. Permanent — a burned cell stays burned.
    Value = 1.0 for fire, 0.0 for confirmed no-fire.
    """
    is_fire:    bool  = True
    confidence: float = 0.95

    def effective_sigma(self, current_time: float, tau: float) -> float:
        return self.sigma

    def is_expired(self, current_time: float, tau: float,
                   drop_factor: float = 10.0) -> bool:
        return False


@dataclass(frozen=True)
class SatelliteObservation(Observation):
    """
    Satellite observation with a spatial footprint.
    footprint_cells lists all grid cells covered by the pixel.
    Decays normally (satellite FMC estimates go stale like drone obs).
    """
    footprint_cells: tuple[tuple[int, int], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Observation store
# ---------------------------------------------------------------------------

class ObservationStore:
    """
    Centralized, thread-safe observation management.

    Thread safety: an RLock protects all mutations. During a cycle's prior
    computation call lock_for_cycle() / unlock_cycle() to prevent concurrent
    ingestion from partially-updated state.

    Decay config maps ObservationType → tau (seconds). Observations are pruned
    when their effective sigma exceeds drop_factor × original sigma.
    """

    def __init__(self, decay_config: dict[ObservationType, float]):
        self._decay_config = decay_config
        self._current_time: float = 0.0

        # RAWS: keyed by station_id → {ObservationType → RAWSObservation}.
        # Re-adding a station replaces all its previous readings atomically.
        self._raws: dict[str, dict[ObservationType, RAWSObservation]] = {}

        # Drone / satellite: append-only (until pruned).
        self._drone:     dict[ObservationType, list[Observation]] = defaultdict(list)
        self._satellite: dict[ObservationType, list[SatelliteObservation]] = defaultdict(list)

        # Fire detections: permanent, never pruned.
        self._fire_detections: list[FireDetectionObservation] = []

        self._lock             = threading.RLock()
        self._locked_for_cycle = False

        self._total_ingested = 0
        self._total_pruned   = 0

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    def update_time(self, t: float) -> None:
        """Advance the store clock. Call once per cycle."""
        self._current_time = t

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_raws(self, station_id: str, observations: list[RAWSObservation]) -> None:
        """Replace all readings for this station atomically."""
        with self._lock:
            if self._locked_for_cycle:
                raise RuntimeError(
                    "Cannot add observations during cycle computation. "
                    "Buffer externally and add after unlock_cycle()."
                )
            self._raws[station_id] = {obs.obs_type: obs for obs in observations}
            self._total_ingested += len(observations)

    def add_drone_observations(self, observations: list[Observation]) -> None:
        """Append drone observations. Subject to temporal decay and pruning."""
        with self._lock:
            if self._locked_for_cycle:
                raise RuntimeError(
                    "Cannot add observations during cycle computation."
                )
            for obs in observations:
                self._drone[obs.obs_type].append(obs)
            self._total_ingested += len(observations)

    def add_satellite_observations(self, observations: list[SatelliteObservation]) -> None:
        with self._lock:
            if self._locked_for_cycle:
                raise RuntimeError(
                    "Cannot add observations during cycle computation."
                )
            for obs in observations:
                self._satellite[obs.obs_type].append(obs)
            self._total_ingested += len(observations)

    def add_fire_detections(self, detections: list[FireDetectionObservation]) -> None:
        with self._lock:
            if self._locked_for_cycle:
                raise RuntimeError(
                    "Cannot add observations during cycle computation."
                )
            self._fire_detections.extend(detections)
            self._total_ingested += len(detections)

    # ------------------------------------------------------------------
    # Cycle lock
    # ------------------------------------------------------------------

    def lock_for_cycle(self) -> None:
        """Block observation ingestion during cycle prior computation."""
        self._lock.acquire()
        self._locked_for_cycle = True

    def unlock_cycle(self) -> None:
        """Re-allow observation ingestion after cycle completes."""
        self._locked_for_cycle = False
        self._lock.release()

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune(self, drop_factor: float = 10.0) -> int:
        """
        Remove expired drone and satellite observations.
        RAWS and fire detections are never pruned.
        Returns number of observations removed.
        """
        pruned = 0
        for obs_type in list(self._drone.keys()):
            tau    = self._decay_config.get(obs_type, float('inf'))
            before = len(self._drone[obs_type])
            self._drone[obs_type] = [
                obs for obs in self._drone[obs_type]
                if not obs.is_expired(self._current_time, tau, drop_factor)
            ]
            pruned += before - len(self._drone[obs_type])

        for obs_type in list(self._satellite.keys()):
            tau    = self._decay_config.get(obs_type, float('inf'))
            before = len(self._satellite[obs_type])
            self._satellite[obs_type] = [
                obs for obs in self._satellite[obs_type]
                if not obs.is_expired(self._current_time, tau, drop_factor)
            ]
            pruned += before - len(self._satellite[obs_type])

        self._total_pruned += pruned
        return pruned

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all_for_type(
        self,
        obs_type: ObservationType,
        include_raws:      bool = True,
        include_drone:     bool = True,
        include_satellite: bool = True,
    ) -> list[Observation]:
        """All observations of a type across all sources (original sigma)."""
        result: list[Observation] = []
        if include_raws:
            for station in self._raws.values():
                if obs_type in station:
                    result.append(station[obs_type])
        if include_drone:
            result.extend(self._drone.get(obs_type, []))
        if include_satellite:
            result.extend(self._satellite.get(obs_type, []))
        return result

    def get_decayed_for_type(
        self, obs_type: ObservationType
    ) -> tuple[list[Observation], list[float]]:
        """
        All observations of a type with their effective (age-inflated) sigmas.
        RAWS returns original sigma (no decay). Drone/satellite return decayed sigma.

        Returns (observations, effective_sigmas) — parallel lists.
        This is what the GP consumes for fitting.
        """
        tau = self._decay_config.get(obs_type, float('inf'))
        observations: list[Observation] = []
        effective_sigmas: list[float]   = []

        for station in self._raws.values():
            if obs_type in station:
                obs = station[obs_type]
                observations.append(obs)
                effective_sigmas.append(obs.sigma)

        for obs in self._drone.get(obs_type, []):
            observations.append(obs)
            effective_sigmas.append(obs.effective_sigma(self._current_time, tau))

        for obs in self._satellite.get(obs_type, []):
            observations.append(obs)
            effective_sigmas.append(obs.effective_sigma(self._current_time, tau))

        return observations, effective_sigmas

    def get_thinned_for_type(
        self,
        obs_type: ObservationType,
        min_spacing_cells: int,
    ) -> tuple[list[Observation], list[float]]:
        """
        Return observations thinned to one per min_spacing_cells spatial bin.
        Keeps the lowest-sigma observation in each bin.
        Used by the GP to avoid singularity from dense satellite swath obs.
        """
        obs_list, sigmas = self.get_decayed_for_type(obs_type)
        paired = sorted(zip(obs_list, sigmas), key=lambda x: x[1])
        thinned_obs: list[Observation] = []
        thinned_sigs: list[float]      = []
        for obs, sigma in paired:
            if all(
                abs(obs.location[0] - t.location[0]) > min_spacing_cells or
                abs(obs.location[1] - t.location[1]) > min_spacing_cells
                for t in thinned_obs
            ):
                thinned_obs.append(obs)
                thinned_sigs.append(sigma)
        return thinned_obs, thinned_sigs

    def get_fire_detections(
        self, since: Optional[float] = None
    ) -> list[FireDetectionObservation]:
        """Fire detections, optionally filtered to those after a timestamp."""
        if since is None:
            return list(self._fire_detections)
        return [d for d in self._fire_detections if d.timestamp >= since]

    def get_observations_near(
        self,
        location: tuple[int, int],
        radius_cells: int,
        obs_type: Optional[ObservationType] = None,
    ) -> list[Observation]:
        """Spatial query: all observations within radius_cells of location."""
        result = []
        for obs in self._iter_all(obs_type):
            dr = obs.location[0] - location[0]
            dc = obs.location[1] - location[1]
            if dr * dr + dc * dc <= radius_cells * radius_cells:
                result.append(obs)
        return result

    def get_observation_locations(
        self, obs_type: Optional[ObservationType] = None
    ) -> list[tuple[int, int]]:
        """All unique observation locations. Useful for visualization overlays."""
        return list({obs.location for obs in self._iter_all(obs_type)})

    def _iter_all(
        self, obs_type: Optional[ObservationType] = None
    ) -> Iterator[Observation]:
        """Iterate over all stored observations, optionally filtered by type."""
        for station in self._raws.values():
            for ot, obs in station.items():
                if obs_type is None or ot == obs_type:
                    yield obs
        for ot, obs_list in self._drone.items():
            if obs_type is None or ot == obs_type:
                yield from obs_list
        for ot, obs_list in self._satellite.items():
            if obs_type is None or ot == obs_type:
                yield from obs_list
        if obs_type is None or obs_type == ObservationType.FIRE_DETECTION:
            yield from self._fire_detections

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def count(self, obs_type: Optional[ObservationType] = None) -> dict:
        counts = {
            "raws":           0,
            "drone":          0,
            "satellite":      0,
            "fire_detection": len(self._fire_detections),
            "total":          0,
        }
        for station in self._raws.values():
            for ot in station:
                if obs_type is None or ot == obs_type:
                    counts["raws"] += 1
        for ot, obs_list in self._drone.items():
            if obs_type is None or ot == obs_type:
                counts["drone"] += len(obs_list)
        for ot, obs_list in self._satellite.items():
            if obs_type is None or ot == obs_type:
                counts["satellite"] += len(obs_list)
        counts["total"] = sum(
            counts[k] for k in ("raws", "drone", "satellite", "fire_detection")
        )
        return counts

    def age_summary(self) -> dict[ObservationType, dict]:
        """Min/max/mean age of drone observations per type. Diagnostic."""
        summary = {}
        for obs_type, obs_list in self._drone.items():
            if not obs_list:
                continue
            ages = [self._current_time - obs.timestamp for obs in obs_list]
            summary[obs_type] = {
                "count":      len(ages),
                "min_age_s":  min(ages),
                "max_age_s":  max(ages),
                "mean_age_s": sum(ages) / len(ages),
            }
        return summary

    def snapshot(self) -> ObservationSnapshot:
        """
        Immutable snapshot of current state.
        Used for counterfactual evaluation (fork without affecting live store).
        """
        return ObservationSnapshot(
            raws={sid: dict(obs) for sid, obs in self._raws.items()},
            drone={ot: list(obs) for ot, obs in self._drone.items()},
            satellite={ot: list(obs) for ot, obs in self._satellite.items()},
            fire_detections=list(self._fire_detections),
            current_time=self._current_time,
            decay_config=dict(self._decay_config),
        )

    def __deepcopy__(self, memo: dict) -> "ObservationStore":
        """
        Custom deepcopy that creates a fresh RLock instead of copying the
        existing one (threading.RLock objects are not copyable).
        All observation data is deep-copied normally.
        """
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

    def fork(self) -> ObservationStore:
        """
        Create a mutable deep copy of this store.
        Used by LiveEstimator to build a temporary working copy without affecting
        the live store.
        """
        new = ObservationStore(dict(self._decay_config))
        new._current_time   = self._current_time
        new._raws           = {sid: dict(obs) for sid, obs in self._raws.items()}
        new._drone          = defaultdict(list,
                               {ot: list(obs) for ot, obs in self._drone.items()})
        new._satellite      = defaultdict(list,
                               {ot: list(obs) for ot, obs in self._satellite.items()})
        new._fire_detections = list(self._fire_detections)
        return new

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def total_ingested(self) -> int:
        return self._total_ingested

    @property
    def total_pruned(self) -> int:
        return self._total_pruned


# ---------------------------------------------------------------------------
# Immutable snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservationSnapshot:
    """Immutable snapshot for counterfactual evaluation."""
    raws:            dict
    drone:           dict
    satellite:       dict
    fire_detections: list
    current_time:    float
    decay_config:    dict


# ---------------------------------------------------------------------------
# Ingestion buffer
# ---------------------------------------------------------------------------

class IngestionBuffer:
    """
    Buffers observations while the store is locked during a cycle.
    External sources (drone telemetry, satellite) push here; flush() after
    the cycle's unlock_cycle() delivers them to the store.
    """

    def __init__(self, store: ObservationStore) -> None:
        self._store   = store
        self._pending: list[Observation] = []

    def add(self, obs: Observation) -> None:
        """Always succeeds — buffers if store is locked."""
        try:
            if isinstance(obs, RAWSObservation):
                self._store.add_raws(obs.source_id, [obs])
            elif isinstance(obs, FireDetectionObservation):
                self._store.add_fire_detections([obs])
            elif isinstance(obs, SatelliteObservation):
                self._store.add_satellite_observations([obs])
            else:
                self._store.add_drone_observations([obs])
        except RuntimeError:
            self._pending.append(obs)

    def flush(self) -> None:
        """Deliver all buffered observations to the store. Call after unlock_cycle()."""
        pending = list(self._pending)
        self._pending.clear()
        for obs in pending:
            self.add(obs)
