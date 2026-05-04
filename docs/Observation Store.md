# Observation Store — Revised Design Specification

---

## Core Design Principles

1. **Store physical observations, serve mathematical data points.** A drone sortie is one observation containing FMC + wind + thermal readings. The GP needs separate lists of (location, value, sigma) per variable type. The observation class handles the conversion internally.
    
2. **Time-parameterized queries, not mutable state.** The store doesn't decay observations in place. Callers pass a query time; the store returns observations with sigmas adjusted for that time. The store is append-only (plus pruning). This enables historical queries and makes the store thread-safe for reads.
    
3. **Each observation type controls its own post-processing.** RAWS observations never decay. Drone FMC observations decay with tau=1hr. Fire detections never expire. This logic lives in the observation class via overridable methods, not in the store.
    
4. **Prune for efficiency, not correctness.** Pruning removes observations that are so decayed they contribute nothing. It's an optimization, not a semantic operation. The system produces identical results with or without pruning — pruning just avoids iterating over dead observations.
    

---

## Observation Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class VariableType(Enum):
    FMC = "fmc"
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    FIRE_DETECTION = "fire_detection"

@dataclass(frozen=True)
class DataPoint:
    """
    A single (location, value, sigma) tuple consumed by the GP/EnKF.
    This is what mathematical consumers see. They never see the
    observation object that produced it.
    """
    location: tuple[int, int]
    variable: VariableType
    value: float
    sigma: float              # effective sigma (after decay)
    timestamp: float          # when the measurement was taken

class Observation(ABC):
    """
    Interface for all observation types.
    
    An observation represents a physical measurement event.
    It may contain multiple variable types (a drone measures FMC + wind
    simultaneously). It knows how to decay itself and how to decide
    when it's expired.
    
    Subclasses MUST implement:
      - to_data_points(query_time) -> list[DataPoint]
      - is_expired(query_time) -> bool
    """
    
    @abstractmethod
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        """
        Convert this observation to mathematical data points at query_time.
        
        Each returned DataPoint has sigma adjusted for temporal decay
        relative to query_time. A fresh observation returns original sigma.
        An old observation returns inflated sigma.
        
        May return multiple DataPoints (one per variable measured).
        """
        ...
    
    @abstractmethod
    def is_expired(self, query_time: float) -> bool:
        """
        Should this observation be pruned at query_time?
        
        Returns True if the observation has decayed beyond usefulness.
        RAWS observations return False (never expire — replaced instead).
        Fire detections return False (permanent).
        Drone observations return True when sigma has inflated >10× original.
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
```

---

## Observation Implementations

### RAWS Observation

```python
@dataclass(frozen=True)
class RAWSObservation(Observation):
    """
    Fixed ground station reading. Contains FMC + wind speed + wind direction.
    Never decays. Replaced entirely when a new reading arrives from the
    same station.
    """
    _source_id: str              # e.g., "RAWS_CEDU"
    _timestamp: float
    location: tuple[int, int]
    fmc: float
    fmc_sigma: float
    wind_speed: float
    wind_speed_sigma: float
    wind_direction: float
    wind_direction_sigma: float
    
    @property
    def source_id(self): return self._source_id
    
    @property
    def timestamp(self): return self._timestamp
    
    @property
    def variables(self):
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
```

### Drone Observation

```python
# Decay constants (seconds)
TAU = {
    VariableType.FMC: 3600.0,
    VariableType.WIND_SPEED: 7200.0,
    VariableType.WIND_DIRECTION: 3600.0,
    VariableType.TEMPERATURE: 3600.0,
    VariableType.HUMIDITY: 3600.0,
}
EXPIRY_FACTOR = 10.0  # expire when sigma inflates >10× original

@dataclass(frozen=True)
class DroneObservation(Observation):
    """
    Single measurement from a drone at one location.
    Contains whichever variables the drone's sensors captured.
    Decays over time — sigma inflates exponentially with age.
    """
    _source_id: str              # e.g., "drone_03"
    _timestamp: float
    location: tuple[int, int]
    
    # Optional per variable — None means this drone didn't measure it here
    fmc: float | None = None
    fmc_sigma: float | None = None
    wind_speed: float | None = None
    wind_speed_sigma: float | None = None
    wind_direction: float | None = None
    wind_direction_sigma: float | None = None
    
    @property
    def source_id(self): return self._source_id
    
    @property
    def timestamp(self): return self._timestamp
    
    @property
    def variables(self):
        v = []
        if self.fmc is not None: v.append(VariableType.FMC)
        if self.wind_speed is not None: v.append(VariableType.WIND_SPEED)
        if self.wind_direction is not None: v.append(VariableType.WIND_DIRECTION)
        return v
    
    def _effective_sigma(self, original_sigma: float, tau: float,
                          query_time: float) -> float:
        age = max(0.0, query_time - self._timestamp)
        return original_sigma * np.exp(age / tau)
    
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        points = []
        if self.fmc is not None:
            sigma = self._effective_sigma(
                self.fmc_sigma, TAU[VariableType.FMC], query_time)
            points.append(DataPoint(
                self.location, VariableType.FMC,
                self.fmc, sigma, self._timestamp))
        
        if self.wind_speed is not None:
            sigma = self._effective_sigma(
                self.wind_speed_sigma, TAU[VariableType.WIND_SPEED], query_time)
            points.append(DataPoint(
                self.location, VariableType.WIND_SPEED,
                self.wind_speed, sigma, self._timestamp))
        
        if self.wind_direction is not None:
            sigma = self._effective_sigma(
                self.wind_direction_sigma, TAU[VariableType.WIND_DIRECTION], query_time)
            points.append(DataPoint(
                self.location, VariableType.WIND_DIRECTION,
                self.wind_direction, sigma, self._timestamp))
        
        return points
    
    def is_expired(self, query_time: float) -> bool:
        # Expired if ALL contained variables have decayed beyond threshold
        for var in self.variables:
            sigma_orig = getattr(self, f"{var.value}_sigma")
            tau = TAU[var]
            sigma_eff = self._effective_sigma(sigma_orig, tau, query_time)
            if sigma_eff <= EXPIRY_FACTOR * sigma_orig:
                return False  # at least one variable is still informative
        return True  # everything has decayed
```

### Fire Detection Observation

```python
@dataclass(frozen=True)
class FireDetectionObservation(Observation):
    """
    Binary fire/no-fire observation from thermal camera or satellite.
    Never decays — a cell that was burning at time T was burning at time T forever.
    Consumed by the particle filter, not the GP.
    """
    _source_id: str              # e.g., "VIIRS_pass_042", "drone_03_thermal"
    _timestamp: float
    location: tuple[int, int]
    is_fire: bool
    confidence: float            # P(observation is correct), typically 0.8-0.99
    
    @property
    def source_id(self): return self._source_id
    
    @property
    def timestamp(self): return self._timestamp
    
    @property
    def variables(self): return [VariableType.FIRE_DETECTION]
    
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        # Fire detection as a data point:
        # value = 1.0 (fire) or 0.0 (no fire)
        # sigma = 1 - confidence (lower confidence = higher noise)
        return [DataPoint(
            self.location, VariableType.FIRE_DETECTION,
            1.0 if self.is_fire else 0.0,
            1.0 - self.confidence,
            self._timestamp
        )]
    
    def is_expired(self, query_time: float) -> bool:
        return False  # permanent
```

### Satellite FMC Observation

```python
@dataclass(frozen=True)
class SatelliteFMCObservation(Observation):
    """
    Coarse FMC estimate from satellite (MODIS/VIIRS).
    Covers a footprint of multiple cells. Decays like drone observations.
    """
    _source_id: str
    _timestamp: float
    center_location: tuple[int, int]
    footprint_cells: tuple[tuple[int, int], ...]
    fmc: float
    fmc_sigma: float              # higher than drone — coarser measurement
    
    @property
    def source_id(self): return self._source_id
    
    @property
    def timestamp(self): return self._timestamp
    
    @property
    def location(self): return self.center_location
    
    @property
    def variables(self): return [VariableType.FMC]
    
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        # Return one data point per footprint cell
        # All share the same value (satellite pixel average)
        # but with inflated sigma reflecting the coarseness
        age = max(0.0, query_time - self._timestamp)
        sigma = self.fmc_sigma * np.exp(age / TAU[VariableType.FMC])
        
        return [DataPoint(cell, VariableType.FMC, self.fmc, sigma, self._timestamp)
                for cell in self.footprint_cells]
    
    def is_expired(self, query_time: float) -> bool:
        age = max(0.0, query_time - self._timestamp)
        return self.fmc_sigma * np.exp(age / TAU[VariableType.FMC]) > EXPIRY_FACTOR * self.fmc_sigma
```

---

## Observation Store

```python
import threading
from typing import Optional

class ObservationStore:
    """
    Central repository for all observations.
    
    Append-only (plus pruning). Queries are time-parameterized —
    callers pass a query time and receive data points with
    appropriately decayed sigmas. The store never mutates
    observation objects.
    
    Thread-safe for concurrent reads. Locked during cycle computation
    to prevent observation ingestion mid-cycle.
    """
    
    def __init__(self):
        # RAWS: keyed by source_id. New readings replace old.
        self._raws: dict[str, RAWSObservation] = {}
        
        # All other observations: append-only list.
        # Pruning removes expired entries periodically.
        self._observations: list[Observation] = []
        
        # Lock for cycle computation
        self._lock = threading.RLock()
        self._locked = False
        
        # Pruning cache: last time we pruned and the timestamp used
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
    
    def get_data_points(self, query_time: float,
                         variable: Optional[VariableType] = None,
                         min_spacing_cells: Optional[int] = None
                         ) -> list[DataPoint]:
        """
        Primary query method. Returns processed data points with
        decayed sigmas appropriate for query_time.
        
        All observation types are flattened into DataPoints.
        Optionally filter by variable type.
        Optionally thin to one point per min_spacing_cells.
        
        This is what the GP calls.
        """
        points = []
        
        # RAWS data points (no decay)
        for raws in self._raws.values():
            points.extend(raws.to_data_points(query_time))
        
        # All other observations (decayed)
        for obs in self._observations:
            if obs.is_expired(query_time):
                continue  # skip without removing — pruning handles removal
            points.extend(obs.to_data_points(query_time))
        
        # Filter by variable type
        if variable is not None:
            points = [p for p in points if p.variable == variable]
        
        # Thin spatially if requested
        if min_spacing_cells is not None:
            points = self._thin(points, min_spacing_cells)
        
        return points
    
    def get_fire_detections(self, since: Optional[float] = None
                            ) -> list[FireDetectionObservation]:
        """
        Return fire detection observations for the particle filter.
        These are returned as full observation objects (not DataPoints)
        because the particle filter needs the is_fire and confidence fields.
        """
        detections = [obs for obs in self._observations
                      if isinstance(obs, FireDetectionObservation)]
        if since is not None:
            detections = [d for d in detections if d.timestamp >= since]
        return detections
    
    def get_raw_observations(self, variable: Optional[VariableType] = None
                              ) -> list[Observation]:
        """
        Return raw observation objects (undecayed).
        For diagnostics, visualization, and debugging.
        """
        result = list(self._raws.values())
        result.extend(self._observations)
        if variable is not None:
            result = [o for o in result if variable in o.variables]
        return result
    
    # ------------------------------------------------------------------
    # Thinning
    # ------------------------------------------------------------------
    
    def _thin(self, points: list[DataPoint], 
              min_spacing: int) -> list[DataPoint]:
        """
        Keep lowest-sigma point per spatial neighborhood.
        Operates on DataPoints (already processed), not observations.
        """
        # Sort by sigma ascending — keep most precise first
        sorted_points = sorted(points, key=lambda p: p.sigma)
        
        thinned = []
        for point in sorted_points:
            if all(
                abs(point.location[0] - t.location[0]) > min_spacing or
                abs(point.location[1] - t.location[1]) > min_spacing
                for t in thinned
            ):
                thinned.append(point)
        
        return thinned
    
    # ------------------------------------------------------------------
    # Pruning — removes expired observations for efficiency
    # ------------------------------------------------------------------
    
    def prune(self, current_time: float) -> int:
        """
        Remove observations expired as of current_time.
        
        Uses cache: if current_time > last prune time, only checks
        observations that weren't pruned last time. Observations
        are ordered by insertion time, and expiry is monotonic in
        query time, so we can skip observations we've already
        determined are still alive at an earlier time.
        
        Returns: number of observations pruned.
        """
        if current_time <= self._last_prune_time:
            return 0  # nothing new to prune
        
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
    
    def _check_not_locked(self):
        if self._locked:
            raise RuntimeError(
                "Cannot add observations during cycle computation. "
                "Buffer externally and add after unlock().")
    
    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    
    def count(self) -> dict:
        """Observation counts by type."""
        counts = {"raws": len(self._raws), "other": 0}
        by_type = {}
        for obs in self._observations:
            name = type(obs).__name__
            by_type[name] = by_type.get(name, 0) + 1
        counts.update(by_type)
        counts["total"] = counts["raws"] + len(self._observations)
        return counts
```

---

## How the GP Consumes This

The GP no longer stores observations internally. It calls `store.get_data_points()` at fit time:

```python
class IGNISGPPrior:
    def __init__(self, obs_store: ObservationStore, terrain, ...):
        self._store = obs_store
        # No internal observation lists. The store owns all observations.
    
    def fit(self, current_time: float):
        """Fit GPs from observation store data."""
        
        # FMC: get all FMC data points with decay applied,
        # thinned to one per correlation-length spacing
        fmc_points = self._store.get_data_points(
            query_time=current_time,
            variable=VariableType.FMC,
            min_spacing_cells=int(self.correlation_length / self.resolution_m)
        )
        
        # These are DataPoints — just (location, value, sigma)
        fmc_locs = [p.location for p in fmc_points]
        fmc_vals = [p.value for p in fmc_points]
        fmc_sigmas = [p.sigma for p in fmc_points]
        
        # Subtract Nelson mean (residual fitting)
        if self._nelson_mean is not None:
            fmc_vals = [v - self._nelson_mean[r, c] 
                        for v, (r, c) in zip(fmc_vals, fmc_locs)]
        
        # Fit GP
        self._gp_fmc = self._fit_variable(fmc_locs, fmc_vals, fmc_sigmas, ...)
        
        # Same for wind speed and direction
        ws_points = self._store.get_data_points(
            current_time, VariableType.WIND_SPEED,
            min_spacing_cells=int(self.wind_corr_length / self.resolution_m))
        # ... fit wind GP ...
        
        wd_points = self._store.get_data_points(
            current_time, VariableType.WIND_DIRECTION,
            min_spacing_cells=int(self.wind_corr_length / self.resolution_m))
        # ... fit wind direction GP ...
```

The GP doesn't know whether a data point came from RAWS, a drone, or a satellite. It sees (location, value, sigma). A RAWS FMC reading with sigma=0.03 and a drone FMC reading from 2 hours ago with sigma=0.08 (decayed from 0.03) are just two data points with different precisions. The GP weights them accordingly.

---

## Orchestrator Integration

```python
class Orchestrator:
    def run_cycle(self):
        self.obs_store.lock()
        try:
            # Prune expired observations
            self.obs_store.prune(self.current_time)
            
            # GP reads from store — gets decayed, thinned DataPoints
            self.gp.fit(self.current_time)
            gp_prior = self.gp.predict(self.shape)
            
            # Fire detections for particle filter (full objects, not DataPoints)
            fire_obs = self.obs_store.get_fire_detections(
                since=self.last_cycle_time)
            
            # ... rest of cycle (ensemble, info field, selection) ...
            
        finally:
            self.obs_store.unlock()
```

---

## Ingestion Buffer

```python
class IngestionBuffer:
    """
    Buffers observations while the store is locked during cycle computation.
    External sources (drone telemetry, satellite passes) push here.
    Flushed to store after each cycle.
    """
    def __init__(self, store: ObservationStore):
        self._store = store
        self._pending: list[Observation] = []
        self._pending_raws: list[RAWSObservation] = []
    
    def add(self, obs: Observation):
        if isinstance(obs, RAWSObservation):
            self._pending_raws.append(obs)
        else:
            self._pending.append(obs)
    
    def flush(self):
        """Push all buffered observations to store. Call after unlock()."""
        for raws in self._pending_raws:
            self._store.add_raws(raws)
        if self._pending:
            self._store.add_batch(self._pending)
        self._pending.clear()
        self._pending_raws.clear()
```

---

## What This Clarifies vs the Previous Spec

|Issue|Previous spec|This spec|
|---|---|---|
|Where decay logic lives|In the store's `_prune_and_decay()`|In each observation's `to_data_points()`|
|Where thinning happens|Separate function, unclear when called|Inside `get_data_points()` via optional parameter|
|How multi-variable observations work|Separate lists per variable|One observation object produces multiple DataPoints|
|What the GP sees|Lists of locs/vals/sigmas managed by GP|`list[DataPoint]` from store — GP has zero storage responsibility|
|RAWS handling|Special parallel lists in GP|Keyed dict in store, standard `Observation` interface|
|Fire detection handling|Stored but unused|Stored, returned as full objects for particle filter via `get_fire_detections()`|
|Historical queries|Not possible (observations mutated by decay)|Time-parameterized queries — pass any time, get appropriate decay|
|Thread safety|Lock on store|Lock on store + IngestionBuffer for concurrent drone telemetry|