# IGNIS: Observation Types Specification

---

## Submodule Structure

```
ignis/
└── observations/
    ├── __init__.py              # exports ObservationStore, all types
    ├── interface.py             # Observation ABC, DataPoint, VariableType, VARIABLE_TAU
    ├── store.py                 # ObservationStore, IngestionBuffer
    ├── types/
    │   ├── __init__.py          # re-exports all observation classes
    │   ├── raws.py              # RAWSObservation
    │   ├── drone.py             # DroneWindObs, DroneFMCObs, DroneThermalObs,
    │   │                        # DroneLiDARObs, DroneGasObs
    │   └── satellite.py         # GOESFireDetection, VIIRSFireDetection,
    │                            # SatelliteFMCObservation, HRRRWindObservation
    └── utils.py                 # spatial thinning, coordinate helpers
```

---

## Core Interface (interface.py)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

class VariableType(Enum):
    """Variables the GP and ensemble consume."""
    FMC = "fmc"
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    FIRE_DETECTION = "fire_detection"

# Decay timescale by VARIABLE TYPE, not by sensor.
# FMC staleness is a property of FMC, not of what measured it.
VARIABLE_TAU: dict[VariableType, float] = {
    VariableType.FMC: 3600.0,             # 1 hour (1-hr fuel timelag)
    VariableType.WIND_SPEED: 7200.0,      # 2 hours (mean flow persistence)
    VariableType.WIND_DIRECTION: 3600.0,  # 1 hour (direction drifts faster)
    VariableType.TEMPERATURE: 7200.0,     # 2 hours
    VariableType.HUMIDITY: 3600.0,        # 1 hour
    VariableType.FIRE_DETECTION: None,    # permanent — fire doesn't un-burn
}

EXPIRY_FACTOR: float = 10.0  # prune when sigma inflated >10× original

@dataclass(frozen=True)
class DataPoint:
    """
    A single (location, variable, value, sigma) tuple.
    
    This is what mathematical consumers (GP, EnKF) see.
    They never see the Observation object that produced it.
    All processing (decay, footprint expansion, etc.) is done
    before the DataPoint is created.
    """
    location: tuple[int, int]     # (row, col) in grid coordinates
    variable: VariableType
    value: float                  # measured value in SI units
    sigma: float                  # effective sigma (after decay, after repr. error)
    timestamp: float              # when the measurement was taken (sim seconds)

class Observation(ABC):
    """
    Interface for all observation types.
    
    An observation represents a physical measurement event — one sensor
    reading at one time. It may contain multiple variable types (a drone
    measures FMC + wind simultaneously). It knows how to:
    
    1. Convert itself to DataPoints with appropriate decay and footprint
    2. Decide when it's expired (too stale to contribute)
    
    Subclasses MUST implement:
      to_data_points(query_time) -> list[DataPoint]
      is_expired(query_time) -> bool
      source_id -> str
      timestamp -> float
      variables -> list[VariableType]
    """
    
    def _decay_sigma(self, original_sigma: float,
                      variable: VariableType,
                      query_time: float) -> float:
        """Inflate sigma by temporal decay. Shared by all subclasses."""
        tau = VARIABLE_TAU.get(variable)
        if tau is None:
            return original_sigma  # non-decaying variable
        age = max(0.0, query_time - self.timestamp)
        return original_sigma * np.exp(age / tau)
    
    def _is_variable_expired(self, original_sigma: float,
                              variable: VariableType,
                              query_time: float) -> bool:
        """Check if a single variable has decayed beyond usefulness."""
        tau = VARIABLE_TAU.get(variable)
        if tau is None:
            return False  # non-decaying
        age = max(0.0, query_time - self.timestamp)
        return np.exp(age / tau) > EXPIRY_FACTOR
    
    @abstractmethod
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        """
        Convert to mathematical data points at query_time.
        Sigmas are adjusted for temporal decay.
        May return multiple DataPoints (per variable, per footprint cell).
        """
        ...
    
    @abstractmethod
    def is_expired(self, query_time: float) -> bool:
        """Should this observation be pruned at query_time?"""
        ...
    
    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique source identifier (station ID, drone ID, satellite pass)."""
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
    
    @property
    @abstractmethod
    def location(self) -> tuple[int, int]:
        """Primary location (center of footprint for multi-cell obs)."""
        ...
```

---

## RAWS Observations (types/raws.py)

```python
@dataclass(frozen=True)
class RAWSObservation(Observation):
    """
    Fixed ground weather station reading.
    
    Contains FMC (derived from T/RH via Nelson at the station) +
    wind speed + wind direction + temperature + humidity.
    
    Never decays — each call to add_raws() replaces previous reading
    for that station. Identified by station ID matching real RAWS
    identifiers (e.g., "CEDU" for Cerro Negro, "PTGU" for Point Mugu).
    
    Point measurement at the station's grid cell.
    """
    _source_id: str                # real RAWS station ID, e.g., "RAWS_CEDU"
    _timestamp: float              # simulation time of this reading
    _location: tuple[int, int]     # station's grid cell
    
    # All variables measured simultaneously
    fmc: float                     # fraction, derived from T/RH via Nelson at station
    fmc_sigma: float               # ~0.03 (well-maintained station)
    wind_speed: float              # m/s at 10m (standard RAWS height)
    wind_speed_sigma: float        # ~1.0 m/s
    wind_direction: float          # degrees
    wind_direction_sigma: float    # ~10 degrees
    temperature: float             # °C
    temperature_sigma: float       # ~0.5 °C
    humidity: float                # fraction 0-1
    humidity_sigma: float          # ~0.03
    
    @property
    def source_id(self): return self._source_id
    @property
    def timestamp(self): return self._timestamp
    @property
    def location(self): return self._location
    
    @property
    def variables(self):
        return [VariableType.FMC, VariableType.WIND_SPEED,
                VariableType.WIND_DIRECTION, VariableType.TEMPERATURE,
                VariableType.HUMIDITY]
    
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        # No decay — RAWS readings are always treated as current
        return [
            DataPoint(self._location, VariableType.FMC,
                      self.fmc, self.fmc_sigma, self._timestamp),
            DataPoint(self._location, VariableType.WIND_SPEED,
                      self.wind_speed, self.wind_speed_sigma, self._timestamp),
            DataPoint(self._location, VariableType.WIND_DIRECTION,
                      self.wind_direction, self.wind_direction_sigma, self._timestamp),
            DataPoint(self._location, VariableType.TEMPERATURE,
                      self.temperature, self.temperature_sigma, self._timestamp),
            DataPoint(self._location, VariableType.HUMIDITY,
                      self.humidity, self.humidity_sigma, self._timestamp),
        ]
    
    def is_expired(self, query_time: float) -> bool:
        return False  # never expires — replaced by fresh readings
```

---

## Drone Observations (types/drone.py)

### DroneWindObservation

```python
@dataclass(frozen=True)
class DroneWindObservation(Observation):
    """
    Wind measurement from drone-mounted gyroscope/accelerometer or anemometer.
    
    Point measurement at the drone's position. Best obtained by hovering
    for 10-30 seconds to get a stable temporal average.
    
    Accuracy: ~0.22 m/s speed error, <7° direction error.
    """
    _source_id: str                # e.g., "drone_03"
    _timestamp: float
    _location: tuple[int, int]
    
    wind_speed: float              # m/s
    wind_speed_sigma: float        # ~0.22 m/s (gyro/accel), ~0.3 m/s (cup anemometer)
    wind_direction: float          # degrees
    wind_direction_sigma: float    # ~7 degrees
    
    @property
    def source_id(self): return self._source_id
    @property
    def timestamp(self): return self._timestamp
    @property
    def location(self): return self._location
    
    @property
    def variables(self):
        return [VariableType.WIND_SPEED, VariableType.WIND_DIRECTION]
    
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        return [
            DataPoint(
                self._location, VariableType.WIND_SPEED,
                self.wind_speed,
                self._decay_sigma(self.wind_speed_sigma, VariableType.WIND_SPEED, query_time),
                self._timestamp),
            DataPoint(
                self._location, VariableType.WIND_DIRECTION,
                self.wind_direction,
                self._decay_sigma(self.wind_direction_sigma, VariableType.WIND_DIRECTION, query_time),
                self._timestamp),
        ]
    
    def is_expired(self, query_time: float) -> bool:
        # Expired when BOTH speed and direction have decayed past threshold
        ws_expired = self._is_variable_expired(
            self.wind_speed_sigma, VariableType.WIND_SPEED, query_time)
        wd_expired = self._is_variable_expired(
            self.wind_direction_sigma, VariableType.WIND_DIRECTION, query_time)
        return ws_expired and wd_expired
```

### DroneFMCObservation

```python
@dataclass(frozen=True)
class DroneFMCObservation(Observation):
    """
    Fuel moisture from multispectral/IR camera + reflectance-to-FMC model.
    
    Covers a footprint — the camera FOV observes multiple grid cells.
    At 50m altitude over a 50m grid, footprint is roughly 2-3 cells radius.
    
    Accuracy: ~5% RMSE. Dependent on fuel type calibration.
    """
    _source_id: str
    _timestamp: float
    _location: tuple[int, int]     # center of footprint
    
    fmc: float                     # fraction
    fmc_sigma: float               # ~0.05 (5% RMSE)
    footprint_radius_cells: int    # camera FOV radius in grid cells (~2-3)
    
    @property
    def source_id(self): return self._source_id
    @property
    def timestamp(self): return self._timestamp
    @property
    def location(self): return self._location
    
    @property
    def variables(self):
        return [VariableType.FMC]
    
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        base_sigma = self._decay_sigma(self.fmc_sigma, VariableType.FMC, query_time)
        
        points = []
        r0, c0 = self._location
        for dr in range(-self.footprint_radius_cells,
                         self.footprint_radius_cells + 1):
            for dc in range(-self.footprint_radius_cells,
                             self.footprint_radius_cells + 1):
                dist = np.sqrt(dr**2 + dc**2)
                if dist > self.footprint_radius_cells:
                    continue
                
                # Sigma increases toward edge of FOV
                # (optical quality degrades, parallax, etc.)
                edge_factor = 1.0 + 0.5 * (dist / (self.footprint_radius_cells + 1e-6))
                
                points.append(DataPoint(
                    location=(r0 + dr, c0 + dc),
                    variable=VariableType.FMC,
                    value=self.fmc,
                    sigma=base_sigma * edge_factor,
                    timestamp=self._timestamp
                ))
        return points
    
    def is_expired(self, query_time: float) -> bool:
        return self._is_variable_expired(
            self.fmc_sigma, VariableType.FMC, query_time)
```

### DroneThermalObservation

```python
@dataclass(frozen=True)
class DroneThermalObservation(Observation):
    """
    Fire detection from IR/thermal camera.
    
    The thermal camera observes a swath of cells each frame. Each cell
    in the swath gets a binary fire/no-fire classification with a
    confidence score.
    
    Also used for: spot fire / ember detection, search & rescue
    heat signatures (not modeled here).
    
    Never decays — fire state is permanent.
    """
    _source_id: str
    _timestamp: float
    
    # Each detection: (location, is_fire, confidence)
    # A single frame may contain hundreds of cell detections
    detections: tuple[tuple[tuple[int, int], bool, float], ...]
    
    @property
    def source_id(self): return self._source_id
    @property
    def timestamp(self): return self._timestamp
    @property
    def location(self): return self.detections[0][0] if self.detections else (0, 0)
    
    @property
    def variables(self):
        return [VariableType.FIRE_DETECTION]
    
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        return [
            DataPoint(
                location=loc,
                variable=VariableType.FIRE_DETECTION,
                value=1.0 if is_fire else 0.0,
                sigma=1.0 - confidence,  # high confidence = low sigma
                timestamp=self._timestamp
            )
            for loc, is_fire, confidence in self.detections
        ]
    
    def is_expired(self, query_time: float) -> bool:
        return False  # fire state is permanent


@dataclass(frozen=True)
class DroneWeatherObservation(Observation):
    """
    Onboard temperature + humidity sensor reading.
    
    Every drone carries these for flight controller telemetry.
    Used to compute local dead FMC via Nelson model — the drone
    doesn't measure DFMC directly, but measures the weather
    that determines DFMC at that specific microclimate.
    
    Point measurement at drone position.
    """
    _source_id: str
    _timestamp: float
    _location: tuple[int, int]
    
    temperature: float             # °C
    temperature_sigma: float       # ~0.5 °C
    humidity: float                # fraction 0-1
    humidity_sigma: float          # ~0.03
    
    @property
    def source_id(self): return self._source_id
    @property
    def timestamp(self): return self._timestamp
    @property
    def location(self): return self._location
    
    @property
    def variables(self):
        return [VariableType.TEMPERATURE, VariableType.HUMIDITY]
    
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        return [
            DataPoint(
                self._location, VariableType.TEMPERATURE,
                self.temperature,
                self._decay_sigma(self.temperature_sigma, VariableType.TEMPERATURE, query_time),
                self._timestamp),
            DataPoint(
                self._location, VariableType.HUMIDITY,
                self.humidity,
                self._decay_sigma(self.humidity_sigma, VariableType.HUMIDITY, query_time),
                self._timestamp),
        ]
    
    def is_expired(self, query_time: float) -> bool:
        t_expired = self._is_variable_expired(
            self.temperature_sigma, VariableType.TEMPERATURE, query_time)
        h_expired = self._is_variable_expired(
            self.humidity_sigma, VariableType.HUMIDITY, query_time)
        return t_expired and h_expired
```

### DroneLiDARObs

```python
@dataclass(frozen=True)
class DroneLiDARObservation(Observation):
    """
    3D fuel structure from LiDAR scan.
    
    Provides sub-canopy information satellites cannot see:
    canopy height, canopy base height, fuel density.
    
    This does NOT feed the GP — it updates static terrain layers
    (specifically CBH and CBD which affect crown fire threshold).
    The observation store stores it; the orchestrator routes it
    to PriorState for terrain layer updates.
    
    Never decays — fuel structure doesn't change over hours
    (unless the fire burns through it, handled separately).
    """
    _source_id: str
    _timestamp: float
    _location: tuple[int, int]
    
    canopy_height: float           # meters
    canopy_base_height: float      # meters — critical for Van Wagner
    canopy_bulk_density: float     # kg/m³
    fuel_density: float            # kg/m² surface fuel
    confidence: float              # scan quality
    scan_radius_cells: int         # LiDAR footprint (~1-2 cells)
    
    @property
    def source_id(self): return self._source_id
    @property
    def timestamp(self): return self._timestamp
    @property
    def location(self): return self._location
    
    @property
    def variables(self):
        return []  # does not produce standard DataPoints for the GP
    
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        return []  # consumed by PriorState terrain update, not GP
    
    def is_expired(self, query_time: float) -> bool:
        return False  # fuel structure is static until fire burns it
```

### DroneGasObservation

```python
@dataclass(frozen=True)
class DroneGasObservation(Observation):
    """
    Gas/particulate sensors: CO, PM2.5, PM10.
    
    Used for: early fire detection (smoke before visible fire),
    air quality assessment, smoke plume boundary detection.
    
    Not consumed by current GP or fire model — stored for
    situational awareness and potential future smoke transport modeling.
    
    Decays very fast — atmospheric mixing disperses smoke quickly.
    """
    _source_id: str
    _timestamp: float
    _location: tuple[int, int]
    
    pm25: float | None             # µg/m³
    pm10: float | None             # µg/m³
    co_ppm: float | None           # ppm
    
    GAS_EXPIRY_SECONDS: float = 900.0  # 15 minutes — smoke disperses fast
    
    @property
    def source_id(self): return self._source_id
    @property
    def timestamp(self): return self._timestamp
    @property
    def location(self): return self._location
    
    @property
    def variables(self):
        return []  # not consumed by current model
    
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        return []  # stored for future use / visualization only
    
    def is_expired(self, query_time: float) -> bool:
        return (query_time - self._timestamp) > self.GAS_EXPIRY_SECONDS
```

---

## Satellite Observations (types/satellite.py)

### GOESFireDetection

```python
@dataclass(frozen=True)
class GOESFireDetection(Observation):
    """
    GOES-16/17/18 geostationary fire detection.
    
    Frequency: every 1-5 minutes.
    Resolution: ~2 km per pixel.
    Strength: near-continuous temporal coverage.
    Weakness: coarse spatial resolution — 2km pixel = ~40×40 grid cells at 50m.
    
    Can detect fires 10-15 minutes before ground-based notification in
    clear conditions. Degrades under heavy smoke or cloud cover.
    
    Emits a thinned grid of DataPoints across the pixel footprint.
    Confidence decreases toward pixel edges (geolocation uncertainty).
    """
    _source_id: str                # e.g., "GOES16_20250107_1830"
    _timestamp: float
    _location: tuple[int, int]     # pixel center in grid coords
    
    is_fire: bool
    confidence: float              # 0.0-1.0
    frp: float | None              # fire radiative power (kW), if available
    pixel_radius_cells: int = 20   # 2000m / 50m / 2 = 20
    
    THIN_STEP_CELLS: int = 10      # one DataPoint per ~500m within pixel
    
    @property
    def source_id(self): return self._source_id
    @property
    def timestamp(self): return self._timestamp
    @property
    def location(self): return self._location
    
    @property
    def variables(self):
        return [VariableType.FIRE_DETECTION]
    
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        points = []
        r0, c0 = self._location
        for dr in range(-self.pixel_radius_cells,
                         self.pixel_radius_cells + 1, self.THIN_STEP_CELLS):
            for dc in range(-self.pixel_radius_cells,
                             self.pixel_radius_cells + 1, self.THIN_STEP_CELLS):
                dist = np.sqrt(dr**2 + dc**2)
                if dist > self.pixel_radius_cells:
                    continue
                
                # Confidence degrades toward pixel edge
                edge_fraction = dist / (self.pixel_radius_cells + 1e-6)
                local_confidence = self.confidence * max(0.5, 1.0 - 0.5 * edge_fraction)
                
                points.append(DataPoint(
                    location=(r0 + dr, c0 + dc),
                    variable=VariableType.FIRE_DETECTION,
                    value=1.0 if self.is_fire else 0.0,
                    sigma=1.0 - local_confidence,
                    timestamp=self._timestamp
                ))
        return points
    
    def is_expired(self, query_time: float) -> bool:
        return False  # fire state is permanent


@dataclass(frozen=True)
class VIIRSFireDetection(Observation):
    """
    VIIRS (Suomi-NPP, NOAA-20) polar orbiter fire detection.
    
    Frequency: every ~12 hours.
    Resolution: 375 m per pixel.
    Strength: 3× more detailed than MODIS, detects smaller fires.
    Weakness: infrequent — only 2 passes per day.
    
    Used for: consistency check (triggers hard reset if ensemble
    disagrees significantly), fire perimeter reconstruction.
    
    Small enough footprint that 1-5 DataPoints suffice.
    """
    _source_id: str                # e.g., "VIIRS_SNPP_20250107_2045"
    _timestamp: float
    _location: tuple[int, int]
    
    is_fire: bool
    confidence: float              # "low"→0.5, "nominal"→0.8, "high"→0.95
    frp: float | None
    pixel_radius_cells: int = 4    # 375m / 50m / 2 ≈ 4
    
    @property
    def source_id(self): return self._source_id
    @property
    def timestamp(self): return self._timestamp
    @property
    def location(self): return self._location
    
    @property
    def variables(self):
        return [VariableType.FIRE_DETECTION]
    
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        points = [
            # Center at full confidence
            DataPoint(self._location, VariableType.FIRE_DETECTION,
                      1.0 if self.is_fire else 0.0,
                      1.0 - self.confidence, self._timestamp)
        ]
        # Cardinal neighbors at reduced confidence
        r0, c0 = self._location
        for dr, dc in [(-2,0), (2,0), (0,-2), (0,2)]:
            points.append(DataPoint(
                (r0+dr, c0+dc), VariableType.FIRE_DETECTION,
                1.0 if self.is_fire else 0.0,
                1.0 - self.confidence * 0.85,
                self._timestamp))
        return points
    
    def is_expired(self, query_time: float) -> bool:
        return False


@dataclass(frozen=True)
class SatelliteFMCObservation(Observation):
    """
    FMC estimate from MODIS, VIIRS, or Sentinel-2.
    
    Resolution: 250m-500m (MODIS), 10m (Sentinel-2), 375m (VIIRS).
    Frequency: daily (coarse res), weekly (10m).
    Accuracy: sigma ~0.10-0.15, much higher than drone (~0.05).
    
    Provides regional FMC baseline in areas drones haven't visited.
    Decays at the same rate as drone FMC (same physical variable),
    but starts with higher sigma so becomes useless sooner in
    absolute terms.
    
    Emits thinned grid of DataPoints across the pixel footprint
    with representativeness error added to sigma.
    """
    _source_id: str                # e.g., "MODIS_Terra_20250107"
    _timestamp: float
    _location: tuple[int, int]     # pixel center
    
    fmc: float                     # fraction (pixel-average FMC)
    fmc_sigma: float               # ~0.10-0.15 (instrument + model error)
    pixel_radius_cells: int        # varies by satellite
    correlation_length_cells: int = 10  # ~500m, for thinning
    
    # Representativeness error: pixel average ≠ cell value
    # Sub-pixel variability adds uncertainty at each DataPoint
    REPRESENTATIVENESS_FRACTION: float = 0.5  # sigma_repr = 0.5 × sigma_instrument
    
    @property
    def source_id(self): return self._source_id
    @property
    def timestamp(self): return self._timestamp
    @property
    def location(self): return self._location
    
    @property
    def variables(self):
        return [VariableType.FMC]
    
    def to_data_points(self, query_time: float) -> list[DataPoint]:
        decayed = self._decay_sigma(self.fmc_sigma, VariableType.FMC, query_time)
        repr_sigma = self.fmc_sigma * self.REPRESENTATIVENESS_FRACTION
        total_sigma = np.sqrt(decayed**2 + repr_sigma**2)
        
        step = max(1, self.correlation_length_cells)
        points = []
        r0, c0 = self._location
        for dr in range(-self.pixel_radius_cells,
                         self.pixel_radius_cells + 1, step):
            for dc in range(-self.pixel_radius_cells,
                             self.pixel_radius_cells + 1, step):
                if np.sqrt(dr**2 + dc**2) > self.pixel_radius_cells:
                    continue
                points.append(DataPoint(
                    (r0+dr, c0+dc), VariableType.FMC,
                    self.fmc, total_sigma, self._timestamp))
        return points
    
    def is_expired(self, query_time: float) -> bool:
        return self._is_variable_expired(
            self.fmc_sigma, VariableType.FMC, query_time)
```

---

## Summary: What Each Type Produces

| Observation class | DataPoint variables | Footprint | Decays? | Typical count per observation |
|---|---|---|---|---|
| RAWSObservation | FMC, WS, WD, T, RH | 1 cell | Never | 5 points |
| DroneWindObservation | WS, WD | 1 cell | Yes (1-2 hr tau) | 2 points |
| DroneFMCObservation | FMC | ~25 cells (footprint) | Yes (1 hr tau) | ~25 points |
| DroneWeatherObservation | T, RH | 1 cell | Yes (1-2 hr tau) | 2 points |
| DroneThermalObservation | FIRE_DETECTION | swath (~100-1000 cells) | Never | ~100-1000 points |
| DroneLiDARObservation | (none — updates terrain) | ~5 cells | Never | 0 points (routed to PriorState) |
| DroneGasObservation | (none — stored for SA) | 1 cell | 15 min | 0 points |
| GOESFireDetection | FIRE_DETECTION | ~16 thinned across 2km pixel | Never | ~16 points |
| VIIRSFireDetection | FIRE_DETECTION | 5 cells | Never | 5 points |
| SatelliteFMCObservation | FMC | thinned across pixel | Yes (1 hr tau) | ~4-64 points depending on pixel size |
