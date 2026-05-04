# IGNIS: Dynamic Prior Data Specification

---

## The Two Types of Prior

**Static Prior (already implemented as TerrainData):**
Loaded once from LANDFIRE. Never changes during a simulation.
Elevation, slope, aspect, fuel model, canopy cover, canopy height, CBH, CBD.
Owned by the existing terrain loader.

**Dynamic Prior (this spec):**
Updated each cycle (or less frequently for slow-changing data). Represents the best available model-derived estimates of environmental conditions BEFORE observations correct them. These are the mean functions the GP corrects around.

The key distinction: static prior is geography. Dynamic prior is weather and derived fields.

---

## What the Dynamic Prior Contains

| Field | Source | Update frequency | What it provides |
|---|---|---|---|
| Nelson FMC field | Nelson model (T, RH, solar, terrain) | Each cycle (hourly weather changes) | GP prior mean for FMC |
| Wind speed field | HRRR / GFS forecast or scenario definition | Each cycle or on forecast update | GP prior mean for wind speed |
| Wind direction field | HRRR / GFS forecast or scenario definition | Each cycle or on forecast update | GP prior mean for wind direction |
| Temperature field | HRRR / GFS + lapse rate correction | Each cycle | Input to Nelson model |
| Humidity field | HRRR / GFS + terrain correction | Each cycle | Input to Nelson model |
| Solar radiation estimate | Computed from time-of-day + terrain aspect/slope | Each cycle | Input to Nelson model |
| Fire state estimate | Ensemble consensus or reconstruction | Each cycle | Ensemble initialization (per-member fire fronts) |
| Fire state uncertainty | From ensemble spread or observation recency | Each cycle | Perturbation scale for fire front |

---

## Class Design

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class DynamicPrior:
    """
    Time-varying environmental baseline that the GP corrects around.
    
    Updated at the start of each IGNIS cycle from weather forecasts,
    model computations, and ensemble outputs. Everything here is
    model-derived — no raw observations. Observations live in the
    ObservationStore and correct these fields via the GP.
    
    The GP uses these fields as its prior mean function:
      GP_posterior = DynamicPrior_field + GP_correction(observations)
    
    Where observations are sparse, the GP correction is near zero
    and the output equals the DynamicPrior field.
    """
    
    grid_shape: tuple[int, int]
    resolution_m: float
    
    # --- Current simulation time ---
    timestamp: float = 0.0
    
    # --- Weather fields (from forecast model) ---
    # These drive the Nelson FMC model and serve as wind prior means
    temperature: Optional[np.ndarray] = None       # °C, (rows, cols)
    humidity: Optional[np.ndarray] = None           # fraction 0-1, (rows, cols)
    solar_radiation: Optional[np.ndarray] = None    # W/m², (rows, cols)
    
    # --- Derived fields (computed from weather + terrain) ---
    nelson_fmc: Optional[np.ndarray] = None         # fraction, (rows, cols)
    wind_speed_prior: Optional[np.ndarray] = None   # m/s, (rows, cols)
    wind_direction_prior: Optional[np.ndarray] = None # degrees, (rows, cols)
    
    # --- Fire state (from ensemble or reconstruction) ---
    fire_arrival_time: Optional[np.ndarray] = None  # seconds, (rows, cols)
    fire_burn_probability: Optional[np.ndarray] = None  # 0-1, (rows, cols)
    fire_uncertainty: Optional[np.ndarray] = None   # seconds, (rows, cols)
    
    # --- Metadata: when each field was last updated ---
    _temperature_updated: float = -1.0
    _humidity_updated: float = -1.0
    _wind_updated: float = -1.0
    _nelson_updated: float = -1.0
    _fire_state_updated: float = -1.0
    
    # --- Weather source tracking ---
    weather_source: str = "default"  # "HRRR", "GFS", "scenario", "RAWS_interpolated"
    
    def is_initialized(self) -> bool:
        """Check that minimum required fields are set."""
        return (self.nelson_fmc is not None and
                self.wind_speed_prior is not None and
                self.wind_direction_prior is not None)
```

---

## Update Methods

### Weather Update

```python
    def update_weather(self, temperature: np.ndarray, 
                        humidity: np.ndarray,
                        source: str = "HRRR",
                        timestamp: Optional[float] = None):
        """
        Update temperature and humidity fields from weather forecast.
        
        In operational use: called when a new HRRR forecast arrives (~hourly).
        In simulation: called each cycle with ground truth or scenario weather.
        
        Triggers Nelson FMC recomputation via recompute_nelson().
        """
        self.temperature = temperature.astype(np.float32)
        self.humidity = humidity.astype(np.float32)
        self.weather_source = source
        self._temperature_updated = timestamp or self.timestamp
        self._humidity_updated = timestamp or self.timestamp
    
    def update_wind(self, wind_speed: np.ndarray,
                     wind_direction: np.ndarray,
                     source: str = "HRRR",
                     timestamp: Optional[float] = None):
        """
        Update wind prior mean fields.
        
        Sources (in order of preference):
        1. HRRR 3km forecast (interpolated to grid)
        2. GFS 28km forecast (interpolated to grid)
        3. Scenario-defined wind field
        4. Spatial interpolation from RAWS (worst case)
        
        The GP corrects this field using RAWS + drone wind observations.
        Regions far from any observation revert to this prior.
        """
        self.wind_speed_prior = np.clip(
            wind_speed, 0.1, 50.0).astype(np.float32)
        self.wind_direction_prior = (wind_direction % 360).astype(np.float32)
        self.weather_source = source
        self._wind_updated = timestamp or self.timestamp
    
    def update_solar(self, terrain, hour_utc: float, 
                      latitude: float):
        """
        Compute solar radiation from time-of-day and terrain.
        
        Simplified model: solar angle from latitude + hour,
        modified by slope and aspect (south-facing slopes get more).
        Used as input to Nelson FMC model.
        """
        # Solar elevation angle
        # Simplified: peak at solar noon (local), zero at night
        hour_local = (hour_utc - self._utc_offset) % 24
        solar_elevation = max(0, 90 - abs(hour_local - 12) * 15)
        solar_elevation_rad = np.radians(solar_elevation)
        
        if solar_elevation <= 0:
            self.solar_radiation = np.zeros(self.grid_shape, dtype=np.float32)
            return
        
        # Base radiation at this solar angle
        base_radiation = 1000 * np.sin(solar_elevation_rad)  # W/m² max
        
        # Terrain modification: slope/aspect interaction with solar angle
        # South-facing slopes receive more when sun is in the south
        # (Northern hemisphere)
        aspect_rad = np.radians(terrain.aspect)
        slope_rad = np.radians(terrain.slope)
        
        # Cosine of incidence angle between sun and slope normal
        cos_incidence = (np.sin(solar_elevation_rad) * np.cos(slope_rad) +
                         np.cos(solar_elevation_rad) * np.sin(slope_rad) *
                         np.cos(aspect_rad - np.pi))  # sun from south
        cos_incidence = np.clip(cos_incidence, 0, 1)
        
        # Canopy shading reduces radiation reaching the surface
        canopy_transmission = 1.0 - 0.7 * terrain.canopy_cover
        
        self.solar_radiation = (
            base_radiation * cos_incidence * canopy_transmission
        ).astype(np.float32)
```

### Nelson FMC Recomputation

```python
    def recompute_nelson(self, terrain):
        """
        Recompute the Nelson dead fuel moisture field from current
        weather and terrain.
        
        This is the GP's prior mean for FMC. Where no observations
        exist, the GP output equals this field. Where drone or
        satellite FMC observations exist, the GP corrects this.
        
        Should be called after update_weather() and update_solar().
        """
        if self.temperature is None or self.humidity is None:
            return
        
        rh_pct = self.humidity * 100
        T = self.temperature
        
        # Nelson equilibrium moisture content (EMC)
        # Piecewise function of relative humidity
        emc = np.where(
            rh_pct < 10,
            0.03229 + 0.281073 * rh_pct - 0.000578 * rh_pct * T,
            np.where(
                rh_pct < 50,
                2.22749 + 0.160107 * rh_pct - 0.01478 * T,
                21.0606 + 0.005565 * rh_pct**2 - 0.00035 * rh_pct * T
            )
        )
        emc_fraction = emc / 100.0
        
        # Solar radiation effect: high radiation dries fuel
        if self.solar_radiation is not None:
            # Normalize radiation to 0-1 scale
            solar_factor = np.clip(self.solar_radiation / 1000.0, 0, 1)
            # Radiation reduces EMC by up to 3%
            emc_fraction -= 0.03 * solar_factor
        
        # Canopy shading retains moisture
        if terrain.canopy_cover is not None:
            emc_fraction += 0.02 * terrain.canopy_cover
        
        # Elevation effect: higher = cooler = moister (lapse rate)
        if terrain.elevation is not None:
            elev_norm = (terrain.elevation - terrain.elevation.mean()) / (
                terrain.elevation.std() + 1e-6)
            emc_fraction += 0.015 * np.clip(elev_norm, -2, 2)
        
        self.nelson_fmc = np.clip(
            emc_fraction, 0.02, 0.40).astype(np.float32)
        self._nelson_updated = self.timestamp
```

### Fire State Update

```python
    def update_fire_state(self, ensemble_arrival_times: np.ndarray,
                           current_time: float,
                           last_observed: Optional[np.ndarray] = None,
                           max_ros: float = 5.0):
        """
        Update fire state from ensemble output.
        
        ensemble_arrival_times: (N, rows, cols) from the latest ensemble run
        current_time: simulation time in seconds
        last_observed: (rows, cols) timestamp of last fire observation per cell
                       (from ObservationStore fire detections)
        max_ros: maximum credible ROS in m/s (for uncertainty computation)
        
        Called at the end of each cycle, after ensemble has run.
        The fire state serves as the basis for next cycle's ensemble
        initialization (per-member fire fronts).
        """
        # Burn probability from ensemble consensus
        self.fire_burn_probability = (
            ensemble_arrival_times < current_time
        ).mean(axis=0).astype(np.float32)
        
        # Best-estimate arrival time: ensemble median
        # (median is more robust than mean for bimodal distributions)
        self.fire_arrival_time = np.median(
            ensemble_arrival_times, axis=0).astype(np.float32)
        
        # Fire state uncertainty: ensemble spread at the fire boundary
        # Interior (certainly burned) and exterior (certainly unburned)
        # have low uncertainty. The uncertain zone is the fire perimeter.
        arrival_std = ensemble_arrival_times.std(axis=0)
        self.fire_uncertainty = arrival_std.astype(np.float32)
        
        # Increase uncertainty where fire hasn't been observed recently
        if last_observed is not None:
            time_since_obs = np.clip(current_time - last_observed, 0, 14400)
            # Uncertainty grows with time since observation and max possible advance
            obs_uncertainty = max_ros * time_since_obs  # meters of possible advance
            obs_uncertainty_s = obs_uncertainty / (max_ros + 1e-6)  # back to seconds
            self.fire_uncertainty = np.sqrt(
                self.fire_uncertainty**2 + obs_uncertainty_s**2
            ).astype(np.float32)
        
        self._fire_state_updated = current_time
```

---

## Cycle Update Method

```python
    def update_cycle(self, current_time: float, terrain,
                      weather_source: Optional[dict] = None,
                      ensemble_result=None,
                      fire_observations_last_observed=None):
        """
        Master update called once per cycle by the orchestrator.
        
        Updates all dynamic fields in the correct order:
        1. Weather (T, RH) from forecast or scenario
        2. Solar radiation from time + terrain
        3. Nelson FMC from weather + solar + terrain
        4. Wind prior from forecast or scenario
        5. Fire state from ensemble (if available)
        
        weather_source: dict with keys "temperature", "humidity",
                        "wind_speed", "wind_direction" as (rows, cols) arrays.
                        If None, retains previous fields.
        """
        self.timestamp = current_time
        hour_utc = (current_time / 3600.0) % 24
        
        # 1. Weather fields
        if weather_source is not None:
            if "temperature" in weather_source:
                self.update_weather(
                    weather_source["temperature"],
                    weather_source["humidity"],
                    source=weather_source.get("source", "scenario"))
            
            if "wind_speed" in weather_source:
                self.update_wind(
                    weather_source["wind_speed"],
                    weather_source["wind_direction"],
                    source=weather_source.get("source", "scenario"))
        
        # 2. Solar radiation
        latitude = weather_source.get("latitude", 34.0) if weather_source else 34.0
        self.update_solar(terrain, hour_utc, latitude)
        
        # 3. Nelson FMC
        self.recompute_nelson(terrain)
        
        # 4. Fire state from last cycle's ensemble
        if ensemble_result is not None:
            self.update_fire_state(
                ensemble_result.member_arrival_times,
                current_time,
                fire_observations_last_observed)
    
    def get_gp_prior_means(self) -> dict[str, np.ndarray]:
        """
        Return the prior mean fields the GP uses.
        
        The GP fits residuals against these — observations that
        agree with these fields produce zero residual, observations
        that disagree produce corrections.
        """
        return {
            "fmc": self.nelson_fmc,
            "wind_speed": self.wind_speed_prior,
            "wind_direction": self.wind_direction_prior,
        }
    
    def get_weather_for_nelson(self) -> dict[str, np.ndarray]:
        """
        Return weather fields needed for Nelson FMC computation.
        Used when drone T/RH observations update local weather
        and Nelson should be recomputed locally.
        """
        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "solar_radiation": self.solar_radiation,
        }
```

---

## Integration with GP

```python
class IGNISGPPrior:
    def fit(self, dynamic_prior: DynamicPrior, 
            obs_store: ObservationStore,
            current_time: float):
        """
        Fit GP using dynamic prior as mean function and
        observations from store as corrections.
        """
        # Set prior means from DynamicPrior
        prior_means = dynamic_prior.get_gp_prior_means()
        self.set_nelson_mean(prior_means["fmc"])
        self.set_wind_prior_mean(
            prior_means["wind_speed"],
            prior_means["wind_direction"])
        
        # Get observations from store (decayed, thinned)
        fmc_points = obs_store.get_data_points(
            current_time, VariableType.FMC,
            min_spacing_cells=self.fmc_thin_spacing)
        
        ws_points = obs_store.get_data_points(
            current_time, VariableType.WIND_SPEED,
            min_spacing_cells=self.wind_thin_spacing)
        
        wd_points = obs_store.get_data_points(
            current_time, VariableType.WIND_DIRECTION,
            min_spacing_cells=self.wind_thin_spacing)
        
        # Fit GP on residuals (observation - prior mean)
        # The GP models the CORRECTION to the dynamic prior
        self._fit_fmc(fmc_points)
        self._fit_wind_speed(ws_points)
        self._fit_wind_direction(wd_points)
```

---

## Integration with Orchestrator

```python
class Orchestrator:
    def __init__(self, terrain, ...):
        self.dynamic_prior = DynamicPrior(
            grid_shape=terrain.shape,
            resolution_m=terrain.resolution_m)
        self.obs_store = ObservationStore()
        self.gp = IGNISGPPrior(...)
    
    def run_cycle(self):
        self.obs_store.lock()
        try:
            # 1. Update dynamic prior from weather + last ensemble
            self.dynamic_prior.update_cycle(
                current_time=self.current_time,
                terrain=self.terrain,
                weather_source=self._get_current_weather(),
                ensemble_result=self._last_ensemble_result,
                fire_observations_last_observed=self._get_fire_obs_times())
            
            # 2. GP fits corrections to dynamic prior using observations
            self.gp.fit(self.dynamic_prior, self.obs_store, self.current_time)
            gp_estimate = self.gp.predict(self.terrain.shape)
            
            # 3. Ensemble uses GP estimate (= dynamic prior + corrections)
            # ... fire engine, info field, selection, etc. ...
            
        finally:
            self.obs_store.unlock()
            self.obs_store.prune(self.current_time)
    
    def _get_current_weather(self) -> dict:
        """
        Get weather for this cycle.
        
        In simulation: from ground truth weather evolution.
        In operational: from latest HRRR/GFS download.
        """
        # Simulation mode:
        return {
            "temperature": self.weather_model.temperature_at(self.current_time),
            "humidity": self.weather_model.humidity_at(self.current_time),
            "wind_speed": self.weather_model.wind_speed_at(self.current_time),
            "wind_direction": self.weather_model.wind_direction_at(self.current_time),
            "source": "simulation",
            "latitude": self.terrain_latitude,
        }
```

---

## Data Flow Summary

```
STATIC PRIOR (TerrainData)               DYNAMIC PRIOR (DynamicPrior)
Loaded once, never changes                Updated each cycle
├── elevation                             ├── temperature field (from HRRR/scenario)
├── slope                                 ├── humidity field (from HRRR/scenario)
├── aspect                                ├── solar radiation (from time + terrain)
├── fuel model                            ├── Nelson FMC (from T + RH + solar + terrain)
├── canopy cover                          ├── wind speed prior (from HRRR/scenario)
├── canopy height                         ├── wind direction prior (from HRRR/scenario)
├── canopy base height                    ├── fire arrival time (from ensemble)
└── canopy bulk density                   ├── fire burn probability (from ensemble)
     │                                    └── fire uncertainty (from ensemble + obs recency)
     │                                         │
     │  terrain inputs to Nelson               │  prior means for GP
     └──────────────┬──────────────────────────┘
                    │
                    ▼
              GP fits corrections using ObservationStore
                    │
                    ▼
              GPEstimate (posterior = dynamic prior + corrections)
                    │
                    ▼
              Ensemble + Fire Engine + Information Field
```

---

## What Triggers Updates

| Field | Trigger | How often |
|---|---|---|
| Temperature, humidity | New HRRR forecast arrives, or scenario advances | Every 1-3 cycles (~20-60 min) |
| Solar radiation | Time advances (automatic in update_cycle) | Every cycle |
| Nelson FMC | Temperature, humidity, or solar changes | Every cycle (after weather update) |
| Wind speed/direction | New HRRR forecast, or scenario wind event | Every 1-3 cycles, or on wind shift event |
| Fire arrival time | Ensemble completes a cycle | Every cycle |
| Fire burn probability | Ensemble completes a cycle | Every cycle |
| Fire uncertainty | Ensemble completes + fire observation recency | Every cycle |

---

## Files to Create

| File | Contents |
|---|---|
| `ignis/prior/dynamic_prior.py` | DynamicPrior class with all update methods |
| `ignis/prior/__init__.py` | Exports DynamicPrior |
| `ignis/tests/test_dynamic_prior.py` | Tests for Nelson, solar, wind update, fire state |

The existing `TerrainData` (static prior) stays where it is. `DynamicPrior` is a new class that sits alongside it. The orchestrator holds both:

```python
self.terrain = TerrainData(...)        # static, loaded once
self.dynamic_prior = DynamicPrior(...) # updated each cycle
```
