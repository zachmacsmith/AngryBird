"""
Dynamic prior: time-varying environmental baseline the GP corrects around.

Updated at the start of each IGNIS cycle from weather forecasts, model
computations, and ensemble outputs.  Everything here is model-derived —
no raw observations.  Observations live in ObservationStore and correct
these fields via the GP.

Data flow:
  GP_posterior = DynamicPrior_field + GP_correction(observations)

Where observations are sparse the GP correction is near zero and the
output equals the DynamicPrior field.

Arrival times in EnsembleResult are in **minutes** (GPU engine convention).
DynamicPrior stores fire_arrival_time / fire_uncertainty in **seconds** so
they share units with simulation time (current_time is always in seconds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DynamicPrior:
    """
    Time-varying environmental baseline that the GP corrects around.

    Updated at the start of each IGNIS cycle from weather forecasts,
    model computations, and ensemble outputs.  Everything here is
    model-derived — no raw observations.  Observations live in the
    ObservationStore and correct these fields via the GP.
    """

    grid_shape: tuple[int, int]
    resolution_m: float

    # --- Current simulation time ---
    timestamp: float = 0.0

    # --- Weather fields (from forecast model) ---
    temperature: Optional[np.ndarray] = None       # °C, (rows, cols)
    humidity: Optional[np.ndarray] = None          # fraction 0-1, (rows, cols)
    solar_radiation: Optional[np.ndarray] = None   # W/m², (rows, cols)

    # --- Derived fields (computed from weather + terrain) ---
    nelson_fmc: Optional[np.ndarray] = None            # fraction, (rows, cols)
    wind_speed_prior: Optional[np.ndarray] = None      # m/s, (rows, cols)
    wind_direction_prior: Optional[np.ndarray] = None  # degrees, (rows, cols)

    # --- Fire state (from ensemble or reconstruction) ---
    fire_arrival_time: Optional[np.ndarray] = None    # seconds, (rows, cols)
    fire_burn_probability: Optional[np.ndarray] = None  # 0-1, (rows, cols)
    fire_uncertainty: Optional[np.ndarray] = None     # seconds, (rows, cols)

    # --- Metadata: when each field was last updated ---
    _temperature_updated: float = field(default=-1.0, repr=False)
    _humidity_updated: float = field(default=-1.0, repr=False)
    _wind_updated: float = field(default=-1.0, repr=False)
    _nelson_updated: float = field(default=-1.0, repr=False)
    _fire_state_updated: float = field(default=-1.0, repr=False)

    # --- Weather source tracking ---
    weather_source: str = "default"

    # ----------------------------------------------------------------
    # Predicates
    # ----------------------------------------------------------------

    def is_initialized(self) -> bool:
        """True when minimum fields needed by the GP are available."""
        return (
            self.nelson_fmc is not None
            and self.wind_speed_prior is not None
            and self.wind_direction_prior is not None
        )

    # ----------------------------------------------------------------
    # Weather updates
    # ----------------------------------------------------------------

    def update_weather(
        self,
        temperature: "np.ndarray | float",
        humidity: "np.ndarray | float",
        source: str = "scenario",
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update temperature and humidity fields.

        In simulation: called each cycle with scenario values.
        In operational: called when a new HRRR/GFS forecast arrives.
        Triggers Nelson FMC recomputation via recompute_nelson().

        temperature: °C, scalar or (rows, cols)
        humidity:    fraction 0-1, scalar or (rows, cols)
        """
        rows, cols = self.grid_shape
        self.temperature = np.broadcast_to(
            np.asarray(temperature, dtype=np.float32), (rows, cols)
        ).copy()
        self.humidity = np.broadcast_to(
            np.asarray(humidity, dtype=np.float32), (rows, cols)
        ).copy()
        self.weather_source = source
        ts = timestamp if timestamp is not None else self.timestamp
        self._temperature_updated = ts
        self._humidity_updated = ts

    def update_wind(
        self,
        wind_speed: "np.ndarray | float",
        wind_direction: "np.ndarray | float",
        source: str = "scenario",
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update wind prior mean fields.

        The GP corrects this field using RAWS + drone wind observations.
        Regions far from any observation revert to this prior.

        wind_speed:     m/s, scalar or (rows, cols)
        wind_direction: degrees, scalar or (rows, cols)
        """
        rows, cols = self.grid_shape
        ws = np.broadcast_to(
            np.asarray(wind_speed, dtype=np.float32), (rows, cols)
        ).copy()
        wd = np.broadcast_to(
            np.asarray(wind_direction, dtype=np.float32), (rows, cols)
        ).copy()
        self.wind_speed_prior = np.clip(ws, 0.1, 50.0).astype(np.float32)
        self.wind_direction_prior = (wd % 360.0).astype(np.float32)
        self.weather_source = source
        self._wind_updated = timestamp if timestamp is not None else self.timestamp

    def update_solar(self, terrain, hour_local: float, latitude: float) -> None:
        """
        Compute solar radiation from time-of-day and terrain geometry.

        hour_local: local solar hour (0-24)
        latitude:   degrees north
        """
        # Solar elevation angle: peaks at solar noon, zero at night.
        solar_elevation = max(0.0, 90.0 - abs(hour_local - 12.0) * 15.0)

        if solar_elevation <= 0.0:
            self.solar_radiation = np.zeros(self.grid_shape, dtype=np.float32)
            return

        solar_elevation_rad = np.radians(solar_elevation)
        base_radiation = 1000.0 * np.sin(solar_elevation_rad)  # W/m²

        aspect_rad = np.radians(terrain.aspect)
        slope_rad = np.radians(terrain.slope)

        # Cosine of incidence angle between sun ray and slope normal
        # (sun from south in Northern Hemisphere → aspect offset = π)
        cos_incidence = np.clip(
            np.sin(solar_elevation_rad) * np.cos(slope_rad)
            + np.cos(solar_elevation_rad)
            * np.sin(slope_rad)
            * np.cos(aspect_rad - np.pi),
            0.0,
            1.0,
        )

        canopy_transmission = (
            1.0 - 0.7 * terrain.canopy_cover
            if terrain.canopy_cover is not None
            else np.ones(self.grid_shape, dtype=np.float32)
        )

        self.solar_radiation = (
            base_radiation * cos_incidence * canopy_transmission
        ).astype(np.float32)

    # ----------------------------------------------------------------
    # Nelson FMC recomputation
    # ----------------------------------------------------------------

    def recompute_nelson(self, terrain) -> None:
        """
        Recompute the Nelson dead fuel moisture field from current weather
        and terrain.  This is the GP's prior mean for FMC.

        Should be called after update_weather() and update_solar().
        """
        if self.temperature is None or self.humidity is None:
            return

        rh_pct = self.humidity * 100.0
        T = self.temperature

        # Fosberg-Deeming three-segment EMC formula
        emc = np.where(
            rh_pct < 10.0,
            0.03229 + 0.281073 * rh_pct - 0.000578 * rh_pct * T,
            np.where(
                rh_pct < 50.0,
                2.22749 + 0.160107 * rh_pct - 0.01478 * T,
                21.0606 + 0.005565 * rh_pct ** 2 - 0.00035 * rh_pct * T,
            ),
        )
        emc_fraction = emc / 100.0

        # Solar radiation dries fuel (up to −3%)
        if self.solar_radiation is not None:
            solar_factor = np.clip(self.solar_radiation / 1000.0, 0.0, 1.0)
            emc_fraction -= 0.03 * solar_factor

        # Canopy shading retains moisture
        if terrain.canopy_cover is not None:
            emc_fraction += 0.02 * terrain.canopy_cover

        # Elevation: higher = cooler = moister
        if terrain.elevation is not None:
            elev_norm = (terrain.elevation - terrain.elevation.mean()) / (
                terrain.elevation.std() + 1e-6
            )
            emc_fraction += 0.015 * np.clip(elev_norm, -2.0, 2.0)

        self.nelson_fmc = np.clip(emc_fraction, 0.02, 0.40).astype(np.float32)
        self._nelson_updated = self.timestamp

    # ----------------------------------------------------------------
    # Fire state update
    # ----------------------------------------------------------------

    def update_fire_state(
        self,
        ensemble_arrival_times: np.ndarray,
        current_time: float,
        last_observed: Optional[np.ndarray] = None,
        max_ros: float = 5.0,
    ) -> None:
        """
        Update fire state from ensemble output.

        ensemble_arrival_times: float32[N, rows, cols] in **minutes**
                                (GPU engine convention; sentinel = 2×horizon_min)
        current_time:           simulation time in seconds
        last_observed:          (rows, cols) timestamp (seconds) of last fire
                                observation per cell, from ObservationStore
        max_ros:                maximum credible ROS in m/s
        """
        current_time_min = current_time / 60.0

        # Burn probability: fraction of members that have arrived by now
        self.fire_burn_probability = (
            ensemble_arrival_times < current_time_min
        ).mean(axis=0).astype(np.float32)

        # Best-estimate arrival time in seconds (median is robust to bimodal)
        self.fire_arrival_time = (
            np.median(ensemble_arrival_times, axis=0) * 60.0
        ).astype(np.float32)

        # Uncertainty: ensemble spread at fire boundary, converted to seconds
        arrival_std_s = ensemble_arrival_times.std(axis=0) * 60.0
        self.fire_uncertainty = arrival_std_s.astype(np.float32)

        # Inflate uncertainty where fire hasn't been observed recently
        if last_observed is not None:
            time_since_obs = np.clip(current_time - last_observed, 0.0, 14400.0)
            obs_uncertainty_s = time_since_obs  # max_ros * t / max_ros = t
            self.fire_uncertainty = np.sqrt(
                self.fire_uncertainty ** 2 + obs_uncertainty_s ** 2
            ).astype(np.float32)

        self._fire_state_updated = current_time

    # ----------------------------------------------------------------
    # Master cycle update
    # ----------------------------------------------------------------

    def update_cycle(
        self,
        current_time: float,
        terrain,
        weather_source: Optional[dict] = None,
        ensemble_result=None,
        fire_observations_last_observed: Optional[np.ndarray] = None,
    ) -> None:
        """
        Master update called once per cycle by the orchestrator.

        Updates all dynamic fields in order:
          1. Weather (T, RH) from forecast or scenario dict
          2. Wind prior from forecast or scenario dict
          3. Solar radiation from time + terrain
          4. Nelson FMC from weather + solar + terrain
          5. Fire state from ensemble (if available)

        weather_source keys (all optional):
          "temperature"   — scalar or (rows, cols) °C
          "humidity"      — scalar or (rows, cols) fraction 0-1
          "wind_speed"    — scalar or (rows, cols) m/s
          "wind_direction"— scalar or (rows, cols) degrees
          "source"        — str label ("HRRR", "GFS", "scenario", …)
          "latitude"      — float degrees north (default 34.0)
          "hour_local"    — float local solar hour; if absent computed from
                            current_time (simulation clock starts at 06:00)
        """
        self.timestamp = current_time

        # 1 & 2. Weather and wind from source dict
        if weather_source is not None:
            source_label = weather_source.get("source", "scenario")
            if "temperature" in weather_source:
                self.update_weather(
                    weather_source["temperature"],
                    weather_source["humidity"],
                    source=source_label,
                )
            if "wind_speed" in weather_source:
                self.update_wind(
                    weather_source["wind_speed"],
                    weather_source["wind_direction"],
                    source=source_label,
                )

        # 3. Solar radiation
        latitude = (
            weather_source.get("latitude", 34.0) if weather_source else 34.0
        )
        if weather_source is not None and "hour_local" in weather_source:
            hour_local = float(weather_source["hour_local"])
        else:
            # Simulation clock starts at 06:00 local solar time by convention.
            hour_local = (6.0 + current_time / 3600.0) % 24.0
        self.update_solar(terrain, hour_local, latitude)

        # 4. Nelson FMC
        self.recompute_nelson(terrain)

        # 5. Fire state from last cycle's ensemble
        if ensemble_result is not None:
            self.update_fire_state(
                ensemble_result.member_arrival_times,
                current_time,
                fire_observations_last_observed,
            )

    # ----------------------------------------------------------------
    # GP interface
    # ----------------------------------------------------------------

    def get_gp_prior_means(self) -> dict[str, Optional[np.ndarray]]:
        """
        Return the prior mean fields the GP uses.

        The GP fits residuals against these — observations that agree
        with these fields produce zero residual; observations that
        disagree produce corrections.  Returns None for fields not yet
        computed.
        """
        return {
            "fmc": self.nelson_fmc,
            "wind_speed": self.wind_speed_prior,
            "wind_direction": self.wind_direction_prior,
        }

    def get_weather_for_nelson(self) -> dict[str, Optional[np.ndarray]]:
        """
        Return current weather fields.  Used when drone T/RH observations
        should trigger a local Nelson recomputation.
        """
        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "solar_radiation": self.solar_radiation,
        }
