"""
Dynamic prior: time-varying environmental baseline the GP corrects around.

Data flow
---------
  GP_posterior = DynamicPrior_field + GP_correction(observations)

Where observations are sparse the correction is near zero and the output
equals the DynamicPrior field.  Where drones or RAWS stations have measured,
the GP posterior pulls toward those measurements.

Each cycle:
  1. An EnvironmentalDataSource is queried for the latest weather / wind /
     satellite measurements.
  2. Those raw Measurement objects are stored in `inputs` (for inspection).
  3. Private _compute_* methods derive the GP prior means from the raw inputs
     plus terrain and time.
  4. get_gp_prior_means() returns the result for the GP.

Adding a new data type (e.g. RAWS T/RH, aircraft soundings):
  - Add a Measurement subclass in measurements.py.
  - Add a _apply_<type> method here that knows how to use it.
  - Add the get_<type>() call inside compute_cycle().

Arrival times in EnsembleResult are in **minutes** (GPU engine convention).
fire_arrival_time / fire_uncertainty here are in **seconds** to share units
with simulation time (current_time is always in seconds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from .measurements import (
    Measurement,
    NWPWeatherMeasurement,
    NWPWindMeasurement,
    SatelliteFMCMeasurement,
)

if TYPE_CHECKING:
    from .sources import EnvironmentalDataSource


@dataclass
class DynamicPrior:
    """
    Time-varying environmental baseline updated once per IGNIS cycle.

    Public state
    ------------
    inputs            Raw Measurement objects applied in the last compute_cycle()
                      call — kept for logging, diagnostics, and replay.
    nelson_fmc        GP FMC prior mean: Fosberg-Deeming EMC + terrain corrections.
    wind_speed_prior  GP wind speed prior mean (m/s).
    wind_direction_prior  GP wind direction prior mean (degrees).
    fire_burn_probability  Fraction of ensemble members burned by current_time.
    fire_arrival_time      Median ensemble arrival time (seconds).
    fire_uncertainty       Ensemble arrival time std dev (seconds).
    """

    grid_shape: tuple[int, int]
    resolution_m: float

    # ── Current simulation time ──────────────────────────────────────────────
    timestamp: float = 0.0

    # ── Raw inputs from the last compute_cycle() call ────────────────────────
    # Populated by add_input(); cleared at the start of each compute_cycle().
    inputs: list[Measurement] = field(default_factory=list)

    # ── Intermediate weather fields (kept so _compute_nelson can rerun) ──────
    temperature: Optional[np.ndarray] = None   # °C, float32[rows, cols]
    humidity: Optional[np.ndarray] = None      # fraction, float32[rows, cols]
    solar_radiation: Optional[np.ndarray] = None  # W/m², float32[rows, cols]

    # ── Derived GP prior means ───────────────────────────────────────────────
    nelson_fmc: Optional[np.ndarray] = None           # fraction, float32[rows, cols]
    wind_speed_prior: Optional[np.ndarray] = None     # m/s, float32[rows, cols]
    wind_direction_prior: Optional[np.ndarray] = None # degrees, float32[rows, cols]

    # ── Fire state (from ensemble — model output, not measurement) ───────────
    fire_arrival_time: Optional[np.ndarray] = None    # seconds, float32[rows, cols]
    fire_burn_probability: Optional[np.ndarray] = None  # [0,1], float32[rows, cols]
    fire_uncertainty: Optional[np.ndarray] = None     # seconds, float32[rows, cols]

    # ── Source label for the last cycle ─────────────────────────────────────
    last_source: str = "none"

    # ----------------------------------------------------------------
    # Predicate
    # ----------------------------------------------------------------

    def is_initialized(self) -> bool:
        """True when the GP prior means are all populated."""
        return (
            self.nelson_fmc is not None
            and self.wind_speed_prior is not None
            and self.wind_direction_prior is not None
        )

    # ----------------------------------------------------------------
    # Input registration
    # ----------------------------------------------------------------

    def add_input(self, m: Measurement) -> None:
        """
        Register a raw measurement to be applied in the next compute_cycle().

        Call this before compute_cycle() for any measurement type not provided
        by the DataSource (e.g. a one-off satellite pass, aircraft sounding).
        """
        self.inputs.append(m)

    # ----------------------------------------------------------------
    # Master cycle entry point
    # ----------------------------------------------------------------

    def compute_cycle(
        self,
        source: "EnvironmentalDataSource",
        terrain,
        current_time: float,
        ensemble_result=None,
        latitude: float = 37.5,
    ) -> None:
        """
        Derive all prior means for one cycle.

        Steps
        -----
        1. Query `source` for all supported measurement types and register them.
        2. Apply each registered Measurement to update intermediate weather /
           wind fields.
        3. Compute solar radiation from terrain + time.
        4. Compute Nelson FMC from weather + solar + terrain.
        5. Update fire state from ensemble (if available).

        The raw inputs are stored in `self.inputs` after the call.
        """
        self.timestamp = current_time
        self.inputs = []  # clear previous cycle's inputs

        # 1. Query source
        weather_m = source.get_weather(current_time)
        wind_m    = source.get_wind(current_time)
        sat_fmc_m = source.get_satellite_fmc(current_time)

        for m in (weather_m, wind_m, sat_fmc_m):
            if m is not None:
                self.inputs.append(m)

        # Record which source provided data this cycle
        if self.inputs:
            self.last_source = self.inputs[0].source

        # 2. Apply each measurement
        for m in self.inputs:
            if isinstance(m, NWPWeatherMeasurement):
                self._apply_weather(m)
            elif isinstance(m, NWPWindMeasurement):
                self._apply_wind(m)
            elif isinstance(m, SatelliteFMCMeasurement):
                self._apply_satellite_fmc(m)

        # 3. Solar radiation (time + terrain — no measurement needed)
        hour_local = (6.0 + current_time / 3600.0) % 24.0
        self._compute_solar(terrain, hour_local, latitude)

        # 4. Nelson FMC
        self._compute_nelson(terrain)

        # 5. Fire state from ensemble
        if ensemble_result is not None:
            self._compute_fire_state(ensemble_result, current_time)

    # ----------------------------------------------------------------
    # Private: measurement application
    # ----------------------------------------------------------------

    def _apply_weather(self, m: NWPWeatherMeasurement) -> None:
        rows, cols = self.grid_shape
        self.temperature = np.broadcast_to(
            np.asarray(m.temperature_c, dtype=np.float32), (rows, cols)
        ).copy()
        self.humidity = np.broadcast_to(
            np.asarray(m.relative_humidity, dtype=np.float32), (rows, cols)
        ).copy()

    def _apply_wind(self, m: NWPWindMeasurement) -> None:
        rows, cols = self.grid_shape
        ws = np.broadcast_to(
            np.asarray(m.wind_speed, dtype=np.float32), (rows, cols)
        ).copy()
        wd = np.broadcast_to(
            np.asarray(m.wind_direction, dtype=np.float32), (rows, cols)
        ).copy()
        self.wind_speed_prior    = np.clip(ws, 0.1, 50.0).astype(np.float32)
        self.wind_direction_prior = (wd % 360.0).astype(np.float32)

    def _apply_satellite_fmc(self, m: SatelliteFMCMeasurement) -> None:
        """Confidence-weighted blend with existing Nelson FMC field."""
        mask = m.confidence > 0.3
        if self.nelson_fmc is None:
            self.nelson_fmc = np.where(mask, m.fmc, np.nan).astype(np.float32)
        else:
            self.nelson_fmc = np.where(
                mask,
                m.confidence * m.fmc + (1.0 - m.confidence) * self.nelson_fmc,
                self.nelson_fmc,
            ).astype(np.float32)

    # ----------------------------------------------------------------
    # Private: derived field computation
    # ----------------------------------------------------------------

    def _compute_solar(
        self, terrain, hour_local: float, latitude: float = 37.5
    ) -> None:
        solar_elevation = max(0.0, 90.0 - abs(hour_local - 12.0) * 15.0)
        if solar_elevation <= 0.0:
            self.solar_radiation = np.zeros(self.grid_shape, dtype=np.float32)
            return

        solar_elevation_rad = np.radians(solar_elevation)
        base_radiation = 1000.0 * np.sin(solar_elevation_rad)

        aspect_rad = np.radians(terrain.aspect)
        slope_rad  = np.radians(terrain.slope)
        cos_incidence = np.clip(
            np.sin(solar_elevation_rad) * np.cos(slope_rad)
            + np.cos(solar_elevation_rad)
            * np.sin(slope_rad)
            * np.cos(aspect_rad - np.pi),
            0.0, 1.0,
        )
        canopy_transmission = (
            1.0 - 0.7 * terrain.canopy_cover
            if terrain.canopy_cover is not None
            else np.ones(self.grid_shape, dtype=np.float32)
        )
        self.solar_radiation = (
            base_radiation * cos_incidence * canopy_transmission
        ).astype(np.float32)

    def _compute_nelson(self, terrain) -> None:
        """Fosberg-Deeming three-segment EMC → Nelson FMC prior."""
        if self.temperature is None or self.humidity is None:
            return

        rh_pct = self.humidity * 100.0
        T = self.temperature
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

        if self.solar_radiation is not None:
            solar_factor = np.clip(self.solar_radiation / 1000.0, 0.0, 1.0)
            emc_fraction -= 0.03 * solar_factor

        if terrain.canopy_cover is not None:
            emc_fraction += 0.02 * terrain.canopy_cover

        if terrain.elevation is not None:
            elev_norm = (terrain.elevation - terrain.elevation.mean()) / (
                terrain.elevation.std() + 1e-6
            )
            emc_fraction += 0.015 * np.clip(elev_norm, -2.0, 2.0)

        nelson = np.clip(emc_fraction, 0.02, 0.40).astype(np.float32)

        # If satellite FMC was already blended in, preserve it; otherwise replace.
        self.nelson_fmc = nelson

    def _compute_fire_state(
        self,
        ensemble_result,
        current_time: float,
        last_observed: Optional[np.ndarray] = None,
    ) -> None:
        """Update fire state statistics from the previous cycle's ensemble."""
        arrival_min = ensemble_result.member_arrival_times  # float32[N, rows, cols]
        current_time_min = current_time / 60.0

        self.fire_burn_probability = (
            (arrival_min < current_time_min).mean(axis=0).astype(np.float32)
        )
        self.fire_arrival_time = (
            np.median(arrival_min, axis=0) * 60.0
        ).astype(np.float32)
        arrival_std_s = arrival_min.std(axis=0) * 60.0
        if last_observed is not None:
            time_since_obs = np.clip(current_time - last_observed, 0.0, 14400.0)
            arrival_std_s = np.sqrt(arrival_std_s ** 2 + time_since_obs ** 2)
        self.fire_uncertainty = arrival_std_s.astype(np.float32)

    # ----------------------------------------------------------------
    # GP interface
    # ----------------------------------------------------------------

    def get_gp_prior_means(self) -> dict[str, Optional[np.ndarray]]:
        """
        Return the prior mean fields the GP corrects around.

        The GP fits residuals against these — observations that agree
        produce near-zero correction; disagreeing observations produce
        corrections that decay away with the kernel length scale.
        Returns None for fields not yet populated.
        """
        return {
            "fmc":            self.nelson_fmc,
            "wind_speed":     self.wind_speed_prior,
            "wind_direction": self.wind_direction_prior,
        }

    def get_weather_fields(self) -> dict[str, Optional[np.ndarray]]:
        """Intermediate weather fields — used if drone T/RH obs trigger local Nelson recomputation."""
        return {
            "temperature":    self.temperature,
            "humidity":       self.humidity,
            "solar_radiation": self.solar_radiation,
        }

    # ----------------------------------------------------------------
    # Convenience shims (backward compat for direct calls in tests)
    # ----------------------------------------------------------------

    def update_weather(
        self,
        temperature: "float | np.ndarray",
        humidity: "float | np.ndarray",
        source: str = "scenario",
        timestamp: Optional[float] = None,
    ) -> None:
        """Convenience method: directly set weather without a DataSource."""
        m = NWPWeatherMeasurement(
            source=source,
            timestamp=timestamp if timestamp is not None else self.timestamp,
            temperature_c=temperature,
            relative_humidity=humidity,
        )
        self._apply_weather(m)
        self.last_source = source

    def update_wind(
        self,
        wind_speed: "float | np.ndarray",
        wind_direction: "float | np.ndarray",
        source: str = "scenario",
        timestamp: Optional[float] = None,
    ) -> None:
        """Convenience method: directly set wind without a DataSource."""
        m = NWPWindMeasurement(
            source=source,
            timestamp=timestamp if timestamp is not None else self.timestamp,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
        )
        self._apply_wind(m)
        self.last_source = source

    def update_solar(
        self,
        terrain,
        hour_local: float,
        latitude: float = 37.5,
    ) -> None:
        """Convenience shim: compute solar radiation directly."""
        self._compute_solar(terrain, hour_local, latitude)

    def recompute_nelson(self, terrain) -> None:
        """Convenience shim: recompute Nelson FMC from current T/RH/solar."""
        self._compute_nelson(terrain)

    def update_fire_state(
        self,
        ensemble_arrival_times: np.ndarray,
        current_time: float,
        last_observed: Optional[np.ndarray] = None,
        max_ros: float = 5.0,
    ) -> None:
        """Convenience shim: update fire state from raw arrival times array."""
        from angrybird.types import EnsembleResult
        import dataclasses

        n, r, c = ensemble_arrival_times.shape
        zeros = np.zeros((r, c), dtype=np.float32)
        ens = EnsembleResult(
            member_arrival_times=ensemble_arrival_times,
            member_fmc_fields=np.zeros((n, r, c), dtype=np.float32),
            member_wind_fields=np.zeros((n, r, c), dtype=np.float32),
            burn_probability=zeros,
            mean_arrival_time=zeros,
            arrival_time_variance=zeros,
            n_members=n,
        )
        self._compute_fire_state(ens, current_time, last_observed)

    # ----------------------------------------------------------------
    # Legacy entry point (kept so existing orchestrator tests still pass)
    # ----------------------------------------------------------------

    def update_cycle(
        self,
        current_time: float,
        terrain,
        weather_source: Optional[dict] = None,
        ensemble_result=None,
        fire_observations_last_observed=None,
    ) -> None:
        """
        Legacy entry point.  Prefer compute_cycle(source, terrain, t).

        Converts the weather_source dict to a StaticDataSource and delegates
        to compute_cycle so existing callers keep working.
        """
        from .sources import StaticDataSource

        if weather_source is not None:
            source = StaticDataSource(
                temperature_c=float(
                    np.mean(weather_source["temperature"])
                    if "temperature" in weather_source
                    else 30.0
                ),
                relative_humidity=float(
                    np.mean(weather_source["humidity"])
                    if "humidity" in weather_source
                    else 0.25
                ),
                wind_speed=float(
                    np.mean(weather_source["wind_speed"])
                    if "wind_speed" in weather_source
                    else 5.0
                ),
                wind_direction=float(
                    np.mean(weather_source["wind_direction"])
                    if "wind_direction" in weather_source
                    else 270.0
                ),
                source_label=weather_source.get("source", "scenario"),
            )
            # For array wind fields passed in legacy dict, apply directly
            if "wind_speed" in weather_source and isinstance(
                weather_source["wind_speed"], np.ndarray
            ):
                self._apply_wind(
                    NWPWindMeasurement(
                        source=weather_source.get("source", "scenario"),
                        timestamp=current_time,
                        wind_speed=weather_source["wind_speed"],
                        wind_direction=weather_source.get("wind_direction", 270.0),
                    )
                )
                # Don't query wind from the StaticDataSource shim
                _skip_wind = True
            else:
                _skip_wind = False

<<<<<<< HEAD
    def compute_cycle(
        self,
        source,
        terrain,
        current_time: float,
        ensemble_result=None,
    ) -> None:
        """
        Update dynamic prior from an EnvironmentalDataSource object.
        Delegates to update_cycle using weather data provided by the source.
        """
        weather = source.get_weather_at(current_time) if hasattr(source, "get_weather_at") else None
        self.update_cycle(
            current_time=current_time,
            terrain=terrain,
            weather_source=weather,
            ensemble_result=ensemble_result,
        )

    # ----------------------------------------------------------------
    # GP interface
    # ----------------------------------------------------------------
=======
            self.timestamp = current_time
            self.inputs = []
>>>>>>> 601e2e99cfdfdb5a77a8ea8f791078c1c00159e9

            weather_m = source.get_weather(current_time)
            if weather_m:
                self.inputs.append(weather_m)
                self._apply_weather(weather_m)

            if not _skip_wind:
                wind_m = source.get_wind(current_time)
                if wind_m:
                    self.inputs.append(wind_m)
                    self._apply_wind(wind_m)

            lat = float(weather_source.get("latitude", 37.5))
            if "hour_local" in weather_source:
                hour_local = float(weather_source["hour_local"])
            else:
                hour_local = (6.0 + current_time / 3600.0) % 24.0
            self._compute_solar(terrain, hour_local, lat)
            self._compute_nelson(terrain)
            self.last_source = weather_source.get("source", "scenario")
        else:
            self.timestamp = current_time

        if ensemble_result is not None:
            self._compute_fire_state(ensemble_result, current_time)
