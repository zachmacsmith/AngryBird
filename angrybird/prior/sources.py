"""
Environmental data source protocol and built-in implementations.

The DynamicPrior does not know or care where its inputs come from — it only
knows how to compute a prior from typed Measurement objects.  This module
defines the source side: a Protocol that any provider must satisfy, plus the
two built-in implementations you can swap between.

To add a new source (e.g. HRRR API, GOES satellite, RAWS network):
  1. Create a class that satisfies EnvironmentalDataSource.
  2. Implement get_weather / get_wind / get_satellite_fmc.
  3. Return None from any method you don't support yet.
  4. Pass an instance to IGNISOrchestrator(data_source=...) at construction.

Spec gaps (not yet implemented)
--------------------------------
The following items are described in the observation-types spec or sensor docs
but differ from the current implementation:

1. DroneWeatherObservation (T/RH per cell) — not in observations.py.
2. DroneThermalObservation (forward-looking fire swath) — not in observations.py.
3. DroneLiDARObservation — no terrain-update pipeline to consume it.
4. DroneGasObservation — not in observations.py.
5. RAWSObservation missing T/RH DataPoints — spec says RAWS emits T and RH;
   implementation only stores FMC + wind speed + wind direction.
6. SatelliteFMCObservation missing representativeness error —
   spec: sigma_total = sqrt(sigma_instrument² + sigma_repr²), where
   sigma_repr ≈ 0.5 × sigma_instrument for coarse pixels.
7. GOES/VIIRS as distinct typed classes — spec distinguishes them;
   both currently map to the same FireDetectionObservation.
8. VIIRS confidence tiers ("low"→0.5, "nominal"→0.8, "high"→0.95) —
   not reflected in observations.py.
9. ObservationStore lock/unlock during cycle — store has lock() / unlock()
   methods but orchestrator does not call them around run_cycle(), so
   concurrent ingestion is technically possible.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .measurements import (
    NWPWeatherMeasurement,
    NWPWindMeasurement,
    SatelliteFMCMeasurement,
)


@runtime_checkable
class EnvironmentalDataSource(Protocol):
    """
    Protocol for any source of environmental prior data.

    Each method returns a typed measurement or None if that data type is not
    available from this source.  The DynamicPrior skips None measurements
    and leaves the corresponding prior field unchanged from its last value.
    """

    def get_weather(self, timestamp: float) -> Optional[NWPWeatherMeasurement]:
        """Temperature + RH valid at `timestamp` (seconds)."""
        ...

    def get_wind(self, timestamp: float) -> Optional[NWPWindMeasurement]:
        """Wind speed + direction valid at `timestamp` (seconds)."""
        ...

    def get_satellite_fmc(self, timestamp: float) -> Optional[SatelliteFMCMeasurement]:
        """Satellite FMC retrieval valid at or before `timestamp` (seconds)."""
        ...


class GroundTruthDataSource:
    """
    Simulation data source — reads directly from the oracle GroundTruth.

    Represents a "perfect" NWP forecast with no error.  Used by
    SimulationRunner so that the dynamic prior is driven by the actual
    scenario conditions rather than hardcoded defaults.

    To simulate a realistic (imperfect) NWP forecast, replace with a
    NoisyGroundTruthDataSource that adds bias/noise to the returned fields.
    """

    def __init__(self, ground_truth, source_label: str = "ground_truth") -> None:
        self._truth = ground_truth
        self._label = source_label

    def get_weather(self, timestamp: float) -> NWPWeatherMeasurement:
        return NWPWeatherMeasurement(
            source=self._label,
            timestamp=timestamp,
            temperature_c=self._truth.temperature_c,
            relative_humidity=self._truth.relative_humidity,
        )

    def get_wind(self, timestamp: float) -> NWPWindMeasurement:
        return NWPWindMeasurement(
            source=self._label,
            timestamp=timestamp,
            wind_speed=self._truth.wind_speed.copy(),
            wind_direction=self._truth.wind_direction.copy(),
        )

    def get_satellite_fmc(self, timestamp: float) -> None:
        return None  # no satellite in simulation


class StaticDataSource:
    """
    Constant-value data source — hardcoded scalars for all fields.

    Used for tests, offline development, and as a safe fallback when no
    real data connection is available.  All fields return the same value
    on every call regardless of timestamp.
    """

    def __init__(
        self,
        temperature_c: float = 30.0,
        relative_humidity: float = 0.25,
        wind_speed: float = 5.0,
        wind_direction: float = 270.0,
        source_label: str = "static",
    ) -> None:
        self._t    = temperature_c
        self._rh   = relative_humidity
        self._ws   = wind_speed
        self._wd   = wind_direction
        self._label = source_label

    def get_weather(self, timestamp: float) -> NWPWeatherMeasurement:
        return NWPWeatherMeasurement(
            source=self._label,
            timestamp=timestamp,
            temperature_c=self._t,
            relative_humidity=self._rh,
        )

    def get_wind(self, timestamp: float) -> NWPWindMeasurement:
        return NWPWindMeasurement(
            source=self._label,
            timestamp=timestamp,
            wind_speed=self._ws,
            wind_direction=self._wd,
        )

    def get_satellite_fmc(self, timestamp: float) -> None:
        return None


class SimulatedEnvironmentalSource:
    """
    Simulated sensor source for WispSim.

    Produces frequency-gated, noisy measurements derived from the oracle
    GroundTruth.  Designed to mimic a realistic mix of NWP model output and
    satellite-derived products without requiring external data connections.

    Three measurement channels feed the DynamicPrior (called by compute_cycle):
      • get_weather  — NWP T/RH: hourly scalar regional mean + Gaussian noise
      • get_wind     — NWP/GOES-AMV wind: every 5 min, circular mean + noise
      • get_satellite_fmc — satellite FMC: daily, full noisy field

    Two additional channels feed the GP ObservationStore directly via the
    separate collect_obs_store_inputs() call in the runner:
      • GOES fire detection: every 5 min, 2 km pixels (40 cells), detect_prob=0.90
      • VIIRS fire detection: every 12 hr, 375 m pixels (7 cells), detect_prob=0.95
      • Satellite FMC point obs: daily, 500 m thinned grid (10-cell stride)

    Each channel is frequency-gated: calls arriving before the interval has
    elapsed return None (prior channels) or an empty list (store channels).
    """

    # Sensor update intervals (seconds)
    WEATHER_INTERVAL_S:    float = 3_600.0    # NWP weather: hourly
    WIND_INTERVAL_S:       float = 300.0      # NWP/GOES-AMV wind: every 5 min
    SAT_FMC_INTERVAL_S:    float = 86_400.0   # Satellite FMC prior: daily
    GOES_FIRE_INTERVAL_S:  float = 300.0      # GOES fire detect: every 5 min
    VIIRS_FIRE_INTERVAL_S: float = 43_200.0   # VIIRS fire detect: every 12 hr
    SAT_OBS_INTERVAL_S:    float = 86_400.0   # Satellite FMC GP obs: daily

    # Sensor spatial resolutions in grid cells (grid assumed 50 m)
    GOES_PIXEL_CELLS:  int = 40   # 2 000 m / 50 m
    VIIRS_PIXEL_CELLS: int = 7    # 375 m / 50 m (rounded)
    SAT_OBS_STRIDE:    int = 10   # 500 m / 50 m FMC point-obs thinning

    def __init__(
        self,
        ground_truth,
        source_label: str = "simulated",
        rng: Optional[np.random.Generator] = None,
        weather_t_sigma: float = 2.0,
        weather_rh_sigma: float = 0.05,
        wind_speed_sigma: float = 1.5,
        wind_dir_sigma_deg: float = 15.0,
        sat_fmc_sigma: float = 0.12,
        goes_confidence: float = 0.85,
        goes_detect_prob: float = 0.90,
        viirs_confidence: float = 0.90,
        viirs_detect_prob: float = 0.95,
    ) -> None:
        self._truth = ground_truth
        self._label = source_label
        self._rng   = rng if rng is not None else np.random.default_rng(42)

        self._weather_t_sigma   = weather_t_sigma
        self._weather_rh_sigma  = weather_rh_sigma
        self._wind_speed_sigma  = wind_speed_sigma
        self._wind_dir_sigma    = np.deg2rad(wind_dir_sigma_deg)
        self._sat_fmc_sigma     = sat_fmc_sigma
        self._goes_confidence   = goes_confidence
        self._goes_detect_prob  = goes_detect_prob
        self._viirs_confidence  = viirs_confidence
        self._viirs_detect_prob = viirs_detect_prob

        # Frequency-gate timers — initialised to -inf so first call fires immediately
        self._last_weather_time:  float = -np.inf
        self._last_wind_time:     float = -np.inf
        self._last_sat_fmc_time:  float = -np.inf
        self._last_goes_time:     float = -np.inf
        self._last_viirs_time:    float = -np.inf
        self._last_sat_obs_time:  float = -np.inf

    # ------------------------------------------------------------------
    # EnvironmentalDataSource protocol — feeds DynamicPrior
    # ------------------------------------------------------------------

    def get_weather(self, timestamp: float) -> Optional[NWPWeatherMeasurement]:
        if timestamp - self._last_weather_time < self.WEATHER_INTERVAL_S:
            return None
        self._last_weather_time = timestamp
        t  = self._truth.temperature_c  + self._rng.normal(0.0, self._weather_t_sigma)
        rh = self._truth.relative_humidity + self._rng.normal(0.0, self._weather_rh_sigma)
        return NWPWeatherMeasurement(
            source=self._label,
            timestamp=timestamp,
            temperature_c=float(t),
            relative_humidity=float(np.clip(rh, 0.01, 0.99)),
        )

    def get_wind(self, timestamp: float) -> Optional[NWPWindMeasurement]:
        if timestamp - self._last_wind_time < self.WIND_INTERVAL_S:
            return None
        self._last_wind_time = timestamp
        ws_mean = float(np.mean(self._truth.wind_speed))
        wd_rad   = np.deg2rad(self._truth.wind_direction)
        wd_mean  = float(np.rad2deg(
            np.arctan2(np.mean(np.sin(wd_rad)), np.mean(np.cos(wd_rad)))
        )) % 360.0
        ws = float(np.clip(ws_mean + self._rng.normal(0.0, self._wind_speed_sigma), 0.1, None))
        wd = float((wd_mean + np.rad2deg(self._rng.normal(0.0, self._wind_dir_sigma))) % 360.0)
        return NWPWindMeasurement(
            source=self._label,
            timestamp=timestamp,
            wind_speed=ws,
            wind_direction=wd,
        )

    def get_satellite_fmc(self, timestamp: float) -> Optional[SatelliteFMCMeasurement]:
        if timestamp - self._last_sat_fmc_time < self.SAT_FMC_INTERVAL_S:
            return None
        self._last_sat_fmc_time = timestamp
        fmc   = self._truth.fmc.astype(np.float32)
        noise = self._rng.normal(0.0, self._sat_fmc_sigma, size=fmc.shape).astype(np.float32)
        return SatelliteFMCMeasurement(
            source=self._label,
            timestamp=timestamp,
            fmc=np.clip(fmc + noise, 0.01, None),
            confidence=np.full(fmc.shape, 0.7, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # GP ObservationStore inputs — fire detections + FMC point obs
    # ------------------------------------------------------------------

    def collect_obs_store_inputs(
        self,
        timestamp: float,
        fire_state: np.ndarray,
    ) -> list:
        """
        Generate fire detection and satellite FMC observations for the GP store.

        Returns a list of FireDetectionObservation and SatelliteFMCObservation
        ready for ObservationStore.add_batch().  Call this once per WISP cycle
        before orchestrator.run_cycle() so the observations are visible to the
        GP during the upcoming ensemble.

        GOES fire detections  — every 5 min, 2 km pixels (stride=40 cells)
        VIIRS fire detections — every 12 hr, 375 m pixels (stride=7 cells)
        Satellite FMC obs    — daily, 500 m thinned grid (stride=10 cells)
        """
        from angrybird.observations import FireDetectionObservation, SatelliteFMCObservation

        obs = []
        rows, cols = fire_state.shape

        # ---- GOES fire detection (every 5 min) ----
        if timestamp - self._last_goes_time >= self.GOES_FIRE_INTERVAL_S:
            self._last_goes_time = timestamp
            stride = self.GOES_PIXEL_CELLS
            for r in range(0, rows, stride):
                for c in range(0, cols, stride):
                    r_end, c_end = min(r + stride, rows), min(c + stride, cols)
                    if not np.any(fire_state[r:r_end, c:c_end] > 0):
                        continue
                    if self._rng.random() >= self._goes_detect_prob:
                        continue
                    cr, cc = (r + r_end) // 2, (c + c_end) // 2
                    obs.append(FireDetectionObservation(
                        _source_id=f"goes_{timestamp:.0f}_{cr}_{cc}",
                        _timestamp=timestamp,
                        location=(cr, cc),
                        is_fire=True,
                        confidence=self._goes_confidence,
                    ))

        # ---- VIIRS fire detection (every 12 hr) ----
        if timestamp - self._last_viirs_time >= self.VIIRS_FIRE_INTERVAL_S:
            self._last_viirs_time = timestamp
            stride = self.VIIRS_PIXEL_CELLS
            for r in range(0, rows, stride):
                for c in range(0, cols, stride):
                    r_end, c_end = min(r + stride, rows), min(c + stride, cols)
                    if not np.any(fire_state[r:r_end, c:c_end] > 0):
                        continue
                    if self._rng.random() >= self._viirs_detect_prob:
                        continue
                    cr, cc = (r + r_end) // 2, (c + c_end) // 2
                    obs.append(FireDetectionObservation(
                        _source_id=f"viirs_{timestamp:.0f}_{cr}_{cc}",
                        _timestamp=timestamp,
                        location=(cr, cc),
                        is_fire=True,
                        confidence=self._viirs_confidence,
                    ))

        # ---- Satellite FMC GP point observations (daily, 500 m thinned grid) ----
        if timestamp - self._last_sat_obs_time >= self.SAT_OBS_INTERVAL_S:
            self._last_sat_obs_time = timestamp
            stride = self.SAT_OBS_STRIDE
            fmc = self._truth.fmc
            for r in range(0, rows, stride):
                for c in range(0, cols, stride):
                    noisy_fmc = float(fmc[r, c]) + float(
                        self._rng.normal(0.0, self._sat_fmc_sigma)
                    )
                    obs.append(SatelliteFMCObservation(
                        _source_id=f"sat_fmc_{timestamp:.0f}_{r}_{c}",
                        _timestamp=timestamp,
                        center_location=(r, c),
                        footprint_cells=((r, c),),
                        fmc=max(0.01, noisy_fmc),
                        fmc_sigma=self._sat_fmc_sigma,
                    ))

        return obs
