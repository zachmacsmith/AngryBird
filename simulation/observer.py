"""
ObservationSource protocol + SimulatedObserver.

Phase 4b — simulation harness only.

The ObservationSource protocol defines the interface any observation source must
implement. In production, real UAV telemetry implements this. In the simulation
harness, SimulatedObserver samples from a hidden GroundTruth + calibrated noise.

Noise parameters reflect published UAV sensor accuracy:
  FMC:        σ ≈ 0.05  (multispectral R² ≈ 0.86, Architecture §3.8)
  Wind speed: σ ≈ 1.0 m/s
  Wind dir:   σ ≈ 10°
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from angrybird.config import OBS_FMC_SIGMA, OBS_WIND_DIR_SIGMA, OBS_WIND_SPEED_SIGMA
from angrybird.types import DroneObservation
from .ground_truth import GroundTruth


@runtime_checkable
class ObservationSource(Protocol):
    """Interface any observation source must satisfy (real or simulated)."""

    def observe(self, cells: list[tuple[int, int]]) -> list[DroneObservation]: ...


class SimulatedObserver:
    """
    Generates synthetic drone observations by sampling GroundTruth + sensor noise.

    The noise model is fixed per observer instance so noise parameters can be
    varied to study sensor quality effects on strategy performance.
    """

    def __init__(
        self,
        ground_truth: GroundTruth,
        fmc_sigma: float = OBS_FMC_SIGMA,
        wind_speed_sigma: float = OBS_WIND_SPEED_SIGMA,
        wind_dir_sigma: float = OBS_WIND_DIR_SIGMA,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.truth            = ground_truth
        self.fmc_sigma        = fmc_sigma
        self.wind_speed_sigma = wind_speed_sigma
        self.wind_dir_sigma   = wind_dir_sigma
        self._rng             = rng or np.random.default_rng()

    def observe(self, cells: list[tuple[int, int]]) -> list[DroneObservation]:
        """
        Sample one DroneObservation per cell with additive Gaussian sensor noise.
        Wind speed is clipped to ≥ 0; wind direction is wrapped to [0, 360).
        """
        observations: list[DroneObservation] = []
        for r, c in cells:
            true_fmc = float(self.truth.fmc[r, c])
            true_ws  = float(self.truth.wind_speed[r, c])
            true_wd  = float(self.truth.wind_direction[r, c])

            obs_fmc = true_fmc + float(self._rng.normal(0.0, self.fmc_sigma))
            obs_ws  = max(0.0, true_ws + float(self._rng.normal(0.0, self.wind_speed_sigma)))
            obs_wd  = (true_wd + float(self._rng.normal(0.0, self.wind_dir_sigma))) % 360.0

            observations.append(DroneObservation(
                location=(r, c),
                fmc=obs_fmc,
                fmc_sigma=self.fmc_sigma,
                wind_speed=obs_ws,
                wind_speed_sigma=self.wind_speed_sigma,
                wind_dir=obs_wd,
                wind_dir_sigma=self.wind_dir_sigma,
            ))
        return observations
