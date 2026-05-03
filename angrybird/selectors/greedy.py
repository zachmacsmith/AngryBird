"""
Greedy selector: iterative highest-value selection with GP variance update.

Each iteration:
  1. Find the cell with highest current w (respecting min-spacing)
  2. Update the GP posterior variance as if that cell were observed
  3. Recompute w from the updated variance (sensitivity unchanged)
  4. Repeat until K locations selected

Theoretical guarantee: ≥ (1 - 1/e) ≈ 63% of the mutual-information optimum
(Krause et al. 2008 — submodular maximisation).
"""

from __future__ import annotations

import time

import numpy as np

from ..config import GRID_RESOLUTION_M, MIN_SELECTION_SPACING_M, SENSOR_FMC_R2, SENSOR_WIND_ACCURACY
from ..gp import IGNISGPPrior
from ..information import compute_observability
from ..types import EnsembleResult, InformationField, SelectionResult
from .base import spacing_mask


class GreedySelector:
    name = "greedy"

    def __init__(
        self,
        min_spacing_m: float = MIN_SELECTION_SPACING_M,
        resolution_m: float = GRID_RESOLUTION_M,
    ) -> None:
        self.min_spacing_m = min_spacing_m
        self.resolution_m = resolution_m

    def select(
        self,
        info_field: InformationField,
        gp: IGNISGPPrior,
        ensemble: EnsembleResult,
        k: int,
    ) -> SelectionResult:
        t0 = time.perf_counter()
        shape = info_field.w.shape
        min_cells = self.min_spacing_m / self.resolution_m

        # We recompute w at each step from the updated variance.
        # Sensitivity and observability are fixed (they don't depend on which
        # cell we measure next, only on the ensemble — which is unchanged).
        sens_fmc  = np.abs(info_field.sensitivity.get("fmc",
                           np.zeros(shape, dtype=np.float32)))
        sens_wind = np.abs(info_field.sensitivity.get("wind_speed",
                           np.zeros(shape, dtype=np.float32)))

        # Recompute observability directly — avoids back-calculating from w_by_variable
        # which is numerically unstable when sensitivity is near zero.
        obs = compute_observability(ensemble, shape, self.resolution_m)
        obs_fmc  = obs["fmc"]
        obs_wind = obs["wind_speed"]

        # Live GP variance fields — updated each iteration
        gp_prior = gp.predict(shape)
        var_fmc  = gp_prior.fmc_variance.copy()
        var_wind = gp_prior.wind_speed_variance.copy()

        selected: list[tuple[int, int]] = []
        marginal_gains: list[float] = []
        cumulative_gain: list[float] = []
        running_total = 0.0

        for _ in range(k):
            # Recompute w from current (updated) variance
            w = (var_fmc  * sens_fmc  * obs_fmc +
                 var_wind * sens_wind * obs_wind).astype(np.float32)

            # Zero out already-excluded zones (spacing + burned cells)
            if selected:
                w[spacing_mask(shape, selected, min_cells)] = 0.0
            w[ensemble.burn_probability > 0.95] = 0.0

            if w.max() <= 0.0:
                break  # no more useful locations

            flat_idx = int(np.argmax(w))
            loc = (int(flat_idx // shape[1]), int(flat_idx % shape[1]))
            gain = float(w[loc])

            selected.append(loc)
            marginal_gains.append(gain)
            running_total += gain
            cumulative_gain.append(running_total)

            # Update GP variance to reflect the information this observation provides
            var_fmc  = gp.conditional_variance(var_fmc,  loc)
            var_wind = gp.conditional_variance(var_wind, loc)

        return SelectionResult(
            selected_locations=selected,
            marginal_gains=marginal_gains,
            cumulative_gain=cumulative_gain,
            strategy_name=self.name,
            compute_time_s=time.perf_counter() - t0,
        )
