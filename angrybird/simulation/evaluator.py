"""
Counterfactual evaluator: measures information gain for each selection strategy.

Phase 4b — simulation harness only.

For each strategy's drone plans:
  1. Collect the union of all cells those drones would overfly
  2. Simulate what observations they would have collected (from GroundTruth + noise)
  3. Compute the hypothetical GP posterior variance after those observations
     using conditional_variance — no GP refit, no state mutation
  4. Recompute the information field w with updated variance
     (sensitivity and observability are fixed — they only depend on the ensemble)
  5. Entropy reduction  = w_before.sum() - w_after.sum()
     PERR               = entropy_reduction / n_drones

Only the primary strategy's observations ever feed into the real GP update.
Counterfactual evaluation is read-only: it never mutates gp, ensemble, or
anything else. (Architecture §3.6)
"""

from __future__ import annotations

import logging

import numpy as np

from ..gp import IGNISGPPrior
from ..types import (
    DronePlan,
    GPPrior,
    InformationField,
    SelectionResult,
    StrategyEvaluation,
)
from .observer import ObservationSource

logger = logging.getLogger(__name__)


class CounterfactualEvaluator:
    """
    Evaluates all strategies on equal footing without mutating any shared state.
    """

    def __init__(
        self,
        obs_source: ObservationSource,
        gp: IGNISGPPrior,
    ) -> None:
        self.obs_source = obs_source
        self.gp         = gp

    def evaluate(
        self,
        strategy_name: str,
        selection_result: SelectionResult,
        drone_plans: list[DronePlan],
        info_field: InformationField,
        gp_prior: GPPrior,
        n_drones: int,
    ) -> StrategyEvaluation:
        """
        Simulate what would have happened if this strategy's drones flew their paths.

        The GP is queried via conditional_variance (closed-form, O(D) per call)
        and is NEVER modified. The updated variance is used purely for metric
        computation and then discarded.

        Note: conditional_variance uses the FMC kernel for both FMC and wind
        speed updates. This is a reasonable approximation at hackathon scale —
        the kernel length scales differ but the decay shape is identical.
        """
        # Union of all cells observed across all drone plans
        seen: set[tuple[int, int]] = set()
        all_cells: list[tuple[int, int]] = []
        for plan in drone_plans:
            for cell in plan.cells_observed:
                if cell not in seen:
                    all_cells.append(cell)
                    seen.add(cell)

        entropy_before = float(info_field.w.sum())

        if not all_cells:
            return StrategyEvaluation(
                strategy_name=strategy_name,
                selected_locations=selection_result.selected_locations,
                entropy_before=entropy_before,
                entropy_after=entropy_before,
                entropy_reduction=0.0,
                perr=0.0,
                cells_observed=[],
            )

        # Counterfactual observations (read-only — does not feed into real state)
        simulated_obs = self.obs_source.observe(all_cells)

        # Propagate hypothetical variance reduction via conditional_variance
        # Work on copies — never touch the real GP prior arrays
        var_fmc  = gp_prior.fmc_variance.copy()
        var_wind = gp_prior.wind_speed_variance.copy()
        for obs in simulated_obs:
            var_fmc  = self.gp.conditional_variance(var_fmc,  obs.location, obs.fmc_sigma)
            var_wind = self.gp.conditional_variance(var_wind, obs.location, obs.wind_speed_sigma)

        # Recompute w with updated variance:
        #   w_after = w_before * (var_after / var_before)
        # This factors out sensitivity × observability — they are invariant to
        # which cells we measure (they depend on the ensemble, not the GP).
        shape = gp_prior.fmc_variance.shape
        orig_var_fmc  = info_field.gp_variance.get("fmc",        np.ones(shape, dtype=np.float32))
        orig_var_wind = info_field.gp_variance.get("wind_speed", np.ones(shape, dtype=np.float32))
        orig_w_fmc    = info_field.w_by_variable.get("fmc",        np.zeros(shape, dtype=np.float32))
        orig_w_wind   = info_field.w_by_variable.get("wind_speed", np.zeros(shape, dtype=np.float32))

        safe_fmc  = np.where(orig_var_fmc  > 1e-12, orig_var_fmc,  1e-12)
        safe_wind = np.where(orig_var_wind > 1e-12, orig_var_wind, 1e-12)

        w_fmc_after  = (orig_w_fmc  * var_fmc  / safe_fmc ).astype(np.float32)
        w_wind_after = (orig_w_wind * var_wind / safe_wind).astype(np.float32)
        w_after = w_fmc_after + w_wind_after

        entropy_after = float(w_after.sum())
        reduction     = max(0.0, entropy_before - entropy_after)
        perr          = reduction / max(n_drones, 1)

        logger.debug(
            "Counterfactual [%s]: %d cells, entropy %.4f → %.4f (Δ=%.4f, PERR=%.4f)",
            strategy_name, len(all_cells),
            entropy_before, entropy_after, reduction, perr,
        )
        return StrategyEvaluation(
            strategy_name=strategy_name,
            selected_locations=selection_result.selected_locations,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            entropy_reduction=reduction,
            perr=perr,
            cells_observed=all_cells,
        )
