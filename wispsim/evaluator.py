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

from angrybird.gp import IGNISGPPrior
from angrybird.types import (
    DronePlan,
    EnsembleResult,
    GPPrior,
    InformationField,
    SelectionResult,
    StrategyEvaluation,
)
from .observer import ObservationSource

logger = logging.getLogger(__name__)


def compute_arrival_accuracy(
    ensemble: EnsembleResult,
    truth_arrival_s: np.ndarray,
    horizon_s: float,
) -> tuple[float, float, float]:
    """
    Compare ensemble arrival time predictions to oracle ground truth.

    Only evaluates cells that the ground truth fire actually burns within the
    planning horizon — cells with truth_arrival_s >= horizon_s are excluded
    (fire hasn't reached there yet, so prediction error is not meaningful).

    Ensemble sentinels (2×horizon_min for unburned members) are kept as-is:
    if the ensemble thinks a cell won't burn but the truth says it will, those
    large sentinel values correctly penalise the score.

    Returns
    -------
    crps_minutes : float
        Continuous Ranked Probability Score averaged over burned cells (minutes).
        Proper scoring rule — rewards accuracy *and* calibration simultaneously.
        Lower is better.
    rmse_minutes : float
        RMSE of ensemble mean_arrival_time vs truth over burned cells (minutes).
        Lower is better.
    spread_skill_ratio : float
        sqrt(mean ensemble variance) / RMSE.  ~1.0 means the spread correctly
        predicts the magnitude of errors; >1 = over-dispersed, <1 = over-confident.
    """
    burned_mask = (truth_arrival_s < horizon_s) & np.isfinite(truth_arrival_s)
    if not burned_mask.any():
        return 0.0, 0.0, 1.0

    truth_min = (truth_arrival_s[burned_mask] / 60.0).astype(np.float64)

    # members_min: float64[N, n_burned]
    members_min = ensemble.member_arrival_times[:, burned_mask].astype(np.float64)
    N = members_min.shape[0]

    # ── CRPS via the sorted-member identity ──────────────────────────────────
    # CRPS(F,y) = E[|X-y|] - ½ E[|X-X'|]
    # With N ensemble members:
    #   E[|X-y|]   = (1/N) Σ_i |x_i - y|
    #   ½ E[|X-X'|] = (1/N²) Σ_{i=0}^{N-1} x_(i) * (2i - N + 1)   (x sorted)
    term1 = np.mean(np.abs(members_min - truth_min[np.newaxis, :]), axis=0)

    sorted_m = np.sort(members_min, axis=0)
    ranks = np.arange(N, dtype=np.float64)[:, np.newaxis]          # [N, 1]
    term2 = np.sum(sorted_m * (2.0 * ranks - N + 1.0), axis=0) / (N * N)

    crps = float(np.mean(term1 - term2))

    # ── RMSE of ensemble mean ─────────────────────────────────────────────────
    mean_min = ensemble.mean_arrival_time[burned_mask].astype(np.float64)
    rmse = float(np.sqrt(np.mean((mean_min - truth_min) ** 2)))

    # ── Spread-skill ratio ────────────────────────────────────────────────────
    spread = float(np.sqrt(np.mean(ensemble.arrival_time_variance[burned_mask])))
    spread_skill = spread / (rmse + 1e-6)

    return crps, rmse, spread_skill


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


def compute_arrival_accuracy(
    ensemble: EnsembleResult,
    truth_arrival_s: np.ndarray,
    horizon_s: float,
) -> tuple[float, float, float]:
    """
    Compute CRPS, RMSE, and spread-skill ratio for ensemble arrival-time forecasts.

    Parameters
    ----------
    ensemble       : EnsembleResult — member_arrival_times in minutes
    truth_arrival_s: (rows, cols) ground-truth arrival times in seconds
    horizon_s      : planning horizon in seconds (unburned sentinel threshold)

    Returns
    -------
    (crps_minutes, rmse_minutes, spread_skill_ratio)
    Returns (0.0, 0.0, 0.0) when no ground-truth cells have burned yet.
    """
    burned_truth = truth_arrival_s < horizon_s
    if not burned_truth.any():
        return 0.0, 0.0, 0.0

    truth_min = truth_arrival_s[burned_truth] / 60.0          # (n_burned,)
    at_pred   = ensemble.member_arrival_times[:, burned_truth] # (N, n_burned)
    N         = ensemble.n_members

    # --- CRPS (fair energy-score form) ---
    # CRPS = E|X - y| - ½ E|X - X'|
    # Efficient O(N log N) computation via sorted ensemble:
    #   E|X - X'| = (2/N²) Σ_i x_{(i)} * (2i - N + 1)   (i = 0 … N-1, sorted)
    abs_err    = np.abs(at_pred - truth_min[None, :]).mean(axis=0)       # (n_burned,)
    sorted_p   = np.sort(at_pred, axis=0)                                # (N, n_burned)
    weights    = (2 * np.arange(N, dtype=np.float32) - N + 1)           # (N,)
    spread_crps = (sorted_p * weights[:, None]).sum(axis=0) / (N * N)   # (n_burned,)
    crps_min   = float(np.mean(abs_err - spread_crps))

    # --- RMSE of ensemble mean ---
    mean_pred = at_pred.mean(axis=0)
    rmse_min  = float(np.sqrt(np.mean((mean_pred - truth_min) ** 2)))

    # --- Spread-skill ratio ---
    spread    = float(at_pred.std(axis=0).mean())
    spread_skill = spread / max(rmse_min, 1e-10)

    return max(crps_min, 0.0), rmse_min, spread_skill
