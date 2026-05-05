"""
Static prior baseline evaluator for WISPsim.

Runs the fire engine once at t=0 with the initial data prior (Nelson FMC +
background wind, no observations) and evaluates the frozen ensemble against
ground truth at each cycle boundary.  The gap between this and the full IGNIS
run quantifies the value of the drone observation network.

This is NOT an alternative orchestrator — it has no GP update loop, no drone
simulation, and no information field.  It is a pure post-hoc evaluation over
a frozen ensemble.
"""

from __future__ import annotations

import logging

import numpy as np

from angrybird.types import EnsembleResult, GPPrior
from wispsim.ground_truth import GroundTruth
from wispsim.runner import SimulationConfig

logger = logging.getLogger(__name__)


class StaticPriorEvaluator:
    """
    Evaluate forecast skill of the initial data prior against ground truth.

    Parameters
    ----------
    config             : SimulationConfig — provides cycle_interval_s, total_time_s, scenario_name
    terrain            : TerrainData — passed through to the fire engine
    ground_truth       : GroundTruth — provides arrival_times after the main run completes
    initial_gp_prior   : GPPrior — GP state before any observations (Nelson FMC + default wind)
    fire_engine        : object implementing FireEngineProtocol.run()
    initial_fire_state : np.ndarray — fire CA state at t=0 (fallback if no phi)
    n_members          : int — ensemble size (same as the main IGNIS run)
    horizon_min        : int — fire-spread horizon in minutes (same as main run)
    initial_phi        : np.ndarray or None — phi level-set from orchestrator cycle 1;
                         when provided this is used instead of initial_fire_state so both
                         IGNIS and the static baseline start from the same fire perimeter.
    """

    def __init__(
        self,
        config: SimulationConfig,
        terrain,
        ground_truth: GroundTruth,
        initial_gp_prior: GPPrior,
        fire_engine,
        initial_fire_state: np.ndarray,
        n_members: int,
        horizon_min: int,
        initial_phi: np.ndarray | None = None,
        oracle_arrival_times: np.ndarray | None = None,
    ) -> None:
        self.config               = config
        self.terrain              = terrain
        self.ground_truth         = ground_truth
        self.initial_gp_prior     = initial_gp_prior
        self.fire_engine          = fire_engine
        self.initial_fire_state   = initial_fire_state
        self.n_members            = n_members
        self.horizon_min          = horizon_min
        self.initial_phi          = initial_phi
        # Use pre-computed oracle arrival times when provided so the same cell set
        # is scored as IGNIS uses.  Falls back to ground truth at evaluation time.
        self._oracle_arrival_times = oracle_arrival_times

    # ------------------------------------------------------------------

    def evaluate(self) -> list[dict]:
        """
        Run the fire engine once, then score the frozen ensemble against
        ground truth at each cycle boundary.

        Returns
        -------
        list[dict] — one row per cycle, with the same keys as
        SimulationRunner._cycle_metrics_rows plus "variant"="static_prior".
        """
        logger.info(
            "StaticPriorEvaluator: running %d-member ensemble from t=0 (horizon=%d min)",
            self.n_members, self.horizon_min,
        )
        rng = np.random.default_rng(1)
        _kwargs = {}
        if self.initial_phi is not None:
            _kwargs["initial_phi"] = self.initial_phi
        ensemble: EnsembleResult = self.fire_engine.run(
            self.terrain,
            self.initial_gp_prior,
            self.initial_fire_state.copy(),
            n_members=self.n_members,
            horizon_min=self.horizon_min,
            rng=rng,
            **_kwargs,
        )
        logger.info("StaticPriorEvaluator: ensemble ready — evaluating at cycle boundaries")

        rows: list[dict] = []
        cycle_s   = self.config.ignis_cycle_interval_s
        total_s   = self.config.total_time_s
        horizon_s = self.horizon_min * 60.0
        # Use oracle arrival times if provided; fall back to live ground truth.
        truth_arr = (
            self._oracle_arrival_times
            if self._oracle_arrival_times is not None
            else self.ground_truth.fire.arrival_times
        )

        t         = 0.0
        cycle_idx = 0
        while t <= total_s:
            cycle_idx += 1
            crps_per_cell_min, n_active = self._score_at_time(
                ensemble, truth_arr, horizon_s, current_time_s=t
            )
            rows.append({
                "scenario_name":         self.config.scenario_name,
                "n_drones":              0,
                "cycle":                 cycle_idx,
                "time_min":              round(t / 60.0, 3),
                "n_obs_cumulative":      0,
                "crps_minutes":          round(crps_per_cell_min, 4),
                "crps_per_cell_minutes": round(crps_per_cell_min, 6),
                "n_active_cells":        n_active,
                "oracle_crps_minutes":   0.0,
                "spread_skill":          0.0,
                "gp_var_fmc_mean":       round(float(self.initial_gp_prior.fmc_variance.mean()), 6),
                "gp_var_wind_mean":      round(float(self.initial_gp_prior.wind_speed_variance.mean()), 6),
                "n_burned_cells":        0,
                "burn_fraction":         0.0,
                "variant":               "static_prior",
            })
            t += cycle_s

        return rows

    # ------------------------------------------------------------------

    def _score_at_time(
        self,
        ensemble: EnsembleResult,
        truth_arrival_s: np.ndarray,
        horizon_s: float,
        current_time_s: float,
    ) -> tuple[float, int]:
        """
        Compute per-cell CRPS for the static ensemble evaluated at a given
        simulation time.

        The ensemble was produced at t=0, so its member_arrival_times are in
        minutes from t=0.  At evaluation time T we shift them by -T/60 to put
        them on the same time base as the truth (minutes from T).
        """
        mask = (
            (truth_arrival_s >= current_time_s) &
            ((truth_arrival_s - current_time_s) < horizon_s) &
            np.isfinite(truth_arrival_s)
        )
        n_active = int(mask.sum())
        if not mask.any():
            return 0.0, 0

        # Truth: minutes from current_time_s
        truth_rel_min = ((truth_arrival_s[mask] - current_time_s) / 60.0).astype(np.float64)

        # Ensemble: member_arrival_times are minutes from t=0; shift to be
        # relative to current_time_s so both axes share the same origin.
        at_pred = (
            ensemble.member_arrival_times[:, mask].astype(np.float64)
            - current_time_s / 60.0
        )

        N = ensemble.n_members
        abs_err     = np.abs(at_pred - truth_rel_min[None, :]).mean(axis=0)
        sorted_p    = np.sort(at_pred, axis=0)
        weights     = (2 * np.arange(N, dtype=np.float64) - N + 1)
        spread_crps = (sorted_p * weights[:, None]).sum(axis=0) / (N * N)
        crps_min    = float(np.mean(abs_err - spread_crps))
        crps_min    = max(crps_min, 0.0)

        # crps_min is already the per-cell mean (via np.mean over cells).
        # Return it directly — no second division by n_active.
        return crps_min, n_active
