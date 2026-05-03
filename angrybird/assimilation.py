"""
Data assimilation: EnKF state update + observation thinning + replan flags.

Phase 4a — ships with the production system.

Shape reference (PotentialBugs1 §2):
  X:    (N, D)       ensemble states, N members, D = rows×cols
  HX:   (N, n_obs)   states at observation locations — index gather, not matrix mul
  PHT:  (D, n_obs)   = A.T @ HA / (N-1)
  K:    (D, n_obs)   Kalman gain

Inflation is applied inside enkf_update before returning (PotentialBugs2 §2).
Without inflation the ensemble collapses after 3-4 cycles and the information
field goes to zero everywhere.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .config import (
    ENKF_INFLATION_FACTOR,
    ENKF_LOCALIZATION_RADIUS_M,
    ENKF_OUTLIER_THRESHOLD,
    GP_CORRELATION_LENGTH_FMC_M,
    GRID_RESOLUTION_M,
    REPLAN_VARIANCE_REDUCTION_THRESHOLD,
    REPLAN_WIND_SHIFT_THRESHOLD_DEG,
)
from .gp import IGNISGPPrior
from .types import DroneObservation, EnsembleResult, GPPrior
from .utils import angular_diff_deg, distance_grid, gaspari_cohn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observation thinning (PotentialBugs2 §1)
# ---------------------------------------------------------------------------

def thin_drone_observations(
    observations: list[DroneObservation],
    min_spacing_m: float = GP_CORRELATION_LENGTH_FMC_M / 3.0,
    resolution_m: float = GRID_RESOLUTION_M,
) -> list[DroneObservation]:
    """
    Reduce dense swath observations before assimilation.
    Keeps the lowest-noise observation when multiple fall within min_spacing_m.
    Reduces ~1000 raw cells to ~50 for stable EnKF matrix inversion.
    """
    if not observations:
        return []
    sorted_obs = sorted(observations, key=lambda o: o.fmc_sigma)
    kept: list[DroneObservation] = []
    min_cells_sq = (min_spacing_m / resolution_m) ** 2
    for obs in sorted_obs:
        r, c = obs.location
        too_close = any(
            (r - k.location[0]) ** 2 + (c - k.location[1]) ** 2 < min_cells_sq
            for k in kept
        )
        if not too_close:
            kept.append(obs)
    logger.debug("Observation thinning: %d → %d", len(observations), len(kept))
    return kept


# ---------------------------------------------------------------------------
# EnKF update
# ---------------------------------------------------------------------------

def enkf_update(
    X: np.ndarray,
    y_obs: np.ndarray,
    obs_indices: list[int],
    obs_sigma: np.ndarray,
    shape: tuple[int, int],
    obs_locations: list[tuple[int, int]],
    resolution_m: float = GRID_RESOLUTION_M,
    localization_radius_m: float = ENKF_LOCALIZATION_RADIUS_M,
    inflation_factor: float = ENKF_INFLATION_FACTOR,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Ensemble Kalman Filter update for one state variable.

    H is an index gather — never materialized as a matrix (PotentialBugs1 §2).
    Inflation applied to anomalies before the update (PotentialBugs2 §2).

    Args:
        X:                (N, D) ensemble states — copied, not modified in place
        y_obs:            (n_obs,) observed values
        obs_indices:      flat indices [r*cols + c] for each observation
        obs_sigma:        (n_obs,) observation noise std devs
        shape:            (rows, cols) for localization distance computation
        obs_locations:    (row, col) pairs — same order as obs_indices
        resolution_m:     grid cell size in metres
        localization_radius_m: Gaspari-Cohn taper radius
        inflation_factor: multiplicative inflation (>1.0) applied to anomalies
        rng:              Generator for perturbed observations

    Returns:
        Updated (N, D) array.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, D = X.shape
    n_obs = len(obs_indices)
    if n_obs == 0:
        return X.copy()

    # 1. Inflate ensemble anomalies before update (PotentialBugs2 §2)
    x_mean = X.mean(axis=0)                              # (D,)
    A = (X - x_mean) * inflation_factor                  # (N, D)
    X_inf = x_mean + A                                   # (N, D)

    # 2. Observation operator — index gather, not matrix multiply (PotentialBugs1 §2)
    HX = X_inf[:, obs_indices]                           # (N, n_obs)
    HA = HX - HX.mean(axis=0)                           # (N, n_obs)

    # 3. Covariances
    PHT  = (A.T @ HA) / (N - 1)                         # (D, n_obs)
    HPHT = (HA.T @ HA) / (N - 1)                        # (n_obs, n_obs)
    R    = np.diag(obs_sigma ** 2)                       # (n_obs, n_obs)

    # 4. Kalman gain
    K = PHT @ np.linalg.inv(HPHT + R)                   # (D, n_obs)

    # 5. Localization: taper K beyond correlation radius (Gaspari-Cohn)
    for j, obs_loc in enumerate(obs_locations):
        dists = distance_grid(obs_loc[0], obs_loc[1], shape, resolution_m).ravel()
        taper = gaspari_cohn(dists, localization_radius_m).astype(np.float64)
        K[:, j] *= taper

    # 6. Perturbed-observations update for each ensemble member
    X_out = X_inf.copy()
    y = y_obs.astype(np.float64)
    for n in range(N):
        y_perturbed = y + rng.normal(0.0, obs_sigma)
        X_out[n] += K @ (y_perturbed - HX[n])

    return X_out


# ---------------------------------------------------------------------------
# Full assimilation step
# ---------------------------------------------------------------------------

def assimilate_observations(
    gp: IGNISGPPrior,
    ensemble: EnsembleResult,
    observations: list[DroneObservation],
    shape: tuple[int, int],
    resolution_m: float = GRID_RESOLUTION_M,
    gp_prior: Optional[GPPrior] = None,
    rng: Optional[np.random.Generator] = None,
    localization_radius_m: float = ENKF_LOCALIZATION_RADIUS_M,
    inflation_factor: float = ENKF_INFLATION_FACTOR,
) -> dict:
    """
    Full assimilation pipeline: thin → outlier check → GP update → EnKF → replan flags.

    Returns dict:
      'fmc_states'   float32[N, rows, cols] updated FMC member fields
      'wind_states'  float32[N, rows, cols] updated wind member fields
      'replan_flags' dict[str, bool]
      'n_obs_used'   int — number of observations after thinning
    """
    thinned = thin_drone_observations(observations, resolution_m=resolution_m)
    N = ensemble.n_members

    if not thinned:
        logger.info("Assimilation: no observations this cycle.")
        return {
            "fmc_states":   ensemble.member_fmc_fields.copy(),
            "wind_states":  ensemble.member_wind_fields.copy(),
            "replan_flags": {"variance_drop": False, "wind_shift": False},
            "n_obs_used":   0,
        }

    locs      = [o.location for o in thinned]
    fmc_vals  = [o.fmc for o in thinned]
    fmc_sigs  = [o.fmc_sigma for o in thinned]
    ws_vals   = [o.wind_speed for o in thinned]
    ws_sigs   = [o.wind_speed_sigma for o in thinned]

    # Outlier check — log but assimilate anyway (localization limits damage)
    fmc_arr = np.array(fmc_vals)
    ens_fmc_at_obs = np.stack(
        [ensemble.member_fmc_fields[:, r, c] for r, c in locs], axis=1
    )  # (N, n_obs)
    ens_mean = ens_fmc_at_obs.mean(axis=0)
    ens_std  = ens_fmc_at_obs.std(axis=0) + 1e-6
    n_outliers = int((np.abs(fmc_arr - ens_mean) / ens_std > ENKF_OUTLIER_THRESHOLD).sum())
    if n_outliers:
        logger.warning(
            "EnKF: %d observation(s) flagged as outliers (>%.1fσ) — assimilating anyway.",
            n_outliers, ENKF_OUTLIER_THRESHOLD,
        )

    # GP update: add observations to conditioning set (triggers refit on next predict)
    gp.add_observations(locs, fmc_vals, fmc_sigs, ws_vals, ws_sigs)

    # EnKF updates — separate pass per variable
    obs_idx = [r * shape[1] + c for r, c in locs]

    fmc_X = ensemble.member_fmc_fields.reshape(N, -1).astype(np.float64)
    fmc_X = enkf_update(
        fmc_X, np.array(fmc_vals), obs_idx, np.array(fmc_sigs),
        shape, locs, resolution_m, localization_radius_m, inflation_factor, rng,
    )

    ws_X = ensemble.member_wind_fields.reshape(N, -1).astype(np.float64)
    ws_X = enkf_update(
        ws_X, np.array(ws_vals), obs_idx, np.array(ws_sigs),
        shape, locs, resolution_m, localization_radius_m, inflation_factor, rng,
    )

    # Replan flags
    var_drop_flag  = False
    wind_shift_flag = False
    if gp_prior is not None:
        var_before = float(gp_prior.fmc_variance.sum())
        new_prior  = gp.predict(shape)
        var_after  = float(new_prior.fmc_variance.sum())
        var_drop   = (var_before - var_after) / max(var_before, 1e-9)
        var_drop_flag = var_drop > REPLAN_VARIANCE_REDUCTION_THRESHOLD

        for obs in thinned:
            if obs.wind_dir is not None:
                r, c = obs.location
                prior_dir = float(gp_prior.wind_dir_mean[r, c])
                if float(angular_diff_deg(obs.wind_dir, prior_dir)) > REPLAN_WIND_SHIFT_THRESHOLD_DEG:
                    wind_shift_flag = True
                    break

    logger.info(
        "Assimilation: n_obs=%d, var_drop=%s, wind_shift=%s",
        len(thinned), var_drop_flag, wind_shift_flag,
    )
    return {
        "fmc_states":   fmc_X.reshape(N, *shape).astype(np.float32),
        "wind_states":  ws_X.reshape(N, *shape).astype(np.float32),
        "replan_flags": {"variance_drop": var_drop_flag, "wind_shift": wind_shift_flag},
        "n_obs_used":   len(thinned),
    }
