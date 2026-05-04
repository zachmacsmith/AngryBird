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
    AGGREGATION_SIGMA_FLOOR,
    ENKF_INFLATION_FACTOR,
    ENKF_LOCALIZATION_RADIUS_M,
    ENKF_OUTLIER_THRESHOLD,
    GP_CORRELATION_LENGTH_FMC_M,
    GRID_RESOLUTION_M,
    OBS_FMC_SIGMA,
    OBS_WIND_DIR_SIGMA,
    OBS_WIND_SPEED_SIGMA,
    REPLAN_VARIANCE_REDUCTION_THRESHOLD,
    REPLAN_WIND_SHIFT_THRESHOLD_DEG,
)
from .gp import IGNISGPPrior
from .observations import DroneObservation as DroneObs, ObservationStore, VariableType
from .types import DroneObservation, EnsembleResult, GPPrior
from .utils import angular_diff_deg, distance_grid, gaspari_cohn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observation aggregation (replaces thinning, PotentialBugs2 §1)
# ---------------------------------------------------------------------------

def thin_drone_observations(
    observations: list[DroneObservation],
    min_spacing_m: float = GP_CORRELATION_LENGTH_FMC_M / 3.0,
    resolution_m: float = GRID_RESOLUTION_M,
) -> list[DroneObservation]:
    """
    Greedy thinning: keep at most one observation per min_spacing_m radius.

    Observations are sorted by fmc_sigma ascending so the lowest-noise
    observation wins when two are too close together.
    """
    if not observations:
        return []

    min_cells_sq = (min_spacing_m / resolution_m) ** 2
    sorted_obs = sorted(observations, key=lambda o: o.fmc_sigma)
    kept: list[DroneObservation] = []

    for obs in sorted_obs:
        r, c = obs.location
        too_close = any(
            (r - kr) ** 2 + (c - kc) ** 2 < min_cells_sq
            for kr, kc in (k.location for k in kept)
        )
        if not too_close:
            kept.append(obs)

    return kept


def aggregate_drone_observations(
    observations: list[DroneObservation],
    spacing_m: float = GP_CORRELATION_LENGTH_FMC_M,
    resolution_m: float = GRID_RESOLUTION_M,
) -> list[DroneObservation]:
    """
    Aggregate spatially dense observations into one representative observation per
    spatial cell (radius = spacing_m).

    Instead of keeping only the best observation per cell (thinning), this averages
    ALL observations within each cell using inverse-variance weighting.  The
    resulting aggregate has:

        fmc_agg   = Σ(fmc_i / σ_i²) / Σ(1 / σ_i²)          [precision-weighted mean]
        σ_agg     = 1 / √Σ(1 / σ_i²)                         [combined precision → std]

    This prevents the GP from being over-constrained by hundreds of correlated
    individual footprint readings (which collapse `gp_var` to ~1e-6 and make the
    information field uniformly black).  With aggregation:

        - ~22 raw obs/cycle → ~3–5 aggregate obs/cycle
        - σ_agg ≈ σ_raw / √n  (correctly reflects reduced uncertainty from averaging)
        - GP sees well-separated, genuinely informative training points
        - min(gp_var) stays above ~0.001 → info field remains meaningful

    Spatial cells are assigned greedily: the first un-assigned observation seeds
    each cell; all remaining observations within spacing_m are merged into it.

    Wind speed / direction use the same precision-weighted aggregation, but only
    among nadir-only observations that carry valid wind readings.
    """
    if not observations:
        return []

    min_cells_sq = (spacing_m / resolution_m) ** 2
    remaining   = list(observations)
    aggregated: list[DroneObservation] = []

    while remaining:
        seed = remaining.pop(0)
        sr, sc = seed.location

        # Collect all obs (including seed) within spacing_m of the seed
        group   = [seed]
        outside = []
        for obs in remaining:
            dr = obs.location[0] - sr
            dc = obs.location[1] - sc
            if dr * dr + dc * dc < min_cells_sq:
                group.append(obs)
            else:
                outside.append(obs)
        remaining = outside

        # ── FMC: precision-weighted mean ─────────────────────────────────────
        fmc_weights  = np.array([1.0 / (o.fmc_sigma ** 2) for o in group])
        fmc_wsum     = fmc_weights.sum()
        fmc_agg      = float(np.dot(fmc_weights, [o.fmc for o in group]) / fmc_wsum)
        # Floor at half the individual sensor sigma — prevents averaging many
        # readings from collapsing sigma to near-zero and over-constraining the GP
        # (which drives info_field.w → 0 everywhere after the first cycle).
        sigma_fmc    = max(float(1.0 / np.sqrt(fmc_wsum)), AGGREGATION_SIGMA_FLOOR)

        # Centroid location (rounded to nearest cell)
        r_agg = int(round(float(np.mean([o.location[0] for o in group]))))
        c_agg = int(round(float(np.mean([o.location[1] for o in group]))))

        # ── Wind speed: precision-weighted mean (nadir obs only) ─────────────
        ws_group = [o for o in group
                    if o.wind_speed is not None and np.isfinite(o.wind_speed)
                    and o.wind_speed_sigma is not None and np.isfinite(o.wind_speed_sigma)
                    and o.wind_speed_sigma > 0]
        if ws_group:
            ws_weights = np.array([1.0 / (o.wind_speed_sigma ** 2) for o in ws_group])
            ws_wsum    = ws_weights.sum()
            ws_agg     = float(np.dot(ws_weights, [o.wind_speed for o in ws_group]) / ws_wsum)
            sigma_ws   = max(float(1.0 / np.sqrt(ws_wsum)), OBS_WIND_SPEED_SIGMA / 2.0)
        else:
            ws_agg   = None
            sigma_ws = None

        # ── Wind direction: circular precision-weighted mean ──────────────────
        wd_group = [o for o in ws_group
                    if o.wind_dir is not None and np.isfinite(o.wind_dir)]
        if wd_group:
            wd_weights = np.array([1.0 / ((o.wind_dir_sigma or 10.0) ** 2) for o in wd_group])
            wd_wsum    = wd_weights.sum()
            dirs_rad   = np.deg2rad([o.wind_dir for o in wd_group])
            sin_mean   = float(np.dot(wd_weights, np.sin(dirs_rad)) / wd_wsum)
            cos_mean   = float(np.dot(wd_weights, np.cos(dirs_rad)) / wd_wsum)
            wd_agg     = float(np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0)
            sigma_wd   = max(float(1.0 / np.sqrt(wd_wsum)), OBS_WIND_DIR_SIGMA / 2.0)
        else:
            wd_agg   = None
            sigma_wd = None

        aggregated.append(DroneObservation(
            location         = (r_agg, c_agg),
            fmc              = fmc_agg,
            fmc_sigma        = sigma_fmc,
            wind_speed       = ws_agg,
            wind_speed_sigma = sigma_ws,
            wind_dir         = wd_agg,
            wind_dir_sigma   = sigma_wd,
            timestamp        = seed.timestamp,
            drone_id         = seed.drone_id,
        ))

    logger.debug(
        "Observation aggregation: %d → %d (spacing=%.0f m)",
        len(observations), len(aggregated), spacing_m,
    )
    return aggregated


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
    obs_store_or_ensemble,  # ObservationStore (new API) or EnsembleResult (old API)
    ensemble,               # EnsembleResult (new API) or list[DroneObservation] (old API)
    observations,           # list[DroneObservation] (new API) or tuple[int,int] shape (old API)
    shape=None,             # tuple[int, int] (new API) or None when old API used
    resolution_m: float = GRID_RESOLUTION_M,
    gp_prior: Optional[GPPrior] = None,
    rng: Optional[np.random.Generator] = None,
    localization_radius_m: float = ENKF_LOCALIZATION_RADIUS_M,
    inflation_factor: float = ENKF_INFLATION_FACTOR,
) -> dict:
    """
    Full assimilation pipeline: thin → outlier check → store update → GP refit → EnKF → replan flags.

    Supports two calling conventions:
      New: assimilate_observations(gp, obs_store, ensemble, observations, shape, ...)
      Old: assimilate_observations(gp, ensemble, observations, shape, ...)

    Returns dict:
      'fmc_states'   float32[N, rows, cols] updated FMC member fields
      'wind_states'  float32[N, rows, cols] updated wind member fields
      'replan_flags' dict[str, bool]
      'n_obs_used'   int — number of observations after thinning
    """
    # Detect calling convention by inspecting arg types
    if isinstance(obs_store_or_ensemble, ObservationStore):
        obs_store  = obs_store_or_ensemble
        # ensemble, observations, shape already bound correctly
        _ensemble    = ensemble
        _observations = observations
        _shape       = shape
    else:
        # Old API: (gp, ensemble, observations, shape, ...)
        obs_store    = gp._store
        _ensemble    = obs_store_or_ensemble
        _observations = ensemble
        _shape       = observations  # type: ignore[assignment]
    ensemble     = _ensemble
    observations = _observations
    shape        = _shape

    thinned = aggregate_drone_observations(
        observations,
        spacing_m=GP_CORRELATION_LENGTH_FMC_M,
        resolution_m=resolution_m,
    )
    N = ensemble.n_members

    if not thinned:
        logger.info("Assimilation: no observations this cycle.")
        return {
            "fmc_states":   ensemble.member_fmc_fields.copy(),
            "wind_states":  ensemble.member_wind_fields.copy(),
            "replan_flags": {"variance_drop": False, "wind_shift": False},
            "n_obs_used":   0,
        }

    locs     = [o.location for o in thinned]
    fmc_vals = [o.fmc for o in thinned]
    fmc_sigs = [o.fmc_sigma for o in thinned]

    # Off-centre footprint cells carry FMC only — wind obs are nadir-only.
    # Separate wind-valid observations so NaNs never reach the GP or EnKF.
    wind_obs  = [o for o in thinned
                 if o.wind_speed is not None and np.isfinite(o.wind_speed)
                 and o.wind_speed_sigma is not None and np.isfinite(o.wind_speed_sigma)]
    ws_locs   = [o.location     for o in wind_obs]
    ws_vals   = [o.wind_speed   for o in wind_obs]
    ws_sigs   = [o.wind_speed_sigma for o in wind_obs]

    # Wind direction — also nadir-only; same obs that carry wind speed.
    dir_obs   = [o for o in wind_obs
                 if o.wind_dir is not None and np.isfinite(o.wind_dir)]
    wd_locs   = [o.location        for o in dir_obs]
    wd_vals   = [o.wind_dir        for o in dir_obs]
    wd_sigs   = [o.wind_dir_sigma if o.wind_dir_sigma is not None else 10.0
                 for o in dir_obs]

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

    # Store update: one DroneObs per thinned telemetry location.
    # Multi-variable observations collapse into a single store object; the
    # GP unwraps them via to_data_points() at query time.
    new_obs: list[DroneObs] = []
    wind_by_loc = {o.location: o for o in wind_obs}
    dir_by_loc  = {o.location: o for o in dir_obs}
    for obs in thinned:
        t  = obs.timestamp if obs.timestamp is not None else 0.0
        wo = wind_by_loc.get(obs.location)
        do = dir_by_loc.get(obs.location)
        new_obs.append(DroneObs(
            _source_id           = obs.drone_id or "drone",
            _timestamp           = t,
            location             = obs.location,
            fmc                  = obs.fmc,
            fmc_sigma            = obs.fmc_sigma,
            wind_speed           = wo.wind_speed if wo else None,
            wind_speed_sigma     = wo.wind_speed_sigma if wo else None,
            wind_direction       = do.wind_dir if do else None,
            wind_direction_sigma = (do.wind_dir_sigma if do and do.wind_dir_sigma is not None
                                    else (10.0 if do else None)),
        ))
    if new_obs:
        obs_store.add_batch(new_obs)

    # EnKF updates — separate pass per variable, wind uses its own location subset
    obs_idx    = [r * shape[1] + c for r, c in locs]
    ws_obs_idx = [r * shape[1] + c for r, c in ws_locs]

    fmc_X = ensemble.member_fmc_fields.reshape(N, -1).astype(np.float64)
    fmc_X = enkf_update(
        fmc_X, np.array(fmc_vals), obs_idx, np.array(fmc_sigs),
        shape, locs, resolution_m, localization_radius_m, inflation_factor, rng,
    )

    ws_X = ensemble.member_wind_fields.reshape(N, -1).astype(np.float64)
    if ws_vals:
        ws_X = enkf_update(
            ws_X, np.array(ws_vals), ws_obs_idx, np.array(ws_sigs),
            shape, ws_locs, resolution_m, localization_radius_m, inflation_factor, rng,
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
