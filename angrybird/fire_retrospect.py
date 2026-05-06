"""
Fire Retrospect Observation System.

One synthetic observation per new fire detection, placed at the correlation
domain where that detection's weight vector most diverges from uniform — i.e.
where surviving members' FMC differs most from non-survivors'. Deduplicated
across detections sharing a domain (keep highest divergence). Observation count
is bounded by the number of new fire detections, never by fire front length.

Three design constraints prevent GP variance collapse:

  1. N_eff gate  — skip entirely when N_eff > 0.8 N (uniform weights = the
                   detection added no discriminating information).

  2. One-per-detection with divergence placement  — each detection independently
                   finds its best domain (max normalized FMC divergence in the
                   fire front mask). Domains with zero divergence are skipped.
                   Two detections mapping to the same domain deduplicate to the
                   one with higher divergence. Five GOES pixels → ≤ five obs.

  3. ESS variance inflation  — weighted variance is multiplied by N / N_eff
                   before the floor, so a single dominant member's FMC doesn't
                   get injected as a precise measurement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .observations import FireDetectionObservation, FireRetrospectObservation

if TYPE_CHECKING:
    from .selectors.correlation_path import CorrelationDomain
    from .types import EnsembleResult, GPPrior


def _find_best_placement(
    weights: np.ndarray,
    fmc_fields: np.ndarray,
    domains: "list[CorrelationDomain]",
    fire_front_mask: np.ndarray,
    gp_prior: "GPPrior",
) -> "tuple[Optional[CorrelationDomain], float]":
    """
    Return the fire-front domain where the weight vector diverges most from
    uniform, normalised by the GP prior sigma.

    Returns (best_domain, normalized_divergence). normalized_divergence is
    |weighted_mean - unweighted_mean| / sigma_prior, so the caller can apply
    a threshold in units of sigma without knowing the raw FMC scale.
    """
    max_norm_div = 0.0
    best_domain: Optional[CorrelationDomain] = None

    for domain in domains:
        r, c = domain.representative_cell
        if not fire_front_mask[r, c]:
            continue

        vals = fmc_fields[:, r, c]
        unweighted = float(np.mean(vals))
        weighted   = float(np.dot(weights, vals))
        divergence = abs(weighted - unweighted)

        prior_sigma = float(np.sqrt(max(float(gp_prior.fmc_variance[r, c]), 1e-12)))
        norm_div    = divergence / prior_sigma

        if norm_div > max_norm_div:
            max_norm_div = norm_div
            best_domain  = domain

    return best_domain, max_norm_div


def generate_fire_retrospect_observations(
    weights: np.ndarray,
    fire_detections: "list[FireDetectionObservation]",
    ensemble: "EnsembleResult",
    fire_front_mask: np.ndarray,
    domains: "list[CorrelationDomain]",
    gp_prior: "GPPrior",
    n_eff: float,
    n_members: int,
    variance_floor_frac: float = 0.25,
    min_divergence_sigma_frac: float = 0.5,
    current_time: float = 0.0,
    min_resultant_length: float = 0.5,
) -> list[FireRetrospectObservation]:
    """
    Generate at most one synthetic FMC/wind observation per new fire detection.

    For each detection, find the fire-front domain whose weighted FMC mean
    diverges most from the unweighted mean (i.e. where surviving members'
    conditions differ most from non-survivors'). Detections that map to the
    same domain deduplicate to the one with higher divergence.

    Args:
        weights:                  float64[N] normalized particle filter weights.
        fire_detections:          New FireDetectionObservations this cycle.
        ensemble:                 EnsembleResult from the reweighted cycle.
        fire_front_mask:          bool[R,C] — 0.05 < burn_fraction < 0.95.
        domains:                  Terrain-based correlation domains.
        gp_prior:                 Current GP posterior (floors + thresholds).
        n_eff:                    Effective sample size 1/Σw².
        n_members:                Total ensemble size N.
        variance_floor_frac:      Minimum obs variance as fraction of prior var.
        min_divergence_sigma_frac: Skip a detection when its best domain's
                                  normalized divergence < this value (default 0.5σ).
        current_time:             Simulation time in seconds (obs timestamp).
        min_resultant_length:     Omit wind dir obs when mean resultant R < this.

    Returns:
        List of FireRetrospectObservations, length ≤ len(fire_detections).
    """
    if not fire_detections:
        return []
    if ensemble.member_fmc_fields is None or ensemble.member_wind_fields is None:
        return []
    if not fire_front_mask.any():
        return []

    # ESS variance inflation — large when only a few members dominate weights
    n_eff_safe   = max(n_eff, 1.0)
    ess_inflation = float(n_members) / n_eff_safe

    w          = weights.astype(np.float64)
    fmc_fields = ensemble.member_fmc_fields.astype(np.float64)   # (N, R, C)
    ws_fields  = ensemble.member_wind_fields.astype(np.float64)  # (N, R, C)
    wd_fields: Optional[np.ndarray] = (
        ensemble.member_wind_dir_fields.astype(np.float64)
        if ensemble.member_wind_dir_fields is not None
        else None
    )

    # For each detection, find its best domain.
    # best_per_domain: domain_id → (normalized_divergence, CorrelationDomain)
    best_per_domain: dict[int, tuple[float, CorrelationDomain]] = {}

    for detection in fire_detections:
        best_domain, norm_div = _find_best_placement(
            w, fmc_fields, domains, fire_front_mask, gp_prior,
        )
        if best_domain is None or norm_div < min_divergence_sigma_frac:
            continue

        d_id = best_domain.domain_id
        if d_id not in best_per_domain or norm_div > best_per_domain[d_id][0]:
            best_per_domain[d_id] = (norm_div, best_domain)

    if not best_per_domain:
        return []

    # Build one observation per surviving domain
    observations: list[FireRetrospectObservation] = []

    for _norm_div, domain in best_per_domain.values():
        r, c = domain.representative_cell

        # ── FMC ──────────────────────────────────────────────────────────────
        fmc_vals    = fmc_fields[:, r, c]
        fmc_weighted = float(np.dot(w, fmc_vals))
        fmc_var      = float(np.dot(w, (fmc_vals - fmc_weighted) ** 2))
        fmc_floor    = variance_floor_frac * float(gp_prior.fmc_variance[r, c])
        fmc_sigma    = float(np.sqrt(max(fmc_var * ess_inflation, fmc_floor)))

        # ── Wind speed ────────────────────────────────────────────────────────
        ws_vals     = ws_fields[:, r, c]
        ws_weighted  = float(np.dot(w, ws_vals))
        ws_var       = float(np.dot(w, (ws_vals - ws_weighted) ** 2))
        ws_floor     = variance_floor_frac * float(gp_prior.wind_speed_variance[r, c])
        ws_sigma     = float(np.sqrt(max(ws_var * ess_inflation, ws_floor)))

        # ── Wind direction (circular) ─────────────────────────────────────────
        wd_mean_out:  Optional[float] = None
        wd_sigma_out: Optional[float] = None
        if wd_fields is not None:
            wd_rad   = np.radians(wd_fields[:, r, c])
            mean_sin = float(np.dot(w, np.sin(wd_rad)))
            mean_cos = float(np.dot(w, np.cos(wd_rad)))
            R_len    = float(np.sqrt(mean_sin ** 2 + mean_cos ** 2))
            if R_len >= min_resultant_length:
                circ_mean_deg   = float(np.degrees(np.arctan2(mean_sin, mean_cos))) % 360.0
                circ_var        = 1.0 - R_len
                wd_prior_sigma  = float(np.sqrt(max(float(gp_prior.wind_dir_variance[r, c]), 1e-6)))
                sigma_from_circ = float(np.degrees(np.sqrt(circ_var)))
                wd_floor        = variance_floor_frac * wd_prior_sigma
                wd_sigma_out    = max(sigma_from_circ * float(np.sqrt(ess_inflation)), wd_floor)
                wd_mean_out     = circ_mean_deg

        observations.append(FireRetrospectObservation(
            _source_id           = f"fire_retrospect_{domain.domain_id}",
            _timestamp           = current_time,
            location             = (r, c),
            fmc                  = fmc_weighted,
            fmc_sigma            = fmc_sigma,
            wind_speed           = ws_weighted,
            wind_speed_sigma     = ws_sigma,
            wind_direction       = wd_mean_out,
            wind_direction_sigma = wd_sigma_out,
        ))

    return observations
