"""
Fire Retrospect Observation System.

Converts particle filter weights into GP observations of FMC and wind at the
active fire front.  Each cycle, weighted ensemble statistics over fire-active
correlation domains are emitted as FireRetrospectObservations and injected into
the GP store, preserving implied physical conditions learned from fire perimeter
history across forecast cycles.

See docs/Fire State Estimation.md §Fire Retrospect for design rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .observations import FireRetrospectObservation

if TYPE_CHECKING:
    from .selectors.correlation_path import CorrelationDomain
    from .types import EnsembleResult, GPPrior


def generate_fire_retrospect_observations(
    weights: np.ndarray,
    ensemble: "EnsembleResult",
    fire_front_mask: np.ndarray,
    domains: "list[CorrelationDomain]",
    gp_prior: "GPPrior",
    variance_floor_frac: float = 0.25,
    current_time: float = 0.0,
    min_resultant_length: float = 0.5,
) -> list[FireRetrospectObservation]:
    """
    Generate synthetic FMC/wind observations from weighted ensemble statistics.

    For each correlation domain whose representative cell falls inside the
    active fire front, emit one FireRetrospectObservation encoding the
    particle-filter-weighted mean and uncertainty of FMC, wind speed, and
    (when members agree sufficiently) wind direction.

    Args:
        weights:              float64[N] normalized particle filter weights (sum=1).
        ensemble:             EnsembleResult from the cycle that was just reweighted.
        fire_front_mask:      bool[R,C] — True for cells in the active perimeter zone
                              (0.05 < burn_fraction < 0.95).
        domains:              Terrain-based correlation domains from CorrelationGraph.
        gp_prior:             Current GP posterior — used only for variance floors.
        variance_floor_frac:  Fraction of prior variance used as minimum observation
                              variance.  Default 0.25 (25%).
        current_time:         Simulation time in seconds (attached as timestamp).
        min_resultant_length: Mean resultant length R below which the wind direction
                              obs is omitted (members disagree too much to be useful).

    Returns:
        List of FireRetrospectObservations, one per fire-front domain.
        Empty if no fire-front domains or ensemble fields are missing.
    """
    if ensemble.member_fmc_fields is None or ensemble.member_wind_fields is None:
        return []
    if not fire_front_mask.any():
        return []

    w = weights.astype(np.float64)
    fmc_fields = ensemble.member_fmc_fields.astype(np.float64)   # (N, R, C)
    ws_fields  = ensemble.member_wind_fields.astype(np.float64)  # (N, R, C)
    wd_fields: Optional[np.ndarray] = (
        ensemble.member_wind_dir_fields.astype(np.float64)
        if ensemble.member_wind_dir_fields is not None
        else None
    )

    observations: list[FireRetrospectObservation] = []

    for domain in domains:
        r, c = domain.representative_cell

        if not fire_front_mask[r, c]:
            continue

        # ── FMC ──────────────────────────────────────────────────────────────
        fmc_vals  = fmc_fields[:, r, c]
        fmc_mean  = float(np.dot(w, fmc_vals))
        fmc_var   = float(np.dot(w, (fmc_vals - fmc_mean) ** 2))
        fmc_floor = variance_floor_frac * float(gp_prior.fmc_variance[r, c])
        fmc_sigma = float(np.sqrt(max(fmc_var, fmc_floor)))

        # ── Wind speed ────────────────────────────────────────────────────────
        ws_vals  = ws_fields[:, r, c]
        ws_mean  = float(np.dot(w, ws_vals))
        ws_var   = float(np.dot(w, (ws_vals - ws_mean) ** 2))
        ws_floor = variance_floor_frac * float(gp_prior.wind_speed_variance[r, c])
        ws_sigma = float(np.sqrt(max(ws_var, ws_floor)))

        # ── Wind direction (circular statistics) ──────────────────────────────
        wd_mean_out:  Optional[float] = None
        wd_sigma_out: Optional[float] = None
        if wd_fields is not None:
            wd_rad   = np.radians(wd_fields[:, r, c])
            mean_sin = float(np.dot(w, np.sin(wd_rad)))
            mean_cos = float(np.dot(w, np.cos(wd_rad)))
            R_len    = float(np.sqrt(mean_sin ** 2 + mean_cos ** 2))
            if R_len >= min_resultant_length:
                circ_mean_deg = float(np.degrees(np.arctan2(mean_sin, mean_cos))) % 360.0
                circ_var      = 1.0 - R_len   # dimensionless, 0=tight, 1=uniform
                # Convert to angular sigma in degrees; floor from GP prior variance
                wd_prior_sigma_deg = float(np.sqrt(max(float(gp_prior.wind_dir_variance[r, c]), 1e-6)))
                sigma_from_circ    = float(np.degrees(np.sqrt(circ_var)))
                wd_sigma_out  = max(sigma_from_circ, variance_floor_frac * wd_prior_sigma_deg)
                wd_mean_out   = circ_mean_deg

        observations.append(FireRetrospectObservation(
            _source_id           = f"fire_retrospect_{domain.domain_id}",
            _timestamp           = current_time,
            location             = (r, c),
            fmc                  = fmc_mean,
            fmc_sigma            = fmc_sigma,
            wind_speed           = ws_mean,
            wind_speed_sigma     = ws_sigma,
            wind_direction       = wd_mean_out,
            wind_direction_sigma = wd_sigma_out,
        ))

    return observations
