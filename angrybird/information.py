"""
Information field computation.

Public surface:
  compute_information_field(ensemble, gp_prior, ...) -> InformationField

Pipeline:
  1. Sensitivity   — per-cell correlation between arrival times and perturbation fields
  2. Observability — sensor quality, degraded near active fire
  3. Info value    — w = gp_variance * |sensitivity| * observability  (one NumPy line)
  4. Overlays      — priority weight field, exclusion mask
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from .config import (
    FIRE_DEGRADATION_RADIUS_M,
    GRID_RESOLUTION_M,
    SENSOR_FMC_R2,
    SENSOR_WIND_ACCURACY,
    SIMULATION_HORIZON_MIN,
)
from .types import EnsembleResult, GPPrior, InformationField


# ---------------------------------------------------------------------------
# Unburned-cell sentinel (PotentialBugs1 §3)
# ---------------------------------------------------------------------------

def _max_arrival_sentinel(horizon_minutes: float = SIMULATION_HORIZON_MIN) -> float:
    """2× simulation horizon in hours — large but finite, preserves variance structure."""
    return 2.0 * (horizon_minutes / 60.0)


# ---------------------------------------------------------------------------
# Step 1: Sensitivity
# ---------------------------------------------------------------------------

def compute_sensitivity(
    ensemble: EnsembleResult,
    gp_prior: GPPrior,
    horizon_minutes: float = SIMULATION_HORIZON_MIN,
) -> dict[str, np.ndarray]:
    """
    Per-cell correlation between fire arrival times and the perturbation field
    used for each variable.

    Uses PotentialBugs1 §3 sentinel:
      - NaN arrival times → MAX_ARRIVAL (finite, preserves inter-member variance)
      - Zero-variance cells: std floored at 1e-10 to prevent NaN propagation

    Returns:
        {"fmc": float32[rows,cols], "wind_speed": float32[rows,cols]}
    """
    N = ensemble.n_members
    rows, cols = ensemble.burn_probability.shape
    D = rows * cols

    max_arrival = _max_arrival_sentinel(horizon_minutes)

    # (N, D) — replace NaN unburned sentinel with MAX_ARRIVAL
    A_flat = ensemble.member_arrival_times.reshape(N, D).astype(np.float64)
    A_flat = np.where(np.isnan(A_flat), max_arrival, A_flat)

    A_c = A_flat - A_flat.mean(axis=0)          # (N, D)
    std_A = A_flat.std(axis=0)                   # (D,)
    # Floor prevents NaN when every member has identical arrival time (PotentialBugs1 §3)
    std_A = np.where(std_A < 1e-10, 1e-10, std_A)

    sensitivities: dict[str, np.ndarray] = {}

    def _corr(perturbation_field: np.ndarray, label: str) -> np.ndarray:
        """Per-cell correlation between A and a perturbation field, both (N, D)."""
        P = perturbation_field.reshape(N, D).astype(np.float64)
        P_c = P - P.mean(axis=0)
        std_P = P.std(axis=0)
        std_P = np.where(std_P < 1e-10, 1e-10, std_P)
        cov = (A_c * P_c).sum(axis=0) / max(N - 1, 1)   # (D,)
        corr = cov / (std_A * std_P)                       # (D,) in [-1, 1]
        return corr.reshape(rows, cols).astype(np.float32)

    # FMC perturbation = member field − prior mean
    fmc_perturb = ensemble.member_fmc_fields - gp_prior.fmc_mean[None, :, :]
    sensitivities["fmc"] = _corr(fmc_perturb, "fmc")

    # Wind speed perturbation = member field − prior mean
    ws_perturb = ensemble.member_wind_fields - gp_prior.wind_speed_mean[None, :, :]
    sensitivities["wind_speed"] = _corr(ws_perturb, "wind_speed")

    return sensitivities


# ---------------------------------------------------------------------------
# Step 2: Observability
# ---------------------------------------------------------------------------

def compute_observability(
    ensemble: EnsembleResult,
    shape: tuple[int, int],
    resolution_m: float = GRID_RESOLUTION_M,
    degradation_radius_m: float = FIRE_DEGRADATION_RADIUS_M,
    fmc_accuracy: float = SENSOR_FMC_R2,
    wind_accuracy: float = SENSOR_WIND_ACCURACY,
) -> dict[str, np.ndarray]:
    """
    D_v[r, c]: fraction of sensor accuracy retained at cell (r, c).

    Accuracy degrades smoothly within `degradation_radius_m` of the active fire
    front (smoke, turbulence). Uses a proper distance transform so every cell
    gets the distance to the nearest burning cell (not just the centroid).
    """
    rows, cols = shape
    D_fmc = np.full((rows, cols), fmc_accuracy, dtype=np.float32)
    D_wind = np.full((rows, cols), wind_accuracy, dtype=np.float32)

    # Degrade near the ACTIVE FIRE PERIMETER (0.1 < p < 0.9) — not the full burned area.
    # Drones fly ahead of the fire; cells far ahead have full observability.
    # Cells close to where fire is actively burning get reduced accuracy (smoke, turbulence).
    perimeter_mask = (ensemble.burn_probability > 0.1) & (ensemble.burn_probability < 0.9)
    source_mask = ensemble.burn_probability > 0.5   # fallback: use full burned area if no perimeter
    active_mask = perimeter_mask if perimeter_mask.any() else source_mask
    if active_mask.any():
        # EDT gives distance in cells from each unmasked cell to nearest masked cell
        dist_cells = distance_transform_edt(~active_mask)
        dist_m = (dist_cells * resolution_m).astype(np.float32)
        degradation = np.clip(dist_m / degradation_radius_m, 0.0, 1.0)
        D_fmc *= degradation
        D_wind *= degradation

    return {"fmc": D_fmc, "wind_speed": D_wind}


# ---------------------------------------------------------------------------
# Step 3 + 4: Information value + overlays
# ---------------------------------------------------------------------------

def compute_information_field(
    ensemble: EnsembleResult,
    gp_prior: GPPrior,
    resolution_m: float = GRID_RESOLUTION_M,
    horizon_minutes: float = SIMULATION_HORIZON_MIN,
    priority_weight_field: Optional[np.ndarray] = None,
    exclusion_mask: Optional[np.ndarray] = None,
) -> InformationField:
    """
    Compute the information value w at every grid cell.

        w = gp_variance_fmc   * |sensitivity_fmc|   * D_fmc
          + gp_variance_wind  * |sensitivity_wind|  * D_wind

    Args:
        ensemble:              EnsembleResult from fire engine
        gp_prior:              GPPrior from GP estimator
        resolution_m:          grid cell size in metres
        horizon_minutes:       simulation horizon (for unburned sentinel)
        priority_weight_field: float32[rows,cols], >1 in priority regions (optional)
        exclusion_mask:        bool[rows,cols], True where drones are excluded (optional)

    Returns:
        InformationField with total w and per-variable breakdowns.
    """
    shape = ensemble.burn_probability.shape

    sensitivity = compute_sensitivity(ensemble, gp_prior, horizon_minutes)
    observability = compute_observability(ensemble, shape, resolution_m)

    # Per-variable information maps
    w_fmc = (
        gp_prior.fmc_variance
        * np.abs(sensitivity["fmc"])
        * observability["fmc"]
    ).astype(np.float32)

    w_wind = (
        gp_prior.wind_speed_variance
        * np.abs(sensitivity["wind_speed"])
        * observability["wind_speed"]
    ).astype(np.float32)

    w = w_fmc + w_wind

    # Cells already burned have zero information value (fire won't spread back)
    burned = ensemble.burn_probability > 0.95
    w[burned] = 0.0
    w_fmc[burned] = 0.0
    w_wind[burned] = 0.0

    # Operator overlays
    if priority_weight_field is not None:
        w = (w * priority_weight_field).astype(np.float32)
        w_fmc = (w_fmc * priority_weight_field).astype(np.float32)
        w_wind = (w_wind * priority_weight_field).astype(np.float32)

    if exclusion_mask is not None:
        w[exclusion_mask] = 0.0
        w_fmc[exclusion_mask] = 0.0
        w_wind[exclusion_mask] = 0.0

    return InformationField(
        w=w,
        w_by_variable={"fmc": w_fmc, "wind_speed": w_wind},
        sensitivity=sensitivity,
        gp_variance={
            "fmc": gp_prior.fmc_variance,
            "wind_speed": gp_prior.wind_speed_variance,
        },
    )
