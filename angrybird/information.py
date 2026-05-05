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
    BIMODAL_UNBURNED_FACTOR,
    BURNED_PROBABILITY_THRESHOLD,
    FIRE_DEGRADATION_RADIUS_M,
    FIRE_PERIMETER_HI_PROB,
    FIRE_PERIMETER_LO_PROB,
    GRID_RESOLUTION_M,
    SENSOR_FMC_R2,
    SENSOR_WIND_ACCURACY,
    SIMULATION_HORIZON_MIN,
)
from .types import EnsembleResult, GPPrior, InformationField

# Fire type constant (mirrors fire_oracle and gpu_fire_engine)
_CROWN_FIRE = 2


# ---------------------------------------------------------------------------
# Bimodal detection (Extended Fire Physics §3)
# ---------------------------------------------------------------------------

def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """
    Binary entropy H(p) = -(p·log₂p + (1-p)·log₂(1-p)), clipped to [0, 1].
    Maximum = 1.0 at p = 0.5; zero at p ∈ {0, 1}.
    """
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def detect_bimodality(
    member_arrival_times: np.ndarray,
    max_arrival: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify each cell by burn/no-burn bimodality across ensemble members.

    member_arrival_times : float32[N, rows, cols] — sentinel = max_arrival for unburned
    max_arrival          : sentinel value used for unburned cells

    Returns:
        burn_fraction  : float32[rows, cols] — fraction of members that burned
        bimodal_score  : float32[rows, cols] — 0=consensus, 1=50/50 split
    """
    burns          = member_arrival_times < (max_arrival * BIMODAL_UNBURNED_FACTOR)  # (N, rows, cols)
    burn_fraction  = burns.mean(axis=0).astype(np.float32)       # (rows, cols)
    bimodal_score  = (1.0 - 2.0 * np.abs(burn_fraction - 0.5)).astype(np.float32)
    return burn_fraction, bimodal_score


def detect_regime_split(
    member_fire_types: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect disagreement about surface vs crown fire regime across ensemble members.

    member_fire_types : int8[N, rows, cols] — 1=surface, 2=crown

    Returns:
        crown_fraction  : float32[rows, cols] — fraction of members with crown fire
        regime_bimodal  : float32[rows, cols] — 0=consensus, 1=50/50 split
    """
    crown_fraction = (member_fire_types == _CROWN_FIRE).mean(axis=0).astype(np.float32)
    regime_bimodal = (1.0 - 2.0 * np.abs(crown_fraction - 0.5)).astype(np.float32)
    return crown_fraction, regime_bimodal


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
        # Use population mean (N denominator) to be consistent with std().
        # N-1 Bessel correction mixed with population std can push |corr| > 1.
        cov = (A_c * P_c).mean(axis=0)                              # (D,) population cov
        corr = np.clip(cov / (std_A * std_P), -1.0, 1.0)           # (D,) guaranteed in [-1, 1]
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)  # guard any residual NaN
        return corr.reshape(rows, cols).astype(np.float32)

    # FMC perturbation = member field − prior mean
    fmc_perturb = ensemble.member_fmc_fields - gp_prior.fmc_mean[None, :, :]
    sensitivities["fmc"] = _corr(fmc_perturb, "fmc")

    # Wind speed perturbation = member field − prior mean
    ws_perturb = ensemble.member_wind_fields - gp_prior.wind_speed_mean[None, :, :]
    sensitivities["wind_speed"] = _corr(ws_perturb, "wind_speed")

    # Wind direction perturbation — circular wrap to [-180, +180] avoids 360° spikes
    if ensemble.member_wind_dir_fields is not None:
        wd_perturb = (
            (ensemble.member_wind_dir_fields - gp_prior.wind_dir_mean[None, :, :] + 180.0)
            % 360.0
        ) - 180.0
        sensitivities["wind_dir"] = _corr(wd_perturb, "wind_dir")

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
    D_fmc      = np.full((rows, cols), fmc_accuracy,  dtype=np.float32)
    D_wind     = np.full((rows, cols), wind_accuracy, dtype=np.float32)
    D_wind_dir = np.full((rows, cols), wind_accuracy, dtype=np.float32)

    # Degrade near the ACTIVE FIRE PERIMETER (0.1 < p < 0.9) — not the full burned area.
    # Drones fly ahead of the fire; cells far ahead have full observability.
    # Cells close to where fire is actively burning get reduced accuracy (smoke, turbulence).
    perimeter_mask = (ensemble.burn_probability > FIRE_PERIMETER_LO_PROB) & (ensemble.burn_probability < FIRE_PERIMETER_HI_PROB)
    source_mask = ensemble.burn_probability > 0.5   # fallback: use full burned area if no perimeter
    active_mask = perimeter_mask if perimeter_mask.any() else source_mask
    if active_mask.any():
        # EDT gives distance in cells from each unmasked cell to nearest masked cell
        dist_cells = distance_transform_edt(~active_mask)
        dist_m = (dist_cells * resolution_m).astype(np.float32)
        degradation = np.clip(dist_m / degradation_radius_m, 0.0, 1.0)
        D_fmc      *= degradation
        D_wind     *= degradation
        D_wind_dir *= degradation

    return {"fmc": D_fmc, "wind_speed": D_wind, "wind_dir": D_wind_dir}


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
    bimodal_alpha: float = 0.0,
    bimodal_beta: float = 0.0,
    fire_state_alpha: float = 0.0,
    fire_state_burn_prob: Optional[np.ndarray] = None,
) -> InformationField:
    """
    Compute the information value w at every grid cell.

        w = gp_variance_fmc   * |sensitivity_fmc|   * D_fmc
          + gp_variance_wind  * |sensitivity_wind|  * D_wind
          + bimodal_alpha     * H_binary(burn_fraction)            [optional]
          + bimodal_beta      * H_binary(crown_fraction)           [optional]
          + fire_state_alpha  * H_binary(fire_state_burn_prob)     [optional]

    Args:
        ensemble:              EnsembleResult from fire engine
        gp_prior:              GPPrior from GP estimator
        resolution_m:          grid cell size in metres
        horizon_minutes:       simulation horizon (for unburned sentinel)
        priority_weight_field: float32[rows,cols], >1 in priority regions (optional)
        exclusion_mask:        bool[rows,cols], True where drones are excluded (optional)
        bimodal_alpha:         weight for binary burn-probability entropy (0 = disabled)
        bimodal_beta:          weight for crown fire regime entropy (0 = disabled;
                               requires ensemble.member_fire_types to be set)
        fire_state_alpha:      weight for fire location uncertainty entropy (0 = disabled).
                               When > 0, fire_state_burn_prob must be supplied.
        fire_state_burn_prob:  float32[rows,cols] — P(cell currently burning) derived
                               from EnsembleFireState at cycle start.  Cells at the
                               uncertain fire perimeter have high binary entropy and
                               attract drones to confirm fire position.

    Returns:
        InformationField with total w and per-variable breakdowns.
    """
    shape = ensemble.burn_probability.shape

    sensitivity = compute_sensitivity(ensemble, gp_prior, horizon_minutes)
    observability = compute_observability(ensemble, shape, resolution_m)

    # Per-variable information maps.
    # NaN-guard each GP variance field: ill-conditioned GP fits on large/heterogeneous
    # grids (especially wind direction) can produce NaN posterior variances which would
    # poison the entire w field.  Replace NaN/negative variance with 0 so those cells
    # simply contribute nothing rather than collapsing the whole information field.
    fmc_var  = np.nan_to_num(gp_prior.fmc_variance,  nan=0.0, posinf=0.0, neginf=0.0)
    wind_var = np.nan_to_num(gp_prior.wind_speed_variance, nan=0.0, posinf=0.0, neginf=0.0)
    fmc_var  = np.clip(fmc_var,  0.0, None)
    wind_var = np.clip(wind_var, 0.0, None)

    w_fmc = (
        fmc_var
        * np.abs(sensitivity["fmc"])
        * observability["fmc"]
    ).astype(np.float32)

    w_wind = (
        wind_var
        * np.abs(sensitivity["wind_speed"])
        * observability["wind_speed"]
    ).astype(np.float32)

    w = w_fmc + w_wind

    w_wind_dir: Optional[np.ndarray] = None
    if "wind_dir" in sensitivity:
        wd_var = np.nan_to_num(gp_prior.wind_dir_variance, nan=0.0, posinf=0.0, neginf=0.0)
        wd_var = np.clip(wd_var, 0.0, None)
        w_wind_dir = (
            wd_var
            * np.abs(sensitivity["wind_dir"])
            * observability["wind_dir"]
        ).astype(np.float32)
        w = (w + w_wind_dir).astype(np.float32)

    # Bimodal entropy augmentation (Extended Fire Physics §3)
    max_arrival = _max_arrival_sentinel(horizon_minutes)
    if bimodal_alpha > 0.0:
        burn_frac, _ = detect_bimodality(ensemble.member_arrival_times, max_arrival)
        w = (w + bimodal_alpha * _binary_entropy(burn_frac)).astype(np.float32)

    if bimodal_beta > 0.0 and ensemble.member_fire_types is not None:
        crown_frac, _ = detect_regime_split(ensemble.member_fire_types)
        w = (w + bimodal_beta * _binary_entropy(crown_frac)).astype(np.float32)

    # Fire location uncertainty (from EnsembleFireState per-member fronts)
    # High entropy at cells where members disagree about current fire position.
    # Drives drones to confirm fire perimeter when location is the key uncertainty.
    if fire_state_alpha > 0.0 and fire_state_burn_prob is not None:
        w_fire_state = (fire_state_alpha * _binary_entropy(fire_state_burn_prob)).astype(np.float32)
        w = (w + w_fire_state).astype(np.float32)

    # Cells already burned have zero information value (fire won't spread back)
    burned = ensemble.burn_probability > BURNED_PROBABILITY_THRESHOLD
    w[burned] = 0.0
    w_fmc[burned] = 0.0
    w_wind[burned] = 0.0
    if w_wind_dir is not None:
        w_wind_dir[burned] = 0.0

    # Operator overlays
    if priority_weight_field is not None:
        w = (w * priority_weight_field).astype(np.float32)
        w_fmc = (w_fmc * priority_weight_field).astype(np.float32)
        w_wind = (w_wind * priority_weight_field).astype(np.float32)
        if w_wind_dir is not None:
            w_wind_dir = (w_wind_dir * priority_weight_field).astype(np.float32)

    if exclusion_mask is not None:
        w[exclusion_mask] = 0.0
        w_fmc[exclusion_mask] = 0.0
        w_wind[exclusion_mask] = 0.0
        if w_wind_dir is not None:
            w_wind_dir[exclusion_mask] = 0.0

    w_by_variable: dict[str, np.ndarray] = {"fmc": w_fmc, "wind_speed": w_wind}
    gp_variance: dict[str, np.ndarray] = {
        "fmc": gp_prior.fmc_variance,
        "wind_speed": gp_prior.wind_speed_variance,
    }
    if w_wind_dir is not None:
        w_by_variable["wind_dir"] = w_wind_dir
        gp_variance["wind_dir"] = gp_prior.wind_dir_variance

    return InformationField(
        w=w,
        w_by_variable=w_by_variable,
        sensitivity=sensitivity,
        gp_variance=gp_variance,
    )
