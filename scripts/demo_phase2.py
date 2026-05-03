"""
Demo of Phase 2 outputs: GP prior → ensemble → information field → visualization.

Uses synthetic data (no LANDFIRE, no real fire engine needed).

Usage:
    python -m scripts.demo_phase2               # display plot
    python -m scripts.demo_phase2 --save out.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as `python -m scripts.demo_phase2` from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from angrybird.config import (
    ENSEMBLE_SIZE,
    GP_CORRELATION_LENGTH_FMC_M,
    GRID_RESOLUTION_M,
    SIMULATION_HORIZON_MIN,
)
from angrybird.gp import IGNISGPPrior, draw_gp_scaled_field
from angrybird.information import compute_information_field
from angrybird.types import EnsembleResult
from angrybird.visualization import plot_phase2_summary, save_or_show


# ---------------------------------------------------------------------------
# Scenario parameters
# ---------------------------------------------------------------------------

ROWS, COLS = 120, 150
RES_M      = GRID_RESOLUTION_M       # 50 m/cell → 6 km × 7.5 km domain
N_MEMBERS  = min(ENSEMBLE_SIZE, 100) # keep demo fast

# RAWS station locations (grid row, col)
# Three stations in the eastern half — western half is data-sparse
RAWS_LOCS    = [(20, 110), (60, 125), (100, 105)]
RAWS_FMC     = [0.07,  0.12,  0.09]   # fraction
RAWS_WS      = [6.5,   4.8,   7.2]    # m/s
RAWS_WD      = [270.,  255.,  285.]   # degrees (westerly)

# Fire ignition: south-west corner, spreading north-east
IGNITION     = (105, 10)
BASE_SPEED   = 14.0   # cells / hour  (= 700 m/hr ≈ 12 m/min, moderate grass fire)
HORIZON_H    = 3.0    # hours  → fire reaches ~40-cell radius before stopping

# Greedy placeholder locations (before selectors are built)
# These are just for visual reference in the demo
DEMO_TARGETS = [(40, 20), (70, 35), (55, 60), (25, 45), (85, 50)]


# ---------------------------------------------------------------------------
# Build GP prior
# ---------------------------------------------------------------------------

def build_gp_prior() -> IGNISGPPrior:
    gp = IGNISGPPrior(terrain=None, resolution_m=RES_M)
    gp.add_raws(RAWS_LOCS, RAWS_FMC, RAWS_WS, RAWS_WD)
    return gp


# ---------------------------------------------------------------------------
# Synthetic ensemble (fire engine placeholder)
# ---------------------------------------------------------------------------

def build_synthetic_ensemble(gp_prior, rng: np.random.Generator) -> EnsembleResult:
    """
    Simulate N fire spread scenarios perturbed by GP variance.
    Fire starts at IGNITION and spreads radially with FMC-dependent speed.
    """
    shape = (ROWS, COLS)
    rr, cc = np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing="ij")

    member_arrivals = np.zeros((N_MEMBERS, ROWS, COLS), dtype=np.float32)
    member_fmc      = np.zeros((N_MEMBERS, ROWS, COLS), dtype=np.float32)
    member_wind     = np.zeros((N_MEMBERS, ROWS, COLS), dtype=np.float32)

    for i in range(N_MEMBERS):
        # GP-scaled FMC perturbation: higher uncertainty in data-sparse west
        fmc_pert = draw_gp_scaled_field(
            shape,
            correlation_length=GP_CORRELATION_LENGTH_FMC_M,
            resolution=RES_M,
            gp_variance=gp_prior.fmc_variance,
        )
        fmc_field = np.clip(gp_prior.fmc_mean + fmc_pert, 0.01, 0.50)

        # Wind perturbation (isotropic, smaller magnitude)
        ws_field = gp_prior.wind_speed_mean + rng.normal(0, 1.0, shape).astype(np.float32)
        ws_field = np.clip(ws_field, 0.5, 20.0)

        # Speed decreases with FMC (fire is harder to spread in wet fuels).
        # No upper clip — the sensitivity test needs speed to vary between members.
        speed = BASE_SPEED * np.clip(1 - 5 * (fmc_field - 0.09), 0.3, 2.0)

        # Radial arrival time from ignition (simplified, no terrain)
        dist = np.sqrt((rr - IGNITION[0]) ** 2 + (cc - IGNITION[1]) ** 2)
        arr = (dist / speed).astype(np.float32)

        # Cells beyond the horizon don't burn
        burned = arr <= HORIZON_H
        member_arrivals[i] = np.where(burned, arr, np.nan)
        member_fmc[i]      = fmc_field
        member_wind[i]     = ws_field

    burn_prob  = (~np.isnan(member_arrivals)).mean(axis=0).astype(np.float32)
    mean_arr   = np.nanmean(member_arrivals, axis=0).astype(np.float32)
    var_arr    = np.nanvar( member_arrivals, axis=0).astype(np.float32)

    return EnsembleResult(
        member_arrival_times=member_arrivals,
        member_fmc_fields=member_fmc,
        member_wind_fields=member_wind,
        burn_probability=burn_prob,
        mean_arrival_time=mean_arr,
        arrival_time_variance=var_arr,
        n_members=N_MEMBERS,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(save_path: str | None = None) -> None:
    rng = np.random.default_rng(seed=42)

    print("Building GP prior from 3 RAWS stations …")
    gp = build_gp_prior()
    prior = gp.predict((ROWS, COLS))
    print(f"  FMC variance — RAWS sites: {min(prior.fmc_variance[r,c] for r,c in RAWS_LOCS):.6f}")
    print(f"  FMC variance — data-sparse west: {prior.fmc_variance[60, 10]:.6f}")

    print(f"Running {N_MEMBERS}-member ensemble …")
    ensemble = build_synthetic_ensemble(prior, rng)
    n_burned = (ensemble.burn_probability > 0.5).sum()
    print(f"  Cells with >50% burn probability: {n_burned} / {ROWS*COLS}")

    print("Computing information field …")
    info = compute_information_field(ensemble, prior, resolution_m=RES_M)
    print(f"  max w = {info.w.max():.6f}  (at cell {np.unravel_index(info.w.argmax(), info.w.shape)})")

    print("Plotting …")
    fig = plot_phase2_summary(
        gp_prior=prior,
        ensemble=ensemble,
        info_field=info,
        station_locs=RAWS_LOCS,
        selected_locs=DEMO_TARGETS,
        resolution_m=RES_M,
        title=f"IGNIS · Phase 2 Demo  ({ROWS}×{COLS} grid, {N_MEMBERS} members, 3 RAWS)",
    )
    save_or_show(fig, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IGNIS Phase 2 visualization demo")
    parser.add_argument("--save", metavar="PATH", default=None,
                        help="Save figure to this path instead of displaying")
    args = parser.parse_args()
    main(args.save)
