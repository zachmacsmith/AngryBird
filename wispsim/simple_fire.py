"""
SimpleFire — Huygens-style elliptical fire spread engine.

Extracted from demo_sim.py so run_sim.py can import it without pulling
in the entire deprecated demo script.

Implements FireEngineProtocol via a CPU-parallel KD-tree approach:
  - Base spread rate R [m/min] ∝ wind_speed × (1/FMC) × fuel_factor
  - Ellipse eccentricity ∝ wind speed
  - Arrival time at cell P from ignition I:
        t = sqrt( (along_wind_m / R)² + (cross_wind_m / R_cross)² )
  - Cells with t > horizon_min → unburned (NaN)

Use GPUFireEngine (simulation.gpu_fire_engine) for production runs
— it handles crown fire, terrain slope, and ensemble variance correctly.
SimpleFire is kept here as a lightweight fallback for environments without
PyTorch or for quick smoke-testing.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from angrybird.config import PROCESS_NOISE_FLOOR
from angrybird.types import EnsembleResult, GPPrior, TerrainData


def _run_fire_member(args: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute arrival times for one SimpleFire ensemble member.

    Module-level so ThreadPoolExecutor can submit it without pickling a
    class instance.  Uses scipy.cKDTree for O((N+K) log K) nearest-neighbour
    search in scaled wind-frame coordinates instead of O(N×K) brute force.
    """
    from scipy.spatial import cKDTree

    (m_seed, fmc_mean, fmc_std, ws_mean, ws_std, wd_mean,
     ig_r, ig_c, ignition, rows, cols, res, fuel_arr, horizon_min) = args

    rng = np.random.default_rng(int(m_seed))

    fmc_m = np.clip(fmc_mean + rng.standard_normal((rows, cols)) * fmc_std,
                    0.04, 0.60).astype(np.float32)
    ws_m  = np.clip(ws_mean  + rng.standard_normal((rows, cols)) * ws_std,
                    0.5, 20.0).astype(np.float32)
    wd_m  = (wd_mean + rng.normal(0, 8, (rows, cols))).astype(np.float32)

    mean_fmc   = float(fmc_m.mean())
    mean_ws    = float(ws_m.mean())
    wd_rad     = np.deg2rad(float(wd_m.mean()))
    fmc_factor = float(np.clip(np.exp(-6.0 * (mean_fmc - 0.06)), 0.1, 8.0))
    R          = max(3.0 * mean_ws * fmc_factor * float(fuel_arr.mean()), 1e-6)
    R_cross    = max(R * (1.0 - float(np.clip(0.08 * mean_ws, 0.0, 0.85))), 1e-6)
    wx         = float(np.sin(wd_rad))
    wy         = float(-np.cos(wd_rad))

    ig_r_m = ig_r.astype(np.float64) * res
    ig_c_m = ig_c.astype(np.float64) * res
    ig_uv  = np.column_stack([
        (ig_r_m * wy + ig_c_m * wx) / R,
        (ig_r_m * wx - ig_c_m * wy) / R_cross,
    ])

    r_m = np.arange(rows, dtype=np.float64) * res
    c_m = np.arange(cols, dtype=np.float64) * res
    RR, CC = np.meshgrid(r_m, c_m, indexing="ij")
    cell_uv = np.column_stack([
        (RR.ravel() * wy + CC.ravel() * wx) / R,
        (RR.ravel() * wx - CC.ravel() * wy) / R_cross,
    ])

    dist, _ = cKDTree(ig_uv).query(cell_uv, k=1, workers=-1)
    best = dist.reshape(rows, cols).astype(np.float32)
    best[ignition] = 0.0
    best[best > horizon_min] = np.nan
    return best, fmc_m, ws_m


class SimpleFire:
    """
    Huygens elliptical fire spread — CPU parallel, no PyTorch required.

    Fuel spread multipliers (Anderson 13):
      Fast fuels (grass/shrub): 1.0–1.6×
      Slow fuels (timber litter): 0.5–0.65×
    """

    _FUEL_SPREAD = {
        1: 0.8, 2: 1.0, 3: 1.6, 4: 1.3, 5: 0.9,
        6: 1.0, 7: 1.1, 8: 0.55, 9: 0.65, 10: 1.0,
        11: 0.5, 12: 0.9, 13: 1.0,
    }

    def run(
        self,
        terrain: TerrainData,
        gp_prior: GPPrior,
        fire_state: np.ndarray,
        n_members: int,
        horizon_min: int,
        rng: np.random.Generator,
        initial_phi: "np.ndarray | None" = None,
    ) -> EnsembleResult:
        rows, cols = terrain.shape
        res        = terrain.resolution_m

        ignition = fire_state.astype(bool)
        if not ignition.any():
            ignition[rows // 2, cols // 2] = True
        ig_r, ig_c = np.where(ignition)

        fuel_arr = np.vectorize(self._FUEL_SPREAD.get)(terrain.fuel_model, 1.0)
        fmc_std  = np.maximum(
            np.sqrt(np.clip(gp_prior.fmc_variance, 0.0, None)),
            PROCESS_NOISE_FLOOR,
        ).astype(np.float32)
        ws_std = np.sqrt(np.clip(gp_prior.wind_speed_variance, 0.0, None)).astype(np.float32)

        seeds = rng.integers(0, 2**31, size=n_members)
        member_args = [
            (seeds[m],
             gp_prior.fmc_mean.astype(np.float32), fmc_std,
             gp_prior.wind_speed_mean.astype(np.float32), ws_std,
             gp_prior.wind_dir_mean.astype(np.float32),
             ig_r, ig_c, ignition,
             rows, cols, res, fuel_arr, horizon_min)
            for m in range(n_members)
        ]

        n_workers = min(os.cpu_count() or 1, n_members)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_run_fire_member, member_args))

        all_arrival_min = np.stack([r[0] for r in results])
        all_fmc         = np.stack([r[1] for r in results])
        all_wind        = np.stack([r[2] for r in results])

        # KD-tree distance is in minutes; downstream expects hours.
        all_arrival = all_arrival_min / 60.0

        burned    = np.isfinite(all_arrival)
        burn_prob = burned.mean(axis=0).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_arr = np.where(burned.any(axis=0),
                                np.nanmean(all_arrival, axis=0), np.nan)
            var_arr  = np.where(burned.any(axis=0),
                                np.nanvar(all_arrival, axis=0), 0.0)

        return EnsembleResult(
            member_arrival_times  = all_arrival,
            member_fmc_fields     = all_fmc,
            member_wind_fields    = all_wind,
            burn_probability      = burn_prob,
            mean_arrival_time     = mean_arr.astype(np.float32),
            arrival_time_variance = var_arr.astype(np.float32),
            n_members             = n_members,
        )
