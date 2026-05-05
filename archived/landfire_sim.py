"""
Full IGNIS ensemble fire simulation on real LANDFIRE terrain.

Runs the GPU fire engine (Rothermel + level-set) on the cached Northern
California LANDFIRE tile (249×322 cells, 100 m resolution, ~39.2°N −121.2°W).

Three horizon scenarios: 0.4 h (24 min), 1 h, 4 h.

Two output families:
  fire_only_<horizon>.png  — ensemble burn-probability + arrival isochrones
  drone_<horizon>.png      — same fire map + 1-drone information routing

Usage
-----
  python scripts/landfire_sim.py [--device cpu|mps|cuda] [--members N]

"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import gaussian_filter

from angrybird.config import (
    FUEL_MINERAL_CONTENT, FUEL_MINERAL_SILICA_FREE,
)
from angrybird.fire_engines.gpu_fire_engine import GPUFireEngine
from angrybird.information import compute_information_field
from angrybird.landfire import load_from_directory
from angrybird.nelson import nelson_fmc_field
from angrybird.path_planner import plan_paths
from angrybird.selectors.greedy import GreedySelector
from angrybird.gp import IGNISGPPrior
from angrybird.observations import ObservationStore
from angrybird.types import GPPrior, TerrainData
from angrybird.visualization.core import plot_fire_prediction_map
from angrybird.visualization._style import compute_hillshade, VIZ_CONFIG

OUT_DIR = "out/landfire_sim"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device",  default="cpu",
                   help="Torch device: cpu | mps | cuda")
    p.add_argument("--members", type=int, default=30,
                   help="Ensemble size (default 30)")
    p.add_argument("--cache",   default="landfire_cache",
                   help="Path to LANDFIRE GeoTIFF cache directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Terrain-correlated GP prior (no observations)
# ---------------------------------------------------------------------------

def build_prior(terrain: TerrainData, seed: int = 0) -> GPPrior:
    """
    Build a spatially realistic GPPrior for a mid-afternoon California
    fire-weather day: T=35 °C, RH=15 %, 14:00 local, SW wind at 5 m/s.

    FMC: Nelson model (terrain-correlated via slope/aspect/elevation).
    Wind speed/dir: Gaussian-smoothed spatial field, SW base with orographic
    perturbations.  Variance captures analyst uncertainty (no drone obs yet).
    """
    rng = np.random.default_rng(seed)
    rows, cols = terrain.shape

    # ── FMC via Nelson ────────────────────────────────────────────────────
    fmc_mean = nelson_fmc_field(
        terrain     = terrain,
        T_C         = 35.0,
        RH          = 0.15,
        hour_of_day = 14.0,
        latitude_deg= terrain.origin[0] if terrain.origin_latlon else 39.2,
    ).astype(np.float32)
    fmc_mean = np.clip(fmc_mean, 0.05, 0.35)
    fmc_var  = np.full((rows, cols), 0.012**2, dtype=np.float32)  # ±1.2% std

    # ── Wind — SW synoptic base + terrain perturbation ────────────────────
    base_ws  = 5.0   # m/s  (SW, ~11 mph)
    base_wd  = 225.0 # degrees  (SW → NE)

    # Spatial wind-speed noise correlated at ~1 km
    ws_noise = rng.standard_normal((rows, cols)).astype(np.float32)
    ws_noise = gaussian_filter(ws_noise, sigma=10.0)  # 10 cells × 100m = 1 km
    ws_noise = ws_noise / (ws_noise.std() + 1e-8) * 0.8  # ±0.8 m/s std

    # Ridge acceleration: wind picks up over high terrain (simple proxy)
    tpi = terrain.elevation - gaussian_filter(terrain.elevation, sigma=15)
    tpi_norm = np.clip(tpi / (tpi.std() + 1e-8), -2, 2).astype(np.float32)
    ws_mean = np.clip(base_ws + ws_noise + 0.4 * tpi_norm, 0.5, 15.0).astype(np.float32)
    ws_var  = np.full((rows, cols), 1.5**2, dtype=np.float32)   # ±1.5 m/s std

    # Wind direction: small terrain-induced deflections near ridges
    wd_noise = rng.standard_normal((rows, cols)).astype(np.float32)
    wd_noise = gaussian_filter(wd_noise, sigma=8.0) * 12.0  # ±12° spatial noise
    wd_mean  = ((base_wd + wd_noise) % 360.0).astype(np.float32)
    wd_var   = np.full((rows, cols), 15.0**2, dtype=np.float32)  # ±15° std

    return GPPrior(
        fmc_mean            = fmc_mean,
        fmc_variance        = fmc_var,
        wind_speed_mean     = ws_mean,
        wind_speed_variance = ws_var,
        wind_dir_mean       = wd_mean,
        wind_dir_variance   = wd_var,
    )


# ---------------------------------------------------------------------------
# Ignition — single cell near domain centre in burnable fuel
# ---------------------------------------------------------------------------

def find_ignition(terrain: TerrainData) -> tuple[int, int]:
    """Return (row, col) of the first burnable cell within 20 cells of centre."""
    R, C  = terrain.shape
    cr, cc = R // 2, C // 2
    fm    = terrain.fuel_model
    for radius in range(0, 40):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = cr + dr, cc + dc
                if 0 <= r < R and 0 <= c < C and fm[r, c] > 99:
                    return r, c
    return cr, cc   # fallback


def make_fire_state(terrain: TerrainData, ignition: tuple[int, int]) -> np.ndarray:
    fs = np.zeros(terrain.shape, dtype=np.float32)
    fs[ignition] = 1.0
    return fs


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _km_ticks(ax, shape: tuple[int, int], resolution_m: float) -> None:
    """Label axes in kilometres instead of cell indices."""
    rows, cols = shape
    xticks = np.arange(0, cols + 1, max(1, cols // 5))
    yticks = np.arange(0, rows + 1, max(1, rows // 5))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x * resolution_m / 1000:.0f}" for x in xticks], fontsize=6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y * resolution_m / 1000:.0f}" for y in yticks], fontsize=6)
    ax.set_xlabel("East (km)", fontsize=7)
    ax.set_ylabel("North (km)", fontsize=7)


def fire_only_figure(
    results:    list,           # list of EnsembleResult
    horizons_h: list[float],
    terrain:    TerrainData,
    ignition:   tuple[int, int],
    prior:      GPPrior,
    out_path:   str,
) -> None:
    """3-panel figure: burn-probability maps for each horizon."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 7), constrained_layout=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "IGNIS Ensemble Fire Prediction — LANDFIRE Terrain (Northern California)\n"
        f"Ignition: cell ({ignition[0]}, {ignition[1]})  ·  "
        f"SW wind 5 m/s  ·  FMC 8–15%  ·  {results[0].n_members} ensemble members",
        fontsize=9, fontweight="bold",
    )

    rows, cols = terrain.shape
    extent = [0, cols, rows, 0]
    hs = compute_hillshade(terrain.elevation, resolution_m=terrain.resolution_m)

    cmap_bp = plt.get_cmap("YlOrRd")

    for ax, res, h in zip(axes, results, horizons_h):
        # Hillshade base
        ax.imshow(hs, origin="upper", cmap="gray", vmin=0, vmax=1,
                  extent=extent, interpolation="bilinear", zorder=0)

        # Burn probability
        bp = res.burn_probability
        cf = ax.contourf(bp,
                         levels=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0],
                         cmap=cmap_bp, alpha=0.75, zorder=1)
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04,
                     label="Burn Probability").ax.tick_params(labelsize=6)

        # Arrival-time isochrones
        arr = np.where(np.isnan(res.mean_arrival_time),
                       np.inf, res.mean_arrival_time)
        # Choose isochrone intervals based on horizon
        if h <= 0.5:
            iso_min = [5, 10, 15, 20]
        elif h <= 1.5:
            iso_min = [15, 30, 45, 60]
        else:
            iso_min = [30, 60, 120, 180, 240]
        iso_min = [m for m in iso_min if m < np.nanmax(res.mean_arrival_time)]
        if iso_min:
            cs = ax.contour(arr, levels=iso_min, colors="white",
                            linewidths=0.8, linestyles="--", zorder=2)
            ax.clabel(cs, fmt="%d min", fontsize=5.5, inline=True,
                      inline_spacing=2, colors="white")

        # 90% perimeter
        ax.contour(bp, levels=[0.9], colors=["#CC0000"],
                   linewidths=1.5, zorder=3)

        # Ignition marker
        ax.scatter([ignition[1]], [ignition[0]], marker="*", s=120,
                   c="yellow", edgecolors="black", linewidths=0.6,
                   zorder=5, label="Ignition")

        # FMC field contour (thin grey, to show spatial structure)
        fmc_cs = ax.contour(prior.fmc_mean, levels=[0.08, 0.10, 0.12],
                            colors=["#555555"], linewidths=0.4,
                            linestyles=":", alpha=0.6, zorder=2)

        ax.set_title(f"Horizon: {h:.1f} h ({int(h * 60)} min)\n"
                     f"Burned: {bp[bp > 0.5].shape[0] / (rows * cols):.1%} > 50% prob",
                     fontsize=8, fontweight="bold")
        _km_ticks(ax, terrain.shape, terrain.resolution_m)

        # Mini legend
        legend_items = [
            plt.Line2D([0], [0], color="#CC0000", lw=1.5, label="P>0.9 perimeter"),
            plt.Line2D([0], [0], color="white", lw=0.8, ls="--", label="Arrival (min)"),
            plt.scatter([], [], marker="*", c="yellow", edgecolors="black",
                        s=60, label="Ignition"),
        ]
        ax.legend(handles=legend_items, fontsize=5.5, loc="lower right",
                  framealpha=0.75)

    # Domain footnote
    km_ns = rows * terrain.resolution_m / 1000
    km_ew = cols * terrain.resolution_m / 1000
    fig.text(0.5, 0.01,
             f"Domain: {km_ew:.1f} km E–W × {km_ns:.1f} km N–S  "
             f"| Resolution: {terrain.resolution_m:.0f} m  "
             f"| Origin: {terrain.origin[0]:.3f}°N  {terrain.origin[1]:.3f}°E",
             ha="center", fontsize=6, color="gray")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def drone_figure(
    result:     object,         # EnsembleResult
    horizon_h:  float,
    terrain:    TerrainData,
    ignition:   tuple[int, int],
    prior:      GPPrior,
    gp:         IGNISGPPrior,
    out_path:   str,
    n_drones:   int = 1,
) -> None:
    """Fire prediction + information field + drone routing."""
    from angrybird.visualization.core import plot_fire_prediction_map, plot_information_field

    # ── Information field ─────────────────────────────────────────────────
    info = compute_information_field(
        ensemble      = result,
        gp_prior      = prior,
        resolution_m  = terrain.resolution_m,
        horizon_minutes = horizon_h * 60,
    )

    # ── Greedy drone selection ────────────────────────────────────────────
    sel = GreedySelector()
    selection = sel.select(info, gp, result, k=n_drones * 3)  # 3 waypoints / drone

    # ── Path planning ─────────────────────────────────────────────────────
    staging = (terrain.shape[0] - 1, terrain.shape[1] // 2)  # S edge, centre
    drone_plans = plan_paths(
        selected_locations = selection.selected_locations,
        staging_area       = staging,
        n_drones           = n_drones,
        shape              = terrain.shape,
        resolution_m       = terrain.resolution_m,
        drone_range_m      = 40000.0,   # 40 km range — covers full domain
    )

    # ── Figure: 2 panels (fire prediction | information + drone) ─────────
    fig, (ax_fire, ax_info) = plt.subplots(1, 2, figsize=(14, 7),
                                            constrained_layout=True)

    rows, cols = terrain.shape
    extent = [0, cols, rows, 0]
    hs = compute_hillshade(terrain.elevation, resolution_m=terrain.resolution_m)

    # --- Left: fire prediction ---
    ax_fire.imshow(hs, origin="upper", cmap="gray", vmin=0, vmax=1,
                   extent=extent, interpolation="bilinear", zorder=0)
    bp = result.burn_probability
    cf = ax_fire.contourf(bp,
                          levels=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0],
                          cmap="YlOrRd", alpha=0.75, zorder=1)
    plt.colorbar(cf, ax=ax_fire, fraction=0.046, pad=0.04,
                 label="Burn Probability").ax.tick_params(labelsize=6)

    arr = np.where(np.isnan(result.mean_arrival_time),
                   np.inf, result.mean_arrival_time)
    iso_min = [15, 30, 45, 60] if horizon_h <= 1.5 else [30, 60, 120, 180, 240]
    iso_min = [m for m in iso_min if m < np.nanmax(result.mean_arrival_time)]
    if iso_min:
        cs = ax_fire.contour(arr, levels=iso_min, colors="white",
                             linewidths=0.8, linestyles="--", zorder=2)
        ax_fire.clabel(cs, fmt="%d min", fontsize=5.5, inline=True,
                       colors="white")
    ax_fire.contour(bp, levels=[0.9], colors=["#CC0000"], linewidths=1.5, zorder=3)
    ax_fire.scatter([ignition[1]], [ignition[0]], marker="*", s=150,
                    c="yellow", edgecolors="black", zorder=5)
    ax_fire.set_title(f"Ensemble Fire Prediction — {horizon_h:.1f} h",
                      fontsize=9, fontweight="bold")
    _km_ticks(ax_fire, terrain.shape, terrain.resolution_m)

    # --- Right: information field + drone path ---
    im = ax_info.imshow(info.w, origin="upper", cmap="inferno",
                        interpolation="nearest", zorder=1)
    plt.colorbar(im, ax=ax_info, fraction=0.046, pad=0.04,
                 label="Information Value w(x)").ax.tick_params(labelsize=6)

    # Burn perimeter overlay (thin white)
    ax_info.contour(bp, levels=[0.5], colors=["white"],
                    linewidths=0.8, linestyles="--", zorder=2)

    # Drone path(s)
    drone_colors = ["#00FFFF", "#FFD700", "#FF69B4", "#00FF7F", "#FF4500"]
    for d_idx, plan in enumerate(drone_plans):
        col = drone_colors[d_idx % len(drone_colors)]
        wps = plan.waypoints  # list of (row, col)
        if len(wps) < 2:
            continue
        path_r = [staging[0]] + [w[0] for w in wps] + [staging[0]]
        path_c = [staging[1]] + [w[1] for w in wps] + [staging[1]]
        ax_info.plot(path_c, path_r, "-", color=col, lw=1.8,
                     zorder=4, label=f"Drone {d_idx + 1}")
        ax_info.scatter(path_c[1:-1], path_r[1:-1], marker="D", s=25,
                        c=col, edgecolors="white", lw=0.5, zorder=5)
        # Start/end base
        ax_info.scatter([staging[1]], [staging[0]], marker="s", s=60,
                        c="#FFFFFF", edgecolors="black", lw=0.8, zorder=6)

    ax_info.set_title(
        f"Information Field  +  {n_drones}-Drone Routing\n"
        f"(Red=FMC-driven · Blue=Wind-driven uncertainty)",
        fontsize=8, fontweight="bold",
    )
    _km_ticks(ax_info, terrain.shape, terrain.resolution_m)
    if drone_plans:
        ax_info.legend(fontsize=6, loc="lower right", framealpha=0.75)

    fig.suptitle(
        "IGNIS — LANDFIRE Terrain  ·  Northern California  ·  "
        f"Horizon {horizon_h:.1f} h  ·  {n_drones} Drone{'s' if n_drones > 1 else ''}",
        fontsize=9, fontweight="bold",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load terrain ──────────────────────────────────────────────────────
    print("Loading LANDFIRE terrain …")
    terrain = load_from_directory(args.cache, resolution_m=100.0)
    R, C = terrain.shape
    print(f"  Shape: {R}×{C}  Resolution: {terrain.resolution_m:.0f} m  "
          f"({R * terrain.resolution_m / 1000:.1f} km × "
          f"{C * terrain.resolution_m / 1000:.1f} km)")
    print(f"  Origin: {terrain.origin[0]:.3f}°N  {terrain.origin[1]:.3f}°W")
    print(f"  Fuel codes: {np.unique(terrain.fuel_model).tolist()}")

    # ── Prior and ignition ────────────────────────────────────────────────
    print("Building terrain-correlated GP prior …")
    prior = build_prior(terrain)
    print(f"  FMC mean: {prior.fmc_mean.mean():.3f} ± {prior.fmc_mean.std():.3f}")
    print(f"  Wind:     {prior.wind_speed_mean.mean():.1f} m/s "
          f"from {prior.wind_dir_mean.mean():.0f}°")

    ignition   = find_ignition(terrain)
    fire_state = make_fire_state(terrain, ignition)
    print(f"  Ignition: {ignition}  fuel={terrain.fuel_model[ignition]}  "
          f"slope={terrain.slope[ignition]:.1f}°  "
          f"elev={terrain.elevation[ignition]:.0f} m")

    # ── Build a minimal IGNISGPPrior for drone selector ───────────────────
    gp = IGNISGPPrior(
        terrain      = terrain,
        resolution_m = terrain.resolution_m,
    )

    # ── Engine ────────────────────────────────────────────────────────────
    print(f"\nInitialising GPUFireEngine (device={args.device}) …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        engine = GPUFireEngine(terrain, device=args.device)
    print("  Engine ready.")

    # ── Horizons ─────────────────────────────────────────────────────────
    HORIZONS_H = [0.4, 1.0, 4.0]
    results = []

    for h in HORIZONS_H:
        horizon_min = int(round(h * 60))
        print(f"\nRunning ensemble — horizon {h:.1f} h ({horizon_min} min)  "
              f"N={args.members} …")
        t0 = time.time()
        rng = np.random.default_rng(42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = engine.run(
                terrain     = terrain,
                gp_prior    = prior,
                fire_state  = fire_state,
                n_members   = args.members,
                horizon_min = horizon_min,
                rng         = rng,
            )
        elapsed = time.time() - t0
        n_burned = (res.burn_probability > 0.5).sum()
        area_km2 = n_burned * (terrain.resolution_m / 1000) ** 2
        print(f"  Done in {elapsed:.1f} s  |  "
              f"P>0.5 burned: {n_burned} cells ({area_km2:.1f} km²)  |  "
              f"Max arrival: {np.nanmax(res.mean_arrival_time):.1f} min")
        results.append(res)

    # ── Fire-only 3-panel figure ──────────────────────────────────────────
    print(f"\nSaving fire-only 3-panel figure …")
    fire_only_figure(
        results     = results,
        horizons_h  = HORIZONS_H,
        terrain     = terrain,
        ignition    = ignition,
        prior       = prior,
        out_path    = f"{OUT_DIR}/fire_only_all.png",
    )

    # ── Per-horizon fire-only figures ─────────────────────────────────────
    for res, h in zip(results, HORIZONS_H):
        label = f"{int(h * 60)}min"
        fire_only_figure(
            results     = [res],
            horizons_h  = [h],
            terrain     = terrain,
            ignition    = ignition,
            prior       = prior,
            out_path    = f"{OUT_DIR}/fire_only_{label}.png",
        )

    # ── 1-drone routing figures (1 h and 4 h horizons) ───────────────────
    for res, h in zip(results, HORIZONS_H):
        label = f"{int(h * 60)}min"
        print(f"\nComputing information field + drone routing ({h:.1f} h) …")
        drone_figure(
            result     = res,
            horizon_h  = h,
            terrain    = terrain,
            ignition   = ignition,
            prior      = prior,
            gp         = gp,
            out_path   = f"{OUT_DIR}/drone_1_{label}.png",
            n_drones   = 1,
        )

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
