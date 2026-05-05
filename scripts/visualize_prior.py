"""
Visualize static prior ensemble vs ground truth fire spread.

Two-panel animation:
  Left  — Ground truth fire spreading over time (arrival time colour map +
           current perimeter contour)
  Right — Static prior burn probability (frozen ensemble from t=0, no obs)

Shows exactly what the prior "thinks" vs what is actually happening on the
ground, making the value of drone observations visually obvious.

Usage:
    PYTHONPATH=. python scripts/visualize_prior.py
    PYTHONPATH=. python scripts/visualize_prior.py --hours 2 --members 50 --out out/prior_viz
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress noisy library loggers
for _lib in ("matplotlib", "PIL", "rasterio", "numexpr"):
    logging.getLogger(_lib).setLevel(logging.WARNING)


def _load_terrain(terrain_dir: str):
    from run import _load_landfire
    return _load_landfire(terrain_dir)


def _find_ignition(terrain):
    from run import find_burnable_cell
    return find_burnable_cell(terrain)


def _make_fire_engine(terrain, device: str):
    from run import _make_fire_engine as _rfe
    return _rfe(terrain, device)


def _hillshade(elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    gy, gx = np.gradient(elevation, resolution_m)
    sun_az, sun_el = np.radians(315), np.radians(45)
    normal = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    normal /= norm
    sun = np.array([np.cos(sun_el) * np.sin(sun_az),
                    np.cos(sun_el) * np.cos(sun_az),
                    np.sin(sun_el)])
    hs = np.einsum("ijk,k->ij", normal, sun)
    return np.clip(hs, 0, 1).astype(np.float32)


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Terrain ──────────────────────────────────────────────────────────────
    log.info("Loading terrain from '%s' …", args.terrain)
    terrain = _load_terrain(args.terrain)
    R, C = terrain.shape
    resolution_m = terrain.resolution_m
    log.info("  %d×%d  res=%.0f m", R, C, resolution_m)

    ignition_cell = _find_ignition(terrain)
    log.info("  Ignition: (%d, %d)", *ignition_cell)

    # ── Ground truth ─────────────────────────────────────────────────────────
    from wispsim.ground_truth import generate_ground_truth, WindEvent
    wind_events = [WindEvent(time_s=1800.0, direction_change=20.0,
                             speed_change=1.0, ramp_duration_s=300.0)]
    ground_truth = generate_ground_truth(
        terrain           = terrain,
        ignition_cell     = ignition_cell,
        base_fmc          = args.base_fmc,
        base_ws           = args.wind_speed,
        base_wd           = args.wind_direction,
        wind_events       = wind_events,
        seed              = args.seed,
        temperature_c     = args.temperature,
        relative_humidity = args.humidity,
    )

    # ── Static prior ensemble ─────────────────────────────────────────────────
    from angrybird.nelson import nelson_fmc_field
    from angrybird.types import GPPrior
    from angrybird.config import (
        GP_DEFAULT_FMC_VARIANCE, GP_DEFAULT_WIND_SPEED_MEAN,
        GP_DEFAULT_WIND_SPEED_VARIANCE, GP_DEFAULT_WIND_DIR_MEAN,
        GP_DEFAULT_WIND_DIR_VARIANCE,
    )

    _lat = float(terrain.origin_latlon[0]) if terrain.origin_latlon else 37.5
    _nelson = nelson_fmc_field(terrain, T_C=args.temperature, RH=args.humidity,
                               hour_of_day=14.0, latitude_deg=_lat)
    prior = GPPrior(
        fmc_mean            = _nelson.astype(np.float32),
        fmc_variance        = np.full(terrain.shape, GP_DEFAULT_FMC_VARIANCE,        dtype=np.float32),
        wind_speed_mean     = np.full(terrain.shape, GP_DEFAULT_WIND_SPEED_MEAN,     dtype=np.float32),
        wind_speed_variance = np.full(terrain.shape, GP_DEFAULT_WIND_SPEED_VARIANCE, dtype=np.float32),
        wind_dir_mean       = np.full(terrain.shape, GP_DEFAULT_WIND_DIR_MEAN,       dtype=np.float32),
        wind_dir_variance   = np.full(terrain.shape, GP_DEFAULT_WIND_DIR_VARIANCE,   dtype=np.float32),
    )

    fire_engine = _make_fire_engine(terrain, args.device)
    log.info("Running static prior ensemble (%d members) …", args.members)
    rng = np.random.default_rng(0)
    ensemble = fire_engine.run(
        terrain,
        prior,
        ground_truth.fire.fire_state.copy(),
        n_members   = args.members,
        horizon_min = args.horizon_min,
        rng         = rng,
    )
    burn_prob = ensemble.burn_probability   # (R, C) float in [0, 1]
    log.info("  Ensemble ready — burn_prob range [%.2f, %.2f]",
             float(burn_prob.min()), float(burn_prob.max()))

    # ── Advance ground truth to full duration ─────────────────────────────────
    total_s  = args.hours * 3600.0
    dt       = 10.0
    n_steps  = int(total_s / dt)
    render_every = max(1, int(args.frame_interval))

    log.info("Pre-stepping ground truth for %d h …", args.hours)
    from wispsim.ground_truth import compute_wind_field
    _rng2 = np.random.default_rng(1)
    for step in range(n_steps):
        ws, wd = compute_wind_field(
            ground_truth.base_wind_speed,
            ground_truth.base_wind_direction,
            terrain,
            step * dt,
            ground_truth.wind_events,
            rng=_rng2,
        )
        ground_truth.wind_speed     = ws
        ground_truth.wind_direction = wd
        ground_truth.fire.step(dt, ws, wd, ground_truth.fmc)

    arrival_times_s = ground_truth.fire.arrival_times   # seconds from t=0
    log.info("  Ground truth stepped — %.0f cells burned",
             float(np.isfinite(arrival_times_s).sum()))

    # ── Render frames ─────────────────────────────────────────────────────────
    hs = _hillshade(terrain.elevation, resolution_m)
    horizon_s = args.horizon_min * 60.0
    _vmin_h = 0.0
    _vmax_h = args.horizon_min / 60.0   # hours

    log.info("Rendering %d frames …", n_steps // render_every + 1)
    frame_idx = 0

    fig = plt.figure(figsize=(16, 7), facecolor="#12121f")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12)
    ax_truth = fig.add_subplot(gs[0, 0])
    ax_prior = fig.add_subplot(gs[0, 1])
    fig.patch.set_facecolor("#12121f")

    for ax, title in ((ax_truth, "Ground Truth: Fire Spread"),
                      (ax_prior, "Static Prior: Ensemble Burn Probability")):
        ax.set_facecolor("#12121f")
        ax.imshow(hs, origin="upper", cmap="gray", vmin=0, vmax=1,
                  alpha=0.5, interpolation="bilinear")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("East →", color="#aaaaaa", fontsize=7)
        ax.set_ylabel("↑ North", color="#aaaaaa", fontsize=7)
        ax.tick_params(colors="#aaaaaa", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # Static burn probability panel — drawn once, never changes
    _bp_img = ax_prior.imshow(burn_prob, origin="upper", cmap="YlOrRd",
                               alpha=0.75, vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(_bp_img, ax=ax_prior, fraction=0.03, pad=0.02,
                 label="Burn probability").ax.tick_params(labelsize=7)

    # Truth panel dynamic artists (cleared each frame)
    _truth_artists: list = []

    _render_times = list(range(0, n_steps + 1, render_every))

    for step in _render_times:
        current_s = step * dt

        # Clear previous truth frame artists
        for a in _truth_artists:
            try:
                if hasattr(a, "collections"):
                    for c in a.collections:
                        try: c.remove()
                        except Exception: pass
                else:
                    a.remove()
            except Exception:
                pass
        _truth_artists.clear()

        # Compute arrival hours relative to now (for colour scale)
        with np.errstate(invalid="ignore"):
            rel_h = (arrival_times_s - current_s) / 3600.0
        rel_h_clamped = np.where(np.isfinite(rel_h), rel_h, np.nan)

        # Burned cells: arrival_time <= current_time
        burned_mask = (arrival_times_s <= current_s) & np.isfinite(arrival_times_s)

        # Colour the imminent-to-future fire spread
        future_mask = np.isfinite(rel_h) & (rel_h >= 0)
        if future_mask.any():
            _arr = np.where(future_mask, rel_h_clamped, np.nan)
            im = ax_truth.imshow(_arr, origin="upper", cmap="hot_r",
                                 alpha=0.6, vmin=0, vmax=_vmax_h,
                                 interpolation="nearest")
            _truth_artists.append(im)

        # Orange fill for burned area
        if burned_mask.any():
            _burnt_rgba = np.zeros((*burned_mask.shape, 4), dtype=np.float32)
            _burnt_rgba[burned_mask] = [0.9, 0.4, 0.1, 0.7]
            im2 = ax_truth.imshow(_burnt_rgba, origin="upper", interpolation="nearest")
            _truth_artists.append(im2)
            # Perimeter contour
            cl = ax_truth.contour(burned_mask.astype(float), levels=[0.5],
                                   colors=["#FF4400"], linewidths=1.8)
            _truth_artists.append(cl)

        # Time annotation
        _t_min = current_s / 60.0
        ann = ax_truth.text(
            0.02, 0.97, f"t = {_t_min:.0f} min",
            transform=ax_truth.transAxes, color="white",
            fontsize=9, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#22223baa", ec="none"),
        )
        _truth_artists.append(ann)

        frame_path = frames_dir / f"frame_{frame_idx:05d}.png"
        fig.savefig(frame_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        frame_idx += 1

    plt.close(fig)
    log.info("Rendered %d frames → %s", frame_idx, frames_dir)

    # ── Assemble video ─────────────────────────────────────────────────────────
    out_video = out_dir / "prior_vs_truth.mp4"
    try:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_video),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        log.info("Video saved → %s", out_video)
    except Exception as exc:
        log.warning("Video assembly failed (%s). Frames are in %s.", exc, frames_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualise static prior vs ground truth fire spread",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--terrain",          default="landfire_cache")
    p.add_argument("--hours",            type=float, default=2.0)
    p.add_argument("--members",          type=int,   default=50)
    p.add_argument("--horizon-min",      type=int,   default=240)
    p.add_argument("--base-fmc",         type=float, default=0.08)
    p.add_argument("--wind-speed",       type=float, default=5.0)
    p.add_argument("--wind-direction",   type=float, default=225.0)
    p.add_argument("--temperature",      type=float, default=32.0)
    p.add_argument("--humidity",         type=float, default=0.20)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--device",           default="mps",
                   choices=["cpu", "mps", "cuda"])
    p.add_argument("--frame-interval",   type=int,   default=60,
                   help="Render one frame every N simulation steps (10 s each)")
    p.add_argument("--fps",              type=int,   default=10)
    p.add_argument("--out",              default="out/prior_viz")
    return p


if __name__ == "__main__":
    main(_build_parser().parse_args())
