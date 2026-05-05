"""
run_landfire_wispsim.py — Full WISPsim on real LANDFIRE terrain with GPU fire engine.

Loads the cached Northern California LANDFIRE tile (249×322 cells, 100 m/cell,
~39.2°N −121.2°W) and runs the complete SimulationRunner loop:

  • Ground truth fire advancing via GroundTruthFire CA each 10 s step
  • Drone fleet collecting FMC + wind observations in flight
  • WISP cycles every 10 min (GPUFireEngine ensemble → info field → drone selection)
  • 6-panel frame renderer → PNG sequence + MP4

Duration: 1 hour (3600 s), 360 simulation steps, 60 rendered frames.

Usage
-----
    PYTHONPATH=. python scripts/run_landfire_wispsim.py
    PYTHONPATH=. python scripts/run_landfire_wispsim.py --device mps --members 20
    PYTHONPATH=. python scripts/run_landfire_wispsim.py --cache /path/to/landfire --out out/lf_run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from angrybird.config import TAU_FMC_S, TAU_WIND_SPEED_S, TAU_WIND_DIR_S
from angrybird.fire_engines.gpu_fire_engine import GPUFireEngine
from angrybird.gp import IGNISGPPrior
from angrybird.landfire import load_from_directory
from angrybird.observations import ObservationStore, ObservationType
from angrybird.orchestrator import IGNISOrchestrator
from angrybird.selectors.base import SelectorRegistry
from angrybird.selectors.greedy import GreedySelector
from angrybird.selectors.baselines import UniformSelector, FireFrontSelector
from angrybird.selectors.correlation_path import CorrelationPathSelector
from angrybird.types import TerrainData
from wispsim.ground_truth import generate_ground_truth, WindEvent
from wispsim.runner import SimulationConfig, SimulationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("run_landfire_wispsim")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gp(terrain: TerrainData) -> tuple[IGNISGPPrior, ObservationStore]:
    decay_config = {
        ObservationType.FMC:            TAU_FMC_S,
        ObservationType.WIND_SPEED:     TAU_WIND_SPEED_S,
        ObservationType.WIND_DIRECTION: TAU_WIND_DIR_S,
    }
    try:
        obs_store = ObservationStore(decay_config)
    except TypeError:
        obs_store = ObservationStore()

    gp = IGNISGPPrior(obs_store, terrain=terrain, resolution_m=terrain.resolution_m)
    return gp, obs_store


def find_ignition(terrain: TerrainData) -> tuple[int, int]:
    """
    Same cell as the static landfire_sim.py: centre of grid, choose
    a burnable SB40 fuel.  Scan around (R/2, C/2) for the first non-NB cell.
    """
    NB_CODES = {91, 92, 93, 98, 99}
    R, C = terrain.shape
    r0, c0 = R // 2, C // 2
    for dr in range(0, max(R, C)):
        for dc in range(0, dr + 1):
            for r, c in {(r0 + dr, c0 + dc), (r0 - dr, c0 + dc),
                         (r0 + dr, c0 - dc), (r0 - dr, c0 - dc)}:
                if 0 <= r < R and 0 <= c < C:
                    if int(terrain.fuel_model[r, c]) not in NB_CODES:
                        return (r, c)
    return (r0, c0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    cache_dir: str = "landfire_cache",
    device: str = "mps",
    n_drones: int = 2,
    n_targets: int = 6,
    n_members: int = 20,
    hours: float = 1.0,
    out_dir: str = "out/landfire_wispsim",
    selector: str = "correlation_path",
) -> None:
    t0 = time.time()

    # ── Resolve cache_dir relative to repo root if it doesn't exist as-is ──
    _cache = Path(cache_dir)
    if not _cache.is_absolute() and not _cache.is_dir():
        _cache = Path(__file__).resolve().parent.parent / cache_dir
    cache_dir = str(_cache)

    # ── Load LANDFIRE terrain ─────────────────────────────────────────────
    log.info("Loading LANDFIRE terrain from '%s' …", cache_dir)
    terrain = load_from_directory(cache_dir, resolution_m=100.0)
    R, C = terrain.shape
    log.info(
        "  Shape: %d×%d  |  %.1f km × %.1f km  |  origin: %.3f°N %.3f°W",
        R, C,
        R * terrain.resolution_m / 1000,
        C * terrain.resolution_m / 1000,
        terrain.origin_latlon[0] if terrain.origin_latlon else 0.0,
        -terrain.origin_latlon[1] if terrain.origin_latlon else 0.0,
    )

    # ── Ignition and ground truth ─────────────────────────────────────────
    ignition = find_ignition(terrain)
    log.info(
        "  Ignition cell: (%d, %d)  fuel=%d  slope=%.1f°  elev=%.0f m",
        ignition[0], ignition[1],
        terrain.fuel_model[ignition],
        terrain.slope[ignition],
        terrain.elevation[ignition],
    )

    # SW wind (225°), moderate speed; one wind event at 30 min
    wind_events = [
        WindEvent(
            time_s=1800.0,          # 30 min
            direction_change=20.0,  # veer slightly clockwise
            speed_change=1.0,       # gust
            ramp_duration_s=300.0,
        ),
    ]

    ground_truth = generate_ground_truth(
        terrain=terrain,
        ignition_cell=ignition,
        base_fmc=0.08,
        base_ws=5.0,
        base_wd=225.0,   # SW synoptic — consistent with prev landfire_sim
        wind_events=wind_events,
        seed=42,
    )

    # ── GP + ObservationStore ─────────────────────────────────────────────
    gp, obs_store = make_gp(terrain)

    # ── GPU fire engine ───────────────────────────────────────────────────
    log.info("Initialising GPUFireEngine (device=%s) …", device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fire_engine = GPUFireEngine(terrain, device=device, target_cfl=0.7)
    log.info("  Engine ready.")

    # ── Orchestrator ──────────────────────────────────────────────────────
    # Drone base: 10% inset from south edge, centre column — inside the map
    base_cell = (int(R * 0.90), C // 2)

    # Build a selector registry with the correct resolution and drone parameters
    # for this terrain.  The default global registry uses GRID_RESOLUTION_M=50 m
    # and default drone specs; LANDFIRE runs use 100 m cells and a different drone.
    res            = terrain.resolution_m
    drone_speed    = 23.25                          # m/s (52 mph)
    cycle_s        = 600.0                          # 10-min IGNIS cycle
    endurance_s    = 48_462.0                       # ~700 mi range @ 52 mph
    d_cycle_m      = drone_speed * cycle_s          # 13 950 m per cycle
    drone_range_m  = drone_speed * endurance_s      # ~1 128 km total endurance

    local_registry = SelectorRegistry()
    local_registry.register(GreedySelector(resolution_m=res))
    local_registry.register(UniformSelector())
    local_registry.register(FireFrontSelector())
    local_registry.register(CorrelationPathSelector(
        drone_range_m=drone_range_m,
        d_cycle_m=d_cycle_m,
        correlation_length_m=2000.0,   # ~2 km for 100 m cells (≈ 20-cell domains)
        min_domain_cells=15,
    ))

    # Planning horizon: 4 hours ahead, even though the sim runs for 1 hour.
    # With a 1-hour horizon the fire footprint (~475 cells) is too small —
    # the sensitivity zone (perimeter) sits inside the observability-degraded
    # zone and w≈0 everywhere.  A 4-hour ensemble gives a footprint of ~5000
    # cells with a wide uncertain buffer, producing a meaningful info field.
    orchestrator = IGNISOrchestrator(
        terrain=terrain,
        gp=gp,
        obs_store=obs_store,
        fire_engine=fire_engine,
        selector_name=selector,
        selector_registry=local_registry,
        n_drones=n_drones,
        n_targets=n_targets,      # select more points than drones for multi-waypoint paths
        staging_area=base_cell,
        n_members=n_members,
        horizon_min=240,          # 4-hour planning horizon
        resolution_m=res,         # 100 m — used by compute_information_field
    )

    # ── Simulation config ─────────────────────────────────────────────────
    config = SimulationConfig(
        dt=10.0,
        total_time_s=hours * 3600.0,
        ignis_cycle_interval_s=600.0,   # WISP every 10 min → 6 cycles/hour
        n_drones=n_drones,
        drone_speed_ms=23.25,           # 52 mph → 23.25 m/s
        drone_endurance_s=48462.0,      # 700 mi range @ 52 mph → 48 462 s
        camera_footprint_m=150.0,       # wider footprint for 100 m cells
        base_cell=base_cell,
        frame_interval=6,               # render every 60 s of sim time
        fps=10,
        output_path=out_dir,
        scenario_name="landfire_1h",
        n_raws=2,
        enable_mesh_network=False,      # instant delivery: obs added as made
    )

    # ── Run ───────────────────────────────────────────────────────────────
    log.info(
        "Starting SimulationRunner: %.1f h | dt=%.0f s | %d drones | %d members",
        hours, config.dt, n_drones, n_members,
    )

    runner = SimulationRunner(
        config=config,
        terrain=terrain,
        ground_truth=ground_truth,
        orchestrator=orchestrator,
    )

    reports = runner.run()

    elapsed = time.time() - t0
    out     = runner.renderer.out_dir
    frames  = runner.renderer._frame_count
    cycles  = len(reports)

    log.info(
        "Done. %d WISP cycles | %d frames | %.1f s wall-clock",
        cycles, frames, elapsed,
    )
    log.info("Frames: %s", out.resolve())

    video = out.parent / f"{out.name}.mp4"
    if video.exists():
        log.info("Video:  %s  (%.1f MB)", video.resolve(), video.stat().st_size / 1e6)
    else:
        log.info("MP4 not produced (ffmpeg not available) — PNG frames in %s", out)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WISPsim on LANDFIRE terrain with GPU engine")
    parser.add_argument("--cache",   default="landfire_cache",
                        help="Path to LANDFIRE GeoTIFF cache directory (default: landfire_cache)")
    parser.add_argument("--device",  default="mps", choices=["cpu", "mps", "cuda"],
                        help="PyTorch device for GPUFireEngine (default: cpu)")
    parser.add_argument("--drones",  type=int, default=1,
                        help="Number of drones (default: 1)")
    parser.add_argument("--targets", type=int, default=6,
                        help="Waypoints selected per cycle (default: 6)")
    parser.add_argument("--members", type=int, default=200,
                        help="Ensemble size (default: 20)")
    parser.add_argument("--hours",   type=float, default=1.0,
                        help="Simulation duration in hours (default: 1.0)")
    parser.add_argument("--out",      default="out/landfire_wispsim",
                        help="Output directory (default: out/landfire_wispsim)")
    parser.add_argument("--selector", default="correlation_path",
                        choices=["correlation_path", "greedy", "uniform", "fire_front"],
                        help="Path/point selector strategy (default: correlation_path)")
    args = parser.parse_args()

    main(
        cache_dir=args.cache,
        device=args.device,
        n_drones=args.drones,
        n_targets=args.targets,
        n_members=args.members,
        hours=args.hours,
        out_dir=args.out,
        selector=args.selector,
    )
