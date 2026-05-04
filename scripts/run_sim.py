"""
run_sim.py — Clock-based IGNIS simulation with animated video output.

PRIMARY DEMO SCRIPT. Runs the full SimulationRunner:
  - Unified simulation clock (dt=10 s, 6 h total by default)
  - Dynamic wind evolution + scheduled wind-shift events (wind_shift scenario)
  - CA-based ground truth fire advancing every timestep
  - Simulated drone fleet: movement, FMC + wind sensor collection
  - IGNIS cycles every 20 minutes (ensemble → info field → selection → assimilation)
  - 6-panel frame renderer → PNG sequence + MP4 video

Scenarios (auto-discovered from scenarios.py — add a function there to register it)
---------
  hilly_heterogeneous  (default)  Complex ridge/valley terrain, mixed fuels, SW wind + event
  wind_shift                      Same terrain + 45° wind shift at hour 3
  flat_homogeneous                Flat/uniform control — minimal PERR advantage expected
  dual_ignition                   Two simultaneous fires, 1-hour, 45° wind event at 30 min
  crown_fire_risk                 Dense timber, high FMC, crown fire conditions

Usage
-----
    cd /path/to/AngryBird
    python scripts/run_sim.py                                        # hilly_heterogeneous, 6 h
    python scripts/run_sim.py --scenario wind_shift                  # wind-shift stress test
    python scripts/run_sim.py --scenario dual_ignition --hours 1     # dual-ignition, 1 h
    python scripts/run_sim.py --drones 3 --hours 1                   # quick 1-hour test run
    python scripts/run_sim.py --out out/my_run                       # custom output directory

No real terrain, no QPU, no cloud required.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inspect
from angrybird.config import TAU_FMC_S, TAU_WIND_SPEED_S, TAU_WIND_DIR_S
from angrybird.gp import IGNISGPPrior
from angrybird.observations import ObservationStore, ObservationType
from angrybird.orchestrator import IGNISOrchestrator
from angrybird.types import TerrainData
import simulation.scenarios as _scenario_module
from simulation import SimpleFire, SimulationConfig, SimulationRunner


def make_gp(terrain: TerrainData) -> tuple[IGNISGPPrior, ObservationStore]:
    """Return an (IGNISGPPrior, ObservationStore) pair backed by the same store."""
    decay_config = {
        ObservationType.FMC:            TAU_FMC_S,
        ObservationType.WIND_SPEED:     TAU_WIND_SPEED_S,
        ObservationType.WIND_DIRECTION: TAU_WIND_DIR_S,
    }
    obs_store = ObservationStore(decay_config)
    gp = IGNISGPPrior(obs_store, terrain=terrain, resolution_m=terrain.resolution_m)
    return gp, obs_store

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("run_sim")


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

# Auto-discover scenario factories defined in scenarios.py — no manual registration needed.
# Adding a new scenario only requires writing the function there; it appears here automatically.
_SCENARIOS: dict = {
    name: fn
    for name, fn in inspect.getmembers(_scenario_module, inspect.isfunction)
    if not name.startswith("_") and fn.__module__ == _scenario_module.__name__
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    scenario: str = "hilly_heterogeneous",
    n_drones: int = 5,
    n_members: int = 30,
    hours: float = 6.0,
    out_dir: str | None = None,
    frame_interval: int | None = None,
) -> None:

    t0 = time.time()

    # ── scenario ──────────────────────────────────────────────────────────
    factory = _SCENARIOS[scenario]
    terrain, ground_truth, config = factory(seed=42)

    # Override hours and drones from CLI
    config.total_time_s = hours * 3600.0
    config.n_drones     = n_drones
    if out_dir:
        config.output_path = out_dir
    if frame_interval is not None:
        config.frame_interval = frame_interval

    log.info(
        "=== run_sim | scenario=%s | %.1f h | dt=%.0f s | %d drones | %d members ===",
        scenario, hours, config.dt, config.n_drones, n_members,
    )

    # ── IGNIS components ──────────────────────────────────────────────────
    gp, obs_store = make_gp(terrain)   # make_gp returns (IGNISGPPrior, ObservationStore)
    fire_engine   = SimpleFire()

    rows, cols   = terrain.shape
    staging_area = config.base_cell

    orchestrator = IGNISOrchestrator(
        terrain=terrain,
        gp=gp,
        obs_store=obs_store,
        fire_engine=fire_engine,
        selector_name="greedy",
        n_drones=config.n_drones,
        staging_area=staging_area,
        n_members=n_members,
        horizon_min=360,         # 6-hour fire spread horizon
    )

    # ── run ───────────────────────────────────────────────────────────────
    runner = SimulationRunner(
        config=config,
        terrain=terrain,
        ground_truth=ground_truth,
        orchestrator=orchestrator,
    )

    reports = runner.run()

    elapsed = time.time() - t0
    out    = runner.renderer.out_dir          # actual dir chosen (may have _N suffix)
    frames = runner.renderer._frame_count
    cycles = len(reports)

    log.info(
        "Done. %d IGNIS cycles | %d frames | %.1f s wall-clock | output: %s",
        cycles, frames, elapsed, out.resolve(),
    )

    video = out.parent / f"{out.name}.mp4"
    if video.exists():
        log.info("Video: %s  (%.1f MB)", video, video.stat().st_size / 1e6)
    else:
        log.info("PNG frames in %s  (ffmpeg not available — no MP4 assembled)", out)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IGNIS clock-based simulation")
    parser.add_argument(
        "--scenario", choices=list(_SCENARIOS), default="hilly_heterogeneous",
        help=f"Scenario name — auto-discovered from scenarios.py: {sorted(_SCENARIOS)}",
    )
    parser.add_argument("--drones",  type=int,   default=5,
                        help="Number of drones (default 5)")
    parser.add_argument("--members", type=int,   default=30,
                        help="Ensemble size (default 30; lower = faster)")
    parser.add_argument("--hours",   type=float, default=6.0,
                        help="Simulation duration in hours (default 6)")
    parser.add_argument("--out",     default=None,
                        help="Output directory (overrides scenario default)")
    parser.add_argument("--frame-interval", type=int, default=None,
                        help="Override frame_interval (steps between renders; default from scenario config)")
    args = parser.parse_args()

    main(
        scenario=args.scenario,
        n_drones=args.drones,
        n_members=args.members,
        hours=args.hours,
        out_dir=args.out,
        frame_interval=args.frame_interval,
    )
