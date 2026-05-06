"""
compare_selectors.py — Compare drone path-selection strategies over a 2-hour run.

Runs the following strategies with a fixed 3-drone fleet and produces a combined
comparison chart + CSV updated after each variant completes:

  static prior      — no drones; frozen ensemble baseline
  lawnmower         — systematic boustrophedon strip sweep
  fire_front_orbit  — orbit fire centroid at fixed standoff radius
  fire_front        — evenly-spaced points along the fire perimeter
  greedy            — marginal-gain greedy (info field)
  correlation_path  — terrain-adaptive domain path planner

Usage
-----
    PYTHONPATH=. python scripts/compare_selectors.py
    PYTHONPATH=. python scripts/compare_selectors.py --out out/selectors --hours 2
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", message="Mean of empty slice",       category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-4s  %(name)-14s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
    level=logging.INFO,
)
for _noisy in ("matplotlib", "PIL", "rasterio", "numexpr"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("compare_selectors")

# ── Project imports ────────────────────────────────────────────────────────────

from angrybird.config import (
    GP_DEFAULT_FMC_VARIANCE,
    GP_DEFAULT_WIND_DIR_MEAN, GP_DEFAULT_WIND_DIR_VARIANCE,
    GP_DEFAULT_WIND_SPEED_MEAN, GP_DEFAULT_WIND_SPEED_VARIANCE,
    TAU_FMC_S, TAU_WIND_SPEED_S, TAU_WIND_DIR_S,
)
from angrybird.gp import IGNISGPPrior
from angrybird.nelson import nelson_fmc_field
from angrybird.observations import (
    FireDetectionObservation, ObservationStore, ObservationType,
)
from angrybird.orchestrator import IGNISOrchestrator
from angrybird.types import GPPrior

from wispsim.ground_truth import WindEvent, generate_ground_truth
from wispsim.runner import SimulationConfig, SimulationRunner
from wispsim.static_prior_evaluator import StaticPriorEvaluator

from run import (
    find_burnable_cell,
    cells_within_radius,
    make_gp,
    make_registry,
    seed_fire_report,
    _load_landfire,
    _make_fire_engine,
)

# ── Variant definitions ────────────────────────────────────────────────────────

_VARIANTS = [
    {"label": "lawnmower",        "selector": "lawnmower"},
    {"label": "fire front orbit", "selector": "fire_front_orbit"},
    {"label": "fire front",       "selector": "fire_front"},
    {"label": "greedy",           "selector": "greedy"},
    {"label": "correlation path", "selector": "correlation_path"},
]

_COLORS: dict[str, tuple[str, str]] = {
    "static prior":   ("#9E9E9E", "--"),
    "lawnmower":      ("#A5D6A7", "-"),
    "fire front orbit": ("#FFB74D", "-"),
    "fire front":     ("#EF9A9A", "-"),
    "greedy":         ("#80DEEA", "-"),
    "correlation path": ("#1565C0", "-"),
}


# ── Chart helper ──────────────────────────────────────────────────────────────

def _save_chart(completed: dict[str, list[dict]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#12121f")

    for label, rows in completed.items():
        color, ls = _COLORS.get(label, ("#ffffff", "-"))
        t = [r["time_min"] for r in rows]
        v = [r["crps_minutes"] for r in rows]
        lw = 1.6 if label == "static prior" else 2.0
        ms = 3   if label == "static prior" else 4
        marker = "s" if label == "static prior" else "o"
        ax.plot(t, v, marker=marker, linestyle=ls, color=color,
                linewidth=lw, markersize=ms, label=label)

    ax.axhline(0.0, color="#555577", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Simulation time (min)", color="white", fontsize=9)
    ax.set_ylabel("Per-cell CRPS (min)", color="white", fontsize=9)
    ax.set_title("Model Error — selector strategy comparison (3 drones)",
                 color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.legend(fontsize=8, facecolor="#22223b", labelcolor="white",
              edgecolor="#444466", loc="upper right")
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Chart updated → %s", out_path.resolve())


def _save_csv(completed: dict[str, list[dict]], out_path: Path) -> None:
    all_rows = [r for rows in completed.values() for r in rows]
    if not all_rows:
        return
    keys = list(all_rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    log.info("CSV updated → %s", out_path.resolve())


# ── Core: run one variant ──────────────────────────────────────────────────────

def run_variant(
    *,
    label: str,
    selector_name: str,
    n_drones: int,
    terrain,
    ignition_cells: list[tuple[int, int]],
    base_cell: tuple[int, int],
    fire_engine,
    resolution_m: float,
    args: argparse.Namespace,
):
    log.info("")
    log.info("═══ Variant: %s (selector=%s, %d drone(s)) ═══",
             label, selector_name, n_drones)
    t_start = time.time()

    gp, obs_store = make_gp(terrain)

    wind_events = [WindEvent(time_s=1800.0, direction_change=20.0,
                             speed_change=1.0, ramp_duration_s=300.0)]
    ground_truth = generate_ground_truth(
        terrain           = terrain,
        ignition_cell     = ignition_cells[0] if len(ignition_cells) == 1 else ignition_cells,
        base_fmc          = args.base_fmc,
        base_ws           = args.wind_speed,
        base_wd           = args.wind_direction,
        wind_events       = wind_events,
        seed              = args.seed,
        temperature_c     = args.temperature,
        relative_humidity = args.humidity,
    )

    for ic in ignition_cells:
        seed_fire_report(obs_store, ic, terrain, args.fire_radius_m,
                         args.fire_confidence, timestamp=0.0)

    cycle_s     = args.cycle_min * 60.0
    horizon_min = args.horizon_min
    corr_len_m  = max(2000.0, 20 * resolution_m)

    registry = make_registry(
        terrain              = terrain,
        drone_speed_ms       = args.drone_speed_ms,
        drone_endurance_s    = args.drone_endurance_s,
        cycle_s              = cycle_s,
        correlation_length_m = corr_len_m,
        horizon_cycles       = float(horizon_min) / float(args.cycle_min),
    )

    orchestrator = IGNISOrchestrator(
        terrain           = terrain,
        gp                = gp,
        obs_store         = obs_store,
        fire_engine       = fire_engine,
        selector_name     = selector_name,
        selector_registry = registry,
        n_drones          = n_drones,
        n_targets         = n_drones * 3,
        staging_area      = base_cell,
        n_members         = args.members,
        horizon_min       = horizon_min,
        resolution_m      = resolution_m,
    )

    config = SimulationConfig(
        dt                     = 10.0,
        total_time_s           = args.hours * 3600.0,
        ignis_cycle_interval_s = cycle_s,
        n_drones               = n_drones,
        drone_speed_ms         = args.drone_speed_ms,
        drone_endurance_s      = args.drone_endurance_s,
        camera_footprint_m     = max(150.0, resolution_m * 1.5),
        base_cell              = base_cell,
        frame_interval         = 6,
        fps                    = 10,
        output_path            = f"{args.out}/{label.replace(' ', '_')}",
        scenario_name          = label.replace(" ", "_"),
        n_raws                 = args.n_raws,
        enable_mesh_network    = False,
        selector_name          = selector_name,
        live_fire_horizon_h    = float(horizon_min) / 60.0,
        drone_spawn_times_s    = [],
    )

    runner = SimulationRunner(
        config       = config,
        terrain      = terrain,
        ground_truth = ground_truth,
        orchestrator = orchestrator,
    )
    runner.run()

    rows = [{**r, "variant": label} for r in runner._cycle_metrics_rows]
    log.info("Variant '%s' done in %.0fs — %d cycles",
             label, time.time() - t_start, len(rows))
    return rows, orchestrator, ground_truth, runner


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_path = out_dir / "comparison.png"
    csv_path   = out_dir / "comparison.csv"

    log.info("Loading LANDFIRE terrain from '%s' …", args.terrain)
    terrain      = _load_landfire(args.terrain)
    resolution_m = terrain.resolution_m
    R, C = terrain.shape
    log.info("  Shape: %d×%d  res=%.0f m", R, C, resolution_m)

    ignition_cell = find_burnable_cell(terrain)
    log.info("  Ignition cell: (%d, %d)", *ignition_cell)

    _ic_row   = ignition_cell[0]
    _base_row = int(np.clip(_ic_row + R * 0.25, R * 0.10, R * 0.85))
    base_cell = (_base_row, C // 2)
    log.info("  Staging area: (%d, %d)", *base_cell)

    ignition_cells = [ignition_cell]

    log.info("Initialising fire engine …")
    fire_engine = _make_fire_engine(terrain, args.device)

    _init_lat     = float(terrain.origin_latlon[0]) if terrain.origin_latlon else 37.5
    _nelson       = nelson_fmc_field(
        terrain, T_C=args.temperature, RH=args.humidity,
        hour_of_day=14.0, latitude_deg=_init_lat,
    )
    initial_gp_prior = GPPrior(
        fmc_mean             = _nelson.astype(np.float32),
        fmc_variance         = np.full(terrain.shape, GP_DEFAULT_FMC_VARIANCE,        dtype=np.float32),
        wind_speed_mean      = np.full(terrain.shape, GP_DEFAULT_WIND_SPEED_MEAN,     dtype=np.float32),
        wind_speed_variance  = np.full(terrain.shape, GP_DEFAULT_WIND_SPEED_VARIANCE, dtype=np.float32),
        wind_dir_mean        = np.full(terrain.shape, GP_DEFAULT_WIND_DIR_MEAN,       dtype=np.float32),
        wind_dir_variance    = np.full(terrain.shape, GP_DEFAULT_WIND_DIR_VARIANCE,   dtype=np.float32),
    )

    completed: dict[str, list[dict]] = {}
    static_rows = None

    for v in _VARIANTS:
        rows, orchestrator, ground_truth, runner = run_variant(
            label          = v["label"],
            selector_name  = v["selector"],
            n_drones       = args.n_drones,
            terrain        = terrain,
            ignition_cells = ignition_cells,
            base_cell      = base_cell,
            fire_engine    = fire_engine,
            resolution_m   = resolution_m,
            args           = args,
        )
        completed[v["label"]] = rows

        if static_rows is None:
            log.info("Computing static prior baseline …")
            _se = StaticPriorEvaluator(
                config               = SimulationConfig(
                    ignis_cycle_interval_s = args.cycle_min * 60.0,
                    total_time_s           = args.hours * 3600.0,
                    scenario_name          = "static_prior",
                ),
                terrain              = terrain,
                ground_truth         = ground_truth,
                initial_gp_prior     = initial_gp_prior,
                fire_engine          = fire_engine,
                initial_fire_state   = ground_truth.fire.fire_state.copy(),
                n_members            = args.members,
                horizon_min          = args.horizon_min,
                initial_phi          = orchestrator._cycle1_initial_phi,
                oracle_arrival_times = runner._oracle_arrival_times,
            )
            static_rows = _se.evaluate()
            completed["static prior"] = static_rows
            log.info("Static prior baseline ready (%d rows)", len(static_rows))

        _save_chart(completed, chart_path)
        _save_csv(completed, csv_path)

    log.info("")
    log.info("All variants complete.")
    log.info("Chart: %s", chart_path.resolve())
    log.info("CSV:   %s", csv_path.resolve())


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare drone selector strategies over a 2-hour simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--terrain",           default="landfire_cache")
    p.add_argument("--hours",             type=float, default=2.0)
    p.add_argument("--cycle-min",         type=float, default=10.0)
    p.add_argument("--horizon-min",       type=int,   default=120)
    p.add_argument("--members",           type=int,   default=50)
    p.add_argument("--n-drones",          type=int,   default=3)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--base-fmc",          type=float, default=0.08)
    p.add_argument("--wind-speed",        type=float, default=5.0)
    p.add_argument("--wind-direction",    type=float, default=225.0)
    p.add_argument("--temperature",       type=float, default=32.0)
    p.add_argument("--humidity",          type=float, default=0.20)
    p.add_argument("--fire-radius-m",     type=float, default=300.0)
    p.add_argument("--fire-confidence",   type=float, default=0.80)
    p.add_argument("--n-raws",            type=int,   default=2)
    p.add_argument("--drone-speed-ms",    type=float, default=23.25)
    p.add_argument("--drone-endurance-s", type=float, default=48_462.0)
    p.add_argument("--device",            default="mps",
                   choices=["cpu", "mps", "cuda"])
    p.add_argument("--out",               default="out/selectors")
    return p


if __name__ == "__main__":
    main(_build_parser().parse_args())
