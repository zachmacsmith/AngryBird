"""
benchmark_wispsim.py — Full WispSim stack profiler.

Profiles every layer of the simulation independently, then runs a
short clock-based sim and reports the step-bucket breakdown.

Components timed:
  1. Ground-truth fire CA step       (wispsim/fire_oracle.py)
  2. Wind field update               (wispsim/ground_truth.py)
  3. GP predict                      (angrybird/gp.py)
  4. Renderer frame                  (wispsim/renderer.py)
  5. compute_information_field       (angrybird/information.py)
  6. LiveEstimator.compute_estimate  (wispsim/runner.py) — GP fork + 1 fire member
  7. Mesh network step               (wispsim/network.py)
  8. Full SimulationRunner loop      (360 steps, 6 WISP cycles) — step-bucket summary

Usage
-----
    PYTHONPATH=. python3 scripts/benchmark_wispsim.py
    PYTHONPATH=. python3 scripts/benchmark_wispsim.py --device mps --members 20 --hours 0.25
    PYTHONPATH=. python3 scripts/benchmark_wispsim.py --skip-runner  # component benchmarks only
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from angrybird.config import (
    TAU_FMC_S, TAU_WIND_SPEED_S, TAU_WIND_DIR_S,
    GP_DEFAULT_WIND_SPEED_MEAN, GP_DEFAULT_WIND_DIR_MEAN,
)
from angrybird.fire_engines.gpu_fire_engine import GPUFireEngine
from angrybird.gp import IGNISGPPrior
from angrybird.information import compute_information_field
from angrybird.landfire import load_from_directory
from angrybird.observations import ObservationStore, ObservationType
from angrybird.orchestrator import IGNISOrchestrator
from angrybird.selectors.base import SelectorRegistry
from angrybird.selectors.greedy import GreedySelector
from angrybird.selectors.baselines import UniformSelector, FireFrontSelector
from wispsim.ground_truth import generate_ground_truth, compute_wind_field, WindEvent
from wispsim.runner import SimulationConfig, SimulationRunner, LiveEstimator
from wispsim.renderer import FrameRenderer
from wispsim.network import PingMeshNetwork, make_improved_mesh_config
from wispsim.drone_sim import DroneState, cell_to_pos_m

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

HEADER  = "\033[1;36m"
SECTION = "\033[1;33m"
OK      = "\033[0;32m"
WARN    = "\033[0;33m"
RESET   = "\033[0m"

TRIALS = 5   # micro-benchmark repetitions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_row(label: str, val: float, total: float, unit: str = "s") -> str:
    pct = 100.0 * val / total if total > 0 else 0.0
    bar = "█" * int(pct / 3)
    return f"  {label:<34s} {val:7.3f}{unit}  {pct:5.1f}%  {bar}"


def median_time(fn, n: int = TRIALS) -> float:
    times = []
    for _ in range(n):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[n // 2]


def make_stack(terrain, device: str, n_members: int):
    """Build the full orchestrator + fire engine stack."""
    R, C = terrain.shape
    res  = terrain.resolution_m

    decay_config = {
        ObservationType.FMC:            TAU_FMC_S,
        ObservationType.WIND_SPEED:     TAU_WIND_SPEED_S,
        ObservationType.WIND_DIRECTION: TAU_WIND_DIR_S,
    }
    try:
        obs_store = ObservationStore(decay_config)
    except TypeError:
        obs_store = ObservationStore()

    gp = IGNISGPPrior(obs_store, terrain=terrain, resolution_m=res)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fire_engine = GPUFireEngine(terrain, device=device, target_cfl=0.7)

    base_cell = (R - 1, C // 2)
    local_reg = SelectorRegistry()
    local_reg.register(GreedySelector(resolution_m=res))
    local_reg.register(UniformSelector())
    local_reg.register(FireFrontSelector())

    orch = IGNISOrchestrator(
        terrain=terrain,
        gp=gp,
        obs_store=obs_store,
        fire_engine=fire_engine,
        selector_name="greedy",
        selector_registry=local_reg,
        n_drones=2,
        n_targets=6,
        staging_area=base_cell,
        n_members=n_members,
        horizon_min=240,
        resolution_m=res,
    )
    return orch, fire_engine, obs_store, gp


# ---------------------------------------------------------------------------
# 1. Ground-truth fire CA step
# ---------------------------------------------------------------------------

def bench_fire_ca(terrain, ground_truth, n_steps: int = 50) -> float:
    """Median time for a single fire_oracle.step() call."""
    fire = ground_truth.fire
    ws   = ground_truth.wind_speed
    wd   = ground_truth.wind_direction
    fmc  = ground_truth.fmc

    # Prime — advance to ~10 min so there are active cells to process
    for _ in range(60):
        fire.step(10.0, ws, wd, fmc)

    active = int((fire.fire_state > 0).sum())
    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        fire.step(10.0, ws, wd, fmc)
        times.append(time.perf_counter() - t0)

    med = sorted(times)[len(times) // 2]
    return med, active


# ---------------------------------------------------------------------------
# 2. Wind field update
# ---------------------------------------------------------------------------

def bench_wind(terrain, ground_truth) -> float:
    rng = np.random.default_rng(0)
    return median_time(lambda: compute_wind_field(
        ground_truth.base_wind_speed,
        ground_truth.base_wind_direction,
        terrain, 300.0,
        ground_truth.wind_events,
        rng=rng,
    ))


# ---------------------------------------------------------------------------
# 3. GP predict
# ---------------------------------------------------------------------------

def bench_gp_predict(gp, shape) -> float:
    return median_time(lambda: gp.predict(shape))


# ---------------------------------------------------------------------------
# 4. Renderer frame (single savefig)
# ---------------------------------------------------------------------------

def bench_renderer(terrain, ground_truth, orch, tmp_dir: Path) -> float:
    renderer = FrameRenderer(
        terrain=terrain,
        out_dir=str(tmp_dir / "bench_render"),
        frame_interval=1,
        fps=10,
        horizon_h=4.0,
    )
    shape = terrain.shape
    gp_prior = orch.gp.predict(shape)
    zeros = np.zeros(shape, dtype=np.float32)

    # First call triggers figure creation — exclude from timing
    renderer.render_frame(
        step=0, time_s=0.0,
        ground_truth=ground_truth,
        drones=[],
        gp_prior=gp_prior,
        burn_probability=zeros,
        info_field=None,
        mission_targets=[],
        drone_plans=None,
        cycle_reports=[],
        live_gp_prior=gp_prior,
        live_arrival_times_h=None,
    )

    times = []
    for i in range(TRIALS):
        t0 = time.perf_counter()
        renderer.render_frame(
            step=i + 1, time_s=(i + 1) * 60.0,
            ground_truth=ground_truth,
            drones=[],
            gp_prior=gp_prior,
            burn_probability=zeros,
            info_field=None,
            mission_targets=[],
            drone_plans=None,
            cycle_reports=[],
            live_gp_prior=gp_prior,
            live_arrival_times_h=None,
        )
        times.append(time.perf_counter() - t0)

    import matplotlib.pyplot as plt
    plt.close(renderer.fig)
    return sorted(times)[len(times) // 2]


# ---------------------------------------------------------------------------
# 5. compute_information_field  (numpy only, cheap path)
# ---------------------------------------------------------------------------

def bench_info_field(terrain, orch, fire_engine, n_members: int) -> float:
    R, C = terrain.shape
    rng  = np.random.default_rng(0)
    gp_prior = orch.gp.predict((R, C))
    fire_state = np.zeros((R, C), dtype=np.float32)
    fire_state[R // 2, C // 2] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble = fire_engine.run(
            terrain, gp_prior, fire_state, n_members, 240, rng
        )

    return median_time(lambda: compute_information_field(
        ensemble, gp_prior,
        resolution_m=terrain.resolution_m,
        horizon_minutes=240,
    ))


# ---------------------------------------------------------------------------
# 6. LiveEstimator.compute_estimate  (GP fork + 1 fire member)
# ---------------------------------------------------------------------------

def bench_live_estimator(terrain, orch, ground_truth) -> float:
    live_est = LiveEstimator(orchestrator=orch, terrain=terrain, horizon_h=1.0)
    shape = terrain.shape
    fire_state = ground_truth.fire.fire_state.copy()
    rng = np.random.default_rng(0)

    # Warmup
    live_est.compute_estimate(shape, fire_state, rng)

    times = []
    for i in range(TRIALS):
        gc.collect()
        t0 = time.perf_counter()
        live_est.compute_estimate(shape, fire_state, np.random.default_rng(i))
        times.append(time.perf_counter() - t0)

    return sorted(times)[len(times) // 2]


# ---------------------------------------------------------------------------
# 7. Mesh network step
# ---------------------------------------------------------------------------

def bench_network(terrain) -> float:
    R, C = terrain.shape
    res  = terrain.resolution_m
    base_pos = cell_to_pos_m((R - 1, C // 2), res)
    net = PingMeshNetwork(
        ground_station_position=base_pos,
        drone_ids=["drone_0", "drone_1"],
        config=make_improved_mesh_config(),
        rng=np.random.default_rng(0),
    )
    drone_positions = {
        "drone_0": cell_to_pos_m((R // 2, C // 2), res),
        "drone_1": cell_to_pos_m((R // 4, C // 4), res),
    }
    return median_time(lambda: net.step(drone_positions, current_time=300.0))


# ---------------------------------------------------------------------------
# 8. Full SimulationRunner
# ---------------------------------------------------------------------------

def run_full_sim(
    terrain, device: str, n_members: int, hours: float, tmp_dir: Path
) -> None:
    R, C = terrain.shape
    orch, fire_engine, obs_store, gp = make_stack(terrain, device, n_members)

    ignition = (R // 2, C // 2)
    ground_truth = generate_ground_truth(
        terrain=terrain,
        ignition_cell=ignition,
        base_fmc=0.08,
        base_ws=5.0,
        base_wd=225.0,
        wind_events=[WindEvent(1800.0, 20.0, 1.0, 300.0)],
        seed=42,
    )

    config = SimulationConfig(
        dt=10.0,
        total_time_s=hours * 3600.0,
        ignis_cycle_interval_s=600.0,
        n_drones=2,
        drone_speed_ms=23.25,
        drone_endurance_s=48462.0,
        camera_footprint_m=150.0,
        base_cell=(R - 1, C // 2),
        frame_interval=6,
        fps=10,
        output_path=str(tmp_dir / "runner_out"),
        scenario_name="benchmark",
        n_raws=2,
        enable_mesh_network=False,
    )

    # Enable INFO so the step-bucket summary prints
    logging.getLogger("wispsim.runner").setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    runner = SimulationRunner(
        config=config,
        terrain=terrain,
        ground_truth=ground_truth,
        orchestrator=orch,
    )

    print(f"\n  Running {hours:.2f}h sim ({int(hours*3600/10)} steps, "
          f"{int(hours*6)} WISP cycles, N={n_members})...", flush=True)
    t0 = time.perf_counter()
    reports = runner.run()
    elapsed = time.perf_counter() - t0

    print(f"\n  {OK}Wall clock: {elapsed:.1f}s for {hours:.2f}h sim "
          f"({elapsed/(hours*3600):.2f}s wall per sim-second){RESET}")
    print(f"  Real-time factor: {(hours*3600)/elapsed:.1f}× "
          f"({'faster' if elapsed < hours*3600 else 'slower'} than real time)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="WispSim full-stack performance benchmark")
    ap.add_argument("--cache",       default="landfire_cache")
    ap.add_argument("--device",      default="mps", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--members",     type=int, default=20)
    ap.add_argument("--hours",       type=float, default=0.5,
                    help="Sim duration for full runner benchmark (default 0.5h = 3 cycles)")
    ap.add_argument("--skip-runner", action="store_true",
                    help="Skip the full SimulationRunner benchmark")
    ap.add_argument("--tmp",         default="/tmp/wisp_bench")
    args = ap.parse_args()

    tmp_dir = Path(args.tmp)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{HEADER}WispSim Full-Stack Benchmark{RESET}")
    print(f"  device={args.device}  N={args.members}  torch={torch.__version__}")

    terrain = load_from_directory(args.cache, resolution_m=100.0)
    R, C = terrain.shape
    print(f"  Terrain: {R}×{C} = {R*C:,} cells  ({R*100/1000:.1f}km × {C*100/1000:.1f}km)")

    # ── Build shared stack ────────────────────────────────────────────────
    print("\nBuilding stack...", end="", flush=True)
    orch, fire_engine, obs_store, gp = make_stack(terrain, args.device, args.members)

    ignition = (R // 2, C // 2)
    ground_truth = generate_ground_truth(
        terrain=terrain,
        ignition_cell=ignition,
        base_fmc=0.08, base_ws=5.0, base_wd=225.0,
        wind_events=[], seed=42,
    )
    # Prime ground truth fire to ~10 min so there is something burning
    ws, wd = ground_truth.wind_speed, ground_truth.wind_direction
    for _ in range(60):
        ground_truth.fire.step(10.0, ws, wd, ground_truth.fmc)
    print(" done")

    # ── Component benchmarks ──────────────────────────────────────────────
    print(f"\n{SECTION}══ Component Benchmarks  ({TRIALS} trials each, median) ══{RESET}\n")

    results: dict[str, float] = {}

    # 1. Fire CA
    print("  [1/7] fire_ca.step ...", end="", flush=True)
    t_ca, active_cells = bench_fire_ca(terrain, ground_truth)
    results["fire_ca.step (×360/h)"] = t_ca
    print(f" {t_ca*1000:.2f}ms  (active_cells≈{active_cells})")

    # 2. Wind field
    print("  [2/7] compute_wind_field ...", end="", flush=True)
    t_wind = bench_wind(terrain, ground_truth)
    results["compute_wind_field (×360/h)"] = t_wind
    print(f" {t_wind*1000:.2f}ms")

    # 3. GP predict
    print("  [3/7] gp.predict ...", end="", flush=True)
    t_gp = bench_gp_predict(gp, terrain.shape)
    results["gp.predict (×6+/cycle)"] = t_gp
    print(f" {t_gp*1000:.1f}ms")

    # 4. Renderer frame
    print("  [4/7] renderer.render_frame (savefig) ...", end="", flush=True)
    t_render = bench_renderer(terrain, ground_truth, orch, tmp_dir)
    results["render_frame/savefig (×60/h)"] = t_render
    print(f" {t_render:.3f}s")

    # 5. compute_information_field
    print("  [5/7] compute_information_field ...", end="", flush=True)
    t_info = bench_info_field(terrain, orch, fire_engine, args.members)
    results["compute_info_field (×60/h)"] = t_info
    print(f" {t_info*1000:.1f}ms")

    # 6. LiveEstimator (GP fork + 1 fire member)
    print("  [6/7] LiveEstimator.compute_estimate (GP fork + 1 member) ...", end="", flush=True)
    t_live = bench_live_estimator(terrain, orch, ground_truth)
    results["live_estimator (×60/h)"] = t_live
    print(f" {t_live:.3f}s")

    # 7. Network step
    print("  [7/7] mesh_network.step ...", end="", flush=True)
    t_net = bench_network(terrain)
    results["network.step (×360/h)"] = t_net
    print(f" {t_net*1000:.2f}ms")

    # ── Projected hourly cost ─────────────────────────────────────────────
    print(f"\n{SECTION}══ Projected Cost Per Simulated Hour ══{RESET}")
    print(f"  (Assumes 360 steps, 6 WISP cycles, 60 render frames per hour)\n")

    n_steps    = 360
    n_cycles   = 6
    n_frames   = 60
    n_ca       = n_steps        # every step
    n_wind     = n_steps        # every step
    n_render   = n_frames       # every frame_interval steps
    n_gp_pred  = n_cycles * 3   # ~3 predict calls per cycle (orchestrator + renderer + live)
    n_info     = n_frames       # every render frame
    n_live     = n_frames       # every render frame
    n_net      = n_steps        # every step (if enabled)

    projected = {
        "fire_ca.step":              t_ca    * n_ca,
        "compute_wind_field":        t_wind  * n_wind,
        "gp.predict":                t_gp    * n_gp_pred,
        "render_frame/savefig":      t_render * n_render,
        "compute_info_field":        t_info  * n_info,
        "live_estimator":            t_live  * n_live,
        "network.step (if enabled)": t_net   * n_net,
    }

    # WISP ensemble is dominant — add it separately
    # Use actual N=members time from benchmark_performance if available
    # Estimate: t_live is 1 member; scale to n_members for 6 cycles
    t_ensemble_per_cycle = t_live * args.members   # rough scaling
    projected["WISP ensemble (6 cycles)"] = t_ensemble_per_cycle * n_cycles

    total_proj = sum(projected.values())

    print(f"  {'Component':<34s} {'Per call':>9}  {'×/hr':>5}  {'Total/hr':>9}  Bar")
    print(f"  {'─'*75}")

    call_counts = {
        "fire_ca.step":              (t_ca,     n_ca,    t_ca    * n_ca),
        "compute_wind_field":        (t_wind,   n_wind,  t_wind  * n_wind),
        "gp.predict":                (t_gp,     n_gp_pred, t_gp  * n_gp_pred),
        "render_frame/savefig":      (t_render, n_render, t_render * n_render),
        "compute_info_field":        (t_info,   n_info,  t_info  * n_info),
        "live_estimator":            (t_live,   n_live,  t_live  * n_live),
        "network.step (if enabled)": (t_net,    n_net,   t_net   * n_net),
        "WISP ensemble (6 cycles)":  (t_ensemble_per_cycle, n_cycles, t_ensemble_per_cycle * n_cycles),
    }

    for name, (per_call, n_calls, total) in sorted(call_counts.items(), key=lambda x: -x[1][2]):
        pct = 100.0 * total / max(total_proj, 1e-9)
        bar = "█" * int(pct / 3)
        unit = "ms" if per_call < 1.0 else "s "
        per_str = f"{per_call*1000:.1f}ms" if per_call < 1.0 else f"{per_call:.3f}s"
        print(f"  {name:<34s} {per_str:>9}  {n_calls:>5}  {total:>7.1f}s  {pct:5.1f}%  {bar}")

    print(f"  {'─'*75}")
    print(f"  {'ESTIMATED TOTAL':<34s} {'':>9}  {'':>5}  {total_proj:>7.1f}s")
    print(f"\n  {WARN}Note: WISP ensemble estimate is rough (scales t_live_est×N).{RESET}")
    print(f"  {WARN}Run benchmark_performance.py for accurate fire engine numbers.{RESET}")

    # ── Inefficiency analysis ─────────────────────────────────────────────
    print(f"\n{SECTION}══ Inefficiency Analysis ══{RESET}\n")

    issues = []

    if t_live > 2.0:
        issues.append((
            "LiveEstimator runs 1 fire member every render frame (×60/h)",
            f"  Cost: {t_live:.1f}s × 60 = {t_live*60:.0f}s/h",
            "  Fix:  Only re-run fire member when new observations arrive\n"
            "        (cache result; `_has_obs` flag already exists in LiveEstimator)",
        ))

    if t_render > 0.5:
        issues.append((
            "Matplotlib savefig called every render frame (×60/h)",
            f"  Cost: {t_render:.2f}s × 60 = {t_render*60:.0f}s/h",
            "  Fix:  Reduce frame_interval (fewer frames) OR switch to\n"
            "        canvas.draw() + buffer instead of savefig each frame",
        ))

    if t_gp > 0.5:
        issues.append((
            "GP predict called multiple times per cycle",
            f"  Cost: {t_gp:.2f}s × {n_gp_pred} = {t_gp*n_gp_pred:.0f}s/h",
            "  Fix:  Cache gp.predict() result within a cycle; invalidate on new obs",
        ))

    if t_ca > 0.05:
        issues.append((
            "Ground-truth fire CA is pure-Python heap (×360/h)",
            f"  Cost: {t_ca*1000:.0f}ms × 360 = {t_ca*360:.0f}s/h — grows as fire spreads",
            "  Fix:  Vectorize heap expansion with numpy (significant refactor)",
        ))

    if not issues:
        print(f"  {OK}No major inefficiencies detected at N={args.members}.{RESET}")
    else:
        for title, cost, fix in issues:
            print(f"  {WARN}⚠ {title}{RESET}")
            print(f"{cost}")
            print(f"{fix}\n")

    # ── Full runner ───────────────────────────────────────────────────────
    if not args.skip_runner:
        print(f"\n{SECTION}══ Full SimulationRunner  ({args.hours:.2f}h, N={args.members}) ══{RESET}")
        print("  (Step-bucket breakdown will appear in the log below)\n")
        run_full_sim(terrain, args.device, args.members, args.hours, tmp_dir)

    print(f"\n{OK}Benchmark complete.{RESET}\n")


if __name__ == "__main__":
    main()
