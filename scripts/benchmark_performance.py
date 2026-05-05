"""
benchmark_performance.py — End-to-end performance profiler for the WISP stack.

Measures wall-clock time at every layer:

  1. GPU fire engine (N = 20, 50, 75)
     • Per-phase breakdown: setup, perturbation generation, SDF init,
       Rothermel precompute, CFL loop, redistancing, CPU transfer
     • s/member scaling and estimated VRAM footprint

  2. Full WISP cycle (one orchestrator.run_cycle call)
     • Phase log already emitted by IGNISOrchestrator (dynamic_prior,
       gp_fit_pre, assimilation, gp_fit_post, fire_state, ensemble,
       info_field, selection, path_plan)

  3. SimulationRunner step loop
     • Per-bucket averages: wind, fire_ca, drones, network, wisp_cycle,
       live_est, render

Usage
-----
    PYTHONPATH=. python scripts/benchmark_performance.py
    PYTHONPATH=. python scripts/benchmark_performance.py --device mps --members 20 50 75
    PYTHONPATH=. python scripts/benchmark_performance.py --skip-cycle  # engine only
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from angrybird.fire_engines.gpu_fire_engine import GPUFireEngine, _redistance, _phi_gradient_full
from angrybird.gp import IGNISGPPrior
from angrybird.landfire import load_from_directory
from angrybird.observations import ObservationStore, ObservationType
from angrybird.orchestrator import IGNISOrchestrator
from angrybird.selectors.base import SelectorRegistry
from angrybird.selectors.greedy import GreedySelector
from angrybird.selectors.baselines import UniformSelector, FireFrontSelector
from angrybird.config import TAU_FMC_S, TAU_WIND_SPEED_S, TAU_WIND_DIR_S
from angrybird.types import GPPrior, TerrainData

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

HEADER  = "\033[1;36m"
SECTION = "\033[1;33m"
OK      = "\033[0;32m"
RESET   = "\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def timed(label: str, store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0


def mps_sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def free_mps():
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def fmt_row(label: str, val: float, total: float, unit: str = "s") -> str:
    pct = 100.0 * val / total if total > 0 else 0.0
    bar = "█" * int(pct / 3)
    return f"  {label:<28s} {val:7.2f}{unit}  {pct:5.1f}%  {bar}"


def flat_gp_prior(R: int, C: int) -> GPPrior:
    return GPPrior(
        fmc_mean         = np.full((R, C), 0.08,  dtype=np.float32),
        fmc_variance     = np.full((R, C), 0.001, dtype=np.float32),
        wind_speed_mean  = np.full((R, C), 5.0,   dtype=np.float32),
        wind_speed_variance = np.full((R, C), 0.25, dtype=np.float32),
        wind_dir_mean    = np.full((R, C), 225.0, dtype=np.float32),
        wind_dir_variance= np.full((R, C), 100.0, dtype=np.float32),
    )


def make_fire_state(R: int, C: int) -> np.ndarray:
    fs = np.zeros((R, C), dtype=np.float32)
    fs[R // 2, C // 2] = 1.0
    return fs


# ---------------------------------------------------------------------------
# Phase-instrumented fire engine wrapper
# ---------------------------------------------------------------------------

def run_engine_instrumented(
    engine: GPUFireEngine,
    terrain: TerrainData,
    gp_prior: GPPrior,
    fire_state: np.ndarray,
    n_members: int,
    horizon_min: int = 240,
) -> dict[str, float]:
    """Run one ensemble and return wall-clock seconds per phase."""
    from scipy.ndimage import distance_transform_edt
    from angrybird.config import (
        FMC_MIN_FRACTION, FMC_MAX_FRACTION,
        WIND_SPEED_MIN_MS, WIND_SPEED_MAX_MS,
    )
    from angrybird.fire_engines.gpu_fire_engine import (
        _REDISTANCE_EVERY, _rothermel_ros, _crown_fire_check,
        _directional_ros, _redistance, _phi_gradient_full,
        _COL_W0, _COL_ST,
    )
    import torch

    T: dict[str, float] = {}
    dev = engine.device
    rng = np.random.default_rng(42)

    rows, cols = engine.shape
    total_s   = float(horizon_min) * 60.0
    sentinel_s = total_s * 1.1

    # ── 1. Perturbation generation ────────────────────────────────────────
    t0 = time.perf_counter()
    fmc_std = np.sqrt(np.clip(gp_prior.fmc_variance, 0, None)).astype(np.float32)
    ws_std  = np.sqrt(np.clip(gp_prior.wind_speed_variance, 0, None)).astype(np.float32)
    wd_std  = np.sqrt(np.clip(gp_prior.wind_dir_variance, 0, None)).astype(np.float32)

    torch.manual_seed(int(rng.integers(0, 2**31)))
    fmc_mean_t = torch.tensor(gp_prior.fmc_mean,            device=dev)
    fmc_std_t  = torch.tensor(fmc_std,                      device=dev)
    ws_mean_t  = torch.tensor(gp_prior.wind_speed_mean,     device=dev)
    ws_std_t   = torch.tensor(ws_std,                       device=dev)
    wd_mean_t  = torch.tensor(gp_prior.wind_dir_mean,       device=dev)
    wd_std_t   = torch.tensor(wd_std,                       device=dev)

    fmc = (fmc_mean_t + torch.randn(n_members, rows, cols, dtype=torch.float32, device=dev) * fmc_std_t).clamp(FMC_MIN_FRACTION, FMC_MAX_FRACTION)
    ws  = (ws_mean_t  + torch.randn(n_members, rows, cols, dtype=torch.float32, device=dev) * ws_std_t).clamp(WIND_SPEED_MIN_MS, WIND_SPEED_MAX_MS)
    wd  = (wd_mean_t  + torch.randn(n_members, rows, cols, dtype=torch.float32, device=dev) * wd_std_t) % 360.0
    mps_sync()
    T["perturbation_gen"] = time.perf_counter() - t0

    # ── 2. SDF init ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    burned_np = fire_state > 0.5
    d_in  = distance_transform_edt(burned_np).astype(np.float32)
    d_out = distance_transform_edt(~burned_np).astype(np.float32)
    sdf   = np.where(burned_np, -d_in, d_out) * engine.dx
    phi   = (torch.tensor(sdf, dtype=torch.float32, device=dev)
             .unsqueeze(0).expand(n_members, -1, -1).clone())
    arrival_t = torch.full((n_members, rows, cols), sentinel_s, dtype=torch.float32, device=dev)
    burned_t = torch.tensor(burned_np, device=dev)
    arrival_t[:, burned_t] = 0.0
    mps_sync()
    T["sdf_init"] = time.perf_counter() - t0

    # ── 3. Static precomputation ──────────────────────────────────────────
    t0 = time.perf_counter()
    fp  = engine._fuel_params.unsqueeze(0)
    tss = engine._tan_slope_sq.unsqueeze(0)
    asp = engine._aspect.unsqueeze(0)
    cbh = engine._cbh.unsqueeze(0)
    cbd = engine._cbd.unsqueeze(0)
    waf = engine._wind_adj.unsqueeze(0)

    ros_s, _, theta_max = _rothermel_ros(fmc, ws, waf, wd, asp, fp, tss)
    ws_mid_kmh = ws * waf * 3.6
    LB = (1.0 + 0.25 * ws_mid_kmh).clamp(min=1.0, max=8.0)
    e  = (LB - 1.0) / (LB + 1.0)
    ros_crown  = 11.02 * (ws * 3.6).pow(0.854) * cbd.pow(0.19) / 60.0
    I_crit     = (0.01 * cbh * (460.0 + 2590.0)).pow(1.5)
    int_factor = 18000.0 * fp[..., _COL_W0] * (1.0 - fp[..., _COL_ST])
    crown_v    = ros_s.new_full(ros_s.shape, 2.0)
    surface_v  = ros_s.new_ones(ros_s.shape)
    max_ros    = max(ros_s.max().item(), ros_crown.max().item())
    dt         = float(max(0.5, min(engine.target_cfl * engine.dx / max(max_ros, 1e-10), 300.0)))
    mps_sync()
    T["rothermel_precompute"] = time.perf_counter() - t0

    print(f"    dt={dt:.2f}s  steps={int(total_s/dt)}  max_ros={max_ros:.4f}m/s", flush=True)

    # ── 4. CFL loop ───────────────────────────────────────────────────────
    fire_type = surface_v.clone()
    t_sim = 0.0
    step  = 0
    n_redist = 0
    t_grad = t_ros = t_phi = t_redist = 0.0

    loop_t0 = time.perf_counter()
    while t_sim < total_s:
        dt_step = min(dt, total_s - t_sim)
        if dt_step <= 0.0:
            break

        _t = time.perf_counter()
        grad_mag, gx_cd, gy_cd = _phi_gradient_full(phi, engine.dx)
        mps_sync()
        t_grad += time.perf_counter() - _t

        _t = time.perf_counter()
        ros_dir   = _directional_ros(ros_s, e, theta_max, gx_cd, gy_cd)
        intensity = int_factor * ros_dir
        initiates = intensity > I_crit
        ros_f     = torch.where(initiates, torch.maximum(ros_dir, ros_crown), ros_dir)
        fire_type = torch.where(initiates, crown_v, surface_v)
        mps_sync()
        t_ros += time.perf_counter() - _t

        _t = time.perf_counter()
        phi_new   = phi - dt_step * ros_f * grad_mag
        crossing  = (phi > 0) & (phi_new <= 0)
        t_cross   = t_sim + dt_step * phi / (phi - phi_new + 1e-10)
        arrival_t = torch.where(crossing, t_cross, arrival_t)
        phi = phi_new
        mps_sync()
        t_phi += time.perf_counter() - _t

        step += 1
        t_sim += dt_step

        if step % _REDISTANCE_EVERY == 0:
            _t = time.perf_counter()
            phi = _redistance(phi, engine.dx)
            mps_sync()
            t_redist += time.perf_counter() - _t
            n_redist += 1

    T["cfl_loop_total"] = time.perf_counter() - loop_t0
    T["cfl_gradient"]   = t_grad
    T["cfl_ros_dir"]    = t_ros
    T["cfl_phi_advance"]= t_phi
    T["cfl_redistance"] = t_redist
    T["cfl_steps"]      = float(step)
    T["cfl_redist_calls"] = float(n_redist)

    # ── 5. CPU transfer ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    _ = arrival_t.cpu().numpy()
    _ = fire_type.cpu().numpy()
    _ = fmc.cpu().numpy()
    _ = ws.cpu().numpy()
    _ = wd.cpu().numpy()
    T["cpu_transfer"] = time.perf_counter() - t0

    return T


# ---------------------------------------------------------------------------
# Fire engine benchmark
# ---------------------------------------------------------------------------

def benchmark_engine(terrain: TerrainData, device: str, members: list[int]) -> None:
    print(f"\n{SECTION}══ GPU Fire Engine Benchmark  (device={device}, horizon=240 min) ══{RESET}")
    R, C = terrain.shape
    print(f"   Grid: {R}×{C} = {R*C:,} cells  |  {R*terrain.resolution_m/1000:.1f}km × {C*terrain.resolution_m/1000:.1f}km")

    gp_prior   = flat_gp_prior(R, C)
    fire_state = make_fire_state(R, C)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = GPUFireEngine(terrain, device=device, target_cfl=0.7)

    results: list[tuple[int, dict]] = []

    for N in members:
        free_mps()
        print(f"\n{HEADER}─── N={N} members ───{RESET}")
        # Warmup
        print("  warmup...", end="", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine.run(terrain, gp_prior, fire_state, N, 240, np.random.default_rng(0))
        free_mps()
        print(" done")

        # Instrumented run
        T = run_engine_instrumented(engine, terrain, gp_prior, fire_state, N)
        results.append((N, T))

        total = T["perturbation_gen"] + T["sdf_init"] + T["rothermel_precompute"] + T["cfl_loop_total"] + T["cpu_transfer"]
        tensor_gb = N * R * C * 4 / 1e9

        print(f"\n  {'Phase':<28s} {'Time':>7s}  {'%':>5s}  Bar")
        print(f"  {'─'*65}")
        print(fmt_row("perturbation_gen (GPU randn)", T["perturbation_gen"], total))
        print(fmt_row("sdf_init (scipy EDT)",         T["sdf_init"],         total))
        print(fmt_row("rothermel_precompute",         T["rothermel_precompute"], total))
        print(fmt_row("cfl_loop_total",               T["cfl_loop_total"],   total))
        print(fmt_row("  ↳ gradient (F.pad)",         T["cfl_gradient"],     total))
        print(fmt_row("  ↳ ROS + crown",              T["cfl_ros_dir"],      total))
        print(fmt_row("  ↳ phi advance + crossing",   T["cfl_phi_advance"],  total))
        print(fmt_row("  ↳ redistancing",             T["cfl_redistance"],   total))
        print(fmt_row("cpu_transfer",                 T["cpu_transfer"],     total))
        print(f"  {'─'*65}")
        print(f"  {'TOTAL':<28s} {total:7.2f}s")
        print(f"\n  steps={int(T['cfl_steps'])}  redist_calls={int(T['cfl_redist_calls'])}")
        print(f"  s/member={total/N:.3f}  tensor_footprint={tensor_gb:.3f}GB")

    # Scaling table
    if len(results) > 1:
        print(f"\n{SECTION}── Scaling Summary ──{RESET}")
        print(f"  {'N':>6}  {'total':>8}  {'s/member':>9}  {'speedup_vs_N20':>15}  {'tensor_GB':>10}")
        base_spm = None
        for N, T in results:
            total = T["perturbation_gen"] + T["sdf_init"] + T["rothermel_precompute"] + T["cfl_loop_total"] + T["cpu_transfer"]
            spm = total / N
            if base_spm is None:
                base_spm = spm
            speedup = base_spm / spm
            tensor_gb = N * R * C * 4 / 1e9
            print(f"  {N:>6}  {total:>7.1f}s  {spm:>9.3f}  {speedup:>15.2f}×  {tensor_gb:>10.3f}")


# ---------------------------------------------------------------------------
# Full WISP cycle benchmark (one orchestrator cycle on real terrain)
# ---------------------------------------------------------------------------

def benchmark_wisp_cycle(terrain: TerrainData, device: str, n_members: int) -> None:
    from wispsim.ground_truth import generate_ground_truth, WindEvent

    print(f"\n{SECTION}══ Full WISP Cycle Benchmark  (N={n_members}, device={device}) ══{RESET}")
    print("   (orchestrator phase timings logged at INFO level below)\n")

    R, C = terrain.shape
    res  = terrain.resolution_m

    # Ground truth
    ignition = (R // 2, C // 2)
    ground_truth = generate_ground_truth(
        terrain=terrain,
        ignition_cell=ignition,
        base_fmc=0.08,
        base_ws=5.0,
        base_wd=225.0,
        wind_events=[],
        seed=42,
    )

    # GP + obs store
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

    # Fire engine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fire_engine = GPUFireEngine(terrain, device=device, target_cfl=0.7)

    base_cell = (R - 1, C // 2)
    local_registry = SelectorRegistry()
    local_registry.register(GreedySelector(resolution_m=res))
    local_registry.register(UniformSelector())
    local_registry.register(FireFrontSelector())

    orchestrator = IGNISOrchestrator(
        terrain=terrain,
        gp=gp,
        obs_store=obs_store,
        fire_engine=fire_engine,
        selector_name="greedy",
        selector_registry=local_registry,
        n_drones=2,
        n_targets=6,
        staging_area=base_cell,
        n_members=n_members,
        horizon_min=240,
        resolution_m=res,
    )

    # Enable INFO so orchestrator's own timing log prints
    logging.getLogger("angrybird.orchestrator").setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s", force=True)

    # Advance ground truth a bit so there's a fire to work with
    gt_state = ground_truth.get_state(600.0)   # 10 min in

    print("Running one WISP cycle (watch for 'Cycle timing' log line)...", flush=True)
    t0 = time.perf_counter()
    report = orchestrator.run_cycle(
        t=600.0,
        ground_truth_state=gt_state,
        drone_positions=[(base_cell[0] - 10, base_cell[1])],
    )
    total = time.perf_counter() - t0
    print(f"\n  {OK}WISP cycle wall-clock: {total:.2f}s{RESET}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="WISP performance benchmark")
    ap.add_argument("--cache",      default="landfire_cache")
    ap.add_argument("--device",     default="mps", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--members",    type=int, nargs="+", default=[20, 50, 75])
    ap.add_argument("--skip-cycle", action="store_true", help="Skip full WISP cycle benchmark")
    ap.add_argument("--cycle-n",    type=int, default=20, help="N for WISP cycle benchmark")
    args = ap.parse_args()

    print(f"\n{HEADER}WISP Performance Benchmark{RESET}")
    print(f"  device={args.device}  torch={torch.__version__}")
    if torch.backends.mps.is_available():
        print(f"  MPS available ✓")
    if torch.cuda.is_available():
        print(f"  CUDA {torch.version.cuda}  |  {torch.cuda.get_device_name(0)}")

    terrain = load_from_directory(args.cache, resolution_m=100.0)
    print(f"  Terrain: {terrain.shape[0]}×{terrain.shape[1]}  res={terrain.resolution_m}m")

    benchmark_engine(terrain, args.device, args.members)

    if not args.skip_cycle:
        benchmark_wisp_cycle(terrain, args.device, args.cycle_n)

    print(f"\n{OK}Benchmark complete.{RESET}\n")


if __name__ == "__main__":
    main()
