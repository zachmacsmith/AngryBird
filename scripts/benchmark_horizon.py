"""
benchmark_horizon.py — Ensemble simulation horizon sweep.

Measures how changing the fire-spread horizon (currently 240 min in run.py,
60 min library default) affects:

  1. Fire engine wall-clock time (GPU CA → scales linearly with steps)
  2. Information-field quality: fraction of cells with w > threshold
  3. Sensitivity spatial extent: how far from the ignition non-zero
     sensitivity reaches (i.e. how much of the map the path planner "sees")

Two ignition scenarios are tested:
  • SMALL  — single-cell ignition (pre-fire / first-cycle cold start)
  • GROWN  — fire has spread for 60 min (representative of active fire)

Usage
-----
    python scripts/benchmark_horizon.py
    python scripts/benchmark_horizon.py --members 10 --device mps
    python scripts/benchmark_horizon.py --save out/horizon_bench.png
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from angrybird.config import TAU_FMC_S, TAU_WIND_SPEED_S, TAU_WIND_DIR_S
from angrybird.gp import IGNISGPPrior
from angrybird.information import compute_information_field
from angrybird.landfire import load_from_directory
from angrybird.observations import ObservationStore, ObservationType
from wispsim.ground_truth import generate_ground_truth

HORIZONS_MIN = [60, 120, 240]      # representative subset for N-sweep
W_THRESHOLD  = 1e-4               # "meaningful" w floor


def build_gp(terrain):
    decay_config = {
        ObservationType.FMC:            TAU_FMC_S,
        ObservationType.WIND_SPEED:     TAU_WIND_SPEED_S,
        ObservationType.WIND_DIRECTION: TAU_WIND_DIR_S,
    }
    try:
        obs = ObservationStore(decay_config)
    except TypeError:
        obs = ObservationStore()
    return IGNISGPPrior(obs, terrain=terrain, resolution_m=terrain.resolution_m)


def build_fire_engine(terrain, device):
    """Try GPU engine, fall back to SimpleFire."""
    try:
        from angrybird.fire_engines.gpu_fire_engine import GPUFireEngine
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return GPUFireEngine(terrain, device=device, target_cfl=0.7), "GPU"
    except Exception:
        from wispsim.simple_fire import SimpleFire
        return SimpleFire(), "Simple"


def make_fire_state(terrain, grown: bool, base_fmc, base_ws, base_wd, seed) -> np.ndarray:
    """
    SMALL (grown=False): single ignition cell, fire_state zeros everywhere except centre.
    GROWN (grown=True):  run ground-truth fire for 60 min to get a realistic fire front.
    """
    R, C = terrain.shape
    ig = (R // 2, C // 2)
    if not grown:
        fs = np.zeros((R, C), dtype=np.float32)
        fs[ig] = 1.0
        return fs

    gt = generate_ground_truth(
        terrain=terrain,
        ignition_cell=ig,
        base_fmc=base_fmc,
        base_ws=base_ws,
        base_wd=base_wd,
        wind_events=[],
        seed=seed,
    )
    ws, wd, fmc = gt.wind_speed, gt.wind_direction, gt.fmc
    for _ in range(360):           # 360 × 10 s = 60 min
        gt.fire.step(10.0, ws, wd, fmc)
    return (gt.fire.fire_state > 0).astype(np.float32)


def sweep_horizon(terrain, gp_prior, fire_engine, fire_state, n_members, rng_seed, label,
                  bimodal_alpha: float = 0.5, bimodal_beta: float = 0.3):
    """Run ensemble + info-field for each horizon and collect metrics."""
    recs = []
    R, C = terrain.shape
    n_cells = R * C

    for h in HORIZONS_MIN:
        rng = np.random.default_rng(rng_seed)

        # --- Fire engine ---
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ensemble = fire_engine.run(terrain, gp_prior, fire_state, n_members, h, rng)
        t_fire = time.perf_counter() - t0

        # --- Information field ---
        t1 = time.perf_counter()
        info = compute_information_field(
            ensemble, gp_prior,
            resolution_m=terrain.resolution_m,
            horizon_minutes=h,
            bimodal_alpha=bimodal_alpha,
            bimodal_beta=bimodal_beta,
        )
        t_info = time.perf_counter() - t1

        # --- Quality metrics ---
        w = info.w
        frac_nonzero  = float((w > W_THRESHOLD).sum()) / n_cells
        burn_frac     = float((ensemble.burn_probability > 0.01).sum()) / n_cells
        w_max         = float(w.max())
        w_mean_nonzero = float(w[w > W_THRESHOLD].mean()) if (w > W_THRESHOLD).any() else 0.0

        # Spatial reach: distance from ignition to furthest non-zero w cell
        nonzero_mask = w > W_THRESHOLD
        if nonzero_mask.any():
            ig_r, ig_c = R // 2, C // 2
            rs, cs = np.where(nonzero_mask)
            dists = np.sqrt((rs - ig_r)**2 + (cs - ig_c)**2) * terrain.resolution_m
            reach_km = float(dists.max()) / 1000.0
        else:
            reach_km = 0.0

        recs.append(dict(
            horizon=h,
            t_fire_s=t_fire,
            t_info_s=t_info,
            t_total_s=t_fire + t_info,
            frac_nonzero_w=frac_nonzero,
            burn_frac=burn_frac,
            w_max=w_max,
            w_mean_nz=w_mean_nonzero,
            reach_km=reach_km,
        ))

        print(f"  [{label}] H={h:3d}min  fire={t_fire:.2f}s  info={t_info:.3f}s  "
              f"w>0={frac_nonzero:.1%}  burn={burn_frac:.1%}  reach={reach_km:.1f}km")

    return recs


def main():
    ap = argparse.ArgumentParser(description="Ensemble horizon sweep benchmark")
    ap.add_argument("--cache",   default="landfire_cache")
    ap.add_argument("--device",  default="mps", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--members", type=int, default=50,
                    help="Primary N to benchmark (also sweeps N=10 for comparison)")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--save",    default=None)
    args = ap.parse_args()

    print(f"\nLoading terrain …")
    terrain = load_from_directory(args.cache, resolution_m=100.0)
    R, C = terrain.shape
    print(f"  {R}×{C}  ({R*100/1000:.1f}km × {C*100/1000:.1f}km)")

    print(f"Building GP prior …")
    gp = build_gp(terrain)
    gp_prior = gp.predict(terrain.shape)

    print(f"Building fire engine …")
    fire_engine, engine_label = build_fire_engine(terrain, args.device)
    print(f"  Engine: {engine_label}")

    fmc, ws, wd = 0.08, 5.0, 225.0

    fs_small = make_fire_state(terrain, grown=False, base_fmc=fmc, base_ws=ws, base_wd=wd, seed=args.seed)
    fs_grown = make_fire_state(terrain, grown=True,  base_fmc=fmc, base_ws=ws, base_wd=wd, seed=args.seed)
    print(f"  Grown fire: {int(fs_grown.sum())} cells burning")

    # Run at primary N, bimodal ON vs OFF, both fire scenarios
    n = args.members
    configs = [
        ("alpha=0.5 beta=0.3 [OLD]", dict(bimodal_alpha=0.5, bimodal_beta=0.3)),
        ("alpha=0.5 beta=0.0 [FIX]", dict(bimodal_alpha=0.5, bimodal_beta=0.0)),
        ("alpha=0.0 beta=0.0 [OFF]", dict(bimodal_alpha=0.0, bimodal_beta=0.0)),
    ]
    all_results: dict[tuple, list] = {}

    for cfg_label, cfg_kwargs in configs:
        print(f"\n──── N={n}  {cfg_label} ─────────────────────────────────────────────────")
        for scenario, fs in [("SMALL", fs_small), ("GROWN", fs_grown)]:
            print(f"  [{scenario}]")
            recs = sweep_horizon(terrain, gp_prior, fire_engine, fs, n, args.seed,
                                 f"{scenario}/{cfg_label}", **cfg_kwargs)
            all_results[(cfg_label, scenario)] = recs

    # ── Text summary ──────────────────────────────────────────────────────
    print(f"\n{'='*105}")
    print(f"  BIMODAL ON vs OFF  ({engine_label} engine, N={n}, WITH normalization Fix 1)")
    print(f"{'='*105}")
    hdr = (f"  {'Config':<22}  {'H(min)':>6}  {'Fire(s)':>8}  "
           f"{'w>0':>7}  {'Burn':>6}  {'Reach(km)':>10}  Scenario")
    print(hdr)
    print("  " + "-"*90)

    for cfg_label, _ in configs:
        for scenario in ["SMALL", "GROWN"]:
            recs = all_results[(cfg_label, scenario)]
            for r in recs:
                h = r["horizon"]
                tag = " ←" if h == 240 else ""
                print(f"  {cfg_label:<22}  {h:>6}  {r['t_fire_s']:>8.2f}s  "
                      f"{r['frac_nonzero_w']:>6.1%}  {r['burn_frac']:>5.1%}  "
                      f"{r['reach_km']:>9.1f}km  {scenario}{tag}")
        print("  " + "-"*90)

    # ── Key findings ──────────────────────────────────────────────────────
    print(f"\n  KEY FINDINGS (H=240 min):")
    cfg_labels = [c[0] for c in configs]
    for scenario in ["SMALL", "GROWN"]:
        print(f"  {scenario}:")
        for cfg_label in cfg_labels:
            r240 = next(r for r in all_results[(cfg_label, scenario)] if r["horizon"] == 240)
            print(f"    {cfg_label}: w>0={r240['frac_nonzero_w']:.2%}  reach={r240['reach_km']:.1f}km  "
                  f"w_max={r240['w_max']:.4f}")
        print()

    # ── Figures ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    fig.suptitle(
        f"Bimodal ON vs OFF  |  {engine_label} engine, N={n}  |  normalization ON (Fix 1)",
        fontsize=11,
    )
    palette = {
        ("alpha=0.5 beta=0.3 [OLD]", "SMALL"): ("#F44336", "o-"),
        ("alpha=0.5 beta=0.3 [OLD]", "GROWN"): ("#B71C1C", "o--"),
        ("alpha=0.5 beta=0.0 [FIX]", "SMALL"): ("#2196F3", "s-"),
        ("alpha=0.5 beta=0.0 [FIX]", "GROWN"): ("#1565C0", "s--"),
        ("alpha=0.0 beta=0.0 [OFF]", "SMALL"): ("#4CAF50", "^-"),
        ("alpha=0.0 beta=0.0 [OFF]", "GROWN"): ("#2E7D32", "^--"),
    }

    metrics = [
        ("t_fire_s",       "Wall time (s)",        "Fire engine time",        False),
        ("frac_nonzero_w", "% cells w > 1e-4",     "w-field coverage",        True),
        ("w_max",          "Peak w (normalized)",   "Peak information value",  False),
        ("reach_km",       "Reach (km)",            "Spatial reach",           False),
        ("burn_frac",      "Burn fraction",         "Ensemble burn coverage",  True),
        ("w_mean_nz",      "Mean w (non-zero)",     "Mean info (non-zero cells)", False),
    ]

    for idx, (key, ylabel, title, pct) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        scale = 100.0 if pct else 1.0
        for cfg_label, _ in configs:
            for scenario in ["SMALL", "GROWN"]:
                recs = all_results[(cfg_label, scenario)]
                color, ls = palette[(cfg_label, scenario)]
                vals = [r[key] * scale for r in recs]
                ax.plot(HORIZONS_MIN, vals, ls, color=color, lw=2,
                        label=f"{scenario} / {cfg_label.split('(')[1].rstrip(')')}")
        ax.axvline(240, ls=":", color="#4CAF50", lw=1.2, label="H=240 (default)")
        ax.set_xlabel("Horizon (min)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6)
        ax.set_xticks(HORIZONS_MIN)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
