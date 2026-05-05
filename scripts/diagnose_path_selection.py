"""
Diagnostic: is the greedy path planner selecting non-burnable / ocean domains,
and does the score function incentivise long worthless detours?

Bypasses InformationField / GP internals — builds the domain graph directly
and uses raw GP variance as the domain weight (worst-case: no fire yet).

Usage
-----
    python scripts/diagnose_path_selection.py
    python scripts/diagnose_path_selection.py --save out/path_diag.png
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from angrybird.landfire import load_from_directory
from angrybird.selectors.correlation_path import (
    _terrain_features,
    _felzenszwalb_label_map,
    _build_correlation_graph,
    _plan_greedy_path,
    _pos_to_domain,
    BUDGET_BUFFER,
    NB_BURNABLE_THRESHOLD,
    CorrelationGraph,
)
from angrybird.observations import ObservationStore
from angrybird.gp import IGNISGPPrior



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",       default="landfire_cache")
    parser.add_argument("--drones",      type=int,   default=3)
    parser.add_argument("--corr-len",    type=float, default=2000.0)
    parser.add_argument("--min-cells",   type=int,   default=15)
    parser.add_argument("--d-cycle-km",  type=float, default=13.95)
    parser.add_argument("--save",        default=None)
    args = parser.parse_args()

    res       = 100.0
    d_cycle_m = args.d_cycle_km * 1000.0
    budget    = d_cycle_m * BUDGET_BUFFER

    # ── Terrain ────────────────────────────────────────────────────────────
    print("Loading terrain …")
    terrain = load_from_directory(args.cache, resolution_m=res)
    R, C = terrain.shape
    print(f"  {R}×{C}  res={res:.0f} m")

    staging_row = int(R * 0.65)
    staging_col = int(C * 0.60)
    staging_m   = np.array([staging_row * res, staging_col * res])
    print(f"  Staging cell: ({staging_row}, {staging_col})")

    # ── GP variance (prior, no observations) ───────────────────────────────
    print("Building GP prior variance …")
    obs  = ObservationStore()
    gp   = IGNISGPPrior(obs, terrain=terrain, resolution_m=res)
    prior = gp.predict(terrain.shape)
    gp_var_map = prior.fmc_variance.astype(np.float64)   # (R, C)
    print(f"  GP var range: {gp_var_map.min():.2f} – {gp_var_map.max():.2f}")

    # ── Domain graph ───────────────────────────────────────────────────────
    print("Building Felzenszwalb domains …")
    features = _terrain_features(terrain)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        label_map = _felzenszwalb_label_map(
            terrain, features, args.corr_len, res, args.min_cells)
    n_dom = int(label_map.max()) + 1
    print(f"  {n_dom} domains")

    w_zeros = np.zeros((R, C), dtype=np.float32)
    graph = _build_correlation_graph(
        label_map=label_map,
        features=features,
        w_field=w_zeros,
        w_by_variable={"fmc": w_zeros},
        resolution_m=res,
        fuel_model=terrain.fuel_model,
    )

    # ── Per-domain stats ───────────────────────────────────────────────────
    # Note: burnable_fraction is now stored on the domain object (from fuel_model)
    burn_frac  = {d.domain_id: d.burnable_fraction for d in graph.domains}
    area_cells = {d.domain_id: d.area_cells for d in graph.domains}

    blocked_domains = {
        d.domain_id for d in graph.domains
        if d.burnable_fraction < NB_BURNABLE_THRESHOLD
    }
    print(f"  {len(blocked_domains)}/{len(graph.domains)} domains blocked "
          f"(burnable < {NB_BURNABLE_THRESHOLD:.0%})")

    # Use mean GP variance as domain weight (mirrors selector fallback)
    base_domain_w: dict[int, float] = {
        d.domain_id: float(np.mean(gp_var_map[
            [c[0] for c in d.cells],
            [c[1] for c in d.cells],
        ]))
        for d in graph.domains
    }
    current_w   = dict(base_domain_w)
    current_var = gp_var_map.copy()

    start_domain = _pos_to_domain(staging_m, graph, res)
    print(f"  Start domain: {start_domain}  "
          f"burnable={burn_frac[start_domain]:.0%}  "
          f"area={area_cells[start_domain]} cells")

    # ── Run greedy for each drone ──────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  GREEDY PATH ANALYSIS  budget={budget/1000:.1f} km, {args.drones} drones")
    print(f"{'='*90}")

    all_paths: list[list[int]] = []
    for drone_idx in range(args.drones):
        path, dist = _plan_greedy_path(
            graph=graph,
            start_domain=start_domain,
            budget_m=budget,
            current_w=current_w,
            current_var=current_var,
            gp=gp,
            gp_update_budget_m=budget,
            blocked_domains=blocked_domains,
        )
        all_paths.append(path)

        print(f"\nDrone {drone_idx}  ({len(path)-1} hops, {dist/1000:.1f} km total):")
        hdr = f"  {'Hop':>3}  {'DomID':>6}  {'Burn%':>6}  {'Area':>5}  " \
              f"{'w':>8}  {'EdgeInfo':>8}  {'Dist_m':>7}  " \
              f"{'Score':>10}  {'w/km':>8}  NOTE"
        print(hdr)
        print("  " + "-"*95)

        for hop_idx, (did_from, did_to) in enumerate(zip(path[:-1], path[1:])):
            edge = graph.edge(did_from, did_to)
            if edge is None:
                continue
            w_val  = base_domain_w.get(did_to, 0.0)
            e_info = edge.information_gain
            d_m    = edge.real_distance_m
            score  = (w_val + e_info) / (d_m + 1.0)
            w_km   = (w_val + e_info) / max(d_m / 1000.0, 0.001)
            bf     = burn_frac[did_to]
            area   = area_cells[did_to]

            notes = []
            if bf < 0.10:
                notes.append("🌊 OCEAN/NB")
            if area > 300:
                notes.append(f"⚠ BIG({area}cells)")
            if d_m > d_cycle_m * 0.3:
                notes.append("⚠ LONG HOP")

            print(f"  {hop_idx+1:>3}  {did_to:>6}  {bf:>5.0%}  {area:>5}  "
                  f"{w_val:>8.1f}  {e_info:>8.4f}  {d_m:>7.0f}  "
                  f"{score:>10.6f}  {w_km:>8.3f}  {'  '.join(notes)}")

    # ── Hop-0 candidate breakdown ──────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  ALL HOP-0 CANDIDATES from start domain (sorted by score)")
    print(f"{'='*90}")
    edges_data = []
    for edge in graph.adj.get(start_domain, []):
        nxt  = edge.target
        w    = base_domain_w.get(nxt, 0.0)
        d_m  = edge.real_distance_m
        sc   = (w + edge.information_gain) / (d_m + 1.0)
        wkm  = (w + edge.information_gain) / max(d_m / 1000.0, 0.001)
        bf   = burn_frac[nxt]
        edges_data.append(dict(domain=nxt, w=w, dist=d_m, score=sc,
                               w_km=wkm, burn_frac=bf, area=area_cells[nxt]))
    edges_data.sort(key=lambda x: -x["score"])

    print(f"  {'Rank':>4}  {'DomID':>6}  {'Burn%':>6}  {'Area':>5}  "
          f"{'w':>8}  {'Dist_m':>7}  {'Score':>10}  {'w/km':>8}")
    for rank, e in enumerate(edges_data, 1):
        flag = "  🌊 OCEAN/NB" if e["burn_frac"] < 0.10 else ""
        print(f"  {rank:>4}  {e['domain']:>6}  {e['burn_frac']:>5.0%}  "
              f"{e['area']:>5}  {e['w']:>8.1f}  {e['dist']:>7.0f}  "
              f"{e['score']:>10.6f}  {e['w_km']:>8.3f}{flag}")

    selected_first = all_paths[0][1] if len(all_paths[0]) > 1 else None
    if selected_first is not None:
        bf0 = burn_frac.get(selected_first, 1.0)
        print(f"\n  → Drone 0 selected domain {selected_first}  "
              f"(burnable={bf0:.0%})")
        if bf0 < 0.10:
            print("  *** CONFIRMED: greedy selected NON-BURNABLE domain at hop 0 ***")
        else:
            print("  ✓  First hop is on burnable land.")

    # check any hop in any path for ocean
    ocean_hops = [
        (di, hop, did)
        for di, path in enumerate(all_paths)
        for hop, did in enumerate(path[1:], 1)
        if burn_frac.get(did, 1.0) < 0.10
    ]
    if ocean_hops:
        print(f"\n  *** {len(ocean_hops)} ocean/NB domain(s) in selected paths: ***")
        for di, hop, did in ocean_hops:
            print(f"      Drone {di}, hop {hop}: domain {did}  "
                  f"burnable={burn_frac[did]:.0%}  area={area_cells[did]}")
    else:
        print("\n  ✓  No ocean/NB domains in any selected path.")

    # ── Figures ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle("Path Selection Diagnostic — GP-variance fallback (pre-fire)",
                 fontsize=11)

    dom_by_id = {d.domain_id: d for d in graph.domains}
    path_colors = ["#2196F3", "#FF5722", "#4CAF50"]

    # Panel 1 — burnable fraction map + paths
    ax = axes[0]
    burn_map = np.zeros((R, C), dtype=np.float32)
    for d in graph.domains:
        bf = burn_frac[d.domain_id]
        for r_, c_ in d.cells:
            burn_map[r_, c_] = bf
    im = ax.imshow(burn_map, cmap="RdYlGn", vmin=0, vmax=1, origin="upper")
    fig.colorbar(im, ax=ax, fraction=0.03, label="Burnable fraction")
    for di, (path, color) in enumerate(zip(all_paths, path_colors)):
        pts = [dom_by_id[did].representative_cell for did in path]
        ax.plot([p[1] for p in pts], [p[0] for p in pts],
                "-o", color=color, lw=2, ms=4, label=f"Drone {di}", zorder=5)
    ax.scatter([staging_col], [staging_row], marker="D", c="yellow",
               s=120, zorder=10, edgecolors="black", lw=0.8, label="Staging")
    ax.set_title("Burnable fraction + selected paths", fontsize=9)
    ax.set_xlabel("East (cells)"); ax.set_ylabel("North (cells)")
    ax.legend(fontsize=7)

    # Panel 2 — Score vs distance for all hop-0 candidates
    ax = axes[1]
    bf_arr    = np.array([e["burn_frac"] for e in edges_data])
    score_arr = np.array([e["score"]     for e in edges_data])
    dist_arr  = np.array([e["dist"]      for e in edges_data]) / 1000.0
    sc_colors = plt.cm.RdYlGn(bf_arr)
    ax.scatter(dist_arr, score_arr, c=sc_colors, s=60,
               edgecolors="k", linewidths=0.4, zorder=3)
    if selected_first:
        sel = next((e for e in edges_data if e["domain"] == selected_first), None)
        if sel:
            ax.scatter([sel["dist"]/1000], [sel["score"]], marker="*",
                       c="gold", s=300, zorder=10, edgecolors="black",
                       lw=0.8, label="Drone 0 selected")
    ax.set_xlabel("Centroid distance (km)")
    ax.set_ylabel("Score = (w + edge_info) / (d + 1)")
    ax.set_title("Hop-0 candidates: score vs centroid distance\n"
                 "green=burnable, red=ocean/NB", fontsize=9)
    ax.legend(fontsize=7)

    # Panel 3 — area vs burnable fraction, highlight path domains
    ax = axes[2]
    path_set = set(did for p in all_paths for did in p)
    all_ids  = [d.domain_id for d in graph.domains]
    x_bf     = np.array([burn_frac[did]    for did in all_ids])
    x_area   = np.array([area_cells[did]   for did in all_ids])
    in_path  = np.array([did in path_set   for did in all_ids])
    ax.scatter(x_area[~in_path], x_bf[~in_path],
               c="#BDBDBD", s=12, alpha=0.4, label="Not selected")
    ax.scatter(x_area[in_path],  x_bf[in_path],
               c="#F44336", s=60, edgecolors="black", lw=0.5,
               zorder=5, label="In selected path")
    ax.axhline(0.10, color="red", ls="--", lw=1, label="10% burn threshold")
    ax.set_xlabel("Domain area (cells)")
    ax.set_ylabel("Burnable fraction")
    ax.set_title("Domain size vs burnable fraction\n(red = in selected path)",
                 fontsize=9)
    ax.legend(fontsize=7)

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
