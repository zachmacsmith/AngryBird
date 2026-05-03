"""
Simulation & evaluation visualizations (§2 of Vizualisations.md).

These plots exist only for the hackathon to validate the system and
demonstrate comparison results — not part of a deployed system.

Public API
----------
plot_ensemble_spread        2.1  Grid of N member fire perimeters
plot_arrival_distributions  2.2  Histograms at key cells
plot_strategy_comparison    2.3  Four-way strategy comparison
plot_entropy_convergence    2.4  Entropy convergence curve
plot_drone_value_curve      2.5  Marginal information gain vs drone count
plot_qubo_greedy_overlap    2.6  Jaccard similarity + divergent maps
plot_placement_stability    2.7  Jaccard between consecutive cycles
plot_ground_truth_reveal    2.8  Ground truth vs GP estimate vs residual
plot_innovation_tracking    2.9  |obs - pred| over cycles
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ._style import (
    VIZ_CONFIG,
    STRATEGY_STYLES,
    DRONE_COLORS,
    compute_hillshade,
    _imshow,
    _mark_stations,
    _mark_drone_targets,
    _add_legend,
    _domain_footnote,
    save_or_show,
)


# ---------------------------------------------------------------------------
# 2.1  Ensemble Spread
# ---------------------------------------------------------------------------

def plot_ensemble_spread(
    ensemble,          # EnsembleResult
    terrain,           # TerrainGrid
    max_members: int = 9,
    title: str = "Ensemble Spread",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Grid of individual ensemble-member fire perimeters overlaid on terrain.

    Shows how different perturbations produce different fire trajectories,
    illustrating why ensemble methods are necessary.
    """
    members = min(max_members, ensemble.n_members)
    cols = min(3, members)
    rows = (members + cols - 1) // cols
    fw = figsize[0] if figsize else cols * 3.5
    fh = figsize[1] if figsize else rows * 3.0

    fig, axes = plt.subplots(rows, cols, figsize=(fw, fh),
                              squeeze=False)
    fig.suptitle(title, fontsize=VIZ_CONFIG["font_size"] + 2, fontweight="bold")

    hs = compute_hillshade(terrain.elevation, terrain.resolution_m)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        if idx >= members:
            ax.set_visible(False)
            continue

        # Terrain base
        ax.imshow(hs, origin="upper", cmap="gray", vmin=0, vmax=1, alpha=0.6)

        # Per-member burn probability (treat each member as binary burn mask)
        if hasattr(ensemble, "burn_probability") and ensemble.burn_probability is not None:
            # If we have per-member data embedded, reconstruct — otherwise use aggregate
            bp = ensemble.burn_probability
        else:
            bp = np.zeros(terrain.elevation.shape, dtype=np.float32)

        # Per-member arrival times (if available) → perimeter at current horizon
        if (hasattr(ensemble, "arrival_times") and
                ensemble.arrival_times is not None and
                ensemble.arrival_times.ndim == 3):
            at_m = ensemble.arrival_times[idx]  # [rows, cols]
            burned = np.isfinite(at_m) & (at_m >= 0)
            ax.imshow(burned.astype(float), origin="upper",
                      cmap="Reds", alpha=0.6, vmin=0, vmax=1)
            subtitle = f"Member {idx + 1}"
        else:
            # Fall back: show ensemble burn probability with member index label
            ax.imshow(bp, origin="upper",
                      cmap=VIZ_CONFIG["burn_prob_cmap"], alpha=0.6, vmin=0, vmax=1)
            subtitle = f"Member {idx + 1}"

        ax.set_title(subtitle, fontsize=VIZ_CONFIG["font_size"] - 1,
                     fontweight="bold")
        ax.set_xlabel("East →", fontsize=VIZ_CONFIG["label_size"])
        ax.set_ylabel("↑ North", fontsize=VIZ_CONFIG["label_size"])
        ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])

    _domain_footnote(fig, terrain.elevation.shape[0], terrain.elevation.shape[1],
                     terrain.resolution_m, n_members=ensemble.n_members)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# 2.2  Arrival Time Distributions at Key Cells
# ---------------------------------------------------------------------------

def plot_arrival_distributions(
    ensemble,           # EnsembleResult with arrival_times [n, rows, cols]
    key_cells: Optional[list[tuple[int, int]]] = None,
    cell_labels: Optional[list[str]] = None,
    title: str = "Arrival Time Distributions at Key Cells",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Histograms of arrival times across ensemble members for selected cells.

    Shows whether the ensemble is unimodal (low uncertainty) or bimodal
    (high uncertainty, potential crown fire transition).
    """
    if (not hasattr(ensemble, "arrival_times") or
            ensemble.arrival_times is None or
            ensemble.arrival_times.ndim != 3):
        fig, ax = plt.subplots(figsize=figsize or (8, 4))
        ax.text(0.5, 0.5, "Per-member arrival times not available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=VIZ_CONFIG["font_size"])
        ax.set_title(title, fontsize=VIZ_CONFIG["font_size"], fontweight="bold")
        return fig

    arrival = ensemble.arrival_times   # [n_members, rows, cols]
    rows_g, cols_g = arrival.shape[1], arrival.shape[2]

    # Auto-select key cells if not provided
    if key_cells is None:
        bp = np.nanmean(np.isfinite(arrival) & (arrival >= 0), axis=0)
        # cell well behind (high bp), at boundary (~0.5 bp), well ahead (low bp)
        def _pick(cond_map):
            idx = np.argmax(cond_map)
            return divmod(int(idx), cols_g)

        behind = _pick(bp > 0.8)
        boundary = _pick(np.abs(bp - 0.5) < 0.15)
        ahead = _pick(bp < 0.2)
        key_cells = [behind, boundary, ahead]
        cell_labels = ["Well behind fire (high P)", "Fire boundary (~0.5 P)",
                       "Well ahead (low P)"]

    n_cells = len(key_cells)
    labels = cell_labels if cell_labels else [f"Cell {i+1}" for i in range(n_cells)]

    fw = figsize[0] if figsize else n_cells * 3.5
    fh = figsize[1] if figsize else 3.5
    fig, axes = plt.subplots(1, n_cells, figsize=(fw, fh), squeeze=False)
    fig.suptitle(title, fontsize=VIZ_CONFIG["font_size"] + 1, fontweight="bold")

    for i, (cell, label) in enumerate(zip(key_cells, labels)):
        ax = axes[0][i]
        cr, cc = cell
        times = arrival[:, cr, cc]
        finite = times[np.isfinite(times) & (times >= 0)]

        if len(finite) == 0:
            ax.text(0.5, 0.5, "No burns in\nthis cell",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=VIZ_CONFIG["font_size"])
            ax.set_title(label, fontsize=VIZ_CONFIG["font_size"] - 1, fontweight="bold")
            continue

        n_bins = max(8, min(20, ensemble.n_members // 2))
        ax.hist(finite / 60.0, bins=n_bins, color="#FF5722", edgecolor="white",
                alpha=0.85)
        frac_burned = len(finite) / ensemble.n_members
        ax.set_title(f"{label}\nP(burn)={frac_burned:.2f}",
                     fontsize=VIZ_CONFIG["font_size"] - 1, fontweight="bold")
        ax.set_xlabel("Arrival time (hr)", fontsize=VIZ_CONFIG["label_size"])
        ax.set_ylabel("Members", fontsize=VIZ_CONFIG["label_size"])
        ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# 2.3  Four-Way Strategy Comparison
# ---------------------------------------------------------------------------

def plot_strategy_comparison(
    cycle_report,        # CycleReport with evaluations dict keyed by strategy name
    ensemble,            # EnsembleResult
    terrain,             # TerrainGrid
    cycle_number: int = 1,
    title: str = "Strategy Comparison",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Four side-by-side maps showing drone placements per strategy, with a
    bar chart of entropy reduction below.
    """
    strategies = list(STRATEGY_STYLES.keys())
    evals = getattr(cycle_report, "evaluations", {}) or {}

    fw = figsize[0] if figsize else 16.0
    fh = figsize[1] if figsize else 8.0
    fig = plt.figure(figsize=(fw, fh))
    fig.suptitle(f"{title} — Cycle {cycle_number}",
                 fontsize=VIZ_CONFIG["font_size"] + 2, fontweight="bold")

    gs = gridspec.GridSpec(2, len(strategies), figure=fig,
                           height_ratios=[3, 1], hspace=0.35, wspace=0.3)

    hs = compute_hillshade(terrain.elevation, terrain.resolution_m)
    bp = ensemble.burn_probability if hasattr(ensemble, "burn_probability") else None

    perr_values = []
    strategy_labels = []

    for col_idx, sname in enumerate(strategies):
        ax = fig.add_subplot(gs[0, col_idx])
        style = STRATEGY_STYLES[sname]

        # Terrain + burn probability base
        ax.imshow(hs, origin="upper", cmap="gray", vmin=0, vmax=1, alpha=0.6)
        if bp is not None:
            ax.imshow(bp, origin="upper",
                      cmap=VIZ_CONFIG["burn_prob_cmap"],
                      alpha=0.5, vmin=0, vmax=1)

        # Drone placements from evaluation
        ev = evals.get(sname)
        if ev is not None and hasattr(ev, "selected_locations"):
            locs = ev.selected_locations or []
            for rank, (r, c) in enumerate(locs, start=1):
                ax.scatter([c], [r], marker=VIZ_CONFIG["drone_marker"],
                           c=[style["color"]], s=100, zorder=7,
                           edgecolors="black", linewidths=0.7)
                ax.text(c + 0.4, r - 0.4, str(rank),
                        fontsize=5, color="black", fontweight="bold", zorder=8)

            perr = getattr(ev, "perr", None)
            perr_str = f"PERR:{perr:.0f}" if perr is not None else ""
            ax.set_title(f"{style['label']}\n{perr_str}",
                         fontsize=VIZ_CONFIG["font_size"] - 1, fontweight="bold",
                         color=style["color"])
            if perr is not None:
                perr_values.append(perr)
                strategy_labels.append(style["label"])
        else:
            ax.set_title(style["label"],
                         fontsize=VIZ_CONFIG["font_size"] - 1, fontweight="bold",
                         color=style["color"])

        ax.set_xlabel("East →", fontsize=VIZ_CONFIG["label_size"])
        ax.set_ylabel("↑ North", fontsize=VIZ_CONFIG["label_size"])
        ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])

    # Bar chart of PERR / entropy reduction
    ax_bar = fig.add_subplot(gs[1, :])
    if perr_values:
        colors = [STRATEGY_STYLES[s]["color"] for s in strategies
                  if evals.get(s) and getattr(evals.get(s), "perr", None) is not None]
        ax_bar.bar(strategy_labels, perr_values, color=colors[:len(perr_values)],
                   edgecolor="black", linewidth=0.7)
        ax_bar.set_ylabel("Entropy Reduction (PERR)",
                          fontsize=VIZ_CONFIG["label_size"])
        ax_bar.tick_params(labelsize=VIZ_CONFIG["tick_size"])
        ax_bar.set_title("Entropy Reduction by Strategy",
                         fontsize=VIZ_CONFIG["font_size"] - 1)
    else:
        ax_bar.text(0.5, 0.5, "Strategy evaluation data not available",
                    ha="center", va="center", transform=ax_bar.transAxes,
                    fontsize=VIZ_CONFIG["font_size"])
        ax_bar.set_visible(True)

    return fig


# ---------------------------------------------------------------------------
# 2.4  Entropy Convergence Curve
# ---------------------------------------------------------------------------

def plot_entropy_convergence(
    history: list[dict],
    title: str = "Entropy Convergence",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Line plot of total predictive entropy (y) vs cycle number (x), one line
    per strategy.

    `history` is a list of per-cycle dicts:
        {"cycle": int, "greedy": float, "qubo": float, "uniform": float, ...}
    The key name must match STRATEGY_STYLES keys.
    """
    fw = figsize[0] if figsize else 8.0
    fh = figsize[1] if figsize else 5.0
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.suptitle(title, fontsize=VIZ_CONFIG["font_size"] + 1, fontweight="bold")

    cycles = [d.get("cycle", i + 1) for i, d in enumerate(history)]

    plotted = False
    for sname, style in STRATEGY_STYLES.items():
        values = [d.get(sname) for d in history]
        if any(v is not None for v in values):
            clean_x = [c for c, v in zip(cycles, values) if v is not None]
            clean_y = [v for v in values if v is not None]
            ax.plot(clean_x, clean_y,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=5,
                    linewidth=1.5,
                    label=style["label"])
            plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No entropy history provided",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=VIZ_CONFIG["font_size"])
    else:
        ax.set_xlabel("Cycle", fontsize=VIZ_CONFIG["label_size"])
        ax.set_ylabel("Total Predictive Entropy", fontsize=VIZ_CONFIG["label_size"])
        ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])
        ax.legend(fontsize=7, framealpha=0.8)
        ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# 2.5  Drone Value Curve
# ---------------------------------------------------------------------------

def plot_drone_value_curve(
    marginal_gains: Sequence[float],
    strategy_name: str = "greedy",
    title: str = "Drone Value Curve",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Marginal information gain (y) vs drone number (x) from sequential greedy
    selection. Shows diminishing returns; the knee is the optimal fleet size.
    """
    fw = figsize[0] if figsize else 6.0
    fh = figsize[1] if figsize else 4.0
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.suptitle(title, fontsize=VIZ_CONFIG["font_size"] + 1, fontweight="bold")

    style = STRATEGY_STYLES.get(strategy_name, STRATEGY_STYLES["greedy"])
    drone_nums = list(range(1, len(marginal_gains) + 1))

    ax.bar(drone_nums, marginal_gains,
           color=style["color"], edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.plot(drone_nums, marginal_gains,
            color=style["color"], linestyle="--", marker="o", markersize=4,
            linewidth=1.0, alpha=0.7)

    # Mark the knee (largest drop in marginal gain)
    if len(marginal_gains) > 1:
        gains = np.array(marginal_gains, dtype=float)
        drops = gains[:-1] - gains[1:]
        knee = int(np.argmax(drops))  # index before largest drop
        ax.axvline(x=drone_nums[knee], color="red", linestyle=":",
                   linewidth=1.2, label=f"Knee at drone {drone_nums[knee]}")
        ax.legend(fontsize=7)

    ax.set_xlabel("Drone Number (k)", fontsize=VIZ_CONFIG["label_size"])
    ax.set_ylabel("Marginal Information Gain", fontsize=VIZ_CONFIG["label_size"])
    ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# 2.6  QUBO vs Greedy Overlap Analysis
# ---------------------------------------------------------------------------

def plot_qubo_greedy_overlap(
    jaccard_history: Sequence[float],
    cycle_reports: Optional[list] = None,
    ensemble: Optional[object] = None,
    terrain: Optional[object] = None,
    title: str = "QUBO vs Greedy Overlap",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Per-cycle Jaccard similarity between greedy and QUBO selections.
    When provided, also draws side-by-side maps for the cycle of lowest
    agreement.
    """
    show_maps = (
        cycle_reports is not None and
        ensemble is not None and
        terrain is not None and
        len(cycle_reports) > 0
    )

    fw = figsize[0] if figsize else (12.0 if show_maps else 7.0)
    fh = figsize[1] if figsize else 4.5

    if show_maps:
        fig = plt.figure(figsize=(fw, fh))
        fig.suptitle(title, fontsize=VIZ_CONFIG["font_size"] + 1, fontweight="bold")
        gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2, 1, 1], wspace=0.35)
        ax_jac = fig.add_subplot(gs[0])
        ax_g   = fig.add_subplot(gs[1])
        ax_q   = fig.add_subplot(gs[2])
    else:
        fig, ax_jac = plt.subplots(figsize=(fw, fh))
        fig.suptitle(title, fontsize=VIZ_CONFIG["font_size"] + 1, fontweight="bold")
        ax_g = ax_q = None

    # Jaccard line
    cycles = list(range(1, len(jaccard_history) + 1))
    ax_jac.plot(cycles, jaccard_history,
                color=STRATEGY_STYLES["qubo"]["color"],
                linestyle="-", marker="o", markersize=5, linewidth=1.5,
                label="QUBO–Greedy Jaccard")
    ax_jac.axhline(y=0.8, color="green", linestyle=":", linewidth=1.0,
                   label="High agreement (0.8)")
    ax_jac.axhline(y=0.5, color="red", linestyle=":", linewidth=1.0,
                   label="Low agreement (0.5)")
    ax_jac.set_ylim(0, 1.05)
    ax_jac.set_xlabel("Cycle", fontsize=VIZ_CONFIG["label_size"])
    ax_jac.set_ylabel("Jaccard Similarity", fontsize=VIZ_CONFIG["label_size"])
    ax_jac.tick_params(labelsize=VIZ_CONFIG["tick_size"])
    ax_jac.legend(fontsize=6)
    ax_jac.grid(True, linestyle=":", alpha=0.4)

    if show_maps and ax_g is not None and ax_q is not None:
        # Find cycle of lowest agreement
        worst_idx = int(np.argmin(jaccard_history))
        worst_report = cycle_reports[worst_idx]
        evals = getattr(worst_report, "evaluations", {}) or {}

        hs = compute_hillshade(terrain.elevation, terrain.resolution_m)
        bp = (ensemble.burn_probability
              if hasattr(ensemble, "burn_probability") else None)

        for ax, sname, label_str in [(ax_g, "greedy", "Greedy"),
                                     (ax_q, "qubo", "QUBO")]:
            ax.imshow(hs, origin="upper", cmap="gray", vmin=0, vmax=1, alpha=0.6)
            if bp is not None:
                ax.imshow(bp, origin="upper",
                          cmap=VIZ_CONFIG["burn_prob_cmap"],
                          alpha=0.45, vmin=0, vmax=1)
            ev = evals.get(sname)
            if ev is not None and hasattr(ev, "selected_locations"):
                style = STRATEGY_STYLES[sname]
                for rank, (r, c) in enumerate(ev.selected_locations or [], start=1):
                    ax.scatter([c], [r], marker=VIZ_CONFIG["drone_marker"],
                               c=[style["color"]], s=100, zorder=7,
                               edgecolors="black", linewidths=0.7)
                    ax.text(c + 0.4, r - 0.4, str(rank),
                            fontsize=5, color="black", fontweight="bold", zorder=8)
            ax.set_title(f"{label_str}\n(Cycle {worst_idx + 1}, "
                         f"J={jaccard_history[worst_idx]:.2f})",
                         fontsize=VIZ_CONFIG["font_size"] - 2, fontweight="bold")
            ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# 2.7  Placement Stability Plot
# ---------------------------------------------------------------------------

def plot_placement_stability(
    jaccard_history: Sequence[float],
    strategy_name: str = "greedy",
    title: str = "Placement Stability",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Jaccard similarity between the primary strategy's selections at consecutive
    cycles. High stability → system doesn't need full recompute every cycle.
    """
    fw = figsize[0] if figsize else 7.0
    fh = figsize[1] if figsize else 4.0
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.suptitle(title, fontsize=VIZ_CONFIG["font_size"] + 1, fontweight="bold")

    style = STRATEGY_STYLES.get(strategy_name, STRATEGY_STYLES["greedy"])
    cycles = list(range(2, len(jaccard_history) + 2))  # cycle-to-cycle transitions

    ax.plot(cycles, jaccard_history,
            color=style["color"], linestyle="-", marker="o",
            markersize=5, linewidth=1.5, label=style["label"])
    ax.axhline(y=0.7, color="green", linestyle=":", linewidth=1.0,
               label="High stability (0.7)")
    ax.axhline(y=0.3, color="red", linestyle=":", linewidth=1.0,
               label="Low stability (0.3)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Cycle (transition)", fontsize=VIZ_CONFIG["label_size"])
    ax.set_ylabel("Jaccard Similarity (consec. cycles)",
                  fontsize=VIZ_CONFIG["label_size"])
    ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])
    ax.legend(fontsize=7)
    ax.grid(True, linestyle=":", alpha=0.4)

    # Mean stability annotation
    if jaccard_history:
        mean_j = float(np.mean(jaccard_history))
        ax.axhline(y=mean_j, color=style["color"], linestyle="--",
                   linewidth=0.8, alpha=0.6, label=f"Mean={mean_j:.2f}")
        ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# 2.8  Ground Truth Reveal
# ---------------------------------------------------------------------------

def plot_ground_truth_reveal(
    ground_truth,                          # GroundTruth dataclass
    gp_prior,                              # GPPrior
    drone_obs_locs: Optional[list[tuple[int, int]]] = None,
    variable: str = "fmc",                 # "fmc" or "wind_speed"
    title: str = "Ground Truth Reveal",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Side-by-side: hidden ground truth vs GP estimate, with residual map.

    Regions where drones observed should have low residual. Demonstrates that
    the system correctly prioritized measurement where it mattered.
    """
    fw = figsize[0] if figsize else 13.0
    fh = figsize[1] if figsize else 4.5
    fig, axes = plt.subplots(1, 3, figsize=(fw, fh))
    fig.suptitle(title, fontsize=VIZ_CONFIG["font_size"] + 1, fontweight="bold")

    if variable == "fmc":
        truth_field = ground_truth.fmc_field
        est_field   = gp_prior.fmc_mean
        label       = "FMC"
        cmap        = VIZ_CONFIG["fmc_cmap"]
    else:
        truth_field = ground_truth.wind_speed_field
        est_field   = gp_prior.wind_speed_mean
        label       = "Wind Speed (m/s)"
        cmap        = VIZ_CONFIG["uncertainty_cmap"]

    vmin = float(min(truth_field.min(), est_field.min()))
    vmax = float(max(truth_field.max(), est_field.max()))

    _imshow(axes[0], truth_field, f"Ground Truth {label}",
            cmap, vmin, vmax, label)
    _imshow(axes[1], est_field, f"GP Estimate {label}",
            cmap, vmin, vmax, label)

    residual = truth_field - est_field
    abs_r = float(np.abs(residual).max())
    _imshow(axes[2], residual, f"Residual (truth − GP)",
            "RdBu_r", -abs_r, abs_r, f"Δ {label}")

    # Mark drone observation locations on all panels
    if drone_obs_locs:
        for ax in axes:
            _mark_stations(ax, drone_obs_locs, label="Drone obs",
                           color="cyan", marker=VIZ_CONFIG["drone_marker"], size=50)
            _add_legend(ax)

    _domain_footnote(fig, truth_field.shape[0], truth_field.shape[1],
                     getattr(ground_truth, "resolution_m", 50.0))
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# 2.9  Innovation Tracking
# ---------------------------------------------------------------------------

def plot_innovation_tracking(
    innovations_history: list[dict],
    title: str = "Innovation Tracking",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Plot of |observation - prediction| at drone measurement locations over
    cycles.

    `innovations_history` is a list of per-cycle dicts:
        {"cycle": int, "fmc_mean_abs": float, "wind_speed_mean_abs": float}
    If innovations decrease, the model is learning.
    """
    fw = figsize[0] if figsize else 7.0
    fh = figsize[1] if figsize else 4.5
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.suptitle(title, fontsize=VIZ_CONFIG["font_size"] + 1, fontweight="bold")

    cycles = [d.get("cycle", i + 1) for i, d in enumerate(innovations_history)]

    for key, color, label in [
        ("fmc_mean_abs",        "#FF5722", "FMC |innovation|"),
        ("wind_speed_mean_abs", "#2196F3", "Wind speed |innovation|"),
    ]:
        values = [d.get(key) for d in innovations_history]
        if any(v is not None for v in values):
            cx = [c for c, v in zip(cycles, values) if v is not None]
            vy = [v for v in values if v is not None]
            ax.plot(cx, vy, color=color, linestyle="-", marker="o",
                    markersize=5, linewidth=1.5, label=label)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No innovation data provided",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=VIZ_CONFIG["font_size"])

    ax.set_xlabel("Cycle", fontsize=VIZ_CONFIG["label_size"])
    ax.set_ylabel("|Observation − Prediction|", fontsize=VIZ_CONFIG["label_size"])
    ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
