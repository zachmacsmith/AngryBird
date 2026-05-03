"""
Presentation-specific visualizations (§3 of Vizualisations.md).

These are static slides built from system outputs — not part of the
deployed system, used only for hackathon presentations.

Public API
----------
plot_observation_gap    3.1  RAWS station coverage map with fire perimeter
plot_architecture       3.2  Simplified architecture flow diagram
plot_before_after       3.3  Before/after information-field side-by-side
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ._style import (
    VIZ_CONFIG,
    _imshow,
    _mark_stations,
    _add_legend,
    _domain_footnote,
    save_or_show,
)


# ---------------------------------------------------------------------------
# 3.1  Observation Gap Slide
# ---------------------------------------------------------------------------

def plot_observation_gap(
    terrain,                            # TerrainGrid (used for extent background)
    fire_perimeter_mask: np.ndarray,    # bool[rows, cols] — true = inside perimeter
    raws_locs: Optional[list[tuple[int, int]]] = None,  # (row, col) list
    n_raws_within: int = 0,
    domain_name: str = "Study Domain",
    title: str = "The Observation Gap",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Map showing RAWS station locations relative to the fire perimeter.

    Punchline: within the fire perimeter, there are zero (or very few)
    ground-truth measurements. Everything is interpolated from afar.
    """
    fw = figsize[0] if figsize else 8.0
    fh = figsize[1] if figsize else 7.0
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.suptitle(title, fontsize=VIZ_CONFIG["font_size"] + 3,
                 fontweight="bold")

    rows, cols = terrain.elevation.shape

    # Terrain elevation background
    ax.imshow(terrain.elevation, origin="upper",
              cmap=VIZ_CONFIG["terrain_cmap"], alpha=0.7)

    # Fire perimeter shading
    ax.imshow(
        np.where(fire_perimeter_mask, 1.0, np.nan),
        origin="upper", cmap="Reds", alpha=0.55, vmin=0, vmax=1,
    )

    # Fire perimeter outline
    from matplotlib.contour import QuadContourSet  # local import
    ax.contour(
        fire_perimeter_mask.astype(float),
        levels=[0.5], colors=["#D32F2F"], linewidths=[2.0],
    )

    # RAWS station markers
    if raws_locs:
        rs = [r for r, c in raws_locs]
        cs = [c for r, c in raws_locs]
        ax.scatter(cs, rs,
                   marker=VIZ_CONFIG["raws_marker"],
                   c="yellow", s=80, zorder=6,
                   edgecolors="black", linewidths=0.8,
                   label=f"RAWS stations (n={len(raws_locs)})")

    # Annotation box
    km_x = cols * terrain.resolution_m / 1000
    km_y = rows * terrain.resolution_m / 1000
    msg = (
        f"{domain_name}  ·  {km_x:.1f} km × {km_y:.1f} km\n"
        f"RAWS within fire perimeter: {n_raws_within}\n"
        '"Everything the model uses is interpolated from the edges."'
    )
    ax.text(0.02, 0.03, msg, transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.88, edgecolor="gray"))

    ax.set_title(f"Fire Perimeter vs Observation Network",
                 fontsize=VIZ_CONFIG["font_size"], fontweight="bold")
    ax.set_xlabel("East →", fontsize=VIZ_CONFIG["label_size"])
    ax.set_ylabel("↑ North", fontsize=VIZ_CONFIG["label_size"])
    ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=7, loc="upper right",
                  framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# 3.2  Architecture Diagram
# ---------------------------------------------------------------------------

def plot_architecture(
    title: str = "IGNIS System Architecture",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Simplified architecture flow diagram: boxes and arrows for the
    predict → identify uncertainty → route drones → measure → update → repeat
    loop.

    Entirely drawn in matplotlib — no external dependencies.
    """
    fw = figsize[0] if figsize else 12.0
    fh = figsize[1] if figsize else 6.0
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.suptitle(title, fontsize=VIZ_CONFIG["font_size"] + 3, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # --- node definitions: (label, x_center, y_center, color) ---
    nodes = [
        ("Fire\nEnsemble\n(predict)", 1.2, 3.8, "#FF5722"),
        ("GP Uncertainty\n(quantify)", 3.5, 3.8, "#9C27B0"),
        ("Information\nField (w_i)", 5.8, 3.8, "#3F51B5"),
        ("Drone\nSelector\n(greedy / QUBO)", 8.0, 3.8, "#00BCD4"),
        ("UTM /\nDrone Ops", 8.0, 1.8, "#4CAF50"),
        ("GP Update\n(EnKF)", 5.8, 1.8, "#FF9800"),
        ("RAWS /\nSatellite", 3.5, 1.8, "#795548"),
        ("Terrain\n& Fuel", 1.2, 1.8, "#607D8B"),
    ]

    box_w, box_h = 1.5, 0.9

    def _box(ax_, cx, cy, label, color):
        rect = FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.08",
            linewidth=1.2, edgecolor="black",
            facecolor=color, alpha=0.85, zorder=3,
        )
        ax_.add_patch(rect)
        ax_.text(cx, cy, label, ha="center", va="center",
                 fontsize=7, fontweight="bold", color="white",
                 zorder=4, wrap=True)

    for lbl, cx, cy, col in nodes:
        _box(ax, cx, cy, lbl, col)

    # --- arrows ---
    def _arrow(ax_, x1, y1, x2, y2, label=""):
        ax_.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#333333",
                            lw=1.2, mutation_scale=12),
            zorder=2,
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15
            ax_.text(mx, my, label, ha="center", fontsize=6,
                     color="#333333", zorder=5)

    # Top row (left to right)
    _arrow(ax, 1.2 + box_w / 2, 3.8, 3.5 - box_w / 2, 3.8, "ensemble")
    _arrow(ax, 3.5 + box_w / 2, 3.8, 5.8 - box_w / 2, 3.8, "variance")
    _arrow(ax, 5.8 + box_w / 2, 3.8, 8.0 - box_w / 2, 3.8, "w_i field")

    # Down from selector to UTM
    _arrow(ax, 8.0, 3.8 - box_h / 2, 8.0, 1.8 + box_h / 2, "missions")

    # Bottom row (right to left)
    _arrow(ax, 8.0 - box_w / 2, 1.8, 5.8 + box_w / 2, 1.8, "obs")
    _arrow(ax, 5.8 - box_w / 2, 1.8, 3.5 + box_w / 2, 1.8, "GP update")
    _arrow(ax, 3.5 - box_w / 2, 1.8, 1.2 + box_w / 2, 1.8, "prior")

    # Terrain feeds fire ensemble (up)
    _arrow(ax, 1.2, 1.8 + box_h / 2, 1.2, 3.8 - box_h / 2, "DEM/fuel")

    # GP Update feeds GP uncertainty (up)
    _arrow(ax, 5.8, 1.8 + box_h / 2, 5.8, 3.8 - box_h / 2, "posterior")

    # Cycle label
    ax.text(5.0, 0.3, "← IGNIS continuous assimilation loop →",
            ha="center", fontsize=8, style="italic", color="#555555")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# 3.3  Before / After Information Field
# ---------------------------------------------------------------------------

def plot_before_after(
    info_field_before: np.ndarray,     # w_total at cycle 1
    info_field_after: np.ndarray,      # w_total at cycle N
    terrain,                            # TerrainGrid (for footnote)
    drone_obs_locs: Optional[list[tuple[int, int]]] = None,
    cycle_before: int = 1,
    cycle_after: int = 5,
    title: str = "Before / After Information Field",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Side-by-side information field heatmaps from cycle 1 (before any drone
    observations) and cycle N (after targeted sensing).

    The progressive darkening of the heatmap (uncertainty reduction) is the
    visual proof that the system works.
    """
    fw = figsize[0] if figsize else 11.0
    fh = figsize[1] if figsize else 5.0
    fig, axes = plt.subplots(1, 2, figsize=(fw, fh))
    fig.suptitle(title, fontsize=VIZ_CONFIG["font_size"] + 2, fontweight="bold")

    # Shared color scale so comparison is meaningful
    vmax = float(max(info_field_before.max(), info_field_after.max()))
    vmin = 0.0

    _imshow(axes[0], info_field_before,
            f"Cycle {cycle_before} — No drone observations",
            VIZ_CONFIG["info_field_cmap"], vmin, vmax,
            "Information value w_i")

    _imshow(axes[1], info_field_after,
            f"Cycle {cycle_after} — After targeted sensing",
            VIZ_CONFIG["info_field_cmap"], vmin, vmax,
            "Information value w_i")

    # Mark drone observation locations on the "after" panel
    if drone_obs_locs:
        _mark_stations(axes[1], drone_obs_locs, label="Drone obs",
                       color="cyan", marker=VIZ_CONFIG["drone_marker"], size=50)
        _add_legend(axes[1])

    # Reduction annotation
    total_before = float(info_field_before.sum())
    total_after  = float(info_field_after.sum())
    if total_before > 1e-9:
        pct = 100.0 * (total_before - total_after) / total_before
        axes[1].text(
            0.02, 0.98,
            f"Σw reduction: {pct:.1f}%",
            transform=axes[1].transAxes,
            fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="gray"),
        )

    _domain_footnote(fig, info_field_before.shape[0],
                     info_field_before.shape[1], terrain.resolution_m)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    return fig
