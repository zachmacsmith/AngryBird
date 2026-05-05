"""
Shared style constants, colormaps, and low-level rendering helpers.

All visualization modules import from here so the system is visually consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Global style config (mirrors Vizualisations.md §4)
# ---------------------------------------------------------------------------

VIZ_CONFIG: dict = {
    "terrain_cmap":      "terrain",
    "burn_prob_cmap":    "YlOrRd",
    "info_field_cmap":   "inferno",
    "uncertainty_cmap":  "viridis",
    "fmc_cmap":          "YlGn",
    "sensitivity_cmap":  "RdBu_r",
    "crown_risk_cmap":   "PuRd",
    "attribution_alpha": 0.90,
    "drone_marker":      "^",
    "raws_marker":       "o",
    "satellite_marker":  "s",
    "figure_dpi":        150,
    "font_size":         10,
    "tick_size":         8,
    "label_size":        8,
}

# Per-strategy consistent colours / line styles
STRATEGY_STYLES: dict[str, dict] = {
    "greedy":     {"color": "#2196F3", "linestyle": "-",  "marker": "o", "label": "Greedy"},
    "qubo":       {"color": "#FF9800", "linestyle": "--", "marker": "s", "label": "QUBO"},
    "uniform":    {"color": "#9E9E9E", "linestyle": ":",  "marker": "D", "label": "Uniform"},
    "fire_front": {"color": "#4CAF50", "linestyle": "-.", "marker": "^", "label": "Fire Front"},
}

# Drone-assignment colours (up to 10 drones)
DRONE_COLORS = [
    "#E91E63", "#9C27B0", "#3F51B5", "#00BCD4",
    "#8BC34A", "#FF5722", "#795548", "#607D8B",
    "#F44336", "#2196F3",
]


# ---------------------------------------------------------------------------
# Hillshade
# ---------------------------------------------------------------------------

def compute_hillshade(
    elevation: np.ndarray,
    resolution_m: float = 50.0,
    azdeg: float = 315.0,
    altdeg: float = 45.0,
    vert_exag: float = 2.0,
) -> np.ndarray:
    """Compute hillshade from a DEM. Returns float32 array in [0, 1]."""
    ls = mcolors.LightSource(azdeg=azdeg, altdeg=altdeg)
    hs = ls.hillshade(
        elevation.astype(np.float64),
        vert_exag=vert_exag,
        dx=resolution_m,
        dy=resolution_m,
    )
    return hs.astype(np.float32)


# ---------------------------------------------------------------------------
# Low-level panel helpers
# ---------------------------------------------------------------------------

def _imshow(
    ax: Axes,
    data: np.ndarray,
    title: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: str = "",
    alpha: float = 1.0,
    interpolation: str = "nearest",
    extent: Optional[Sequence[float]] = None,  # [left, right, bottom, top] in km
) -> None:
    """Standard imshow with title and colorbar, matching global style."""
    im = ax.imshow(
        data, origin="upper", cmap=cmap,
        vmin=vmin, vmax=vmax,
        alpha=alpha, interpolation=interpolation,
        extent=extent, aspect="auto",
    )
    ax.set_title(title, fontsize=VIZ_CONFIG["font_size"], fontweight="bold")
    xlabel = "Easting (km) →" if extent is not None else "Easting (px) →"
    ylabel = "→ Northing (km)" if extent is not None else "→ Northing (px)"
    ax.set_xlabel(xlabel, fontsize=VIZ_CONFIG["label_size"])
    ax.set_ylabel(ylabel, fontsize=VIZ_CONFIG["label_size"])
    ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(colorbar_label, fontsize=VIZ_CONFIG["label_size"])
    cb.ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])


def _mark_stations(
    ax: Axes,
    locs: list[tuple[int, int]],
    label: str = "RAWS",
    color: str = "white",
    marker: str = "o",
    size: int = 70,
) -> None:
    if not locs:
        return
    rs = [r for r, c in locs]
    cs = [c for r, c in locs]
    ax.scatter(cs, rs, marker=marker, c=color, s=size, zorder=6,
               edgecolors="black", linewidths=0.8, label=label)


def _mark_drone_targets(
    ax: Axes,
    locations: list[tuple[int, int]],
    info_w: Optional[np.ndarray] = None,
    color: str = "cyan",
    label: str = "Drone targets",
    annotate_rank: bool = True,
) -> None:
    """Mark drone targets. Size proportional to w if info_w provided."""
    if not locations:
        return
    w_max = float(info_w.max()) if info_w is not None else 1.0
    for rank, (r, c) in enumerate(locations, start=1):
        w = float(info_w[r, c]) if info_w is not None else 1.0
        size = 120 + 300 * (w / max(w_max, 1e-9))
        ax.scatter([c], [r], marker=VIZ_CONFIG["drone_marker"],
                   c=[color], s=size, zorder=7,
                   edgecolors="black", linewidths=0.7)
        if annotate_rank:
            ax.text(c + 0.5, r - 0.5, str(rank),
                    fontsize=6, color="black",
                    fontweight="bold", zorder=8)
    # Legend proxy
    ax.scatter([], [], marker=VIZ_CONFIG["drone_marker"],
               c=color, s=120, edgecolors="black",
               linewidths=0.7, label=label)


def _add_legend(ax: Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=6, loc="upper right",
                  framealpha=0.75, markerscale=0.9)


def _domain_footnote(fig: Figure, rows: int, cols: int, resolution_m: float,
                     n_members: Optional[int] = None) -> None:
    km_x = cols * resolution_m / 1000
    km_y = rows * resolution_m / 1000
    note = f"Domain: {km_x:.1f} km × {km_y:.1f} km  |  Resolution: {resolution_m:.0f} m"
    if n_members is not None:
        note += f"  |  N={n_members} ensemble members"
    fig.text(0.5, 0.005, note, ha="center", fontsize=7, color="gray")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def save_or_show(
    fig: Figure,
    path: Optional[str | Path] = None,
    dpi: int = VIZ_CONFIG["figure_dpi"],
) -> None:
    """Save figure to path (if given), otherwise display it."""
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {path}")
    else:
        plt.show()
    plt.close(fig)
