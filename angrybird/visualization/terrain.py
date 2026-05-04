"""Birds-eye terrain visualizer for TerrainData."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from ..types import TerrainData
from ._style import (
    VIZ_CONFIG, compute_hillshade, _imshow, save_or_show, _domain_footnote,
)


# ---------------------------------------------------------------------------
# SB40 fuel model category colormap
# ---------------------------------------------------------------------------

# Maps SB40 code → (category label, hex color)
_SB40_CATEGORIES: list[tuple[range | set, str, str]] = [
    (set([0]),          "Non-burnable",  "#AAAAAA"),
    (range(91, 100),    "Non-burnable",  "#AAAAAA"),
    (range(101, 110),   "Grass",         "#F5E642"),
    (range(121, 125),   "Grass-Shrub",   "#A8CC3C"),
    (range(141, 150),   "Shrub",         "#7A8C32"),
    (range(161, 166),   "Timber Under.", "#C49A3C"),
    (range(181, 190),   "Timber Litter", "#8B6530"),
    (range(201, 205),   "Slash",         "#4A2800"),
]

_CATEGORY_COLORS = {label: color for _, label, color in _SB40_CATEGORIES}
_CATEGORY_ORDER  = list(dict.fromkeys(label for _, label, _ in _SB40_CATEGORIES))


def _fuel_model_rgb(fuel_model: np.ndarray) -> np.ndarray:
    """Convert SB40 code array to an RGB image using category colors."""
    rgb = np.full((*fuel_model.shape, 3), 0.67, dtype=np.float32)  # default gray
    for code_range, _, hex_color in _SB40_CATEGORIES:
        r, g, b = mcolors.to_rgb(hex_color)
        mask = np.isin(fuel_model, list(code_range))
        rgb[mask] = (r, g, b)
    return rgb


def _fuel_model_legend(ax) -> None:
    seen: set[str] = set()
    handles = []
    for _, label, color in _SB40_CATEGORIES:
        if label not in seen:
            seen.add(label)
            handles.append(mpatches.Patch(facecolor=color, edgecolor="gray",
                                          linewidth=0.4, label=label))
    ax.legend(handles=handles, fontsize=5, loc="lower right",
              framealpha=0.8, ncol=2, borderpad=0.4, labelspacing=0.3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_terrain_overview(
    terrain: TerrainData,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """
    8-panel birds-eye view of all TerrainData layers.

    Layout (2 rows × 4 cols):
        Elevation  |  Slope  |  Aspect  |  Fuel Model
        Canopy Cov |  Canopy H  |  Canopy CBH  |  Canopy CBD
    """
    rows, cols = terrain.shape
    res = terrain.resolution_m
    hs = compute_hillshade(terrain.elevation, res)

    # Extent in km: [left, right, bottom, top] with origin="upper"
    w_km = cols * res / 1000
    h_km = rows * res / 1000
    ext = [0, w_km, h_km, 0]

    fig, axes = plt.subplots(
        2, 4,
        figsize=(18, 9),
        dpi=VIZ_CONFIG["figure_dpi"],
        constrained_layout=True,
    )
    fig.suptitle("Terrain Overview", fontsize=13, fontweight="bold")

    # --- Row 0 ---

    # Elevation with hillshade overlay
    ax = axes[0, 0]
    norm_elev = plt.Normalize(terrain.elevation.min(), terrain.elevation.max())
    elev_rgb = plt.get_cmap("terrain")(norm_elev(terrain.elevation))
    elev_rgb[..., :3] *= hs[..., np.newaxis]
    ax.imshow(elev_rgb, origin="upper", interpolation="nearest", extent=ext)
    sm = plt.cm.ScalarMappable(cmap="terrain", norm=norm_elev)
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("m", fontsize=VIZ_CONFIG["label_size"])
    cb.ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])
    _panel_labels(ax, "Elevation", has_extent=True)

    # Slope
    _imshow(axes[0, 1], terrain.slope, "Slope", "YlOrBr",
            vmin=0, colorbar_label="°", extent=ext)

    # Aspect
    _imshow(axes[0, 2], terrain.aspect, "Aspect", "hsv",
            vmin=0, vmax=360, colorbar_label="° from N", extent=ext)

    # Fuel model
    ax = axes[0, 3]
    ax.imshow(_fuel_model_rgb(terrain.fuel_model), origin="upper",
              interpolation="nearest", extent=ext)
    _panel_labels(ax, "Fuel Model (SB40)", has_extent=True)
    _fuel_model_legend(ax)

    # --- Row 1 ---

    _imshow(axes[1, 0], terrain.canopy_cover, "Canopy Cover", "YlGn",
            vmin=0, vmax=1, colorbar_label="fraction", extent=ext)

    _imshow(axes[1, 1], terrain.canopy_height, "Canopy Height", "Greens",
            vmin=0, colorbar_label="m", extent=ext)

    _imshow(axes[1, 2], terrain.canopy_base_height, "Canopy Base Height", "BuGn",
            vmin=0, colorbar_label="m", extent=ext)

    _imshow(axes[1, 3], terrain.canopy_bulk_density, "Canopy Bulk Density", "Blues",
            vmin=0, colorbar_label="kg/m³", extent=ext)

    _domain_footnote(fig, rows, cols, res)

    if save_path is not None:
        save_or_show(fig, save_path)
    return fig


def plot_terrain_elevation(
    terrain: TerrainData,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """Single-panel hillshaded elevation map."""
    res = terrain.resolution_m
    hs = compute_hillshade(terrain.elevation, res)

    rows, cols = terrain.shape
    w_km = cols * res / 1000
    h_km = rows * res / 1000
    ext = [0, w_km, h_km, 0]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=VIZ_CONFIG["figure_dpi"],
                           constrained_layout=True)

    norm_elev = plt.Normalize(terrain.elevation.min(), terrain.elevation.max())
    elev_rgb = plt.get_cmap("terrain")(norm_elev(terrain.elevation))
    elev_rgb[..., :3] *= hs[..., np.newaxis]
    ax.imshow(elev_rgb, origin="upper", interpolation="nearest", extent=ext)

    sm = plt.cm.ScalarMappable(cmap="terrain", norm=norm_elev)
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("Elevation (m)", fontsize=VIZ_CONFIG["label_size"])
    cb.ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])

    _panel_labels(ax, "Elevation", has_extent=True)
    _domain_footnote(fig, rows, cols, res)

    if save_path is not None:
        save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _panel_labels(ax, title: str, has_extent: bool = False) -> None:
    ax.set_title(title, fontsize=VIZ_CONFIG["font_size"], fontweight="bold")
    xlabel = "Easting (km) →" if has_extent else "Easting (px) →"
    ylabel = "→ Northing (km)" if has_extent else "→ Northing (px)"
    ax.set_xlabel(xlabel, fontsize=VIZ_CONFIG["label_size"])
    ax.set_ylabel(ylabel, fontsize=VIZ_CONFIG["label_size"])
    ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])
