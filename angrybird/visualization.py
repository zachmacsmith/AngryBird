"""
Visualization utilities for IGNIS outputs.

Public surface:
  plot_gp_prior(gp_prior, ...)                    — FMC mean + variance
  plot_information_field(info_field, ensemble, ...) — w heatmap + burn prob
  plot_phase2_summary(...)                          — 2×3 dashboard
  save_or_show(fig, path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

from .types import EnsembleResult, GPPrior, InformationField


# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------

_CMAP_VAR  = "YlOrRd"      # GP variance — yellow → orange → red
_CMAP_BURN = "hot_r"       # burn probability — white → orange → black
_CMAP_SENS = "RdBu_r"      # sensitivity — blue (neg) → white → red (pos)
_CMAP_INFO = "viridis"     # information field
_CMAP_FMC  = "YlGn"       # FMC mean — yellow (dry) → green (wet)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _imshow(
    ax: Axes,
    data: np.ndarray,
    title: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: str = "",
) -> None:
    im = ax.imshow(
        data, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("East →", fontsize=8)
    ax.set_ylabel("↑ North", fontsize=8)
    ax.tick_params(labelsize=7)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(colorbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def _mark_stations(
    ax: Axes,
    station_locs: list[tuple[int, int]],
    label: str = "RAWS",
) -> None:
    if not station_locs:
        return
    rows = [r for r, c in station_locs]
    cols = [c for r, c in station_locs]
    ax.scatter(cols, rows, marker="^", c="white", s=80, zorder=5,
               edgecolors="black", linewidths=0.8)
    ax.scatter([], [], marker="^", c="white", edgecolors="black",
               linewidths=0.8, s=80, label=label)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)


def _mark_selections(
    ax: Axes,
    locations: list[tuple[int, int]],
    color: str = "cyan",
    marker: str = "*",
    label: str = "",
    size: int = 120,
) -> None:
    if not locations:
        return
    rows = [r for r, c in locations]
    cols = [c for r, c in locations]
    ax.scatter(cols, rows, marker=marker, c=color, s=size, zorder=6,
               edgecolors="black", linewidths=0.6, label=label)
    if label:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)


# ---------------------------------------------------------------------------
# Individual panel plots
# ---------------------------------------------------------------------------

def plot_gp_prior(
    ax_mean: Axes,
    ax_var: Axes,
    gp_prior: GPPrior,
    station_locs: Optional[list[tuple[int, int]]] = None,
) -> None:
    """Two panels: FMC posterior mean and FMC posterior variance."""
    _imshow(ax_mean, gp_prior.fmc_mean,
            "FMC Posterior Mean", _CMAP_FMC, colorbar_label="FMC (fraction)")
    _imshow(ax_var, gp_prior.fmc_variance,
            "FMC Posterior Variance (GP uncertainty)", _CMAP_VAR,
            colorbar_label="σ² (fraction²)")
    for ax in (ax_mean, ax_var):
        _mark_stations(ax, station_locs or [])


def plot_ensemble(ax: Axes, ensemble: EnsembleResult) -> None:
    """Burn probability from ensemble."""
    _imshow(ax, ensemble.burn_probability,
            "Ensemble Burn Probability", _CMAP_BURN,
            vmin=0.0, vmax=1.0, colorbar_label="P(burn)")


def plot_sensitivity(
    ax: Axes,
    info_field: InformationField,
    variable: str = "fmc",
) -> None:
    sens = info_field.sensitivity.get(variable)
    if sens is None:
        ax.set_visible(False)
        return
    vabs = max(np.abs(sens).max(), 1e-9)
    _imshow(ax, sens,
            f"Sensitivity  ∂arrival/∂{variable}",
            _CMAP_SENS, vmin=-vabs, vmax=vabs,
            colorbar_label="correlation coefficient")


def plot_information_field(
    ax: Axes,
    info_field: InformationField,
    selected_locs: Optional[list[tuple[int, int]]] = None,
    station_locs: Optional[list[tuple[int, int]]] = None,
    title: str = "Information Field  w(x)",
) -> None:
    """Information field heatmap with optional drone waypoints overlaid."""
    _imshow(ax, info_field.w, title, _CMAP_INFO, vmin=0,
            colorbar_label="w  (GP var × sensitivity × observability)")
    _mark_stations(ax, station_locs or [])
    if selected_locs:
        _mark_selections(ax, selected_locs, color="cyan", marker="*",
                         label="Drone targets", size=140)


# ---------------------------------------------------------------------------
# Full Phase 2 dashboard
# ---------------------------------------------------------------------------

def plot_phase2_summary(
    gp_prior: GPPrior,
    ensemble: EnsembleResult,
    info_field: InformationField,
    station_locs: Optional[list[tuple[int, int]]] = None,
    selected_locs: Optional[list[tuple[int, int]]] = None,
    resolution_m: float = 50.0,
    title: str = "IGNIS · Phase 2 Summary",
    figsize: tuple[int, int] = (16, 10),
) -> Figure:
    """
    2×3 dashboard:
      [FMC mean]  [FMC variance]       [Burn probability]
      [Sensitivity FMC]  [Sensitivity wind]  [Information field]
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # Row 0
    plot_gp_prior(axes[0, 0], axes[0, 1], gp_prior, station_locs)
    plot_ensemble(axes[0, 2], ensemble)

    # Row 1
    plot_sensitivity(axes[1, 0], info_field, variable="fmc")
    axes[1, 0].set_title("FMC Sensitivity  ∂arrival/∂FMC", fontsize=10, fontweight="bold")

    plot_sensitivity(axes[1, 1], info_field, variable="wind_speed")
    axes[1, 1].set_title("Wind Sensitivity  ∂arrival/∂wind", fontsize=10, fontweight="bold")

    plot_information_field(axes[1, 2], info_field,
                           selected_locs=selected_locs,
                           station_locs=station_locs)

    # Domain scale annotation
    rows, cols = gp_prior.fmc_mean.shape
    domain_km_x = cols * resolution_m / 1000
    domain_km_y = rows * resolution_m / 1000
    fig.text(0.5, 0.01,
             f"Domain: {domain_km_x:.1f} km × {domain_km_y:.1f} km  |  "
             f"Resolution: {resolution_m:.0f} m  |  "
             f"N={ensemble.n_members} ensemble members",
             ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# Entropy reduction curve (for multi-cycle output)
# ---------------------------------------------------------------------------

def plot_entropy_curve(
    ax: Axes,
    cycles: list[int],
    entropy_by_strategy: dict[str, list[float]],
    title: str = "Total Information Field Entropy by Cycle",
) -> None:
    """Line chart of total w across cycles for each strategy."""
    styles = {"greedy": ("tab:blue", "-"), "qubo": ("tab:orange", "--"),
              "uniform": ("tab:gray", ":"), "fire_front": ("tab:green", "-.")}
    for name, values in entropy_by_strategy.items():
        color, ls = styles.get(name, ("black", "-"))
        ax.plot(cycles, values, label=name.replace("_", " ").title(),
                color=color, linestyle=ls, linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("Cycle", fontsize=9)
    ax.set_ylabel("Total entropy  Σ w(x)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def save_or_show(fig: Figure, path: Optional[str | Path] = None, dpi: int = 150) -> None:
    """Save figure to `path` if given, otherwise display it."""
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {path}")
    else:
        plt.show()
    plt.close(fig)
