"""
Core system visualizations (§1 of Vizualisations.md).

These answer operational questions and would exist in a deployed system.

1.1  plot_fire_prediction_map     — Where is the fire going?
1.2  plot_information_field       — Where should drones go and why?
1.3  plot_gp_uncertainty          — Where do we lack data?
1.4  plot_drone_placement         — Where are drones going?
1.5  plot_mission_queue_table     — What should the UTM act on?
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..types import (
    DronePlan, EnsembleResult, GPPrior, InformationField,
    MissionQueue, SelectionResult, TerrainData,
)
from ..fire_state import FireStateEstimator
from ..observations import FireDetectionObservation
from ._style import (
    DRONE_COLORS, STRATEGY_STYLES, VIZ_CONFIG,
    _add_legend, _domain_footnote, _imshow,
    _mark_drone_targets, _mark_stations,
    compute_hillshade, save_or_show,
)


# ---------------------------------------------------------------------------
# 1.1  Fire Prediction Map
# ---------------------------------------------------------------------------

def plot_fire_prediction_map(
    ensemble: EnsembleResult,
    terrain: Optional[TerrainData] = None,
    station_locs: Optional[list[tuple[int, int]]] = None,
    title: str = "Fire Prediction Map",
    figsize: tuple[int, int] = (10, 8),
) -> Figure:
    """
    §1.1 — 'Where is the fire going?'

    Terrain hillshade base (if terrain provided) with overlaid:
      • Burn-probability gradient (YlOrRd, contourf)
      • Mean arrival-time isochrones (1 hr contour lines, white)
      • Confirmed perimeter (burn_prob > 0.9, red outline)
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    rows, cols = ensemble.burn_probability.shape
    extent = [0, cols, rows, 0]

    # Terrain hillshade base layer
    if terrain is not None:
        hs = compute_hillshade(terrain.elevation,
                               resolution_m=terrain.resolution_m)
        ax.imshow(hs, origin="upper", cmap="gray", vmin=0, vmax=1,
                  extent=extent, interpolation="bilinear")
        alpha_bp = 0.70
    else:
        alpha_bp = 1.0

    # Burn probability — filled contours
    bp = ensemble.burn_probability
    bp_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cmap_bp = plt.get_cmap(VIZ_CONFIG["burn_prob_cmap"])
    cf = ax.contourf(bp, levels=bp_levels, cmap=cmap_bp,
                     alpha=alpha_bp)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Burn Probability", fontsize=VIZ_CONFIG["label_size"])
    cb.ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])

    # Arrival-time isochrones (hours)
    arr = np.where(np.isnan(ensemble.mean_arrival_time),
                   np.inf, ensemble.mean_arrival_time)
    iso_levels = [h for h in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
                  if h < np.nanmax(ensemble.mean_arrival_time)]
    if iso_levels:
        cs = ax.contour(arr, levels=iso_levels, colors="white",
                        linewidths=0.9, linestyles="--")
        ax.clabel(cs, fmt="%.0fh", fontsize=6, colors="white",
                  inline=True, inline_spacing=2)

    # Confirmed perimeter (burn_prob > 0.9) — bold red outline
    ax.contour(bp, levels=[0.9], colors=["#CC0000"],
               linewidths=1.8)

    # RAWS stations
    _mark_stations(ax, station_locs or [], label="RAWS", marker="o")
    _add_legend(ax)

    ax.set_title(title, fontsize=VIZ_CONFIG["font_size"] + 1, fontweight="bold")
    ax.set_xlabel("East →", fontsize=VIZ_CONFIG["label_size"])
    ax.set_ylabel("↑ North", fontsize=VIZ_CONFIG["label_size"])
    ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])

    # Legend patches
    from matplotlib.lines import Line2D
    legend_els = [
        mcolors.to_rgba("#CC0000"),  # just for patches below
    ]
    proxy = [
        plt.Line2D([0], [0], color="#CC0000", lw=1.8, label="Perimeter (P>0.9)"),
        plt.Line2D([0], [0], color="white", lw=0.9, ls="--", label="Arrival isochrone"),
    ]
    if station_locs:
        proxy.append(plt.scatter([], [], marker="o", c="white",
                                 edgecolors="black", s=50, label="RAWS"))
    ax.legend(handles=proxy, fontsize=6, loc="upper right",
              framealpha=0.75, labelcolor="black")

    _domain_footnote(fig, rows, cols,
                     terrain.resolution_m if terrain else 50.0,
                     ensemble.n_members)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# 1.2  Information Field Heatmap
# ---------------------------------------------------------------------------

def plot_information_field(
    info_field: InformationField,
    ensemble: Optional[EnsembleResult] = None,
    station_locs: Optional[list[tuple[int, int]]] = None,
    selected_locs: Optional[list[tuple[int, int]]] = None,
    title: str = "Information Field",
    figsize: tuple[int, int] = (14, 6),
) -> Figure:
    """
    §1.2 — 'Where should drones go and why?'

    Left panel  : Total w_i heatmap (inferno)
    Right panel : Attribution RGB composite
                    Red   = FMC-driven uncertainty
                    Blue  = Wind-driven uncertainty
                    Green = 0 (placeholder for crown-fire / bimodal term)
    """
    fig, (ax_total, ax_attr) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    rows, cols = info_field.w.shape

    # --- Left: total w ---
    _imshow(ax_total, info_field.w,
            "Total Information Value  w(x)",
            VIZ_CONFIG["info_field_cmap"], vmin=0,
            colorbar_label="w  (GP var × sensitivity × observability)")
    _mark_stations(ax_total, station_locs or [])
    _mark_drone_targets(ax_total, selected_locs or [],
                        info_w=info_field.w, color="cyan")
    if ensemble is not None:
        # Burn perimeter overlay (thin white contour)
        ax_total.contour(ensemble.burn_probability, levels=[0.5],
                         colors=["white"], linewidths=0.8, origin="upper",
                         linestyles="--")
    _add_legend(ax_total)

    # --- Right: RGB attribution composite ---
    w_fmc  = info_field.w_by_variable.get("fmc",        np.zeros((rows, cols), np.float32))
    w_wind = info_field.w_by_variable.get("wind_speed",  np.zeros((rows, cols), np.float32))
    total  = w_fmc + w_wind + 1e-12

    # Fractions
    fmc_frac  = w_fmc  / total          # → R channel
    wind_frac = w_wind / total          # → B channel

    # Brightness by total w (normalised)
    brightness = np.clip(info_field.w / (info_field.w.max() + 1e-12), 0, 1)

    rgb = np.zeros((rows, cols, 3), dtype=np.float32)
    rgb[:, :, 0] = (fmc_frac  * brightness)   # R = FMC
    rgb[:, :, 2] = (wind_frac * brightness)   # B = wind
    # Gamma-correct for perceptual linearity
    rgb = np.clip(rgb ** 0.5, 0, 1)

    ax_attr.imshow(rgb, origin="upper", interpolation="nearest")
    ax_attr.set_title("Attribution: Red=FMC · Blue=Wind",
                      fontsize=VIZ_CONFIG["font_size"], fontweight="bold")
    ax_attr.set_xlabel("East →", fontsize=VIZ_CONFIG["label_size"])
    ax_attr.set_ylabel("↑ North", fontsize=VIZ_CONFIG["label_size"])
    ax_attr.tick_params(labelsize=VIZ_CONFIG["tick_size"])
    _mark_stations(ax_attr, station_locs or [])
    _mark_drone_targets(ax_attr, selected_locs or [],
                        info_w=info_field.w, color="yellow")
    _add_legend(ax_attr)

    # Legend chips
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=(0.9, 0, 0), label="FMC uncertainty"),
        Patch(facecolor=(0, 0, 0.9), label="Wind uncertainty"),
        Patch(facecolor=(0.7, 0, 0.7), label="Both"),
    ]
    ax_attr.legend(handles=legend_els, fontsize=6, loc="upper right",
                   framealpha=0.75)

    _domain_footnote(fig, rows, cols, 50.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# 1.3  GP Uncertainty Field
# ---------------------------------------------------------------------------

def plot_gp_uncertainty(
    gp_prior: GPPrior,
    raws_locs: Optional[list[tuple[int, int]]] = None,
    drone_obs_locs: Optional[list[tuple[int, int]]] = None,
    title: str = "GP Uncertainty Field",
    figsize: tuple[int, int] = (13, 5),
) -> Figure:
    """
    §1.3 — 'Where do we lack data?'

    Left  : FMC posterior std (sqrt of variance)
    Right : Wind speed posterior std
    Overlaid: RAWS (circles), drone observations (triangles)
    """
    fig, (ax_fmc, ax_ws) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    fmc_std = np.sqrt(np.clip(gp_prior.fmc_variance, 0, None))
    ws_std  = np.sqrt(np.clip(gp_prior.wind_speed_variance, 0, None))

    _imshow(ax_fmc, fmc_std,
            "FMC Uncertainty  σ(FMC)",
            VIZ_CONFIG["uncertainty_cmap"],
            vmin=0, colorbar_label="σ (fraction)")
    _imshow(ax_ws, ws_std,
            "Wind Speed Uncertainty  σ(wind)",
            VIZ_CONFIG["uncertainty_cmap"],
            vmin=0, colorbar_label="σ (m/s)")

    for ax in (ax_fmc, ax_ws):
        _mark_stations(ax, raws_locs or [], label="RAWS",
                       marker="o", color="lime", size=80)
        _mark_stations(ax, drone_obs_locs or [], label="Drone obs",
                       marker=VIZ_CONFIG["drone_marker"],
                       color="cyan", size=60)
        _add_legend(ax)

    rows, cols = gp_prior.fmc_variance.shape
    _domain_footnote(fig, rows, cols, 50.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# 1.4  Drone Placement Map
# ---------------------------------------------------------------------------

def plot_drone_placement(
    info_field: InformationField,
    selection_result: SelectionResult,
    ensemble: Optional[EnsembleResult] = None,
    drone_plans: Optional[list[DronePlan]] = None,
    staging_area: Optional[tuple[int, int]] = None,
    station_locs: Optional[list[tuple[int, int]]] = None,
    title: str = "Drone Placement",
    figsize: tuple[int, int] = (10, 8),
) -> Figure:
    """
    §1.4 — 'Where are drones going and why these locations?'

    Background: burn probability (if ensemble) or info field.
    Targets: ranked markers, size ∝ w_i, annotated with rank number.
    Paths: flight paths per drone, colour-coded by drone ID.
    """
    fig, ax = plt.subplots(figsize=figsize)

    rows, cols = info_field.w.shape

    # Background: burn probability or info field
    if ensemble is not None:
        bg = ensemble.burn_probability
        bg_cmap = VIZ_CONFIG["burn_prob_cmap"]
        bg_label = "Burn Probability"
    else:
        bg = info_field.w
        bg_cmap = VIZ_CONFIG["info_field_cmap"]
        bg_label = "Information Value"

    im = ax.imshow(bg, origin="upper", cmap=bg_cmap,
                   vmin=0, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(bg_label, fontsize=VIZ_CONFIG["label_size"])
    cb.ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])

    # Fire perimeter contour
    if ensemble is not None:
        ax.contour(ensemble.burn_probability, levels=[0.5],
                   colors=["white"], linewidths=1.0, linestyles="--",
                   origin="upper")

    # Drone flight paths
    if drone_plans:
        for plan in drone_plans:
            if not plan.waypoints:
                continue
            color = DRONE_COLORS[plan.drone_id % len(DRONE_COLORS)]
            wps = np.array(plan.waypoints)
            ax.plot(wps[:, 1], wps[:, 0], "-",
                    color=color, linewidth=1.5, alpha=0.80, zorder=5,
                    label=f"Drone {plan.drone_id}")

    # Staging area
    if staging_area is not None:
        ax.scatter([staging_area[1]], [staging_area[0]],
                   marker="s", c="yellow", s=120, zorder=8,
                   edgecolors="black", linewidths=0.8, label="Staging")

    # Ranked targets (size ∝ w_i)
    _mark_drone_targets(ax, selection_result.selected_locations,
                        info_w=info_field.w,
                        color="cyan", annotate_rank=True,
                        label=f"{selection_result.strategy_name.title()} targets")

    # RAWS stations
    _mark_stations(ax, station_locs or [], label="RAWS")

    _add_legend(ax)
    ax.set_title(
        f"{title}  [{selection_result.strategy_name.upper()}]  "
        f"K={len(selection_result.selected_locations)}",
        fontsize=VIZ_CONFIG["font_size"], fontweight="bold",
    )
    ax.set_xlabel("East →", fontsize=VIZ_CONFIG["label_size"])
    ax.set_ylabel("↑ North", fontsize=VIZ_CONFIG["label_size"])
    ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])

    _domain_footnote(fig, rows, cols, 50.0,
                     ensemble.n_members if ensemble else None)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# 1.5  Mission Queue Table
# ---------------------------------------------------------------------------

def plot_mission_queue_table(
    mission_queue: MissionQueue,
    title: str = "Mission Queue",
    figsize: tuple[int, int] = (11, 5),
) -> Figure:
    """
    §1.5 — 'What should the UTM act on?'

    Renders the MissionQueue as a matplotlib table with columns:
    Rank | Lat | Lon | Info Value | Dominant Variable | Expiry (min)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    if not mission_queue.requests:
        ax.text(0.5, 0.5, "No active requests", ha="center", va="center",
                fontsize=12, color="gray")
        return fig

    col_labels = ["Drone", "Waypoints", "First (Lat, Lon)", "Info Value", "Dominant Variable", "Expiry (min)"]
    rows_data = []
    for req in mission_queue.requests:
        first = req.path[0] if req.path else (float("nan"), float("nan"))
        rows_data.append([
            str(req.drone_id),
            str(len(req.path)),
            f"{first[0]:.4f}, {first[1]:.4f}",
            f"{req.information_value:.4f}",
            req.dominant_variable,
            f"{req.expiry_minutes:.0f}",
        ])

    table = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    # Header style
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1976D2")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for i in range(1, len(rows_data) + 1):
        bg = "#E3F2FD" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 1.6  Fire State Estimation
# ---------------------------------------------------------------------------

def plot_fire_state_estimation(
    estimator: FireStateEstimator,
    fire_obs: Optional[list[FireDetectionObservation]] = None,
    terrain: Optional[TerrainData] = None,
    ground_truth_mask: Optional[np.ndarray] = None,
    title: str = "Fire State Estimation",
    figsize: tuple[int, int] = (18, 5),
) -> Figure:
    """
    §1.6 — 'How confident are we in the current fire location?'

    Left panel   : Reconstructed arrival time (isochrones).
    Center panel : Confidence (0.0 to 1.0).
    Right panel  : Arrival time uncertainty (seconds).
    """
    fig, (ax_arr, ax_conf, ax_unc) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    rows, cols = estimator.grid_shape
    extent = [0, cols, rows, 0]
    
    # Pre-compute hillshade if terrain is available
    hs = None
    if terrain is not None:
        hs = compute_hillshade(terrain.elevation, resolution_m=terrain.resolution_m)

    # Helper to plot background and overlay points
    def setup_ax(ax, title_str):
        if hs is not None:
            ax.imshow(hs, origin="upper", cmap="gray", vmin=0, vmax=1,
                      extent=extent, interpolation="bilinear")
        ax.set_title(title_str, fontsize=VIZ_CONFIG["font_size"], fontweight="bold")
        ax.set_xlabel("East →", fontsize=VIZ_CONFIG["label_size"])
        ax.set_ylabel("↑ North", fontsize=VIZ_CONFIG["label_size"])
        ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])

    # --- Left: Arrival Time ---
    setup_ax(ax_arr, "Reconstructed Arrival Time")
    
    arr = np.where(estimator.arrival_time >= estimator.max_arrival * 0.9, np.nan, estimator.arrival_time)
    
    cmap_arr = plt.get_cmap("viridis")
    if not np.isnan(arr).all():
        im_arr = ax_arr.imshow(arr, origin="upper", cmap=cmap_arr, extent=extent, 
                               alpha=0.8 if hs is not None else 1.0)
        cb1 = fig.colorbar(im_arr, ax=ax_arr, fraction=0.046, pad=0.04)
        cb1.set_label("Arrival Time (s)", fontsize=VIZ_CONFIG["label_size"])
        cb1.ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])
        
        # Draw 1-hour (3600s) isochrones
        iso_levels = np.arange(3600, np.nanmax(arr) + 1, 3600)
        if len(iso_levels) > 0:
            cs = ax_arr.contour(arr, levels=iso_levels, colors="white", linewidths=0.8, linestyles="--")
            ax_arr.clabel(cs, fmt="%ds", fontsize=6, colors="white")
    else:
        ax_arr.text(0.5, 0.5, "No Fire", ha="center", va="center", transform=ax_arr.transAxes, color="red")

    # --- Center: Confidence ---
    setup_ax(ax_conf, "Observation Confidence")
    cmap_conf = plt.get_cmap("Blues")
    im_conf = ax_conf.imshow(estimator.confidence, origin="upper", cmap=cmap_conf, vmin=0.0, vmax=1.0, 
                             extent=extent, alpha=0.8 if hs is not None else 1.0)
    cb2 = fig.colorbar(im_conf, ax=ax_conf, fraction=0.046, pad=0.04)
    cb2.set_label("Confidence (0=Model, 1=Observed)", fontsize=VIZ_CONFIG["label_size"])
    cb2.ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])

    # --- Right: Uncertainty ---
    setup_ax(ax_unc, "Arrival Time Uncertainty")
    unc = np.where(estimator.arrival_uncertainty >= estimator.max_arrival * 0.9, np.nan, estimator.arrival_uncertainty)
    cmap_unc = plt.get_cmap("plasma")
    
    if not np.isnan(unc).all():
        im_unc = ax_unc.imshow(unc, origin="upper", cmap=cmap_unc, extent=extent, 
                               alpha=0.8 if hs is not None else 1.0)
        cb3 = fig.colorbar(im_unc, ax=ax_unc, fraction=0.046, pad=0.04)
        cb3.set_label("Uncertainty (s)", fontsize=VIZ_CONFIG["label_size"])
        cb3.ax.tick_params(labelsize=VIZ_CONFIG["tick_size"])
    else:
        ax_unc.text(0.5, 0.5, "Infinite Uncertainty", ha="center", va="center", transform=ax_unc.transAxes, color="red")

    # --- Plot Observations on all axes ---
    if fire_obs:
        fire_r, fire_c = [], []
        nofire_r, nofire_c = [], []
        
        for obs in fire_obs:
            if obs.is_fire:
                fire_r.append(obs.location[0])
                fire_c.append(obs.location[1])
            else:
                nofire_r.append(obs.location[0])
                nofire_c.append(obs.location[1])
                
        for ax in [ax_arr, ax_conf, ax_unc]:
            if fire_r:
                ax.scatter(fire_c, fire_r, marker="^", color="red", edgecolors="black", 
                           s=40, label="Fire Obs" if ax == ax_arr else "")
            if nofire_r:
                ax.scatter(nofire_c, nofire_r, marker="o", color="dodgerblue", edgecolors="black", 
                           s=30, label="No-Fire Obs" if ax == ax_arr else "")
            
            # Ground truth outline
            if ground_truth_mask is not None:
                ax.contour(ground_truth_mask, levels=[0.5], colors=["#00FF00"], 
                           linewidths=1.5, linestyles="-", origin="upper")
                if ax == ax_arr:
                    from matplotlib.lines import Line2D
                    proxy = [Line2D([0], [0], color="#00FF00", lw=1.5, label="Ground Truth")]
                    current_handles, current_labels = ax.get_legend_handles_labels()
                    ax.legend(handles=current_handles + proxy, labels=current_labels + ["Ground Truth"], 
                              fontsize=6, loc="upper right", framealpha=0.75, labelcolor="black")
            elif ax == ax_arr and (fire_r or nofire_r):
                _add_legend(ax)

    _domain_footnote(fig, rows, cols, terrain.resolution_m if terrain else 50.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

