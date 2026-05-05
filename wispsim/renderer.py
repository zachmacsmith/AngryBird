"""
Frame renderer: produces the simulation video or PNG frame sequence.

Phase 4b — simulation harness only.

Layout (6-panel + 2 metric strips + header):

  ┌───────────────────────────────────────────────────────────────────┐
  │  SIMULATION TIME  |  CYCLE  |  OBSERVATIONS  |  LIVE UPDATES ▲   │
  ├──────────────┬──────────────┬─────────────────┬───────────────────┤
  │ GROUND TRUTH │ WISP LIVE   │ EST. ARRIVAL    │ INFORMATION FIELD │
  │ True FMC     │ Est. FMC     │ TIME            │ GP uncertainty    │
  │ True wind    │ Est. wind    │ (single-member  │ + drone targets   │
  │ True fire    │ (per-obs)    │  per-obs)       │ (per WISP cycle) │
  ├──────────────────────────┬──────────────────────────────────────┤
  │  GP VARIANCE CONVERGENCE │  PREDICTION ACCURACY (CRPS / RMSE)  │
  │  (RAWS + drone obs)      │  ensemble vs oracle fire arrival     │
  └──────────────────────────┴──────────────────────────────────────┘

Panels 2 and 3 update live: every render frame reflects the latest drone
observations, NOT just the 20-minute WISP cycle boundary.  Panel 4 still
shows the per-cycle info field since it requires the full ensemble.

Video output uses matplotlib FFMpegWriter when ffmpeg is available,
falling back to PNG frame sequences (assembler-agnostic).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from angrybird.types import CycleReport, DronePlan, InformationField, GPPrior
from angrybird.visualization._style import compute_hillshade, STRATEGY_STYLES
from .drone_sim import DroneState, pos_m_to_cell
from .ground_truth import GroundTruth

logger = logging.getLogger(__name__)

_DRONE_COLORS = ["#E91E63", "#9C27B0", "#3F51B5", "#00BCD4", "#8BC34A",
                 "#FF5722", "#795548", "#607D8B", "#F44336", "#2196F3"]

# Fill colour by planning mode — outline is always the per-drone path colour.
_MODE_FILL_COLORS = {
    "NORMAL":    "#4CAF50",  # green  — operational correlation-path planning
    "TRACKING":  "#2196F3",  # blue   — greedy search (tracking toward fire)
    "RETURN":    "#FFEB3B",  # yellow — returning to ground station
    "EMERGENCY": "#F44336",  # red    — emergency direct flight
}


# ---------------------------------------------------------------------------
# Arrival time colour helper
# ---------------------------------------------------------------------------

def _arrival_rgba(
    arrival_times_h: np.ndarray,
    current_h: float,
    horizon_h: float = 6.0,
) -> np.ndarray:
    """
    Build an RGBA image from estimated arrival times.

    Colour scale (time from NOW until ignition):
      Already burned  (≤ 0 h)         →  dark crimson
      Imminent        (0 – horizon_h)  →  RdYlGn_r  (red → yellow → green)
      Beyond horizon / NaN             →  light grey

    Returns float32 RGBA array (rows, cols, 4).
    """
    time_until = arrival_times_h - current_h          # hours from now; < 0 = burned
    rows, cols = arrival_times_h.shape

    cmap = plt.cm.YlOrRd_r
    # Use a negative vmin so that 0.0 (about to burn) starts slightly into the orange
    # rather than at the absolute dark red of the colormap, providing better contrast
    # with the already-burned red.
    norm = mcolors.Normalize(vmin=-0.2 * horizon_h, vmax=horizon_h)
    rgba = cmap(norm(np.clip(time_until, 0.0, horizon_h))).astype(np.float32)
    rgba[..., 3] = 0.72

    # Cells with no predicted ignition within the horizon
    beyond_mask = (time_until >= horizon_h) | np.isnan(arrival_times_h)
    # A pleasant semi-transparent green for safe areas
    rgba[beyond_mask] = [0.30, 0.70, 0.30, 0.45]

    # Already-burned cells (dark red)
    burned_mask = (~beyond_mask) & (time_until <= 0.0)
    rgba[burned_mask] = [0.60, 0.0, 0.0, 0.85]

    return rgba


# ---------------------------------------------------------------------------
# Map panel
# ---------------------------------------------------------------------------

class MapPanel:
    """
    One terrain-base map panel with swappable dynamic layer overlays.

    The hillshade base is rendered once; only dynamic artists are replaced
    each frame (avoids re-drawing the static background).
    """

    def __init__(self, ax, terrain, title: str) -> None:
        self.ax      = ax
        self.terrain = terrain
        self.title   = title
        self._dyn: list = []   # dynamic artists to clear each frame

        hs = compute_hillshade(terrain.elevation, terrain.resolution_m)
        ax.imshow(hs, origin="upper", cmap="gray", vmin=0, vmax=1, alpha=0.5,
                  interpolation="bilinear")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("East →", fontsize=7)
        ax.set_ylabel("↑ North", fontsize=7)
        ax.tick_params(labelsize=6)

    def _clear(self) -> None:
        for a in self._dyn:
            try:
                # ContourSet objects (from ax.contour/contourf) need their
                # internal collections removed individually; ContourSet.remove()
                # is unreliable across matplotlib versions.
                if hasattr(a, "collections"):
                    for coll in a.collections:
                        try:
                            coll.remove()
                        except Exception:
                            pass
                else:
                    a.remove()
            except Exception:
                pass
        self._dyn.clear()

    def update(
        self,
        fmc: Optional[np.ndarray] = None,
        wind_speed: Optional[np.ndarray] = None,
        wind_direction: Optional[np.ndarray] = None,
        fire_arrival: Optional[np.ndarray] = None,
        current_time: Optional[float] = None,
        burn_probability: Optional[np.ndarray] = None,
        uncertainty: Optional[np.ndarray] = None,
        arrival_times_h: Optional[np.ndarray] = None,
        horizon_h: float = 6.0,
        drone_targets: Optional[list[tuple[int, int]]] = None,
        drone_plans: Optional[list] = None,
        drones: Optional[list[DroneState]] = None,
        resolution_m: float = 50.0,
        shape: Optional[tuple[int, int]] = None,
        raws_locations: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        self._clear()
        ax = self.ax

        # ── Scalar field overlays ─────────────────────────────────────────

        if fmc is not None:
            im = ax.imshow(fmc, origin="upper", cmap="YlOrBr_r",
                           alpha=0.65, vmin=0.02, vmax=0.25,
                           interpolation="nearest")
            self._dyn.append(im)

        if burn_probability is not None:
            im = ax.imshow(burn_probability, origin="upper", cmap="YlOrRd",
                           alpha=0.55, vmin=0, vmax=1, interpolation="nearest")
            self._dyn.append(im)

        if uncertainty is not None:
            vmax = float(np.nanpercentile(uncertainty, 95)) or 1.0
            im = ax.imshow(uncertainty, origin="upper", cmap="inferno",
                           alpha=0.70, vmin=0, vmax=vmax,
                           interpolation="nearest")
            self._dyn.append(im)

        # ── Estimated arrival time map ────────────────────────────────────
        # Shown as a continuous colour field; burned contour drawn separately.

        if arrival_times_h is not None and current_time is not None:
            current_h = current_time / 3600.0
            rgba = _arrival_rgba(arrival_times_h, current_h, horizon_h)
            im = ax.imshow(rgba, origin="upper", interpolation="nearest")
            self._dyn.append(im)

            # Thin contour marking the current predicted fire perimeter (t = now)
            near_mask = (arrival_times_h - current_h <= 0.0).astype(float)
            if near_mask.any():
                cl = ax.contour(near_mask, levels=[0.5],
                                colors=["#CC3300"], linewidths=1.5)
                self._dyn.append(cl)

        # ── True fire contour (ground-truth panels) ───────────────────────
        # Fill uses imshow RGBA (reliably removable); perimeter uses contour.

        if fire_arrival is not None and current_time is not None:
            burned = fire_arrival <= current_time
            if burned.any():
                rgba_fire = np.zeros((*fire_arrival.shape, 4), dtype=np.float32)
                rgba_fire[burned] = [0.80, 0.20, 0.0, 0.40]
                im_f = ax.imshow(rgba_fire, origin="upper", interpolation="nearest")
                self._dyn.append(im_f)
                cl = ax.contour(burned.astype(float), levels=[0.5],
                                colors=["#CC3300"], linewidths=1.5)
                self._dyn.append(cl)

        # ── Wind quivers ─────────────────────────────────────────────────

        if wind_speed is not None and wind_direction is not None:
            step = max(1, wind_speed.shape[0] // 18)
            ys = np.arange(0, wind_speed.shape[0], step) + step // 2
            xs = np.arange(0, wind_speed.shape[1], step) + step // 2
            Y, X = np.meshgrid(ys, xs, indexing="ij")
            ws = wind_speed[::step, ::step]
            wd = wind_direction[::step, ::step]
            U = ws * np.sin(np.radians(wd))
            V = -ws * np.cos(np.radians(wd))
            q = ax.quiver(X, Y, U, V, color="white", alpha=0.70,
                          scale=150, width=0.003)
            self._dyn.append(q)

        # ── RAWS station markers ─────────────────────────────────────────

        if raws_locations:
            for j, (rr, rc) in enumerate(raws_locations):
                sc = ax.scatter([rc], [rr], marker="D", c="yellow",
                                s=55, zorder=8, edgecolors="black",
                                linewidths=0.8)
                tx = ax.text(rc + 1, rr - 1, f"W{j+1}", fontsize=5,
                             color="yellow", fontweight="bold", zorder=9)
                self._dyn += [sc, tx]

        # ── Drone targets ────────────────────────────────────────────────

        if drone_targets:
            for i, (r, c) in enumerate(drone_targets):
                sc = ax.scatter([c], [r], marker="^", c="cyan",
                                s=80, zorder=7, edgecolors="black",
                                linewidths=0.6)
                tx = ax.text(c + 1, r - 1, str(i + 1), fontsize=6,
                             color="cyan", fontweight="bold", zorder=8)
                self._dyn += [sc, tx]

        # ── Planned drone paths ──────────────────────────────────────────
        # Draw the full multi-waypoint route for each drone, color-coded by
        # drone index.  Waypoints are (row, col) grid cells from DronePlan.

        if drone_plans:
            for i, plan in enumerate(drone_plans):
                if not plan.waypoints or len(plan.waypoints) < 2:
                    continue
                color = _DRONE_COLORS[i % len(_DRONE_COLORS)]
                rows_ = [wp[0] for wp in plan.waypoints]
                cols_ = [wp[1] for wp in plan.waypoints]
                # Planned path: solid line, no waypoint markers — the line shows
                # the intended trajectory clearly without cluttering the map.
                ln, = ax.plot(cols_, rows_, "-", color=color,
                              alpha=0.90, linewidth=2.2, zorder=10)
                self._dyn.append(ln)
                # Mark only the final endpoint with a small arrow/dot so the
                # destination is visible without obscuring intermediate domains.
                sc = ax.scatter([cols_[-1]], [rows_[-1]], marker=">",
                                c=color, s=55, zorder=11,
                                edgecolors="white", linewidths=0.8)
                self._dyn.append(sc)

        # ── Drone positions + trails ──────────────────────────────────────
        # Outline: permanent per-drone path colour (matches the route line).
        # Fill:    mode-dependent — TRACKING=blue, NORMAL=green,
        #          RETURN=yellow, EMERGENCY=red.

        if drones and shape:
            plan_mode: dict[int, str] = {
                p.drone_id: p.drone_mode for p in (drone_plans or [])
            }
            for i, drone in enumerate(drones):
                cell = pos_m_to_cell(drone.position, resolution_m, shape)
                outline = _DRONE_COLORS[i % len(_DRONE_COLORS)]
                try:
                    drone_idx = int(drone.drone_id.split("_")[-1])
                except (ValueError, AttributeError):
                    drone_idx = i
                mode = plan_mode.get(drone_idx, "NORMAL")
                fill = _MODE_FILL_COLORS.get(mode, "#9E9E9E")
                sc = ax.scatter([cell[1]], [cell[0]], marker="o",
                                c=fill, s=70, zorder=12,
                                edgecolors=outline, linewidths=2.0)
                self._dyn.append(sc)

                # Historical trail suppressed — planned path lines are sufficient.


# ---------------------------------------------------------------------------
# Arrival time colour bar (drawn once, replaced each frame label only)
# ---------------------------------------------------------------------------

def _draw_arrival_colorbar(ax, horizon_h: float = 6.0) -> None:
    """
    Draw a compact colorbar legend INSIDE the arrival time panel axes.

    Uses inset_axes so the colorbar floats over the map and never calls
    subplots_adjust — neighbouring panels are completely unaffected.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cax = inset_axes(
        ax,
        width="62%", height="5%",
        loc="lower center",
        bbox_to_anchor=(0.0, 0.03, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cmap = plt.cm.YlOrRd_r
    norm = mcolors.Normalize(vmin=-0.2 * horizon_h, vmax=horizon_h)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Hours until ignition", fontsize=5)
    cbar.ax.tick_params(labelsize=4)
    ax.annotate(
        "▪ burned (red)   ▪ no threat (green)",
        xy=(0.99, 0.10), xycoords="axes fraction",
        fontsize=4.5, va="bottom", ha="right", color="#333",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.65, ec="none"),
    )


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def _unique_frame_dir(requested: Path) -> Path:
    """
    Return a directory that contains no existing frame_*.png files.

    Strips any trailing ``_<N>`` suffix from *requested* to get a canonical
    base, then walks the sequence  base → base_1 → base_2 → …  and returns
    the first entry that is either absent or contains no frame files.

    Examples
    --------
    out/sim_hilly          (has frames) → out/sim_hilly_1
    out/sim_hilly_1        (has frames) → out/sim_hilly_2   (strips _1 first)
    out/sim_hilly_2        (empty)      → out/sim_hilly_2   (returned as-is)
    """
    # Strip a trailing _<integer> to get the canonical root name.
    m = re.match(r"^(.+?)_(\d+)$", requested.name)
    root = requested.parent / (m.group(1) if m else requested.name)

    # Try the root (no number), then root_1, root_2, ...
    for candidate in [root] + [root.parent / f"{root.name}_{n}"
                                for n in range(1, 10_000)]:
        if not candidate.exists() or not any(candidate.glob("frame_*.png")):
            return candidate

    return root  # unreachable in practice


# ---------------------------------------------------------------------------
# Frame renderer
# ---------------------------------------------------------------------------

class FrameRenderer:
    """
    Renders simulation frames to a PNG sequence or MP4 video.

    Four map panels (top row):
      1. Ground Truth   — true FMC, true wind, true fire boundary
      2. WISP Estimate — GP-posterior FMC + wind, updated LIVE per observation
      3. Arrival Time   — single-member estimated time-to-fire, updated LIVE
      4. Info Field     — GP uncertainty × sensitivity, updated per WISP cycle

    Panels 2 & 3 update between WISP cycles because they only need
    a GP predict() call + one fire member, not the full 30-member ensemble.

    Args:
        terrain:        TerrainData (for hillshade base)
        out_dir:        directory for PNG frames (and optionally the MP4)
        figsize:        figure size inches (wider default for 4-panel layout)
        frame_interval: render one frame every N simulation steps
        fps:            frames per second for the output video
        make_video:     try to assemble PNG frames into MP4 via FFMpeg
        horizon_h:      fire spread horizon in hours (for arrival time colorbar)
    """

    def __init__(
        self,
        terrain,
        out_dir: str | Path = "out/simulation",
        figsize: tuple[float, float] = (24, 12),
        frame_interval: int = 6,
        fps: int = 10,
        make_video: bool = True,
        horizon_h: float = 6.0,
        raws_locations: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        self.terrain        = terrain
        self.out_dir        = _unique_frame_dir(Path(out_dir))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.frame_interval = frame_interval
        self.fps            = fps
        self.make_video     = make_video
        self.horizon_h      = horizon_h
        self._frame_count   = 0
        self._step_count    = 0
        self._cbs_drawn: set[str] = set()   # track which panels have colorbars
        self._raws_locations = raws_locations or []
        # Set by SimulationRunner after RAWS seeding: used to draw the RAWS-only
        # GP-variance-reduction baseline alongside the full-model curve.
        self._raws_only_gp_var_sum: float = 0.0
        self._initial_gp_var_sum: float = 0.0

        # Continuous GP variance trace: sampled every render frame from live_gp_prior.
        # Each entry is (time_min, gp_var_sum).  Gives a smooth curve that updates
        # as observations arrive through the network, not just at cycle boundaries.
        self._live_gp_trace: list[tuple[float, float]] = []

        # Build figure — 3x2 layout
        # Row 1: Truth Arrival, Truth Env
        # Row 2: Est Arrival,   Est Env
        # Row 3: Info Field,    Accuracy
        self.fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            3, 2,
            figure=self.fig,
            height_ratios=[3, 3, 2.5],
            hspace=0.35, wspace=0.18,
        )
        ax_truth_arr = self.fig.add_subplot(gs[0, 0])
        ax_truth_env = self.fig.add_subplot(gs[0, 1])
        ax_est_arr   = self.fig.add_subplot(gs[1, 0])
        ax_est_env   = self.fig.add_subplot(gs[1, 1])
        ax_info      = self.fig.add_subplot(gs[2, 0])
        self.ax_accuracy = self.fig.add_subplot(gs[2, 1])

        self.panels = {
            "truth_arr": MapPanel(ax_truth_arr, terrain, "Ground Truth: Arrival Time"),
            "truth_env": MapPanel(ax_truth_env, terrain, "Ground Truth: FMC + Wind"),
            "est_arr":   MapPanel(ax_est_arr,   terrain, "Estimated Arrival Time  ▲live"),
            "est_env":   MapPanel(ax_est_env,   terrain, "Estimated FMC + Wind  ▲live"),
            "info":      MapPanel(ax_info,      terrain, "Information Field + Drone Targets"),
        }

    def render_frame(
        self,
        step: int,
        time_s: float,
        ground_truth: GroundTruth,
        drones: list[DroneState],
        gp_prior: Optional[GPPrior] = None,
        burn_probability: Optional[np.ndarray] = None,
        info_field: Optional[InformationField] = None,
        mission_targets: Optional[list[tuple[int, int]]] = None,
        drone_plans: Optional[list[DronePlan]] = None,
        cycle_reports: Optional[list[CycleReport]] = None,
        live_gp_prior: Optional[GPPrior] = None,
        live_arrival_times_h: Optional[np.ndarray] = None,
        truth_arrival_times_h: Optional[np.ndarray] = None,
        accuracy_trace: Optional[list[dict]] = None,
    ) -> None:
        """
        Render one frame if step % frame_interval == 0.

        Args:
            gp_prior:             GP posterior from last WISP cycle (for info field)
            live_gp_prior:        GP posterior updated per-observation (FMC + wind panels)
            live_arrival_times_h: single-member arrival time estimate in hours,
                                  NaN for cells not reached within the horizon;
                                  updated per-observation between WISP cycles
            truth_arrival_times_h: single-member arrival time from oracle fire run using
                                  true FMC/wind, same format as live_arrival_times_h;
                                  falls back to raw CA arrival times if None
        """
        self._step_count = step
        if step % self.frame_interval != 0:
            return

        res = self.terrain.resolution_m
        sh  = self.terrain.shape

        # Choose the most current GP prior available for the estimate panels
        est_prior = live_gp_prior if live_gp_prior is not None else gp_prior

        # ── Header ────────────────────────────────────────────────────────
        hrs  = int(time_s // 3600)
        mins = int((time_s % 3600) // 60)
        secs = int(time_s % 60)
        n_cy = len(cycle_reports) if cycle_reports else 0
        n_ob = sum(r.ensemble_summary.get("n_obs_assimilated", 0)
                   for r in (cycle_reports or []))
        live_label = "▲ live estimate" if live_gp_prior is not None else ""
        self.fig.suptitle(
            f"Time: {hrs:02d}:{mins:02d}:{secs:02d}  |  "
            f"Cycle: {n_cy}  |  Obs assimilated: {n_ob}  {live_label}",
            fontsize=11, fontweight="bold",
        )

        rl = self._raws_locations  # shorthand

        # ── Panel 1: Ground truth Arrival ─────────────────────────────────
        # Prefer the fire-engine projection using true conditions (same method
        # as the estimate panel) so both panels are directly comparable.
        # Falls back to the raw CA arrival times if the oracle run isn't ready yet.
        _truth_arr_h = (
            truth_arrival_times_h
            if truth_arrival_times_h is not None
            else ground_truth.fire.arrival_times.astype(np.float32) / 3600.0
        )
        self.panels["truth_arr"].update(
            arrival_times_h=_truth_arr_h,
            current_time=time_s,
            horizon_h=self.horizon_h,
            drone_plans=drone_plans,
            drones=drones, resolution_m=res, shape=sh,
            raws_locations=rl,
        )
        if "truth_arr" not in self._cbs_drawn:
            _draw_arrival_colorbar(self.panels["truth_arr"].ax, self.horizon_h)
            self._cbs_drawn.add("truth_arr")

        # ── Panel 2: Ground truth FMC + Wind ──────────────────────────────
        self.panels["truth_env"].update(
            fmc=ground_truth.fmc,
            wind_speed=ground_truth.wind_speed,
            wind_direction=ground_truth.wind_direction,
            current_time=time_s,
            drone_plans=drone_plans,
            drones=drones, resolution_m=res, shape=sh,
            raws_locations=rl,
        )

        # ── Panel 3: Estimated Arrival ────────────────────────────────────
        self.panels["est_arr"].update(
            arrival_times_h=live_arrival_times_h,
            current_time=time_s,
            horizon_h=self.horizon_h,
            drone_plans=drone_plans,
            drones=drones, resolution_m=res, shape=sh,
            raws_locations=rl,
        )
        if "est_arr" not in self._cbs_drawn and live_arrival_times_h is not None:
            _draw_arrival_colorbar(self.panels["est_arr"].ax, self.horizon_h)
            self._cbs_drawn.add("est_arr")

        # ── Panel 4: Estimated FMC + Wind ─────────────────────────────────
        self.panels["est_env"].update(
            fmc=est_prior.fmc_mean if est_prior is not None else None,
            wind_speed=(est_prior.wind_speed_mean if est_prior is not None else None),
            wind_direction=(est_prior.wind_dir_mean if est_prior is not None else None),
            drone_plans=drone_plans,
            drones=drones, resolution_m=res, shape=sh,
            raws_locations=rl,
        )

        # ── Panel 5: Information Field ────────────────────────────────────
        self.panels["info"].update(
            uncertainty=info_field.w if info_field is not None else None,
            drone_targets=mission_targets or [],
            drone_plans=drone_plans,
            drones=drones, resolution_m=res, shape=sh,
            raws_locations=rl,
        )

        self._update_accuracy(accuracy_trace or [])

        # ── Save frame ────────────────────────────────────────────────────
        frame_path = self.out_dir / f"frame_{self._frame_count:05d}.png"
        self.fig.savefig(frame_path, dpi=100, bbox_inches="tight")
        self._frame_count += 1
        logger.debug("Frame %d saved (t=%.0fs)", self._frame_count, time_s)


    def _update_accuracy(self, accuracy_trace: list[dict]) -> None:
        """
        Right bottom strip — per-burning-cell CRPS over time.

        Dividing by the number of cells that will burn within the planning
        horizon removes the confound where raw CRPS grows simply because the
        fire is larger.  Ideal value is 0 (perfect arrival-time forecast).
        """
        ax = self.ax_accuracy
        ax.clear()
        ax.set_title(
            "CRPS per Active Burning Cell (per WISP Cycle)",
            fontsize=8, fontweight="bold",
        )
        ax.set_xlabel("Simulation Time (min)", fontsize=7)
        ax.set_ylabel("CRPS / burning cell (min)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

        if not accuracy_trace:
            ax.text(0.5, 0.5, "Accuracy metrics\navailable once\nfire spreads",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=7, color="#888888")
            return

        t_min  = [row["time_min"]              for row in accuracy_trace]
        pccrps = [row["crps_per_cell_minutes"]  for row in accuracy_trace]

        ax.plot(t_min, pccrps, "o-", color="#F44336", linewidth=1.8,
                markersize=4, label="CRPS / burning cell")
        ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="Ideal (0)")
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=6, loc="upper right")

    def finalize(self) -> None:
        """Close figure and optionally assemble video from PNG frames."""
        plt.close(self.fig)
        logger.info("Renderer finalised: %d frames in %s", self._frame_count, self.out_dir)

        if self.make_video and self._frame_count > 0:
            self._assemble_video()

    def _assemble_video(self) -> None:
        """Assemble PNG frames into MP4 via FFMpeg (if available)."""
        try:
            import subprocess
            out_video = self.out_dir.parent / f"{self.out_dir.name}.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(self.fps),
                "-i", str(self.out_dir / "frame_%05d.png"),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(out_video),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("Video saved: %s", out_video)
        except Exception as exc:
            logger.warning("Video assembly failed (%s). PNG frames are in %s.",
                           exc, self.out_dir)
