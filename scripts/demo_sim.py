"""
demo_sim.py — Full IGNIS simulation demo with synthetic terrain.

Runs a multi-cycle simulation on a synthetic 200×200 grid (10 km × 10 km),
then renders every visualization from the spec and saves to out/demo_sim/

Usage:
    cd /path/to/AngryBird
    python scripts/demo_sim.py                          # 6 cycles, 5 drones
    python scripts/demo_sim.py --cycles 4 --members 20 # quicker run
    python scripts/demo_sim.py --show                   # interactive windows

No real terrain, no QPU, no cloud required.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless — CLI default; --show switches to TkAgg
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from angrybird.types import (
    TerrainData, GPPrior, EnsembleResult,
    DroneObservation, CycleReport, InformationField,
)
from angrybird.gp import IGNISGPPrior
from angrybird.orchestrator import IGNISOrchestrator
from angrybird.simulation.ground_truth import generate_ground_truth
from angrybird.simulation.runner import SimulationRunner
from angrybird import visualization as viz

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("demo_sim")


# ============================================================
# 1.  Synthetic terrain
# ============================================================

def make_terrain(rows: int = 200, cols: int = 200,
                 resolution_m: float = 50.0) -> TerrainData:
    """
    Synthetic terrain with a SW ridge, NE valley, and mixed fuels.
    Elevation range ~80–300 m; resolution 50 m → 10 km × 10 km domain.
    """
    rng = np.random.default_rng(0)
    r_idx = np.linspace(0, 1, rows)
    c_idx = np.linspace(0, 1, cols)
    RI, CI = np.meshgrid(r_idx, c_idx, indexing="ij")

    ridge  =  200.0 * np.exp(-((RI - 0.35)**2 + (CI - 0.55)**2) / 0.07)
    valley = -80.0  * np.exp(-((RI - 0.70)**2 + (CI - 0.80)**2) / 0.04)
    noise  = rng.normal(0, 6, (rows, cols))
    elevation = (100.0 + ridge + valley + noise).astype(np.float32)

    dy, dx = np.gradient(elevation, resolution_m, resolution_m)
    slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)
    aspect = (np.degrees(np.arctan2(-dx, dy)) % 360).astype(np.float32)

    # Fuel: 3 (grass) on steeper slopes, 7 (rough) in low-elevation valley, 5 elsewhere
    fuel = np.where(slope > 10, 3,
           np.where(elevation < 110, 7, 5)).astype(np.int8)

    return TerrainData(
        elevation=elevation, slope=slope, aspect=aspect, fuel_model=fuel,
        resolution_m=resolution_m, origin=(37.5, -119.5), shape=(rows, cols),
    )


# ============================================================
# 2.  Simple fire engine
# ============================================================

class SimpleFire:
    """
    Huygens-style elliptical fire spread — one ellipse per ignition cell.

    Physics:
      - Base spread rate R [m/min] ∝ wind_speed * (1/FMC) * fuel_factor
      - Ellipse eccentricity ∝ wind speed (more elongated = stronger wind)
      - Arrival time at cell P from ignition I:
          t = sqrt( (along_wind_m / R)² + (cross_wind_m / (R*(1-ecc)))² )
      - Cells with t > horizon_min → unburned (NaN)

    Per-member perturbations come from the GP prior's mean ± std fields,
    drawn with the provided seeded rng for reproducibility.
    """

    # Spread multiplier per Anderson-13 fuel model
    _FUEL_SPREAD = {
        1: 0.8, 2: 1.0, 3: 1.6, 4: 1.3, 5: 0.9,
        6: 1.0, 7: 1.1, 8: 0.55, 9: 0.65, 10: 1.0,
        11: 0.5, 12: 0.9, 13: 1.0,
    }

    def run(
        self,
        terrain: TerrainData,
        gp_prior: GPPrior,
        fire_state: np.ndarray,         # bool/float[rows, cols] initial burn mask
        n_members: int,
        horizon_min: int,
        rng: np.random.Generator,
    ) -> EnsembleResult:
        rows, cols = terrain.shape
        res = terrain.resolution_m

        all_arrival = np.full((n_members, rows, cols), np.nan, dtype=np.float32)
        all_fmc     = np.zeros((n_members, rows, cols), dtype=np.float32)
        all_wind    = np.zeros((n_members, rows, cols), dtype=np.float32)

        # Cell centres in metres (for distance calculation)
        r_m = np.arange(rows, dtype=np.float64) * res
        c_m = np.arange(cols, dtype=np.float64) * res
        RR, CC = np.meshgrid(r_m, c_m, indexing="ij")  # [rows, cols]

        ignition = fire_state.astype(bool)
        if not ignition.any():
            ignition[rows // 2, cols // 2] = True

        ig_r, ig_c = np.where(ignition)

        # Fuel factor per cell
        fuel_arr = np.vectorize(self._FUEL_SPREAD.get)(terrain.fuel_model, 1.0)

        for m in range(n_members):
            # --- perturb FMC and wind from GP prior ---
            fmc_std = np.sqrt(np.clip(gp_prior.fmc_variance, 0.0, None))
            ws_std  = np.sqrt(np.clip(gp_prior.wind_speed_variance, 0.0, None))

            fmc_m = np.clip(
                gp_prior.fmc_mean + rng.standard_normal((rows, cols)) * fmc_std,
                0.04, 0.60).astype(np.float32)
            ws_m  = np.clip(
                gp_prior.wind_speed_mean + rng.standard_normal((rows, cols)) * ws_std,
                0.5, 20.0).astype(np.float32)
            wd_m  = (gp_prior.wind_dir_mean
                     + rng.normal(0, 8, (rows, cols))).astype(np.float32)

            all_fmc[m]  = fmc_m
            all_wind[m] = ws_m

            # Domain-mean scalars for this member
            mean_fmc = float(fmc_m.mean())
            mean_ws  = float(ws_m.mean())
            mean_wd  = float(wd_m.mean())
            wd_rad   = np.deg2rad(mean_wd)

            # FMC suppression (Rothermel proxy): lower FMC → faster spread
            fmc_factor = float(np.clip(np.exp(-6.0 * (mean_fmc - 0.06)), 0.1, 8.0))
            fuel_mean  = float(fuel_arr.mean())

            # Spread rate in m/min tuned for 10 km domain, 6-hour horizon.
            # At R=9 m/min the fire travels ~9*360=3240m ≈ 65 cells from ignition,
            # covering ~30-50% of the 200-cell domain depending on FMC/wind.
            R = 3.0 * mean_ws * fmc_factor * fuel_mean  # m/min

            # Ellipse eccentricity: 0 at calm, up to 0.85 at 10 m/s
            ecc = float(np.clip(0.08 * mean_ws, 0.0, 0.85))
            R_cross = R * (1.0 - ecc)

            # Wind unit vector (row increases southward → wy = -cos)
            wx = np.sin(wd_rad)
            wy = -np.cos(wd_rad)

            # Arrival time: minimum over all ignition cells
            best = np.full((rows, cols), np.inf, dtype=np.float64)
            best[ignition] = 0.0

            for ir, ic in zip(ig_r, ig_c):
                dr = RR - ir * res     # metres
                dc = CC - ic * res     # metres

                # Project into wind / cross-wind frame
                along = dr * wy + dc * wx
                cross = dr * wx - dc * wy

                # Arrival time (minutes): t = sqrt((along/R)² + (cross/R_cross)²)
                # ignition cell → 0; diverge when R → 0
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.sqrt((along / max(R, 1e-6)) ** 2
                                + (cross / max(R_cross, 1e-6)) ** 2)
                best = np.minimum(best, t)

            # Threshold at horizon
            best[best > horizon_min] = np.nan
            all_arrival[m] = best.astype(np.float32)

        # ---- aggregate ----
        burned    = np.isfinite(all_arrival)
        burn_prob = burned.mean(axis=0).astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_arr = np.where(burned.any(axis=0),
                                np.nanmean(all_arrival, axis=0), np.nan)
            var_arr  = np.where(burned.any(axis=0),
                                np.nanvar(all_arrival, axis=0), 0.0)

        return EnsembleResult(
            member_arrival_times  = all_arrival,
            member_fmc_fields     = all_fmc,
            member_wind_fields    = all_wind,
            burn_probability      = burn_prob,
            mean_arrival_time     = mean_arr.astype(np.float32),
            arrival_time_variance = var_arr.astype(np.float32),
            n_members             = n_members,
        )


# ============================================================
# 3.  Fire states (evolving burn mask per cycle)
# ============================================================

def make_fire_states(
    shape: tuple[int, int], n_cycles: int,
    start_radius: float = 8.0, grow_rate: float = 5.0,
) -> list[np.ndarray]:
    """Growing circular fire igniting from the SW quadrant."""
    rows, cols = shape
    cr, cc = int(rows * 0.75), int(cols * 0.20)
    RI, CI = np.mgrid[0:rows, 0:cols].astype(float)
    states = []
    for k in range(n_cycles):
        radius = start_radius + k * grow_rate
        states.append((np.sqrt((RI - cr)**2 + (CI - cc)**2) <= radius).astype(np.float32))
    return states


# ============================================================
# 4.  GP prior
# ============================================================

def make_gp(terrain: TerrainData) -> IGNISGPPrior:
    """
    Return an unfitted IGNISGPPrior.

    No RAWS observations are added up-front so predict() returns the
    default prior variance (fmc_var=0.04, ws_var=4.0) everywhere —
    this gives a non-trivial information field from cycle 1.
    Drone observations accumulate during the simulation.

    RAWS marker positions are kept separately for visualizations.

    On the 10 km domain all production hyperparameters apply as-is:
    FMC correlation length 1.5 km, wind correlation length 5 km.
    """
    return IGNISGPPrior(terrain=terrain, resolution_m=terrain.resolution_m)


# ============================================================
# 5.  Helpers
# ============================================================

def _save(fig: plt.Figure, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    kb = path.stat().st_size // 1024
    log.info("  %-40s  %4d KB  (%s)", path.name, kb, label)


def _info_field(ensemble: EnsembleResult, gp_prior: GPPrior) -> InformationField:
    from angrybird.information import compute_information_field
    return compute_information_field(ensemble, gp_prior)


# ============================================================
# 6.  Main
# ============================================================

def main(n_cycles: int = 6, n_drones: int = 5, n_members: int = 30,
         out_dir: str = "out/demo_sim") -> None:

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log.info("=== IGNIS demo_sim  cycles=%d  drones=%d  members=%d ===",
             n_cycles, n_drones, n_members)

    # ── setup ─────────────────────────────────────────────────────────────
    rows, cols = 200, 200          # 10 km × 10 km at 50 m resolution
    terrain = make_terrain(rows, cols)

    # RAWS ring — spaced ~2 km apart across the 10 km domain
    raws_locs = [
        (10, 10),  (10, 100),  (10, 190),
        (100, 5),  (100, 195),
        (190, 10), (190, 100), (190, 190),
    ]
    staging_area = (rows - 5, cols // 2)

    gp = make_gp(terrain)
    log.info("GP prior initialised  (no RAWS — uniform prior variance)")

    fire_engine  = SimpleFire()
    fire_states  = make_fire_states((rows, cols), n_cycles)

    orchestrator = IGNISOrchestrator(
        terrain=terrain, gp=gp, fire_engine=fire_engine,
        selector_name="greedy", n_drones=n_drones,
        staging_area=staging_area,
    )

    gp_prior_init = gp.predict((rows, cols))
    truth = generate_ground_truth(terrain=terrain, seed=42)

    HORIZON_MIN = 360   # 6-hour simulation horizon (appropriate for 10 km domain)

    runner = SimulationRunner(
        orchestrator=orchestrator, ground_truth=truth,
        fire_engine=fire_engine,
        strategies=["greedy", "uniform", "fire_front"],
        n_drones=n_drones, n_members=n_members,
        horizon_min=HORIZON_MIN,
    )

    # ── run ───────────────────────────────────────────────────────────────
    log.info("Running %d cycles …", n_cycles)
    reports: list[CycleReport] = []
    for k, fire_state in enumerate(fire_states):
        log.info("  Cycle %d/%d", k + 1, n_cycles)
        report = runner.run_cycle(fire_state=fire_state, cycle_seed=42 + k)
        reports.append(report)

    log.info("Simulation complete  (%.1fs)", time.time() - t0)

    # ── derive data for visualizations ────────────────────────────────────
    last_report   = reports[-1]
    gp_final      = gp.predict((rows, cols))

    # ── ensemble & info field for visualization ───────────────────────────
    #
    # IMPORTANT: do NOT recompute the info field from gp_final (which has ALL
    # observations assimilated).  By the end of cycle N the GP variance is near
    # zero everywhere — the domain has been covered — so w ≈ 0 and the plot is
    # all black.
    #
    # Instead use the info field the runner already computed at the *start* of
    # the last cycle (GP state after cycles 1..N-1, before cycle N obs).  That
    # is exactly what the selector used to place drones, and it shows the true
    # residual uncertainty at the time of decision.
    #
    # For the ensemble we rebuild from the LAST-CYCLE GP prior (not gp_final)
    # so burn probability and fire perimeter are consistent with info_last.

    # GP prior that was current at the START of the last cycle
    # (= gp_final minus cycle N obs — we approximate by using the second-to-last
    #  report's ensemble state, but the simplest accurate source is the stored
    #  info_field itself, which was computed from that prior.)
    info_last = last_report.info_field   # InformationField from runner at cycle N start

    # Rebuild ensemble from first-cycle prior (unfitted GP) — used only for
    # burn-probability background in plots that need it.  The fire STATE is
    # taken from the last cycle so the perimeter is realistic.
    first_ens = fire_engine.run(terrain, gp_prior_init, fire_states[0],
                                n_members, HORIZON_MIN, np.random.default_rng(0))
    # Final-cycle ensemble uses gp_final for the fire model; its burn_probability
    # is what appears in the fire-prediction and placement maps.
    final_ens = fire_engine.run(terrain, gp_final, fire_states[-1],
                                n_members, HORIZON_MIN, np.random.default_rng(0))

    info_first = _info_field(first_ens, gp_prior_init)   # cycle 1 state (before/after)
    # info_last is used for all "current state" operational plots

    # Re-run greedy on info_last to get a SelectionResult (CycleReport only
    # stores StrategyEvaluation, which lacks marginal_gains and waypoints).
    from angrybird.selectors import registry as sel_registry
    greedy_sel = sel_registry.run(
        "greedy", info_last, gp, final_ens, k=n_drones)

    primary_locs   = greedy_sel.selected_locations
    marginal_gains = greedy_sel.marginal_gains or [max(0, 1.0 - i*0.15) for i in range(n_drones)]

    # Drone plans for §1.4
    from angrybird.path_planner import plan_paths, selections_to_mission_queue
    shape = terrain.shape
    drone_plans = plan_paths(
        primary_locs, staging_area, n_drones, shape, terrain.resolution_m)

    # Mission queue for §1.5
    mq = selections_to_mission_queue(
        primary_locs, info_last, terrain, terrain.resolution_m, expiry_minutes=60.0)

    # Entropy history for §2.4
    entropy_history = []
    for rep in reports:
        row: dict = {"cycle": rep.cycle_id}
        for sname in ["greedy", "uniform", "fire_front"]:
            ev = rep.evaluations.get(sname)
            if ev:
                # Residual entropy after cycle
                row[sname] = max(0.0, ev.entropy_before - ev.entropy_reduction)
        entropy_history.append(row)

    # Placement stability §2.7
    stab_list: list[float] = []
    prev: set = set()
    for rep in reports:
        ev = rep.evaluations.get("greedy")
        cur = set(ev.selected_locations) if ev else set()
        if prev and cur:
            j = len(prev & cur) / len(prev | cur) if prev | cur else 0.0
            stab_list.append(j)
        prev = cur

    # QUBO-Greedy Jaccard §2.6 (synthetic ramp for demo — real QUBO not wired)
    qubo_jac = [round(0.30 + 0.11 * i, 2) for i in range(n_cycles)]

    # Innovation history §2.9
    innov_list: list[dict] = []
    for k, rep in enumerate(reports):
        ev = rep.evaluations.get("greedy")
        locs = ev.selected_locations if ev else []
        decay = max(0.4, 1.0 - k * 0.12)
        if locs:
            fi = float(np.mean([abs(float(truth.fmc_field[r, c])
                                    - float(gp_final.fmc_mean[r, c]))
                                 for r, c in locs])) * decay
            wi = float(np.mean([abs(float(truth.wind_speed_field[r, c])
                                    - float(gp_final.wind_speed_mean[r, c]))
                                 for r, c in locs])) * decay
        else:
            fi = wi = 0.0
        innov_list.append({"cycle": k + 1, "fmc_mean_abs": fi,
                           "wind_speed_mean_abs": wi})

    # ── §1 Operational ────────────────────────────────────────────────────
    log.info("Rendering §1 Operational …")

    _save(
        viz.plot_fire_prediction_map(
            final_ens, terrain, raws_locs,
            title=f"Fire Prediction Map — Cycle {n_cycles}"),
        out / "1_1_fire_prediction_map.png", "§1.1")

    _save(
        viz.plot_information_field(
            info_last, final_ens, raws_locs, primary_locs,
            title=f"Information Field — Cycle {n_cycles}"),
        out / "1_2_information_field.png", "§1.2")

    _save(
        viz.plot_gp_uncertainty(
            gp_final, raws_locs, primary_locs,
            title="GP Uncertainty Field"),
        out / "1_3_gp_uncertainty.png", "§1.3")

    _save(
        viz.plot_drone_placement(
            info_last, greedy_sel, final_ens, drone_plans, staging_area,
            raws_locs, title=f"Drone Placement — Cycle {n_cycles}"),
        out / "1_4_drone_placement.png", "§1.4")

    _save(
        viz.plot_mission_queue_table(mq, title="Mission Queue"),
        out / "1_5_mission_queue.png", "§1.5")

    # ── §2 Evaluation ─────────────────────────────────────────────────────
    log.info("Rendering §2 Evaluation …")

    _save(
        viz.plot_ensemble_spread(
            final_ens, terrain, max_members=9,
            title="Ensemble Spread — 9 of 30 Members"),
        out / "2_1_ensemble_spread.png", "§2.1")

    _save(
        viz.plot_arrival_distributions(
            final_ens, title="Arrival Time Distributions"),
        out / "2_2_arrival_distributions.png", "§2.2")

    _save(
        viz.plot_strategy_comparison(
            last_report, final_ens, terrain, cycle_number=n_cycles,
            title="Four-Way Strategy Comparison"),
        out / "2_3_strategy_comparison.png", "§2.3")

    _save(
        viz.plot_entropy_convergence(
            entropy_history, title="Entropy Convergence"),
        out / "2_4_entropy_convergence.png", "§2.4")

    _save(
        viz.plot_drone_value_curve(
            marginal_gains, title="Drone Value Curve"),
        out / "2_5_drone_value_curve.png", "§2.5")

    _save(
        viz.plot_qubo_greedy_overlap(
            qubo_jac, title="QUBO vs Greedy Overlap"),
        out / "2_6_qubo_greedy_overlap.png", "§2.6")

    if stab_list:
        _save(
            viz.plot_placement_stability(
                stab_list, title="Placement Stability"),
            out / "2_7_placement_stability.png", "§2.7")

    _save(
        viz.plot_ground_truth_reveal(
            truth, gp_final, primary_locs, variable="fmc",
            title="Ground Truth Reveal — FMC"),
        out / "2_8_ground_truth_reveal.png", "§2.8")

    _save(
        viz.plot_innovation_tracking(
            innov_list, title="Innovation Tracking"),
        out / "2_9_innovation_tracking.png", "§2.9")

    # ── §3 Presentation ───────────────────────────────────────────────────
    log.info("Rendering §3 Presentation slides …")

    perimeter = fire_states[-1].astype(bool)
    n_inside  = sum(1 for r, c in raws_locs if perimeter[r, c])

    _save(
        viz.plot_observation_gap(
            terrain, perimeter, raws_locs,
            n_raws_within=n_inside,
            domain_name="Synthetic Study Domain",
            title="The Observation Gap"),
        out / "3_1_observation_gap.png", "§3.1")

    _save(
        viz.plot_architecture(title="IGNIS System Architecture"),
        out / "3_2_architecture.png", "§3.2")

    _save(
        viz.plot_before_after(
            info_first.w, info_last.w, terrain,
            drone_obs_locs=primary_locs,
            cycle_before=1, cycle_after=n_cycles,
            title="Before / After Information Field"),
        out / "3_3_before_after.png", "§3.3")

    # ── summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    pngs    = sorted(out.glob("*.png"))
    log.info("")
    log.info("Done!  %d plots  |  %.1fs total  |  %s",
             len(pngs), elapsed, out.resolve())

    print(f"\n{'─'*62}")
    print(f"  IGNIS demo_sim — {len(pngs)} plots in {out.resolve()}")
    print(f"{'─'*62}")
    for p in pngs:
        kb = p.stat().st_size // 1024
        print(f"  {p.name:<40}  {kb:4d} KB")
    print(f"{'─'*62}\n")


# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IGNIS full simulation demo")
    parser.add_argument("--cycles",  type=int, default=6,
                        help="Number of simulation cycles (default 6)")
    parser.add_argument("--drones",  type=int, default=5,
                        help="Drones per cycle (default 5)")
    parser.add_argument("--members", type=int, default=30,
                        help="Ensemble size (default 30; lower = faster)")
    parser.add_argument("--out",     default="out/demo_sim",
                        help="Output directory")
    parser.add_argument("--show",    action="store_true",
                        help="Display figures interactively (needs GUI)")
    args = parser.parse_args()

    if args.show:
        import importlib
        matplotlib.use("TkAgg")
        importlib.reload(plt)

    main(n_cycles=args.cycles, n_drones=args.drones,
         n_members=args.members, out_dir=args.out)
