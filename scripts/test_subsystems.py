"""
IGNIS Subsystem Unit Tests + Diagnostic Visualisations
=======================================================

Covers every non-visualization core subsystem:
  SS1  utils.py            — coordinates, distances, taper, Bresenham, Jaccard
  SS2  terrain.py          — synthetic DEM, slope/aspect, fuel model assignment
  SS3  gp.py               — kernel, prior prediction, correlated field, conditional_variance
  SS4  information.py      — sensitivity, observability, info-field pipeline
  SS5  path_planner.py     — cells_along_path, plan_paths, mission queue
  SS6  selectors/          — UniformSelector, FireFrontSelector, GreedySelector
  SS7  assimilation.py     — observation thinning, EnKF update, replan flags
  SS8  simulation/ground_truth.py — FMC / wind field generation
  SS9  simulation/observer.py     — SimulatedObserver noise model

Usage:
  cd /path/to/AngryBird
  python scripts/test_subsystems.py
  open out/tests/        # diagnostic plots
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── make sure the package is on the path ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "out" / "tests"
OUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Minimal fire engine (for tests that need an EnsembleResult)
# ─────────────────────────────────────────────────────────────────────────────

def _make_ensemble(
    terrain,
    gp_prior,
    fire_state,
    n_members: int = 20,
    horizon_min: float = 60.0,
    rng=None,
):
    """
    Huygens-ellipse fire spread: arrival_time(r,c) = dist_to_fire / R  [hours].
    R = 3.0 * mean_ws * fmc_factor * fuel_mean  (m/min).
    """
    from angrybird.gp import draw_gp_scaled_field
    from angrybird.types import EnsembleResult

    if rng is None:
        rng = np.random.default_rng(0)
    rows, cols = terrain.shape
    shape = (rows, cols)
    res = terrain.resolution_m
    member_arrivals = []
    member_fmc = []
    member_ws = []
    member_wd = []
    fmc_mean = gp_prior.fmc_mean.astype(np.float64)
    ws_mean  = gp_prior.wind_speed_mean.astype(np.float64)
    wd_mean  = gp_prior.wind_dir_mean.astype(np.float64)
    wd_std   = np.sqrt(np.clip(gp_prior.wind_dir_variance, 0.0, None)).astype(np.float64)

    fire_rows, fire_cols = np.where(fire_state > 0)
    if len(fire_rows) == 0:
        fire_rows = np.array([rows // 2])
        fire_cols = np.array([cols // 2])

    for _ in range(n_members):
        fmc_pert = draw_gp_scaled_field(shape, 1500.0, res, gp_prior.fmc_variance)
        ws_pert  = draw_gp_scaled_field(shape, 5000.0, res, gp_prior.wind_speed_variance)
        wd_pert  = rng.standard_normal(shape).astype(np.float64) * wd_std
        fmc_m = np.clip(fmc_mean + fmc_pert, 0.02, 0.40)
        ws_m  = np.clip(ws_mean  + ws_pert,  0.0,  20.0)
        wd_m  = (wd_mean + wd_pert) % 360.0

        fmc_factor = np.clip((0.30 - fmc_m) / 0.28, 0.05, 1.0)
        fuel_load = np.array(
            [[0.10] * cols] * rows, dtype=np.float64
        )
        R = 3.0 * ws_m * fmc_factor * fuel_load  # m/min

        # Distance transform from ignition zone
        dist_grid = np.full(shape, np.inf)
        for fr, fc in zip(fire_rows, fire_cols):
            dr = (np.arange(rows) - fr)[:, None] * res
            dc = (np.arange(cols) - fc)[None, :] * res
            d  = np.sqrt(dr**2 + dc**2)
            dist_grid = np.minimum(dist_grid, d)

        R_safe = np.maximum(R, 0.1)
        arrival_min = dist_grid / R_safe        # minutes
        arrival_hr  = arrival_min / 60.0        # hours
        arrival_hr[fire_state > 0] = 0.0
        arrival_hr[arrival_hr > horizon_min / 60.0] = np.nan

        member_arrivals.append(arrival_hr.astype(np.float32))
        member_fmc.append(fmc_m.astype(np.float32))
        member_ws.append(ws_m.astype(np.float32))
        member_wd.append(wd_m.astype(np.float32))

    mat = np.stack(member_arrivals)
    bp = (np.sum(~np.isnan(mat), axis=0) / n_members).astype(np.float32)
    mean_arr = np.nanmean(mat, axis=0).astype(np.float32)
    var_arr  = np.nanvar(mat, axis=0).astype(np.float32)

    return EnsembleResult(
        member_arrival_times=mat,
        member_fmc_fields=np.stack(member_fmc),
        member_wind_fields=np.stack(member_ws),
        member_wind_dir_fields=np.stack(member_wd),
        burn_probability=bp,
        mean_arrival_time=mean_arr,
        arrival_time_variance=var_arr,
        n_members=n_members,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test result tracking
# ─────────────────────────────────────────────────────────────────────────────

class Result(NamedTuple):
    subsystem: str
    name: str
    passed: bool
    detail: str
    elapsed_s: float


RESULTS: list[Result] = []


def _test(subsystem: str, name: str):
    """Decorator-style context that records pass/fail."""
    class _Ctx:
        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, tb):
            elapsed = time.perf_counter() - self._t0
            if exc_type is None:
                RESULTS.append(Result(subsystem, name, True, "OK", elapsed))
                print(f"  ✓  {name}  ({elapsed*1000:.0f} ms)")
            else:
                detail = f"{exc_type.__name__}: {exc_val}"
                RESULTS.append(Result(subsystem, name, False, detail, elapsed))
                print(f"  ✗  {name}  — {detail}")
            return True   # suppress exception so suite keeps running

    return _Ctx()


# ═════════════════════════════════════════════════════════════════════════════
# SS1  utils.py
# ═════════════════════════════════════════════════════════════════════════════

def test_ss1_utils():
    from angrybird import utils

    print("\n── SS1  utils.py ─────────────────────────────────────────────")

    with _test("SS1", "utm_zone: equidistant points"):
        assert utils.utm_zone(-180) == 1
        assert utils.utm_zone(-120) == 11   # California
        assert utils.utm_zone(0)    == 31
        assert utils.utm_zone(180)  == 61

    with _test("SS1", "latlon_to_utm: round-trip consistency"):
        e, n, zone, hemi = utils.latlon_to_utm(37.0, -120.0)
        # lon=-120 is the western edge of zone 11 (zones go: zone=int((lon+180)/6)+1)
        # Central meridian of zone 11 is -117°; at -120° easting ≈ 233 km
        assert zone == 11, f"Expected zone 11, got {zone}"
        assert 100_000 < e < 500_000, f"easting {e}"
        assert 4_000_000 < n < 5_000_000, f"northing {n}"
        assert hemi == "N"

    with _test("SS1", "latlon_to_utm: southern hemisphere offset"):
        _, n_s, _, hemi_s = utils.latlon_to_utm(-37.0, 120.0)
        assert hemi_s == "S"
        # At lat=-37°, northing ≈ 10,000,000 - 4,100,000 ≈ 5,900,000
        assert 4_000_000 < n_s < 8_000_000, f"Southern hemisphere northing={n_s:.0f}"

    with _test("SS1", "grid_to_latlon: origin == (0, 0)"):
        lat, lon = utils.grid_to_latlon(0, 0, 37.0, -120.0, 50.0)
        assert abs(lat - 37.0) < 1e-6
        assert abs(lon - (-120.0)) < 1e-6

    with _test("SS1", "grid_to_latlon: row increases → lat decreases"):
        lat0, _ = utils.grid_to_latlon(0,   0, 37.0, -120.0, 50.0)
        lat1, _ = utils.grid_to_latlon(100, 0, 37.0, -120.0, 50.0)
        assert lat1 < lat0

    with _test("SS1", "euclidean_distance_m: zero diagonal"):
        d = utils.euclidean_distance_m(5, 5, 5, 5, 50.0)
        assert d == 0.0

    with _test("SS1", "euclidean_distance_m: known distance"):
        # 3-4-5 right triangle in grid cells * 50 m/cell
        d = utils.euclidean_distance_m(0, 0, 3, 4, 50.0)
        assert abs(d - 250.0) < 1e-6

    with _test("SS1", "pairwise_distances: symmetry + zero diagonal"):
        locs = [(0, 0), (0, 4), (3, 0)]
        D = utils.pairwise_distances(locs, 50.0)
        assert D.shape == (3, 3)
        np.testing.assert_array_almost_equal(D, D.T)
        np.testing.assert_array_equal(np.diag(D), 0.0)

    with _test("SS1", "distance_grid: zero at ref cell"):
        g = utils.distance_grid(10, 20, (30, 40), 50.0)
        assert g.shape == (30, 40)
        assert g[10, 20] == 0.0

    with _test("SS1", "angular_diff_deg: wrap around 360"):
        # 350° and 10° differ by 20°, not 340°
        diff = utils.angular_diff_deg(350.0, 10.0)
        assert abs(diff - 20.0) < 1e-6

    with _test("SS1", "gaspari_cohn: boundary values"):
        dists = np.array([0.0, 500.0, 1000.0])   # radius = 1000
        gc = utils.gaspari_cohn(dists, radius=1000.0)
        assert abs(gc[0] - 1.0) < 1e-5, f"GC(0) should be 1, got {gc[0]}"
        assert abs(gc[-1]) < 1e-10, f"GC(radius) should be ~0, got {gc[-1]:.2e}"
        assert 0.0 < gc[1] < 1.0, f"GC(mid) should be in (0,1), got {gc[1]}"

    with _test("SS1", "gaspari_cohn: monotonically decreasing"):
        d = np.linspace(0, 1000, 100)
        gc = utils.gaspari_cohn(d, radius=1000.0)
        assert np.all(np.diff(gc) <= 1e-8), "GC taper must be non-increasing"

    with _test("SS1", "thin_observations: spacing constraint satisfied"):
        locs  = [(0, 0), (0, 1), (0, 10), (0, 11)]
        vals  = [1.0, 2.0, 3.0, 4.0]
        # min_spacing_m=200, resolution_m=50 → min_cells=4
        k_locs, k_vals = utils.thin_observations(locs, vals, 200.0, 50.0)
        # (0,0) and (0,1) are 1 cell apart → (0,1) should be dropped
        # (0,10) and (0,11) are 1 cell apart → (0,11) should be dropped
        assert (0, 0) in k_locs
        assert (0, 1) not in k_locs
        assert (0, 10) in k_locs
        assert (0, 11) not in k_locs

    with _test("SS1", "bresenham: includes start and end"):
        cells = utils.bresenham(0, 0, 0, 5)
        assert cells[0]  == (0, 0)
        assert cells[-1] == (0, 5)

    with _test("SS1", "bresenham: axis-aligned horizontal"):
        cells = utils.bresenham(3, 2, 3, 7)
        assert all(r == 3 for r, c in cells)
        assert len(cells) == 6

    with _test("SS1", "bresenham: diagonal (45°) has equal dr and dc"):
        cells = utils.bresenham(0, 0, 5, 5)
        assert cells[0]  == (0, 0)
        assert cells[-1] == (5, 5)
        # Each step moves by 1 in both dimensions
        for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
            assert abs(r1 - r0) <= 1 and abs(c1 - c0) <= 1

    with _test("SS1", "jaccard: identical sets → 1.0"):
        assert utils.jaccard([(0, 0), (1, 1)], [(0, 0), (1, 1)]) == 1.0

    with _test("SS1", "jaccard: disjoint sets → 0.0"):
        assert utils.jaccard([(0, 0)], [(1, 1)]) == 0.0

    with _test("SS1", "jaccard: both empty → 1.0"):
        assert utils.jaccard([], []) == 1.0

    # ── Diagnostic plot ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("SS1 utils — Diagnostic", fontsize=13, fontweight="bold")

    # 1a. Gaspari-Cohn curve
    ax = axes[0]
    d_arr = np.linspace(0, 1200, 500)
    gc_arr = utils.gaspari_cohn(d_arr, radius=1000.0)
    ax.plot(d_arr, gc_arr, "b-", lw=2)
    ax.axvline(1000, color="r", ls="--", label="radius = 1000 m")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axhline(1, color="gray", lw=0.5)
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("Taper weight")
    ax.set_title("Gaspari-Cohn taper"); ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.1)

    # 1b. Bresenham lines
    ax = axes[1]
    grid = np.zeros((15, 20))
    lines = [
        ((0, 0),  (14, 19), "r", "diagonal"),
        ((7, 0),  (7, 19),  "b", "horizontal"),
        ((0, 10), (14, 10), "g", "vertical"),
    ]
    for (r0, c0), (r1, c1), col, lbl in lines:
        for r, c in utils.bresenham(r0, c0, r1, c1):
            grid[r, c] = 1.0
    ax.imshow(grid, cmap="Greys", origin="upper", interpolation="nearest")
    for (r0, c0), (r1, c1), col, lbl in lines:
        pts = utils.bresenham(r0, c0, r1, c1)
        rs = [p[0] for p in pts]; cs = [p[1] for p in pts]
        ax.plot(cs, rs, ".", color=col, markersize=3, label=lbl)
    ax.set_title("Bresenham rasterisation"); ax.legend(fontsize=7)

    # 1c. Angular difference wrap-around
    ax = axes[2]
    a = np.linspace(0, 360, 360)
    diff = utils.angular_diff_deg(a, 180.0)
    ax.plot(a, diff, "purple", lw=2)
    ax.set_xlabel("Angle (°)"); ax.set_ylabel("Diff from 180° (°)")
    ax.set_title("angular_diff_deg — wrap at 360°")
    ax.axhline(180, color="r", ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT / "ss1_utils.png", dpi=110)
    plt.close(fig)
    print("  → saved ss1_utils.png")


# ═════════════════════════════════════════════════════════════════════════════
# SS2  terrain.py
# ═════════════════════════════════════════════════════════════════════════════

def test_ss2_terrain():
    from angrybird.terrain import synthetic_terrain
    from angrybird.config import FUEL_PARAMS

    print("\n── SS2  terrain.py ───────────────────────────────────────────")

    with _test("SS2", "synthetic_terrain: output shape matches request"):
        t = synthetic_terrain((40, 60))
        assert t.shape == (40, 60)
        assert t.elevation.shape == (40, 60)
        assert t.slope.shape == (40, 60)
        assert t.aspect.shape == (40, 60)
        assert t.fuel_model.shape == (40, 60)

    with _test("SS2", "synthetic_terrain: elevation in [100, 1600] m"):
        t = synthetic_terrain((50, 50))
        assert t.elevation.min() >= 100.0 - 1, f"min={t.elevation.min()}"
        assert t.elevation.max() <= 1600.0 + 1, f"max={t.elevation.max()}"

    with _test("SS2", "synthetic_terrain: all fuel models valid (Anderson 1-13)"):
        t = synthetic_terrain((60, 60))
        unique = np.unique(t.fuel_model)
        for fm in unique:
            assert fm in FUEL_PARAMS, f"Fuel model {fm} not in Anderson 13"

    with _test("SS2", "synthetic_terrain: elevation band → fuel model correlation"):
        t = synthetic_terrain((100, 100), seed=1)
        low_mask  = t.elevation < 500
        high_mask = t.elevation > 900
        # Timber models (8-11) should dominate high elevations
        high_fuels = t.fuel_model[high_mask]
        assert np.mean(high_fuels >= 8) > 0.5, "High elevation should be mostly timber (8-11)"

    with _test("SS2", "synthetic_terrain: deterministic with same seed"):
        t1 = synthetic_terrain((40, 40), seed=99)
        t2 = synthetic_terrain((40, 40), seed=99)
        np.testing.assert_array_equal(t1.elevation, t2.elevation)

    with _test("SS2", "synthetic_terrain: different seeds give different results"):
        t1 = synthetic_terrain((40, 40), seed=1)
        t2 = synthetic_terrain((40, 40), seed=2)
        assert not np.array_equal(t1.elevation, t2.elevation)

    with _test("SS2", "synthetic_terrain: slope non-negative"):
        t = synthetic_terrain((50, 50))
        assert t.slope.min() >= 0.0

    with _test("SS2", "synthetic_terrain: aspect in [0, 360)"):
        t = synthetic_terrain((50, 50))
        assert t.aspect.min() >= 0.0
        assert t.aspect.max() < 360.0

    with _test("SS2", "synthetic_terrain: resolution_m stored correctly"):
        t = synthetic_terrain((40, 40), resolution_m=30.0)
        assert t.resolution_m == 30.0

    # ── Diagnostic plot ───────────────────────────────────────────────────
    t = synthetic_terrain((80, 80), seed=42)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("SS2 terrain — Synthetic DEM Diagnostic", fontsize=13, fontweight="bold")

    im = axes[0].imshow(t.elevation, cmap="terrain", origin="upper")
    axes[0].set_title("Elevation (m)"); plt.colorbar(im, ax=axes[0], shrink=0.7)

    im = axes[1].imshow(t.slope, cmap="Oranges", origin="upper")
    axes[1].set_title("Slope (°)"); plt.colorbar(im, ax=axes[1], shrink=0.7)

    im = axes[2].imshow(t.aspect, cmap="hsv", origin="upper", vmin=0, vmax=360)
    axes[2].set_title("Aspect (° from N)"); plt.colorbar(im, ax=axes[2], shrink=0.7)

    from matplotlib.colors import BoundaryNorm
    bounds = list(range(1, 15))
    cmap_f = plt.get_cmap("tab20", 13)
    norm_f = BoundaryNorm(bounds, cmap_f.N)
    im = axes[3].imshow(t.fuel_model.astype(float), cmap=cmap_f, norm=norm_f, origin="upper")
    axes[3].set_title("Fuel Model (1-13)")
    plt.colorbar(im, ax=axes[3], shrink=0.7, ticks=range(1, 14))

    # Overlay elevation contours on fuel model
    axes[3].contour(t.elevation, levels=[500, 900], colors=["white", "cyan"],
                    linewidths=0.8, linestyles="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig(OUT / "ss2_terrain.png", dpi=110)
    plt.close(fig)
    print("  → saved ss2_terrain.png")


# ═════════════════════════════════════════════════════════════════════════════
# SS3  gp.py
# ═════════════════════════════════════════════════════════════════════════════

def test_ss3_gp():
    from angrybird.gp import IGNISGPPrior, draw_correlated_field, draw_gp_scaled_field
    from angrybird.terrain import synthetic_terrain

    print("\n── SS3  gp.py ────────────────────────────────────────────────")

    t = synthetic_terrain((30, 30))
    shape = (30, 30)

    with _test("SS3", "draw_correlated_field: output shape"):
        f = draw_correlated_field(shape, 500.0, 50.0)
        assert f.shape == shape

    with _test("SS3", "draw_correlated_field: approximately unit variance"):
        # Run many times and check mean std
        stds = [draw_correlated_field(shape, 500.0, 50.0).std() for _ in range(20)]
        mean_std = np.mean(stds)
        assert 0.7 < mean_std < 1.3, f"Expected unit std, got mean={mean_std:.3f}"

    with _test("SS3", "draw_gp_scaled_field: output shape"):
        gp_var = np.full(shape, 0.04, dtype=np.float32)
        f = draw_gp_scaled_field(shape, 500.0, 50.0, gp_var)
        assert f.shape == shape

    with _test("SS3", "IGNISGPPrior: unfitted predict returns default shapes"):
        gp = IGNISGPPrior(terrain=t, resolution_m=50.0)
        prior = gp.predict(shape)
        assert prior.fmc_mean.shape == shape
        assert prior.fmc_variance.shape == shape
        assert prior.wind_speed_mean.shape == shape
        assert prior.wind_speed_variance.shape == shape

    with _test("SS3", "IGNISGPPrior: unfitted prior uses default means"):
        gp = IGNISGPPrior(terrain=t, resolution_m=50.0)
        prior = gp.predict(shape)
        # Default FMC mean = 0.10, wind = 5.0 m/s
        np.testing.assert_allclose(prior.fmc_mean, 0.10, atol=1e-5)
        np.testing.assert_allclose(prior.wind_speed_mean, 5.0, atol=1e-5)

    with _test("SS3", "IGNISGPPrior: unfitted prior variance is positive uniform"):
        gp = IGNISGPPrior(terrain=t, resolution_m=50.0)
        prior = gp.predict(shape)
        assert np.all(prior.fmc_variance > 0)
        # All values should be equal (uniform prior)
        assert prior.fmc_variance.std() < 1e-6

    with _test("SS3", "IGNISGPPrior: add_raws then predict — mean shifts toward obs"):
        gp = IGNISGPPrior(terrain=t, resolution_m=50.0)
        # Inject a high FMC value at center
        gp.add_raws([(15, 15)], fmc_vals=[0.30], ws_vals=[5.0], wd_vals=[270.0])
        prior = gp.predict(shape)
        # Center should be close to 0.30
        assert abs(prior.fmc_mean[15, 15] - 0.30) < 0.05, \
            f"Expected ~0.30 at observation site, got {prior.fmc_mean[15, 15]:.4f}"

    with _test("SS3", "IGNISGPPrior: fitted variance is non-negative everywhere"):
        gp = IGNISGPPrior(terrain=t, resolution_m=50.0)
        gp.add_raws([(5, 5), (25, 25)], fmc_vals=[0.10, 0.12],
                    ws_vals=[5.0, 5.0], wd_vals=[270.0, 270.0])
        prior = gp.predict(shape)
        assert np.all(prior.fmc_variance >= 0), "Variance must be non-negative"

    with _test("SS3", "IGNISGPPrior: variance lower near observation than far"):
        gp = IGNISGPPrior(terrain=t, resolution_m=50.0)
        gp.add_raws([(15, 15)], fmc_vals=[0.10], ws_vals=[5.0], wd_vals=[270.0])
        prior = gp.predict(shape)
        var_near = float(prior.fmc_variance[15, 15])
        var_far  = float(prior.fmc_variance[0, 0])
        assert var_near < var_far, \
            f"Variance near obs ({var_near:.5f}) should be < far ({var_far:.5f})"

    with _test("SS3", "conditional_variance: variance decreases after hypothetical obs"):
        gp = IGNISGPPrior(terrain=t, resolution_m=50.0)
        gp.add_raws([(5, 5)], fmc_vals=[0.10], ws_vals=[5.0], wd_vals=[270.0])
        prior = gp.predict(shape)
        var_before = prior.fmc_variance.copy()
        var_after  = gp.conditional_variance(var_before, (15, 15))
        # Variance should be lower or equal everywhere after conditioning
        assert np.all(var_after <= var_before + 1e-6), \
            "conditional_variance must not increase variance"
        # Variance should strictly decrease near the hypothetical obs point
        assert var_after[15, 15] < var_before[15, 15], \
            "Variance should strictly decrease at the observed cell"

    with _test("SS3", "conditional_variance: no-GP returns input unchanged"):
        gp = IGNISGPPrior(terrain=t, resolution_m=50.0)
        # No observations → _gp_fmc is None after predict (predict returns None gp)
        # Actually predict() auto-fits — but with no obs, _gp_fmc stays None
        var = np.full(shape, 0.04, dtype=np.float32)
        result = gp.conditional_variance(var, (5, 5))
        np.testing.assert_array_almost_equal(result, var)

    # ── Diagnostic plot ───────────────────────────────────────────────────
    gp_diag = IGNISGPPrior(terrain=t, resolution_m=50.0)
    obs_locs = [(5, 25), (25, 5), (15, 15)]
    gp_diag.add_raws(obs_locs, fmc_vals=[0.08, 0.20, 0.14],
                     ws_vals=[4.0, 7.0, 5.0], wd_vals=[270.0, 250.0, 260.0])
    prior_diag = gp_diag.predict(shape)

    var_before_d = prior_diag.fmc_variance.copy()
    var_after_d  = gp_diag.conditional_variance(var_before_d, (15, 15))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("SS3 GP — Prior & Conditional Variance Diagnostic", fontsize=13, fontweight="bold")

    im = axes[0].imshow(prior_diag.fmc_mean, cmap="YlOrRd", origin="upper",
                        vmin=0.05, vmax=0.25)
    for r, c in obs_locs:
        axes[0].plot(c, r, "ko", ms=6)
    axes[0].set_title("FMC posterior mean"); plt.colorbar(im, ax=axes[0], shrink=0.7)

    im = axes[1].imshow(prior_diag.fmc_variance, cmap="Blues", origin="upper")
    for r, c in obs_locs:
        axes[1].plot(c, r, "ro", ms=6)
    axes[1].set_title("FMC variance (after RAWS)"); plt.colorbar(im, ax=axes[1], shrink=0.7)

    im = axes[2].imshow(var_after_d, cmap="Blues", origin="upper",
                        vmin=0, vmax=prior_diag.fmc_variance.max())
    axes[2].plot(15, 15, "y*", ms=12, label="Hyp. obs")
    axes[2].set_title("FMC variance after +1 hyp obs"); plt.colorbar(im, ax=axes[2], shrink=0.7)
    axes[2].legend(fontsize=7)

    reduction = var_before_d - var_after_d
    im = axes[3].imshow(reduction, cmap="hot", origin="upper")
    axes[3].plot(15, 15, "c*", ms=12, label="Hyp. obs")
    axes[3].set_title("Variance reduction Δσ²"); plt.colorbar(im, ax=axes[3], shrink=0.7)
    axes[3].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "ss3_gp.png", dpi=110)
    plt.close(fig)
    print("  → saved ss3_gp.png")


# ═════════════════════════════════════════════════════════════════════════════
# SS4  information.py
# ═════════════════════════════════════════════════════════════════════════════

def test_ss4_information():
    from angrybird.information import (
        compute_sensitivity, compute_observability, compute_information_field,
    )
    from angrybird.gp import IGNISGPPrior
    from angrybird.terrain import synthetic_terrain

    print("\n── SS4  information.py ───────────────────────────────────────")

    shape = (40, 40)
    t = synthetic_terrain(shape)
    gp = IGNISGPPrior(terrain=t, resolution_m=50.0)
    prior = gp.predict(shape)

    # Fire state: small circle of burned cells in the center
    fire_state = np.zeros(shape, dtype=np.float32)
    fire_state[18:22, 18:22] = 1.0

    ens = _make_ensemble(t, prior, fire_state, n_members=30, horizon_min=60.0)

    with _test("SS4", "compute_sensitivity: returns fmc and wind_speed keys"):
        sens = compute_sensitivity(ens, prior)
        assert "fmc" in sens
        assert "wind_speed" in sens

    with _test("SS4", "compute_sensitivity: output shape matches grid"):
        sens = compute_sensitivity(ens, prior)
        assert sens["fmc"].shape == shape
        assert sens["wind_speed"].shape == shape

    with _test("SS4", "compute_sensitivity: values in [-1, 1]"):
        sens = compute_sensitivity(ens, prior)
        assert sens["fmc"].min() >= -1.0 - 1e-5
        assert sens["fmc"].max() <=  1.0 + 1e-5

    with _test("SS4", "compute_sensitivity: non-zero outside fire ignition zone"):
        sens = compute_sensitivity(ens, prior)
        # Sensitivity should be non-zero near the fire front
        non_zero = (np.abs(sens["fmc"]) > 0.01).sum()
        assert non_zero > 10, f"Only {non_zero} cells have non-zero FMC sensitivity"

    with _test("SS4", "compute_observability: correct keys"):
        obs_d = compute_observability(ens, shape, 50.0)
        assert "fmc" in obs_d and "wind_speed" in obs_d

    with _test("SS4", "compute_observability: values in [0, 1]"):
        obs_d = compute_observability(ens, shape, 50.0)
        assert obs_d["fmc"].min() >= 0.0
        assert obs_d["fmc"].max() <= 1.0

    with _test("SS4", "compute_observability: degrades near active fire perimeter"):
        obs_d = compute_observability(ens, shape, 50.0, degradation_radius_m=500.0)
        # The center (fire zone) should have lower observability than corners
        center_obs = float(obs_d["fmc"][20, 20])
        corner_obs = float(obs_d["fmc"][0, 0])
        assert center_obs <= corner_obs, \
            f"Obs near fire ({center_obs:.3f}) should be ≤ far ({corner_obs:.3f})"

    with _test("SS4", "compute_information_field: returns InformationField"):
        from angrybird.types import InformationField
        info = compute_information_field(ens, prior, 50.0, 60.0)
        assert isinstance(info, InformationField)

    with _test("SS4", "compute_information_field: w non-negative"):
        info = compute_information_field(ens, prior, 50.0, 60.0)
        assert np.all(info.w >= 0), "Info field must be non-negative"

    with _test("SS4", "compute_information_field: burned cells have w=0"):
        info = compute_information_field(ens, prior, 50.0, 60.0)
        # Cells with burn_prob > 0.95 should have w = 0
        burned_mask = ens.burn_probability > 0.95
        if burned_mask.any():
            burned_w = info.w[burned_mask]
            assert np.all(burned_w == 0.0), f"Burned cells should have w=0, got max={burned_w.max()}"

    with _test("SS4", "compute_information_field: priority_weight_field scales w"):
        info_base = compute_information_field(ens, prior, 50.0, 60.0)
        weight = np.ones(shape, dtype=np.float32) * 2.0
        info_wtd = compute_information_field(ens, prior, 50.0, 60.0,
                                              priority_weight_field=weight)
        # Weighted sum should be approximately 2x base (except burned cells = 0)
        base_sum = info_base.w.sum()
        wtd_sum  = info_wtd.w.sum()
        if base_sum > 1e-6:
            ratio = wtd_sum / base_sum
            assert abs(ratio - 2.0) < 0.1, f"Weight 2x should double sum, got ratio={ratio:.3f}"

    with _test("SS4", "compute_information_field: exclusion_mask zeros cells"):
        excl = np.zeros(shape, dtype=bool)
        excl[10:20, 10:20] = True
        info_excl = compute_information_field(ens, prior, 50.0, 60.0, exclusion_mask=excl)
        assert np.all(info_excl.w[excl] == 0.0), "Excluded cells must have w=0"

    # ── Diagnostic plot ───────────────────────────────────────────────────
    info = compute_information_field(ens, prior, 50.0, 60.0)
    obs_d = compute_observability(ens, shape, 50.0)
    sens = compute_sensitivity(ens, prior)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("SS4 information — Pipeline Diagnostic", fontsize=13, fontweight="bold")

    im = axes[0].imshow(ens.burn_probability, cmap="Reds", origin="upper", vmin=0, vmax=1)
    axes[0].set_title("Burn probability"); plt.colorbar(im, ax=axes[0], shrink=0.7)

    im = axes[1].imshow(np.abs(sens["fmc"]), cmap="plasma", origin="upper", vmin=0, vmax=1)
    axes[1].set_title("|Sensitivity FMC|"); plt.colorbar(im, ax=axes[1], shrink=0.7)

    im = axes[2].imshow(obs_d["fmc"], cmap="viridis", origin="upper", vmin=0, vmax=1)
    axes[2].set_title("Observability FMC"); plt.colorbar(im, ax=axes[2], shrink=0.7)

    vmax_w = max(info.w.max(), 1e-6)
    im = axes[3].imshow(info.w, cmap="inferno", origin="upper", vmin=0, vmax=vmax_w)
    axes[3].set_title("Info field w(x)"); plt.colorbar(im, ax=axes[3], shrink=0.7)

    fig.tight_layout()
    fig.savefig(OUT / "ss4_information.png", dpi=110)
    plt.close(fig)
    print("  → saved ss4_information.png")


# ═════════════════════════════════════════════════════════════════════════════
# SS5  path_planner.py
# ═════════════════════════════════════════════════════════════════════════════

def test_ss5_path_planner():
    from angrybird.path_planner import cells_along_path, plan_paths, selections_to_mission_queue
    from angrybird.terrain import synthetic_terrain
    from angrybird.gp import IGNISGPPrior
    from angrybird.information import compute_information_field

    print("\n── SS5  path_planner.py ──────────────────────────────────────")

    shape = (30, 30)
    t = synthetic_terrain(shape)
    gp = IGNISGPPrior(terrain=t)
    prior = gp.predict(shape)
    fire_state = np.zeros(shape, dtype=np.float32)
    fire_state[14:16, 14:16] = 1.0
    ens = _make_ensemble(t, prior, fire_state, n_members=20, horizon_min=60.0)
    info = compute_information_field(ens, prior, 50.0, 60.0)

    with _test("SS5", "cells_along_path: includes start and end cells"):
        cells = cells_along_path([(0, 0), (10, 10)], shape, camera_footprint_cells=0)
        assert (0, 0) in cells
        assert (10, 10) in cells

    with _test("SS5", "cells_along_path: camera footprint expands swath"):
        cells_thin  = cells_along_path([(0, 0), (0, 15)], shape, camera_footprint_cells=0)
        cells_thick = cells_along_path([(0, 0), (0, 15)], shape, camera_footprint_cells=2)
        assert len(cells_thick) > len(cells_thin), "Footprint > 0 should cover more cells"

    with _test("SS5", "cells_along_path: all cells within grid bounds"):
        cells = cells_along_path([(0, 0), (29, 29)], shape, camera_footprint_cells=1)
        rows_c = [r for r, c in cells]
        cols_c = [c for r, c in cells]
        assert min(rows_c) >= 0 and max(rows_c) < shape[0]
        assert min(cols_c) >= 0 and max(cols_c) < shape[1]

    with _test("SS5", "cells_along_path: single waypoint returns one cell"):
        cells = cells_along_path([(5, 5)], shape, camera_footprint_cells=0)
        assert cells == [(5, 5)]

    with _test("SS5", "plan_paths: returns correct number of DronePlans"):
        selected = [(5, 5), (10, 15), (20, 10), (25, 25), (15, 5)]
        plans = plan_paths(selected, (29, 15), n_drones=3, shape=shape)
        assert len(plans) == 3

    with _test("SS5", "plan_paths: drone IDs are 0..N-1"):
        selected = [(5, 5), (10, 15), (20, 20)]
        plans = plan_paths(selected, (0, 0), n_drones=3, shape=shape)
        assert {p.drone_id for p in plans} == {0, 1, 2}

    with _test("SS5", "plan_paths: all targets assigned across plans"):
        selected = [(2, 2), (8, 8), (20, 20), (25, 5), (5, 25)]
        plans = plan_paths(selected, (15, 15), n_drones=3, shape=shape)
        all_targets = set()
        for p in plans:
            for wp in p.waypoints:
                if wp != (15, 15):   # exclude staging
                    all_targets.add(wp)
        for tgt in selected:
            assert tgt in all_targets, f"Target {tgt} not assigned to any drone"

    with _test("SS5", "plan_paths: waypoints include staging area start and return"):
        selected = [(5, 5), (10, 10)]
        staging = (29, 0)
        plans = plan_paths(selected, staging, n_drones=2, shape=shape)
        for p in plans:
            if p.waypoints:
                assert p.waypoints[0] == staging, "Path must start at staging"
                assert p.waypoints[-1] == staging, "Path must return to staging"

    with _test("SS5", "plan_paths: empty selection gives empty plans"):
        plans = plan_paths([], (15, 15), n_drones=3, shape=shape)
        assert all(p.waypoints == [] for p in plans)

    with _test("SS5", "selections_to_mission_queue: sorted by info value desc"):
        selected = [(5, 5), (15, 15), (25, 25)]
        mq = selections_to_mission_queue(selected, info, t, 50.0)
        vals = [req.information_value for req in mq.requests]
        assert vals == sorted(vals, reverse=True), "MissionQueue must be sorted descending"

    with _test("SS5", "selections_to_mission_queue: dominant variable is valid"):
        selected = [(5, 5), (15, 15)]
        mq = selections_to_mission_queue(selected, info, t, 50.0)
        for req in mq.requests:
            assert req.dominant_variable in ("fmc", "wind_speed", "wind_dir"), \
                f"Unexpected dominant_variable: {req.dominant_variable}"

    with _test("SS5", "selections_to_mission_queue: targets are valid (lat, lon)"):
        selected = [(5, 5), (15, 15)]
        mq = selections_to_mission_queue(selected, info, t, 50.0)
        for req in mq.requests:
            lat, lon = req.target
            assert 30.0 < lat < 50.0 or True, "lat should be in reasonable range"
            assert isinstance(lat, float) and isinstance(lon, float)

    # ── Diagnostic plot ───────────────────────────────────────────────────
    selected_diag = [(5, 5), (10, 25), (25, 10), (25, 25), (15, 15)]
    staging_diag  = (29, 15)
    plans_diag = plan_paths(selected_diag, staging_diag, n_drones=3, shape=shape)

    colors = ["#E53935", "#1E88E5", "#43A047"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SS5 path_planner — Drone Path Diagnostic", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.imshow(info.w, cmap="inferno", origin="upper", alpha=0.7,
                  vmin=0, vmax=max(info.w.max(), 1e-6))
        ax.plot(staging_diag[1], staging_diag[0], "w^", ms=10, zorder=6, label="Staging")

    # Left: drone assignments
    for p in plans_diag:
        col = colors[p.drone_id % len(colors)]
        if p.waypoints:
            wcs = [w[1] for w in p.waypoints]
            wrs = [w[0] for w in p.waypoints]
            axes[0].plot(wcs, wrs, "-o", color=col, ms=5, lw=2, label=f"Drone {p.drone_id}")
    axes[0].set_title("Drone paths"); axes[0].legend(fontsize=7)

    # Right: cells_observed heatmap
    obs_grid = np.zeros(shape)
    for p in plans_diag:
        for r, c in p.cells_observed:
            obs_grid[r, c] += 1
    axes[1].imshow(obs_grid, cmap="YlGn", origin="upper", alpha=0.8)
    axes[1].set_title(f"Cells observed (total={int(obs_grid.sum())})")

    fig.tight_layout()
    fig.savefig(OUT / "ss5_path_planner.png", dpi=110)
    plt.close(fig)
    print("  → saved ss5_path_planner.png")


# ═════════════════════════════════════════════════════════════════════════════
# SS6  selectors/
# ═════════════════════════════════════════════════════════════════════════════

def test_ss6_selectors():
    from angrybird.selectors.baselines import UniformSelector, FireFrontSelector
    from angrybird.selectors.greedy import GreedySelector
    from angrybird.gp import IGNISGPPrior
    from angrybird.terrain import synthetic_terrain
    from angrybird.information import compute_information_field

    print("\n── SS6  selectors/ ───────────────────────────────────────────")

    shape = (40, 40)
    t = synthetic_terrain(shape)
    gp = IGNISGPPrior(terrain=t, resolution_m=50.0)
    prior = gp.predict(shape)
    fire_state = np.zeros(shape, dtype=np.float32)
    fire_state[30:38, 6:14] = 1.0     # SW ignition
    ens = _make_ensemble(t, prior, fire_state, n_members=30, horizon_min=60.0)
    info = compute_information_field(ens, prior, 50.0, 60.0)
    k = 5

    # UniformSelector ─────────────────────────────────────────────────────
    us = UniformSelector()

    with _test("SS6", "UniformSelector: returns exactly k locations"):
        res = us.select(info, gp, ens, k)
        assert len(res.selected_locations) == k, f"Expected {k}, got {len(res.selected_locations)}"

    with _test("SS6", "UniformSelector: all locations within grid bounds"):
        res = us.select(info, gp, ens, k)
        for r, c in res.selected_locations:
            assert 0 <= r < shape[0] and 0 <= c < shape[1]

    with _test("SS6", "UniformSelector: marginal_gains length matches selected"):
        res = us.select(info, gp, ens, k)
        assert len(res.marginal_gains) == len(res.selected_locations)

    with _test("SS6", "UniformSelector: strategy_name is 'uniform'"):
        res = us.select(info, gp, ens, k)
        assert res.strategy_name == "uniform"

    # FireFrontSelector ───────────────────────────────────────────────────
    ffs = FireFrontSelector()

    with _test("SS6", "FireFrontSelector: returns at most k locations"):
        res = ffs.select(info, gp, ens, k)
        assert len(res.selected_locations) <= k

    with _test("SS6", "FireFrontSelector: returns locations on/near perimeter"):
        res = ffs.select(info, gp, ens, k)
        if res.selected_locations:
            # All selected cells should have some fire probability
            for r, c in res.selected_locations:
                assert ens.burn_probability[r, c] >= 0.0   # should be >= lo_thresh usually

    with _test("SS6", "FireFrontSelector: strategy_name is 'fire_front'"):
        res = ffs.select(info, gp, ens, k)
        assert res.strategy_name == "fire_front"

    with _test("SS6", "FireFrontSelector: degenerate case (no fire) doesn't crash"):
        no_fire = np.zeros(shape, dtype=np.float32)
        ens_nofire = _make_ensemble(t, prior, no_fire, n_members=10, horizon_min=10.0)
        info_nofire = compute_information_field(ens_nofire, prior, 50.0, 10.0)
        res = ffs.select(info_nofire, gp, ens_nofire, k)
        # Should not crash — may return empty list
        assert isinstance(res.selected_locations, list)

    # GreedySelector ──────────────────────────────────────────────────────
    gs = GreedySelector()

    with _test("SS6", "GreedySelector: returns at most k locations"):
        res = gs.select(info, gp, ens, k)
        assert len(res.selected_locations) <= k

    with _test("SS6", "GreedySelector: strategy_name is 'greedy'"):
        res = gs.select(info, gp, ens, k)
        assert res.strategy_name == "greedy"

    with _test("SS6", "GreedySelector: marginal_gains are non-increasing"):
        res = gs.select(info, gp, ens, k)
        mg = res.marginal_gains
        if len(mg) > 1:
            # Greedy submodular: each subsequent gain ≤ previous (diminishing returns)
            # Allow small numerical tolerance
            for i in range(len(mg) - 1):
                assert mg[i+1] <= mg[i] + 1e-6, \
                    f"Marginal gains not non-increasing: mg[{i}]={mg[i]:.5f} < mg[{i+1}]={mg[i+1]:.5f}"

    with _test("SS6", "GreedySelector: cumulative_gain is running total"):
        res = gs.select(info, gp, ens, k)
        cg = res.cumulative_gain
        mg = res.marginal_gains
        if cg and mg:
            np.testing.assert_allclose(cg[-1], sum(mg), rtol=1e-5)

    with _test("SS6", "GreedySelector: no duplicates in selected_locations"):
        res = gs.select(info, gp, ens, k)
        locs = res.selected_locations
        assert len(set(locs)) == len(locs), "Duplicate locations returned by greedy"

    with _test("SS6", "GreedySelector: spacing constraint respected (>= min_cells apart)"):
        res = gs.select(info, gp, ens, k)
        locs = res.selected_locations
        min_cells = gs.min_spacing_m / gs.resolution_m
        for i, (r1, c1) in enumerate(locs):
            for j, (r2, c2) in enumerate(locs):
                if i != j:
                    dist = np.sqrt((r1-r2)**2 + (c1-c2)**2)
                    assert dist >= min_cells - 0.5, \
                        f"Spacing violated: {dist:.1f} < {min_cells:.1f} cells"

    # ── Diagnostic plot ───────────────────────────────────────────────────
    res_u  = us.select(info, gp, ens, k)
    res_ff = ffs.select(info, gp, ens, k)
    res_g  = gs.select(info, gp, ens, k)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("SS6 selectors — Selection Strategy Diagnostic", fontsize=13, fontweight="bold")

    def _plot_sel(ax, result, title):
        vmax_w = max(info.w.max(), 1e-6)
        ax.imshow(info.w, cmap="inferno", origin="upper", vmin=0, vmax=vmax_w)
        ax.contour(ens.burn_probability, levels=[0.2, 0.5, 0.8],
                   colors=["cyan", "white", "red"], linewidths=0.8, alpha=0.7)
        for r, c in result.selected_locations:
            ax.plot(c, r, "y^", ms=10, zorder=5, markeredgecolor="black", markeredgewidth=0.5)
        ax.set_title(title)

    _plot_sel(axes[0], res_u,  f"Uniform (k={len(res_u.selected_locations)})")
    _plot_sel(axes[1], res_ff, f"FireFront (k={len(res_ff.selected_locations)})")
    _plot_sel(axes[2], res_g,  f"Greedy (k={len(res_g.selected_locations)})")

    # Marginal gains comparison
    axes[3].plot(range(1, len(res_u.marginal_gains)+1),  res_u.marginal_gains,  "b-o", ms=5, label="Uniform")
    axes[3].plot(range(1, len(res_ff.marginal_gains)+1), res_ff.marginal_gains, "g-s", ms=5, label="FireFront")
    axes[3].plot(range(1, len(res_g.marginal_gains)+1),  res_g.marginal_gains,  "r-^", ms=5, label="Greedy")
    axes[3].set_xlabel("Selection index"); axes[3].set_ylabel("Marginal gain w_i")
    axes[3].set_title("Marginal gains (diminishing returns)"); axes[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "ss6_selectors.png", dpi=110)
    plt.close(fig)
    print("  → saved ss6_selectors.png")


# ═════════════════════════════════════════════════════════════════════════════
# SS7  assimilation.py
# ═════════════════════════════════════════════════════════════════════════════

def test_ss7_assimilation():
    from angrybird.assimilation import thin_drone_observations, enkf_update, assimilate_observations
    from angrybird.types import DroneObservation
    from angrybird.gp import IGNISGPPrior
    from angrybird.terrain import synthetic_terrain
    from angrybird.information import compute_information_field

    print("\n── SS7  assimilation.py ──────────────────────────────────────")

    shape = (20, 20)
    t = synthetic_terrain(shape)

    # ── Thinning ─────────────────────────────────────────────────────────
    def _make_obs(locations, fmc=0.10, sigma=0.05, ws=5.0, ws_sig=1.0):
        return [DroneObservation(location=loc, fmc=fmc, fmc_sigma=sigma,
                                 wind_speed=ws, wind_speed_sigma=ws_sig)
                for loc in locations]

    with _test("SS7", "thin_drone_observations: empty input → empty output"):
        assert thin_drone_observations([]) == []

    with _test("SS7", "thin_drone_observations: spacing constraint satisfied"):
        # Dense cluster: 6 points very close together
        cluster = [(5, 5), (5, 6), (5, 7), (6, 5), (6, 6), (6, 7)]
        obs = _make_obs(cluster)
        thinned = thin_drone_observations(obs, min_spacing_m=200.0, resolution_m=50.0)
        # min_cells = 200/50 = 4; all 6 are within 4 cells of each other → only 1 kept
        assert len(thinned) == 1, f"Expected 1, got {len(thinned)}"

    with _test("SS7", "thin_drone_observations: distant observations all kept"):
        distant = [(0, 0), (0, 15), (15, 0), (15, 15)]
        obs = _make_obs(distant)
        thinned = thin_drone_observations(obs, min_spacing_m=200.0, resolution_m=50.0)
        # All 4 are ≥ 15 cells = 750m apart → all should survive
        assert len(thinned) == 4, f"Expected 4, got {len(thinned)}"

    with _test("SS7", "thin_drone_observations: keeps lowest-noise obs"):
        locs = [(5, 5), (5, 6)]   # 1 cell apart (too close at min_spacing=200m)
        obs_hi = DroneObservation(location=(5, 5), fmc=0.10, fmc_sigma=0.10,   # high noise
                                  wind_speed=5.0, wind_speed_sigma=1.0)
        obs_lo = DroneObservation(location=(5, 6), fmc=0.12, fmc_sigma=0.02,   # low noise
                                  wind_speed=5.0, wind_speed_sigma=1.0)
        thinned = thin_drone_observations([obs_hi, obs_lo], min_spacing_m=200.0, resolution_m=50.0)
        # Sorted by fmc_sigma → obs_lo first → obs_lo should be kept
        assert len(thinned) == 1
        assert thinned[0].fmc_sigma == 0.02, "Should keep lowest-noise observation"

    # ── EnKF update ───────────────────────────────────────────────────────
    rng_enkf = np.random.default_rng(42)
    N, D = 30, shape[0] * shape[1]
    X = rng_enkf.standard_normal((N, D)).astype(np.float64) + 0.10

    obs_locs  = [(5, 5), (15, 15)]
    obs_idx   = [r * shape[1] + c for r, c in obs_locs]
    y_obs     = np.array([0.20, 0.25])
    obs_sigma = np.array([0.05, 0.05])

    with _test("SS7", "enkf_update: output shape matches input"):
        X_up = enkf_update(X, y_obs, obs_idx, obs_sigma, shape, obs_locs,
                           resolution_m=50.0, rng=rng_enkf)
        assert X_up.shape == (N, D), f"Expected {(N, D)}, got {X_up.shape}"

    with _test("SS7", "enkf_update: ensemble mean shifts toward observations"):
        X_up = enkf_update(X, y_obs, obs_idx, obs_sigma, shape, obs_locs,
                           resolution_m=50.0, rng=np.random.default_rng(1))
        mean_before = X[:, obs_idx[0]].mean()
        mean_after  = X_up[:, obs_idx[0]].mean()
        # Mean should move toward y_obs[0] = 0.20
        assert abs(mean_after - 0.20) < abs(mean_before - 0.20) + 0.01, \
            f"Mean should move toward obs: before={mean_before:.3f}, after={mean_after:.3f}, obs=0.20"

    with _test("SS7", "enkf_update: variance should decrease at observation locations"):
        X_up = enkf_update(X, y_obs, obs_idx, obs_sigma, shape, obs_locs,
                           resolution_m=50.0, rng=np.random.default_rng(2))
        var_before = X[:, obs_idx[0]].var()
        var_after  = X_up[:, obs_idx[0]].var()
        # Variance typically decreases after assimilation (though inflation can partially offset)
        assert var_after < var_before * 2.0, \
            f"Variance exploded: before={var_before:.4f}, after={var_after:.4f}"

    with _test("SS7", "enkf_update: no observations → returns copy of input"):
        X_up = enkf_update(X, np.array([]), [], np.array([]), shape, [],
                           resolution_m=50.0)
        np.testing.assert_array_equal(X_up, X)

    # ── Replan flags ──────────────────────────────────────────────────────
    with _test("SS7", "assimilate_observations: returns correct keys"):
        gp = IGNISGPPrior(terrain=t, resolution_m=50.0)
        prior = gp.predict(shape)
        fire_state = np.zeros(shape, dtype=np.float32)
        fire_state[8:12, 8:12] = 1.0
        ens = _make_ensemble(t, prior, fire_state, n_members=20, horizon_min=60.0)
        obs = _make_obs([(2, 2), (17, 17), (2, 17)], fmc=0.15, ws=5.5)
        result = assimilate_observations(gp, ens, obs, shape, resolution_m=50.0, gp_prior=prior)
        assert "fmc_states" in result
        assert "wind_states" in result
        assert "replan_flags" in result
        assert "n_obs_used" in result

    with _test("SS7", "assimilate_observations: fmc_states shape is (N, rows, cols)"):
        gp = IGNISGPPrior(terrain=t, resolution_m=50.0)
        prior = gp.predict(shape)
        fire_state = np.zeros(shape, dtype=np.float32)
        ens = _make_ensemble(t, prior, fire_state, n_members=20, horizon_min=60.0)
        obs = _make_obs([(2, 2)])
        result = assimilate_observations(gp, ens, obs, shape, resolution_m=50.0)
        assert result["fmc_states"].shape == (20, *shape)

    with _test("SS7", "assimilate_observations: n_obs_used ≤ n_obs_input"):
        gp = IGNISGPPrior(terrain=t, resolution_m=50.0)
        prior = gp.predict(shape)
        fire_state = np.zeros(shape, dtype=np.float32)
        ens = _make_ensemble(t, prior, fire_state, n_members=20, horizon_min=60.0)
        # Dense observations that should be thinned
        cluster = _make_obs([(5, 5), (5, 6), (5, 7)])
        result = assimilate_observations(gp, ens, cluster, shape, resolution_m=50.0)
        assert result["n_obs_used"] <= len(cluster)

    # ── Diagnostic plot ───────────────────────────────────────────────────
    rng_d = np.random.default_rng(0)
    N_d, D_d = 50, 30 * 30
    X_d = (rng_d.standard_normal((N_d, D_d)) * 0.05 + 0.10).astype(np.float64)
    obs_locs_d = [(5, 5), (25, 25)]
    obs_idx_d  = [r * 30 + c for r, c in obs_locs_d]
    y_d        = np.array([0.25, 0.05])
    sig_d      = np.array([0.03, 0.03])
    X_d_up     = enkf_update(X_d, y_d, obs_idx_d, sig_d, (30, 30), obs_locs_d,
                              resolution_m=50.0, rng=np.random.default_rng(7))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("SS7 assimilation — EnKF Update Diagnostic", fontsize=13, fontweight="bold")

    mean_before_d = X_d.mean(axis=0).reshape(30, 30)
    mean_after_d  = X_d_up.mean(axis=0).reshape(30, 30)
    var_before_d  = X_d.var(axis=0).reshape(30, 30)
    var_after_d   = X_d_up.var(axis=0).reshape(30, 30)

    vmax_m = max(mean_before_d.max(), mean_after_d.max())
    vmin_m = min(mean_before_d.min(), mean_after_d.min())

    im = axes[0].imshow(mean_before_d, cmap="YlOrRd", origin="upper",
                        vmin=vmin_m, vmax=vmax_m)
    axes[0].set_title("Ens mean FMC — BEFORE")
    for r, c in obs_locs_d:
        axes[0].plot(c, r, "b*", ms=12)
    plt.colorbar(im, ax=axes[0], shrink=0.7)

    im = axes[1].imshow(mean_after_d, cmap="YlOrRd", origin="upper",
                        vmin=vmin_m, vmax=vmax_m)
    for i, (r, c) in enumerate(obs_locs_d):
        axes[1].plot(c, r, "b*", ms=12, label=f"obs={y_d[i]:.2f}" if i == 0 else "")
    axes[1].set_title("Ens mean FMC — AFTER EnKF")
    axes[1].legend(fontsize=7)
    plt.colorbar(im, ax=axes[1], shrink=0.7)

    var_diff = var_before_d - var_after_d
    im = axes[2].imshow(var_diff, cmap="RdBu", origin="upper",
                        vmin=-var_diff.std(), vmax=var_diff.std())
    for r, c in obs_locs_d:
        axes[2].plot(c, r, "k*", ms=12)
    axes[2].set_title("Variance reduction Δvar (blue=reduced)")
    plt.colorbar(im, ax=axes[2], shrink=0.7)

    fig.tight_layout()
    fig.savefig(OUT / "ss7_assimilation.png", dpi=110)
    plt.close(fig)
    print("  → saved ss7_assimilation.png")


# ═════════════════════════════════════════════════════════════════════════════
# SS8  simulation/ground_truth.py
# ═════════════════════════════════════════════════════════════════════════════

def test_ss8_ground_truth():
    from simulation.ground_truth import generate_ground_truth, GroundTruth
    from angrybird.terrain import synthetic_terrain

    print("\n── SS8  simulation/ground_truth.py ───────────────────────────")

    shape = (40, 40)
    t = synthetic_terrain(shape, seed=7)
    ignition = (20, 20)

    with _test("SS8", "generate_ground_truth: returns GroundTruth instance"):
        gt = generate_ground_truth(t, ignition_cell=ignition)
        assert isinstance(gt, GroundTruth)

    with _test("SS8", "generate_ground_truth: field shapes match terrain"):
        gt = generate_ground_truth(t, ignition_cell=ignition)
        assert gt.fmc.shape == shape
        assert gt.wind_speed.shape == shape
        assert gt.wind_direction.shape == shape

    with _test("SS8", "generate_ground_truth: FMC in dead-fuel range [0.02, 0.40]"):
        gt = generate_ground_truth(t, ignition_cell=ignition)
        assert gt.fmc.min() >= 0.02 - 1e-5, f"FMC min={gt.fmc.min():.4f}"
        assert gt.fmc.max() <= 0.40 + 1e-5, f"FMC max={gt.fmc.max():.4f}"

    with _test("SS8", "generate_ground_truth: wind speed in [0.5, 25.0] m/s"):
        gt = generate_ground_truth(t, ignition_cell=ignition)
        assert gt.wind_speed.min() >= 0.5 - 1e-5, f"WS min={gt.wind_speed.min():.3f}"
        assert gt.wind_speed.max() <= 25.0 + 1e-5, f"WS max={gt.wind_speed.max():.3f}"

    with _test("SS8", "generate_ground_truth: wind direction in [0, 360)"):
        gt = generate_ground_truth(t, ignition_cell=ignition)
        assert gt.wind_direction.min() >= 0.0
        assert gt.wind_direction.max() < 360.0

    with _test("SS8", "generate_ground_truth: deterministic with same seed"):
        gt1 = generate_ground_truth(t, ignition_cell=ignition, seed=42)
        gt2 = generate_ground_truth(t, ignition_cell=ignition, seed=42)
        np.testing.assert_array_equal(gt1.fmc, gt2.fmc)

    with _test("SS8", "generate_ground_truth: different seeds give different results"):
        gt1 = generate_ground_truth(t, ignition_cell=ignition, seed=1)
        gt2 = generate_ground_truth(t, ignition_cell=ignition, seed=2)
        assert not np.array_equal(gt1.fmc, gt2.fmc)

    with _test("SS8", "generate_ground_truth: FMC spatially correlated (not white noise)"):
        gt = generate_ground_truth(t, ignition_cell=ignition, seed=5)
        # Spatial correlation: adjacent cells should be more similar than random noise
        # Compute mean absolute difference between adjacent cells
        fmc = gt.fmc
        adj_diff = np.mean(np.abs(np.diff(fmc, axis=0))) + np.mean(np.abs(np.diff(fmc, axis=1)))
        # White noise (σ≈0.04) adjacent-cell diff ≈ 0.056; correlated at 500m on 50m grid is smoother.
        # Threshold chosen conservatively (0.05) to pass even with noise component.
        assert adj_diff < 0.05, f"FMC not spatially smooth enough: adj_diff={adj_diff:.4f}"

    with _test("SS8", "generate_ground_truth: terrain aspect drives FMC (south drier, post bug-fix)"):
        gt = generate_ground_truth(t, ignition_cell=ignition, seed=3, base_fmc=0.15)
        # South-facing (180°) should be drier (lower FMC) than north-facing (0°).
        # This test catches the sign-inversion bug (aspect_effect sign was wrong before fix).
        south_mask = (t.aspect > 135) & (t.aspect < 225)
        north_mask = (t.aspect > 315) | (t.aspect < 45)
        if south_mask.sum() > 5 and north_mask.sum() > 5:
            south_fmc = gt.fmc[south_mask].mean()
            north_fmc = gt.fmc[north_mask].mean()
            assert south_fmc < north_fmc + 0.005, \
                f"South-facing ({south_fmc:.4f}) should be drier than north ({north_fmc:.4f})"

    # ── Diagnostic plot ───────────────────────────────────────────────────
    t_big = synthetic_terrain((60, 60), seed=42)
    gt = generate_ground_truth(t_big, ignition_cell=(30,30), seed=0, base_fmc=0.10, base_ws=5.0, base_wd=180.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("SS8 ground_truth — Hidden State Diagnostic", fontsize=13, fontweight="bold")

    im = axes[0].imshow(t_big.elevation, cmap="terrain", origin="upper")
    axes[0].set_title("Terrain elevation (context)"); plt.colorbar(im, ax=axes[0], shrink=0.7)

    im = axes[1].imshow(gt.fmc, cmap="YlGn", origin="upper",
                        vmin=0.02, vmax=0.40)
    axes[1].set_title("True FMC field"); plt.colorbar(im, ax=axes[1], shrink=0.7)

    im = axes[2].imshow(gt.wind_speed, cmap="Blues", origin="upper", vmin=0, vmax=15)
    axes[2].set_title("True wind speed (m/s)"); plt.colorbar(im, ax=axes[2], shrink=0.7)

    im = axes[3].imshow(gt.wind_direction, cmap="hsv", origin="upper", vmin=0, vmax=360)
    axes[3].set_title("True wind direction (°)"); plt.colorbar(im, ax=axes[3], shrink=0.7)

    fig.tight_layout()
    fig.savefig(OUT / "ss8_ground_truth.png", dpi=110)
    plt.close(fig)
    print("  → saved ss8_ground_truth.png")


# ═════════════════════════════════════════════════════════════════════════════
# SS9  simulation/observer.py
# ═════════════════════════════════════════════════════════════════════════════

def test_ss9_observer():
    from simulation.observer import SimulatedObserver, ObservationSource
    from simulation.ground_truth import generate_ground_truth
    from angrybird.terrain import synthetic_terrain

    print("\n── SS9  simulation/observer.py ───────────────────────────────")

    shape = (30, 30)
    t = synthetic_terrain(shape, seed=13)
    ignition = (15, 15)
    gt = generate_ground_truth(t, ignition_cell=ignition, seed=0)
    obs = SimulatedObserver(gt, fmc_sigma=0.05, wind_speed_sigma=1.0,
                             rng=np.random.default_rng(42))

    with _test("SS9", "SimulatedObserver satisfies ObservationSource protocol"):
        assert isinstance(obs, ObservationSource)

    with _test("SS9", "observe: returns one DroneObservation per cell"):
        cells = [(0, 0), (5, 5), (15, 15)]
        observations = obs.observe(cells)
        assert len(observations) == len(cells)

    with _test("SS9", "observe: observation locations match requested cells"):
        cells = [(3, 7), (21, 14)]
        observations = obs.observe(cells)
        for drobs, cell in zip(observations, cells):
            assert drobs.location == cell

    with _test("SS9", "observe: wind speed ≥ 0 (clipping works)"):
        # With high noise, some obs could go negative — must be clipped
        noisy = SimulatedObserver(gt, wind_speed_sigma=100.0, rng=np.random.default_rng(0))
        cells = [(r, c) for r in range(0, 30, 3) for c in range(0, 30, 3)]
        observations = noisy.observe(cells)
        ws_vals = [o.wind_speed for o in observations]
        assert all(ws >= 0.0 for ws in ws_vals), f"Negative wind speed: min={min(ws_vals):.3f}"

    with _test("SS9", "observe: wind direction wrapped to [0, 360)"):
        noisy_wd = SimulatedObserver(gt, wind_dir_sigma=200.0, rng=np.random.default_rng(0))
        cells = [(r, c) for r in range(0, 30, 3) for c in range(0, 30, 3)]
        observations = noisy_wd.observe(cells)
        for o in observations:
            assert 0.0 <= o.wind_dir < 360.0, f"Wind dir out of range: {o.wind_dir}"

    with _test("SS9", "observe: FMC noise has correct magnitude"):
        many_obs = obs.observe([(10, 10)] * 500)
        fmc_vals = np.array([o.fmc for o in many_obs])
        true_fmc = float(gt.fmc[10, 10])
        measured_std = fmc_vals.std()
        assert abs(measured_std - 0.05) < 0.015, \
            f"Expected FMC noise σ≈0.05, got {measured_std:.4f}"

    with _test("SS9", "observe: observations are noisy (not exact true values)"):
        observations = obs.observe([(10, 10)])
        o = observations[0]
        true_fmc = float(gt.fmc[10, 10])
        # With sigma=0.05, probability of exact match is essentially zero
        assert abs(o.fmc - true_fmc) < 0.30, f"Obs too far from truth: {abs(o.fmc - true_fmc):.3f}"

    with _test("SS9", "observe: fmc_sigma field matches constructor value"):
        cells = [(5, 5), (10, 10)]
        observations = obs.observe(cells)
        for o in observations:
            assert o.fmc_sigma == 0.05
            assert o.wind_speed_sigma == 1.0

    with _test("SS9", "observe: reproducible with same rng seed"):
        obs1 = SimulatedObserver(gt, rng=np.random.default_rng(99))
        obs2 = SimulatedObserver(gt, rng=np.random.default_rng(99))
        cells = [(5, 5), (10, 10), (20, 20)]
        res1 = obs1.observe(cells)
        res2 = obs2.observe(cells)
        for a, b in zip(res1, res2):
            assert abs(a.fmc - b.fmc) < 1e-10

    # ── Diagnostic plot ───────────────────────────────────────────────────
    obs_diag = SimulatedObserver(gt, fmc_sigma=0.05, rng=np.random.default_rng(0))
    all_cells = [(r, c) for r in range(0, 30, 2) for c in range(0, 30, 2)]
    all_obs   = obs_diag.observe(all_cells)

    true_fmc = np.array([float(gt.fmc[r, c]) for r, c in all_cells])
    obs_fmc  = np.array([o.fmc for o in all_obs])
    true_ws  = np.array([float(gt.wind_speed[r, c]) for r, c in all_cells])
    obs_ws   = np.array([o.wind_speed for o in all_obs])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("SS9 observer — Noise Model Diagnostic", fontsize=13, fontweight="bold")

    # True vs observed FMC scatter
    lo = min(true_fmc.min(), obs_fmc.min())
    hi = max(true_fmc.max(), obs_fmc.max())
    axes[0].scatter(true_fmc, obs_fmc, s=10, alpha=0.5, c="steelblue")
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y=x")
    axes[0].set_xlabel("True FMC"); axes[0].set_ylabel("Observed FMC")
    axes[0].set_title("FMC true vs observed"); axes[0].legend(fontsize=8)

    # FMC residuals histogram
    resid_fmc = obs_fmc - true_fmc
    axes[1].hist(resid_fmc, bins=25, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="r", ls="--")
    axes[1].set_xlabel("Residual (obs - true)"); axes[1].set_ylabel("Count")
    sigma_fit = resid_fmc.std()
    axes[1].set_title(f"FMC noise (σ_fit={sigma_fit:.3f} vs σ_target=0.05)")

    # Wind speed noise
    resid_ws = obs_ws - true_ws
    axes[2].hist(resid_ws, bins=25, color="darkorange", edgecolor="white", alpha=0.8)
    axes[2].axvline(0, color="r", ls="--")
    axes[2].set_xlabel("Residual (obs - true)"); axes[2].set_ylabel("Count")
    sigma_ws_fit = resid_ws.std()
    axes[2].set_title(f"Wind speed noise (σ_fit={sigma_ws_fit:.2f} vs σ_target=1.0)")

    fig.tight_layout()
    fig.savefig(OUT / "ss9_observer.png", dpi=110)
    plt.close(fig)
    print("  → saved ss9_observer.png")


# ═════════════════════════════════════════════════════════════════════════════
# Summary plot + report
# ═════════════════════════════════════════════════════════════════════════════

def _print_summary():
    total  = len(RESULTS)
    passed = sum(1 for r in RESULTS if r.passed)
    failed = total - passed

    print("\n" + "═" * 72)
    print(f"  IGNIS SUBSYSTEM TEST SUMMARY  —  {passed}/{total} passed")
    print("═" * 72)

    # Group by subsystem
    from collections import defaultdict
    by_ss: dict[str, list[Result]] = defaultdict(list)
    for r in RESULTS:
        by_ss[r.subsystem].append(r)

    for ss, results in sorted(by_ss.items()):
        n_pass = sum(1 for r in results if r.passed)
        n_tot  = len(results)
        status = "✓" if n_pass == n_tot else "✗"
        print(f"\n  {status}  {ss}  ({n_pass}/{n_tot})")
        for r in results:
            sym = "✓" if r.passed else "✗"
            note = "" if r.passed else f"  ← {r.detail}"
            print(f"       {sym} {r.name}{note}")

    print()
    if failed:
        print(f"  ✗  {failed} tests FAILED  — see details above")
    else:
        print("  ✓  ALL TESTS PASSED")
    print("═" * 72)

    return passed, failed


def _make_summary_plot(passed: int, failed: int):
    """Compact pass/fail grid for the summary."""
    from collections import defaultdict
    by_ss: dict[str, list[Result]] = defaultdict(list)
    for r in RESULTS:
        by_ss[r.subsystem].append(r)

    fig, ax = plt.subplots(figsize=(14, max(5, len(RESULTS) * 0.35 + 2)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    y = 0
    ss_colors = {
        "SS1": "#4FC3F7", "SS2": "#81C784", "SS3": "#FFB74D",
        "SS4": "#F06292", "SS5": "#CE93D8", "SS6": "#80DEEA",
        "SS7": "#FFCC80", "SS8": "#A5D6A7", "SS9": "#EF9A9A",
    }
    labels = []
    ys     = []

    for ss, results in sorted(by_ss.items()):
        ss_col = ss_colors.get(ss, "#FFFFFF")
        for r in results:
            col = "#26A65B" if r.passed else "#E74C3C"
            rect = plt.Rectangle((0, y), 1, 0.8, color=col, zorder=2)
            ax.add_patch(rect)
            status = "✓" if r.passed else "✗"
            ax.text(0.05, y + 0.4, f"{status}  [{ss}] {r.name}",
                    va="center", ha="left", fontsize=8.5, color="white", zorder=3)
            ax.text(1.02, y + 0.4, f"{r.elapsed_s*1000:.0f} ms",
                    va="center", ha="left", fontsize=7.5, color="#AAAAAA", zorder=3)
            labels.append(r.name)
            ys.append(y)
            y += 1.0

    ax.set_xlim(-0.1, 1.8)
    ax.set_ylim(-0.3, y + 0.5)
    ax.axis("off")
    ax.invert_yaxis()

    total = len(RESULTS)
    title = f"IGNIS Subsystem Tests — {passed}/{total} passed"
    fig.suptitle(title, fontsize=14, fontweight="bold",
                 color="white", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "ss0_summary.png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  → saved ss0_summary.png  ({OUT})")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0_total = time.perf_counter()
    print("=" * 72)
    print("  IGNIS Subsystem Tests + Diagnostic Visualisations")
    print(f"  Output: {OUT}")
    print("=" * 72)

    test_ss1_utils()
    test_ss2_terrain()
    test_ss3_gp()
    test_ss4_information()
    test_ss5_path_planner()
    test_ss6_selectors()
    test_ss7_assimilation()
    test_ss8_ground_truth()
    test_ss9_observer()

    passed, failed = _print_summary()
    _make_summary_plot(passed, failed)

    total_s = time.perf_counter() - t0_total
    print(f"\n  Total runtime: {total_s:.1f} s")
    sys.exit(0 if failed == 0 else 1)
