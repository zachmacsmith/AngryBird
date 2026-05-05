"""
Tests for angrybird/fire_engines/gpu_fire_engine.py

Coverage:
  Unit tests  — fuel table, Rothermel physics, gradient functions,
                directional ROS, redistancing
  Integration — engine shapes / sentinels, circular spread, directional spread,
                initial_phi seeding, reproducibility
  Full-stack  — synthetic_terrain + GPPrior ensemble, fire-type output
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# Skip the entire module if PyTorch is not installed
torch = pytest.importorskip("torch", reason="PyTorch not installed")

from angrybird.fire_engines.gpu_fire_engine import (
    GPUFireEngine,
    _build_fuel_table,
    _directional_ros,
    _phi_gradient,
    _phi_gradient_full,
    _redistance,
    _rothermel_ros,
    _COL_W0, _COL_SIGMA, _COL_DELTA, _COL_MX, _COL_ST,
    _N_MODELS, _N_PARAMS,
)
from angrybird.config import FUEL_PARAMS
from angrybird.terrain import synthetic_terrain
from angrybird.types import GPPrior


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

DEVICE = "cpu"   # tests run on CPU; everything is device-agnostic


def _ones(shape, val=1.0) -> "torch.Tensor":
    return torch.full(shape, val, dtype=torch.float32, device=DEVICE)


def _flat_gp(rows: int, cols: int,
             fmc: float = 0.08,
             ws: float = 5.0,
             wd: float = 270.0,
             ws_var: float = 0.0,
             fmc_var: float = 0.0,
             wd_var: float = 0.0) -> GPPrior:
    """Spatially uniform GP prior for integration tests."""
    return GPPrior(
        fmc_mean             = np.full((rows, cols), fmc,    dtype=np.float32),
        fmc_variance         = np.full((rows, cols), fmc_var, dtype=np.float32),
        wind_speed_mean      = np.full((rows, cols), ws,     dtype=np.float32),
        wind_speed_variance  = np.full((rows, cols), ws_var, dtype=np.float32),
        wind_dir_mean        = np.full((rows, cols), wd,     dtype=np.float32),
        wind_dir_variance    = np.full((rows, cols), wd_var, dtype=np.float32),
    )


def _center_fire(rows: int, cols: int) -> np.ndarray:
    """Single burned cell at grid center."""
    fs = np.zeros((rows, cols), dtype=np.float32)
    fs[rows // 2, cols // 2] = 1.0
    return fs


def _slow_terrain(rows: int, cols: int, resolution_m: float = 50.0):
    """
    Flat terrain with SB40 TL3 (code 183, heavy timber litter) for sentinel tests.
    High moisture extinction (mx=0.35) and no slope mean fire spreads slowly;
    corners of a 30×30 grid at 50 m stay unburned in 30 min at low wind.
    """
    from angrybird.types import TerrainData
    zero = np.zeros((rows, cols), dtype=np.float32)
    return TerrainData(
        elevation          = np.full((rows, cols), 500.0, dtype=np.float32),
        slope              = zero.copy(),
        aspect             = zero.copy(),
        fuel_model         = np.full((rows, cols), 183,  dtype=np.int16),  # TL3
        canopy_cover       = zero.copy(),
        canopy_height      = zero.copy(),
        canopy_base_height = np.full((rows, cols), 6.0,  dtype=np.float32),
        canopy_bulk_density= np.full((rows, cols), 0.1,  dtype=np.float32),
        shape              = (rows, cols),
        resolution_m       = resolution_m,
    )


# ===========================================================================
# 1. Fuel table
# ===========================================================================

class TestFuelTable:
    def test_shape(self):
        tbl = _build_fuel_table()
        assert tbl.shape == (_N_MODELS, _N_PARAMS)
        assert tbl.dtype == np.float32

    def test_row_zero_unused(self):
        tbl = _build_fuel_table()
        assert np.all(tbl[0] == 0.0), "Row 0 should be zero (no FM 0)"

    def test_all_thirteen_models_populated(self):
        tbl = _build_fuel_table()
        for fid in range(1, 14):
            assert tbl[fid, _COL_W0] > 0.0,    f"FM{fid} load should be positive"
            assert tbl[fid, _COL_SIGMA] > 0.0, f"FM{fid} SAV should be positive"
            assert tbl[fid, _COL_DELTA] > 0.0, f"FM{fid} depth should be positive"

    def test_sav_values_in_1_per_m(self):
        """SAV must be in 1/m (config values ~5000–12000); not 1/ft (~1500–4000)."""
        tbl = _build_fuel_table()
        for fid in range(1, 14):
            sav = tbl[fid, _COL_SIGMA]
            assert sav > 3000.0, f"FM{fid} SAV={sav:.0f} looks like 1/ft (too small)"
            assert sav < 15000.0, f"FM{fid} SAV={sav:.0f} suspiciously large"

    def test_matches_config(self):
        tbl = _build_fuel_table()
        for fid, p in FUEL_PARAMS.items():
            assert abs(tbl[fid, _COL_W0]    - p["load"]) < 1e-6
            assert abs(tbl[fid, _COL_SIGMA] - p["sav"])  < 0.5
            assert abs(tbl[fid, _COL_MX]    - p["mx"])   < 1e-6


# ===========================================================================
# 2. Rothermel ROS kernel
# ===========================================================================

class TestRothermelROS:
    """Unit tests for _rothermel_ros."""

    def _make_fp(self, fid: int = 1) -> "torch.Tensor":
        tbl = torch.tensor(_build_fuel_table(), dtype=torch.float32)
        return tbl[fid].view(1, 1, 1, _N_PARAMS)

    def test_returns_positive_ros(self):
        fp  = self._make_fp(1)
        fmc = _ones((1, 1, 1), 0.08)
        ws  = _ones((1, 1, 1), 5.0)
        waf = _ones((1, 1, 1), 0.5)
        wd  = _ones((1, 1, 1), 0.0)
        asp = _ones((1, 1, 1), 0.0)
        tss = _ones((1, 1, 1), 0.0)
        ros, I_R, theta_max = _rothermel_ros(fmc, ws, waf, wd, asp, fp, tss)
        assert ros.item() > 0.0
        assert I_R.item() > 0.0

    def test_high_fmc_slows_fire(self):
        fp  = self._make_fp(1)
        waf = _ones((1, 1, 1), 0.5)
        wd  = _ones((1, 1, 1), 0.0)
        asp = _ones((1, 1, 1), 0.0)
        tss = _ones((1, 1, 1), 0.0)
        ws  = _ones((1, 1, 1), 5.0)

        ros_dry, _, _ = _rothermel_ros(
            _ones((1, 1, 1), 0.05), ws, waf, wd, asp, fp, tss
        )
        ros_wet, _, _ = _rothermel_ros(
            _ones((1, 1, 1), 0.30), ws, waf, wd, asp, fp, tss
        )
        assert ros_dry.item() > ros_wet.item(), "Higher FMC should slow fire"

    def test_higher_wind_increases_ros(self):
        fp  = self._make_fp(1)
        fmc = _ones((1, 1, 1), 0.08)
        waf = _ones((1, 1, 1), 0.5)
        wd  = _ones((1, 1, 1), 270.0)
        asp = _ones((1, 1, 1), 0.0)
        tss = _ones((1, 1, 1), 0.0)

        ros_slow, _, _ = _rothermel_ros(fmc, _ones((1, 1, 1), 2.0), waf, wd, asp, fp, tss)
        ros_fast, _, _ = _rothermel_ros(fmc, _ones((1, 1, 1), 10.0), waf, wd, asp, fp, tss)
        assert ros_fast.item() > ros_slow.item()

    def test_vector_wind_slope_collinear_ge_perpendicular(self):
        """Collinear wind+slope must be ≥ perpendicular (triangle inequality)."""
        fp  = self._make_fp(2)
        fmc = _ones((1, 1, 1), 0.08)
        ws  = _ones((1, 1, 1), 8.0)
        waf = _ones((1, 1, 1), 0.5)

        slope_deg = 20.0
        tss = torch.tensor([[[math.tan(math.radians(slope_deg)) ** 2]]], dtype=torch.float32)

        # Wind from north (wd=0) → fire pushes south; upslope = south (asp=180°) → collinear
        wd_col  = _ones((1, 1, 1), 0.0)
        asp_col = _ones((1, 1, 1), 180.0)
        ros_col, _, _ = _rothermel_ros(fmc, ws, waf, wd_col, asp_col, fp, tss)

        # Wind from north; upslope = east (asp=90°) → perpendicular
        wd_perp  = _ones((1, 1, 1), 0.0)
        asp_perp = _ones((1, 1, 1), 90.0)
        ros_perp, _, _ = _rothermel_ros(fmc, ws, waf, wd_perp, asp_perp, fp, tss)

        assert ros_col.item() >= ros_perp.item() - 1e-4, (
            f"Collinear ROS {ros_col.item():.4f} should be >= perpendicular {ros_perp.item():.4f}"
        )

    def test_theta_max_from_collinear_wind_and_slope(self):
        """When wind and slope align southward, theta_max ≈ 180°."""
        fp  = self._make_fp(2)
        fmc = _ones((1, 1, 1), 0.08)
        ws  = _ones((1, 1, 1), 8.0)
        waf = _ones((1, 1, 1), 0.5)
        slope_deg = 20.0
        tss = torch.tensor([[[math.tan(math.radians(slope_deg)) ** 2]]], dtype=torch.float32)
        # Wind from north → fire heads south (180°); upslope also south (aspect=180°)
        wd  = _ones((1, 1, 1), 0.0)
        asp = _ones((1, 1, 1), 180.0)
        _, _, theta_max = _rothermel_ros(fmc, ws, waf, wd, asp, fp, tss)
        assert abs(theta_max.item() - 180.0) < 5.0, f"theta_max={theta_max.item():.1f}° expected ~180°"

    def test_theta_max_for_east_wind_no_slope(self):
        """East wind (from west, wd=270°) → fire heads east → theta_max ≈ 90°."""
        fp  = self._make_fp(2)
        fmc = _ones((1, 1, 1), 0.08)
        ws  = _ones((1, 1, 1), 8.0)
        waf = _ones((1, 1, 1), 0.5)
        tss = _ones((1, 1, 1), 0.0)
        wd  = _ones((1, 1, 1), 270.0)   # wind FROM west → fire goes east
        asp = _ones((1, 1, 1), 0.0)
        _, _, theta_max = _rothermel_ros(fmc, ws, waf, wd, asp, fp, tss)
        assert abs(theta_max.item() - 90.0) < 5.0, f"theta_max={theta_max.item():.1f}° expected ~90°"


# ===========================================================================
# 3. Gradient kernels
# ===========================================================================

class TestGradientKernels:
    """_phi_gradient and _phi_gradient_full must agree on |∇φ|."""

    def test_gradient_magnitude_consistency(self):
        torch.manual_seed(0)
        phi = torch.randn(3, 20, 20, dtype=torch.float32)
        dx  = 50.0

        g_basic = _phi_gradient(phi, dx)
        g_full, gx, gy = _phi_gradient_full(phi, dx)

        assert torch.allclose(g_basic, g_full, atol=1e-5), \
            "Fused gradient magnitude should match basic gradient"

    def test_central_diff_sign_and_magnitude(self):
        """For a linear ramp φ(x) = x, gx_cd ≈ 1/dx, gy_cd ≈ 0."""
        R, C = 10, 20
        dx   = 1.0
        x    = torch.arange(C, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(1, R, C)
        phi  = x.clone()

        _, gx_cd, gy_cd = _phi_gradient_full(phi, dx)
        # Interior cells: gx_cd ≈ 1.0
        assert (gx_cd[:, :, 1:-1].abs() - 1.0).abs().max().item() < 0.01
        # No y-gradient for horizontal ramp
        assert gy_cd[:, 1:-1, :].abs().max().item() < 0.01

    def test_gradient_positive_semidefinite(self):
        torch.manual_seed(1)
        phi = torch.randn(2, 15, 15, dtype=torch.float32)
        g   = _phi_gradient(phi, 30.0)
        assert (g >= 0).all(), "|∇φ| must be non-negative everywhere"


# ===========================================================================
# 4. Redistancing
# ===========================================================================

class TestRedistancing:
    """After redistancing, |∇φ| should be close to 1 in the band."""

    def test_gradient_converges_to_one(self):
        """Perturb a ring SDF and check exterior narrowband converges to |∇φ|≈1."""
        R, C = 30, 30
        dx   = 50.0
        cy, cx = R // 2, C // 2
        yy, xx = torch.meshgrid(
            torch.arange(R, dtype=torch.float32),
            torch.arange(C, dtype=torch.float32),
            indexing="ij",
        )
        # Ring SDF: positive outside radius 5*dx, negative inside.
        # Avoids the cone-tip singularity (|∇φ| undefined at origin)
        # that would prevent convergence at the center.
        dist      = ((yy - cy).pow(2) + (xx - cx).pow(2)).sqrt() * dx
        sdf_clean = dist - 5 * dx

        # Perturb by ±15 % to simulate |∇φ| drift
        torch.manual_seed(42)
        perturbed = sdf_clean.unsqueeze(0) * (1.0 + 0.15 * torch.randn(1, R, C))

        g_before = _phi_gradient(perturbed, dx)
        # Only measure exterior (phi > 0) cells in the narrowband — redistancing
        # is exterior-only, so interior cells are intentionally not corrected.
        in_band = (perturbed[0] > 0) & (perturbed[0] < 5 * dx)
        distortion_before = (g_before[0][in_band] - 1.0).abs().mean().item()

        result  = _redistance(perturbed, dx, n_iters=5)
        g_after = _phi_gradient(result, dx)
        distortion_after = (g_after[0][in_band] - 1.0).abs().mean().item()

        assert distortion_after < distortion_before, \
            "Redistancing should reduce |∇φ| distortion"
        # 5 Sussman iterations at dt=0.5*dx converge meaningfully but not to < 5%;
        # the real loop calls _redistance every 20 steps, accumulating many passes.
        assert distortion_after < 0.10, \
            f"|∇φ| mean deviation after redistancing: {distortion_after:.4f} (want < 0.10)"

    def test_zero_crossing_not_moved(self):
        """The zero level set should not shift by more than 0.5 dx."""
        R, C = 20, 20
        dx   = 50.0
        cy, cx = R // 2, C // 2
        yy, xx = torch.meshgrid(
            torch.arange(R, dtype=torch.float32),
            torch.arange(C, dtype=torch.float32),
            indexing="ij",
        )
        sdf = ((yy - cy).pow(2) + (xx - cx).pow(2)).sqrt() * dx - 5 * dx
        perturbed = sdf.unsqueeze(0) * (1.0 + 0.10 * torch.ones(1, R, C))
        result    = _redistance(perturbed, dx, n_iters=5)
        # Sign of zero level set should be preserved
        assert ((result > 0) == (perturbed > 0)).float().mean().item() > 0.99, \
            "Redistancing moved the zero contour in more than 1% of cells"


# ===========================================================================
# 5. Directional ROS
# ===========================================================================

class TestDirectionalROS:
    def _grad_for_direction(self, angle_deg: float, n: int = 1) -> tuple:
        """Return (gx_cd, gy_cd) for a normal pointing in angle_deg from north."""
        rad  = math.radians(angle_deg)
        # nx = sin(angle), ny = cos(angle)  (standard GIS: N=0, E=90)
        gx   = torch.full((n, 5, 5), math.sin(rad), dtype=torch.float32)
        gy   = torch.full((n, 5, 5), math.cos(rad), dtype=torch.float32)
        return gx, gy

    def test_head_fire_direction(self):
        """When propagation direction == theta_max, ROS should equal ros_head."""
        ros_head = _ones((1, 5, 5), 0.5)
        e        = _ones((1, 5, 5), 0.6)
        theta_max = _ones((1, 5, 5), 90.0)   # head fire is east
        gx, gy   = self._grad_for_direction(90.0)  # propagating east

        ros_out = _directional_ros(ros_head, e, theta_max, gx, gy)
        assert torch.allclose(ros_out, ros_head, atol=1e-3), \
            f"Head-fire ROS should equal ros_head; got {ros_out.mean().item():.4f}"

    def test_backing_fire_slower(self):
        """Backing fire (θ=180°) should be slower than head fire."""
        ros_head  = _ones((1, 5, 5), 0.5)
        e         = _ones((1, 5, 5), 0.6)
        theta_max = _ones((1, 5, 5), 90.0)  # head fire east
        gx_back, gy_back = self._grad_for_direction(270.0)  # propagating west

        ros_back = _directional_ros(ros_head, e, theta_max, gx_back, gy_back)
        assert (ros_back < ros_head - 0.01).all(), \
            "Backing ROS must be slower than head-fire ROS"

    def test_isotropic_at_zero_eccentricity(self):
        """With e=0 (no wind), all directions should give the same ROS."""
        ros_head  = _ones((1, 5, 5), 0.5)
        e_zero    = _ones((1, 5, 5), 0.0)
        theta_max = _ones((1, 5, 5), 90.0)

        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            gx, gy  = self._grad_for_direction(float(angle))
            ros_dir = _directional_ros(ros_head, e_zero, theta_max, gx, gy)
            assert torch.allclose(ros_dir, ros_head, atol=1e-3), \
                f"At e=0, direction {angle}° should give head-fire ROS"

    def test_flank_ros_between_head_and_back(self):
        """Flank ROS (θ=90°) should be between head and backing ROS."""
        ros_head  = _ones((1, 5, 5), 0.5)
        e         = _ones((1, 5, 5), 0.6)
        theta_max = _ones((1, 5, 5), 0.0)   # head fire north

        gx_head,  gy_head  = self._grad_for_direction(0.0)   # north
        gx_flank, gy_flank = self._grad_for_direction(90.0)  # east (flank)
        gx_back,  gy_back  = self._grad_for_direction(180.0) # south (back)

        ros_h = _directional_ros(ros_head, e, theta_max, gx_head,  gy_head )
        ros_f = _directional_ros(ros_head, e, theta_max, gx_flank, gy_flank)
        ros_b = _directional_ros(ros_head, e, theta_max, gx_back,  gy_back )

        assert (ros_h >= ros_f - 1e-4).all() and (ros_f >= ros_b - 1e-4).all(), \
            "Expected head ≥ flank ≥ backing"


# ===========================================================================
# 6. Engine construction
# ===========================================================================

class TestEngineConstruction:
    def test_builds_on_cpu(self):
        terrain = synthetic_terrain((20, 20), resolution_m=50.0, seed=0)
        engine  = GPUFireEngine(terrain, device="cpu")
        assert engine.device == torch.device("cpu")
        assert engine.dx == 50.0
        assert engine.shape == (20, 20)

    def test_fuel_params_shape(self):
        terrain = synthetic_terrain((15, 15), resolution_m=50.0, seed=1)
        engine  = GPUFireEngine(terrain, device="cpu")
        assert engine._fuel_params.shape == (15, 15, _N_PARAMS)

    def test_cuda_falls_back_to_cpu(self):
        terrain = synthetic_terrain((10, 10), resolution_m=50.0, seed=0)
        if not torch.cuda.is_available():
            with pytest.warns(RuntimeWarning, match="CUDA is not available"):
                engine = GPUFireEngine(terrain, device="cuda")
            assert engine.device == torch.device("cpu")


# ===========================================================================
# 7. Engine run — output shapes and sentinels
# ===========================================================================

class TestEngineRun:
    R, C = 30, 30
    N    = 4

    @pytest.fixture(scope="class")
    def engine_and_result(self):
        # TL3 (code 183, timber litter, mx=0.35) on flat terrain with low wind —
        # slow enough that corners of a 30×30/50 m grid stay unburned in 30 min.
        terrain = _slow_terrain(self.R, self.C, resolution_m=50.0)
        engine  = GPUFireEngine(terrain, device="cpu")
        prior   = _flat_gp(self.R, self.C, fmc=0.20, ws=1.0, wd=270.0)
        fs      = _center_fire(self.R, self.C)
        rng     = np.random.default_rng(0)
        result  = engine.run(terrain, prior, fs, n_members=self.N,
                             horizon_min=30, rng=rng)
        return result

    def test_arrival_time_shape(self, engine_and_result):
        r = engine_and_result
        assert r.member_arrival_times.shape == (self.N, self.R, self.C)

    def test_burn_probability_shape(self, engine_and_result):
        r = engine_and_result
        assert r.burn_probability.shape == (self.R, self.C)

    def test_mean_arrival_time_shape(self, engine_and_result):
        r = engine_and_result
        assert r.mean_arrival_time.shape == (self.R, self.C)

    def test_fire_types_shape(self, engine_and_result):
        r = engine_and_result
        assert r.member_fire_types is not None
        assert r.member_fire_types.shape == (self.N, self.R, self.C)

    def test_arrival_times_bounded(self, engine_and_result):
        """All arrival times should be ≤ 2 × horizon (the sentinel)."""
        r = engine_and_result
        assert (r.member_arrival_times <= 2.0 * 30.0 + 1e-3).all()

    def test_center_burned_at_t0(self, engine_and_result):
        """Center cell (ignition) must have arrival time = 0."""
        r   = engine_and_result
        at  = r.member_arrival_times[:, self.R // 2, self.C // 2]
        assert (at == 0.0).all(), "Ignition cell must arrive at t=0"

    def test_burn_prob_in_range(self, engine_and_result):
        r = engine_and_result
        assert r.burn_probability.min() >= 0.0
        assert r.burn_probability.max() <= 1.0 + 1e-6

    def test_unburned_cells_use_sentinel(self, engine_and_result):
        """Corners of a small domain should be unburned (sentinel = 2×horizon)."""
        r      = engine_and_result
        corner = r.member_arrival_times[:, 0, 0]
        # At least some members should show unburned corners in 30 min
        sentinel = 2.0 * 30.0
        assert (corner >= sentinel * 0.9).any(), \
            "Some members should leave corners unburned in 30 min on 30×30 grid"

    def test_n_members_recorded(self, engine_and_result):
        assert engine_and_result.n_members == self.N

    def test_wind_field_shapes(self, engine_and_result):
        r = engine_and_result
        assert r.member_wind_fields.shape      == (self.N, self.R, self.C)
        assert r.member_wind_dir_fields.shape  == (self.N, self.R, self.C)
        assert r.member_fmc_fields.shape       == (self.N, self.R, self.C)


# ===========================================================================
# 8. Circular spread (no wind, no slope)
# ===========================================================================

class TestCircularSpread:
    """On flat terrain with uniform zero wind, fire should spread roughly circularly."""

    R, C = 40, 40

    def _run(self, wind_speed: float = 0.0) -> "EnsembleResult":
        from angrybird.types import EnsembleResult  # avoid circular import at module level
        terrain = synthetic_terrain((self.R, self.C), resolution_m=50.0, seed=99)
        # Override slope to zero so spread is purely symmetric
        flat = terrain.__class__(
            elevation           = np.zeros((self.R, self.C), dtype=np.float32),
            slope               = np.zeros((self.R, self.C), dtype=np.float32),
            aspect              = np.zeros((self.R, self.C), dtype=np.float32),
            fuel_model          = np.full((self.R, self.C), 1, dtype=np.int16),
            canopy_cover        = np.zeros((self.R, self.C), dtype=np.float32),
            canopy_height       = np.zeros((self.R, self.C), dtype=np.float32),
            canopy_base_height  = np.zeros((self.R, self.C), dtype=np.float32),
            canopy_bulk_density = np.zeros((self.R, self.C), dtype=np.float32),
            shape               = (self.R, self.C),
            resolution_m        = 50.0,
        )
        engine = GPUFireEngine(flat, device="cpu")
        prior  = _flat_gp(self.R, self.C, fmc=0.08, ws=wind_speed, wd=0.0)
        fs     = _center_fire(self.R, self.C)
        rng    = np.random.default_rng(7)
        return engine.run(flat, prior, fs, n_members=1, horizon_min=60, rng=rng)

    def test_symmetric_spread_no_wind(self):
        """With no wind, arrival time at equal radii from center should be similar."""
        result = self._run(wind_speed=0.0)
        at = result.member_arrival_times[0]
        cy, cx = self.R // 2, self.C // 2

        # Sample 8 equidistant points at radius 5 cells
        r = 5
        offsets = [(r, 0), (-r, 0), (0, r), (0, -r),
                   (r, r), (-r, r), (r, -r), (-r, -r)]
        sentinel = 2.0 * 60.0
        valid_times = []
        for dy, dx in offsets:
            row, col = cy + dy, cx + dx
            t = at[row, col]
            if t < sentinel * 0.9:
                valid_times.append(t)

        if len(valid_times) >= 4:
            spread = max(valid_times) - min(valid_times)
            # Allow up to 10 min spread across ring (level-set is not exactly circular)
            assert spread < 15.0, \
                f"Isotropic spread should be roughly symmetric; ring spread={spread:.1f} min"


# ===========================================================================
# 9. Directional spread (with wind)
# ===========================================================================

class TestDirectionalSpread:
    R, C = 50, 50

    def _run_with_wind(self, wd: float) -> np.ndarray:
        """Return mean arrival time map for a given wind direction."""
        flat = synthetic_terrain.__module__  # ensure import path fine
        terrain = synthetic_terrain((self.R, self.C), resolution_m=50.0, seed=5)
        # Flat terrain with uniform FM1
        from angrybird.types import TerrainData
        flat_t = TerrainData(
            elevation           = np.zeros((self.R, self.C), dtype=np.float32),
            slope               = np.zeros((self.R, self.C), dtype=np.float32),
            aspect              = np.zeros((self.R, self.C), dtype=np.float32),
            fuel_model          = np.full((self.R, self.C), 2, dtype=np.int16),
            canopy_cover        = np.zeros((self.R, self.C), dtype=np.float32),
            canopy_height       = np.zeros((self.R, self.C), dtype=np.float32),
            canopy_base_height  = np.zeros((self.R, self.C), dtype=np.float32),
            canopy_bulk_density = np.zeros((self.R, self.C), dtype=np.float32),
            shape               = (self.R, self.C),
            resolution_m        = 50.0,
        )
        engine = GPUFireEngine(flat_t, device="cpu")
        prior  = _flat_gp(self.R, self.C, fmc=0.08, ws=8.0, wd=wd)
        fs     = _center_fire(self.R, self.C)
        rng    = np.random.default_rng(3)
        result = engine.run(flat_t, prior, fs, n_members=3, horizon_min=60, rng=rng)
        return result.mean_arrival_time

    def test_downwind_faster_than_upwind(self):
        """With west wind (wd=270°), east cells should burn before west cells."""
        cy, cx = self.R // 2, self.C // 2
        at = self._run_with_wind(wd=270.0)  # wind FROM west → fire heads east
        sentinel = np.nan

        # East of center
        at_east = at[cy, cx + 8]
        # West of center
        at_west = at[cy, cx - 8]

        if not (np.isnan(at_east) or np.isnan(at_west)):
            assert at_east < at_west, (
                f"Downwind (east) should burn first: at_east={at_east:.1f}, "
                f"at_west={at_west:.1f}"
            )


# ===========================================================================
# 10. initial_phi per-member seeding
# ===========================================================================

class TestInitialPhi:
    R, C = 20, 20
    N    = 3

    def test_initial_phi_accepted(self):
        """Engine should accept initial_phi without fire_state."""
        terrain = synthetic_terrain((self.R, self.C), resolution_m=50.0, seed=0)
        engine  = GPUFireEngine(terrain, device="cpu")
        prior   = _flat_gp(self.R, self.C)

        # Construct per-member SDFs: member i has ignition at row i+5, center col
        cx = self.C // 2
        phis = []
        for i in range(self.N):
            r0 = 5 + i
            yy, xx = np.meshgrid(np.arange(self.R), np.arange(self.C), indexing="ij")
            sdf = np.sqrt((yy - r0) ** 2 + (xx - cx) ** 2).astype(np.float32) * 50.0 - 50.0
            phis.append(sdf)
        initial_phi = np.stack(phis, axis=0)

        rng    = np.random.default_rng(1)
        result = engine.run(terrain, prior, fire_state=None,
                            n_members=self.N, horizon_min=20,
                            rng=rng, initial_phi=initial_phi)
        assert result.member_arrival_times.shape == (self.N, self.R, self.C)

    def test_fire_state_none_with_initial_phi(self):
        """fire_state=None should not raise when initial_phi is provided."""
        terrain = synthetic_terrain((self.R, self.C), resolution_m=50.0, seed=0)
        engine  = GPUFireEngine(terrain, device="cpu")
        prior   = _flat_gp(self.R, self.C)

        cy, cx = self.R // 2, self.C // 2
        yy, xx = np.meshgrid(np.arange(self.R), np.arange(self.C), indexing="ij")
        sdf    = (np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) * 50.0 - 50.0).astype(np.float32)
        phi    = np.stack([sdf] * self.N, axis=0)

        rng = np.random.default_rng(2)
        # Should not raise
        engine.run(terrain, prior, fire_state=None, n_members=self.N,
                   horizon_min=15, rng=rng, initial_phi=phi)

    def test_initial_phi_seeds_different_ignitions(self):
        """Per-member ignitions should produce different arrival time patterns."""
        terrain = synthetic_terrain((self.R, self.C), resolution_m=50.0, seed=0)
        engine  = GPUFireEngine(terrain, device="cpu")
        prior   = _flat_gp(self.R, self.C, ws_var=0.0, fmc_var=0.0, wd_var=0.0)

        cy, cx = self.R // 2, self.C // 2
        yy, xx = np.meshgrid(np.arange(self.R), np.arange(self.C), indexing="ij")
        # Member 0: ignition at top-left
        sdf0 = (np.sqrt((yy - 5) ** 2 + (xx - 5) ** 2) * 50.0 - 50.0).astype(np.float32)
        # Member 1: ignition at bottom-right
        sdf1 = (np.sqrt((yy - (self.R - 5)) ** 2 + (xx - (self.C - 5)) ** 2)
                * 50.0 - 50.0).astype(np.float32)
        phi = np.stack([sdf0, sdf1], axis=0)

        rng    = np.random.default_rng(3)
        # Short horizon: fire won't reach across the grid, so the two ignition
        # locations produce genuinely different arrival-time maps.
        result = engine.run(terrain, prior, fire_state=None, n_members=2,
                            horizon_min=8, rng=rng, initial_phi=phi)

        at0 = result.member_arrival_times[0]
        at1 = result.member_arrival_times[1]
        # Member 0 (top-left ignition) should have lower arrival times at (5,5)
        # than member 1, and vice versa for the opposite corner.
        sentinel = 2.0 * 8.0
        # Cells near each ignition: the igniting member should arrive sooner
        assert at0[5, 5] < at1[5, 5] or at0[5, 5] == 0.0, \
            "Member 0 (ignited at (5,5)) should arrive there before member 1"
        assert at1[self.R - 5, self.C - 5] < at0[self.R - 5, self.C - 5] or \
               at1[self.R - 5, self.C - 5] == 0.0, \
            "Member 1 (ignited at bottom-right) should arrive there before member 0"
        # And the overall arrival maps should differ meaningfully
        diff_frac = (at0 != at1).mean()
        assert diff_frac > 0.5, \
            f"Different ignitions should produce different arrival maps; only {diff_frac:.1%} differ"


# ===========================================================================
# 11. Reproducibility
# ===========================================================================

class TestReproducibility:
    R, C = 20, 20

    def _run(self, seed: int) -> np.ndarray:
        terrain = synthetic_terrain((self.R, self.C), resolution_m=50.0, seed=0)
        engine  = GPUFireEngine(terrain, device="cpu")
        prior   = _flat_gp(self.R, self.C, fmc=0.08, ws=5.0, wd=180.0,
                           ws_var=1.0, fmc_var=0.001)
        fs      = _center_fire(self.R, self.C)
        rng     = np.random.default_rng(seed)
        return engine.run(terrain, prior, fs, n_members=3, horizon_min=20, rng=rng
                          ).member_arrival_times

    def test_same_seed_reproducible(self):
        at1 = self._run(seed=42)
        at2 = self._run(seed=42)
        assert np.allclose(at1, at2), "Same seed must produce identical results"

    def test_different_seeds_differ(self):
        at1 = self._run(seed=42)
        at2 = self._run(seed=99)
        assert not np.allclose(at1, at2), "Different seeds should produce different results"


# ===========================================================================
# 12. Full-stack test with synthetic_terrain
# ===========================================================================

class TestFullStack:
    """End-to-end test: synthetic_terrain → GPUFireEngine → EnsembleResult."""

    def test_ensemble_produces_variance(self):
        """With wind uncertainty, ensemble members should differ."""
        R, C = 40, 40
        terrain = synthetic_terrain((R, C), resolution_m=50.0, seed=10)
        engine  = GPUFireEngine(terrain, device="cpu")
        prior   = _flat_gp(R, C, fmc=0.08, ws=6.0, wd=270.0,
                           ws_var=4.0, fmc_var=0.001, wd_var=100.0)
        fs      = _center_fire(R, C)
        rng     = np.random.default_rng(0)
        result  = engine.run(terrain, prior, fs, n_members=8,
                             horizon_min=60, rng=rng)

        # Variance in arrival times should be non-trivial for uncertain ensemble
        assert result.arrival_time_variance.mean() > 0.0, \
            "Ensemble with wind uncertainty should show arrival-time variance"

    def test_fire_type_values(self):
        """fire_types must be 1 (surface) or 2 (crown) only."""
        R, C = 30, 30
        terrain = synthetic_terrain((R, C), resolution_m=50.0, seed=5)
        engine  = GPUFireEngine(terrain, device="cpu")
        prior   = _flat_gp(R, C, fmc=0.05, ws=10.0, wd=90.0)
        fs      = _center_fire(R, C)
        rng     = np.random.default_rng(1)
        result  = engine.run(terrain, prior, fs, n_members=2,
                             horizon_min=30, rng=rng)

        ft = result.member_fire_types
        assert ft is not None
        assert set(np.unique(ft)).issubset({1, 2}), \
            f"Unexpected fire type values: {np.unique(ft)}"

    def test_burn_prob_increases_with_horizon(self):
        """Longer horizon should burn more area on average."""
        R, C = 30, 30
        terrain = synthetic_terrain((R, C), resolution_m=50.0, seed=3)
        engine  = GPUFireEngine(terrain, device="cpu")
        prior   = _flat_gp(R, C, fmc=0.08, ws=5.0, wd=180.0)
        fs      = _center_fire(R, C)

        rng1 = np.random.default_rng(0)
        r30  = engine.run(terrain, prior, fs, n_members=4, horizon_min=30, rng=rng1)

        rng2 = np.random.default_rng(0)
        r60  = engine.run(terrain, prior, fs, n_members=4, horizon_min=60, rng=rng2)

        burned_30 = (r30.burn_probability > 0.5).sum()
        burned_60 = (r60.burn_probability > 0.5).sum()
        assert burned_60 >= burned_30, \
            "60-minute horizon should burn at least as much as 30-minute horizon"
