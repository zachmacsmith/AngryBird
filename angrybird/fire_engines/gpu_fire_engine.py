"""
GPU-accelerated ensemble fire engine using PyTorch.

Implements full Rothermel surface ROS + Van Wagner crown fire +
level-set propagation with CFL adaptive timestepping, as specified in
docs/GPU Acceleration.md.

Satisfies FireEngineProtocol — drop-in replacement for SimpleFire:
    run(terrain, gp_prior, fire_state, n_members, horizon_min, rng) → EnsembleResult

Falls back to CPU transparently when CUDA is unavailable.
All GPU allocation and computation is internal.  Boundary: numpy in, numpy out.

Arrival times in EnsembleResult are in **minutes**.  Unburned cells use the
finite sentinel 2 × horizon_min (PotentialBugs1 §3) in member_arrival_times.
mean_arrival_time uses NaN for fully-unburned cells (for visualization only).

Performance design
------------------
Three optimisations versus the naïve implementation:

1. Static hoisting — fmc, ws, wd are sampled once per run() call and never
   change within the simulation loop.  All Rothermel quantities (ros_s,
   theta_max, eccentricity, crown-fire ROS, intensity scaling factor, I_crit)
   are therefore precomputed before the loop.  The hot loop contains only the
   gradient computation, ellipse projection, crown check, and φ advance.

2. Fused gradient — _phi_gradient_full() computes both the Godunov upwind
   magnitude |∇φ| (for the level-set advance) and the central-difference
   components (gx, gy) needed for the outward-normal direction in a single
   pass over φ with 4 roll+clone operations.  The previous design called
   _phi_gradient and _phi_grad_components separately (8 roll+clones per step).

3. Fixed CFL dt — ros_s and ros_crown are static, so the maximum possible
   ROS is known before the loop.  A single GPU→CPU sync replaces the per-step
   max().item() sync, eliminating ~500 serialisation points per run.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from angrybird.config import (
    CANOPY_CBH_M,
    CANOPY_CBD_KGM3,
    CANOPY_COVER_FRACTION,
    FMC_MAX_FRACTION,
    FMC_MIN_FRACTION,
    FUEL_MINERAL_CONTENT,
    FUEL_MINERAL_SILICA_FREE,
    FUEL_PARAMS,
    FUEL_PARTICLE_DENSITY,
    SB40_FUEL_PARAMS,
    WIND_SPEED_MAX_MS,
    WIND_SPEED_MIN_MS,
)
from angrybird.types import EnsembleResult, GPPrior, TerrainData

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Unified table covers Anderson-13 (codes 1-13) AND SB40 (codes 91-204).
# Row index = fuel code; rows 0 and 14-90 are zero (unused).
_N_MODELS = 205   # codes 0-204; only occupied rows contribute
_N_PARAMS = 8
_COL_W0, _COL_SIGMA, _COL_DELTA = 0, 1, 2
_COL_MX, _COL_HEAT, _COL_RHOP   = 3, 4, 5
_COL_ST, _COL_SE                  = 6, 7

_CBH_PROXY = CANOPY_CBH_M
_CBD_PROXY = CANOPY_CBD_KGM3
_CC_PROXY  = CANOPY_COVER_FRACTION

_REDISTANCE_EVERY = 20  # redistancing pass every N CFL steps

_RAD2DEG = 180.0 / math.pi
_DEG2RAD = math.pi / 180.0


# ---------------------------------------------------------------------------
# Fuel table
# ---------------------------------------------------------------------------

def _array_from_lookup(
    fuel_model: np.ndarray, lookup: dict[int, float]
) -> np.ndarray:
    out = np.zeros(fuel_model.shape, dtype=np.float32)
    for fid, val in lookup.items():
        out[fuel_model == fid] = float(val)
    return out


def _build_fuel_table() -> np.ndarray:
    """
    (205, 8) float32 parameter table indexed by fuel model code.

    Anderson-13 (codes 1-13) use the simple {"load", "sav", "depth", "mx", "h"}
    schema from FUEL_PARAMS.

    Scott & Burgan 40 (codes 91-204) use the per-class schema from SB40_FUEL_PARAMS:
    load_1h/10h/100h/live_herb/live_woody, sav_1h/sav_10h/sav_100h/sav_live_herb/
    sav_live_woody, depth, mx, h.  These are aggregated to a single-sigma Rothermel
    input using the area-weighted characteristic SAV from Rothermel (1972):

        sigma = Σ(w_i · σ_i²) / Σ(w_i · σ_i)   (dead fuel classes)
        w0    = Σ w_i                              (total dead load)

    Live fuels (herb, woody) are omitted from the simplified dead-fuel Rothermel.
    """
    tbl = np.zeros((_N_MODELS, _N_PARAMS), dtype=np.float32)

    # ── Anderson-13 (simple format) ───────────────────────────────────────
    for fid, p in FUEL_PARAMS.items():
        if not (0 <= fid < _N_MODELS):
            continue
        tbl[fid, _COL_W0]    = p["load"]
        tbl[fid, _COL_SIGMA] = p["sav"]
        tbl[fid, _COL_DELTA] = p["depth"]
        tbl[fid, _COL_MX]    = p["mx"]
        tbl[fid, _COL_HEAT]  = p["h"]
        tbl[fid, _COL_RHOP]  = FUEL_PARTICLE_DENSITY
        tbl[fid, _COL_ST]    = FUEL_MINERAL_CONTENT
        tbl[fid, _COL_SE]    = FUEL_MINERAL_SILICA_FREE

    # ── SB40 (multi-class format) ─────────────────────────────────────────
    for fid, p in SB40_FUEL_PARAMS.items():
        if not (0 <= fid < _N_MODELS):
            continue
        w_1h   = float(p.get("load_1h",   0.0))
        w_10h  = float(p.get("load_10h",  0.0))
        w_100h = float(p.get("load_100h", 0.0))
        s_1h   = float(p.get("sav_1h",    0.0))
        s_10h  = float(p.get("sav_10h",   0.0))
        s_100h = float(p.get("sav_100h",  0.0))

        w0 = w_1h + w_10h + w_100h
        # Area-weighted characteristic SAV (Rothermel 1972, eq. 31-33):
        # σ̄ = Σ(w_i σ_i²) / Σ(w_i σ_i)
        num   = w_1h * s_1h**2 + w_10h * s_10h**2 + w_100h * s_100h**2
        denom = w_1h * s_1h    + w_10h * s_10h    + w_100h * s_100h
        sigma = num / denom if denom > 0 else 0.0

        tbl[fid, _COL_W0]    = w0
        tbl[fid, _COL_SIGMA] = sigma
        tbl[fid, _COL_DELTA] = float(p.get("depth", 0.0))
        tbl[fid, _COL_MX]    = float(p.get("mx",    0.0))
        tbl[fid, _COL_HEAT]  = float(p.get("h",     0.0))
        tbl[fid, _COL_RHOP]  = FUEL_PARTICLE_DENSITY
        tbl[fid, _COL_ST]    = FUEL_MINERAL_CONTENT
        tbl[fid, _COL_SE]    = FUEL_MINERAL_SILICA_FREE

    return tbl


# ---------------------------------------------------------------------------
# Rothermel surface ROS kernel
# ---------------------------------------------------------------------------

def _rothermel_ros(
    fmc: "torch.Tensor",          # (N, R, C)
    ws: "torch.Tensor",           # (N, R, C) m/s at 10 m
    wind_adj: "torch.Tensor",     # (1, R, C) midflame wind adjustment factor
    wind_dir: "torch.Tensor",     # (N, R, C) degrees — direction wind blows FROM
    aspect_deg: "torch.Tensor",   # (1, R, C) degrees from north — uphill direction
    fp: "torch.Tensor",           # (1, R, C, N_PARAMS)
    tan_slope_sq: "torch.Tensor", # (1, R, C)
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Rothermel (1972) surface ROS with vector wind+slope combination.

    Wind and slope contributions are resolved as 2-D vectors before being
    summed, so the ROS is correct when wind is not aligned with the upslope
    direction.  Scalar addition over-predicts ROS and always assigns the
    head-fire direction to the wind direction alone.

    Called once per run() — all inputs are static within a simulation.

    Returns
    -------
    ros       : (N, R, C) m/s — head-fire ROS in the resultant direction
    I_R       : (N, R, C) kW/m² — reaction intensity (diagnostics only)
    theta_max : (N, R, C) degrees — resultant head-fire direction from north
    """
    w0    = fp[..., _COL_W0]
    sigma = fp[..., _COL_SIGMA]
    delta = fp[..., _COL_DELTA]
    mx    = fp[..., _COL_MX]
    heat  = fp[..., _COL_HEAT]
    rho_p = fp[..., _COL_RHOP]
    st    = fp[..., _COL_ST]
    se    = fp[..., _COL_SE]

    # Rothermel (1972) empirical constants are calibrated for σ in 1/ft.
    # Our fuel table stores σ in 1/m (SI).  Convert at the boundary so all
    # coefficient formulas (beta_op, A, gamma_max, xi, phi_w, ε) use 1/ft.
    _INV_FT_TO_INV_M = 3.28084
    sigma_ft = sigma / _INV_FT_TO_INV_M

    beta       = w0 / (delta * rho_p + 1e-10)
    beta_op    = 3.348 * sigma_ft.pow(-0.8189)
    beta_ratio = beta / (beta_op + 1e-10)

    A         = 133.0 * sigma_ft.pow(-0.7913)
    gamma_max = sigma_ft.pow(1.5) / (495.0 + 0.0594 * sigma_ft.pow(1.5))
    gamma     = gamma_max * beta_ratio.pow(A) * (A * (1.0 - beta_ratio)).exp()

    wn    = w0 * (1.0 - st)
    eta_s = (0.174 * se.pow(-0.19)).clamp(max=1.0)
    rm    = fmc / (mx + 1e-10)
    eta_m = (1.0 - 2.59 * rm + 5.11 * rm.pow(2) - 3.52 * rm.pow(3)).clamp(0.0, 1.0)

    I_R = gamma * wn * heat * eta_m * eta_s

    xi = ((0.792 + 0.681 * sigma_ft.pow(0.5)) * (beta + 0.1)).exp() \
         / (192.0 + 0.2595 * sigma_ft)

    # phi_w uses sigma in 1/ft AND wind in ft/min (Rothermel empirical calibration)
    C = 7.47 * (-0.133 * sigma_ft.pow(0.55)).exp()
    B = 0.02526 * sigma_ft.pow(0.54)
    E = 0.715 * (-3.59e-4 * sigma_ft).exp()
    ws_midflame = ws * (60.0 / 0.3048) * wind_adj   # m/s → ft/min
    # NB cells: sigma=0 → beta_ratio→0 → beta_ratio^(-E)=inf; phi_s also inf at β=0.
    # inf×sin(θ)=NaN when sin(θ)=0, propagating through theta_max into ros_dir.
    # Zero out both multipliers for non-burnable cells; ros is nan_to_num'd below.
    phi_w = torch.nan_to_num(C * ws_midflame.pow(B) * beta_ratio.pow(-E), nan=0.0, posinf=0.0)

    phi_s = torch.nan_to_num(5.275 * beta.pow(-0.3) * tan_slope_sq, nan=0.0, posinf=0.0)

    # ── Vector wind + slope combination ───────────────────────────────────
    wind_to_rad = ((wind_dir + 180.0) % 360.0) * _DEG2RAD   # (N,R,C)
    upslope_rad = aspect_deg * _DEG2RAD                       # (1,R,C)

    rx = phi_w * torch.sin(wind_to_rad) + phi_s * torch.sin(upslope_rad)
    ry = phi_w * torch.cos(wind_to_rad) + phi_s * torch.cos(upslope_rad)

    phi_combined = (rx.pow(2) + ry.pow(2) + 1e-10).sqrt()
    theta_max    = (torch.atan2(rx, ry) * _RAD2DEG) % 360.0

    # Q_ig: Rothermel formula gives BTU/lb → convert to J/kg so ROS comes out m/s
    Q_ig  = (250.0 + 1116.0 * fmc) * 2326.0
    eps   = (-138.0 / (sigma_ft + 1e-10)).exp()
    rho_b = w0 / (delta + 1e-10)

    ros = ((I_R * xi * (1.0 + phi_combined)) / (rho_b * eps * Q_ig + 1e-10) / 60.0
           ).clamp(min=0.0)
    # Non-burnable cells (sigma=0, w0=0) produce NaN via 0^inf intermediates.
    # Replace with 0 — no fire spread in non-burnable pixels.
    ros = torch.nan_to_num(ros, nan=0.0)

    return ros, I_R, theta_max


# ---------------------------------------------------------------------------
# Van Wagner crown fire kernel
# ---------------------------------------------------------------------------

def _crown_fire_check(
    surface_intensity: "torch.Tensor",  # (N, R, C) kW/m
    cbh: "torch.Tensor",                # (1, R, C) m
    cbd: "torch.Tensor",                # (1, R, C) kg/m³
    ws: "torch.Tensor",                 # (N, R, C) m/s
    ros_surface: "torch.Tensor",        # (N, R, C) m/s
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Van Wagner (1977) crown fire initiation."""
    I_crit    = (0.01 * cbh * (460.0 + 2590.0)).pow(1.5)
    initiates = surface_intensity > I_crit
    ros_crown = 11.02 * (ws * 3.6).pow(0.854) * cbd.pow(0.19) / 60.0
    ros_final = torch.where(initiates, torch.maximum(ros_surface, ros_crown), ros_surface)
    fire_type = torch.where(
        initiates,
        torch.full_like(ros_surface, 2.0),
        torch.full_like(ros_surface, 1.0),
    )
    return ros_final, fire_type


# ---------------------------------------------------------------------------
# Gradient kernels
# ---------------------------------------------------------------------------

def _phi_gradient(phi: "torch.Tensor", dx: float) -> "torch.Tensor":
    """
    Godunov upwind |∇φ| for an expanding front.
    Used only by _redistance — the main loop uses _phi_gradient_full.
    """
    phi_xp = torch.roll(phi, -1, dims=2).clone(); phi_xp[:, :, -1] = phi[:, :, -1]
    phi_xm = torch.roll(phi,  1, dims=2).clone(); phi_xm[:, :,  0] = phi[:, :,  0]
    phi_yp = torch.roll(phi, -1, dims=1).clone(); phi_yp[:,  -1, :] = phi[:,  -1, :]
    phi_ym = torch.roll(phi,  1, dims=1).clone(); phi_ym[:,   0, :] = phi[:,   0, :]

    Dxm = (phi - phi_xm) / dx;  Dxp = (phi_xp - phi) / dx
    Dym = (phi - phi_ym) / dx;  Dyp = (phi_yp - phi) / dx

    gx = Dxm.clamp(min=0).pow(2) + Dxp.clamp(max=0).pow(2)
    gy = Dym.clamp(min=0).pow(2) + Dyp.clamp(max=0).pow(2)
    return (gx + gy + 1e-10).sqrt()


def _phi_gradient_full(
    phi: "torch.Tensor", dx: float
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Fused gradient: Godunov upwind |∇φ| AND central-difference components.

    Performs 4 roll+clone operations (vs 8 if _phi_gradient and
    _phi_grad_components are called separately), halving the memory traffic
    of the gradient pass in the hot loop.

    Returns
    -------
    grad_mag : (N, R, C) — upwind |∇φ| for the level-set advance
    gx_cd    : (N, R, C) — ∂φ/∂x (central diff, signed, for normal direction)
    gy_cd    : (N, R, C) — ∂φ/∂y (central diff, signed, for normal direction)
    """
    phi_xp = torch.roll(phi, -1, dims=2).clone(); phi_xp[:, :, -1] = phi[:, :, -1]
    phi_xm = torch.roll(phi,  1, dims=2).clone(); phi_xm[:, :,  0] = phi[:, :,  0]
    phi_yp = torch.roll(phi, -1, dims=1).clone(); phi_yp[:,  -1, :] = phi[:,  -1, :]
    phi_ym = torch.roll(phi,  1, dims=1).clone(); phi_ym[:,   0, :] = phi[:,   0, :]

    Dxm = (phi - phi_xm) / dx;  Dxp = (phi_xp - phi) / dx
    Dym = (phi - phi_ym) / dx;  Dyp = (phi_yp - phi) / dx

    gx_up    = Dxm.clamp(min=0).pow(2) + Dxp.clamp(max=0).pow(2)
    gy_up    = Dym.clamp(min=0).pow(2) + Dyp.clamp(max=0).pow(2)
    grad_mag = (gx_up + gy_up + 1e-10).sqrt()

    gx_cd = (phi_xp - phi_xm) / (2.0 * dx)
    gy_cd = (phi_yp - phi_ym) / (2.0 * dx)

    return grad_mag, gx_cd, gy_cd


# ---------------------------------------------------------------------------
# Redistancing (Sussman-Smereka-Osher 1994)
# ---------------------------------------------------------------------------

def _redistance(
    phi: "torch.Tensor", dx: float, n_iters: int = 5
) -> "torch.Tensor":
    """
    Restore |∇φ| = 1 on the exterior (unburned) side without moving φ = 0.

    Only cells with φ > 0 (ahead of the fire front) are updated.  The Godunov
    upwind scheme used by _phi_gradient is calibrated for an EXPANDING front
    (the exterior, φ > 0 side).  Applying it to φ < 0 cells uses the wrong
    characteristic direction and incorrectly pushes burned cells back toward
    zero, which undoes crossings detected in the level-set loop.

    Additionally, only cells within n_iters grid spacings of the front are
    updated (narrowband restriction) to avoid divergent edge artefacts.
    """
    eps     = dx
    s       = phi / (phi.pow(2) + eps ** 2).sqrt()
    dt_τ    = 0.5 * dx
    # Exterior narrowband: positive phi within n_iters cells of the front
    in_band = (phi > 0) & (phi < (n_iters * dx + dx))
    zero    = torch.zeros_like(phi)
    for _ in range(n_iters):
        update = dt_τ * s * (_phi_gradient(phi, dx) - 1.0)
        phi    = phi - torch.where(in_band, update, zero)
    return phi


# ---------------------------------------------------------------------------
# Elliptical (directional) ROS — Anderson (1983) / GPU spec Section 5
# ---------------------------------------------------------------------------

def _directional_ros(
    ros_head: "torch.Tensor",   # (N, R, C) m/s — Rothermel head-fire ROS
    e: "torch.Tensor",          # (N, R, C) — ellipse eccentricity (precomputed)
    theta_max: "torch.Tensor",  # (N, R, C) degrees — head-fire direction
    gx_cd: "torch.Tensor",      # (N, R, C) — central-diff ∂φ/∂x
    gy_cd: "torch.Tensor",      # (N, R, C) — central-diff ∂φ/∂y
) -> "torch.Tensor":
    """
    Anderson (1983) elliptical spread projected onto the local outward normal.

    Accepts precomputed gradient components (from _phi_gradient_full) and
    precomputed eccentricity (static within a run) to avoid redundant work
    in the hot loop.

    R(θ) = R_head * (1−e) / (1−e·cos θ)
    At θ=0  (head fire) → R_head
    At θ=180° (backing) → R_head·(1−e)/(1+e)
    Degrades to isotropic spread when e=0 (no wind).
    """
    gmag     = (gx_cd.pow(2) + gy_cd.pow(2) + 1e-10).sqrt()
    nx, ny   = gx_cd / gmag, gy_cd / gmag
    prop_deg = (torch.atan2(nx, ny) * _RAD2DEG) % 360.0

    delta     = (prop_deg - theta_max) % 360.0
    delta     = torch.minimum(delta, 360.0 - delta)
    theta_rad = delta * _DEG2RAD

    return ros_head * (1.0 - e) / (1.0 - e * torch.cos(theta_rad) + 1e-10)


# ---------------------------------------------------------------------------
# GPU Fire Engine
# ---------------------------------------------------------------------------

class GPUFireEngine:
    """
    Rothermel + level-set ensemble fire engine backed by PyTorch.

    Drop-in replacement for SimpleFire.  Terrain is loaded to the compute
    device once at construction; each run() call handles N ensemble members
    in parallel as a batch dimension.

    Parameters
    ----------
    terrain : TerrainData
    device  : "cuda" | "mps" | "cpu".  "cuda" falls back to "cpu" when CUDA
              is unavailable.  Pass "mps" to use Apple Silicon GPU.
    target_cfl : CFL safety factor (0.4 = conservative, 0.8 = aggressive).
    """

    def __init__(
        self,
        terrain: TerrainData,
        device: str = "cuda",
        target_cfl: float = 0.4,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed — cannot use GPUFireEngine.\n"
                "Install with:  pip install torch"
            )

        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "CUDA is not available — GPUFireEngine falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            device = "cpu"

        self.device     = torch.device(device)
        self.dx         = float(terrain.resolution_m)
        self.shape      = terrain.shape
        self.target_cfl = target_cfl

        slope_rad = np.radians(terrain.slope).astype(np.float32)
        self._tan_slope_sq = torch.tensor(
            np.tan(slope_rad) ** 2, dtype=torch.float32, device=self.device
        )
        self._aspect = torch.tensor(
            terrain.aspect.astype(np.float32), dtype=torch.float32, device=self.device
        )

        if hasattr(terrain, "canopy_base_height") and terrain.canopy_base_height is not None:
            cbh_np = np.asarray(terrain.canopy_base_height, dtype=np.float32)
            cbd_np = np.asarray(terrain.canopy_bulk_density, dtype=np.float32)
        else:
            cbh_np = _array_from_lookup(terrain.fuel_model, _CBH_PROXY)
            cbd_np = _array_from_lookup(terrain.fuel_model, _CBD_PROXY)
        self._cbh = torch.tensor(cbh_np, dtype=torch.float32, device=self.device)
        self._cbd = torch.tensor(cbd_np, dtype=torch.float32, device=self.device)

        # Wind adjustment factor: prefer direct canopy_cover field (available from
        # both LANDFIRE and synthetic_terrain); fall back to proxy lookup.
        if hasattr(terrain, "canopy_cover") and terrain.canopy_cover is not None:
            cc_np = np.asarray(terrain.canopy_cover, dtype=np.float32)
        else:
            cc_np = _array_from_lookup(terrain.fuel_model, _CC_PROXY)
        waf_np = np.where(cc_np > 0.5, 0.4, np.where(cc_np > 0.1, 0.5, 0.6)).astype(np.float32)
        self._wind_adj = torch.tensor(waf_np, dtype=torch.float32, device=self.device)

        fuel_table = torch.tensor(_build_fuel_table(), dtype=torch.float32, device=self.device)
        # Clamp to valid table range; any unknown code maps to row 0 (zero load = no fire).
        fuel_idx   = torch.tensor(
            np.clip(terrain.fuel_model.astype(np.int64), 0, _N_MODELS - 1),
            dtype=torch.long, device=self.device,
        )
        self._fuel_params = fuel_table[fuel_idx]  # (R, C, N_PARAMS)

    def run(
        self,
        terrain: TerrainData,
        gp_prior: GPPrior,
        fire_state: Optional[np.ndarray],
        n_members: int,
        horizon_min: int,
        rng: Optional[np.random.Generator] = None,
        initial_phi: Optional[np.ndarray] = None,
    ) -> EnsembleResult:
        """
        Run an N-member Rothermel + level-set ensemble.

        Parameters
        ----------
        terrain     : used for shape validation; physics tensors loaded at init.
        gp_prior    : GP posterior — source of per-member FMC/wind fields.
        fire_state  : float32[R, C] burn mask (1=burned).  Used to compute the
                      SDF when initial_phi is None.  May be None when
                      initial_phi is provided.
        n_members   : ensemble size N.
        horizon_min : simulation horizon in minutes.
        rng         : numpy Generator for reproducible sampling.
        initial_phi : float32[N, R, C] per-member SDF from EnsembleFireState.
                      When provided, fire_state is ignored.
        """
        if rng is None:
            rng = np.random.default_rng()

        rows, cols  = self.shape
        total_s     = float(horizon_min) * 60.0
        sentinel_s  = total_s * 1.1
        max_arrival = 2.0 * float(horizon_min)

        # ── Per-member perturbations ──────────────────────────────────────
        fmc_std = np.sqrt(np.clip(gp_prior.fmc_variance,        0.0, None)).astype(np.float32)
        ws_std  = np.sqrt(np.clip(gp_prior.wind_speed_variance, 0.0, None)).astype(np.float32)
        wd_std  = np.sqrt(np.clip(gp_prior.wind_dir_variance,   0.0, None)).astype(np.float32)

        n_fmc = rng.standard_normal((n_members, rows, cols)).astype(np.float32)
        n_ws  = rng.standard_normal((n_members, rows, cols)).astype(np.float32)
        n_wd  = rng.standard_normal((n_members, rows, cols)).astype(np.float32)

        fmc_np = np.clip(
            gp_prior.fmc_mean[np.newaxis] + n_fmc * fmc_std[np.newaxis],
            FMC_MIN_FRACTION, FMC_MAX_FRACTION,
        ).astype(np.float32)
        ws_np = np.clip(
            gp_prior.wind_speed_mean[np.newaxis] + n_ws * ws_std[np.newaxis],
            WIND_SPEED_MIN_MS, WIND_SPEED_MAX_MS,
        ).astype(np.float32)
        wd_np = ((gp_prior.wind_dir_mean[np.newaxis] + n_wd * wd_std[np.newaxis]) % 360.0
                 ).astype(np.float32)

        fmc = torch.tensor(fmc_np, dtype=torch.float32, device=self.device)
        ws  = torch.tensor(ws_np,  dtype=torch.float32, device=self.device)
        wd  = torch.tensor(wd_np,  dtype=torch.float32, device=self.device)

        # ── SDF initialisation ────────────────────────────────────────────
        arrival_t = torch.full(
            (n_members, rows, cols), sentinel_s,
            dtype=torch.float32, device=self.device,
        )

        if initial_phi is not None:
            phi = torch.tensor(initial_phi, dtype=torch.float32, device=self.device)
            member_burned = torch.tensor(initial_phi < 0, device=self.device)
            arrival_t[member_burned] = 0.0
        else:
            assert fire_state is not None, "fire_state required when initial_phi is None"
            burned_np = (fire_state > 0.5)
            if not burned_np.any():
                burned_np[rows // 2, cols // 2] = True
            d_in  = distance_transform_edt(burned_np).astype(np.float32)
            d_out = distance_transform_edt(~burned_np).astype(np.float32)
            sdf   = np.where(burned_np, -d_in, d_out) * self.dx
            phi   = (torch.tensor(sdf, dtype=torch.float32, device=self.device)
                     .unsqueeze(0).expand(n_members, -1, -1).clone())
            burned_t = torch.tensor(burned_np, device=self.device)
            arrival_t[:, burned_t] = 0.0

        # ── Pre-broadcast terrain tensors ─────────────────────────────────
        fp  = self._fuel_params.unsqueeze(0)   # (1, R, C, N_PARAMS)
        tss = self._tan_slope_sq.unsqueeze(0)  # (1, R, C)
        asp = self._aspect.unsqueeze(0)        # (1, R, C)
        cbh = self._cbh.unsqueeze(0)           # (1, R, C)
        cbd = self._cbd.unsqueeze(0)           # (1, R, C)
        waf = self._wind_adj.unsqueeze(0)      # (1, R, C)

        # ── Static precomputation (Optimisation 1: hoist Rothermel) ──────
        # fmc, ws, wd are fixed for the entire simulation — compute all
        # derived quantities once before the loop.

        ros_s, _, theta_max = _rothermel_ros(fmc, ws, waf, wd, asp, fp, tss)

        # Ellipse eccentricity from midflame wind (Anderson 1983)
        ws_mid_kmh = ws * waf * 3.6
        LB = (1.0 + 0.25 * ws_mid_kmh).clamp(min=1.0, max=8.0)
        e  = (LB - 1.0) / (LB + 1.0)   # (N, R, C)

        # Crown fire static quantities
        ros_crown  = 11.02 * (ws * 3.6).pow(0.854) * cbd.pow(0.19) / 60.0
        I_crit     = (0.01 * cbh * (460.0 + 2590.0)).pow(1.5)
        int_factor = 18000.0 * fp[..., _COL_W0] * (1.0 - fp[..., _COL_ST])  # (1,R,C)

        # Pre-allocated fire_type fill values (avoid torch.full_like in loop)
        crown_v   = ros_s.new_full(ros_s.shape, 2.0)
        surface_v = ros_s.new_ones(ros_s.shape)

        # ── Fixed CFL dt (Optimisation 3: single GPU→CPU sync) ───────────
        # ros_s is the head-fire surface ROS; ros_crown is the head-fire crown
        # ROS.  Directional attenuation only reduces surface ROS, so the maximum
        # possible speed in any cell of the simulation is bounded by this.
        max_ros = max(ros_s.max().item(), ros_crown.max().item())  # one sync
        dt = float(max(0.5, min(self.target_cfl * self.dx / max(max_ros, 1e-10), 300.0)))

        # ── CFL-adaptive level-set loop ───────────────────────────────────
        fire_type = surface_v.clone()
        t    = 0.0
        step = 0

        while t < total_s:
            dt_step = min(dt, total_s - t)
            if dt_step <= 0.0:
                break

            # Optimisation 2: single fused gradient pass (4 rolls, not 8)
            grad_mag, gx_cd, gy_cd = _phi_gradient_full(phi, self.dx)

            ros_dir   = _directional_ros(ros_s, e, theta_max, gx_cd, gy_cd)
            intensity = int_factor * ros_dir
            initiates = intensity > I_crit
            ros_f     = torch.where(initiates, torch.maximum(ros_dir, ros_crown), ros_dir)
            fire_type = torch.where(initiates, crown_v, surface_v)

            phi_new  = phi - dt_step * ros_f * grad_mag

            # Sub-timestep arrival: linear interpolation to the zero crossing
            crossing = (phi > 0) & (phi_new <= 0)
            t_cross  = t + dt_step * phi / (phi - phi_new + 1e-10)
            arrival_t[crossing] = t_cross[crossing]

            phi   = phi_new
            step += 1
            t    += dt_step

            if step % _REDISTANCE_EVERY == 0:
                phi = _redistance(phi, self.dx)

        # ── CPU transfer and statistics ───────────────────────────────────
        at_s  = arrival_t.cpu().numpy()
        ft_np = fire_type.cpu().numpy().astype(np.int8)

        at_min = np.where(at_s >= sentinel_s * 0.9, max_arrival, at_s / 60.0)

        truly_burned = at_min < max_arrival * 0.9
        burn_prob    = truly_burned.mean(axis=0).astype(np.float32)

        with np.errstate(invalid="ignore"):
            masked  = np.where(truly_burned, at_min, np.nan)
            mean_at = np.where(truly_burned.any(axis=0), np.nanmean(masked, axis=0), np.nan)
            var_at  = at_min.var(axis=0)

        return EnsembleResult(
            member_arrival_times   = at_min.astype(np.float32),
            member_fmc_fields      = fmc_np,
            member_wind_fields     = ws_np,
            member_wind_dir_fields = wd_np,
            burn_probability       = burn_prob,
            mean_arrival_time      = mean_at.astype(np.float32),
            arrival_time_variance  = var_at.astype(np.float32),
            n_members              = n_members,
            member_fire_types      = ft_np,
        )
