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
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..config import (
    CANOPY_CBH_M,
    CANOPY_CBD_KGM3,
    CANOPY_COVER_FRACTION,
    FUEL_MINERAL_CONTENT,
    FUEL_MINERAL_SILICA_FREE,
    FUEL_PARAMS,
    FUEL_PARTICLE_DENSITY,
)
from ..types import EnsembleResult, GPPrior, TerrainData

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fuel table
# ---------------------------------------------------------------------------

_N_MODELS = 14   # rows 0-13; row 0 unused (no FM 0)
_N_PARAMS = 8
_COL_W0, _COL_SIGMA, _COL_DELTA = 0, 1, 2
_COL_MX, _COL_HEAT, _COL_RHOP   = 3, 4, 5
_COL_ST, _COL_SE                  = 6, 7

_SAV_FT_TO_M = 0.3048  # SAV in config is 1/ft; Rothermel SI uses 1/m

# Proxy canopy tables — single source of truth in config.py
_CBH_PROXY = CANOPY_CBH_M
_CBD_PROXY = CANOPY_CBD_KGM3
_CC_PROXY  = CANOPY_COVER_FRACTION


def _array_from_lookup(
    fuel_model: np.ndarray, lookup: dict[int, float]
) -> np.ndarray:
    out = np.zeros(fuel_model.shape, dtype=np.float32)
    for fid, val in lookup.items():
        out[fuel_model == fid] = float(val)
    return out


def _build_fuel_table() -> np.ndarray:
    """
    (14, 8) float32 parameter table.  Row index = fuel model ID.
    SAV converted from 1/ft (config) to 1/m (Rothermel SI).
    """
    tbl = np.zeros((_N_MODELS, _N_PARAMS), dtype=np.float32)
    for fid, p in FUEL_PARAMS.items():
        if not (1 <= fid < _N_MODELS):
            continue
        tbl[fid, _COL_W0]    = p["load"]
        tbl[fid, _COL_SIGMA] = p["sav"] * _SAV_FT_TO_M
        tbl[fid, _COL_DELTA] = p["depth"]
        tbl[fid, _COL_MX]    = p["mx"]
        tbl[fid, _COL_HEAT]  = p["h"]
        tbl[fid, _COL_RHOP]  = FUEL_PARTICLE_DENSITY
        tbl[fid, _COL_ST]    = FUEL_MINERAL_CONTENT
        tbl[fid, _COL_SE]    = FUEL_MINERAL_SILICA_FREE
    return tbl


# ---------------------------------------------------------------------------
# Rothermel surface ROS kernel
# ---------------------------------------------------------------------------

def _rothermel_ros(
    fmc: "torch.Tensor",        # (N, R, C)
    ws: "torch.Tensor",         # (N, R, C) m/s at 10 m
    wind_adj: "torch.Tensor",   # (1, R, C) midflame wind adjustment factor
    fp: "torch.Tensor",         # (1, R, C, N_PARAMS)
    tan_slope_sq: "torch.Tensor",  # (1, R, C)
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Rothermel (1972) surface ROS for a batched ensemble.
    All spatial tensors broadcast over the N (ensemble) dimension.
    Returns (ros m/s, I_R kW/m²), both shape (N, R, C).
    """
    w0    = fp[..., _COL_W0]
    sigma = fp[..., _COL_SIGMA]
    delta = fp[..., _COL_DELTA]
    mx    = fp[..., _COL_MX]
    heat  = fp[..., _COL_HEAT]
    rho_p = fp[..., _COL_RHOP]
    st    = fp[..., _COL_ST]
    se    = fp[..., _COL_SE]

    # Packing ratio and relative packing ratio
    beta       = w0 / (delta * rho_p + 1e-10)
    beta_op    = 3.348 * sigma.pow(-0.8189)
    beta_ratio = beta / (beta_op + 1e-10)

    # Optimum reaction velocity (1/min)
    A         = 133.0 * sigma.pow(-0.7913)
    gamma_max = sigma.pow(1.5) / (495.0 + 0.0594 * sigma.pow(1.5))
    gamma     = gamma_max * beta_ratio.pow(A) * (A * (1.0 - beta_ratio)).exp()

    # Net fuel load, mineral damping, moisture damping
    wn    = w0 * (1.0 - st)
    eta_s = (0.174 * se.pow(-0.19)).clamp(max=1.0)
    rm    = fmc / (mx + 1e-10)
    eta_m = (1.0 - 2.59 * rm + 5.11 * rm.pow(2) - 3.52 * rm.pow(3)).clamp(0.0, 1.0)

    # Reaction intensity (kW/m²)
    I_R = gamma * wn * heat * eta_m * eta_s

    # Propagating flux ratio
    xi = ((0.792 + 0.681 * sigma.pow(0.5)) * (beta + 0.1)).exp() \
         / (192.0 + 0.2595 * sigma)

    # Wind factor: Rothermel coefficients C, B, E
    C = 7.47 * (-0.133 * sigma.pow(0.55)).exp()
    B = 0.02526 * sigma.pow(0.54)
    E = 0.715 * (-3.59e-4 * sigma).exp()
    # 10m wind → midflame wind in m/min
    ws_midflame = ws * 60.0 * wind_adj
    phi_w = C * ws_midflame.pow(B) * beta_ratio.pow(-E)

    # Slope factor (precomputed tan²(slope))
    phi_s = 5.275 * beta.pow(-0.3) * tan_slope_sq

    # Heat of preignition and effective heating number
    Q_ig = 250.0 + 1116.0 * fmc
    eps  = (-138.0 / (sigma + 1e-10)).exp()

    # Bulk density
    rho_b = w0 / (delta + 1e-10)

    # Rate of spread: m/min → m/s
    ros = ((I_R * xi * (1.0 + phi_w + phi_s)) / (rho_b * eps * Q_ig + 1e-10) / 60.0
           ).clamp(min=0.0)

    return ros, I_R


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
    """
    Van Wagner (1977) crown fire initiation.
    Returns (ros_final, fire_type) where fire_type: 1=surface, 2=crown.
    """
    # Critical intensity for crown fire initiation (foliar FMC = 100%)
    I_crit    = (0.01 * cbh * (460.0 + 2590.0)).pow(1.5)
    initiates = surface_intensity > I_crit

    # Rothermel (1991) crown ROS in m/s
    ros_crown = 11.02 * (ws * 3.6).pow(0.854) * cbd.pow(0.19) / 60.0

    ros_final = torch.where(initiates, torch.maximum(ros_surface, ros_crown), ros_surface)
    fire_type = torch.where(
        initiates,
        torch.full_like(ros_surface, 2.0),
        torch.full_like(ros_surface, 1.0),
    )
    return ros_final, fire_type


# ---------------------------------------------------------------------------
# Level-set gradient (Godunov upwind scheme)
# ---------------------------------------------------------------------------

def _phi_gradient(phi: "torch.Tensor", dx: float) -> "torch.Tensor":
    """
    Upwind |∇φ| for an expanding front (speed always positive).
    phi: (N, R, C).  Uses zero-flux boundary conditions.
    """
    # Shift neighbours
    phi_xp = torch.roll(phi, -1, dims=2).clone(); phi_xp[:, :, -1] = phi[:, :, -1]
    phi_xm = torch.roll(phi,  1, dims=2).clone(); phi_xm[:, :,  0] = phi[:, :,  0]
    phi_yp = torch.roll(phi, -1, dims=1).clone(); phi_yp[:,  -1, :] = phi[:,  -1, :]
    phi_ym = torch.roll(phi,  1, dims=1).clone(); phi_ym[:,   0, :] = phi[:,   0, :]

    # One-sided finite differences
    Dxm = (phi - phi_xm) / dx;  Dxp = (phi_xp - phi) / dx
    Dym = (phi - phi_ym) / dx;  Dyp = (phi_yp - phi) / dx

    # Godunov: max(Dxm, 0)² + min(Dxp, 0)² (expanding front → use positive speeds)
    z  = phi.new_zeros(1)
    gx = Dxm.clamp(min=0).pow(2) + Dxp.clamp(max=0).pow(2)
    gy = Dym.clamp(min=0).pow(2) + Dyp.clamp(max=0).pow(2)
    return (gx + gy + 1e-10).sqrt()


# ---------------------------------------------------------------------------
# CFL adaptive timestep
# ---------------------------------------------------------------------------

def _cfl_dt(ros: "torch.Tensor", dx: float, cfl: float = 0.4) -> float:
    """CFL-stable timestep in seconds. One GPU→CPU sync per call (~10 µs)."""
    max_ros = ros.max().item()
    if max_ros < 1e-10:
        return 300.0
    return float(max(0.5, min(cfl * dx / max_ros, 300.0)))


# ---------------------------------------------------------------------------
# GPU Fire Engine
# ---------------------------------------------------------------------------

class GPUFireEngine:
    """
    Rothermel + level-set ensemble fire engine backed by PyTorch.

    Drop-in replacement for SimpleFire.  Terrain is loaded to the compute
    device once at construction; each ``run()`` call handles N ensemble
    members in parallel as a batch dimension.

    Parameters
    ----------
    terrain : TerrainData
        Static terrain.  Loaded to device at construction; reused across runs.
    device : str
        ``"cuda"`` falls back to ``"cpu"`` if no CUDA device is found.
    target_cfl : float
        CFL safety factor (0.4 = conservative, 0.8 = aggressive).
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

        # ── Terrain tensors (loaded once) ──────────────────────────────────
        slope_rad = np.radians(terrain.slope).astype(np.float32)
        self._tan_slope_sq = torch.tensor(
            np.tan(slope_rad) ** 2, dtype=torch.float32, device=self.device
        )

        # CBH / CBD: use terrain attributes if present, else derive from fuel model
        if hasattr(terrain, "canopy_base_height") and terrain.canopy_base_height is not None:
            cbh_np = np.asarray(terrain.canopy_base_height, dtype=np.float32)
            cbd_np = np.asarray(terrain.canopy_bulk_density, dtype=np.float32)
        else:
            cbh_np = _array_from_lookup(terrain.fuel_model, _CBH_PROXY)
            cbd_np = _array_from_lookup(terrain.fuel_model, _CBD_PROXY)
        self._cbh = torch.tensor(cbh_np, dtype=torch.float32, device=self.device)
        self._cbd = torch.tensor(cbd_np, dtype=torch.float32, device=self.device)

        # Wind adjustment factor: 10m → midflame wind height
        cc_np  = _array_from_lookup(terrain.fuel_model, _CC_PROXY)
        waf_np = np.where(cc_np > 0.5, 0.4, np.where(cc_np > 0.1, 0.5, 0.6)).astype(np.float32)
        self._wind_adj = torch.tensor(waf_np, dtype=torch.float32, device=self.device)

        # Per-cell fuel parameter gather: (R, C, N_PARAMS)
        fuel_table = torch.tensor(_build_fuel_table(), dtype=torch.float32, device=self.device)
        fuel_idx   = torch.tensor(
            np.clip(terrain.fuel_model.astype(np.int64), 0, _N_MODELS - 1),
            dtype=torch.long,
            device=self.device,
        )
        self._fuel_params = fuel_table[fuel_idx]  # (R, C, N_PARAMS)

    # ──────────────────────────────────────────────────────────────────────
    # FireEngineProtocol
    # ──────────────────────────────────────────────────────────────────────

    def run(
        self,
        terrain: TerrainData,
        gp_prior: GPPrior,
        fire_state: np.ndarray,
        n_members: int,
        horizon_min: int,
        rng: Optional[np.random.Generator] = None,
    ) -> EnsembleResult:
        """
        Run an N-member Rothermel + level-set ensemble.

        Parameters
        ----------
        terrain     : TerrainData — used for shape validation only; physics
                      tensors were loaded at construction.
        gp_prior    : GP posterior used to sample per-member FMC/wind fields.
        fire_state  : float32[R, C] — initial burn mask (1=burned, 0=unburned).
        n_members   : ensemble size N.
        horizon_min : simulation horizon in minutes.
        rng         : numpy Generator for reproducible perturbation sampling.

        Returns
        -------
        EnsembleResult with arrival times in **minutes**.
        Unburned cells use sentinel = 2 × horizon_min (PotentialBugs1 §3).
        mean_arrival_time uses NaN for fully-unburned cells (visualization only).
        """
        if rng is None:
            rng = np.random.default_rng()

        rows, cols   = self.shape
        total_s      = float(horizon_min) * 60.0
        # PotentialBugs1 §3: use finite sentinel (2× horizon) for unburned cells.
        # NaN propagates through corrcoef → NaN sensitivity → broken information field.
        # 2× horizon is large enough that unburned/uncertain cells get high variance
        # without overwhelming the correlation signal.
        sentinel_s   = total_s * 1.1          # GPU loop uses seconds internally
        max_arrival  = 2.0 * float(horizon_min)  # output sentinel in minutes

        # ── Per-member perturbations from GP prior (vectorised) ───────────
        fmc_std = np.sqrt(np.clip(gp_prior.fmc_variance, 0.0, None)).astype(np.float32)
        ws_std  = np.sqrt(np.clip(gp_prior.wind_speed_variance, 0.0, None)).astype(np.float32)
        wd_std  = np.sqrt(np.clip(gp_prior.wind_dir_variance, 0.0, None)).astype(np.float32)

        n_fmc = rng.standard_normal((n_members, rows, cols)).astype(np.float32)
        n_ws  = rng.standard_normal((n_members, rows, cols)).astype(np.float32)
        n_wd  = rng.standard_normal((n_members, rows, cols)).astype(np.float32)

        fmc_np = np.clip(
            gp_prior.fmc_mean[np.newaxis] + n_fmc * fmc_std[np.newaxis],
            0.02, 0.40,
        ).astype(np.float32)
        ws_np = np.clip(
            gp_prior.wind_speed_mean[np.newaxis] + n_ws * ws_std[np.newaxis],
            0.5, 25.0,
        ).astype(np.float32)
        # wind direction perturbation stored for EnsembleResult but not used
        # in the scalar level-set (ROS magnitude only)

        fmc = torch.tensor(fmc_np, dtype=torch.float32, device=self.device)
        ws  = torch.tensor(ws_np,  dtype=torch.float32, device=self.device)

        # ── Level-set initialisation: signed distance in metres ──────────
        burned_np = (fire_state > 0.5)
        if not burned_np.any():
            # No ignition supplied — seed centre of domain
            burned_np[rows // 2, cols // 2] = True

        # SDF: negative inside fire perimeter, positive outside (metres).
        # d_in:  for burned cells,   distance to nearest unburned edge  (interior)
        # d_out: for unburned cells, distance to nearest burned cell     (exterior)
        d_in  = distance_transform_edt(burned_np).astype(np.float32)
        d_out = distance_transform_edt(~burned_np).astype(np.float32)
        sdf   = np.where(burned_np, -d_in, d_out) * self.dx
        phi   = (torch.tensor(sdf, dtype=torch.float32, device=self.device)
                 .unsqueeze(0).expand(n_members, -1, -1).clone())

        arrival_t = torch.full(
            (n_members, rows, cols), sentinel_s,
            dtype=torch.float32, device=self.device,
        )
        burned_t = torch.tensor(burned_np, device=self.device)
        arrival_t[:, burned_t] = 0.0

        # ── Pre-broadcast terrain tensors ─────────────────────────────────
        fp  = self._fuel_params.unsqueeze(0)    # (1, R, C, N_PARAMS)
        tss = self._tan_slope_sq.unsqueeze(0)   # (1, R, C)
        cbh = self._cbh.unsqueeze(0)            # (1, R, C)
        cbd = self._cbd.unsqueeze(0)            # (1, R, C)
        waf = self._wind_adj.unsqueeze(0)       # (1, R, C)

        # ── CFL-adaptive level-set simulation loop ───────────────────────
        fire_type = torch.ones(
            (n_members, rows, cols), dtype=torch.float32, device=self.device
        )
        t = 0.0
        while t < total_s:
            ros_s, I_R = _rothermel_ros(fmc, ws, waf, fp, tss)

            # Byram fireline intensity (kW/m) for crown fire criterion
            intensity = 18000.0 * fp[..., _COL_W0] * ros_s

            ros_f, fire_type = _crown_fire_check(intensity, cbh, cbd, ws, ros_s)

            dt = _cfl_dt(ros_f, self.dx, self.target_cfl)
            dt = min(dt, total_s - t)
            if dt <= 0.0:
                break

            phi_old  = phi.clone()
            phi      = phi - dt * ros_f * _phi_gradient(phi, self.dx)

            # Record arrival: cells that just crossed φ = 0 (unburned → burning)
            arrival_t[(phi_old > 0) & (phi <= 0)] = t

            t += dt

        # ── Transfer to CPU and compute statistics ────────────────────────
        at_s      = arrival_t.cpu().numpy()   # seconds
        ft_np     = fire_type.cpu().numpy().astype(np.int8)  # (N, R, C): 1=surface, 2=crown

        # Convert to minutes; cells that never crossed φ=0 get the finite sentinel
        # (PotentialBugs1 §3 — finite sentinel preserves variance structure so that
        # corrcoef in sensitivity computation sees real disagreement between members
        # that burned a cell and those that didn't, rather than collapsing to NaN).
        at_min = np.where(at_s >= sentinel_s * 0.9, max_arrival, at_s / 60.0)

        # burn_probability and mean/variance are derived from the sentinel-filled array.
        # Cells that all members agree are unburned (arrival == max_arrival) have
        # burn_prob ≈ 0; cells all members burned have burn_prob ≈ 1.
        truly_burned  = at_min < max_arrival * 0.9
        burn_prob     = truly_burned.mean(axis=0).astype(np.float32)

        # mean_arrival_time uses NaN for fully-unburned cells — valid for visualization.
        with np.errstate(invalid="ignore"):
            masked = np.where(truly_burned, at_min, np.nan)
            mean_at = np.where(truly_burned.any(axis=0), np.nanmean(masked, axis=0), np.nan)
            var_at  = at_min.var(axis=0)  # variance over sentinel-filled array per §3

        return EnsembleResult(
            member_arrival_times  = at_min.astype(np.float32),
            member_fmc_fields     = fmc_np,
            member_wind_fields    = ws_np,
            burn_probability      = burn_prob,
            mean_arrival_time     = mean_at.astype(np.float32),
            arrival_time_variance = var_at.astype(np.float32),
            n_members             = n_members,
            member_fire_types     = ft_np,
        )
