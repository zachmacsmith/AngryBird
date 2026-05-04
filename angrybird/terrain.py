"""
Terrain Manager — static, loaded once per run.

  synthetic_terrain(shape, ...) → TerrainData   fractal DEM for testing / development

Real terrain is loaded via angrybird.landfire.load_from_directory(cache_dir).

Coordinate contract (PotentialBugs1 §1):
  All internal arrays are in a local UTM projection.
  lat/lon only appears at the boundary (origin_latlon output field).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.ndimage import sobel

from .config import GRID_RESOLUTION_M
from .types import TerrainData


class TerrainLoadError(RuntimeError):
    """Raised when terrain data cannot be loaded."""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slope and aspect from elevation array
# ---------------------------------------------------------------------------

def _slope_aspect(
    elevation: np.ndarray, resolution_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (degrees) and aspect (degrees from north, 0-360) from a DEM.
    Uses Sobel finite differences; edges are mirrored.
    """
    dz_dy = sobel(elevation, axis=0) / (8 * resolution_m)
    dz_dx = sobel(elevation, axis=1) / (8 * resolution_m)

    slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))).astype(np.float32)

    # Aspect: downslope direction, degrees clockwise from north (standard GIS convention).
    # The gradient vector (dz_dx, -dz_dy) = (∂z/∂east, ∂z/∂north) points UPslope.
    # Negate it to get the DOWNslope direction.
    # Verification: south-facing slope (dz_dy<0, dz_dx=0) → arctan2(0, negative) = 180° ✓
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect = (np.degrees(aspect_rad) % 360).astype(np.float32)

    return slope, aspect


# ---------------------------------------------------------------------------
# Synthetic terrain (fractal DEM for testing / development)
# ---------------------------------------------------------------------------

def synthetic_terrain(
    shape: tuple[int, int],
    resolution_m: float = GRID_RESOLUTION_M,
    origin: tuple[float, float] = (37.0, -120.0),
    seed: int = 42,
) -> TerrainData:
    """
    Generate a physically plausible synthetic terrain using an FFT power-law
    spectrum (1/f² — Brownian surface) for the DEM.  Fuel models are assigned
    by elevation band to mimic real vegetation zonation.

    Used when LANDFIRE is unavailable and for unit tests.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape

    # ── Fractal DEM ───────────────────────────────────────────────────────
    noise = rng.standard_normal(shape)
    freqs_r = np.fft.fftfreq(rows, d=resolution_m)
    freqs_c = np.fft.fftfreq(cols, d=resolution_m)
    freq_grid = np.sqrt(freqs_r[:, None] ** 2 + freqs_c[None, :] ** 2)
    freq_grid[0, 0] = 1.0           # avoid divide-by-zero at DC
    spectrum = 1.0 / (freq_grid ** 2)
    spectrum[0, 0] = 0.0            # zero mean (no DC component)
    elevation_raw = np.real(np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(spectrum)))

    # Normalise to 100–1600 m (representative Western US terrain)
    lo, hi = 100.0, 1600.0
    e_min, e_max = elevation_raw.min(), elevation_raw.max()
    elevation = ((elevation_raw - e_min) / max(e_max - e_min, 1e-6) * (hi - lo) + lo
                 ).astype(np.float32)

    slope, aspect = _slope_aspect(elevation, resolution_m)

    # ── Fuel models by elevation band (SB40 codes, vegetation zonation) ──
    # Low  (<500m):   grass / grass-shrub  (GR2=102, GR3=103, GS1=121, GS2=122)
    # Mid  (500-900m): shrub / timber under.(SH1=141, SH2=142, SH5=145, TU1=161)
    # High (>900m):   timber litter        (TL1=181, TL2=182, TL3=183, TL8=188)
    fuel_model = np.empty(shape, dtype=np.int16)
    low  = elevation <  500
    mid  = (elevation >= 500) & (elevation < 900)
    high = elevation >= 900

    fuel_model[low]  = rng.choice([102, 103, 121, 122], size=low.sum())
    fuel_model[mid]  = rng.choice([141, 142, 145, 161], size=mid.sum())
    fuel_model[high] = rng.choice([181, 182, 183, 188], size=high.sum())

    # ── Canopy arrays from proxy tables + small spatial jitter ───────────
    ch, cbh, cbd, cc = _canopy_from_fuel(fuel_model, rng)

    logger.info(
        "Synthetic terrain generated: shape=%s, elev=[%.0f, %.0f] m",
        shape, elevation.min(), elevation.max(),
    )
    return TerrainData(
        elevation=elevation,
        slope=slope,
        aspect=aspect,
        fuel_model=fuel_model,
        canopy_cover=cc,
        canopy_height=ch,
        canopy_base_height=cbh,
        canopy_bulk_density=cbd,
        shape=shape,
        resolution_m=resolution_m,
        origin_latlon=origin,
    )


# ---------------------------------------------------------------------------
# Canopy array helpers (used by synthetic_terrain)
# ---------------------------------------------------------------------------

_SB40_CH  = {  # canopy height (m)
    102: 0.0, 103: 0.0, 121: 1.0, 122: 1.5,         # GR, GS
    141: 2.0, 142: 2.5, 145: 3.0, 161: 8.0,          # SH, TU
    181: 15.0, 182: 18.0, 183: 20.0, 188: 12.0,       # TL
}
_SB40_CBH = {102: 0.0, 103: 0.0, 121: 0.5, 122: 0.8,
             141: 1.0, 142: 1.2, 145: 1.5, 161: 3.0,
             181: 5.0, 182: 6.0, 183: 7.0, 188: 4.0}
_SB40_CBD = {102: 0.00, 103: 0.00, 121: 0.02, 122: 0.03,
             141: 0.05, 142: 0.05, 145: 0.06, 161: 0.10,
             181: 0.12, 182: 0.14, 183: 0.15, 188: 0.10}
_SB40_CC  = {102: 0.0, 103: 0.05, 121: 0.10, 122: 0.15,
             141: 0.20, 142: 0.25, 145: 0.30, 161: 0.55,
             181: 0.70, 182: 0.75, 183: 0.80, 188: 0.65}


def _canopy_from_fuel(
    fuel_model: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate CH, CBH, CBD, and canopy cover arrays from SB40 proxy tables.
    Adds ±10% multiplicative jitter (when rng is provided) for spatial realism.

    Returns (ch, cbh, cbd, cc) all as float32[rows, cols].
    """
    ch  = np.zeros(fuel_model.shape, dtype=np.float32)
    cbh = np.zeros(fuel_model.shape, dtype=np.float32)
    cbd = np.zeros(fuel_model.shape, dtype=np.float32)
    cc  = np.zeros(fuel_model.shape, dtype=np.float32)
    for fid in _SB40_CH:
        mask = fuel_model == fid
        ch[mask]  = _SB40_CH[fid]
        cbh[mask] = _SB40_CBH[fid]
        cbd[mask] = _SB40_CBD[fid]
        cc[mask]  = _SB40_CC[fid]

    if rng is not None:
        jitter = rng.uniform(0.9, 1.1, fuel_model.shape).astype(np.float32)
        ch  = np.clip(ch  * jitter, 0.0, None)
        cbh = np.clip(cbh * jitter, 0.0, None)
        cbd = np.clip(cbd * jitter, 0.0, None)
        cc  = np.clip(cc  * jitter, 0.0, 1.0)

    return ch, cbh, cbd, cc
