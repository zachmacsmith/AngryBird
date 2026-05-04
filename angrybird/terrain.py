"""
Terrain Manager — static, loaded once per run.

Two explicit modes — caller chooses, no silent fallback:

  load_terrain(bbox, ...)       → TerrainData   real LANDFIRE data; raises TerrainLoadError on failure
  synthetic_terrain(shape, ...) → TerrainData   fractal DEM for testing / development

Coordinate contract (PotentialBugs1 §1):
  All internal arrays are in a local UTM projection.
  lat/lon only appears at the boundary (bbox input, origin output field).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.ndimage import zoom as ndimage_zoom
from scipy.ndimage import sobel

from .config import CANOPY_CBH_M, CANOPY_CBD_KGM3, CANOPY_COVER_FRACTION, FUEL_PARAMS, GRID_RESOLUTION_M
from .types import TerrainData


class TerrainLoadError(RuntimeError):
    """Raised when LANDFIRE terrain data cannot be loaded."""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UTM helpers (PotentialBugs1 §1 — project at ingestion, stay metric internally)
# ---------------------------------------------------------------------------

def _utm_epsg(center_lon: float, center_lat: float) -> str:
    """Return the EPSG code for the UTM zone covering a given lat/lon."""
    zone = int((center_lon + 180) / 6) + 1
    base = 32600 if center_lat >= 0 else 32700
    return f"EPSG:{base + zone}"


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
    # Gradient in x (east) and y (south, because row index increases southward)
    dz_dy = sobel(elevation, axis=0) / (8 * resolution_m)   # north-south gradient
    dz_dx = sobel(elevation, axis=1) / (8 * resolution_m)   # east-west gradient

    slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))).astype(np.float32)

    # Aspect: downslope direction, degrees clockwise from north (standard GIS convention).
    # The gradient vector (dz_dx, -dz_dy) = (∂z/∂east, ∂z/∂north) points UPslope.
    # Negate it to get the DOWNslope direction.
    # arctan2(-dz_dx, dz_dy) = arctan2(-∂z/∂east, -∂z/∂north) corrected for numpy row ordering.
    # Verification: south-facing slope (dz_dy<0, dz_dx=0) → arctan2(0, negative) = 180° ✓
    aspect_rad = np.arctan2(-dz_dx, dz_dy)                # downslope bearing, 0=N 90=E 180=S 270=W
    aspect = (np.degrees(aspect_rad) % 360).astype(np.float32)

    return slope, aspect


# ---------------------------------------------------------------------------
# Synthetic terrain (fractal DEM fallback)
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
# Canopy array helpers
# ---------------------------------------------------------------------------

# SB40 canopy proxy tables (used only by synthetic_terrain)
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


# ---------------------------------------------------------------------------
# LANDFIRE loader
# ---------------------------------------------------------------------------

# LANDFIRE layer names for each product
_LANDFIRE_LAYERS = {
    "elevation":          "ELEV2020",
    "slope":              "SLP2020",
    "aspect":             "ASP2020",
    "fuel_model":         "220F13_22",   # Anderson 13 fuel models
    "canopy_base_height": "CBD_2020",    # canopy base height (m)
    "canopy_bulk_density":"CBH_2020",    # canopy bulk density (kg/m³) [LANDFIRE naming is inverted]
    "canopy_cover":       "CC_2020",     # canopy cover fraction (0-100 in LANDFIRE, scaled to 0-1)
}


def _resample(arr: np.ndarray, target_resolution_m: float, source_resolution_m: float) -> np.ndarray:
    """Resample array from source to target resolution using bilinear zoom."""
    factor = source_resolution_m / target_resolution_m
    if abs(factor - 1.0) < 1e-3:
        return arr
    order = 1 if arr.dtype == np.int8 else 1   # bilinear for all; nearest for ints handled below
    if np.issubdtype(arr.dtype, np.integer):
        # Nearest-neighbour for categorical data (fuel model IDs)
        resampled = ndimage_zoom(arr.astype(np.float32), factor, order=0)
        return resampled.astype(arr.dtype)
    return ndimage_zoom(arr.astype(np.float32), factor, order=1)


def load_terrain(
    bbox: tuple[float, float, float, float],
    target_resolution_m: float = GRID_RESOLUTION_M,
    source_resolution_m: float = 30.0,
) -> TerrainData:
    """
    Load terrain data for a bounding box from LANDFIRE, resampled to
    `target_resolution_m`.

    Args:
        bbox: (min_lat, min_lon, max_lat, max_lon) — WGS84 decimal degrees
        target_resolution_m: desired output resolution (default 50 m)
        source_resolution_m: native LANDFIRE resolution (30 m)

    Returns:
        TerrainData with arrays in the target resolution.

    Raises:
        TerrainLoadError: if LANDFIRE is unreachable, rasterio is missing,
            or any other failure occurs during download or parsing.
            Use synthetic_terrain() for offline / testing workflows.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    origin = (max_lat, min_lon)   # NW corner

    try:
        return _load_from_landfire(
            bbox, origin, center_lat, center_lon,
            target_resolution_m, source_resolution_m,
        )
    except ImportError as exc:
        raise TerrainLoadError(
            f"Missing dependency for LANDFIRE loader: {exc}. "
            "Install with: pip install landfire rasterio"
        ) from exc
    except Exception as exc:
        raise TerrainLoadError(
            f"Failed to load LANDFIRE terrain for bbox={bbox}: {exc}"
        ) from exc


def _load_from_landfire(
    bbox: tuple[float, float, float, float],
    origin: tuple[float, float],
    center_lat: float,
    center_lon: float,
    target_resolution_m: float,
    source_resolution_m: float,
) -> TerrainData:
    """
    Inner function: download from LANDFIRE and parse with rasterio.
    Raises on any error so the caller can fall back to synthetic.
    """
    import rasterio                          # type: ignore
    from landfire import Landfire            # type: ignore

    min_lat, min_lon, max_lat, max_lon = bbox
    utm_epsg = _utm_epsg(center_lon, center_lat)

    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    lf = Landfire(bbox=bbox_str, output_crs=utm_epsg)
    lf.request_data(layers=list(_LANDFIRE_LAYERS.values()))

    arrays: dict[str, np.ndarray] = {}
    for product, layer_name in _LANDFIRE_LAYERS.items():
        # landfire-python saves files to a temp directory; find by layer name
        raster_path = lf.get_raster_path(layer_name)
        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = 0.0
        arrays[product] = _resample(data, target_resolution_m, source_resolution_m)

    elevation  = arrays["elevation"].astype(np.float32)
    slope      = arrays["slope"].astype(np.float32)
    aspect     = arrays["aspect"].astype(np.float32)
    fuel_model = np.clip(arrays["fuel_model"], 1, 13).astype(np.int8)

    # Validate fuel IDs — LANDFIRE may include NB codes; clamp to Anderson 13
    valid = np.isin(fuel_model, list(FUEL_PARAMS.keys()))
    fuel_model[~valid] = 1    # default to grass (model 1) for unknown codes

    # Canopy layers (LANDFIRE values: CBH in m, CBD in kg/m³, CC in 0-100 percent)
    cbh = np.clip(arrays.get("canopy_base_height", np.zeros_like(elevation)), 0.0, 50.0).astype(np.float32)
    cbd = np.clip(arrays.get("canopy_bulk_density", np.zeros_like(elevation)), 0.0,  2.0).astype(np.float32)
    cc  = np.clip(arrays.get("canopy_cover",        np.zeros_like(elevation)) / 100.0, 0.0, 1.0).astype(np.float32)

    shape = (elevation.shape[0], elevation.shape[1])
    logger.info(
        "LANDFIRE terrain loaded: shape=%s, elev=[%.0f, %.0f] m, resolution=%.0f m",
        shape, elevation.min(), elevation.max(), target_resolution_m,
    )
    return TerrainData(
        elevation=elevation,
        slope=slope,
        aspect=aspect,
        fuel_model=fuel_model,
        resolution_m=target_resolution_m,
        origin=origin,
        shape=shape,
        canopy_base_height=cbh,
        canopy_bulk_density=cbd,
        canopy_cover=cc,
    )
