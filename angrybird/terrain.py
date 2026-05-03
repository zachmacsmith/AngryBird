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

from .config import FUEL_PARAMS, GRID_RESOLUTION_M
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

    # Aspect: degrees clockwise from north
    aspect_rad = np.arctan2(dz_dx, -dz_dy)               # north = 0, east = 90
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

    # ── Fuel models by elevation band (vegetation zonation) ───────────────
    # Low  (<500m):   grassland / chaparral (models 1-3, 6-7)
    # Mid  (500-900m): shrub / open woodland (models 4-5, 6)
    # High (>900m):   timber / slash (models 8-11)
    fuel_model = np.empty(shape, dtype=np.int8)
    low  = elevation <  500
    mid  = (elevation >= 500) & (elevation < 900)
    high = elevation >= 900

    fuel_model[low]  = rng.choice([1, 2, 3, 6, 7],  size=low.sum())
    fuel_model[mid]  = rng.choice([4, 5, 6, 7],     size=mid.sum())
    fuel_model[high] = rng.choice([8, 9, 10, 11],   size=high.sum())

    # Sprinkle non-burnable (model 91/93 — water/urban) is outside Anderson 13,
    # so just keep everything burnable for the simulation domain.

    logger.info(
        "Synthetic terrain generated: shape=%s, elev=[%.0f, %.0f] m",
        shape, elevation.min(), elevation.max(),
    )
    return TerrainData(
        elevation=elevation,
        slope=slope,
        aspect=aspect,
        fuel_model=fuel_model,
        resolution_m=resolution_m,
        origin=origin,
        shape=shape,
    )


# ---------------------------------------------------------------------------
# LANDFIRE loader
# ---------------------------------------------------------------------------

# LANDFIRE layer names for each product
_LANDFIRE_LAYERS = {
    "elevation":   "ELEV2020",
    "slope":       "SLP2020",
    "aspect":      "ASP2020",
    "fuel_model":  "220F13_22",   # Anderson 13 fuel models
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
    )
