"""
Download LANDFIRE GeoTIFF products for a bounding box and build a TerrainData.

Uses the LANDFIRE Product Service (LFPS) asynchronous REST API:
    https://lfps.usgs.gov/arcgis/rest/services/LandFireProductService/GPServer/LandFireProductService/

Workflow
--------
1. POST a job to LFPS with the bounding-box AOI and desired layer codes.
2. Poll until the job succeeds (or fails / times out).
3. Download the resulting zip, extract GeoTIFFs.
4. Reproject every layer to a common UTM grid at the requested resolution.
5. Return a TerrainData ready for use by SimulationRunner.

Requirements:
    pip install requests tqdm rasterio pyproj
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from tqdm import tqdm
import requests

from .types import TerrainData


# ---------------------------------------------------------------------------
# LFPS API endpoints
# ---------------------------------------------------------------------------

_LFPS_BASE = (
    "https://lfps.usgs.gov/arcgis/rest/services"
    "/LandFireProductService/GPServer/LandFireProductService"
)
_SUBMIT_URL = f"{_LFPS_BASE}/submitJob"
_JOB_URL    = f"{_LFPS_BASE}/jobs/{{job_id}}"

# ---------------------------------------------------------------------------
# LANDFIRE 2022 (version 220) layer codes
# Docs: https://www.landfire.gov/lf_data_access.php
# ---------------------------------------------------------------------------

_LAYERS: dict[str, str] = {
    "elevation":           "US_220DEM",    # Digital Elevation Model, metres
    "slope":               "US_220SLPD",   # Slope, degrees
    "aspect":              "US_220ASP",    # Aspect, degrees from north
    "fuel_model":          "US_220FBFM13", # Anderson 13 fuel model IDs (1-13)
    "canopy_cover":        "US_220CC",     # Canopy cover, percent (0-100)
    "canopy_base_height":  "US_220CBH",    # Canopy base height, metres
    "canopy_bulk_density": "US_220CBD",    # Canopy bulk density, kg/m³ × 100
}

# Anderson 13 fuel model IDs produced by FBFM13.
# LANDFIRE non-burnable codes (91-99) are remapped to 0 (treated as non-fuel
# by the fire engine). Values already 1-13 pass through unchanged.
_NB_CODES = set(range(91, 100))

_POLL_INTERVAL_S = 5.0
_POLL_TIMEOUT_S  = 600.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_terrain(
    bbox: tuple[float, float, float, float],
    out_dir: Path | str = Path("landfire_cache"),
    resolution_m: float = 50.0,
    version: str = "220",
    timeout_s: float = _POLL_TIMEOUT_S,
) -> TerrainData:
    """
    Download LANDFIRE layers for *bbox* and return a TerrainData.

    Parameters
    ----------
    bbox:
        (min_lon, min_lat, max_lon, max_lat) in WGS84 decimal degrees.
    out_dir:
        Directory for cached zip and extracted GeoTIFFs.  Re-runs with the
        same job ID skip the network download.
    resolution_m:
        Output grid cell size in metres (default 50 m).
    version:
        LANDFIRE version number string (default "220" = LF 2022).
        Change to "230" for LF 2023, etc.
    timeout_s:
        Maximum seconds to wait for the LFPS job to complete.

    Returns
    -------
    TerrainData
        All required fields populated; canopy fields are None if LANDFIRE
        did not return the layer.

    Example
    -------
    >>> from angrybird.tif_getter import download_terrain
    >>> terrain = download_terrain((-122.7, 37.6, -122.2, 38.0))
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = {k: v.replace("220", version) for k, v in _LAYERS.items()}

    tif_files = _fetch_tifs(bbox, layers, out_dir, timeout_s)
    return _build_terrain_data(tif_files, bbox, resolution_m)


# ---------------------------------------------------------------------------
# LFPS download helpers
# ---------------------------------------------------------------------------

def _fetch_tifs(
    bbox: tuple[float, float, float, float],
    layers: dict[str, str],
    out_dir: Path,
    timeout_s: float,
) -> dict[str, Path]:
    """Submit LFPS job, poll to completion, extract GeoTIFFs, return paths."""
    layer_list = ";".join(layers.values())

    aoi_json = json.dumps({
        "xmin": bbox[0], "ymin": bbox[1],
        "xmax": bbox[2], "ymax": bbox[3],
        "spatialReference": {"wkid": 4326},
    })

    print("Submitting LANDFIRE Product Service job…")
    resp = requests.post(
        _SUBMIT_URL,
        data={"Layer_list": layer_list, "Area_of_Interest": aoi_json, "f": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    if "jobId" not in body:
        raise RuntimeError(f"LFPS did not return a jobId: {body}")
    job_id: str = body["jobId"]
    print(f"Job submitted: {job_id}")

    # Poll until complete
    job_url = _JOB_URL.format(job_id=job_id)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status_resp = requests.get(job_url, params={"f": "json"}, timeout=30)
        status_resp.raise_for_status()
        info = status_resp.json()
        status: str = info.get("jobStatus", "")
        if status == "esriJobSucceeded":
            print("Job succeeded.")
            break
        if status in {"esriJobFailed", "esriJobCancelled", "esriJobTimedOut"}:
            msgs = info.get("messages", [])
            raise RuntimeError(
                f"LFPS job {job_id} ended with status '{status}'.\n"
                + "\n".join(str(m) for m in msgs)
            )
        print(f"  status: {status} — retrying in {_POLL_INTERVAL_S:.0f}s…")
        time.sleep(_POLL_INTERVAL_S)
    else:
        raise TimeoutError(
            f"LFPS job {job_id} did not complete within {timeout_s:.0f}s"
        )

    # Fetch the download URL from the job results
    result_resp = requests.get(
        f"{job_url}/results/Output_File", params={"f": "json"}, timeout=30
    )
    result_resp.raise_for_status()
    result_body = result_resp.json()
    download_url: str = result_body["value"]["url"]

    # Download the zip (cached by job_id)
    zip_path = out_dir / f"{job_id}.zip"
    _download_file(download_url, zip_path)

    # Extract
    tif_dir = out_dir / job_id
    tif_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tif_dir)

    # Map layer keys → .tif paths
    tif_files: dict[str, Path] = {}
    for key, code in layers.items():
        matches = list(tif_dir.rglob(f"*{code}*.tif"))
        if not matches:
            # Case-insensitive fallback
            matches = [
                p for p in tif_dir.rglob("*.tif")
                if code.lower() in p.name.lower()
            ]
        if matches:
            tif_files[key] = matches[0]
        else:
            print(f"Warning: no .tif found for layer '{key}' (code {code})")

    return tif_files


def _download_file(url: str, out_path: Path) -> None:
    if out_path.exists():
        print(f"Using cached download: {out_path.name}")
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            desc=out_path.name, total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


# ---------------------------------------------------------------------------
# Terrain assembly
# ---------------------------------------------------------------------------

def _build_terrain_data(
    tif_files: dict[str, Path],
    bbox: tuple[float, float, float, float],
    resolution_m: float,
) -> TerrainData:
    """Read GeoTIFFs, reproject to a common UTM grid, assemble a TerrainData."""
    if "elevation" not in tif_files:
        raise RuntimeError(
            "Elevation (DEM) layer was not downloaded — cannot build TerrainData. "
            "Check that 'US_220DEM' is in the LANDFIRE layer list."
        )

    lon_c = (bbox[0] + bbox[2]) / 2.0
    lat_c = (bbox[1] + bbox[3]) / 2.0
    utm_crs = _utm_epsg(lat_c, lon_c)

    # Read elevation first — its reprojected grid defines rows/cols for all others.
    elev_arr, ref_transform, ref_shape = _reproject_tif(
        tif_files["elevation"], utm_crs, resolution_m
    )
    arrays: dict[str, np.ndarray] = {"elevation": elev_arr}

    for key in ("slope", "aspect", "fuel_model",
                "canopy_cover", "canopy_base_height", "canopy_bulk_density"):
        if key not in tif_files:
            continue
        arr, _, _ = _reproject_tif(
            tif_files[key], utm_crs, resolution_m,
            ref_transform=ref_transform, ref_shape=ref_shape,
        )
        arrays[key] = arr

    rows, cols = ref_shape
    elevation = _mask_nodata(arrays["elevation"]).astype(np.float32)

    # Slope and aspect: use LANDFIRE layers when available, derive from DEM otherwise.
    if "slope" in arrays:
        slope = _mask_nodata(arrays["slope"]).astype(np.float32)
    else:
        slope = _derive_slope(elevation, resolution_m)

    if "aspect" in arrays:
        aspect = _mask_nodata(arrays["aspect"]).astype(np.float32)
    else:
        aspect = _derive_aspect(elevation, resolution_m)

    # Fuel model: FBFM13 already gives Anderson 13 codes; remap non-burnable → 0.
    if "fuel_model" in arrays:
        fm_raw = _mask_nodata(arrays["fuel_model"]).astype(np.int16)
        fuel_model = np.where(
            np.isin(fm_raw, list(_NB_CODES)), 0,
            np.clip(fm_raw, 0, 13),
        ).astype(np.int8)
    else:
        # Default to Anderson FM3 (tall grass) if the layer is missing.
        print("Warning: fuel_model layer missing — defaulting to Anderson FM3.")
        fuel_model = np.full((rows, cols), 3, dtype=np.int8)

    # Canopy layers — units from LANDFIRE LF2022 data dictionary:
    #   CC:  integer percent (0-100)  → fraction (0-1)
    #   CBH: metres (stored as is in LF2022)
    #   CBD: kg/m³ × 100             → kg/m³  (divide by 100)
    canopy_cover = None
    if "canopy_cover" in arrays:
        cc_raw = _mask_nodata(arrays["canopy_cover"], fill=0.0)
        canopy_cover = np.clip(cc_raw / 100.0, 0.0, 1.0).astype(np.float32)

    canopy_base_height = None
    if "canopy_base_height" in arrays:
        canopy_base_height = np.clip(
            _mask_nodata(arrays["canopy_base_height"], fill=0.0), 0.0, None
        ).astype(np.float32)

    canopy_bulk_density = None
    if "canopy_bulk_density" in arrays:
        cbd_raw = _mask_nodata(arrays["canopy_bulk_density"], fill=0.0)
        canopy_bulk_density = np.clip(cbd_raw / 100.0, 0.0, None).astype(np.float32)

    # Convert NW-corner UTM coordinates back to (lat, lon).
    origin_lat, origin_lon = _utm_to_latlon(
        utm_crs, ref_transform.c, ref_transform.f
    )

    return TerrainData(
        elevation=elevation,
        slope=slope,
        aspect=aspect,
        fuel_model=fuel_model,
        resolution_m=resolution_m,
        origin=(float(origin_lat), float(origin_lon)),
        shape=(rows, cols),
        canopy_base_height=canopy_base_height,
        canopy_bulk_density=canopy_bulk_density,
        canopy_cover=canopy_cover,
    )


# ---------------------------------------------------------------------------
# Rasterio helpers
# ---------------------------------------------------------------------------

def _reproject_tif(
    path: Path,
    dst_crs: str,
    resolution_m: float,
    ref_transform=None,
    ref_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, object, tuple[int, int]]:
    """
    Read a GeoTIFF, reproject to *dst_crs* at *resolution_m*.

    When *ref_transform* and *ref_shape* are given the output is snapped to
    that grid (used to align all layers with the elevation DEM).
    """
    with rasterio.open(path) as src:
        if ref_transform is not None and ref_shape is not None:
            dst_transform = ref_transform
            dst_height, dst_width = ref_shape
        else:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                resolution=(resolution_m, resolution_m),
            )

        dst_array = np.zeros((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )

    return dst_array, dst_transform, (dst_height, dst_width)


def _mask_nodata(
    arr: np.ndarray,
    fill: float = 0.0,
    nodata_sentinel: float = -9999.0,
) -> np.ndarray:
    """Replace NaN and LANDFIRE nodata sentinels with *fill*."""
    out = arr.copy()
    out[~np.isfinite(out)] = fill
    out[out <= nodata_sentinel] = fill
    return out


# ---------------------------------------------------------------------------
# Coordinate / terrain math helpers
# ---------------------------------------------------------------------------

def _utm_epsg(lat: float, lon: float) -> str:
    """Return the EPSG string for the UTM zone containing (lat, lon)."""
    zone = int((lon + 180.0) / 6.0) + 1
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"


def _utm_to_latlon(utm_crs: str, easting: float, northing: float) -> tuple[float, float]:
    """Convert a UTM coordinate to (lat, lon) in WGS84."""
    import pyproj
    transformer = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return float(lat), float(lon)


def _derive_slope(elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    """Slope in degrees from elevation array via central differences."""
    dy, dx = np.gradient(elevation.astype(np.float64), resolution_m)
    return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)


def _derive_aspect(elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    """Aspect in degrees from north (clockwise) from elevation array."""
    dy, dx = np.gradient(elevation.astype(np.float64), resolution_m)
    return ((90.0 - np.degrees(np.arctan2(-dy, dx))) % 360.0).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download LANDFIRE terrain and print TerrainData summary"
    )
    parser.add_argument(
        "--bbox", nargs=4, type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=[-122.7, 37.6, -122.2, 38.0],
        help="Bounding box in WGS84 degrees (default: SF Bay area)",
    )
    parser.add_argument("--out",        default="landfire_cache",
                        help="Output cache directory")
    parser.add_argument("--resolution", type=float, default=50.0,
                        help="Grid cell size in metres (default 50)")
    parser.add_argument("--version",    default="220",
                        help="LANDFIRE version (default 220 = LF2022)")
    args = parser.parse_args()

    terrain = download_terrain(
        bbox=tuple(args.bbox),
        out_dir=args.out,
        resolution_m=args.resolution,
        version=args.version,
    )
    print(f"\nTerrainData summary:")
    print(f"  shape:       {terrain.shape}")
    print(f"  resolution:  {terrain.resolution_m} m")
    print(f"  origin:      lat={terrain.origin[0]:.5f}, lon={terrain.origin[1]:.5f}")
    print(f"  elevation:   {terrain.elevation.min():.1f} – {terrain.elevation.max():.1f} m")
    print(f"  slope:       {terrain.slope.min():.1f} – {terrain.slope.max():.1f}°")
    print(f"  fuel models: {sorted(np.unique(terrain.fuel_model).tolist())}")
    print(f"  canopy_cover:        {'present' if terrain.canopy_cover is not None else 'None'}")
    print(f"  canopy_base_height:  {'present' if terrain.canopy_base_height is not None else 'None'}")
    print(f"  canopy_bulk_density: {'present' if terrain.canopy_bulk_density is not None else 'None'}")
