"""
LANDFIRE data access — build a TerrainData from downloaded GeoTIFFs.

Two entry points:

    load_from_directory(cache_dir)   — load from a folder of LANDFIRE GeoTIFFs
                                       (downloaded manually from landfire.gov)

    download_terrain(bbox, ...)      — submit a LFPS API job and download
                                       (NOTE: LFPS API is currently unreliable;
                                        prefer manual download + load_from_directory)

Expected directory layout (one sub-folder per product, as LANDFIRE delivers them):

    cache_dir/
        LF20**_Elev_CONUS/   *Elev*.tif
        LF20**_SlpD_CONUS/   *SlpD*.tif   (optional — derived from DEM if absent)
        LF20**_Asp_CONUS/    *Asp*.tif    (optional — derived from DEM if absent)
        LF20**_FBFM40_CONUS/ *FBFM40*.tif
        LF20**_CC_CONUS/     *CC*.tif
        LF20**_CH_CONUS/     *CH*.tif
        LF20**_CBH_CONUS/    *CBH*.tif
        LF20**_CBD_CONUS/    *CBD*.tif

Unit conventions (LANDFIRE data dictionary):
    Elevation  — integer metres
    Slope      — integer degrees
    Aspect     — integer degrees from north (-1 = flat → remapped to 0)
    FBFM40     — integer SB40 codes (91-204); non-burnable 91-99 kept as-is
    CC         — integer percent  (0-100)  → divided by 100 → fraction
    CH         — integer × 10 m  (0-600)  → divided by 10  → metres
    CBH        — integer × 10 m           → divided by 10  → metres
    CBD        — integer × 100 kg/m³      → divided by 100 → kg/m³
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from tqdm import tqdm
import requests
from pyproj import Transformer

from .config import GRID_RESOLUTION_M
from .types import TerrainData


# ---------------------------------------------------------------------------
# LANDFIRE product name fragments used to match files in a cache directory
# ---------------------------------------------------------------------------

_PRODUCT_GLOBS: dict[str, list[str]] = {
    "elevation":           ["*Elev*", "*elev*", "*DEM*"],
    "slope":               ["*SlpD*", "*Slp*", "*slope*"],
    "aspect":              ["*Asp*", "*aspect*"],
    "fuel_model":          ["*FBFM40*", "*fbfm40*"],
    "canopy_cover":        ["*_CC_*", "*canopy_cover*"],
    "canopy_height":       ["*_CH_*", "*canopy_height*"],
    "canopy_base_height":  ["*CBH*", "*canopy_base*"],
    "canopy_bulk_density": ["*CBD*", "*canopy_bulk*"],
}


# ---------------------------------------------------------------------------
# Public: load from local directory
# ---------------------------------------------------------------------------

def load_from_directory(
    cache_dir: str | Path,
    resolution_m: float = GRID_RESOLUTION_M,
) -> TerrainData:
    """
    Build a TerrainData from a folder of LANDFIRE GeoTIFFs.

    Parameters
    ----------
    cache_dir:
        Path to the directory containing LANDFIRE product sub-folders or tifs.
    resolution_m:
        Target resolution in metres.  Defaults to GRID_RESOLUTION_M from config.
        If the source files are already at this resolution no resampling occurs.
    """
    cache_dir = Path(cache_dir)
    tif_paths = _find_tifs(cache_dir)

    if "elevation" not in tif_paths:
        raise FileNotFoundError(
            f"No elevation GeoTIFF found in {cache_dir}. "
            "Expected a file matching *Elev*.tif."
        )

    return _build_terrain_data(tif_paths, resolution_m)


# ---------------------------------------------------------------------------
# Public: LFPS API download (best-effort — API has been unreliable)
# ---------------------------------------------------------------------------

_LFPS_BASE = (
    "https://lfps.usgs.gov/arcgis/rest/services"
    "/LandfireProductService/GPServer/LandfireProductService"
)
_SUBMIT_URL = f"{_LFPS_BASE}/submitJob"
_JOB_URL    = f"{_LFPS_BASE}/jobs/{{job_id}}"

_LFPS_LAYER_CODES: dict[str, str] = {
    "elevation":           "US_220DEM",
    "slope":               "US_220SLPD",
    "aspect":              "US_220ASP",
    "fuel_model":          "US_220FBFM40",
    "canopy_cover":        "US_220CC",
    "canopy_height":       "US_220CH",
    "canopy_base_height":  "US_220CBH",
    "canopy_bulk_density": "US_220CBD",
}

_POLL_INTERVAL_S = 5.0
_POLL_TIMEOUT_S  = 600.0


def download_terrain(
    bbox: tuple[float, float, float, float],
    out_dir: Path | str = Path("landfire_cache"),
    resolution_m: float = GRID_RESOLUTION_M,
    version: str = "220",
    timeout_s: float = _POLL_TIMEOUT_S,
) -> TerrainData:
    """
    Download LANDFIRE layers via the LFPS API and return a TerrainData.

    Parameters
    ----------
    bbox:
        (min_lon, min_lat, max_lon, max_lat) in WGS84 decimal degrees.
    out_dir:
        Directory for cached zip and extracted GeoTIFFs.
    resolution_m:
        Output grid cell size in metres.
    version:
        LANDFIRE version string (default "220" = LF 2022).
    timeout_s:
        Maximum seconds to wait for the LFPS job.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_codes = {k: v.replace("220", version) for k, v in _LFPS_LAYER_CODES.items()}
    tif_paths = _fetch_tifs_from_api(bbox, layer_codes, out_dir, timeout_s)
    return _build_terrain_data(tif_paths, resolution_m)


# ---------------------------------------------------------------------------
# Core builder — shared by both entry points
# ---------------------------------------------------------------------------

def _build_terrain_data(
    tif_paths: dict[str, Path],
    resolution_m: float,
) -> TerrainData:
    """Read GeoTIFFs, reproject/resample to a common UTM grid, assemble TerrainData."""

    elev_arr, ref_transform, ref_crs, ref_shape = _read_and_reproject(
        tif_paths["elevation"], resolution_m=resolution_m,
    )

    def load(key: str) -> np.ndarray:
        if key not in tif_paths:
            return np.zeros(ref_shape, dtype=np.float32)
        arr, _, _, _ = _read_and_reproject(
            tif_paths[key], resolution_m=resolution_m,
            ref_transform=ref_transform, ref_crs=ref_crs, ref_shape=ref_shape,
        )
        return arr

    elevation = _clean(elev_arr)

    slope = _clean(load("slope")) if "slope" in tif_paths else _derive_slope(elevation, resolution_m)

    if "aspect" in tif_paths:
        raw_asp = load("aspect")
        aspect = np.where((raw_asp < 0) | ~np.isfinite(raw_asp), 0.0, raw_asp).astype(np.float32)
    else:
        aspect = _derive_aspect(elevation, resolution_m)

    fuel_model          = np.nan_to_num(load("fuel_model"),          nan=0).astype(np.int16)
    canopy_cover        = np.clip(_clean(load("canopy_cover"))        / 100.0, 0, 1)
    canopy_height       = np.clip(_clean(load("canopy_height"))       / 10.0,  0, None)
    canopy_base_height  = np.clip(_clean(load("canopy_base_height"))  / 10.0,  0, None)
    canopy_bulk_density = np.clip(_clean(load("canopy_bulk_density")) / 100.0, 0, None)

    transformer = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
    lon_nw, lat_nw = transformer.transform(ref_transform.c, ref_transform.f)

    rows, cols = ref_shape
    return TerrainData(
        elevation=elevation,
        slope=slope,
        aspect=aspect,
        fuel_model=fuel_model,
        canopy_cover=canopy_cover,
        canopy_height=canopy_height,
        canopy_base_height=canopy_base_height,
        canopy_bulk_density=canopy_bulk_density,
        shape=(rows, cols),
        resolution_m=resolution_m,
        origin_latlon=(float(lat_nw), float(lon_nw)),
    )


# ---------------------------------------------------------------------------
# Rasterio helpers
# ---------------------------------------------------------------------------

def _read_and_reproject(
    path: Path,
    resolution_m: float,
    ref_transform=None,
    ref_crs=None,
    ref_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, object, object, tuple[int, int]]:
    with rasterio.open(path) as src:
        src_crs = src.crs

        if ref_transform is not None:
            dst_crs       = ref_crs
            dst_transform = ref_transform
            dst_h, dst_w  = ref_shape
        else:
            cx = (src.bounds.left + src.bounds.right) / 2
            cy = (src.bounds.bottom + src.bounds.top) / 2
            t = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            lon_c, lat_c = t.transform(cx, cy)
            dst_crs = _utm_epsg(lat_c, lon_c)

            dst_transform, dst_w, dst_h = calculate_default_transform(
                src_crs, dst_crs, src.width, src.height, *src.bounds,
                resolution=(resolution_m, resolution_m),
            )

        dst = np.zeros((dst_h, dst_w), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )

    return dst, dst_transform, dst_crs, (dst_h, dst_w)


def _clean(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = arr.copy()
    out[~np.isfinite(out)] = fill
    out[out >= 32000] = fill
    return out.astype(np.float32)


def _derive_slope(elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    dy, dx = np.gradient(elevation.astype(np.float64), resolution_m)
    return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)


def _derive_aspect(elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    dy, dx = np.gradient(elevation.astype(np.float64), resolution_m)
    return ((90.0 - np.degrees(np.arctan2(-dy, dx))) % 360.0).astype(np.float32)


def _utm_epsg(lat: float, lon: float) -> str:
    zone = int((lon + 180.0) / 6.0) + 1
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_tifs(cache_dir: Path) -> dict[str, Path]:
    all_tifs = list(cache_dir.rglob("*.tif"))
    result: dict[str, Path] = {}
    for key, globs in _PRODUCT_GLOBS.items():
        for pattern in globs:
            matches = [p for p in all_tifs if p.match(pattern)]
            if matches:
                result[key] = matches[0]
                break
    return result


# ---------------------------------------------------------------------------
# LFPS API helpers
# ---------------------------------------------------------------------------

def _fetch_tifs_from_api(
    bbox: tuple[float, float, float, float],
    layer_codes: dict[str, str],
    out_dir: Path,
    timeout_s: float,
) -> dict[str, Path]:
    layer_list = ";".join(layer_codes.values())
    aoi = f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"

    print("Submitting LANDFIRE Product Service job…")
    resp = requests.post(
        _SUBMIT_URL,
        data={"Layer_List": layer_list, "Area_of_Interest": aoi, "f": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    try:
        body = resp.json()
    except Exception:
        raise RuntimeError(
            f"LFPS returned non-JSON (status {resp.status_code}). "
            "Download manually from landfire.gov and use load_from_directory()."
        )
    if "jobId" not in body:
        raise RuntimeError(f"LFPS did not return a jobId: {body}")
    job_id: str = body["jobId"]
    print(f"Job submitted: {job_id}")

    job_url = _JOB_URL.format(job_id=job_id)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = requests.get(job_url, params={"f": "json"}, timeout=30).json()
        status: str = info.get("jobStatus", "")
        if status == "esriJobSucceeded":
            break
        if status in {"esriJobFailed", "esriJobCancelled", "esriJobTimedOut"}:
            raise RuntimeError(f"LFPS job {job_id} ended with status '{status}'.")
        print(f"  status: {status} — retrying in {_POLL_INTERVAL_S:.0f}s…")
        time.sleep(_POLL_INTERVAL_S)
    else:
        raise TimeoutError(f"LFPS job {job_id} timed out after {timeout_s:.0f}s")

    result_body = requests.get(
        f"{job_url}/results/Output_File", params={"f": "json"}, timeout=30
    ).json()
    download_url: str = result_body["value"]["url"]

    zip_path = out_dir / f"{job_id}.zip"
    _download_file(download_url, zip_path)

    tif_dir = out_dir / job_id
    tif_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tif_dir)

    return _find_tifs(tif_dir)


def _download_file(url: str, out_path: Path) -> None:
    if out_path.exists():
        print(f"Using cached: {out_path.name}")
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            desc=out_path.name, total=total, unit="B", unit_scale=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
