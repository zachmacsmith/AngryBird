"""Shared utility functions: coordinate projection, distances, taper, thinning."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# UTM projection
# ---------------------------------------------------------------------------

def utm_zone(lon: float) -> int:
    """Return UTM zone number for a given longitude."""
    return int((lon + 180) / 6) + 1


def latlon_to_utm(lat: float, lon: float) -> tuple[float, float, int, str]:
    """
    Convert (lat, lon) in decimal degrees to UTM (easting, northing, zone, hemisphere).
    Uses a simplified spherical approximation sufficient for ~50 km domains.
    Returns (easting_m, northing_m, zone_number, hemisphere).
    """
    zone = utm_zone(lon)
    hemisphere = "N" if lat >= 0 else "S"

    a = 6_378_137.0          # WGS84 semi-major axis
    e2 = 0.00669437999014    # WGS84 first eccentricity squared
    k0 = 0.9996              # scale factor
    E0 = 500_000.0           # false easting
    N0 = 0.0 if hemisphere == "N" else 10_000_000.0

    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)

    N = a / math.sqrt(1 - e2 * math.sin(lat_r) ** 2)
    T = math.tan(lat_r) ** 2
    C = e2 / (1 - e2) * math.cos(lat_r) ** 2
    A = math.cos(lat_r) * (lon_r - lon0)

    e4 = e2 * e2
    e6 = e4 * e2
    M = a * (
        (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * lat_r
        - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * lat_r)
        + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * lat_r)
        - (35 * e6 / 3072) * math.sin(6 * lat_r)
    )

    easting = k0 * N * (
        A + (1 - T + C) * A ** 3 / 6
        + (5 - 18 * T + T ** 2 + 72 * C - 58 * e2 / (1 - e2)) * A ** 5 / 120
    ) + E0

    northing = k0 * (
        M + N * math.tan(lat_r) * (
            A ** 2 / 2
            + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
            + (61 - 58 * T + T ** 2 + 600 * C - 330 * e2 / (1 - e2)) * A ** 6 / 720
        )
    ) + N0

    return easting, northing, zone, hemisphere


def grid_to_latlon(
    row: int,
    col: int,
    origin_lat: float,
    origin_lon: float,
    resolution_m: float,
) -> tuple[float, float]:
    """
    Convert grid (row, col) to (lat, lon).
    Origin is the NW corner. Row increases southward, col increases eastward.
    Approximate: uses metres-per-degree at origin latitude.
    """
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(origin_lat))

    lat = origin_lat - row * resolution_m / m_per_deg_lat
    lon = origin_lon + col * resolution_m / m_per_deg_lon
    return lat, lon


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def euclidean_distance_m(
    r1: int, c1: int, r2: int, c2: int, resolution_m: float
) -> float:
    """Euclidean distance in metres between two grid cells."""
    dr = (r1 - r2) * resolution_m
    dc = (c1 - c2) * resolution_m
    return math.sqrt(dr * dr + dc * dc)


def pairwise_distances(
    locations: Sequence[tuple[int, int]], resolution_m: float
) -> np.ndarray:
    """
    Return (N, N) symmetric distance matrix in metres for a list of grid locations.
    """
    n = len(locations)
    coords = np.array(locations, dtype=np.float64) * resolution_m
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
    return np.sqrt((diff ** 2).sum(axis=-1))


def distance_grid(
    ref_row: int, ref_col: int, shape: tuple[int, int], resolution_m: float
) -> np.ndarray:
    """
    Return float32[rows, cols] array of distances in metres from (ref_row, ref_col).
    """
    rows, cols = shape
    row_idx = np.arange(rows, dtype=np.float32)
    col_idx = np.arange(cols, dtype=np.float32)
    dr = (row_idx - ref_row)[:, None] * resolution_m
    dc = (col_idx - ref_col)[None, :] * resolution_m
    return np.sqrt(dr ** 2 + dc ** 2)


def angular_diff_deg(a: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    """Smallest signed angular difference in degrees (result in [0, 180])."""
    diff = np.abs(np.asarray(a) - np.asarray(b)) % 360
    return np.minimum(diff, 360 - diff)


# ---------------------------------------------------------------------------
# Gaspari-Cohn taper (localization for EnKF)
# ---------------------------------------------------------------------------

def gaspari_cohn(distances: np.ndarray, radius: float) -> np.ndarray:
    """
    Gaspari-Cohn (1999) fifth-order piecewise polynomial taper.
    Returns values in [0, 1]: 1 at distance 0, 0 beyond `radius`.
    `radius` is the half-width at which the function reaches zero.
    """
    r = np.asarray(distances, dtype=np.float64)
    z = r / (radius / 2)   # normalise so z=2 is the zero crossing
    out = np.zeros_like(z)

    m1 = (z <= 1)
    z1 = z[m1]
    out[m1] = (
        1.0
        - (5 / 3) * z1 ** 2
        + (5 / 8) * z1 ** 3
        + (1 / 2) * z1 ** 4
        - (1 / 4) * z1 ** 5
    )

    m2 = (z > 1) & (z <= 2)
    z2 = z[m2]
    out[m2] = (
        4
        - 5 * z2
        + (5 / 3) * z2 ** 2
        + (5 / 8) * z2 ** 3
        - (1 / 2) * z2 ** 4
        + (1 / 12) * z2 ** 5
        - 2 / (3 * z2)
    )

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Observation thinning
# ---------------------------------------------------------------------------

def thin_observations(
    locations: list[tuple[int, int]],
    values: list[float],
    min_spacing_m: float,
    resolution_m: float,
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Greedily keep observations that are at least `min_spacing_m` apart.
    Processes in the order supplied (caller should sort by quality/recency if desired).
    Returns (kept_locations, kept_values).
    """
    kept_locs: list[tuple[int, int]] = []
    kept_vals: list[float] = []
    min_cells = min_spacing_m / resolution_m

    for loc, val in zip(locations, values):
        r, c = loc
        too_close = any(
            math.sqrt((r - kr) ** 2 + (c - kc) ** 2) < min_cells
            for kr, kc in kept_locs
        )
        if not too_close:
            kept_locs.append(loc)
            kept_vals.append(val)

    return kept_locs, kept_vals


# ---------------------------------------------------------------------------
# Bresenham line rasterisation (used by path planner / evaluator)
# ---------------------------------------------------------------------------

def bresenham(
    r0: int, c0: int, r1: int, c1: int
) -> list[tuple[int, int]]:
    """Return grid cells along the line from (r0,c0) to (r1,c1) (inclusive)."""
    cells: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    return cells


# ---------------------------------------------------------------------------
# Jaccard similarity (placement stability metric)
# ---------------------------------------------------------------------------

def jaccard(
    a: list[tuple[int, int]], b: list[tuple[int, int]]
) -> float:
    """Jaccard similarity between two sets of grid locations."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
