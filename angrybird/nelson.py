"""
Nelson (2000) dead fuel moisture content model.

Provides a physics-informed GP prior mean that captures terrain-driven FMC
variation: south-facing slopes dry faster, higher elevations are cooler/moister,
and canopy shading retards drying.

Public surface:
    nelson_fmc_field(terrain, T_C, RH, hour_of_day, ...) → float32[rows, cols]

References:
    Nelson, R.M. 2000. Prediction of diurnal change in 10-h fuel stick
        moisture content. Canadian Journal of Forest Research 30:1071-1087.
    Fosberg, M.A. and Deeming, J.E. 1971. Derivation of the 1- and 10-hour
        timelag fuel moisture calculations for fire danger rating. USDA Forest
        Service Research Note RM-207.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import (
    CANOPY_COVER_FRACTION,
    FMC_MAX_FRACTION,
    FMC_MIN_FRACTION,
    NELSON_CANOPY_ATTENUATION,
    NELSON_ELEV_FACTOR_MAX,
    NELSON_ELEV_FACTOR_MIN,
    NELSON_ELEV_MOISTURE_FACTOR,
    NELSON_EMC_MAX,
    NELSON_EMC_MIN,
    NELSON_LAPSE_RATE_C_PER_M,
    NELSON_SOLAR_WEIGHT,
)
from .types import TerrainData


def nelson_emc(T_C: "float | np.ndarray", RH: "float | np.ndarray") -> np.ndarray:
    """
    Fosberg-Deeming equilibrium moisture content (three-segment).

    T_C: ambient temperature in Celsius
    RH:  relative humidity as fraction (0-1)
    Returns EMC as fraction (0-1).
    """
    H = np.asarray(RH, dtype=np.float64) * 100.0   # convert fraction → percent
    T = np.asarray(T_C, dtype=np.float64)

    emc_pct = np.where(
        H < 10.0,
        0.03229 + 0.281073 * H - 0.000578 * H * T,
        np.where(
            H < 50.0,
            2.22749 + 0.160107 * H - 0.01478 * T,
            21.0606 + 0.005565 * H ** 2 - 0.00035 * H * T - 0.00199 * T,
        ),
    )
    return np.clip(emc_pct / 100.0, NELSON_EMC_MIN, NELSON_EMC_MAX).astype(np.float32)


def solar_correction_factor(
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    canopy_cover: np.ndarray,
    hour_of_day: float,
    latitude_deg: float = 37.0,
) -> np.ndarray:
    """
    Solar radiation factor per cell on [0, 1]: 1 = full direct sun, 0 = shaded/night.

    Uses simplified solar geometry at the equinox (conservative midpoint for
    annual-average FMC estimates).  The returned factor is applied as:
        emc *= (1 - 0.25 × solar_factor)
    so south-facing open slopes have the lowest EMC (driest fuel).

    slope_deg   : float32[rows, cols], degrees
    aspect_deg  : float32[rows, cols], degrees clockwise from north (downslope direction)
    canopy_cover: float32[rows, cols], fraction 0-1
    hour_of_day : solar hour (e.g. 14.0 for 2 pm local solar time)
    latitude_deg: site latitude
    """
    # Solar geometry at equinox (declination = 0)
    hour_angle = math.radians((hour_of_day - 12.0) * 15.0)
    lat_rad    = math.radians(latitude_deg)

    sin_elev = math.sin(lat_rad) * math.cos(hour_angle)  # decl=0 → sin(decl)=0
    # simplified: sin(elev) = cos(lat)·cos(decl)·cos(HA) + sin(lat)·sin(decl)
    #            at decl=0: sin(elev) = cos(lat)·cos(HA)
    sin_elev = float(np.clip(math.cos(lat_rad) * math.cos(hour_angle), 0.0, 1.0))

    if sin_elev < 0.01:   # below horizon or near-horizon → no direct radiation
        return np.zeros_like(slope_deg, dtype=np.float32)

    cos_elev = math.sqrt(max(0.0, 1.0 - sin_elev ** 2))

    # Solar azimuth at equinox (from north, clockwise)
    # At decl=0: sin(az) = -sin(HA)·cos(decl)/cos(elev) = -sin(HA)/cos(elev)
    if cos_elev > 1e-6:
        sin_az = -math.sin(hour_angle) / cos_elev
        sin_az = float(np.clip(sin_az, -1.0, 1.0))
        cos_az_from_south = (math.sin(lat_rad) * sin_elev - 0.0) / (math.cos(lat_rad) * cos_elev + 1e-10)
        sun_az_rad = math.atan2(sin_az, math.cos(math.asin(sin_az)))
        sun_az_deg = (math.degrees(math.asin(sin_az)) + 360.0) % 360.0
        # Simpler: morning sun is in the east (az ~90°), afternoon in the west (az ~270°)
        # At equinox, noon sun is directly south at mid-latitudes
        # az = 90 + 90·sin(hour_angle) approximation for mid-lat equinox
        sun_az_deg = float((180.0 + math.degrees(math.atan2(
            -math.sin(hour_angle),
            -math.cos(lat_rad) * 0.0 + math.sin(lat_rad) * math.cos(hour_angle)
        ))) % 360.0)
    else:
        sun_az_deg = 180.0  # noon → south

    sun_az_rad = math.radians(sun_az_deg)

    # Illumination on a tilted surface: cos(angle between sun ray and surface normal)
    # Normal of a slope (slope_deg, aspect_deg): slope faces aspect direction.
    slope_rad  = np.radians(slope_deg.astype(np.float64))
    aspect_rad = np.radians(aspect_deg.astype(np.float64))

    # cos(angle) = sin(elev)·cos(slope) + cos(elev)·sin(slope)·cos(sun_az - aspect)
    cos_inc = (sin_elev * np.cos(slope_rad)
               + cos_elev * np.sin(slope_rad) * np.cos(sun_az_rad - aspect_rad))
    cos_inc = np.clip(cos_inc, 0.0, 1.0)

    # Canopy attenuation: dense canopy blocks most direct radiation (Beer-Lambert proxy)
    cc = np.asarray(canopy_cover, dtype=np.float64)
    transmittance = np.exp(-NELSON_CANOPY_ATTENUATION * cc)   # T≈1 at cc=0, T≈0.08 at cc=1

    return (cos_inc * transmittance).astype(np.float32)


def nelson_fmc_field(
    terrain: TerrainData,
    T_C: "float | np.ndarray",
    RH: "float | np.ndarray",
    hour_of_day: float,
    latitude_deg: float = 37.0,
    ref_elevation: Optional[float] = None,
) -> np.ndarray:
    """
    Nelson dead fuel FMC estimate at every grid cell.  float32[rows, cols].

    T_C         : ambient temperature in Celsius — scalar or [rows, cols]
    RH          : relative humidity as fraction (0-1) — scalar or [rows, cols]
    hour_of_day : solar hour (e.g. 14.0 for 2 pm local solar time)
    latitude_deg: site latitude for solar geometry
    ref_elevation: elevation at met-station for lapse correction (defaults to domain mean)

    Pipeline per cell:
      1. EMC three-segment Fosberg equation at ambient T/RH
      2. Elevation lapse-rate correction (higher = cooler = moister)
      3. Solar radiation correction (south-facing open → drier)
    """
    rows, cols = terrain.shape

    T_arr  = np.broadcast_to(np.asarray(T_C,  dtype=np.float64), (rows, cols)).copy()
    RH_arr = np.broadcast_to(np.asarray(RH,   dtype=np.float64), (rows, cols)).copy()

    if ref_elevation is None:
        ref_elevation = float(terrain.elevation.mean())

    # Elevation lapse: -0.65 °C / 100 m → higher sites are cooler → higher EMC
    T_local = T_arr - NELSON_LAPSE_RATE_C_PER_M * (terrain.elevation.astype(np.float64) - ref_elevation)

    # Base EMC from Fosberg three-segment equation
    emc = nelson_emc(T_local, RH_arr).astype(np.float64)

    # Elevation moisture correction: simple multiplicative boost per doc spec
    elev_factor = 1.0 + NELSON_ELEV_MOISTURE_FACTOR * (terrain.elevation.astype(np.float64) - ref_elevation)
    emc = emc * np.clip(elev_factor, NELSON_ELEV_FACTOR_MIN, NELSON_ELEV_FACTOR_MAX)

    # Solar radiation drying correction
    cc = (terrain.canopy_cover if terrain.canopy_cover is not None
          else _canopy_cover_from_fuel(terrain.fuel_model))
    solar = solar_correction_factor(terrain.slope, terrain.aspect, cc, hour_of_day, latitude_deg)
    emc = emc * (1.0 - NELSON_SOLAR_WEIGHT * solar.astype(np.float64))

    return np.clip(emc, FMC_MIN_FRACTION, FMC_MAX_FRACTION).astype(np.float32)


def _canopy_cover_from_fuel(fuel_model: np.ndarray) -> np.ndarray:
    """Derive canopy cover fraction from fuel model IDs using proxy table."""
    out = np.zeros(fuel_model.shape, dtype=np.float32)
    for fid, val in CANOPY_COVER_FRACTION.items():
        out[fuel_model == fid] = float(val)
    return out
