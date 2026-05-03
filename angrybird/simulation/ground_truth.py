"""
Ground truth manager: the hidden "true" state that simulated drones observe.

Phase 4b — simulation harness only. Does not ship with the production system.

The GroundTruth is generated from terrain structure (aspect, elevation, TPI,
canopy) plus a smooth spatially correlated random field.  It is UNKNOWN to
every other component; only the SimulatedObserver has a reference to it.

FMC is bounded dead-fuel range [0.02, 0.40], spatially correlated at ~500 m.
Two cells 50 m apart on the same slope are nearly identical; two cells 50 m
apart across a ridge can differ by 5-10% absolute — matching observed FMC
statistics in Western US chaparral and timber-grass zones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from ..config import GRID_RESOLUTION_M
from ..types import TerrainData


def _draw_correlated_field(
    shape: tuple[int, int],
    correlation_length: float,
    resolution_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw a unit-variance spatially correlated Gaussian field via squared-
    exponential spectral filtering (same kernel used by the GP ensemble draws).

    If correlation_length >> domain the FFT produces a near-constant field
    (std ~ 1e-6); normalising would amplify it to garbage.  Guard: return
    zeros when raw std < 1e-3 so callers' scaling factor still makes sense.
    """
    white = rng.standard_normal(shape).astype(np.float64)
    freqs_r = np.fft.fftfreq(shape[0], d=resolution_m)
    freqs_c = np.fft.fftfreq(shape[1], d=resolution_m)
    freq_grid = np.sqrt(freqs_r[:, None] ** 2 + freqs_c[None, :] ** 2)
    kernel_fft = np.exp(-2.0 * (np.pi * correlation_length * freq_grid) ** 2)
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * np.sqrt(kernel_fft)))
    std = field.std()
    if std < 1e-3:
        return np.zeros(shape, dtype=np.float32)
    return (field / std).astype(np.float32)


# Proxy canopy cover fraction per Anderson-13 fuel model
# Grass (1-3): open; Shrub/brush (4-7): partial; Timber (8-13): closed.
_CANOPY_COVER: dict[int, float] = {
    1: 0.00, 2: 0.10, 3: 0.05,
    4: 0.30, 5: 0.40, 6: 0.20, 7: 0.20,
    8: 0.80, 9: 0.75, 10: 0.70, 11: 0.50, 12: 0.40, 13: 0.30,
}


def _canopy_from_fuel(fuel_model: np.ndarray) -> np.ndarray:
    """Vectorised canopy-cover lookup from Anderson-13 fuel model IDs."""
    out = np.zeros(fuel_model.shape, dtype=np.float32)
    for fid, cc in _CANOPY_COVER.items():
        out[fuel_model == fid] = cc
    return out


def _generate_fmc_field(
    terrain: TerrainData,
    base_fmc: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Nelson-model-like terrain-structured FMC field (dead fuel range).

    Deterministic terrain component:
      - aspect:    south-facing (180°) are drier   → ±3%
      - elevation: higher = wetter (orographic)    → ±2%
      - TPI:       ridge tops drier than valleys   → ±1%
      - canopy:    denser canopy retains moisture  →  0–2%

    Smooth correlated random field (~500 m) captures weather micro-patterns
    and soil variation between sampling points.
    """
    aspect_effect = -0.03 * np.cos(np.radians(terrain.aspect))

    elev = terrain.elevation
    elev_norm = (elev - elev.mean()) / (elev.std() + 1e-6)
    elevation_effect = 0.02 * elev_norm

    tpi = elev - uniform_filter(elev, size=20)
    tpi_effect = -0.01 * np.clip(tpi / (tpi.std() + 1e-6), -2.0, 2.0)

    canopy_effect = 0.02 * _canopy_from_fuel(terrain.fuel_model)

    fmc_terrain = base_fmc + aspect_effect + elevation_effect + tpi_effect + canopy_effect

    noise = _draw_correlated_field(
        terrain.shape, correlation_length=500.0,
        resolution_m=terrain.resolution_m, rng=rng,
    )

    return np.clip(fmc_terrain + 0.015 * noise, 0.02, 0.40).astype(np.float32)


def _generate_wind_fields(
    terrain: TerrainData,
    base_ws: float,
    base_wd: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Terrain-structured wind speed and direction fields.

    Speed: ridge tops windier (positive TPI), valleys sheltered (negative TPI),
    plus smooth ~1 km-scale noise (gusts, turbulence).  Bounded [0.5, 25] m/s.

    Direction: prevailing direction plus smooth ~2 km-scale perturbation
    (valley channelling, mesoscale variation).  ±15° typical deviation.
    """
    elev = terrain.elevation
    tpi = elev - uniform_filter(elev, size=20)
    tpi_norm = tpi / (tpi.std() + 1e-6)
    ws_terrain = base_ws * (1.0 + 0.3 * np.clip(tpi_norm, -1.0, 2.0))

    ws_noise = _draw_correlated_field(
        terrain.shape, correlation_length=1000.0,
        resolution_m=terrain.resolution_m, rng=rng,
    )
    ws = np.clip(ws_terrain + 1.0 * ws_noise, 0.5, 25.0).astype(np.float32)

    wd_noise = _draw_correlated_field(
        terrain.shape, correlation_length=2000.0,
        resolution_m=terrain.resolution_m, rng=rng,
    )
    wd = ((base_wd + 15.0 * wd_noise) % 360.0).astype(np.float32)

    return ws, wd


@dataclass
class GroundTruth:
    """Hidden "true" state — accessible only to the simulated observer."""
    fmc_field:        np.ndarray   # float32[rows, cols]
    wind_speed_field: np.ndarray   # float32[rows, cols], m/s
    wind_dir_field:   np.ndarray   # float32[rows, cols], degrees from north
    shape: tuple[int, int]


def generate_ground_truth(
    terrain: TerrainData,
    base_fmc: float = 0.08,
    base_ws: float = 5.0,
    base_wd: float = 180.0,
    seed: int = 0,
) -> GroundTruth:
    """
    Generate a physically plausible ground truth driven by terrain structure.

    FMC is in the dead-fuel range [0.02, 0.40] with spatial correlation ~500 m.
    Two points 50 m apart on the same slope are nearly identical; two points
    50 m apart across a ridge can differ 5-10% absolute.

    Wind speed is TPI-driven (ridges windier) plus ~1 km-scale noise.
    Wind direction is mostly uniform plus ~2 km-scale smooth perturbation.

    Args:
        terrain:  terrain data (elevation, aspect, fuel model, resolution_m)
        base_fmc: domain-mean FMC dead fuel fraction (default 0.08)
        base_ws:  domain-mean wind speed m/s (default 5.0)
        base_wd:  prevailing wind direction degrees from north (default 180)
        seed:     random seed for reproducibility

    Returns:
        GroundTruth with float32 field arrays.
    """
    rng = np.random.default_rng(seed)
    fmc = _generate_fmc_field(terrain, base_fmc, rng)
    ws, wd = _generate_wind_fields(terrain, base_ws, base_wd, rng)
    return GroundTruth(
        fmc_field=fmc,
        wind_speed_field=ws,
        wind_dir_field=wd,
        shape=terrain.shape,
    )
