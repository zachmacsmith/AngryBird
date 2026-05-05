"""
Ground truth manager: the hidden "true" state that simulated drones observe.

Phase 4b — simulation harness only.

FMC is static (dead fuel moisture changes over hours, not minutes).
Wind evolves: a smooth synoptic drift plus discrete scheduled wind-shift events.
The ground truth fire advances incrementally via GroundTruthFire (fire_oracle.py).

Field names follow the spec:
  fmc              — static FMC field
  wind_speed       — current wind speed (updated each simulation step)
  wind_direction   — current wind direction (updated each simulation step)
  fire             — GroundTruthFire oracle (stepped each simulation step)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter

from angrybird.config import (
    FMC_ASPECT_WEIGHT,
    FMC_CANOPY_WEIGHT,
    FMC_ELEVATION_WEIGHT,
    FMC_MAX_FRACTION,
    FMC_MIN_FRACTION,
    FMC_NOISE_SCALE,
    FMC_TERRAIN_CORR_LENGTH_M,
    FMC_TPI_WEIGHT,
    GRID_RESOLUTION_M,
    TPI_FILTER_SIZE_CELLS,
    WIND_DIR_TERRAIN_CORR_LENGTH_M,
    WIND_DIR_TERRAIN_NOISE_SCALE_DEG,
    WIND_DRIFT_RATE_DEG_PER_HR,
    WIND_SPEED_MAX_MS,
    WIND_SPEED_MIN_MS,
    WIND_SPEED_TERRAIN_CORR_LENGTH_M,
    WIND_SPEED_TERRAIN_NOISE_SCALE,
    WIND_TPI_MODULATION,
    WIND_TURBULENCE_SIGMA_DEG,
    WIND_TURBULENCE_SIGMA_MS,
)
from angrybird.types import TerrainData
from .fire_oracle import GroundTruthFire


# ---------------------------------------------------------------------------
# Wind evolution
# ---------------------------------------------------------------------------

@dataclass
class WindEvent:
    """A scheduled, possibly ramped wind shift."""
    time_s:            float   # simulation time when the shift begins
    direction_change:  float   # degrees (positive = clockwise)
    speed_change:      float   # m/s (positive = faster)
    ramp_duration_s:   float = 0.0  # 0 = instant


def _apply_wind_events(
    base_speed: float,
    base_direction: float,
    t: float,
    events: list[WindEvent],
) -> tuple[float, float]:
    """
    Return scalar (speed, direction) after applying all events up to time t.
    Events with ramp_duration_s > 0 are linearly interpolated.
    """
    speed     = base_speed
    direction = base_direction

    for ev in events:
        if t < ev.time_s:
            continue
        if ev.ramp_duration_s <= 0.0:
            frac = 1.0
        else:
            frac = min(1.0, (t - ev.time_s) / ev.ramp_duration_s)
        speed     += frac * ev.speed_change
        direction += frac * ev.direction_change

    return max(0.5, speed), direction % 360.0


def compute_wind_field(
    base_speed_field: np.ndarray,
    base_direction_field: np.ndarray,
    terrain: TerrainData,
    t: float,
    events: list[WindEvent],
    drift_rate_deg_per_hr: float = WIND_DRIFT_RATE_DEG_PER_HR,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spatially varying wind at simulation time t.

    Steps:
      1. Apply scheduled events + smooth synoptic drift to domain-mean values
      2. Modulate speed by terrain position index (ridges windier, valleys calmer)
      3. Add small-scale Gaussian turbulence (~0.3 m/s, ~3°)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Synoptic drift
    dt_hours  = t / 3600.0
    drift_dir = drift_rate_deg_per_hr * dt_hours

    domain_speed = float(base_speed_field.mean())
    domain_dir   = float(base_direction_field.mean())

    # Apply events to domain-mean values
    eff_speed, eff_dir = _apply_wind_events(
        domain_speed, domain_dir + drift_dir, t, events)

    # Scale the base field to match the new domain-mean speed
    scale     = eff_speed / max(domain_speed, 0.5)
    ws_field  = base_speed_field * scale

    # Shift direction field by the same angular delta
    delta_dir  = (eff_dir - domain_dir) % 360.0
    wd_field   = (base_direction_field + delta_dir) % 360.0

    # Terrain modulation: TPI (ridges windier, valleys calmer)
    elev       = terrain.elevation
    tpi        = elev - uniform_filter(elev, size=TPI_FILTER_SIZE_CELLS)
    tpi_norm   = tpi / (tpi.std() + 1e-6)
    ws_field   = ws_field * (1.0 + WIND_TPI_MODULATION * np.clip(tpi_norm, -1.0, 2.0))

    # Small-scale turbulence
    ws_field  += rng.normal(0.0, WIND_TURBULENCE_SIGMA_MS, terrain.shape).astype(np.float32)
    wd_field  += rng.normal(0.0, WIND_TURBULENCE_SIGMA_DEG, terrain.shape).astype(np.float32)

    ws_field  = np.clip(ws_field, WIND_SPEED_MIN_MS, WIND_SPEED_MAX_MS).astype(np.float32)
    wd_field  = (wd_field % 360.0).astype(np.float32)

    return ws_field, wd_field


# ---------------------------------------------------------------------------
# Ground truth container
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    """
    Hidden "true" state — accessible only to the simulated observer and renderer.

    fmc              : static terrain-structured FMC [rows, cols]
    wind_speed       : current wind speed — updated each timestep [rows, cols]
    wind_direction   : current wind direction — updated each timestep [rows, cols]
    base_wind_speed  : baseline wind speed field used for evolution [rows, cols]
    base_wind_direction: baseline wind direction field [rows, cols]
    wind_events      : scheduled wind shifts
    fire             : GroundTruthFire oracle — stepped each timestep
    shape            : (rows, cols)
    ignition_cells   : fire start locations (row, col) — for RAWS placement exclusion
    """
    fmc:                  np.ndarray         # float32[rows, cols]
    wind_speed:           np.ndarray         # float32[rows, cols], current
    wind_direction:       np.ndarray         # float32[rows, cols], current
    base_wind_speed:      np.ndarray         # float32[rows, cols], for evolution
    base_wind_direction:  np.ndarray         # float32[rows, cols], for evolution
    wind_events:          list[WindEvent]
    fire:                 GroundTruthFire
    shape:                tuple[int, int]
    ignition_cells:       list[tuple[int, int]] = dataclass_field(default_factory=list)
    # Scenario weather — drives Nelson FMC prior and wind prior in the dynamic prior.
    # Represents what an NWP forecast (e.g. HRRR) would provide.
    temperature_c:        float = 30.0       # air temperature °C
    relative_humidity:    float = 0.25       # relative humidity fraction [0, 1]

    @property
    def burned_mask(self) -> np.ndarray:
        """bool[rows, cols] — cells ignited so far."""
        return self.fire.burned_mask

    @property
    def fire_state(self) -> np.ndarray:
        """float32 burn mask for the ensemble fire engine."""
        return self.fire.fire_state


# ---------------------------------------------------------------------------
# Correlated field generator (shared with ensemble perturbations)
# ---------------------------------------------------------------------------

def _draw_correlated_field(
    shape: tuple[int, int],
    correlation_length: float,
    resolution_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit-variance spatially correlated Gaussian field (squared-exponential kernel)."""
    white    = rng.standard_normal(shape).astype(np.float64)
    freqs_r  = np.fft.fftfreq(shape[0], d=resolution_m)
    freqs_c  = np.fft.fftfreq(shape[1], d=resolution_m)
    freq_grid = np.sqrt(freqs_r[:, None] ** 2 + freqs_c[None, :] ** 2)
    kernel   = np.exp(-2.0 * (np.pi * correlation_length * freq_grid) ** 2)
    field    = np.real(np.fft.ifft2(np.fft.fft2(white) * np.sqrt(kernel)))
    std      = field.std()
    if std < 1e-3:
        return np.zeros(shape, dtype=np.float32)
    return (field / std).astype(np.float32)


# Proxy canopy cover per Anderson-13 fuel model
_CANOPY_COVER: dict[int, float] = {
    1: 0.00, 2: 0.10, 3: 0.05,
    4: 0.30, 5: 0.40, 6: 0.20, 7: 0.20,
    8: 0.80, 9: 0.75, 10: 0.70, 11: 0.50, 12: 0.40, 13: 0.30,
}


def _canopy_from_fuel(fuel_model: np.ndarray) -> np.ndarray:
    out = np.zeros(fuel_model.shape, dtype=np.float32)
    for fid, cc in _CANOPY_COVER.items():
        out[fuel_model == fid] = cc
    return out


def _generate_fmc_field(
    terrain: TerrainData, base_fmc: float, rng: np.random.Generator
) -> np.ndarray:
    """Nelson-model-like terrain-structured FMC (dead fuel range [0.02, 0.40])."""
    # South-facing (180°) are sunnier → drier in the Northern Hemisphere → lower FMC.
    # cos(0°)=+1 → north-facing wetter; cos(180°)=-1 → south-facing drier.
    # The original sign was inverted (bug: made south-facing wetter).
    aspect_effect    =  FMC_ASPECT_WEIGHT * np.cos(np.radians(terrain.aspect))
    elev             = terrain.elevation
    elev_norm        = (elev - elev.mean()) / (elev.std() + 1e-6)
    elevation_effect = FMC_ELEVATION_WEIGHT * elev_norm
    tpi              = elev - uniform_filter(elev, size=TPI_FILTER_SIZE_CELLS)
    tpi_effect       = -FMC_TPI_WEIGHT * np.clip(tpi / (tpi.std() + 1e-6), -2.0, 2.0)
    canopy_effect    = FMC_CANOPY_WEIGHT * _canopy_from_fuel(terrain.fuel_model)
    fmc_terrain      = base_fmc + aspect_effect + elevation_effect + tpi_effect + canopy_effect
    noise            = _draw_correlated_field(terrain.shape, FMC_TERRAIN_CORR_LENGTH_M, terrain.resolution_m, rng)
    return np.clip(fmc_terrain + FMC_NOISE_SCALE * noise, FMC_MIN_FRACTION, FMC_MAX_FRACTION).astype(np.float32)


def _generate_base_wind(
    terrain: TerrainData, base_ws: float, base_wd: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Terrain-structured baseline wind fields (before time-evolution)."""
    elev     = terrain.elevation
    tpi      = elev - uniform_filter(elev, size=TPI_FILTER_SIZE_CELLS)
    ws       = base_ws * (1.0 + WIND_TPI_MODULATION * np.clip(tpi / (tpi.std() + 1e-6), -1.0, 2.0))
    ws_noise = _draw_correlated_field(terrain.shape, WIND_SPEED_TERRAIN_CORR_LENGTH_M, terrain.resolution_m, rng)
    ws       = np.clip(ws + WIND_SPEED_TERRAIN_NOISE_SCALE * ws_noise, WIND_SPEED_MIN_MS, WIND_SPEED_MAX_MS).astype(np.float32)
    wd_noise = _draw_correlated_field(terrain.shape, WIND_DIR_TERRAIN_CORR_LENGTH_M, terrain.resolution_m, rng)
    wd       = ((base_wd + WIND_DIR_TERRAIN_NOISE_SCALE_DEG * wd_noise) % 360.0).astype(np.float32)
    return ws, wd


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def generate_ground_truth(
    terrain: TerrainData,
    ignition_cell: "tuple[int, int] | list[tuple[int, int]]",
    base_fmc: float = 0.08,
    base_ws: float = 5.0,
    base_wd: float = 180.0,
    wind_events: Optional[list[WindEvent]] = None,
    seed: int = 0,
    temperature_c: float = 30.0,
    relative_humidity: float = 0.25,
) -> GroundTruth:
    """
    Generate a physically plausible ground truth.

    FMC is static, terrain-structured, bounded [0.02, 0.40].
    Wind starts from a terrain-structured baseline and evolves during the
    simulation via compute_wind_field (call separately each timestep).
    The GroundTruthFire oracle is initialised at ignition_cell and ready
    to be stepped.

    Args:
        terrain:            TerrainData
        ignition_cell:      (row, col) or list of (row, col) fire start locations
        base_fmc:           domain-mean FMC dead fraction (default 0.08)
        base_ws:            domain-mean wind speed m/s (default 5.0)
        base_wd:            prevailing wind direction degrees (default 180)
        wind_events:        scheduled wind shifts (default: none)
        seed:               random seed
        temperature_c:      air temperature °C — drives Nelson FMC prior (default 30)
        relative_humidity:  RH fraction [0,1]  — drives Nelson FMC prior (default 0.25)
    """
    cells: list[tuple[int, int]] = (
        [ignition_cell] if isinstance(ignition_cell, tuple) else list(ignition_cell)
    )

    rng    = np.random.default_rng(seed)
    fmc    = _generate_fmc_field(terrain, base_fmc, rng)
    ws, wd = _generate_base_wind(terrain, base_ws, base_wd, rng)
    fire   = GroundTruthFire(terrain, ignition_cell)

    return GroundTruth(
        fmc=fmc,
        wind_speed=ws.copy(),
        wind_direction=wd.copy(),
        base_wind_speed=ws,
        base_wind_direction=wd,
        wind_events=wind_events or [],
        fire=fire,
        shape=terrain.shape,
        ignition_cells=cells,
        temperature_c=temperature_c,
        relative_humidity=relative_humidity,
    )
