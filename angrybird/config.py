"""Global constants and lookup tables for IGNIS."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Grid / simulation
# ---------------------------------------------------------------------------

GRID_RESOLUTION_M: float = 50.0       # meters per cell
ENSEMBLE_SIZE: int = 100              # number of ensemble members
SIMULATION_HORIZON_MIN: int = 60      # fire spread simulation duration, minutes
TIME_STEP_S: float = 30.0             # CA time step, seconds
CYCLE_INTERVAL_MIN: int = 20          # orchestrator re-solve cadence

# ---------------------------------------------------------------------------
# GP hyperparameters (defaults — fitted from RAWS data when available)
# ---------------------------------------------------------------------------

GP_CORRELATION_LENGTH_FMC_M: float = 1500.0   # ~1-2 km for fuel moisture
GP_CORRELATION_LENGTH_WIND_M: float = 5000.0  # ~5 km for wind
GP_TERRAIN_ALPHA: float = 0.001               # elevation difference weight in terrain distance
GP_TERRAIN_BETA: float = 0.005                # aspect difference weight in terrain distance
GP_NOISE_FMC: float = 0.05                    # RAWS FMC observation noise std dev
GP_NOISE_WIND_SPEED: float = 1.0              # RAWS wind speed noise std dev (m/s)
GP_NOISE_WIND_DIR: float = 10.0               # RAWS wind dir noise std dev (degrees)

# ---------------------------------------------------------------------------
# Drone / sensor parameters
# ---------------------------------------------------------------------------

N_DRONES: int = 5
DRONE_RANGE_M: float = 20_000.0               # max one-way range in meters
DRONE_SPEED_MS: float = 15.0                  # cruise speed m/s
SENSOR_FMC_R2: float = 0.86                   # multispectral FMC measurement R²
SENSOR_WIND_ACCURACY: float = 0.90            # anemometer baseline accuracy
CAMERA_FOOTPRINT_CELLS: int = 1               # half-width of camera swath (cells each side)
FIRE_DEGRADATION_RADIUS_M: float = 500.0      # sensor degrades within this distance of active fire

# ---------------------------------------------------------------------------
# Observation noise (for simulated observer)
# ---------------------------------------------------------------------------

OBS_FMC_SIGMA: float = 0.05    # fraction
OBS_WIND_SPEED_SIGMA: float = 1.0  # m/s
OBS_WIND_DIR_SIGMA: float = 10.0   # degrees

# ---------------------------------------------------------------------------
# Selector parameters
# ---------------------------------------------------------------------------

MIN_SELECTION_SPACING_M: float = 500.0   # minimum distance between selected locations
QUBO_MAX_CANDIDATES: int = 300           # M: top-M candidates passed to QUBO

# ---------------------------------------------------------------------------
# EnKF parameters
# ---------------------------------------------------------------------------

ENKF_INFLATION_FACTOR: float = 1.05          # multiplicative covariance inflation
ENKF_LOCALIZATION_RADIUS_M: float = 10_000.0 # Gaspari-Cohn localization radius
ENKF_OUTLIER_THRESHOLD: float = 3.0          # flag obs if > N sigma from ensemble mean

# ---------------------------------------------------------------------------
# Replan triggers
# ---------------------------------------------------------------------------

REPLAN_VARIANCE_REDUCTION_THRESHOLD: float = 0.20  # 20% drop → flag
REPLAN_WIND_SHIFT_THRESHOLD_DEG: float = 30.0       # 30° shift → immediate replan

# ---------------------------------------------------------------------------
# Anderson 13 fuel model parameters (Andrews 2018, RMRS-GTR-371)
# Keys: fuel model ID 1-13
# Values: load (kg/m²), sav (1/m), depth (m), mx (fraction), h (kJ/kg)
#   load  = total 1-hr fuel load
#   sav   = surface-area-to-volume ratio
#   depth = fuel bed depth
#   mx    = moisture of extinction
#   h     = dead fuel heat content
# ---------------------------------------------------------------------------

FUEL_PARAMS: dict[int, dict[str, float]] = {
    1:  {"load": 0.034, "sav": 11483, "depth": 0.305, "mx": 0.12, "h": 18608},
    2:  {"load": 0.092, "sav":  9843, "depth": 0.305, "mx": 0.15, "h": 18608},
    3:  {"load": 0.230, "sav":  4921, "depth": 0.762, "mx": 0.25, "h": 18608},
    4:  {"load": 0.230, "sav":  6562, "depth": 1.829, "mx": 0.20, "h": 18608},
    5:  {"load": 0.046, "sav":  6562, "depth": 0.610, "mx": 0.20, "h": 18608},
    6:  {"load": 0.069, "sav":  5741, "depth": 0.762, "mx": 0.25, "h": 18608},
    7:  {"load": 0.052, "sav":  5741, "depth": 0.762, "mx": 0.40, "h": 18608},
    8:  {"load": 0.069, "sav":  6562, "depth": 0.061, "mx": 0.30, "h": 18608},
    9:  {"load": 0.134, "sav":  8202, "depth": 0.061, "mx": 0.25, "h": 18608},
    10: {"load": 0.138, "sav":  6562, "depth": 0.305, "mx": 0.25, "h": 18608},
    11: {"load": 0.069, "sav":  4921, "depth": 0.305, "mx": 0.15, "h": 18608},
    12: {"load": 0.184, "sav":  4921, "depth": 0.701, "mx": 0.20, "h": 18608},
    13: {"load": 0.322, "sav":  4921, "depth": 0.914, "mx": 0.25, "h": 18608},
}

# Packing ratio (relative bulk density) per fuel model — approximate from Anderson (1982)
FUEL_PACKING_RATIO: dict[int, float] = {
    1: 0.00154, 2: 0.00307, 3: 0.00307, 4: 0.00615, 5: 0.00154,
    6: 0.00231, 7: 0.00231, 8: 0.00154, 9: 0.00231, 10: 0.00308,
    11: 0.00154, 12: 0.00308, 13: 0.00385,
}

# Mineral content (silica-free) — standard Rothermel value
FUEL_MINERAL_CONTENT: float = 0.0555
FUEL_MINERAL_SILICA_FREE: float = 0.0100
FUEL_PARTICLE_DENSITY: float = 512.0  # kg/m³

# ---------------------------------------------------------------------------
# Canopy proxy values per Anderson-13 fuel model
# Used when LANDFIRE canopy layers are unavailable.
# CBH = canopy base height (m), CBD = canopy bulk density (kg/m³),
# CC  = canopy cover fraction (0-1).
# ---------------------------------------------------------------------------

CANOPY_CBH_M: dict[int, float] = {
    1: 0.0,  2: 0.0,  3: 0.0,
    4: 1.0,  5: 1.5,  6: 1.0,  7: 1.0,
    8: 2.5,  9: 3.0,  10: 4.0, 11: 2.0, 12: 1.5, 13: 2.0,
}

CANOPY_CBD_KGM3: dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.00,
    4: 0.05, 5: 0.05, 6: 0.05, 7: 0.08,
    8: 0.12, 9: 0.10, 10: 0.15, 11: 0.07, 12: 0.10, 13: 0.12,
}

CANOPY_COVER_FRACTION: dict[int, float] = {
    1: 0.00, 2: 0.10, 3: 0.05,
    4: 0.30, 5: 0.40, 6: 0.20, 7: 0.20,
    8: 0.80, 9: 0.75, 10: 0.70, 11: 0.50, 12: 0.40, 13: 0.30,
}
