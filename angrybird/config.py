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
# Scott & Burgan 40 fuel model parameters (Andrews 2012, RMRS-GTR-265)
#
# Keys: SB40 fuel model codes (integers)
# Values:
#   load   — total 1-hr dead fuel load, kg/m²
#   sav    — surface-area-to-volume ratio of 1-hr fuel, 1/m
#   depth  — fuel bed depth, m
#   mx     — dead fuel moisture of extinction, fraction
#   h      — dead fuel heat content, kJ/kg
#   cbh    — default canopy base height, m (0 = surface fuel only)
#   cbd    — default canopy bulk density, kg/m³ (0 = no canopy)
#   cc     — default canopy cover, fraction (0 = open)
#   ch     — default canopy height, m (0 = no canopy)
#
# Non-burnable codes (91-99) are included with zero load so the lookup
# never raises KeyError for cells ELMFIRE skips anyway.
# ---------------------------------------------------------------------------

FUEL_PARAMS: dict[int, dict[str, float]] = {
    # --- Grass (GR) ---
    101: {"load": 0.045, "sav": 11483, "depth": 0.305, "mx": 0.15, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},
    102: {"load": 0.045, "sav": 11483, "depth": 0.305, "mx": 0.15, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},
    103: {"load": 0.057, "sav":  9840, "depth": 0.762, "mx": 0.25, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},
    104: {"load": 0.114, "sav":  9840, "depth": 2.134, "mx": 0.15, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},
    105: {"load": 0.114, "sav": 11483, "depth": 1.829, "mx": 0.40, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},
    106: {"load": 0.319, "sav":  7546, "depth": 1.829, "mx": 0.40, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},
    107: {"load": 0.479, "sav":  7546, "depth": 1.829, "mx": 0.15, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},
    108: {"load": 0.785, "sav":  7546, "depth": 1.524, "mx": 0.30, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},
    109: {"load": 1.046, "sav":  7546, "depth": 1.524, "mx": 0.40, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},
    # --- Grass-Shrub (GS) ---
    121: {"load": 0.045, "sav":  6890, "depth": 0.610, "mx": 0.15, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.10, "ch": 0.6},
    122: {"load": 0.114, "sav":  6890, "depth": 0.915, "mx": 0.15, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.20, "ch": 0.6},
    123: {"load": 0.114, "sav":  6890, "depth": 1.829, "mx": 0.40, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.30, "ch": 1.2},
    124: {"load": 0.239, "sav":  6890, "depth": 1.829, "mx": 0.40, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.40, "ch": 1.8},
    # --- Shrub (SH) ---
    141: {"load": 0.034, "sav":  5249, "depth": 0.610, "mx": 0.15, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.20, "ch": 0.6},
    142: {"load": 0.114, "sav":  4593, "depth": 1.219, "mx": 0.15, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.30, "ch": 1.0},
    143: {"load": 0.068, "sav":  4921, "depth": 1.219, "mx": 0.40, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.20, "ch": 1.2},
    144: {"load": 0.239, "sav":  4921, "depth": 2.438, "mx": 0.40, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.50, "ch": 2.0},
    145: {"load": 0.182, "sav":  4921, "depth": 1.829, "mx": 0.15, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.30, "ch": 1.5},
    146: {"load": 0.568, "sav":  4921, "depth": 1.829, "mx": 0.30, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.60, "ch": 2.0},
    147: {"load": 0.739, "sav":  4593, "depth": 1.524, "mx": 0.15, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.50, "ch": 1.8},
    149: {"load": 1.046, "sav":  4593, "depth": 1.524, "mx": 0.40, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.70, "ch": 2.4},
    # --- Timber Understory (TU) ---
    161: {"load": 0.045, "sav":  6562, "depth": 0.305, "mx": 0.20, "h": 18608,
          "cbh": 0.5, "cbd": 0.020, "cc": 0.20, "ch": 5.0},
    162: {"load": 0.091, "sav":  6890, "depth": 0.305, "mx": 0.30, "h": 18608,
          "cbh": 0.5, "cbd": 0.040, "cc": 0.30, "ch": 8.0},
    163: {"load": 0.136, "sav":  4921, "depth": 0.305, "mx": 0.30, "h": 18608,
          "cbh": 0.3, "cbd": 0.030, "cc": 0.30, "ch": 10.0},
    164: {"load": 0.205, "sav":  4921, "depth": 0.610, "mx": 0.12, "h": 18608,
          "cbh": 0.5, "cbd": 0.060, "cc": 0.40, "ch": 12.0},
    165: {"load": 0.182, "sav":  4921, "depth": 0.914, "mx": 0.25, "h": 18608,
          "cbh": 0.5, "cbd": 0.050, "cc": 0.70, "ch": 15.0},
    # --- Timber Litter (TL) ---
    181: {"load": 0.068, "sav":  6562, "depth": 0.061, "mx": 0.30, "h": 18608,
          "cbh": 1.0, "cbd": 0.050, "cc": 0.60, "ch": 15.0},
    182: {"load": 0.136, "sav":  6562, "depth": 0.061, "mx": 0.25, "h": 18608,
          "cbh": 1.0, "cbd": 0.060, "cc": 0.60, "ch": 15.0},
    183: {"load": 0.205, "sav":  4921, "depth": 0.061, "mx": 0.20, "h": 18608,
          "cbh": 0.5, "cbd": 0.070, "cc": 0.60, "ch": 15.0},
    184: {"load": 0.273, "sav":  4921, "depth": 0.061, "mx": 0.25, "h": 18608,
          "cbh": 1.5, "cbd": 0.080, "cc": 0.80, "ch": 20.0},
    185: {"load": 0.318, "sav":  4921, "depth": 0.122, "mx": 0.25, "h": 18608,
          "cbh": 1.0, "cbd": 0.070, "cc": 0.70, "ch": 20.0},
    186: {"load": 0.375, "sav":  4921, "depth": 0.061, "mx": 0.25, "h": 18608,
          "cbh": 1.5, "cbd": 0.090, "cc": 0.70, "ch": 22.0},
    187: {"load": 0.568, "sav":  4921, "depth": 0.061, "mx": 0.25, "h": 18608,
          "cbh": 2.0, "cbd": 0.100, "cc": 0.80, "ch": 25.0},
    188: {"load": 0.909, "sav":  4921, "depth": 0.061, "mx": 0.35, "h": 18608,
          "cbh": 0.5, "cbd": 0.080, "cc": 0.50, "ch": 12.0},
    189: {"load": 0.136, "sav":  4921, "depth": 0.061, "mx": 0.35, "h": 18608,
          "cbh": 1.0, "cbd": 0.040, "cc": 0.60, "ch": 15.0},
    # --- Slash-Blowdown (SB) ---
    201: {"load": 0.682, "sav":  4921, "depth": 0.914, "mx": 0.25, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},
    202: {"load": 1.364, "sav":  4921, "depth": 0.914, "mx": 0.25, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},
    203: {"load": 1.364, "sav":  4921, "depth": 1.219, "mx": 0.25, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},
    204: {"load": 2.728, "sav":  4921, "depth": 2.743, "mx": 0.25, "h": 18608,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},
    # --- Non-burnable ---
    91:  {"load": 0.000, "sav":     0, "depth": 0.000, "mx": 0.00, "h":     0,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},  # Urban
    92:  {"load": 0.000, "sav":     0, "depth": 0.000, "mx": 0.00, "h":     0,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},  # Snow/Ice
    93:  {"load": 0.000, "sav":     0, "depth": 0.000, "mx": 0.00, "h":     0,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},  # Agriculture
    98:  {"load": 0.000, "sav":     0, "depth": 0.000, "mx": 0.00, "h":     0,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},  # Water
    99:  {"load": 0.000, "sav":     0, "depth": 0.000, "mx": 0.00, "h":     0,
          "cbh": 0.0, "cbd": 0.000, "cc": 0.00, "ch": 0.0},  # Bare ground
}

# Packing ratio per SB40 model — derived from load/depth/particle_density
# β = load / (depth × particle_density), particle_density = 512 kg/m³
FUEL_PACKING_RATIO: dict[int, float] = {
    fid: p["load"] / max(p["depth"] * 512.0, 1e-6)
    for fid, p in FUEL_PARAMS.items()
}

# Mineral content — standard Rothermel values (unchanged from Anderson-13)
FUEL_MINERAL_CONTENT: float = 0.0555
FUEL_MINERAL_SILICA_FREE: float = 0.0100
FUEL_PARTICLE_DENSITY: float = 512.0  # kg/m³

# Canopy cover fraction per SB40 model — extracted from FUEL_PARAMS for
# fast vectorised lookup in ground_truth.py
SB40_CANOPY_COVER: dict[int, float] = {
    fid: p["cc"] for fid, p in FUEL_PARAMS.items()
}
