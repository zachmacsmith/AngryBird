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

# Uninformed prior means — used when no physics model (Nelson/NWP) has been set
# and the GP has no observations to fit.  Also used as neutral-ensemble fill values.
GP_DEFAULT_FMC_MEAN: float = 0.10             # fraction
GP_DEFAULT_WIND_SPEED_MEAN: float = 5.0       # m/s
GP_DEFAULT_WIND_DIR_MEAN: float = 270.0       # degrees (westerly)

# Prior variances — reflect full uninformed uncertainty before any observations.
GP_DEFAULT_FMC_VARIANCE: float = 0.04         # std ≈ 0.20 fraction
GP_DEFAULT_WIND_SPEED_VARIANCE: float = 4.0   # std ≈ 2 m/s
GP_DEFAULT_WIND_DIR_VARIANCE: float = 900.0   # std ≈ 30°

# ---------------------------------------------------------------------------
# Physical bounds (applied as clipping limits throughout the codebase)
# ---------------------------------------------------------------------------

FMC_MIN_FRACTION: float = 0.02    # minimum plausible dead fuel moisture
FMC_MAX_FRACTION: float = 0.40    # maximum plausible dead fuel moisture
WIND_SPEED_MIN_MS: float = 0.5    # minimum plausible wind speed (calm)
WIND_SPEED_MAX_MS: float = 25.0   # maximum plausible wind speed (near-hurricane)

# ---------------------------------------------------------------------------
# Ground truth generation parameters
# ---------------------------------------------------------------------------

TPI_FILTER_SIZE_CELLS: int = 20               # uniform filter size for Terrain Position Index
WIND_TPI_MODULATION: float = 0.3              # fractional WS change per unit normalized TPI
WIND_TURBULENCE_SIGMA_MS: float = 0.3         # small-scale wind speed turbulence std dev (m/s)
WIND_TURBULENCE_SIGMA_DEG: float = 3.0        # small-scale wind direction turbulence std dev (°)
WIND_DRIFT_RATE_DEG_PER_HR: float = 5.0       # synoptic wind drift rate (° per hour)

FMC_ASPECT_WEIGHT: float = 0.03               # south-facing aspect drying effect on FMC
FMC_ELEVATION_WEIGHT: float = 0.02            # elevation drying effect on FMC
FMC_TPI_WEIGHT: float = 0.01                  # topographic position effect on FMC
FMC_CANOPY_WEIGHT: float = 0.02               # canopy shading effect on FMC
FMC_NOISE_SCALE: float = 0.015                # amplitude of spatially correlated FMC noise
FMC_TERRAIN_CORR_LENGTH_M: float = 500.0      # correlation length for FMC terrain noise

WIND_SPEED_TERRAIN_CORR_LENGTH_M: float = 1000.0   # correlation length for WS terrain noise
WIND_SPEED_TERRAIN_NOISE_SCALE: float = 1.0         # amplitude of spatially correlated WS noise
WIND_DIR_TERRAIN_CORR_LENGTH_M: float = 2000.0     # correlation length for WD terrain noise
WIND_DIR_TERRAIN_NOISE_SCALE_DEG: float = 15.0     # amplitude of spatially correlated WD noise (°)

# ---------------------------------------------------------------------------
# Information field thresholds
# ---------------------------------------------------------------------------

BIMODAL_UNBURNED_FACTOR: float = 0.9          # arrival < factor*sentinel → treated as burned
FIRE_PERIMETER_LO_PROB: float = 0.1           # lower burn probability bound for active perimeter
FIRE_PERIMETER_HI_PROB: float = 0.9           # upper burn probability bound for active perimeter
BURNED_PROBABILITY_THRESHOLD: float = 0.95    # above this → cell fully burned (zero info value)

# ---------------------------------------------------------------------------
# Drone simulator
# ---------------------------------------------------------------------------

CAMERA_FOOTPRINT_M: float = 100.0             # FMC observation footprint radius (metres)
OBS_FIRE_DEGRADATION_FACTOR: float = 3.0      # sensor sigma multiplier near active fire front
CAMERA_FOOTPRINT_EDGE_NOISE_FACTOR: float = 0.5  # edge noise relative to centre noise

# ---------------------------------------------------------------------------
# Observation pipeline
# ---------------------------------------------------------------------------

OBSERVATION_THINNING_SPACING_M: float = 200.0  # minimum spacing for spatial thinning / buffering

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------

DRONE_ENDURANCE_S: float = 1800.0             # default drone flight endurance per sortie (30 min)
SIM_TOTAL_TIME_S: float = 21600.0             # default total simulation duration (6 hours)
CYCLE_INTERVAL_S: float = CYCLE_INTERVAL_MIN * 60.0  # IGNIS cycle interval in seconds

# ---------------------------------------------------------------------------
# Nelson FMC model parameters
# ---------------------------------------------------------------------------

NELSON_EMC_MIN: float = 0.01                  # lower clip for Fosberg EMC output
NELSON_EMC_MAX: float = 0.50                  # upper clip for Fosberg EMC output
NELSON_LAPSE_RATE_C_PER_M: float = 0.0065    # temperature lapse rate with elevation (°C/m)
NELSON_ELEV_MOISTURE_FACTOR: float = 0.0001  # elevation moisture correction scale
NELSON_ELEV_FACTOR_MIN: float = 0.85          # lower clip for elevation moisture factor
NELSON_ELEV_FACTOR_MAX: float = 1.20          # upper clip for elevation moisture factor
NELSON_SOLAR_WEIGHT: float = 0.25             # solar radiation drying correction weight
NELSON_CANOPY_ATTENUATION: float = 2.5        # Beer-Lambert canopy attenuation exponent
NELSON_DEFAULT_T_C: float = 28.0              # default ambient temperature for Nelson model (°C)
NELSON_DEFAULT_RH: float = 0.20               # default relative humidity for Nelson model

# ---------------------------------------------------------------------------
# QUBO solver parameters
# ---------------------------------------------------------------------------

QUBO_LAMBDA_INFLATION: float = 1.5           # λ inflation for cardinality constraint robustness
QUBO_SA_NUM_READS: int = 1000                # simulated annealing number of reads
QUBO_DWAVE_NUM_READS: int = 100              # D-Wave QPU number of reads

# ---------------------------------------------------------------------------
# Fire front selector thresholds
# ---------------------------------------------------------------------------

FIRE_FRONT_LO_PROB: float = 0.2              # lower burn probability bound for fire front selection
FIRE_FRONT_HI_PROB: float = 0.8              # upper burn probability bound for fire front selection

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
# RAWS station sensor noise
# Fixed ground instruments are more accurate than drone sensors.
# FMC is derived from T+RH (Nelson/Fosberg) rather than multispectral imagery.
# ---------------------------------------------------------------------------

RAWS_FMC_SIGMA: float = 0.01        # ~Nelson residual — T/RH measurement
RAWS_WIND_SPEED_SIGMA: float = 0.5  # cup anemometer (vs 1.0 m/s drone)
RAWS_WIND_DIR_SIGMA: float = 5.0    # wind vane (vs 10° drone)

# ---------------------------------------------------------------------------
# Selector parameters
# ---------------------------------------------------------------------------

IGNIS_SELECTOR: str = "correlation_path"  # "greedy"|"qubo"|"uniform"|"fire_front" (points) | "correlation_path" (paths)
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
# GP temporal decay
# ---------------------------------------------------------------------------

# Per-variable decay timescales (seconds).  Effective sigma = original_sigma * exp(age / tau).
# Observations are pruned when effective sigma exceeds GP_OBS_DECAY_DROP_FACTOR × original.
TAU: dict[str, float] = {
    "fmc_1hr":        3600.0,    # 1 hour — unchanged, well-characterized
    "fmc_10hr":       36000.0,   # 10 hours — unchanged
    "wind_speed":     7200.0,    # 2 hours — mean speed is fairly persistent
    "wind_direction": 3600.0,    # 1 hour — direction drifts faster than speed
    "fire_state":     float('inf'),  # permanent — fire doesn't un-burn
}

TAU_FMC_S: float          = TAU["fmc_1hr"]        # shorthand used by IGNISGPPrior
TAU_WIND_SPEED_S: float   = TAU["wind_speed"]      # shorthand for wind speed GP
TAU_WIND_DIR_S: float     = TAU["wind_direction"]  # shorthand for wind direction GP
GP_OBS_DECAY_DROP_FACTOR: float = 10.0  # drop obs when effective sigma > 10× original

# ---------------------------------------------------------------------------
# Observation aggregation and ensemble noise floors
# ---------------------------------------------------------------------------

AGGREGATION_SIGMA_FLOOR: float = 0.015  # minimum aggregated FMC sigma (systematic error floor)
PROCESS_NOISE_FLOOR: float = 0.01       # minimum FMC perturbation std per ensemble member

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

# Source: Scott & Burgan 2005 (RMRS-GTR-153), Table 7
# "Standard Fire Behavior Fuel Models: A Comprehensive Set for Use
#  with Rothermel's Surface Fire Spread Model"
#
# All 40 burnable fuel models + 5 non-burnable models.
#
# Units in table: load in tons/acre, SAV in 1/ft, depth in ft,
#   heat content in BTU/lb, moisture of extinction in percent.
# Converted here to SI: load in kg/m², SAV in 1/m, depth in m,
#   heat content in J/kg, moisture of extinction as fraction.
#
# Constant parameters (not per-model, per Rothermel 1972):
#   10-hr dead fuel SAV: 109 1/ft  (357.6 1/m)
#   100-hr dead fuel SAV: 30 1/ft  (98.4 1/m)
#   Total mineral content: 0.0555
#   Effective (silica-free) mineral content: 0.0100
#   Ovendry fuel particle density: 32 lb/ft³ (512.7 kg/m³)

T_ACRE_TO_KG_M2 = 0.22417  # tons/acre -> kg/m²
FT_TO_M = 0.3048           # feet -> meters
INV_FT_TO_INV_M = 3.28084  # 1/ft -> 1/m
BTU_LB_TO_J_KG = 2326.0    # BTU/lb -> J/kg

H_STD = 8000 * BTU_LB_TO_J_KG   # 18,608,000 J/kg
H_GR6 = 9000 * BTU_LB_TO_J_KG   # 20,934,000 J/kg  (GR6 is the only exception)

# Per-model constants
SAV_10H = 109 * INV_FT_TO_INV_M   # ~357.6 1/m
SAV_100H = 30 * INV_FT_TO_INV_M   # ~98.4 1/m
MINERAL_TOTAL = 0.0555
MINERAL_EFFECTIVE = 0.0100
PARTICLE_DENSITY = 32 * 16.0185    # lb/ft³ -> kg/m³ ≈ 512.6 kg/m³


def _fm(load_1h, load_10h, load_100h, load_herb, load_woody,
        sav_1h, sav_herb, sav_woody,
        depth, mx, h, dynamic):
    """Build a fuel model dict with unit conversions applied."""
    return {
        # Fuel loads (tons/acre -> kg/m²)
        "load_1h":          load_1h   * T_ACRE_TO_KG_M2,
        "load_10h":         load_10h  * T_ACRE_TO_KG_M2,
        "load_100h":        load_100h * T_ACRE_TO_KG_M2,
        "load_live_herb":   load_herb * T_ACRE_TO_KG_M2,
        "load_live_woody":  load_woody * T_ACRE_TO_KG_M2,
        # Surface-area-to-volume ratios (1/ft -> 1/m)
        "sav_1h":           sav_1h   * INV_FT_TO_INV_M,
        "sav_live_herb":    sav_herb * INV_FT_TO_INV_M,
        "sav_live_woody":   sav_woody * INV_FT_TO_INV_M,
        "sav_10h":          SAV_10H,
        "sav_100h":         SAV_100H,
        # Fuelbed depth (ft -> m)
        "depth":            depth * FT_TO_M,
        # Dead fuel moisture of extinction (percent -> fraction)
        "mx":               mx / 100.0,
        # Heat content (BTU/lb -> J/kg), same for live and dead
        "h":                h * BTU_LB_TO_J_KG,
        # Fuel model type
        "dynamic":          dynamic,
    }


def _nb():
    """Non-burnable: all zeros."""
    return {
        "load_1h": 0, "load_10h": 0, "load_100h": 0,
        "load_live_herb": 0, "load_live_woody": 0,
        "sav_1h": 0, "sav_live_herb": 0, "sav_live_woody": 0,
        "sav_10h": 0, "sav_100h": 0,
        "depth": 0, "mx": 0, "h": 0,
        "dynamic": False,
    }


SB40_FUEL_PARAMS: dict[int, dict] = {

    # ── Non-burnable (NB) ────────────────────────────────────────────
    91:  _nb(),  # NB1  Urban/Developed
    92:  _nb(),  # NB2  Snow/Ice
    93:  _nb(),  # NB3  Agricultural
    98:  _nb(),  # NB8  Open Water
    99:  _nb(),  # NB9  Bare Ground

    # ── Grass (GR) — all dynamic ─────────────────────────────────────
    #         1h    10h   100h  herb  woody  sav1h  savH   savW   depth  mx   h     dyn
    101: _fm( 0.10, 0.00, 0.00, 0.30, 0.00,  2200,  2000,  9999,  0.4,  15,  8000, True),   # GR1
    102: _fm( 0.10, 0.00, 0.00, 1.00, 0.00,  2000,  1800,  9999,  1.0,  15,  8000, True),   # GR2
    103: _fm( 0.10, 0.40, 0.00, 1.50, 0.00,  1500,  1300,  9999,  2.0,  30,  8000, True),   # GR3
    104: _fm( 0.25, 0.00, 0.00, 1.90, 0.00,  2000,  1800,  9999,  2.0,  15,  8000, True),   # GR4
    105: _fm( 0.40, 0.00, 0.00, 2.50, 0.00,  1800,  1600,  9999,  1.5,  40,  8000, True),   # GR5
    106: _fm( 0.10, 0.00, 0.00, 3.40, 0.00,  2200,  2000,  9999,  1.5,  40,  9000, True),   # GR6
    107: _fm( 1.00, 0.00, 0.00, 5.40, 0.00,  2000,  1800,  9999,  3.0,  15,  8000, True),   # GR7
    108: _fm( 0.50, 1.00, 0.00, 7.30, 0.00,  1500,  1300,  9999,  4.0,  30,  8000, True),   # GR8
    109: _fm( 1.00, 1.00, 0.00, 9.00, 0.00,  1800,  1600,  9999,  5.0,  40,  8000, True),   # GR9

    # ── Grass-Shrub (GS) — all dynamic ──────────────────────────────
    121: _fm( 0.20, 0.00, 0.00, 0.50, 0.65,  2000,  1800,  1800,  0.9,  15,  8000, True),   # GS1
    122: _fm( 0.50, 0.50, 0.00, 0.60, 1.00,  2000,  1800,  1800,  1.5,  15,  8000, True),   # GS2
    123: _fm( 0.30, 0.25, 0.00, 1.45, 1.25,  1800,  1600,  1600,  1.8,  40,  8000, True),   # GS3
    124: _fm( 1.90, 0.30, 0.10, 3.40, 7.10,  1800,  1600,  1600,  2.1,  40,  8000, True),   # GS4

    # ── Shrub (SH) — SH1 & SH9 are dynamic, rest are static ────────
    141: _fm( 0.25, 0.25, 0.00, 0.15, 1.30,  2000,  1800,  1600,  1.0,  15,  8000, True),   # SH1
    142: _fm( 1.35, 2.40, 0.75, 0.00, 3.85,  2000,  9999,  1600,  1.0,  15,  8000, False),  # SH2
    143: _fm( 0.45, 3.00, 0.00, 0.00, 6.20,  1600,  9999,  1400,  2.4,  40,  8000, False),  # SH3
    144: _fm( 0.85, 1.15, 0.20, 0.00, 2.55,  2000,  1800,  1600,  3.0,  30,  8000, False),  # SH4
    145: _fm( 3.60, 2.10, 0.00, 0.00, 2.90,   750,  9999,  1600,  6.0,  15,  8000, False),  # SH5
    146: _fm( 2.90, 1.45, 0.00, 0.00, 1.40,   750,  9999,  1600,  2.0,  30,  8000, False),  # SH6
    147: _fm( 3.50, 5.30, 2.20, 0.00, 3.40,   750,  9999,  1600,  6.0,  15,  8000, False),  # SH7
    148: _fm( 2.05, 3.40, 0.85, 0.00, 4.35,   750,  9999,  1600,  3.0,  40,  8000, False),  # SH8
    149: _fm( 4.50, 2.45, 0.00, 1.55, 7.00,   750,  1800,  1500,  4.4,  40,  8000, True),   # SH9

    # ── Timber-Understory (TU) — TU1 & TU3 are dynamic ──────────────
    161: _fm( 0.20, 0.90, 1.50, 0.20, 0.90,  2000,  1800,  1600,  0.6,  20,  8000, True),   # TU1
    162: _fm( 0.95, 1.80, 1.25, 0.00, 0.20,  2000,  9999,  1600,  1.0,  30,  8000, False),  # TU2
    163: _fm( 1.10, 0.15, 0.25, 0.65, 1.10,  1800,  1600,  1400,  1.3,  30,  8000, True),   # TU3
    164: _fm( 4.50, 0.00, 0.00, 0.00, 2.00,  2300,  9999,  2000,  0.5,  12,  8000, False),  # TU4
    165: _fm( 4.00, 4.00, 3.00, 0.00, 3.00,  1500,  9999,   750,  1.0,  25,  8000, False),  # TU5

    # ── Timber Litter (TL) — all static ──────────────────────────────
    181: _fm( 1.00, 2.20, 3.60, 0.00, 0.00,  2000,  9999,  9999,  0.2,  30,  8000, False),  # TL1
    182: _fm( 1.40, 2.30, 2.20, 0.00, 0.00,  2000,  9999,  9999,  0.2,  25,  8000, False),  # TL2
    183: _fm( 0.50, 2.20, 2.80, 0.00, 0.00,  2000,  9999,  9999,  0.3,  20,  8000, False),  # TL3
    184: _fm( 0.50, 1.50, 4.20, 0.00, 0.00,  2000,  9999,  9999,  0.4,  25,  8000, False),  # TL4
    185: _fm( 1.15, 2.50, 4.40, 0.00, 0.00,  2000,  9999,  1600,  0.6,  25,  8000, False),  # TL5
    186: _fm( 2.40, 1.20, 1.20, 0.00, 0.00,  2000,  9999,  9999,  0.3,  25,  8000, False),  # TL6
    187: _fm( 0.30, 1.40, 8.10, 0.00, 0.00,  2000,  9999,  9999,  0.4,  25,  8000, False),  # TL7
    188: _fm( 5.80, 1.40, 1.10, 0.00, 0.00,  1800,  9999,  9999,  0.3,  35,  8000, False),  # TL8
    189: _fm( 6.65, 3.30, 4.15, 0.00, 0.00,  1800,  9999,  1600,  0.6,  35,  8000, False),  # TL9

    # ── Slash-Blowdown (SB) — all static ─────────────────────────────
    201: _fm( 1.50, 3.00, 11.00, 0.00, 0.00, 2000,  9999,  9999,  1.0,  25,  8000, False),  # SB1
    202: _fm( 4.50, 4.25,  4.00, 0.00, 0.00, 2000,  9999,  9999,  1.0,  25,  8000, False),  # SB2
    203: _fm( 5.50, 2.75,  3.00, 0.00, 0.00, 2000,  9999,  9999,  1.2,  25,  8000, False),  # SB3
    204: _fm( 5.25, 3.50,  5.25, 0.00, 0.00, 2000,  9999,  9999,  2.7,  25,  8000, False),  # SB4
}
