"""
Pre-defined simulation scenarios.

Each factory returns (TerrainData, GroundTruth, SimulationConfig) ready to
pass directly to SimulationRunner.  All scenarios share the same 200×200 grid
(10 km × 10 km at 50 m resolution).

Scenarios
---------
hilly_heterogeneous  Primary demo. Complex ridge/valley terrain, mixed fuels,
                     spatially variable FMC. Targeted placement shows the
                     largest advantage over uniform sampling.

wind_shift           Stress test. Same base terrain, but a 45° wind shift
                     event at hour 3 forces system re-routing and shows
                     real-time adaptability.

flat_homogeneous     Control. Flat terrain, uniform grass fuel (model 3),
                     constant wind. Targeted vs uniform advantage is minimal —
                     provides the lower bound for PERR improvement.
"""

from __future__ import annotations

import numpy as np

from ..config import CANOPY_CBH_M, CANOPY_CBD_KGM3, CANOPY_COVER_FRACTION
from ..types import TerrainData
from .ground_truth import WindEvent, generate_ground_truth, GroundTruth
from .runner import SimulationConfig


def _canopy_arrays(
    fuel_model: np.ndarray,
    rng: np.random.Generator,
    jitter: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CBH / CBD / CC arrays from proxy tables with spatial jitter."""
    cbh = np.zeros(fuel_model.shape, dtype=np.float32)
    cbd = np.zeros(fuel_model.shape, dtype=np.float32)
    cc  = np.zeros(fuel_model.shape, dtype=np.float32)
    for fid in CANOPY_CBH_M:
        mask = fuel_model == fid
        cbh[mask] = CANOPY_CBH_M[fid]
        cbd[mask] = CANOPY_CBD_KGM3[fid]
        cc[mask]  = CANOPY_COVER_FRACTION[fid]
    j = rng.uniform(1.0 - jitter, 1.0 + jitter, fuel_model.shape).astype(np.float32)
    return (np.clip(cbh * j, 0.0, None),
            np.clip(cbd * j, 0.0, None),
            np.clip(cc  * j, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Shared terrain builders
# ---------------------------------------------------------------------------

def _hilly_terrain(rows: int = 200, cols: int = 200,
                   resolution_m: float = 50.0) -> TerrainData:
    """Complex ridge/valley terrain with mixed Anderson-13 fuel models."""
    rng = np.random.default_rng(1)
    r = np.linspace(0, 1, rows)
    c = np.linspace(0, 1, cols)
    RI, CI = np.meshgrid(r, c, indexing="ij")

    ridge  =  200.0 * np.exp(-((RI - 0.35) ** 2 + (CI - 0.55) ** 2) / 0.07)
    valley = -80.0  * np.exp(-((RI - 0.70) ** 2 + (CI - 0.80) ** 2) / 0.04)
    noise  = rng.normal(0, 6, (rows, cols))
    elevation = (100.0 + ridge + valley + noise).astype(np.float32)

    dy, dx = np.gradient(elevation, resolution_m, resolution_m)
    slope  = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2))).astype(np.float32)
    aspect = (np.degrees(np.arctan2(-dx, dy)) % 360).astype(np.float32)

    # Mixed fuels: grass (3) on steep slopes, rough (7) in deep valley, shrub (5) elsewhere
    fuel = np.where(slope > 10, 3,
           np.where(elevation < 110, 7, 5)).astype(np.int8)

    cbh, cbd, cc = _canopy_arrays(fuel, rng)
    return TerrainData(
        elevation=elevation, slope=slope, aspect=aspect, fuel_model=fuel,
        resolution_m=resolution_m, origin=(37.5, -119.5), shape=(rows, cols),
        canopy_base_height=cbh, canopy_bulk_density=cbd, canopy_cover=cc,
    )


def _flat_terrain(rows: int = 200, cols: int = 200,
                  resolution_m: float = 50.0) -> TerrainData:
    """Flat terrain with uniform grass fuel (Anderson model 3)."""
    rng = np.random.default_rng(2)
    elevation = (100.0 + rng.normal(0, 1, (rows, cols))).astype(np.float32)
    slope  = np.zeros((rows, cols), dtype=np.float32)
    aspect = np.full((rows, cols), 180.0, dtype=np.float32)
    fuel   = np.full((rows, cols), 3, dtype=np.int8)

    cbh, cbd, cc = _canopy_arrays(fuel, rng)
    return TerrainData(
        elevation=elevation, slope=slope, aspect=aspect, fuel_model=fuel,
        resolution_m=resolution_m, origin=(37.5, -119.5), shape=(rows, cols),
        canopy_base_height=cbh, canopy_bulk_density=cbd, canopy_cover=cc,
    )


def _timber_terrain(rows: int = 200, cols: int = 200,
                    resolution_m: float = 50.0) -> TerrainData:
    """
    Timber-dominant terrain for crown fire risk scenario.
    Gentle ridges, fuel models 8-10 (timber litter/understory), low CBH
    to make crown fire transition uncertain near FMC thresholds.
    """
    rng = np.random.default_rng(7)
    r = np.linspace(0, 1, rows)
    c = np.linspace(0, 1, cols)
    RI, CI = np.meshgrid(r, c, indexing="ij")

    ridge  = 150.0 * np.exp(-((RI - 0.40) ** 2 + (CI - 0.50) ** 2) / 0.12)
    noise  = rng.normal(0, 4, (rows, cols))
    elevation = (200.0 + ridge + noise).astype(np.float32)

    dy, dx = np.gradient(elevation, resolution_m, resolution_m)
    slope  = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2))).astype(np.float32)
    aspect = (np.degrees(np.arctan2(-dx, dy)) % 360).astype(np.float32)

    # Mostly timber litter/understory; some open timber (10) on steeper slopes
    fuel = np.where(slope > 8, 10,
           np.where(elevation > 290, 9, 8)).astype(np.int8)

    # Deliberately lower CBH for a narrow crown fire initiation margin:
    # reduce proxy CBH by 30% so FMC perturbation determines whether crown fire triggers
    cbh_raw, cbd, cc = _canopy_arrays(fuel, rng, jitter=0.05)
    cbh = (cbh_raw * 0.70).astype(np.float32)   # lower CBH → lower I_crit

    return TerrainData(
        elevation=elevation, slope=slope, aspect=aspect, fuel_model=fuel,
        resolution_m=resolution_m, origin=(37.5, -119.5), shape=(rows, cols),
        canopy_base_height=cbh, canopy_bulk_density=cbd, canopy_cover=cc,
    )


# ---------------------------------------------------------------------------
# Public scenario factories
# ---------------------------------------------------------------------------

def hilly_heterogeneous(
    seed: int = 42,
) -> tuple[TerrainData, GroundTruth, SimulationConfig]:
    """
    Primary demo scenario.

    Complex ridge/valley terrain with mixed fuel types and spatially variable
    FMC.  Fire ignites from the SW quadrant (row 150, col 40) and spreads
    upslope into drier fuels under a southerly wind.  A moderate 30° wind
    shift at hour 2 stresses the wind-estimation pipeline.
    """
    terrain = _hilly_terrain()
    ignition_cell = (150, 40)

    events = [
        WindEvent(
            time_s=7200.0,         # hour 2
            direction_change=30.0, # clockwise: S → SSW
            speed_change=1.5,      # gust from 5 → 6.5 m/s
            ramp_duration_s=900.0, # 15-minute transition
        ),
    ]

    ground_truth = generate_ground_truth(
        terrain=terrain,
        ignition_cell=ignition_cell,
        base_fmc=0.08,
        base_ws=5.0,
        base_wd=180.0,
        wind_events=events,
        seed=seed,
    )

    config = SimulationConfig(
        dt=10.0,
        total_time_s=21600.0,          # 6 hours
        ignis_cycle_interval_s=1200.0, # IGNIS every 20 min
        n_drones=5,
        drone_speed_ms=15.0,
        drone_endurance_s=1800.0,
        camera_footprint_m=100.0,
        base_cell=(195, 100),          # S edge, centre column
        frame_interval=6,
        fps=10,
        output_path="out/sim_hilly",
        scenario_name="hilly_heterogeneous",
    )

    return terrain, ground_truth, config


def wind_shift(
    seed: int = 42,
) -> tuple[TerrainData, GroundTruth, SimulationConfig]:
    """
    Stress-test scenario.

    Same hilly terrain as the primary demo, but a 45° wind-shift event begins
    at hour 3 (t=10800 s) with a 10-minute ramp.  This forces the system to
    re-route drones mid-simulation, demonstrating real-time adaptability.
    """
    terrain = _hilly_terrain()
    ignition_cell = (150, 40)

    events = [
        WindEvent(
            time_s=10800.0,        # hour 3
            direction_change=45.0, # clockwise (SW → W)
            speed_change=3.0,      # gust to 8 m/s
            ramp_duration_s=600.0, # 10-minute transition
        ),
    ]

    ground_truth = generate_ground_truth(
        terrain=terrain,
        ignition_cell=ignition_cell,
        base_fmc=0.08,
        base_ws=5.0,
        base_wd=180.0,
        wind_events=events,
        seed=seed,
    )

    config = SimulationConfig(
        dt=10.0,
        total_time_s=21600.0,
        ignis_cycle_interval_s=1200.0,
        n_drones=5,
        drone_speed_ms=15.0,
        drone_endurance_s=1800.0,
        camera_footprint_m=100.0,
        base_cell=(195, 100),
        frame_interval=6,
        fps=10,
        output_path="out/sim_wind_shift",
        scenario_name="wind_shift",
    )

    return terrain, ground_truth, config


def flat_homogeneous(
    seed: int = 42,
) -> tuple[TerrainData, GroundTruth, SimulationConfig]:
    """
    Control scenario.

    Flat terrain, uniform grass fuel (Anderson 3), constant wind.  Spatial
    FMC variability is minimal so information value is nearly uniform.
    Targeted placement should show only marginal improvement over uniform
    sampling — this provides the lower-bound PERR baseline.
    """
    terrain = _flat_terrain()
    ignition_cell = (150, 40)

    ground_truth = generate_ground_truth(
        terrain=terrain,
        ignition_cell=ignition_cell,
        base_fmc=0.10,
        base_ws=4.0,
        base_wd=180.0,
        wind_events=[],
        seed=seed,
    )

    config = SimulationConfig(
        dt=10.0,
        total_time_s=21600.0,
        ignis_cycle_interval_s=1200.0,
        n_drones=5,
        drone_speed_ms=15.0,
        drone_endurance_s=1800.0,
        camera_footprint_m=100.0,
        base_cell=(195, 100),
        frame_interval=6,
        fps=10,
        output_path="out/sim_flat",
        scenario_name="flat_homogeneous",
    )

    return terrain, ground_truth, config


def dual_ignition(
    seed: int = 42,
) -> tuple[TerrainData, GroundTruth, SimulationConfig]:
    """
    Two simultaneous ignitions on hilly terrain with a mid-simulation wind shift.

    Fire 1 starts in the SW quadrant (row 155, col 35) on lower-elevation
    shrub/rough-wood fuels.  Fire 2 starts in the NE area (row 35, col 155)
    on steep-slope grass fuels with faster spread rates.  A 45° wind shift
    at t=30 min forces the drone fleet to re-route mid-mission and tests
    whether the wind GP estimate tracks the change from drone observations.

    Designed as a 1-hour stress test: both fires are active simultaneously,
    the wind shifts partway through, and FMC is dry (0.07) to maximise spread.
    """
    terrain = _hilly_terrain()

    events = [
        WindEvent(
            time_s=1800.0,         # 30 minutes in
            direction_change=45.0, # clockwise: SW → W
            speed_change=2.5,      # gust from 6 → 8.5 m/s
            ramp_duration_s=300.0, # 5-minute transition
        ),
    ]

    ground_truth = generate_ground_truth(
        terrain=terrain,
        ignition_cell=[(155, 35), (35, 155)],
        base_fmc=0.07,    # dry — maximises spread for the short 1-hour window
        base_ws=6.0,
        base_wd=225.0,    # SW wind — pushes both fires into different terrain features
        wind_events=events,
        seed=seed,
    )

    config = SimulationConfig(
        dt=10.0,
        total_time_s=3600.0,           # 1 hour
        ignis_cycle_interval_s=1200.0, # 3 IGNIS cycles in the hour
        n_drones=5,
        drone_speed_ms=15.0,
        drone_endurance_s=1800.0,
        camera_footprint_m=100.0,
        base_cell=(195, 100),
        frame_interval=6,
        fps=10,
        output_path="out/sim_dual",
        scenario_name="dual_ignition",
    )

    return terrain, ground_truth, config


def crown_fire_risk(
    seed: int = 42,
) -> tuple[TerrainData, GroundTruth, SimulationConfig]:
    """
    Crown fire uncertainty scenario.

    Timber-dominant stands with deliberately low canopy base height so that
    crown fire initiation sits at the boundary of what the ensemble predicts.
    FMC is tuned to the 'uncertain' range: dry enough that some ensemble
    members exceed the Van Wagner threshold, wet enough that others don't.
    The resulting bimodal arrival-time distributions have high binary entropy,
    driving the information field to route drones to FMC measurement in the
    stands where crown fire transition is undecided.

    This is the headline scenario for demonstrating crown fire + bimodal
    detection together: a drone measuring FMC = 0.07 in a stand with CBH ≈ 2m
    can definitively resolve whether crown fire initiates — the most
    consequential single measurement in the domain.
    """
    terrain       = _timber_terrain()
    ignition_cell = (160, 50)

    ground_truth = generate_ground_truth(
        terrain=terrain,
        ignition_cell=ignition_cell,
        base_fmc=0.09,    # near-critical: some members crown, some surface
        base_ws=7.0,      # moderate wind to push towards crown threshold
        base_wd=180.0,
        wind_events=[],
        seed=seed,
    )

    config = SimulationConfig(
        dt=10.0,
        total_time_s=21600.0,
        ignis_cycle_interval_s=1200.0,
        n_drones=5,
        drone_speed_ms=15.0,
        drone_endurance_s=1800.0,
        camera_footprint_m=100.0,
        base_cell=(195, 100),
        frame_interval=6,
        fps=10,
        output_path="out/sim_crown",
        scenario_name="crown_fire_risk",
    )

    return terrain, ground_truth, config
