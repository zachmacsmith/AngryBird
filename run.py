"""
run.py — Unified WISPsim entry point.

Supports two terrain modes:

  LANDFIRE mode (default)
      Loads real GeoTIFF data from a cache directory (same format as the old
      run_landfire_wispsim.py).  Fire ignition is set from a lat/lon report
      (--fire-lat / --fire-lon) with configurable uncertainty radius.  If no
      fire lat/lon is given the nearest burnable cell to the grid centre is used.

  Synthetic scenario mode (--scenario)
      Uses a built-in 200×200 procedural terrain from wispsim/scenarios.py.
      Available: hilly_heterogeneous, wind_shift, flat_homogeneous,
      dual_ignition, crown_fire_risk.

Key design choices vs the old scripts
--------------------------------------
  • Single entry point replaces run_landfire_wispsim.py + run_sim.py.
  • Fire report is *uncertain* (configurable radius + confidence), not oracle.
  • Fire observations are seeded into obs_store *before* SimulationRunner so
    the first IGNIS cycle has real data to bootstrap the ensemble.
  • GPUFireEngine is used for mps/cuda; built-in SimpleFire for cpu.
  • Drone specs, planning horizon, and cycle interval are all CLI args.
  • Selector registry is built with the correct resolution and drone params for
    each terrain rather than relying on the global 50 m defaults.

Usage examples
--------------
    # LANDFIRE terrain — default settings (correlationpath selector, mps GPU)
    PYTHONPATH=. python scripts/run.py

    # LANDFIRE terrain — fire report at specific location
    PYTHONPATH=. python scripts/run.py \\
        --fire-lat 39.18 --fire-lon -121.23 \\
        --fire-radius-m 500 --fire-confidence 0.80

    # LANDFIRE terrain — explicit weather priors
    PYTHONPATH=. python scripts/run.py \\
        --wind-speed 7.0 --wind-direction 225 \\
        --temperature 34 --humidity 0.15 --base-fmc 0.07

    # Synthetic scenario
    PYTHONPATH=. python scripts/run.py --scenario hilly_heterogeneous

    # Full custom run
    PYTHONPATH=. python run.py \\
        --terrain landfire_cache --device mps \\
        --drones 2 --targets 6 --members 30 \\
        --hours 2 --cycle-min 10 --horizon-min 240 \\
        --selector correlation_path --out out/myrun
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Logging — clean, coloured, cycle-aware
# ---------------------------------------------------------------------------

class _Fmt(logging.Formatter):
    """
    Terminal-friendly formatter.

    • Timestamp + short level + short module name + message.
    • ANSI colours: grey (DEBUG), white (INFO), yellow (WARNING), red (ERROR).
    • Blank line injected before the *first* line of each IGNIS cycle so
      cycles are visually grouped without excessive whitespace inside them.
    • Module name shortened to just the last component (e.g. 'orchestrator').
    """

    import re as _re

    _RESET  = "\033[0m"
    _DIM    = "\033[2m"
    _YELLOW = "\033[33m"
    _RED    = "\033[31m"

    _LEVEL_STYLES: dict[int, str] = {
        logging.DEBUG:    "\033[2m",    # dim
        logging.INFO:     "",           # default terminal colour
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[1;31m", # bold red
    }

    # A blank line is inserted before lines that open a new cycle boundary.
    # Specifically: "Cycle N | fire state …" and "WISP cycle N | …" and
    # "SimulationRunner starting" — but NOT "Cycle N timing" or "Cycle N | strategy".
    _CYCLE_OPEN = _re.compile(
        r"^(WISP cycle \d+|Cycle \d+ \| fire state|SimulationRunner starting)"
    )

    def format(self, record: logging.LogRecord) -> str:
        short_name = record.name.split(".")[-1]
        ts  = self.formatTime(record, datefmt="%H:%M:%S")
        lvl = record.levelname[:4]
        colour = self._LEVEL_STYLES.get(record.levelno, "")
        msg = record.getMessage()

        prefix = "\n" if self._CYCLE_OPEN.match(msg) else ""

        return (
            f"{prefix}"
            f"{self._DIM}{ts}{self._RESET}  "
            f"{colour}{lvl:<4}{self._RESET}  "
            f"{self._DIM}{short_name:<14}{self._RESET}  "
            f"{colour}{msg}{self._RESET}"
        )


def _setup_logging() -> None:
    """Configure root logger with the clean formatter."""
    # Suppress the 'Mean of empty slice' RuntimeWarning from the fire engine
    # (expected behaviour during early steps before any cells have burned).
    warnings.filterwarnings(
        "ignore",
        message="Mean of empty slice",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="All-NaN slice encountered",
        category=RuntimeWarning,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Fmt())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Suppress chatty third-party loggers
    for noisy in ("matplotlib", "PIL", "rasterio", "numexpr", "numexpr.utils"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_setup_logging()

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from angrybird.config import (
    TAU_FMC_S, TAU_WIND_SPEED_S, TAU_WIND_DIR_S,
    GP_DEFAULT_FMC_VARIANCE,
    GP_DEFAULT_WIND_SPEED_MEAN, GP_DEFAULT_WIND_SPEED_VARIANCE,
    GP_DEFAULT_WIND_DIR_MEAN, GP_DEFAULT_WIND_DIR_VARIANCE,
)
from angrybird.gp import IGNISGPPrior
from angrybird.nelson import nelson_fmc_field
from angrybird.types import GPPrior
from angrybird.observations import (
    FireDetectionObservation,
    ObservationStore,
    ObservationType,
)
from angrybird.orchestrator import IGNISOrchestrator
from angrybird.selectors.base import SelectorRegistry
from angrybird.selectors.baselines import FireFrontSelector, UniformSelector
from angrybird.selectors.correlation_path import CorrelationPathSelector
from angrybird.selectors.greedy import GreedySelector
from angrybird.selectors.heuristics import FireFrontOrbitSelector, LawnmowerSelector
from angrybird.types import TerrainData
from wispsim.ground_truth import WindEvent, generate_ground_truth
from wispsim.runner import SimulationConfig, SimulationRunner
import wispsim.scenarios as _scenarios

log = logging.getLogger("run")

# Non-burnable SB40 fuel codes
_NB_CODES: frozenset[int] = frozenset({91, 92, 93, 98, 99})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def find_burnable_cell(terrain: TerrainData) -> tuple[int, int]:
    """
    Find the nearest burnable (non-NB) cell to the grid centre.

    Scans outward in a square spiral from (R//2, C//2) until a cell whose
    SB40 fuel model is not in the non-burnable set {91,92,93,98,99} is found.
    Falls back to the centre cell if nothing is found (should never happen on
    real LANDFIRE data).
    """
    R, C = terrain.shape
    r0, c0 = R // 2, C // 2
    for dr in range(0, max(R, C)):
        for dc in range(0, dr + 1):
            for r, c in {
                (r0 + dr, c0 + dc), (r0 - dr, c0 + dc),
                (r0 + dr, c0 - dc), (r0 - dr, c0 - dc),
            }:
                if 0 <= r < R and 0 <= c < C:
                    if int(terrain.fuel_model[r, c]) not in _NB_CODES:
                        return (r, c)
    return (r0, c0)


def latlon_to_cell(
    lat: float,
    lon: float,
    terrain: TerrainData,
) -> tuple[int, int]:
    """
    Convert a WGS-84 lat/lon to the nearest grid (row, col).

    Uses flat-earth approximation (valid at wildfire scales < ~100 km):
      row = (origin_lat - lat)  * 111_000 / resolution_m
      col = (lon - origin_lon)  * 111_000 * cos(lat) / resolution_m

    Clamps to grid bounds.  Requires terrain.origin_latlon to be set.
    """
    if terrain.origin_latlon is None:
        raise ValueError("terrain.origin_latlon is not set — cannot convert lat/lon")
    origin_lat, origin_lon = terrain.origin_latlon
    R, C = terrain.shape
    res  = terrain.resolution_m
    row  = int((origin_lat - lat)  * 111_000.0 / res)
    col  = int((lon - origin_lon) * 111_000.0 * math.cos(math.radians(lat)) / res)
    return (max(0, min(R - 1, row)), max(0, min(C - 1, col)))


def cells_within_radius(
    centre: tuple[int, int],
    radius_m: float,
    terrain: TerrainData,
) -> list[tuple[int, int]]:
    """
    Return all grid cells whose centre lies within radius_m of `centre`.

    Used to convert a fire report's uncertainty radius into a set of seed cells
    for the obs_store.  Only burnable cells are included so we don't seed the
    ensemble with non-physical ignitions in water/rock/urban pixels.
    """
    R, C   = terrain.shape
    res    = terrain.resolution_m
    r0, c0 = centre
    half   = int(math.ceil(radius_m / res))
    result: list[tuple[int, int]] = []
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < R and 0 <= c < C):
                continue
            dist_m = math.sqrt((dr * res) ** 2 + (dc * res) ** 2)
            if dist_m <= radius_m:
                if int(terrain.fuel_model[r, c]) not in _NB_CODES:
                    result.append((r, c))
    return result if result else [centre]


def crop_terrain(
    terrain: TerrainData,
    centre: tuple[int, int],
    factor: float = 3.0,
) -> tuple[TerrainData, tuple[int, int]]:
    """
    Return a cropped TerrainData whose shape is 1/factor of the original,
    centred on `centre`.

    Also returns the new (row, col) of `centre` in the cropped grid.
    All terrain arrays are sliced; origin_latlon is adjusted to the new NW corner.
    """
    R, C    = terrain.shape
    new_R   = max(10, int(R / factor))
    new_C   = max(10, int(C / factor))
    r0, c0  = centre

    # Clamp the crop window to grid bounds
    r_start = max(0, r0 - new_R // 2)
    c_start = max(0, c0 - new_C // 2)
    r_end   = min(R, r_start + new_R)
    c_end   = min(C, c_start + new_C)
    # Shift start back if we hit the far edge
    r_start = max(0, r_end - new_R)
    c_start = max(0, c_end - new_C)

    def _sl(arr):
        return arr[r_start:r_end, c_start:c_end]

    # Adjust origin lat/lon to new NW corner
    new_origin = None
    if terrain.origin_latlon is not None:
        orig_lat, orig_lon = terrain.origin_latlon
        res = terrain.resolution_m
        new_lat = orig_lat - r_start * res / 111_000.0
        new_lon = orig_lon + c_start * res / (111_000.0 * math.cos(math.radians(orig_lat)))
        new_origin = (new_lat, new_lon)

    cropped = TerrainData(
        elevation          = _sl(terrain.elevation),
        slope              = _sl(terrain.slope),
        aspect             = _sl(terrain.aspect),
        fuel_model         = _sl(terrain.fuel_model),
        canopy_cover       = _sl(terrain.canopy_cover)       if terrain.canopy_cover       is not None else None,
        canopy_height      = _sl(terrain.canopy_height)      if terrain.canopy_height      is not None else None,
        canopy_base_height = _sl(terrain.canopy_base_height) if terrain.canopy_base_height is not None else None,
        canopy_bulk_density= _sl(terrain.canopy_bulk_density)if terrain.canopy_bulk_density is not None else None,
        shape              = (r_end - r_start, c_end - c_start),
        resolution_m       = terrain.resolution_m,
        origin_latlon      = new_origin,
    )
    new_centre = (r0 - r_start, c0 - c_start)
    return cropped, new_centre


def random_ignition_cells(
    centre: tuple[int, int],
    n: int,
    terrain: TerrainData,
    rng: np.random.Generator,
    min_radius_m: float = 200.0,
) -> list[tuple[int, int]]:
    """
    Sample `n` random burnable cells around `centre` to use as simultaneous
    ignition points.

    The search radius expands in 100 m increments until at least `n` burnable
    candidates exist, so the cluster is always as compact as the terrain allows.
    Returns exactly `n` cells (or all available if the whole grid has fewer).
    """
    res    = terrain.resolution_m
    radius = max(min_radius_m, math.sqrt(n / math.pi) * res * 1.5)

    candidates: list[tuple[int, int]] = []
    while len(candidates) < n:
        candidates = cells_within_radius(centre, radius, terrain)
        if len(candidates) >= n:
            break
        radius += res  # expand by one cell and retry

    indices = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
    return [candidates[i] for i in indices]


def make_gp(terrain: TerrainData) -> tuple[IGNISGPPrior, ObservationStore]:
    """Build a fresh GP + ObservationStore pair matched to terrain resolution."""
    decay_config = {
        ObservationType.FMC:            TAU_FMC_S,
        ObservationType.WIND_SPEED:     TAU_WIND_SPEED_S,
        ObservationType.WIND_DIRECTION: TAU_WIND_DIR_S,
    }
    try:
        obs_store = ObservationStore(decay_config)
    except TypeError:
        obs_store = ObservationStore()

    gp = IGNISGPPrior(obs_store, terrain=terrain, resolution_m=terrain.resolution_m)
    return gp, obs_store


def make_registry(
    terrain: TerrainData,
    drone_speed_ms: float,
    drone_endurance_s: float,
    cycle_s: float,
    correlation_length_m: float = 2000.0,
    min_domain_cells: int = 15,
    horizon_cycles: float = 1.0,
) -> SelectorRegistry:
    """
    Build a SelectorRegistry with correct per-terrain resolution and drone params.

    The global default registry uses 50 m cells and default drone specs.  Any
    run on a different terrain (e.g., 100 m LANDFIRE) must build its own
    registry so that distance/range conversions are correct.
    """
    res           = terrain.resolution_m
    d_cycle_m     = drone_speed_ms * cycle_s
    drone_range_m = drone_speed_ms * drone_endurance_s

    reg = SelectorRegistry()
    reg.register(GreedySelector(resolution_m=res))
    reg.register(UniformSelector())
    reg.register(FireFrontSelector())
    reg.register(CorrelationPathSelector(
        drone_range_m=drone_range_m,
        d_cycle_m=d_cycle_m,
        correlation_length_m=correlation_length_m,
        min_domain_cells=min_domain_cells,
        horizon_cycles=horizon_cycles,
    ))
    reg.register(LawnmowerSelector(drone_speed_ms=drone_speed_ms, cycle_s=cycle_s, resolution_m=res))
    reg.register(FireFrontOrbitSelector(drone_speed_ms=drone_speed_ms, cycle_s=cycle_s, resolution_m=res))
    return reg


def seed_fire_report(
    obs_store: ObservationStore,
    centre_cell: tuple[int, int],
    terrain: TerrainData,
    radius_m: float,
    confidence: float,
    timestamp: float = 0.0,
    rng: np.random.Generator | None = None,
) -> int:
    """
    Seed obs_store with FireDetectionObservation entries derived from a fire
    report (centre ± radius, specified confidence).

    Cells within radius_m of centre_cell are included (burnable only).  If
    radius_m ≤ 0 or only the centre cell is available, a single observation is
    added at the centre.

    Returns the number of observations added.
    """
    if rng is None:
        rng = np.random.default_rng()

    seed_cells = cells_within_radius(centre_cell, radius_m, terrain)
    for i, (r, c) in enumerate(seed_cells):
        obs_store.add(FireDetectionObservation(
            _source_id  = f"fire_report_{i}_{r}_{c}",
            _timestamp  = timestamp,
            location    = (r, c),
            is_fire     = True,
            confidence  = confidence,
        ))
    return len(seed_cells)


# ---------------------------------------------------------------------------
# LANDFIRE terrain loader
# ---------------------------------------------------------------------------

def _load_landfire(cache_dir: str) -> TerrainData:
    """Load a LANDFIRE GeoTIFF cache directory."""
    from angrybird.landfire import load_from_directory
    terrain = load_from_directory(cache_dir, resolution_m=100.0)
    return terrain


# ---------------------------------------------------------------------------
# Fire engine factory
# ---------------------------------------------------------------------------

def _make_fire_engine(terrain: TerrainData, device: str):
    """
    Return the appropriate fire engine for the requested device.

    mps / cuda → GPUFireEngine (recommended; requires PyTorch).
    cpu         → GPUFireEngine on cpu (slow but correct) if available,
                  otherwise falls back to SimpleFire (CA-based, no GPU).
    """
    try:
        from angrybird.fire_engines.gpu_fire_engine import GPUFireEngine
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            engine = GPUFireEngine(terrain, device=device, target_cfl=0.7)
        log.info("  GPUFireEngine ready (device=%s)", device)
        return engine
    except Exception as exc:
        log.warning("GPUFireEngine unavailable (%s) — falling back to SimpleFire", exc)
        from wispsim.simple_fire import SimpleFire
        return SimpleFire(terrain)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:  # noqa: C901
    t0 = time.time()

    # ── Terrain ───────────────────────────────────────────────────────────
    scenario_mode = args.scenario is not None

    if scenario_mode:
        log.info("Synthetic scenario: %s", args.scenario)
        factory = getattr(_scenarios, args.scenario, None)
        if factory is None:
            log.error(
                "Unknown scenario '%s'. Choose from: %s",
                args.scenario,
                ", ".join(_SCENARIO_NAMES),
            )
            sys.exit(1)
        terrain, ground_truth, _sc = factory(seed=args.seed)
        resolution_m = terrain.resolution_m
        log.info("  Shape: %d×%d  res=%.0f m", *terrain.shape, resolution_m)
    else:
        log.info("LANDFIRE terrain from '%s' …", args.terrain)
        terrain = _load_landfire(args.terrain)
        R, C    = terrain.shape
        resolution_m = terrain.resolution_m
        log.info(
            "  Shape: %d×%d  |  %.1f km × %.1f km  |  origin: %.3f°N %.3f°W",
            R, C,
            R * resolution_m / 1000,
            C * resolution_m / 1000,
            terrain.origin_latlon[0] if terrain.origin_latlon else 0.0,
            -terrain.origin_latlon[1] if terrain.origin_latlon else 0.0,
        )

    R, C = terrain.shape

    # ── Ignition / fire report ─────────────────────────────────────────────
    if scenario_mode:
        # ground_truth already has ignition cells baked in
        ignition_cell = ground_truth.ignition_cells[0] \
            if hasattr(ground_truth, "ignition_cells") \
            else find_burnable_cell(terrain)
    else:
        # Resolve ignition from fire report lat/lon or fall back to grid centre
        if args.fire_lat is not None and args.fire_lon is not None:
            ignition_cell = latlon_to_cell(args.fire_lat, args.fire_lon, terrain)
            log.info(
                "  Fire report: %.4f°N %.4f°W → cell (%d, %d)",
                args.fire_lat, args.fire_lon, *ignition_cell,
            )
        else:
            ignition_cell = find_burnable_cell(terrain)
            log.info(
                "  No fire lat/lon given — using burnable centre cell (%d, %d)",
                *ignition_cell,
            )
        log.info(
            "  Fuel=%d  slope=%.1f°  elev=%.0f m",
            terrain.fuel_model[ignition_cell],
            terrain.slope[ignition_cell],
            terrain.elevation[ignition_cell],
        )

        # ── Terrain crop (optional) ───────────────────────────────────────
        if args.crop_factor > 1.0:
            terrain, ignition_cell = crop_terrain(terrain, ignition_cell, args.crop_factor)
            R, C = terrain.shape
            log.info(
                "  Terrain cropped ×%.0f → %d×%d  (%.1f km × %.1f km)  "
                "new ignition cell (%d, %d)",
                args.crop_factor, R, C,
                R * terrain.resolution_m / 1000,
                C * terrain.resolution_m / 1000,
                *ignition_cell,
            )

        # ── Multi-cell ignition (optional) ────────────────────────────────
        rng_seed = np.random.default_rng(args.seed)
        if args.ignition_cells > 1:
            ignition_cells = random_ignition_cells(
                centre    = ignition_cell,
                n         = args.ignition_cells,
                terrain   = terrain,
                rng       = rng_seed,
            )
            log.info(
                "  Multi-cell ignition: %d cells sampled around (%d, %d)",
                len(ignition_cells), *ignition_cell,
            )
        else:
            ignition_cells = [ignition_cell]

        # ── Ground truth (LANDFIRE mode) ──────────────────────────────────
        wind_events = []
        if args.wind_event_time_s is not None:
            wind_events.append(WindEvent(
                time_s           = args.wind_event_time_s,
                direction_change = args.wind_event_dir_change,
                speed_change     = args.wind_event_speed_change,
                ramp_duration_s  = args.wind_event_ramp_s,
            ))
        else:
            wind_events = [
                WindEvent(
                    time_s=1800.0,
                    direction_change=20.0,
                    speed_change=1.0,
                    ramp_duration_s=300.0,
                ),
            ]

        ground_truth = generate_ground_truth(
            terrain           = terrain,
            ignition_cell     = ignition_cells if len(ignition_cells) > 1 else ignition_cells[0],
            base_fmc          = args.base_fmc,
            base_ws           = args.wind_speed,
            base_wd           = args.wind_direction,
            wind_events       = wind_events,
            seed              = args.seed,
            temperature_c     = args.temperature,
            relative_humidity = args.humidity,
        )

    # ── GP + ObservationStore ──────────────────────────────────────────────
    gp, obs_store = make_gp(terrain)

    # ── Seed fire observations BEFORE building the runner ─────────────────
    # With multi-cell ignition each ignition cell is seeded individually.
    total_seeded = 0
    seed_centres = ignition_cells if not scenario_mode else [ignition_cell]
    for sc in seed_centres:
        total_seeded += seed_fire_report(
            obs_store    = obs_store,
            centre_cell  = sc,
            terrain      = terrain,
            radius_m     = args.fire_radius_m,
            confidence   = args.fire_confidence,
            timestamp    = 0.0,
        )
    log.info(
        "  Seeded %d fire detection observation(s) in obs_store "
        "(%d ignition point(s), radius=%.0f m, confidence=%.2f)",
        total_seeded, len(seed_centres), args.fire_radius_m, args.fire_confidence,
    )

    # Refresh shape in case terrain was cropped
    R, C = terrain.shape

    # ── Fire engine ────────────────────────────────────────────────────────
    log.info("Initialising fire engine (device=%s) …", args.device)
    fire_engine = _make_fire_engine(terrain, args.device)

    # ── Drone / cycle parameters ───────────────────────────────────────────
    drone_speed_ms   = args.drone_speed_ms
    drone_endurance_s = args.drone_endurance_s
    cycle_s          = args.cycle_min * 60.0
    horizon_min      = args.horizon_min

    # Path-planning horizon: how many cycle-lengths ahead each drone claims territory.
    # Default = AngryBird fire-planning horizon (horizon_min / cycle_min).
    path_horizon_cycles = (
        args.path_horizon
        if args.path_horizon is not None
        else float(horizon_min) / float(args.cycle_min)
    )
    log.info("Path planning horizon: %.1f cycles (%.0f min)",
             path_horizon_cycles, path_horizon_cycles * args.cycle_min)

    # Correlation length for CorrelationPathSelector scales with cell size.
    # Use 20 cells as the domain size, matching the LANDFIRE default.
    corr_len_m       = max(2000.0, 20 * resolution_m)

    # ── Selector registry ──────────────────────────────────────────────────
    registry = make_registry(
        terrain          = terrain,
        drone_speed_ms   = drone_speed_ms,
        drone_endurance_s = drone_endurance_s,
        cycle_s          = cycle_s,
        correlation_length_m = corr_len_m,
        horizon_cycles   = path_horizon_cycles,
    )

    # ── Drone base ─────────────────────────────────────────────────────────
    # Place the staging area 25% of the grid height south of the primary
    # ignition cell, clamped to [10%, 85%] of the grid so it stays well inside
    # the map even after terrain cropping.
    _ic_row = ignition_cell[0] if not scenario_mode else (R // 2)
    _base_row = int(np.clip(_ic_row + R * 0.25, R * 0.10, R * 0.85))
    base_cell = (_base_row, C // 2)
    log.info("Staging area: cell (%d, %d)  [map is %d×%d]", *base_cell, R, C)

    # ── Orchestrator ───────────────────────────────────────────────────────
    orchestrator = IGNISOrchestrator(
        terrain          = terrain,
        gp               = gp,
        obs_store        = obs_store,
        fire_engine      = fire_engine,
        selector_name    = args.selector,
        selector_registry = registry,
        n_drones         = args.drones,
        n_targets        = args.targets,
        staging_area     = base_cell,
        n_members        = args.members,
        horizon_min      = horizon_min,
        resolution_m     = resolution_m,
    )

    # ── SimulationConfig ───────────────────────────────────────────────────
    config = SimulationConfig(
        dt                   = 10.0,
        total_time_s         = args.hours * 3600.0,
        ignis_cycle_interval_s = cycle_s,
        n_drones             = args.drones,
        drone_speed_ms       = drone_speed_ms,
        drone_endurance_s    = drone_endurance_s,
        camera_footprint_m   = max(150.0, resolution_m * 1.5),
        base_cell            = base_cell,
        frame_interval       = 6,
        fps                  = 10,
        output_path          = args.out,
        scenario_name        = args.scenario or "landfire",
        n_raws               = args.n_raws,
        enable_mesh_network  = args.mesh_network,
        selector_name        = args.selector,
        live_fire_horizon_h  = float(args.horizon_min) / 60.0,
    )

    # ── Initial prior snapshot (no observations) for static baseline ───────
    # Build Nelson FMC field at pre-observation GP state and construct a
    # GPPrior directly (without calling gp.predict() which would trigger a fit
    # on the empty training set).  This is the "what if we had no drones?"
    # baseline: Nelson FMC + default background wind, full prior variance.
    _init_lat = (
        float(terrain.origin_latlon[0])
        if (not scenario_mode and terrain.origin_latlon is not None)
        else 37.5
    )
    _nelson_init = nelson_fmc_field(
        terrain,
        T_C=args.temperature,
        RH=args.humidity,
        hour_of_day=14.0,
        latitude_deg=_init_lat,
    )
    initial_gp_prior = GPPrior(
        fmc_mean=_nelson_init.astype(np.float32),
        fmc_variance=np.full(terrain.shape, GP_DEFAULT_FMC_VARIANCE, dtype=np.float32),
        wind_speed_mean=np.full(terrain.shape, GP_DEFAULT_WIND_SPEED_MEAN, dtype=np.float32),
        wind_speed_variance=np.full(terrain.shape, GP_DEFAULT_WIND_SPEED_VARIANCE, dtype=np.float32),
        wind_dir_mean=np.full(terrain.shape, GP_DEFAULT_WIND_DIR_MEAN, dtype=np.float32),
        wind_dir_variance=np.full(terrain.shape, GP_DEFAULT_WIND_DIR_VARIANCE, dtype=np.float32),
    )
    initial_fire_state = ground_truth.fire.fire_state.copy()

    # ── Runner ─────────────────────────────────────────────────────────────
    log.info(
        "Starting SimulationRunner: %.1f h | dt=10 s | %d drone(s) | "
        "%d members | cycle=%.0f min | horizon=%d min",
        args.hours, args.drones, args.members, args.cycle_min, horizon_min,
    )

    runner = SimulationRunner(
        config       = config,
        terrain      = terrain,
        ground_truth = ground_truth,
        orchestrator = orchestrator,
    )

    reports = runner.run()

    # ── Static prior baseline comparison ──────────────────────────────────
    from wispsim.static_prior_evaluator import StaticPriorEvaluator
    log.info("Running static prior baseline (no observations) …")
    baseline_eval = StaticPriorEvaluator(
        config                = config,
        terrain               = terrain,
        ground_truth          = ground_truth,
        initial_gp_prior      = initial_gp_prior,
        fire_engine           = fire_engine,
        initial_fire_state    = initial_fire_state,
        n_members             = args.members,
        horizon_min           = horizon_min,
        initial_phi           = orchestrator._cycle1_initial_phi,
        oracle_arrival_times  = runner._oracle_arrival_times,
    )
    baseline_rows = baseline_eval.evaluate()

    ignis_rows = runner._cycle_metrics_rows
    paired = list(zip(ignis_rows, baseline_rows))
    if paired:
        log.info(
            "\n  %-5s  %-8s  %-18s  %-18s  %s",
            "Cycle", "Time(m)", "IGNIS CRPS/cell", "Static CRPS/cell", "Improvement %",
        )
        for ignis, baseline in paired:
            ignis_v  = ignis["crps_per_cell_minutes"]
            static_v = baseline["crps_per_cell_minutes"]
            improvement = (
                100.0 * (static_v - ignis_v) / static_v
                if static_v > 1e-9 else 0.0
            )
            log.info(
                "  %-5d  %-8.1f  %-18.4f  %-18.4f  %+.1f%%",
                ignis["cycle"], ignis["time_min"], ignis_v, static_v, improvement,
            )

    # Combined CSV — IGNIS rows + static prior rows side by side.
    import csv as _csv
    combined_rows = (
        [{**r, "variant": "ignis"} for r in ignis_rows]
        + baseline_rows
    )
    if combined_rows:
        out_dir = runner.renderer.out_dir
        combined_csv = out_dir.parent / f"{config.scenario_name}_comparison.csv"
        _all_keys = list(combined_rows[0].keys())
        with open(combined_csv, "w", newline="") as _f:
            _w = _csv.DictWriter(_f, fieldnames=_all_keys, extrasaction="ignore")
            _w.writeheader()
            _w.writerows(combined_rows)
        log.info("Comparison CSV → %s", combined_csv.resolve())

    # ── Comparison chart ───────────────────────────────────────────────────
    if ignis_rows and baseline_rows:
        import matplotlib.pyplot as _plt
        _fig, _ax = _plt.subplots(figsize=(8, 4))
        _ax.set_facecolor("#1a1a2e")
        _fig.patch.set_facecolor("#12121f")

        _t_ignis  = [r["time_min"]              for r in ignis_rows]
        _v_ignis  = [r["crps_per_cell_minutes"]  for r in ignis_rows]
        _t_static = [r["time_min"]              for r in baseline_rows]
        _v_static = [r["crps_per_cell_minutes"]  for r in baseline_rows]

        _ax.plot(_t_ignis,  _v_ignis,  "o-", color="#F44336", linewidth=2.0,
                 markersize=5, label="IGNIS (with drones)")
        _ax.plot(_t_static, _v_static, "s--", color="#9E9E9E", linewidth=1.6,
                 markersize=4, label="Static prior (no observations)")
        _ax.axhline(0.0, color="#555577", linestyle=":", linewidth=0.8)

        _ax.set_xlabel("Simulation time (min)", color="white", fontsize=9)
        _ax.set_ylabel("CRPS / burning cell (min)", color="white", fontsize=9)
        _ax.set_title("Forecast accuracy: IGNIS vs static prior",
                      color="white", fontsize=10, fontweight="bold")
        _ax.tick_params(colors="white", labelsize=8)
        for spine in _ax.spines.values():
            spine.set_edgecolor("#444466")
        _ax.legend(fontsize=8, facecolor="#22223b", labelcolor="white",
                   edgecolor="#444466")
        _ax.set_ylim(bottom=0.0)

        _chart_path = out_dir.parent / f"{config.scenario_name}_comparison.png"
        _fig.tight_layout()
        _fig.savefig(_chart_path, dpi=150, bbox_inches="tight",
                     facecolor=_fig.get_facecolor())
        _plt.close(_fig)
        log.info("Comparison chart → %s", _chart_path.resolve())

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    out_dir = runner.renderer.out_dir
    frames  = runner.renderer._frame_count
    cycles  = len(reports)

    log.info(
        "Done. %d IGNIS cycles | %d frames | %.1f s wall-clock",
        cycles, frames, elapsed,
    )
    log.info("Frames: %s", out_dir.resolve())

    video = out_dir.parent / f"{out_dir.name}.mp4"
    if video.exists():
        log.info("Video:  %s  (%.1f MB)", video.resolve(), video.stat().st_size / 1e6)
    else:
        log.info("MP4 not produced (ffmpeg not available) — PNG frames in %s", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SCENARIO_NAMES = [
    "hilly_heterogeneous",
    "wind_shift",
    "flat_homogeneous",
    "dual_ignition",
    "crown_fire_risk",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WISPsim unified runner — LANDFIRE terrain or synthetic scenario",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Terrain / scenario ──────────────────────────────────────────────────
    tg = p.add_argument_group("terrain")
    tg.add_argument(
        "--terrain", default="landfire_cache", metavar="DIR",
        help="LANDFIRE GeoTIFF cache directory (ignored if --scenario is set)",
    )
    tg.add_argument(
        "--scenario", default=None, metavar="NAME",
        choices=_SCENARIO_NAMES,
        help="Use a synthetic scenario instead of LANDFIRE terrain. "
             f"Choices: {', '.join(_SCENARIO_NAMES)}",
    )

    # ── Fire report ────────────────────────────────────────────────────────
    fg = p.add_argument_group("fire report (LANDFIRE mode)")
    fg.add_argument("--fire-lat",        type=float, default=None,
                    help="Latitude of initial fire detection (WGS-84)")
    fg.add_argument("--fire-lon",        type=float, default=None,
                    help="Longitude of initial fire detection (WGS-84)")
    fg.add_argument("--fire-radius-m",   type=float, default=300.0,
                    help="Uncertainty radius around fire report (metres)")
    fg.add_argument("--fire-confidence", type=float, default=0.80,
                    help="Detection confidence for seeded fire observations [0, 1]")
    fg.add_argument("--ignition-cells",  type=int,   default=1,
                    help="Number of simultaneous ignition cells (randomly sampled "
                         "around the fire report centre). 1 = single-point ignition.")
    fg.add_argument("--crop-factor",     type=float, default=1.0,
                    help="Crop terrain to 1/N of original size centred on the fire "
                         "report. 3 = 1/3 size. 1 = no crop (default).")

    # ── Weather priors (LANDFIRE mode) ─────────────────────────────────────
    wg = p.add_argument_group("weather priors (LANDFIRE mode)")
    wg.add_argument("--base-fmc",        type=float, default=0.08,
                    help="Base dead fuel moisture content (fraction)")
    wg.add_argument("--wind-speed",      type=float, default=5.0,
                    help="Base wind speed (m/s)")
    wg.add_argument("--wind-direction",  type=float, default=225.0,
                    help="Base wind direction (degrees, meteorological convention)")
    wg.add_argument("--temperature",     type=float, default=32.0,
                    help="Ambient temperature (°C)")
    wg.add_argument("--humidity",        type=float, default=0.20,
                    help="Relative humidity (fraction, 0–1)")

    # ── Wind event (optional, single event for now) ────────────────────────
    we = p.add_argument_group("wind event (optional, single event)")
    we.add_argument("--wind-event-time-s",       type=float, default=None,
                    metavar="S",
                    help="Simulation time for wind shift event (seconds)")
    we.add_argument("--wind-event-dir-change",   type=float, default=20.0,
                    metavar="DEG",
                    help="Direction change for wind event (degrees, clockwise)")
    we.add_argument("--wind-event-speed-change", type=float, default=1.0,
                    metavar="MS",
                    help="Speed change for wind event (m/s)")
    we.add_argument("--wind-event-ramp-s",       type=float, default=300.0,
                    metavar="S",
                    help="Ramp duration for wind event (seconds)")

    # ── Simulation parameters ──────────────────────────────────────────────
    sg = p.add_argument_group("simulation")
    sg.add_argument("--hours",      type=float, default=1.0,
                    help="Simulation duration (hours)")
    sg.add_argument("--cycle-min",  type=float, default=10.0,
                    help="IGNIS cycle interval (minutes)")
    sg.add_argument("--horizon-min", type=int, default=240,
                    help="Fire ensemble planning horizon (minutes)")
    sg.add_argument("--seed",       type=int,   default=42,
                    help="Random seed for ground truth generation")

    # ── Drone fleet ────────────────────────────────────────────────────────
    dg = p.add_argument_group("drone fleet")
    dg.add_argument("--drones",          type=int,   default=2,
                    help="Number of drones in the fleet")
    dg.add_argument("--targets",         type=int,   default=6,
                    help="Waypoints selected per IGNIS cycle")
    dg.add_argument("--drone-speed-ms",  type=float, default=23.25,
                    help="Drone cruise speed (m/s).  Default 52 mph = 23.25 m/s")
    dg.add_argument("--drone-endurance-s", type=float, default=48_462.0,
                    help="Drone total flight endurance (s).  Default 700 mi @ 52 mph")
    dg.add_argument("--n-raws",          type=int,   default=2,
                    help="Number of simulated RAWS stations")
    dg.add_argument("--mesh-network",    action="store_true", default=False,
                    help="Enable mesh radio network simulation (default: off)")

    # ── Ensemble / selector ────────────────────────────────────────────────
    eg = p.add_argument_group("ensemble / selector")
    eg.add_argument("--members",  type=int,   default=50,
                    help="Ensemble size for GPUFireEngine (default 50; N<20 makes binary "
                         "entropy noisy — single outlier member gives burn_prob=0.05 → "
                         "entropy=0.29, comparable to genuine 50/50 split at N=200)")
    eg.add_argument("--selector", default="correlation_path",
                    choices=["correlation_path", "greedy", "uniform", "fire_front"],
                    help="Waypoint selection strategy")
    eg.add_argument("--path-horizon", type=float, default=None,
                    help="Path planning horizon in units of cycle length (1 = one cycle, "
                         "4 = four cycles, etc.).  Higher values force drones to claim broader "
                         "territory during deconfliction, spreading the fleet across the map. "
                         "Only the first cycle's worth of waypoints is sent to the autopilot; "
                         "the rest is used cheaply for deconfliction only (no GP calls). "
                         "Default: horizon_min / cycle_min (= the full AngryBird planning horizon).")
    eg.add_argument("--device",   default="mps",
                    choices=["cpu", "mps", "cuda"],
                    help="PyTorch device for GPUFireEngine")

    # ── Output ─────────────────────────────────────────────────────────────
    og = p.add_argument_group("output")
    og.add_argument("--out", default="out/wispsim",
                    help="Output directory for frames / video")

    return p


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()
    main(args)
