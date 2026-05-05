"""
wispsim/scenario.py — Scenario dataclass and companion types.

A Scenario is the complete, self-contained description of a wildfire simulation
situation.  It bundles:

  terrain        — physical environment (elevations, fuels, slopes)
  ground_truth   — oracle state that simulated sensors observe
  fire_report    — uncertain initial fire detection that seeds the ensemble
  weather        — background weather conditions for GP prior initialization
  data_source    — observation data streams available to the system
                   (NWP models, GOES, VIIRS, satellite FMC)
  raws_locations — optional pre-placed RAWS station grid cells

Separation from SimulationConfig
---------------------------------
SimulationConfig describes *how* we run (drone count, device, output path, frame
rate, etc.).  Scenario describes *what* we are simulating.  The two are
intentionally separate so the same scenario can be replayed with different
fleet sizes or selectors without modifying the scenario object.

Observation data streams
-------------------------
``data_source`` satisfies the EnvironmentalDataSource protocol.  If None, the
runner builds a SimulatedEnvironmentalSource driven by ground_truth — the right
choice for synthetic simulations.  For real-incident replay, pass a source
backed by real NWP/satellite APIs instead.

FireReport
----------
Represents a single uncertain fire detection (e.g. from GOES, 911 call, or
initial VIIRS swath).  The runner converts this into a cluster of
FireDetectionObservation entries in the obs_store *before* the first IGNIS
cycle so the ensemble can bootstrap without oracle knowledge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from angrybird.prior.sources import EnvironmentalDataSource
    from angrybird.types import TerrainData
    from .ground_truth import GroundTruth


# ---------------------------------------------------------------------------
# FireReport
# ---------------------------------------------------------------------------

@dataclass
class FireReport:
    """
    Uncertain initial fire detection report.

    Represents a geolocation report with positional uncertainty — equivalent
    to a GOES fire pixel, VIIRS active fire detection, or a radio dispatch
    with a lat/lon and error circle.

    The runner adds one FireDetectionObservation per burnable cell within
    `radius_m` of `cell` at `timestamp`, all carrying `confidence`.  This
    gives the ensemble a spatially smeared prior rather than a point oracle.

    Attributes
    ----------
    cell        : (row, col) in the simulation grid. Computed from lat/lon by
                  the scenario builder if lat/lon are provided; otherwise set
                  directly.
    radius_m    : Positional uncertainty radius (metres). Smaller = tighter
                  initial ensemble.  0 → single-cell seed.
    confidence  : Instrument confidence [0, 1].  A GOES detection at high
                  confidence is ~0.90; a 911 call might be 0.65.
    timestamp   : Simulation time of the report (seconds). Usually 0.0.
    lat, lon    : Optional raw lat/lon — stored for metadata / logging only
                  after cell has been computed.
    """
    cell:       tuple[int, int]
    radius_m:   float = 300.0
    confidence: float = 0.80
    timestamp:  float = 0.0
    lat:        Optional[float] = None
    lon:        Optional[float] = None


# ---------------------------------------------------------------------------
# WeatherPrior
# ---------------------------------------------------------------------------

@dataclass
class WeatherPrior:
    """
    Background weather conditions used to initialize the GP prior.

    These values represent the system's *prior belief* about conditions
    before any drone observations arrive.  They feed the Nelson FMC model
    and the uninformed wind GP mean — not the ground truth fire spread.

    For LANDFIRE runs the values typically come from a synoptic weather
    analysis or NWP model output.  For synthetic scenarios they are
    specified in the scenario factory.

    Attributes
    ----------
    base_fmc          : Prior dead fuel moisture content (fraction, 0.02–0.40).
    wind_speed_ms     : Prior wind speed (m/s).
    wind_direction_deg: Prior wind direction (degrees, met convention: from which
                        the wind blows, 0=N, 90=E, 180=S, 225=SW).
    temperature_c     : Ambient temperature (°C) for Nelson FMC model.
    relative_humidity : Relative humidity [0, 1] for Nelson FMC model.
    """
    base_fmc:           float = 0.08
    wind_speed_ms:      float = 5.0
    wind_direction_deg: float = 225.0
    temperature_c:      float = 32.0
    relative_humidity:  float = 0.20


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """
    Complete, self-contained wildfire simulation scenario.

    A Scenario fully specifies the physical situation being simulated.  The
    SimulationRunner consumes a Scenario plus a SimulationConfig (operational
    parameters) to produce a simulation run.

    Attributes
    ----------
    name          : Human-readable name used for output labeling.
    terrain       : Static terrain data (elevation, slope, fuels, …).
    ground_truth  : Oracle state — fire spread, FMC field, wind evolution.
                    Mutable: updated in-place each simulation timestep.
    fire_report   : Uncertain initial fire detection that seeds the ensemble.
    weather       : Background weather prior for GP initialization.
    data_source   : Environmental observation data streams (NWP, GOES, VIIRS,
                    satellite FMC).  If None the runner auto-builds a
                    SimulatedEnvironmentalSource from ground_truth.
    raws_locations: Optional list of pre-placed RAWS station grid cells.
                    Empty list = runner auto-places n_raws stations.
    wind_events   : Scheduled wind shift events (from ground_truth generation;
                    stored here for metadata / inspection).
    """
    name:           str
    terrain:        "TerrainData"
    ground_truth:   "GroundTruth"
    fire_report:    FireReport
    weather:        WeatherPrior
    data_source:    Optional["EnvironmentalDataSource"] = None
    raws_locations: list[tuple[int, int]] = field(default_factory=list)
    wind_events:    list = field(default_factory=list)  # list[WindEvent]

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape (rows, cols) of the terrain."""
        return self.terrain.shape

    @property
    def resolution_m(self) -> float:
        """Grid cell size in metres."""
        return self.terrain.resolution_m

    @property
    def ignition_cell(self) -> tuple[int, int]:
        """Primary ignition cell from the ground truth."""
        cells = getattr(self.ground_truth, "ignition_cells", None)
        if cells:
            return (int(cells[0][0]), int(cells[0][1]))
        return self.fire_report.cell

    def __repr__(self) -> str:
        R, C = self.shape
        return (
            f"Scenario(name={self.name!r}, shape={R}×{C}, "
            f"res={self.resolution_m:.0f}m, "
            f"fire_report={self.fire_report.cell}, "
            f"weather=ws{self.weather.wind_speed_ms:.1f}m/s "
            f"wd{self.weather.wind_direction_deg:.0f}°)"
        )
