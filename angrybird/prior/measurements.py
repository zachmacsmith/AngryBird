"""
Typed measurement inputs for the dynamic prior.

Each class represents one kind of environmental data that can be passed to
the DynamicPrior.  All fields are immutable (frozen dataclasses).

Adding a new data type — e.g. a RAWS T/RH point observation — means adding
a new subclass here; nothing else in the core needs changing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

# A measurement value can be a scalar (broadcast to grid) or a 2-D field.
ScalarOrField = Union[float, np.ndarray]


@dataclass(frozen=True)
class Measurement:
    """Base class for all environmental measurements."""
    source: str       # label: "HRRR", "GFS", "ground_truth", "static", …
    timestamp: float  # simulation time (seconds) when this was valid


@dataclass(frozen=True)
class NWPWeatherMeasurement(Measurement):
    """
    Temperature + relative humidity from an NWP forecast or scenario.

    Drives the Nelson/Fosberg-Deeming dead fuel moisture equilibrium model.
    In simulation: sourced from GroundTruthDataSource.
    In production: sourced from HRRR or GFS output.
    """
    temperature_c: ScalarOrField         # air temperature °C
    relative_humidity: ScalarOrField     # fraction [0, 1]


@dataclass(frozen=True)
class NWPWindMeasurement(Measurement):
    """
    Wind speed + direction from an NWP forecast or scenario.

    Sets the GP wind prior mean.  Drone and RAWS observations correct this
    locally; cells far from any observation revert to this field.
    In simulation: sourced from GroundTruthDataSource (live evolving field).
    In production: sourced from HRRR or GFS output.
    """
    wind_speed: ScalarOrField      # m/s
    wind_direction: ScalarOrField  # degrees from north


@dataclass(frozen=True)
class SatelliteFMCMeasurement(Measurement):
    """
    Dead fuel moisture content from a satellite retrieval.

    Confidence-weighted blend into the Nelson FMC prior.  Low-confidence
    pixels (cloud cover, shadow, sensor saturation) are masked out.

    Sources: MODIS, VIIRS, Landsat, GOES-R ABI reflectance retrievals.
    Currently not populated in simulation (placeholder for future work).
    """
    fmc: np.ndarray          # fraction [0, 1], float32[rows, cols]
    confidence: np.ndarray   # retrieval confidence [0, 1], float32[rows, cols]
