from .dynamic_prior import DynamicPrior
from .measurements import (
    Measurement,
    NWPWeatherMeasurement,
    NWPWindMeasurement,
    SatelliteFMCMeasurement,
)
from .sources import (
    EnvironmentalDataSource,
    GroundTruthDataSource,
    SimulatedEnvironmentalSource,
    StaticDataSource,
)

<<<<<<< HEAD

class EnvironmentalDataSource:
    """Base class for environmental data sources (weather API, NWP model, etc.)."""


class StaticDataSource(EnvironmentalDataSource):
    """Environmental data source backed by fixed (non-updating) fields."""


class SimulatedEnvironmentalSource(EnvironmentalDataSource):
    """
    Environmental data source backed by a GroundTruth simulation object.
    Used by SimulationRunner to feed time-varying weather into the orchestrator.
    """

    def __init__(self, ground_truth, rng=None):
        self.ground_truth = ground_truth
        self.rng = rng

    def get_weather_at(self, current_time: float) -> dict:
        """Return a weather_source dict compatible with DynamicPrior.update_cycle."""
        gt = self.ground_truth
        ws, wd = gt.get_wind(current_time) if hasattr(gt, "get_wind") else (None, None)
        d: dict = {"source": "simulated"}
        if ws is not None:
            d["wind_speed"] = ws
        if wd is not None:
            d["wind_direction"] = wd
        return d


__all__ = [
    "DynamicPrior",
    "EnvironmentalDataSource",
    "StaticDataSource",
    "SimulatedEnvironmentalSource",
=======
__all__ = [
    "DynamicPrior",
    "Measurement",
    "NWPWeatherMeasurement",
    "NWPWindMeasurement",
    "SatelliteFMCMeasurement",
    "EnvironmentalDataSource",
    "GroundTruthDataSource",
    "SimulatedEnvironmentalSource",
    "StaticDataSource",
>>>>>>> 601e2e99cfdfdb5a77a8ea8f791078c1c00159e9
]
