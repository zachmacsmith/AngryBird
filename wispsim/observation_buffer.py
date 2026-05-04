"""
Observation buffer with spatial thinning.

Phase 4b — simulation harness only.

Drones accumulate observations continuously at 10-second intervals.
The buffer is flushed at each WISP cycle boundary; before passing
observations to the GP/EnKF we thin spatially so the assimilation
does not receive hundreds of near-identical readings from the same cell.
"""

from __future__ import annotations

from angrybird.config import OBSERVATION_THINNING_SPACING_M
from angrybird.types import DroneObservation


def thin_observations(
    observations: list[DroneObservation],
    min_spacing_m: float,
    resolution_m: float,
) -> list[DroneObservation]:
    """
    Greedy spatial thinning: keep at most one observation per coarse grid cell.

    Cells within min_spacing_m of each other share the same coarse bin.
    The first observation encountered (oldest, since the buffer is FIFO) wins.

    Wind-carrying observations (wind_speed is not NaN) are preferred over
    FMC-only observations if both exist in the same bin.
    """
    if not observations:
        return []

    spacing_cells = max(1, int(min_spacing_m / resolution_m))

    # Two passes: first collect wind observations (they carry both FMC + wind),
    # then fill in FMC-only observations for uncovered bins.
    wind_bin:   dict[tuple[int, int], DroneObservation] = {}
    fmc_bin:    dict[tuple[int, int], DroneObservation] = {}

    for obs in observations:
        r, c = obs.location
        key  = (r // spacing_cells, c // spacing_cells)
        import math
        has_wind = not (isinstance(obs.wind_speed, float) and math.isnan(obs.wind_speed))
        if has_wind:
            wind_bin.setdefault(key, obs)
        else:
            fmc_bin.setdefault(key, obs)

    # Merge: wind observations take priority for each bin
    thinned: list[DroneObservation] = list(wind_bin.values())
    for key, obs in fmc_bin.items():
        if key not in wind_bin:
            thinned.append(obs)

    return thinned


class ObservationBuffer:
    """
    FIFO buffer that accumulates drone observations and flushes with thinning.

    Args:
        min_spacing_m:  minimum spatial separation between retained observations
        resolution_m:   grid cell size in metres
    """

    def __init__(self, min_spacing_m: float = OBSERVATION_THINNING_SPACING_M, resolution_m: float = 50.0) -> None:
        self._buffer:       list[DroneObservation] = []
        self._min_spacing   = min_spacing_m
        self._resolution_m  = resolution_m

    def add(self, observations: list[DroneObservation]) -> None:
        """Append new observations to the buffer."""
        self._buffer.extend(observations)

    def flush_thinned(self) -> list[DroneObservation]:
        """Return spatially thinned observations and clear the buffer."""
        thinned = thin_observations(self._buffer, self._min_spacing, self._resolution_m)
        self._buffer.clear()
        return thinned

    def __len__(self) -> int:
        return len(self._buffer)
