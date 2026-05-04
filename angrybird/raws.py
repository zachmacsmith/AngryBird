"""
RAWS (Remote Automated Weather Station) types and operational observer.

Designed for easy production wire-in:

    Simulation  → provider = SimulatedObserver(ground_truth,
                                               fmc_sigma=RAWS_FMC_SIGMA, ...)
    Production  → provider = <your real-time telemetry client>
    Both        → obs = RAWSObserver(stations, provider).observe_all()
                  gp.add_raws(...)   # identical call either way

RAWSDataProvider satisfies the same Protocol as ObservationSource so any
existing observer (SimulatedObserver, real API client) can be dropped in
without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .types import DroneObservation


# ---------------------------------------------------------------------------
# Station metadata
# ---------------------------------------------------------------------------

@dataclass
class RAWSStation:
    """
    One RAWS station.

    In production, station_id maps to the telemetry API key and name is
    displayed in the operator UI.  In simulation, both are auto-generated.
    """
    location:   tuple[int, int]   # (row, col) grid cell
    name:       str = ""
    station_id: str = ""


# ---------------------------------------------------------------------------
# Provider protocol — swap real vs simulated behind this interface
# ---------------------------------------------------------------------------

@runtime_checkable
class RAWSDataProvider(Protocol):
    """
    Anything that can return a DroneObservation per requested grid cell.

    Matches ObservationSource so SimulatedObserver satisfies it without
    subclassing.  Real implementations fetch from RAWS APIs, ASOS feeds,
    or in-situ sensor arrays.
    """
    def observe(self, locations: list[tuple[int, int]]) -> list[DroneObservation]: ...


# ---------------------------------------------------------------------------
# Observer — queries all registered stations via the provider
# ---------------------------------------------------------------------------

class RAWSObserver:
    """
    Queries all registered RAWS stations through a pluggable provider.

    Usage::

        observer = RAWSObserver(stations, provider)
        obs = observer.observe_all()   # list[DroneObservation], one per station
        gp.add_raws(
            [o.location for o in obs],
            [o.fmc for o in obs],
            [o.wind_speed for o in obs],
            [o.wind_dir for o in obs],
        )
    """

    def __init__(
        self,
        stations: list[RAWSStation],
        provider: RAWSDataProvider,
    ) -> None:
        self.stations = stations
        self.provider = provider

    def observe_all(self) -> list[DroneObservation]:
        """Return one observation per station from the current provider state."""
        return self.provider.observe([s.location for s in self.stations])

    @property
    def locations(self) -> list[tuple[int, int]]:
        return [s.location for s in self.stations]


# ---------------------------------------------------------------------------
# Placement helper
# ---------------------------------------------------------------------------

def place_raws_stations(
    shape:              tuple[int, int],
    n:                  int,
    ignition_cells:     list[tuple[int, int]],
    base_cell:          tuple[int, int],
    exclusion_radius:   int = 15,   # cells (~750 m at 50 m/cell)
    base_exclusion:     int = 5,    # cells around drone staging area
    rng:                Optional[np.random.Generator] = None,
) -> list[RAWSStation]:
    """
    Randomly place n RAWS stations, avoiding fire start zones and the drone base.

    For n > 1 stations, each additional station is placed to maximise its
    minimum distance from already-placed stations (greedy max-min spread).

    Args:
        shape:            (rows, cols) grid dimensions
        n:                number of stations to place
        ignition_cells:   fire start locations to exclude around
        base_cell:        drone staging area to exclude around
        exclusion_radius: cells to exclude around each ignition cell
        base_exclusion:   cells to exclude around base_cell
        rng:              reproducible placement seed

    Returns:
        list of RAWSStation with auto-generated names / IDs
    """
    if rng is None:
        rng = np.random.default_rng()

    rows, cols = shape
    excluded = np.zeros((rows, cols), dtype=bool)

    # Circular exclusion around each ignition cell
    rr, cc = np.mgrid[0:rows, 0:cols]
    for ir, ic in ignition_cells:
        excluded |= (rr - ir) ** 2 + (cc - ic) ** 2 <= exclusion_radius ** 2

    # Square exclusion around drone base
    br, bc = base_cell
    r0, r1 = max(0, br - base_exclusion), min(rows, br + base_exclusion + 1)
    c0, c1 = max(0, bc - base_exclusion), min(cols, bc + base_exclusion + 1)
    excluded[r0:r1, c0:c1] = True

    valid_r, valid_c = np.where(~excluded)
    if len(valid_r) == 0:
        # Fallback: domain centre
        return [RAWSStation((rows // 2, cols // 2), f"RAWS_1", "SIM_001")]

    stations: list[RAWSStation] = []
    placed: list[tuple[int, int]] = []

    for i in range(n):
        candidates = np.column_stack([valid_r, valid_c])  # (m, 2)

        if placed:
            placed_arr = np.array(placed, dtype=np.float64)        # (k, 2)
            diff = candidates[:, None, :] - placed_arr[None, :, :]  # (m, k, 2)
            min_dist_sq = np.min((diff ** 2).sum(axis=2), axis=1)   # (m,)
            idx = int(np.argmax(min_dist_sq))
        else:
            idx = int(rng.integers(0, len(valid_r)))

        r, c = int(valid_r[idx]), int(valid_c[idx])
        placed.append((r, c))
        stations.append(RAWSStation(
            location=(r, c),
            name=f"RAWS_{i + 1}",
            station_id=f"SIM_{i + 1:03d}",
        ))

    return stations
