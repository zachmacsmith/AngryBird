"""
Ground truth fire oracle: a cellular automaton that advances the "real" fire.

Phase 4b — simulation harness only.

Uses a Dijkstra-style wavefront propagation so the fire front is processed
exactly once per cell regardless of timestep size.  Rate of spread uses the
same fuel-factor table as SimpleFire so the ensemble and the ground truth
share the same physical proxy model.

Crown fire is modelled via Van Wagner (1977) initiation + Rothermel (1991)
spread, using canopy layers from TerrainData when available or proxy tables
derived from the fuel model.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

from angrybird.config import CANOPY_CBH_M, CANOPY_CBD_KGM3, FUEL_PARAMS
from angrybird.types import TerrainData

_SQRT2 = math.sqrt(2.0)

# 8-neighbor offsets: (row_delta, col_delta, fire_spread_bearing_deg, dist_factor)
# Bearing = direction the fire is spreading TOWARD (geographic, 0=N, 90=E).
_NEIGHBORS: list[tuple[int, int, float, float]] = [
    (-1,  0,   0.0, 1.0),
    (-1,  1,  45.0, _SQRT2),
    ( 0,  1,  90.0, 1.0),
    ( 1,  1, 135.0, _SQRT2),
    ( 1,  0, 180.0, 1.0),
    ( 1, -1, 225.0, _SQRT2),
    ( 0, -1, 270.0, 1.0),
    (-1, -1, 315.0, _SQRT2),
]

# Spread multiplier per Anderson-13 fuel model (matches SimpleFire)
_FUEL_SPREAD: dict[int, float] = {
    1: 0.8, 2: 1.0, 3: 1.6, 4: 1.3, 5: 0.9,
    6: 1.0, 7: 1.1, 8: 0.55, 9: 0.65, 10: 1.0,
    11: 0.5, 12: 0.9, 13: 1.0,
}

# Fuel load (kg/m²) per model for Byram intensity calculation
_FUEL_LOAD: dict[int, float] = {p: FUEL_PARAMS[p]["load"] for p in FUEL_PARAMS}

# Fire type codes
SURFACE_FIRE: int = 1
CROWN_FIRE: int   = 2


def _proxy_array(fuel_model: np.ndarray, proxy: dict[int, float]) -> np.ndarray:
    """Build a float32 array by mapping each cell's fuel model ID through a proxy table."""
    out = np.zeros(fuel_model.shape, dtype=np.float32)
    for fid, val in proxy.items():
        out[fuel_model == fid] = float(val)
    return out


class GroundTruthFire:
    """
    Incrementally advancing ground truth fire via CA wavefront propagation.

    Fire spread is governed by:
      - Rothermel-proxy ROS (m/min) using the same fuel factors as SimpleFire
      - Wind component in the spread direction
      - FMC extinction: no spread when FMC ≥ fuel moisture of extinction (mx)
      - Slope enhancement toward the upslope direction
      - Crown fire initiation (Van Wagner 1977) + crown ROS (Rothermel 1991)

    Propagation uses a min-heap so each cell is processed exactly once at its
    earliest ignition time, regardless of simulation dt.

    Args:
        terrain:        static TerrainData (canopy fields used when present)
        ignition_cell:  (row, col) where the fire starts
    """

    def __init__(
        self,
        terrain: TerrainData,
        ignition_cell: "tuple[int, int] | list[tuple[int, int]]",
    ) -> None:
        rows, cols = terrain.shape
        self.terrain      = terrain
        self.current_time = 0.0

        self.arrival_times = np.full((rows, cols), np.inf, dtype=np.float64)

        # Fire type per cell: SURFACE_FIRE or CROWN_FIRE
        self.fire_types = np.zeros((rows, cols), dtype=np.int8)

        # processed[r,c] = True once a cell's neighbors have been updated
        self._processed = np.zeros((rows, cols), dtype=bool)

        # Support a single cell or a list of cells for multi-ignition scenarios
        cells: list[tuple[int, int]] = (
            [ignition_cell] if isinstance(ignition_cell, tuple) else list(ignition_cell)
        )
        for cell in cells:
            self.arrival_times[cell] = 0.0
        self._heap: list[tuple[float, int, int]] = [(0.0, r, c) for r, c in cells]

        # Canopy arrays — use terrain attributes when present, else proxy tables
        if terrain.canopy_base_height is not None:
            self._cbh = terrain.canopy_base_height.astype(np.float64)
        else:
            self._cbh = _proxy_array(terrain.fuel_model, CANOPY_CBH_M).astype(np.float64)

        if terrain.canopy_bulk_density is not None:
            self._cbd = terrain.canopy_bulk_density.astype(np.float64)
        else:
            self._cbd = _proxy_array(terrain.fuel_model, CANOPY_CBD_KGM3).astype(np.float64)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        dt: float,
        wind_speed: np.ndarray,
        wind_direction: np.ndarray,
        fmc: np.ndarray,
    ) -> None:
        """
        Advance the fire by dt seconds using current ground truth conditions.

        Processes all wavefront cells whose arrival time ≤ current_time + dt,
        then increments current_time by dt.
        """
        self.current_time += dt
        rows, cols = self.terrain.shape

        while self._heap:
            t_arrive, r, c = self._heap[0]
            if t_arrive > self.current_time:
                break
            heapq.heappop(self._heap)

            if self._processed[r, c]:
                continue
            self._processed[r, c] = True

            # Spread to 8 neighbors
            ws   = float(wind_speed[r, c])
            wd   = float(wind_direction[r, c])
            fid  = int(self.terrain.fuel_model[r, c])
            fmc_ = float(fmc[r, c])
            slp  = float(self.terrain.slope[r, c])
            asp  = float(self.terrain.aspect[r, c])
            cbh  = float(self._cbh[r, c])
            cbd  = float(self._cbd[r, c])

            for dr, dc, spread_deg, dist_factor in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if self._processed[nr, nc]:
                    continue

                ros, ft = self._compute_ros(fid, fmc_, ws, wd, spread_deg, slp, asp, cbh, cbd)
                if ros <= 0.0:
                    continue

                # ROS in m/min → m/s; distance in metres
                ros_ms     = ros / 60.0
                distance_m = self.terrain.resolution_m * dist_factor
                t_new      = t_arrive + distance_m / ros_ms

                if t_new < self.arrival_times[nr, nc]:
                    self.arrival_times[nr, nc] = t_new
                    self.fire_types[nr, nc]     = ft
                    heapq.heappush(self._heap, (t_new, nr, nc))


    @property
    def burned_mask(self) -> np.ndarray:
        """bool[rows, cols] — True for cells that have ignited by current_time."""
        return self.arrival_times <= self.current_time

    @property
    def fire_state(self) -> np.ndarray:
        """float32[rows, cols] burn mask for passing to the ensemble fire engine."""
        return self.burned_mask.astype(np.float32)

    # ------------------------------------------------------------------
    # ROS model
    # ------------------------------------------------------------------

    def _compute_ros(
        self,
        fuel_id: int,
        fmc: float,
        wind_ms: float,
        wind_dir_deg: float,
        spread_deg: float,
        slope_deg: float,
        aspect_deg: float,
        cbh_m: float,
        cbd_kgm3: float,
    ) -> tuple[float, int]:
        """
        Simplified Rothermel surface ROS + Van Wagner crown fire check.

        Returns (ros_m_min, fire_type) where fire_type is SURFACE_FIRE or CROWN_FIRE.
        ROS is in m/min.  Consistent with SimpleFire surface model.
        """
        params = FUEL_PARAMS.get(fuel_id, FUEL_PARAMS[1])

        # Above extinction moisture → no spread
        if fmc >= params["mx"]:
            return 0.0, SURFACE_FIRE

        # Wind component in spread direction
        wind_to_deg     = (wind_dir_deg + 180.0) % 360.0
        angle_diff_rad  = math.radians(spread_deg - wind_to_deg)
        wind_component  = max(0.0, wind_ms * math.cos(angle_diff_rad))

        # FMC suppression
        fmc_factor = math.exp(-6.0 * (fmc - 0.06))
        fmc_factor = max(0.05, min(8.0, fmc_factor))

        # Fuel spread multiplier
        fuel_factor = _FUEL_SPREAD.get(fuel_id, 1.0)

        # Slope enhancement in spread direction
        upslope_deg   = (aspect_deg + 180.0) % 360.0
        slope_diff    = math.radians(spread_deg - upslope_deg)
        slope_project = max(0.0, math.sin(math.radians(slope_deg)) * math.cos(slope_diff))
        slope_factor  = 1.0 + 0.5 * slope_project

        effective_wind = max(wind_component, 0.5)
        ros_surface    = 3.0 * effective_wind * fmc_factor * fuel_factor * slope_factor

        # Byram fireline intensity (kW/m): I = H × w0 × R / 60
        w0        = _FUEL_LOAD.get(fuel_id, 0.10)
        intensity = 18000.0 * w0 * ros_surface / 60.0   # ROS in m/min → divide by 60

        # Van Wagner (1977) crown fire initiation (foliar FMC = 100%)
        if cbh_m > 0.0 and cbd_kgm3 > 0.0:
            I_crit = (0.01 * cbh_m * (460.0 + 2590.0)) ** 1.5
            if intensity > I_crit:
                # Rothermel (1991) crown ROS (m/min)
                wind_kmh  = wind_ms * 3.6
                ros_crown = 11.02 * (wind_kmh ** 0.854) * (cbd_kgm3 ** 0.19)
                ros_final = max(ros_surface, ros_crown)
                return ros_final, CROWN_FIRE

        return ros_surface, SURFACE_FIRE
