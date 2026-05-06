"""
Simple heuristic drone path selectors for baseline comparison.

LawnmowerSelector      — boustrophedon strip sweep; divides the grid into
                         n_drones vertical column strips.  Guaranteed systematic
                         coverage regardless of fire state.

FireFrontOrbitSelector — each drone orbits the predicted fire-front centroid at
                         a fixed standoff radius.  Drones are evenly spaced around
                         the orbit and advance continuously within the cycle budget.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from ..config import (
    CAMERA_FOOTPRINT_CELLS,
    DRONE_CYCLE_DURATION_S,
    DRONE_SPEED_MS,
    GRID_RESOLUTION_M,
)
from ..path_planner import cells_along_path
from ..types import DronePlan, EnsembleResult, InformationField, SelectionResult

if TYPE_CHECKING:
    from ..gp import IGNISGPPrior


class LawnmowerSelector:
    """
    Systematic boustrophedon sweep.

    Divides the grid into n_drones vertical column strips.  Each drone sweeps
    its strip with horizontal rows spaced by the camera swath width.  The path
    length is capped at the cycle budget (drone_speed_ms × cycle_s metres).
    """

    name = "lawnmower"
    kind = "paths"

    def __init__(
        self,
        drone_speed_ms: float = DRONE_SPEED_MS,
        cycle_s: float = DRONE_CYCLE_DURATION_S,
        resolution_m: float = GRID_RESOLUTION_M,
    ) -> None:
        self.drone_speed_ms = drone_speed_ms
        self.cycle_s = cycle_s
        self.resolution_m = resolution_m

    def select(
        self,
        info_field: InformationField,
        gp: "IGNISGPPrior",
        ensemble: EnsembleResult,
        k: int,
        **context,
    ) -> SelectionResult:
        t0 = time.perf_counter()

        resolution_m = float(context.get("resolution_m", self.resolution_m))
        n_drones     = int(context.get("n_drones", max(1, k // 3)))
        staging      = context.get("staging_area", (0, 0))

        shape = info_field.w.shape
        R, C  = shape

        budget_cells = (self.drone_speed_ms * self.cycle_s) / resolution_m
        row_spacing  = max(1, 2 * CAMERA_FOOTPRINT_CELLS + 1)
        strip_width  = max(1, C // n_drones)

        drone_plans: list[DronePlan] = []

        for drone_id in range(n_drones):
            c_start = drone_id * strip_width
            c_end   = (c_start + strip_width - 1) if drone_id < n_drones - 1 else C - 1

            waypoints: list[tuple[int, int]] = [staging]
            cells_traveled = 0.0
            row = 0
            left_to_right = True

            while row < R and cells_traveled < budget_cells:
                if left_to_right:
                    waypoints.append((row, c_start))
                    waypoints.append((row, c_end))
                else:
                    waypoints.append((row, c_end))
                    waypoints.append((row, c_start))
                cells_traveled += (c_end - c_start) + row_spacing
                row += row_spacing
                left_to_right = not left_to_right

            waypoints.append(staging)
            observed = cells_along_path(waypoints, shape)
            drone_plans.append(DronePlan(
                drone_id=drone_id,
                waypoints=waypoints,
                cells_observed=observed,
            ))

        total_info = sum(
            float(info_field.w[r, c]) for p in drone_plans for r, c in p.cells_observed
        )
        return SelectionResult(
            kind="paths",
            drone_plans=drone_plans,
            marginal_gains=[
                float(sum(info_field.w[r, c] for r, c in p.cells_observed))
                for p in drone_plans
            ],
            total_info=total_info,
            strategy_name=self.name,
            compute_time_s=time.perf_counter() - t0,
        )


class FireFrontOrbitSelector:
    """
    Each drone orbits the predicted fire-front centroid at a fixed standoff radius.

    Drones are evenly spaced around the orbit (2π / n_drones apart in starting angle)
    and advance continuously each cycle.  The orbit radius is clamped so the path
    stays within the grid.  Falls back to the grid centre when no fire is detected.
    """

    name = "fire_front_orbit"
    kind = "paths"

    def __init__(
        self,
        standoff_radius_m: float = 1500.0,
        n_arc_waypoints: int = 8,
        lo_prob: float = 0.05,
        drone_speed_ms: float = DRONE_SPEED_MS,
        cycle_s: float = DRONE_CYCLE_DURATION_S,
        resolution_m: float = GRID_RESOLUTION_M,
    ) -> None:
        self.standoff_radius_m = standoff_radius_m
        self.n_arc_waypoints   = n_arc_waypoints
        self.lo_prob           = lo_prob
        self.drone_speed_ms    = drone_speed_ms
        self.cycle_s           = cycle_s
        self.resolution_m      = resolution_m

        # Persistent orbit angle per drone across cycles (drone_id → radians)
        self._drone_angles: dict[int, float] = {}

    def select(
        self,
        info_field: InformationField,
        gp: "IGNISGPPrior",
        ensemble: EnsembleResult,
        k: int,
        **context,
    ) -> SelectionResult:
        t0 = time.perf_counter()

        resolution_m = float(context.get("resolution_m", self.resolution_m))
        n_drones     = int(context.get("n_drones", max(1, k // 3)))
        staging      = context.get("staging_area", (0, 0))

        shape = info_field.w.shape
        R, C  = shape

        # Fire centroid
        bp         = ensemble.burn_probability
        fire_mask  = bp >= self.lo_prob
        if fire_mask.any():
            rows_idx, cols_idx = np.where(fire_mask)
            cy = float(rows_idx.mean())
            cx = float(cols_idx.mean())
        else:
            cy, cx = float(R) / 2.0, float(C) / 2.0

        # Orbit radius in cells — clamped so the orbit stays within the grid
        max_radius_cells = min(cy, R - 1.0 - cy, cx, C - 1.0 - cx) * 0.85
        radius_cells = min(self.standoff_radius_m / resolution_m, max(1.0, max_radius_cells))

        budget_m  = self.drone_speed_ms * self.cycle_s
        radius_m  = radius_cells * resolution_m
        arc_angle = min(budget_m / max(radius_m, 1.0), 2.0 * np.pi)  # cap at one full orbit

        drone_plans: list[DronePlan] = []

        for drone_id in range(n_drones):
            base_angle    = drone_id * (2.0 * np.pi / n_drones)
            current_angle = self._drone_angles.get(drone_id, base_angle)

            angles = np.linspace(current_angle, current_angle + arc_angle,
                                 self.n_arc_waypoints + 1)

            waypoints: list[tuple[int, int]] = [staging]
            prev: tuple[int, int] | None = None
            for theta in angles:
                r = int(round(cy + radius_cells * np.sin(theta)))
                c = int(round(cx + radius_cells * np.cos(theta)))
                r = max(0, min(R - 1, r))
                c = max(0, min(C - 1, c))
                pt = (r, c)
                if pt != prev:
                    waypoints.append(pt)
                    prev = pt
            waypoints.append(staging)

            # Advance angle for next cycle
            self._drone_angles[drone_id] = current_angle + arc_angle

            observed = cells_along_path(waypoints, shape)
            drone_plans.append(DronePlan(
                drone_id=drone_id,
                waypoints=waypoints,
                cells_observed=observed,
            ))

        total_info = sum(
            float(info_field.w[r, c]) for p in drone_plans for r, c in p.cells_observed
        )
        return SelectionResult(
            kind="paths",
            drone_plans=drone_plans,
            marginal_gains=[
                float(sum(info_field.w[r, c] for r, c in p.cells_observed))
                for p in drone_plans
            ],
            total_info=total_info,
            strategy_name=self.name,
            compute_time_s=time.perf_counter() - t0,
        )
