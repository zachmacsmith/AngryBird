"""
Path planner: drone-to-target assignment, observed-cell computation, MissionRequest list.

Phase 4a — ships with the production system.

Strategy: nearest-neighbour routing from a shared staging area through each
drone's assigned targets and back. No airspace constraints. Camera footprint
is a square ± CAMERA_FOOTPRINT_CELLS perpendicular and along the flight path.

Coordinate contract (PotentialBugs1 §1): all arithmetic in grid cells (metres).
Only the mission queue builder converts to lat/lon for the UTM layer.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import (
    CAMERA_FOOTPRINT_CELLS,
    CYCLE_INTERVAL_MIN,
    DRONE_RANGE_M,
    GRID_RESOLUTION_M,
    N_DRONES,
)
from .types import DronePlan, InformationField, MissionRequest, TerrainData
from .utils import bresenham, grid_to_latlon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nearest_neighbor_order(
    targets: list[tuple[int, int]],
    start: tuple[int, int],
) -> list[tuple[int, int]]:
    """Order targets greedily by nearest-neighbour from start."""
    remaining = list(targets)
    path: list[tuple[int, int]] = []
    current = start
    while remaining:
        nearest = min(
            remaining,
            key=lambda t: (t[0] - current[0]) ** 2 + (t[1] - current[1]) ** 2,
        )
        path.append(nearest)
        remaining.remove(nearest)
        current = nearest
    return path


def _assign_targets(
    targets: list[tuple[int, int]],
    n_drones: int,
    staging_area: tuple[int, int],
) -> dict[int, list[tuple[int, int]]]:
    """
    Assign targets to drones by sorting them by angle from the staging area
    and distributing round-robin, so adjacent drones cover adjacent sectors.
    """
    if not targets:
        return {i: [] for i in range(n_drones)}

    sr, sc = staging_area
    angles = np.arctan2(
        [t[0] - sr for t in targets],
        [t[1] - sc for t in targets],
    )
    sorted_targets = [targets[i] for i in np.argsort(angles)]
    assignments: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_drones)}
    for idx, target in enumerate(sorted_targets):
        assignments[idx % n_drones].append(target)
    return assignments


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def cells_along_path(
    waypoints: list[tuple[int, int]],
    shape: tuple[int, int],
    camera_footprint_cells: int = CAMERA_FOOTPRINT_CELLS,
) -> list[tuple[int, int]]:
    """
    Return all grid cells observed along a waypoint path.

    Uses Bresenham rasterisation between consecutive waypoints, then expands
    by ±camera_footprint_cells in each direction to model the sensor swath.
    """
    if len(waypoints) < 2:
        return list(waypoints) if waypoints else []

    rows, cols = shape
    seen: set[tuple[int, int]] = set()
    for i in range(len(waypoints) - 1):
        r0, c0 = waypoints[i]
        r1, c1 = waypoints[i + 1]
        for r, c in bresenham(r0, c0, r1, c1):
            for dr in range(-camera_footprint_cells, camera_footprint_cells + 1):
                for dc in range(-camera_footprint_cells, camera_footprint_cells + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        seen.add((nr, nc))
    return sorted(seen)


def plan_paths(
    selected_locations: list[tuple[int, int]],
    staging_area: tuple[int, int],
    n_drones: int = N_DRONES,
    shape: tuple[int, int] = (100, 100),
    resolution_m: float = GRID_RESOLUTION_M,
    camera_footprint_cells: int = CAMERA_FOOTPRINT_CELLS,
    drone_range_m: float = DRONE_RANGE_M,
) -> list[DronePlan]:
    """
    Assign drones to targets and compute each drone's observed cells.

    Returns one DronePlan per drone. Drones with no targets get an empty plan.
    Drone range is noted in the signature for future constraint enforcement;
    at hackathon scale it is not enforced.
    """
    assignments = _assign_targets(selected_locations, n_drones, staging_area)
    plans: list[DronePlan] = []

    for drone_id in range(n_drones):
        targets = assignments[drone_id]
        if not targets:
            plans.append(DronePlan(drone_id=drone_id, waypoints=[], cells_observed=[]))
            continue

        ordered   = _nearest_neighbor_order(targets, staging_area)
        waypoints = [staging_area] + ordered + [staging_area]
        observed  = cells_along_path(waypoints, shape, camera_footprint_cells)

        plans.append(DronePlan(
            drone_id=drone_id,
            waypoints=waypoints,
            cells_observed=observed,
        ))
        logger.debug(
            "Drone %d: %d targets, %d waypoints, %d cells observed",
            drone_id, len(targets), len(waypoints), len(observed),
        )

    return plans


def selections_to_mission_queue(
    drone_plans: list[DronePlan],
    info_field: InformationField,
    terrain: TerrainData,
    resolution_m: float = GRID_RESOLUTION_M,
    expiry_minutes: float = CYCLE_INTERVAL_MIN * 1.5,
) -> list[MissionRequest]:
    """
    Convert DronePlans to a list of MissionRequests for the UTM layer.

    Each MissionRequest is one drone's ordered path — a list of (lat, lon)
    waypoints derived from the plan's grid-cell waypoints.  Drones with no
    assigned targets produce no request.

    information_value is the sum of info_field.w over the target waypoints
    (i.e. excluding the staging-area start and return).
    Sorted by information_value descending.
    """
    origin_lat, origin_lon = terrain.origin_latlon
    requests: list[MissionRequest] = []

    for plan in drone_plans:
        # Waypoints are [staging, t1, t2, ..., staging]; skip if no targets.
        target_wps = plan.waypoints[1:-1] if len(plan.waypoints) > 2 else []
        if not target_wps:
            continue

        path = [
            grid_to_latlon(r, c, origin_lat, origin_lon, resolution_m)
            for r, c in plan.waypoints   # full path including staging
        ]

        # Total information value over the target cells.
        total_w = sum(float(info_field.w[r, c]) for r, c in target_wps)

        # Dominant variable: most frequent across target cells.
        dominant_counts: dict[str, float] = {k: 0.0 for k in info_field.w_by_variable}
        for r, c in target_wps:
            d = max(info_field.w_by_variable, key=lambda k: float(info_field.w_by_variable[k][r, c]))
            dominant_counts[d] += 1.0
        dominant = max(dominant_counts, key=lambda k: dominant_counts[k])

        requests.append(MissionRequest(
            drone_id=plan.drone_id,
            path=path,
            information_value=total_w,
            dominant_variable=dominant,
            expiry_minutes=expiry_minutes,
        ))

    requests.sort(key=lambda req: req.information_value, reverse=True)
    return requests
