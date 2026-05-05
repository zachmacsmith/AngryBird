"""
Drone simulator: position, movement, and sensor simulation.

Phase 4b — simulation harness only.

Each drone moves at constant cruise speed through a waypoint queue,
collecting observations continuously.  FMC is measured within a camera
footprint (circular radius); wind is a point measurement at the drone's
nadir cell.

Coordinate convention:
  All positions are in metres from the NW corner of the grid.
  Row 0 = northernmost row → y = 0 m.
  Col 0 = westernmost col  → x = 0 m.
  Position array is [y_m, x_m].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from angrybird.config import (
    CAMERA_FOOTPRINT_EDGE_NOISE_FACTOR,
    CAMERA_FOOTPRINT_M,
    DRONE_SPEED_MS,
    FIRE_DEGRADATION_RADIUS_M,
    GRID_RESOLUTION_M,
    OBS_FIRE_DEGRADATION_FACTOR,
    OBS_FMC_SIGMA,
    OBS_WIND_DIR_SIGMA,
    OBS_WIND_SPEED_SIGMA,
    SENSOR_FMC_R2,
)
from angrybird.observations import FireDetectionObservation
from angrybird.types import DroneObservation


# ---------------------------------------------------------------------------
# Noise config
# ---------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    """Sensor noise parameters for the drone simulator."""
    fmc_sigma:               float = OBS_FMC_SIGMA
    ws_sigma:                float = OBS_WIND_SPEED_SIGMA
    wd_sigma:                float = OBS_WIND_DIR_SIGMA
    camera_footprint_m:      float = CAMERA_FOOTPRINT_M
    degrade_near_fire:       bool  = True
    fire_degradation_radius_m: float = FIRE_DEGRADATION_RADIUS_M
    fire_degradation_factor: float = OBS_FIRE_DEGRADATION_FACTOR


# ---------------------------------------------------------------------------
# Drone state
# ---------------------------------------------------------------------------

@dataclass
class DroneState:
    """Mutable state for one simulated drone."""
    drone_id:                str
    position:                np.ndarray       # [y_m, x_m] float64
    speed:                   float            # m/s cruise speed
    status:                  str              # "idle"|"transit"|"returning"
    waypoint_queue:          list[np.ndarray] # remaining target positions (metres)
    current_target:          Optional[np.ndarray]
    path_history:            list[np.ndarray] # positions visited (for trail rendering)
    endurance_remaining_s:   float            # seconds of flight time left
    base_position:           np.ndarray       # home staging area [y_m, x_m]


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def cell_to_pos_m(cell: tuple[int, int], resolution_m: float) -> np.ndarray:
    """Convert grid (row, col) to domain-metres [y_m, x_m] (cell centre)."""
    r, c = cell
    return np.array([(r + 0.5) * resolution_m, (c + 0.5) * resolution_m], dtype=np.float64)


def pos_m_to_cell(
    pos_m: np.ndarray, resolution_m: float, shape: tuple[int, int]
) -> tuple[int, int]:
    """Convert domain-metres [y_m, x_m] to (row, col). Clamps to grid bounds."""
    rows, cols = shape
    r = int(pos_m[0] / resolution_m)
    c = int(pos_m[1] / resolution_m)
    return (max(0, min(rows - 1, r)), max(0, min(cols - 1, c)))


# ---------------------------------------------------------------------------
# Drone movement
# ---------------------------------------------------------------------------

def move_drone(drone: DroneState, dt: float) -> None:
    """
    Advance one drone by dt seconds toward its current target.

    When a waypoint is reached the drone pops the next one.  When the queue
    is empty the drone returns to base and becomes "idle".
    Endurance is decremented; the caller is responsible for refuelling.
    """
    if drone.current_target is None:
        if drone.waypoint_queue:
            drone.current_target = drone.waypoint_queue.pop(0)
            drone.status = "transit"
        else:
            drone.status = "idle"
            return

    direction = drone.current_target - drone.position
    dist      = float(np.linalg.norm(direction))
    step_dist = drone.speed * dt

    if dist <= step_dist:
        # Arrived at current target
        drone.position = drone.current_target.copy()
        drone.path_history.append(drone.position.copy())

        if drone.waypoint_queue:
            drone.current_target = drone.waypoint_queue.pop(0)
            drone.status = "transit"
        else:
            drone.current_target = drone.base_position.copy()
            drone.status = "returning"
    else:
        drone.position = drone.position + (direction / dist) * step_dist
        drone.path_history.append(drone.position.copy())

    drone.endurance_remaining_s -= dt


# ---------------------------------------------------------------------------
# Sensor simulation
# ---------------------------------------------------------------------------

def collect_observations(
    drone: DroneState,
    fmc_field: np.ndarray,
    wind_speed_field: np.ndarray,
    wind_direction_field: np.ndarray,
    terrain_shape: tuple[int, int],
    resolution_m: float,
    noise: NoiseConfig,
    current_time: float,
    fire_arrival_times: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[DroneObservation]:
    """
    Generate sensor observations from the drone's current nadir position.

    FMC is observed within a circular footprint (camera_footprint_m radius),
    with noise increasing toward the footprint edge.  Wind is a point
    measurement at the nadir cell only.

    If fire_arrival_times is provided and degrade_near_fire is set, noise is
    multiplied by fire_degradation_factor within fire_degradation_radius_m of
    the active fire front (smoke + thermal interference).
    """
    if rng is None:
        rng = np.random.default_rng()

    center_cell = pos_m_to_cell(drone.position, resolution_m, terrain_shape)
    rows, cols  = terrain_shape
    r0, c0      = center_cell

    # Fire-proximity degradation factor
    fire_factor = 1.0
    if noise.degrade_near_fire and fire_arrival_times is not None:
        dist_to_front = _dist_to_fire_front_m(
            drone.position, fire_arrival_times, current_time, resolution_m)
        if dist_to_front < noise.fire_degradation_radius_m:
            fire_factor = noise.fire_degradation_factor

    observations: list[DroneObservation] = []
    footprint_cells = int(math.ceil(noise.camera_footprint_m / resolution_m))

    for dr in range(-footprint_cells, footprint_cells + 1):
        for dc in range(-footprint_cells, footprint_cells + 1):
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            dist_cells = math.sqrt(dr ** 2 + dc ** 2)
            if dist_cells > footprint_cells:
                continue

            # Noise increases toward edge of footprint
            edge_factor = 1.0 + CAMERA_FOOTPRINT_EDGE_NOISE_FACTOR * (dist_cells / max(footprint_cells, 1))
            fmc_sigma   = noise.fmc_sigma * edge_factor * fire_factor

            obs_fmc = float(fmc_field[r, c]) + float(rng.normal(0.0, fmc_sigma))

            if dr == 0 and dc == 0:
                # Centre cell: also measure wind
                ws_sigma = noise.ws_sigma * fire_factor
                wd_sigma = noise.wd_sigma * fire_factor
                obs_ws  = max(0.0, float(wind_speed_field[r, c]) + float(rng.normal(0.0, ws_sigma)))
                obs_wd  = (float(wind_direction_field[r, c]) + float(rng.normal(0.0, wd_sigma))) % 360.0
                observations.append(DroneObservation(
                    location=(r, c),
                    fmc=obs_fmc, fmc_sigma=fmc_sigma,
                    wind_speed=obs_ws, wind_speed_sigma=ws_sigma,
                    wind_dir=obs_wd, wind_dir_sigma=wd_sigma,
                    timestamp=current_time, drone_id=drone.drone_id,
                ))
            else:
                # Off-centre: FMC only
                observations.append(DroneObservation(
                    location=(r, c),
                    fmc=obs_fmc, fmc_sigma=fmc_sigma,
                    wind_speed=float("nan"), wind_speed_sigma=float("nan"),
                    wind_dir=None, wind_dir_sigma=None,
                    timestamp=current_time, drone_id=drone.drone_id,
                ))

    return observations


def collect_fire_observation(
    drone: DroneState,
    fire_arrival_times: np.ndarray,
    terrain_shape: tuple[int, int],
    resolution_m: float,
    current_time: float,
    rng: Optional[np.random.Generator] = None,
    confidence: float = 0.85,
) -> Optional[FireDetectionObservation]:
    """
    Thermal camera fire detection at the drone's nadir cell.

    Returns a FireDetectionObservation (fire or no-fire) with the given
    confidence, or None if the drone is outside the grid.  A small fraction
    of detections are flipped to model sensor errors.
    """
    if rng is None:
        rng = np.random.default_rng()

    r, c = pos_m_to_cell(drone.position, resolution_m, terrain_shape)
    rows, cols = terrain_shape
    if not (0 <= r < rows and 0 <= c < cols):
        return None

    is_burning = float(fire_arrival_times[r, c]) <= current_time
    # Sensor error: flip detection with probability (1 - confidence)
    if rng.random() > confidence:
        is_burning = not is_burning

    return FireDetectionObservation(
        _source_id="drone_thermal",
        _timestamp=current_time,
        location=(r, c),
        is_fire=is_burning,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist_to_fire_front_m(
    pos_m: np.ndarray,
    arrival_times: np.ndarray,
    current_time: float,
    resolution_m: float,
) -> float:
    """Approximate distance (metres) from pos_m to the nearest fire front cell."""
    burned = arrival_times <= current_time
    unburned = ~burned
    if not burned.any() or not unburned.any():
        return float("inf")

    # Front = burned cells adjacent to unburned cells
    from scipy.ndimage import binary_dilation
    front_mask = burned & binary_dilation(unburned)
    if not front_mask.any():
        return float("inf")

    front_rows, front_cols = np.where(front_mask)
    fy = (front_rows + 0.5) * resolution_m
    fx = (front_cols + 0.5) * resolution_m

    dy = fy - pos_m[0]
    dx = fx - pos_m[1]
    return float(np.sqrt(dy ** 2 + dx ** 2).min())


def assign_waypoints(
    drone: DroneState,
    path_cells: list[tuple[int, int]],
    resolution_m: float,
) -> None:
    """Replace a drone's waypoint queue with an ordered sequence of cells."""
    positions = [cell_to_pos_m(cell, resolution_m) for cell in path_cells]
    if not positions:
        return
    drone.waypoint_queue = positions
    drone.current_target = positions[0]
    drone.status = "transit"
