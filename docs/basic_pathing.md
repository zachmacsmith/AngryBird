# Basic Pathing: Point Selection → Path Planning → Mission Queue

This document covers the sub-pipeline that runs at the end of every IGNIS cycle:
converting an information field into concrete drone flight paths.

---

## Overview

```
InformationField
      │
      ▼
  Selector  ─────────────────────────  SelectionResult
  (registry lookup by name)             selected_locations: list[(row, col)]
      │
      ▼
  plan_paths()  ──────────────────────  list[DronePlan]
  (assign points to drones,             per drone: ordered waypoints + cells_observed
   nearest-neighbour routing)
      │
      ▼
  selections_to_mission_queue()  ──────  MissionQueue
  (convert grid cells → lat/lon paths)   per drone: ordered (lat, lon) path
      │
      ├── UTM layer (production)
      │
      └── SimulationRunner (simulation)
```

---

## Stage 1: Point Selection

**Trigger:** `IGNISOrchestrator.run_cycle()` after the ensemble and information
field have been computed.

**Call:**
```python
primary_result = self.registry.run(
    self.selector_name, info_field, self.gp, ensemble, k=self.n_drones,
)
```

**Input:** `InformationField` — a grid of per-cell observation value `w`,
broken down by variable (`fmc`, `wind_speed`, `wind_dir`).

**Output:** `SelectionResult`
```python
selected_locations: list[tuple[int, int]]  # N grid cells, ranked by value
marginal_gains:     list[float]
strategy_name:      str
solver_metadata:    Optional[dict]         # QUBO only
```

**Selector registry** — `angrybird/selectors/__init__.py`

All selectors are registered at import time and looked up by name string.
The active selector is set via `IGNIS_SELECTOR` in `angrybird/config.py`
and flows through `SimulationConfig.selector_name` → orchestrator constructor.

| Name | Class | File |
|---|---|---|
| `"greedy"` | `GreedySelector` | `selectors/greedy.py` |
| `"qubo"` | `QUBOSelector` | `selectors/qubo.py` |
| `"uniform"` | `UniformSelector` | `selectors/baselines.py` |
| `"fire_front"` | `FireFrontSelector` | `selectors/baselines.py` |

All selectors implement the `Selector` protocol (`selectors/base.py`):
```python
def select(self, info_field, gp, ensemble, k) -> SelectionResult: ...
```

---

## Stage 2: Path Planning

**Call:** `plan_paths()` in `angrybird/path_planner.py`
```python
drone_plans = plan_paths(
    primary_result.selected_locations,
    staging_area=self.staging_area,
    n_drones=self.n_drones,
    shape=shape,
    resolution_m=self.resolution_m,
)
```

**What it does — two steps internally:**

1. **`_assign_targets()`** — distributes the N selected points across drones.
   Points are sorted by angle from the staging area then distributed
   round-robin, so adjacent drones cover adjacent angular sectors.

2. **`_nearest_neighbor_order()`** — within each drone's assigned points,
   orders them greedily by shortest hop from the staging area.
   Full waypoint list is `[staging] + ordered_targets + [staging]`.

**Output:** `list[DronePlan]`, one entry per drone
```python
@dataclass(frozen=True)
class DronePlan:
    drone_id:       int
    waypoints:      list[tuple[int, int]]   # ordered grid (row, col), starts and ends at staging
    cells_observed: list[tuple[int, int]]   # all cells under the camera swath along the path
```

`cells_observed` is computed by `cells_along_path()`: Bresenham rasterisation
between consecutive waypoints, expanded by `±CAMERA_FOOTPRINT_CELLS` to model
the sensor swath.

---

## Stage 3: Mission Queue

**Call:** `selections_to_mission_queue()` in `angrybird/path_planner.py`
```python
mission_queue = selections_to_mission_queue(
    drone_plans, info_field, self.terrain, self.resolution_m,
)
```

Converts each `DronePlan` into a `MissionRequest` — the external-facing
representation of one drone's mission for the current cycle.

```python
@dataclass(frozen=True)
class MissionRequest:
    drone_id:           int
    path:               list[tuple[float, float]]  # ordered (lat, lon), full route including staging
    information_value:  float                      # sum of info_field.w over target cells
    dominant_variable:  str                        # most frequent dominant variable along path
    expiry_minutes:     float                      # when this plan expires (1.5× cycle interval)
```

`path` covers the complete flight: staging → target 1 → target 2 → … → staging.
Grid coordinates are converted to (lat, lon) using `terrain.origin_latlon` as
the NW corner and the grid resolution.

The `MissionQueue` holds one request per drone that has at least one target,
sorted by `information_value` descending:
```python
@dataclass(frozen=True)
class MissionQueue:
    requests: list[MissionRequest]
```

---

## Downstream Interfaces

### Production (UTM layer)

`run_cycle()` returns `(MissionQueue, CycleReport)`. The `MissionQueue` is the
handoff to the UTM/command layer: each `MissionRequest.path` is an ordered
sequence of (lat, lon) waypoints the UTM should send to the physical drone.

### Simulation (`SimulationRunner`, `wispsim/runner.py`)

The simulation does not consume `MissionQueue` directly. Instead:

- **Drone dispatch:** `_assign_drone_waypoints()` reads
  `orchestrator._previous_selections` (raw grid cells) and calls
  `assign_waypoints()` from `wispsim/drone_sim.py` for each idle drone.
  Currently one waypoint per drone; the full ordered path from `DronePlan`
  is not yet threaded through.

- **Observation collection:** `cells_observed` from each `DronePlan` is
  the set of cells the drone actually scans. This feeds the `ObservationBuffer`
  which accumulates FMC/wind readings for the next cycle's assimilation.

### Strategy comparison (`CycleRunner`, `wispsim/runner.py`)

`CycleRunner` runs all registered selectors on the same ensemble each cycle
for PERR evaluation. For each strategy it calls `plan_paths()` independently,
then passes `drone_plans_by_strategy[name].cells_observed` to the
`CounterfactualEvaluator` to score how much information each strategy would
have gathered.

---

## Coordinate Systems

| Layer | Representation | Used by |
|---|---|---|
| Grid cells | `(row, col)` integers | Selectors, `plan_paths`, fire engine, GP |
| Metric (m) | `(y_m, x_m)` floats | `drone_sim` movement, `cells_along_path` |
| Geographic | `(lat, lon)` floats | `MissionRequest.path`, UTM layer |

Conversion between grid and geographic uses `grid_to_latlon()` in
`angrybird/utils.py`, anchored on `terrain.origin_latlon` (NW corner, degrees).

---

## Key Config Parameters

| Parameter | Default | Effect |
|---|---|---|
| `IGNIS_SELECTOR` | `"greedy"` | Which selector runs in production |
| `N_DRONES` | `5` | Number of drone paths planned per cycle |
| `MIN_SELECTION_SPACING_M` | `500 m` | Minimum distance between selected points |
| `QUBO_MAX_CANDIDATES` | `300` | Top-M candidates passed to QUBO solver |
| `CAMERA_FOOTPRINT_CELLS` | `1` | Half-width of camera swath (cells each side) |
| `DRONE_RANGE_M` | `20 000 m` | Maximum one-way range (noted; not enforced) |
| `CYCLE_INTERVAL_MIN` | — | Plan expiry = 1.5 × cycle interval |
