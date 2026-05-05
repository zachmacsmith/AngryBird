"""
Correlation-Domain Path Selector with mode-based continuous flight planning.

Implements docs/Path Optimization.md (Part 1) and docs/Drone Path.md.

Domains are terrain-adaptive (Felzenszwalb on static terrain features), so
boundaries follow ridgelines, fuel transitions, and aspect breaks.

Three operating modes per drone (all transitions are one-way within a sortie):

  NORMAL    — budget=d_cycle, free exploration, nearest-GS return cost
  RETURN    — budget=min(d_cycle, r-d_safety), reachability invariant enforced,
               target GS locked at NORMAL→RETURN transition
  EMERGENCY — direct shortest path to target GS, no optimisation

Drone states persist across cycles (same sortie). On GS landing the
orchestrator resets battery and mode for the next sortie.
"""

from __future__ import annotations

import copy
import heapq
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..config import (
    CAMERA_FOOTPRINT_CELLS,
    DRONE_CYCLE_DISTANCE_M,
    DRONE_MIN_USEFUL_INFO,
    DRONE_RANGE_M,
    DRONE_RETURN_THRESHOLD,
    DRONE_REVISIT_PERCENTILE,
    DRONE_SAFETY_FRACTION,
    DRONE_SAFETY_MARGIN_M,
    GP_CORRELATION_LENGTH_FMC_M,
    GRID_RESOLUTION_M,
)
from ..path_planner import cells_along_path
from ..types import (
    DronePlan,
    DroneFlightState,
    DroneMode,
    EnsembleResult,
    InformationField,
    PathSelectionResult,
    TerrainData,
)

if TYPE_CHECKING:
    from ..gp import IGNISGPPrior

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain graph data structures
# ---------------------------------------------------------------------------

@dataclass
class CorrelationDomain:
    domain_id: int
    cells: list[tuple[int, int]]
    representative_cell: tuple[int, int]
    centroid_m: np.ndarray          # (row_m, col_m) — row × resolution, col × resolution
    info_value: float
    dominant_variable: str
    area_cells: int


@dataclass
class DomainEdge:
    source: int
    target: int
    cross_correlation: float
    information_gain: float
    real_distance_m: float


@dataclass
class CorrelationGraph:
    domains: list[CorrelationDomain]
    adj: dict[int, list[DomainEdge]]   # domain_id → outbound edges
    label_map: np.ndarray              # int[rows, cols] — domain_id per cell

    def domain_id_for_cell(self, cell: tuple[int, int]) -> int:
        r, c = cell
        return int(self.label_map[r, c])

    def edge(self, src: int, tgt: int) -> Optional[DomainEdge]:
        for e in self.adj.get(src, []):
            if e.target == tgt:
                return e
        return None


# ---------------------------------------------------------------------------
# Terrain feature extraction
# ---------------------------------------------------------------------------

def _terrain_features(terrain: TerrainData) -> np.ndarray:
    """
    Build normalised multi-channel feature array (rows, cols, 6).
    Aspect encoded as (cos, sin) to handle circularity.
    """
    def norm(x: np.ndarray) -> np.ndarray:
        std = float(x.std())
        return (x - x.mean()) / (std + 1e-10)

    aspect_rad = np.radians(terrain.aspect.astype(np.float64))
    aspect_cos = np.cos(aspect_rad)
    aspect_sin = np.sin(aspect_rad)

    return np.stack([
        norm(terrain.elevation.astype(np.float64)) * 1.0,
        norm(terrain.slope.astype(np.float64))    * 0.5,
        norm(aspect_cos)                           * 1.5,
        norm(aspect_sin)                           * 1.5,
        norm(terrain.fuel_model.astype(np.float64)) * 2.0,
        norm(terrain.canopy_cover.astype(np.float64)) * 0.8,
    ], axis=-1).astype(np.float64)


# ---------------------------------------------------------------------------
# Felzenszwalb terrain-adaptive segmentation
# ---------------------------------------------------------------------------

def _felzenszwalb_label_map(
    terrain: TerrainData,
    features: np.ndarray,
    correlation_length_m: float,
    resolution_m: float,
    min_domain_cells: int,
) -> np.ndarray:
    import warnings
    from skimage.segmentation import felzenszwalb

    scale = correlation_length_m / resolution_m * 0.5
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*multichannel.*")
        labels = felzenszwalb(
            features,
            scale=scale,
            sigma=0.8,
            min_size=min_domain_cells,
            channel_axis=-1,
        )
    return labels.astype(np.int32)


def _regular_grid_label_map(
    shape: tuple[int, int],
    correlation_length_m: float,
    resolution_m: float,
) -> np.ndarray:
    rows, cols = shape
    domain_size = max(1, int(round(correlation_length_m / resolution_m)))
    n_dc = (cols + domain_size - 1) // domain_size
    labels = np.zeros((rows, cols), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            labels[r, c] = (r // domain_size) * n_dc + (c // domain_size)
    return labels


def _is_pathological(label_map: np.ndarray) -> bool:
    sizes = np.bincount(label_map.ravel())
    n_cells = label_map.size
    if sizes.max() > 0.5 * n_cells:
        return True
    if (sizes == 1).sum() > 0.25 * n_cells:
        return True
    return False


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_correlation_graph(
    label_map: np.ndarray,
    features: np.ndarray,
    w_field: np.ndarray,
    w_by_variable: dict[str, np.ndarray],
    resolution_m: float,
) -> CorrelationGraph:
    rows, cols = label_map.shape
    unique_ids = np.unique(label_map)

    domain_dict: dict[int, CorrelationDomain] = {}
    for d_id in unique_ids:
        mask = label_map == d_id
        rs, cs = np.where(mask)
        cells = list(zip(rs.tolist(), cs.tolist()))

        w_vals = w_field[mask]
        best_idx = int(np.argmax(w_vals))
        rep = cells[best_idx]

        centroid_m = np.array([rs.mean() * resolution_m, cs.mean() * resolution_m])

        dom_var = max(
            w_by_variable,
            key=lambda v: float(w_by_variable[v][rep[0], rep[1]]),
        )

        domain_dict[int(d_id)] = CorrelationDomain(
            domain_id=int(d_id),
            cells=cells,
            representative_cell=rep,
            centroid_m=centroid_m,
            info_value=float(w_vals.max()),
            dominant_variable=dom_var,
            area_cells=len(cells),
        )

    # 4-connected adjacency scan
    adjacency: set[tuple[int, int]] = set()
    for r in range(rows - 1):
        for c in range(cols - 1):
            here  = int(label_map[r,     c])
            right = int(label_map[r,     c + 1])
            below = int(label_map[r + 1, c])
            if here != right:
                adjacency.add((min(here, right), max(here, right)))
            if here != below:
                adjacency.add((min(here, below), max(here, below)))
    for r in range(rows - 1):
        here  = int(label_map[r,     cols - 1])
        below = int(label_map[r + 1, cols - 1])
        if here != below:
            adjacency.add((min(here, below), max(here, below)))
    for c in range(cols - 1):
        here  = int(label_map[rows - 1, c])
        right = int(label_map[rows - 1, c + 1])
        if here != right:
            adjacency.add((min(here, right), max(here, right)))

    adj: dict[int, list[DomainEdge]] = {int(did): [] for did in unique_ids}
    for d_i, d_j in adjacency:
        di = domain_dict[d_i]
        dj = domain_dict[d_j]
        real_dist = float(np.linalg.norm(di.centroid_m - dj.centroid_m))
        ri, ci = di.representative_cell
        rj, cj = dj.representative_cell
        feat_diff = features[ri, ci] - features[rj, cj]
        dissimilarity = float(np.sqrt((feat_diff ** 2).sum()))
        cross_corr = float(np.exp(-dissimilarity))
        edge_info = (1.0 - cross_corr) * min(di.info_value, dj.info_value)
        adj[d_i].append(DomainEdge(d_i, d_j, cross_corr, edge_info, real_dist))
        adj[d_j].append(DomainEdge(d_j, d_i, cross_corr, edge_info, real_dist))

    return CorrelationGraph(
        domains=list(domain_dict.values()),
        adj=adj,
        label_map=label_map,
    )


# ---------------------------------------------------------------------------
# Dijkstra helpers
# ---------------------------------------------------------------------------

def _dijkstra(graph: CorrelationGraph, start: int) -> dict[int, float]:
    """Single-source Dijkstra. Returns {domain_id: distance}."""
    dist: dict[int, float] = {d.domain_id: np.inf for d in graph.domains}
    dist[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for edge in graph.adj.get(u, []):
            v = edge.target
            nd = d + edge.real_distance_m
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _dijkstra_path(
    graph: CorrelationGraph, start: int, end: int
) -> tuple[float, list[int]]:
    """Dijkstra returning (distance, node_list) shortest path."""
    dist: dict[int, float] = {d.domain_id: np.inf for d in graph.domains}
    prev: dict[int, int] = {}
    dist[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == end:
            break
        if d > dist[u]:
            continue
        for edge in graph.adj.get(u, []):
            v = edge.target
            nd = d + edge.real_distance_m
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    # Reconstruct
    path: list[int] = []
    node = end
    while node in prev:
        path.append(node)
        node = prev[node]
    path.append(start)
    path.reverse()
    return dist.get(end, np.inf), path


# ---------------------------------------------------------------------------
# GS distance tables
# ---------------------------------------------------------------------------

def _pos_to_domain(position_m: np.ndarray, graph: CorrelationGraph, resolution_m: float) -> int:
    """Convert grid-metre position (row_m, col_m) to domain ID."""
    rows, cols = graph.label_map.shape
    row = int(np.clip(position_m[0] / resolution_m, 0, rows - 1))
    col = int(np.clip(position_m[1] / resolution_m, 0, cols - 1))
    return graph.domain_id_for_cell((row, col))


def _compute_all_gs_distances(
    graph: CorrelationGraph,
    ground_stations_m: list[np.ndarray],
    resolution_m: float,
) -> list[dict[int, float]]:
    """
    For each GS, run Dijkstra from its domain outward.
    Returns list[dict[domain_id, distance]] — one dict per GS.
    """
    result = []
    for gs_m in ground_stations_m:
        gs_domain = _pos_to_domain(gs_m, graph, resolution_m)
        result.append(_dijkstra(graph, gs_domain))
    return result


def _min_gs_return_costs(gs_distances: list[dict[int, float]]) -> dict[int, float]:
    """Minimum distance from each domain to any GS."""
    if not gs_distances:
        return {}
    merged: dict[int, float] = dict(gs_distances[0])
    for gsd in gs_distances[1:]:
        for d_id, dist in gsd.items():
            if dist < merged.get(d_id, np.inf):
                merged[d_id] = dist
    return merged


# ---------------------------------------------------------------------------
# Mode transition logic
# ---------------------------------------------------------------------------

def _check_mode_transitions(
    state: DroneFlightState,
    graph: CorrelationGraph,
    ground_stations_m: list[np.ndarray],
    gs_distances: list[dict[int, float]],
    resolution_m: float,
    d_max: float,
    d_safety: float,
    R_threshold: float,
) -> DroneFlightState:
    """
    Apply NORMAL→RETURN and RETURN→EMERGENCY transitions in-place.
    Returns a shallow copy of state with updated mode/target_gs_idx.
    """
    state = copy.copy(state)
    state.visited_domains = set(state.visited_domains)

    if state.mode == DroneMode.EMERGENCY:
        return state  # terminal mode

    current_domain = _pos_to_domain(state.position_m, graph, resolution_m)

    if state.mode == DroneMode.NORMAL:
        # Find nearest reachable GS and its distance
        best_dist = np.inf
        best_gs_idx = -1
        for gs_idx, gs_dist in enumerate(gs_distances):
            d = gs_dist.get(current_domain, np.inf)
            if d < best_dist and d < state.remaining_range_m - d_safety:
                best_dist = d
                best_gs_idx = gs_idx

        if best_gs_idx == -1:
            logger.warning("Drone %d: no reachable GS from domain %d — EMERGENCY",
                           state.drone_id, current_domain)
            state.mode = DroneMode.EMERGENCY
            # Use closest GS even if barely unreachable
            for gs_idx, gs_dist in enumerate(gs_distances):
                d = gs_dist.get(current_domain, np.inf)
                if d < best_dist:
                    best_dist = d
                    best_gs_idx = gs_idx
            state.target_gs_idx = max(best_gs_idx, 0)
            return state

        reserve = (state.remaining_range_m - best_dist) / d_max
        if reserve <= R_threshold:
            logger.info("Drone %d: NORMAL→RETURN (reserve=%.2f ≤ %.2f)",
                        state.drone_id, reserve, R_threshold)
            state.mode = DroneMode.RETURN
            state.target_gs_idx = best_gs_idx

    if state.mode == DroneMode.RETURN:
        tgt_dist = gs_distances[state.target_gs_idx]
        d_return_target = tgt_dist.get(current_domain, np.inf)

        # GS lock override: only if target is physically unreachable (no battery to get there)
        if d_return_target > state.remaining_range_m:
            logger.info("Drone %d: target GS physically unreachable (d=%.0f > r=%.0f) — relocking",
                        state.drone_id, d_return_target, state.remaining_range_m)
            best_dist = np.inf
            best_gs_idx = -1
            for gs_idx, gs_dist in enumerate(gs_distances):
                d = gs_dist.get(current_domain, np.inf)
                if d < best_dist and d <= state.remaining_range_m:
                    best_dist = d
                    best_gs_idx = gs_idx
            if best_gs_idx == -1:
                logger.warning("Drone %d: RETURN→EMERGENCY (all GS physically unreachable)", state.drone_id)
                state.mode = DroneMode.EMERGENCY
                return state
            state.target_gs_idx = best_gs_idx
            d_return_target = best_dist

        # RETURN→EMERGENCY: safety margin exhausted — can no longer explore
        if state.remaining_range_m <= d_return_target + d_safety:
            logger.info("Drone %d: RETURN→EMERGENCY (r=%.0f ≤ d_ret=%.0f + d_sfty=%.0f)",
                        state.drone_id, state.remaining_range_m, d_return_target, d_safety)
            state.mode = DroneMode.EMERGENCY

    return state


# ---------------------------------------------------------------------------
# Greedy path planner (NORMAL and RETURN modes)
# ---------------------------------------------------------------------------

def _plan_greedy_path(
    graph: CorrelationGraph,
    start_domain: int,
    budget_m: float,
    return_costs: dict[int, float],    # domain → distance to return destination
    current_w: dict[int, float],       # mutable — visited domains zeroed on selection
    current_var: np.ndarray,           # GP variance, updated in-place
    gp: "IGNISGPPrior",
    visited_global: set[int],          # already-observed domains (entire sortie)
) -> tuple[list[int], float]:
    """
    Greedy orienteering from start_domain up to budget_m metres.

    return_costs governs the feasibility constraint:
      - NORMAL:  nearest-GS distances (no safety margin in budget)
      - RETURN:  target-GS distances (budget already = r - d_safety)

    Returns (domain_id_path, total_distance_m).
    """
    path = [start_domain]
    visited = {start_domain} | visited_global
    remaining = budget_m
    current = start_domain
    total_dist = 0.0
    rep_by_id = {d.domain_id: d.representative_cell for d in graph.domains}

    while True:
        best_domain: Optional[int] = None
        best_score = -np.inf
        best_travel = 0.0

        for edge in graph.adj.get(current, []):
            nxt = edge.target
            if nxt in visited:
                continue
            w = current_w.get(nxt, 0.0)
            if w <= 0.0:
                continue
            ret = return_costs.get(nxt, np.inf)
            if edge.real_distance_m + ret > remaining:
                continue

            total_info = w + edge.information_gain
            score = total_info / (edge.real_distance_m + 1.0)

            if score > best_score:
                best_score = score
                best_domain = nxt
                best_travel = edge.real_distance_m

        if best_domain is None:
            break

        # Zero selected domain immediately so later drones skip it.
        current_w[best_domain] = 0.0

        # Discount unvisited domains by marginal variance reduction.
        rep_cell = rep_by_id[best_domain]
        updated_var = gp.conditional_variance(current_var, rep_cell)

        for d in graph.domains:
            if d.domain_id in visited or d.domain_id == best_domain:
                continue
            r, c = rep_by_id[d.domain_id]
            old_v = float(current_var[r, c])
            if old_v > 1e-12:
                ratio = float(updated_var[r, c]) / old_v
                current_w[d.domain_id] = current_w.get(d.domain_id, 0.0) * ratio

        np.copyto(current_var, updated_var)

        path.append(best_domain)
        visited.add(best_domain)
        remaining -= best_travel
        total_dist += best_travel
        current = best_domain

    return path, total_dist


# ---------------------------------------------------------------------------
# Revisit threshold
# ---------------------------------------------------------------------------

def _apply_revisit_threshold(
    visited_domains: set[int],
    domain_w: dict[int, float],
    percentile: float,
) -> set[int]:
    """Remove domains from visited_domains if their w_i > top-percentile threshold."""
    all_w = [v for v in domain_w.values() if v > 0]
    if not all_w:
        return set(visited_domains)
    threshold = float(np.percentile(all_w, percentile))
    return {d for d in visited_domains if domain_w.get(d, 0.0) <= threshold}


# ---------------------------------------------------------------------------
# Convert domain path → DronePlan
# ---------------------------------------------------------------------------

def _domains_to_drone_plan(
    drone_id: int,
    domain_ids: list[int],
    graph: CorrelationGraph,
    start_pos_m: np.ndarray,
    shape: tuple[int, int],
    resolution_m: float,
    camera_footprint_cells: int,
    plan_distance_m: float,
    drone_mode: str,
) -> DronePlan:
    if len(domain_ids) <= 1:
        return DronePlan(
            drone_id=drone_id,
            waypoints=[],
            cells_observed=[],
            plan_distance_m=0.0,
            drone_mode=drone_mode,
        )

    domain_by_id = {d.domain_id: d for d in graph.domains}
    # Start waypoint derived from position_m
    rows, cols = shape
    start_row = int(np.clip(start_pos_m[0] / resolution_m, 0, rows - 1))
    start_col = int(np.clip(start_pos_m[1] / resolution_m, 0, cols - 1))
    start_cell = (start_row, start_col)

    target_wps = [
        domain_by_id[did].representative_cell
        for did in domain_ids[1:]
    ]
    waypoints = [start_cell] + target_wps
    observed = cells_along_path(waypoints, shape, camera_footprint_cells)

    return DronePlan(
        drone_id=drone_id,
        waypoints=waypoints,
        cells_observed=observed,
        plan_distance_m=plan_distance_m,
        drone_mode=drone_mode,
    )


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class CorrelationPathSelector:
    """
    Continuous-flight path selector with NORMAL / RETURN / EMERGENCY modes.

    On the first cycle, all drones start at the staging area with full battery
    in NORMAL mode.  Subsequent cycles receive persistent DroneFlightState
    objects from the orchestrator so the planner knows each drone's current
    position, remaining range, and mode.

    Multiple ground stations are supported; pass ground_stations_m to select().
    The default is a single GS at the staging_area.
    """

    name = "correlation_path"
    kind = "paths"

    def __init__(
        self,
        correlation_length_m: float = GP_CORRELATION_LENGTH_FMC_M,
        drone_range_m: float = DRONE_RANGE_M,
        camera_footprint_cells: int = CAMERA_FOOTPRINT_CELLS,
        min_domain_cells: int = 10,
        d_cycle_m: float = DRONE_CYCLE_DISTANCE_M,
        R_threshold: float = DRONE_RETURN_THRESHOLD,
        safety_fraction: float = DRONE_SAFETY_FRACTION,
        revisit_percentile: float = DRONE_REVISIT_PERCENTILE,
        min_useful_info: float = DRONE_MIN_USEFUL_INFO,
    ) -> None:
        self.correlation_length_m   = correlation_length_m
        self.drone_range_m          = drone_range_m
        self.camera_footprint_cells = camera_footprint_cells
        self.min_domain_cells       = min_domain_cells
        self.d_cycle_m              = d_cycle_m
        self.R_threshold            = R_threshold
        self.d_safety               = safety_fraction * drone_range_m
        self.revisit_percentile     = revisit_percentile
        self.min_useful_info        = min_useful_info

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _make_initial_states(
        self,
        n_drones: int,
        staging_m: np.ndarray,
    ) -> list[DroneFlightState]:
        return [
            DroneFlightState(
                drone_id=i,
                position_m=staging_m.copy(),
                remaining_range_m=self.drone_range_m,
                mode=DroneMode.NORMAL,
                target_gs_idx=-1,
                visited_domains=set(),
            )
            for i in range(n_drones)
        ]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def select(
        self,
        info_field: InformationField,
        gp: "IGNISGPPrior",
        ensemble: EnsembleResult,
        k: int,
        *,
        terrain: TerrainData,
        staging_area: tuple[int, int],
        resolution_m: float = GRID_RESOLUTION_M,
        drone_states: Optional[list[DroneFlightState]] = None,
        ground_stations_m: Optional[list[np.ndarray]] = None,
        **_,
    ) -> PathSelectionResult:
        t0 = time.perf_counter()
        shape = terrain.shape
        d_max = self.drone_range_m
        d_safety = self.d_safety

        # Default GS = staging area
        staging_m = np.array(
            [staging_area[0] * resolution_m, staging_area[1] * resolution_m],
            dtype=np.float64,
        )
        if ground_stations_m is None:
            ground_stations_m = [staging_m]

        # 1. Terrain features (static — same every cycle)
        features = _terrain_features(terrain)

        # 2. Terrain-adaptive label map
        try:
            label_map = _felzenszwalb_label_map(
                terrain, features, self.correlation_length_m, resolution_m,
                self.min_domain_cells,
            )
            if _is_pathological(label_map):
                raise ValueError("Felzenszwalb produced degenerate segmentation")
            logger.debug("Felzenszwalb: %d domains", int(label_map.max()) + 1)
        except Exception as exc:
            logger.warning("Felzenszwalb failed (%s); falling back to regular grid", exc)
            label_map = _regular_grid_label_map(
                shape, self.correlation_length_m, resolution_m
            )

        # 3. Correlation graph
        graph = _build_correlation_graph(
            label_map=label_map,
            features=features,
            w_field=info_field.w,
            w_by_variable=info_field.w_by_variable,
            resolution_m=resolution_m,
        )

        # 4. GS distance tables (one Dijkstra per GS)
        gs_distances = _compute_all_gs_distances(graph, ground_stations_m, resolution_m)
        min_return_costs = _min_gs_return_costs(gs_distances)

        # 5. Initial per-domain w values and GP variance (shared across drone plans)
        base_domain_w: dict[int, float] = {d.domain_id: d.info_value for d in graph.domains}
        gp_var = info_field.gp_variance.get("fmc")
        if gp_var is None:
            gp_var = next(iter(info_field.gp_variance.values()))
        current_var = gp_var.astype(np.float64)

        # 6. Initialise drone states on first call
        if drone_states is None:
            drone_states = self._make_initial_states(k, staging_m)
        while len(drone_states) < k:
            drone_states.append(
                DroneFlightState(
                    drone_id=len(drone_states),
                    position_m=staging_m.copy(),
                    remaining_range_m=d_max,
                    mode=DroneMode.NORMAL,
                    target_gs_idx=-1,
                    visited_domains=set(),
                )
            )

        # Shared current_w — zeroed for domains already visited by earlier drones
        # (cross-drone deconfliction via shared GP variance update)
        current_w: dict[int, float] = dict(base_domain_w)

        # 7. Plan each drone
        drone_plans: list[DronePlan] = []
        marginal_gains: list[float] = []
        total_info = 0.0
        updated_states: list[DroneFlightState] = []

        for state in drone_states[:k]:
            # a. Mode transition check
            state = _check_mode_transitions(
                state, graph, ground_stations_m, gs_distances,
                resolution_m, d_max, d_safety, self.R_threshold,
            )

            # b. Re-admit visited domains where w > revisit threshold
            state.visited_domains = _apply_revisit_threshold(
                state.visited_domains, current_w, self.revisit_percentile
            )

            start_domain = _pos_to_domain(state.position_m, graph, resolution_m)
            plan_distance = 0.0
            landed = False

            # c. Plan according to mode
            if state.mode == DroneMode.EMERGENCY:
                # Direct shortest path to target GS
                gs_domain = _pos_to_domain(
                    ground_stations_m[state.target_gs_idx], graph, resolution_m
                )
                plan_distance, domain_ids = _dijkstra_path(graph, start_domain, gs_domain)
                landed = True
                logger.info("Drone %d: EMERGENCY → GS%d (%.0f m)",
                            state.drone_id, state.target_gs_idx, plan_distance)

            elif state.mode == DroneMode.RETURN:
                # Greedy with reachability invariant using target-GS return costs
                tgt_return_costs = gs_distances[state.target_gs_idx]
                budget = min(self.d_cycle_m, state.remaining_range_m - d_safety)
                budget = max(budget, 0.0)

                domain_ids, plan_distance = _plan_greedy_path(
                    graph=graph,
                    start_domain=start_domain,
                    budget_m=budget,
                    return_costs=tgt_return_costs,
                    current_w=current_w,
                    current_var=current_var,
                    gp=gp,
                    visited_global=state.visited_domains,
                )

                # Detect final return cycle: endpoint within d_safety of GS
                gs_domain = _pos_to_domain(
                    ground_stations_m[state.target_gs_idx], graph, resolution_m
                )
                end_to_gs = tgt_return_costs.get(domain_ids[-1], np.inf)
                if end_to_gs <= d_safety:
                    # Append GS domain to path so plan ends at station
                    _, path_to_gs = _dijkstra_path(graph, domain_ids[-1], gs_domain)
                    for nd in path_to_gs[1:]:
                        domain_ids.append(nd)
                    plan_distance = tgt_return_costs.get(start_domain, 0.0) - tgt_return_costs.get(domain_ids[-1], 0.0)
                    # Approximate: use budget used
                    plan_distance = min(state.remaining_range_m - d_safety, plan_distance)
                    landed = True
                    logger.info("Drone %d: RETURN final cycle → GS%d",
                                state.drone_id, state.target_gs_idx)

                # Loiter detection: if plan is trivially short, log
                if len(domain_ids) <= 1:
                    logger.info("Drone %d: RETURN loitering (budget=%.0f m)", state.drone_id, budget)

            else:  # NORMAL
                budget = self.d_cycle_m
                domain_ids, plan_distance = _plan_greedy_path(
                    graph=graph,
                    start_domain=start_domain,
                    budget_m=budget,
                    return_costs=min_return_costs,
                    current_w=current_w,
                    current_var=current_var,
                    gp=gp,
                    visited_global=state.visited_domains,
                )

                # Loiter detection
                if len(domain_ids) <= 1:
                    logger.info("Drone %d: NORMAL loitering (no high-value domains reachable)",
                                state.drone_id)

            # d. Build DronePlan
            plan = _domains_to_drone_plan(
                drone_id=state.drone_id,
                domain_ids=domain_ids,
                graph=graph,
                start_pos_m=state.position_m,
                shape=shape,
                resolution_m=resolution_m,
                camera_footprint_cells=self.camera_footprint_cells,
                plan_distance_m=plan_distance,
                drone_mode=state.mode.value,
            )
            drone_plans.append(plan)

            # e. Info gain accounting
            drone_info = sum(
                float(info_field.w[r, c])
                for r, c in plan.waypoints[1:]
            ) if len(plan.waypoints) > 1 else 0.0
            marginal_gains.append(drone_info)
            total_info += drone_info

            # f. Update drone state
            new_pos = state.position_m.copy()
            if len(domain_ids) > 1:
                end_domain = graph.domains[
                    next(i for i, d in enumerate(graph.domains)
                         if d.domain_id == domain_ids[-1])
                ]
                new_pos = end_domain.centroid_m.copy()

            new_state = DroneFlightState(
                drone_id=state.drone_id,
                position_m=new_pos,
                remaining_range_m=max(0.0, state.remaining_range_m - plan_distance),
                mode=state.mode,
                target_gs_idx=state.target_gs_idx,
                visited_domains=state.visited_domains | set(domain_ids),
                sortie_distance_m=state.sortie_distance_m + plan_distance,
                returned=landed,
            )
            updated_states.append(new_state)

            logger.debug(
                "Drone %d | mode=%s | domains=%d | dist=%.0f m | r=%.0f m | landed=%s",
                state.drone_id, new_state.mode.value, len(domain_ids),
                plan_distance, new_state.remaining_range_m, landed,
            )

        return PathSelectionResult(
            kind="paths",
            drone_plans=drone_plans,
            strategy_name=self.name,
            compute_time_s=time.perf_counter() - t0,
            total_info=total_info,
            marginal_gains=marginal_gains,
            updated_drone_states=updated_states,
        )
