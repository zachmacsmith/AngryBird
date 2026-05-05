"""
Correlation-Domain Path Selector with mode-based continuous flight planning.

Implements docs/Path Optimization.md (Part 1) and docs/Drone Path.md.

Domains are terrain-adaptive (Felzenszwalb on static terrain features), so
boundaries follow ridgelines, fuel transitions, and aspect breaks.

Three operating modes per drone (all transitions are one-way within a sortie):

  NORMAL    — budget = 1.1 × d_cycle, free exploration from current position,
               no endpoint constraint. Greedy stops when budget exhausted.
  RETURN    — budget = min(d_cycle, r - d_safety), battery feasibility enforced,
               target GS locked at NORMAL→RETURN transition. Falls back to
               Dijkstra toward target GS if greedy finds no candidates.
  EMERGENCY — direct shortest path to target GS, no optimisation.

Drone states persist across cycles (same sortie). On GS landing the
orchestrator resets battery and mode for the next sortie.

Visited-domain exclusion is local to each planning call (prevents within-path
cycles). Cross-cycle exclusion is unnecessary: the info field's GP variance
already encodes observation history — w_i falls at observed locations and rises
again only when genuine new uncertainty develops there.
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
    SelectionResult,
    TerrainData,
)
from .greedy import GreedySelector

if TYPE_CHECKING:
    from ..gp import IGNISGPPrior

logger = logging.getLogger(__name__)

# 10 % budget buffer: covers comms delay, wind drift, and autopilot lag
# between receiving the new plan and executing the first waypoint.
BUDGET_BUFFER = 1.10

# SB40 non-burnable fuel-model codes (91–99 = urban/water/ag/snow/bare)
# and 0 (no-data / open ocean).  Domains whose burnable fraction falls
# below NB_BURNABLE_THRESHOLD are excluded from the greedy search.
_NB_FUEL_CODES: frozenset[int] = frozenset(range(91, 100)) | {0}
NB_BURNABLE_THRESHOLD: float = 0.10   # 10 % burnable → treat as non-burnable


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
    burnable_fraction: float = 1.0  # fraction of cells with burnable SB40 fuel code


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
    fuel_model: Optional[np.ndarray] = None,   # SB40 fuel codes; None → all burnable
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

        # Burnable fraction: fraction of cells with SB40 codes NOT in NB set
        if fuel_model is not None:
            codes = fuel_model[rs, cs]
            nb_count = int(np.isin(codes, list(_NB_FUEL_CODES)).sum())
            burn_frac = 1.0 - nb_count / max(len(cells), 1)
        else:
            burn_frac = 1.0

        domain_dict[int(d_id)] = CorrelationDomain(
            domain_id=int(d_id),
            cells=cells,
            representative_cell=rep,
            centroid_m=centroid_m,
            info_value=float(w_vals.max()),
            dominant_variable=dom_var,
            area_cells=len(cells),
            burnable_fraction=burn_frac,
        )

    # 4-connected adjacency scan — also collects one boundary-cell pair per
    # domain pair (used later for representative-cell-to-representative-cell
    # distance, which is far more accurate than centroid-to-centroid for
    # irregular terrain-adaptive domains).
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
        # Use rep-cell-to-rep-cell distance: this is the actual flight distance
        # the drone covers when hopping between representative (highest-w) cells,
        # much more accurate than centroid-to-centroid for irregular domains.
        ri, ci = di.representative_cell
        rj, cj = dj.representative_cell
        real_dist = float(np.sqrt((ri - rj) ** 2 + (ci - cj) ** 2) * resolution_m)
        feat_diff = features[ri, ci] - features[rj, cj]
        dissimilarity = float(np.sqrt((feat_diff ** 2).sum()))
        cross_corr = float(np.exp(-dissimilarity))
        i_val = di.info_value if np.isfinite(di.info_value) else 0.0
        j_val = dj.info_value if np.isfinite(dj.info_value) else 0.0
        edge_info = (1.0 - cross_corr) * min(i_val, j_val)
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
    Apply NORMAL→RETURN and RETURN→EMERGENCY transitions.
    Returns a shallow copy of state with updated mode/target_gs_idx.
    """
    state = copy.copy(state)

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
    current_w: dict[int, float],         # mutable — visited domains zeroed on selection
    current_var: np.ndarray,             # GP variance, updated in-place
    gp: "IGNISGPPrior",
    return_costs: Optional[dict[int, float]] = None,   # RETURN mode: dist to target GS
    remaining_range_m: Optional[float] = None,         # RETURN mode: current battery
    d_safety: float = 0.0,                             # RETURN mode: safety margin
    gp_update_budget_m: Optional[float] = None,        # distance up to which the expensive
                                                       # GP conditional-variance update runs;
                                                       # hops beyond this get cheap zeroing only
    blocked_domains: Optional[set[int]] = None,        # domains permanently excluded from
                                                       # selection (e.g. ocean / non-burnable)
) -> tuple[list[int], float]:
    """
    Greedy orienteering from start_domain up to budget_m metres.

    visited is a LOCAL set, scoped to this call — it only prevents the path from
    revisiting a domain it has already queued in this cycle.  Cross-cycle exclusion
    is handled by the info field: after assimilation, GP variance (and therefore
    w_i) drops at observed locations, so the greedy naturally avoids them without
    any persistent state.

    NORMAL  — budget_m = 1.1 × d_cycle × horizon_cycles.  Only the first-cycle
              portion is sent to the autopilot; the rest is used for deconfliction
              (claiming territory) via current_w zeroing.  To keep selection fast,
              the expensive gp.conditional_variance update is skipped after
              gp_update_budget_m metres (default: the first-cycle portion only).

    RETURN  — budget_m = min(d_cycle, r - d_safety).
              Battery feasibility: traveled + d + dist_to_gs(nxt) + d_safety ≤ r.
              Caller falls back to Dijkstra toward GS if this returns a trivial path.

    Returns (domain_id_path, total_distance_m).
    """
    if gp_update_budget_m is None:
        gp_update_budget_m = budget_m   # default: run GP updates for the full path

    path = [start_domain]
    visited = {start_domain}    # local only — discarded when function returns
    remaining = budget_m
    traveled = 0.0
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
            if blocked_domains and nxt in blocked_domains:
                continue
            w = current_w.get(nxt, 0.0)
            if np.isnan(w):
                w = 0.0
            d = edge.real_distance_m

            # Cycle budget: can we afford this hop?
            if d > remaining:
                continue

            # RETURN mode: battery feasibility — from nxt we must be able to
            # reach target GS with remaining battery minus safety margin.
            if return_costs is not None and remaining_range_m is not None:
                ret = return_costs.get(nxt, np.inf)
                if traveled + d + ret + d_safety > remaining_range_m:
                    continue

            score = (w + edge.information_gain) / (d + 1.0)
            if score > best_score:
                best_score = score
                best_domain = nxt
                best_travel = d

        if best_domain is None:
            break

        # Zero selected domain immediately so later drones skip it.
        current_w[best_domain] = 0.0

        # GP conditional-variance update: only within the first-cycle portion of
        # the path (gp_update_budget_m).  For the horizon-extension portion (beyond
        # the first cycle) we skip this expensive call — domain zeroing above is
        # sufficient for deconfliction, and the extra hops don't generate real
        # observations this cycle anyway.
        if traveled < gp_update_budget_m:
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
        traveled  += best_travel
        total_dist += best_travel
        current = best_domain

    return path, total_dist


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
        min_useful_info: float = DRONE_MIN_USEFUL_INFO,
        horizon_cycles: float = 1.0,
    ) -> None:
        self.correlation_length_m   = correlation_length_m
        self.drone_range_m          = drone_range_m
        self.camera_footprint_cells = camera_footprint_cells
        self.min_domain_cells       = min_domain_cells
        self.d_cycle_m              = d_cycle_m
        self.R_threshold            = R_threshold
        self.d_safety               = safety_fraction * drone_range_m
        self.min_useful_info        = min_useful_info
        self.horizon_cycles         = max(1.0, float(horizon_cycles))

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
        n_drones: Optional[int] = None,
        **_,
    ) -> SelectionResult:
        # n_drones controls how many physical drones are planned for.
        # k is n_targets for point selectors; for path selectors only drone
        # count matters.  Fall back to k for backward compatibility.
        n_fleet = n_drones if n_drones is not None else k
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
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore")
            label_map = _felzenszwalb_label_map(
                terrain, features, self.correlation_length_m, resolution_m,
                self.min_domain_cells,
            )
        logger.info("Felzenszwalb: %d domains", int(label_map.max()) + 1)

        # 3. Correlation graph (pass fuel_model so burnable fraction is computed)
        graph = _build_correlation_graph(
            label_map=label_map,
            features=features,
            w_field=info_field.w,
            w_by_variable=info_field.w_by_variable,
            resolution_m=resolution_m,
            fuel_model=terrain.fuel_model,
        )

        # 4. GS distance tables (one Dijkstra per GS)
        gs_distances = _compute_all_gs_distances(graph, ground_stations_m, resolution_m)

        # 4b. Non-burnable domain exclusion — domains with < NB_BURNABLE_THRESHOLD
        #     burnable fraction are blocked from greedy selection.  We log the
        #     count so callers can spot terrain/resolution mismatches.
        blocked_domains: set[int] = {
            d.domain_id
            for d in graph.domains
            if d.burnable_fraction < NB_BURNABLE_THRESHOLD
        }
        if blocked_domains:
            logger.debug(
                "Blocking %d/%d non-burnable domains (threshold=%.0f%%)",
                len(blocked_domains), len(graph.domains),
                NB_BURNABLE_THRESHOLD * 100,
            )

        # 5. Per-domain w values and GP variance (shared across drone plans for deconfliction)
        gp_var = info_field.gp_variance.get("fmc")
        if gp_var is None:
            gp_var = next(iter(info_field.gp_variance.values()))
        current_var = gp_var.astype(np.float64)

        raw_w = {d.domain_id: d.info_value for d in graph.domains}
        total_w = sum(v for v in raw_w.values() if np.isfinite(v) and v > 0)

        if total_w <= 0:
            # Info field is degenerate (fire hasn't spread yet, sensitivity=0).
            # Fall back to mean GP variance per domain so drones explore where
            # measurement uncertainty is highest rather than loitering.
            logger.debug("Info field degenerate (w≈0); using GP variance as domain score")
            rep_by_id = {d.domain_id: d.representative_cell for d in graph.domains}
            base_domain_w = {
                d.domain_id: float(np.mean(gp_var[
                    [c[0] for c in d.cells],
                    [c[1] for c in d.cells],
                ]))
                for d in graph.domains
            }
        else:
            base_domain_w = raw_w

        # 6. Initialise drone states on first call
        if drone_states is None:
            drone_states = self._make_initial_states(n_fleet, staging_m)
        while len(drone_states) < n_fleet:
            drone_states.append(
                DroneFlightState(
                    drone_id=len(drone_states),
                    position_m=staging_m.copy(),
                    remaining_range_m=d_max,
                    mode=DroneMode.NORMAL,
                    target_gs_idx=-1,
                )
            )

        # Shared current_w: domains visited by earlier drones are zeroed via GP
        # variance updates, ensuring cross-drone deconfliction without persistent state.
        current_w: dict[int, float] = dict(base_domain_w)

        # 7. Plan each drone
        drone_plans: list[DronePlan] = []
        marginal_gains: list[float] = []
        total_info = 0.0
        updated_states: list[DroneFlightState] = []

        # Greedy fallback state — computed lazily on the first NORMAL drone
        # that triggers the low-info threshold.  One globally-optimal cell is
        # assigned per falling-back drone in sequence.
        _greedy_fallback_locs: Optional[list[tuple[int, int]]] = None
        _greedy_fallback_idx: int = 0

        for state in drone_states[:n_fleet]:
            used_greedy_search = False

            # a. Mode transition check
            state = _check_mode_transitions(
                state, graph, ground_stations_m, gs_distances,
                resolution_m, d_max, d_safety, self.R_threshold,
            )

            start_domain = _pos_to_domain(state.position_m, graph, resolution_m)
            plan_distance = 0.0
            landed = False

            # b. Plan according to mode
            if state.mode == DroneMode.EMERGENCY:
                gs_domain = _pos_to_domain(
                    ground_stations_m[state.target_gs_idx], graph, resolution_m
                )
                plan_distance, domain_ids = _dijkstra_path(graph, start_domain, gs_domain)
                landed = True
                logger.info("Drone %d: EMERGENCY → GS%d (%.0f m)",
                            state.drone_id, state.target_gs_idx, plan_distance)

            elif state.mode == DroneMode.RETURN:
                tgt_return_costs = gs_distances[state.target_gs_idx]
                budget = min(self.d_cycle_m, state.remaining_range_m - d_safety)
                budget = max(budget, 0.0)

                domain_ids, plan_distance = _plan_greedy_path(
                    graph=graph,
                    start_domain=start_domain,
                    budget_m=budget,
                    current_w=current_w,
                    current_var=current_var,
                    gp=gp,
                    return_costs=tgt_return_costs,
                    remaining_range_m=state.remaining_range_m,
                    d_safety=d_safety,
                    blocked_domains=blocked_domains,
                )

                # Fallback: if the greedy found no candidates (neighborhood
                # genuinely empty — zero w, not a visited-set issue), walk the
                # Dijkstra shortest path toward the target GS, collecting whatever
                # info lies along the way, subject to budget + battery constraints.
                if len(domain_ids) <= 1:
                    gs_domain = _pos_to_domain(
                        ground_stations_m[state.target_gs_idx], graph, resolution_m
                    )
                    _, dijk_path = _dijkstra_path(graph, start_domain, gs_domain)
                    traveled_fb = 0.0
                    domain_ids = [start_domain]
                    for nxt in dijk_path[1:]:
                        edge = graph.edge(domain_ids[-1], nxt)
                        if edge is None:
                            break
                        d = edge.real_distance_m
                        if traveled_fb + d > budget:
                            break
                        if traveled_fb + d + tgt_return_costs.get(nxt, np.inf) + d_safety > state.remaining_range_m:
                            break
                        domain_ids.append(nxt)
                        current_w[nxt] = 0.0
                        traveled_fb += d
                    plan_distance = traveled_fb
                    logger.info("Drone %d: RETURN fallback → Dijkstra toward GS%d (%.0f m, %d hops)",
                                state.drone_id, state.target_gs_idx, plan_distance, len(domain_ids) - 1)

                if len(domain_ids) <= 1:
                    logger.info("Drone %d: RETURN loitering (budget=%.0f m)", state.drone_id, budget)

                # Detect final return: endpoint within d_safety of GS
                end_to_gs = tgt_return_costs.get(domain_ids[-1], np.inf)
                if end_to_gs <= d_safety:
                    gs_domain = _pos_to_domain(
                        ground_stations_m[state.target_gs_idx], graph, resolution_m
                    )
                    _, path_to_gs = _dijkstra_path(graph, domain_ids[-1], gs_domain)
                    for nd in path_to_gs[1:]:
                        domain_ids.append(nd)
                    landed = True
                    logger.info("Drone %d: RETURN final cycle → GS%d",
                                state.drone_id, state.target_gs_idx)

            else:  # NORMAL
                # Plan over horizon_cycles worth of distance.  Deconfliction
                # zeroes ALL domains in the long path, forcing each subsequent
                # drone to find a completely different trajectory — this is
                # what spreads the fleet across the map.  Only the first
                # d_cycle_m worth of waypoints is sent to the autopilot for
                # this cycle; the drone's state end-position is updated to the
                # first-cycle boundary so replanning starts from there.
                execute_budget = self.d_cycle_m * BUDGET_BUFFER
                plan_budget = execute_budget * self.horizon_cycles
                domain_ids_full, _ = _plan_greedy_path(
                    graph=graph,
                    start_domain=start_domain,
                    budget_m=plan_budget,
                    current_w=current_w,
                    current_var=current_var,
                    gp=gp,
                    # GP variance update only for the first-cycle portion; horizon
                    # extension uses cheap domain zeroing only (no GP calls).
                    gp_update_budget_m=execute_budget,
                    # No return_costs / remaining_range_m: NORMAL has no endpoint constraint.
                    blocked_domains=blocked_domains,
                )

                # Truncate to first-cycle distance for the actual waypoints /
                # observation footprint this cycle.  The domain_ids beyond the
                # truncation already had their current_w zeroed inside
                # _plan_greedy_path, so deconfliction covers the full horizon.
                domain_ids = [domain_ids_full[0]]
                cum_dist    = 0.0
                for i in range(1, len(domain_ids_full)):
                    edge = graph.edge(domain_ids_full[i - 1], domain_ids_full[i])
                    if edge is None:
                        break
                    if cum_dist + edge.real_distance_m > execute_budget:
                        break
                    domain_ids.append(domain_ids_full[i])
                    cum_dist += edge.real_distance_m
                plan_distance = cum_dist

                # Low-info fallback: if the first-cycle segment yields insufficient
                # information, the drone is likely far from the fire.  Call the
                # global GreedySelector to find the best cell on the entire map,
                # then Dijkstra-route toward it within the cycle budget.  This
                # makes drones track toward the fire rather than wander locally;
                # once they arrive and information is high, planning reverts to the
                # correlation path approach automatically.
                cycle_info = sum(base_domain_w.get(did, 0.0) for did in domain_ids[1:])
                if cycle_info < self.min_useful_info:
                    # Compute global greedy targets on first trigger (shared across
                    # all drones this cycle so each gets a distinct assigned cell).
                    if _greedy_fallback_locs is None:
                        _gs = GreedySelector(resolution_m=resolution_m)
                        _gr = _gs.select(info_field, gp, ensemble, k=n_fleet)
                        _greedy_fallback_locs = _gr.selected_locations

                    if _greedy_fallback_idx < len(_greedy_fallback_locs):
                        target_cell = _greedy_fallback_locs[_greedy_fallback_idx]
                        _greedy_fallback_idx += 1
                        target_domain = graph.domain_id_for_cell(target_cell)
                        _, dijk_path = _dijkstra_path(graph, start_domain, target_domain)
                        domain_ids = [start_domain]
                        cum_dist = 0.0
                        for j in range(1, len(dijk_path)):
                            edge = graph.edge(dijk_path[j - 1], dijk_path[j])
                            if edge is None:
                                break
                            if cum_dist + edge.real_distance_m > execute_budget:
                                break
                            domain_ids.append(dijk_path[j])
                            current_w[dijk_path[j]] = 0.0
                            cum_dist += edge.real_distance_m
                        plan_distance = cum_dist
                        used_greedy_search = True
                        logger.info(
                            "Drone %d: low cycle info (%.4f < %.4f) → greedy target %s",
                            state.drone_id, cycle_info, self.min_useful_info, target_cell,
                        )

                if len(domain_ids) <= 1:
                    logger.info("Drone %d: NORMAL loitering (no high-value domains reachable)",
                                state.drone_id)

            # c. Build DronePlan
            plan = _domains_to_drone_plan(
                drone_id=state.drone_id,
                domain_ids=domain_ids,
                graph=graph,
                start_pos_m=state.position_m,
                shape=shape,
                resolution_m=resolution_m,
                camera_footprint_cells=self.camera_footprint_cells,
                plan_distance_m=plan_distance,
                drone_mode="TRACKING" if used_greedy_search else state.mode.value,
            )
            drone_plans.append(plan)

            # d. Info gain accounting
            drone_info = sum(
                float(info_field.w[r, c])
                for r, c in plan.waypoints[1:]
            ) if len(plan.waypoints) > 1 else 0.0
            marginal_gains.append(drone_info)
            total_info += drone_info

            # e. Update drone state
            new_pos = state.position_m.copy()
            if len(domain_ids) > 1:
                end_domain = next(
                    d for d in graph.domains if d.domain_id == domain_ids[-1]
                )
                new_pos = end_domain.centroid_m.copy()

            new_state = DroneFlightState(
                drone_id=state.drone_id,
                position_m=new_pos,
                remaining_range_m=max(0.0, state.remaining_range_m - plan_distance),
                mode=state.mode,
                target_gs_idx=state.target_gs_idx,
                sortie_distance_m=state.sortie_distance_m + plan_distance,
                returned=landed,
            )
            updated_states.append(new_state)

            logger.debug(
                "Drone %d | mode=%s | domains=%d | dist=%.0f m | r=%.0f m | landed=%s",
                state.drone_id, new_state.mode.value, len(domain_ids),
                plan_distance, new_state.remaining_range_m, landed,
            )

        return SelectionResult(
            kind="paths",
            drone_plans=drone_plans,
            strategy_name=self.name,
            compute_time_s=time.perf_counter() - t0,
            total_info=total_info,
            marginal_gains=marginal_gains,
            updated_drone_states=updated_states,
        )
