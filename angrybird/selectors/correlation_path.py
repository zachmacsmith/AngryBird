"""
Correlation-Domain Path Selector (Part 1 of docs/Path Optimization.md).

Instead of selecting K points then routing, this selector:
  1. Partitions the grid into correlation-length-sized domains
  2. Builds a domain adjacency graph with GP-derived cross-correlations
  3. Precomputes return costs (Dijkstra from staging area)
  4. Plans one greedy path per drone with exact GP variance updates
  5. Returns PathSelectionResult with fully-formed DronePlans

No point→path step is needed — the orchestrator uses the plans directly.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..config import (
    CAMERA_FOOTPRINT_CELLS,
    DRONE_RANGE_M,
    DRONE_SPEED_MS,
    GP_CORRELATION_LENGTH_FMC_M,
    GRID_RESOLUTION_M,
)
from ..path_planner import cells_along_path
from ..types import DronePlan, EnsembleResult, InformationField, PathSelectionResult, TerrainData

if TYPE_CHECKING:
    from ..gp import IGNISGPPrior


# ---------------------------------------------------------------------------
# Domain graph data structures
# ---------------------------------------------------------------------------

@dataclass
class CorrelationDomain:
    domain_id: int
    cells: list[tuple[int, int]]
    representative_cell: tuple[int, int]
    centroid_m: np.ndarray          # (y_m, x_m) — row × resolution, col × resolution
    info_value: float
    dominant_variable: str


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
    adj: dict[int, list[DomainEdge]]     # source_id → outbound edges
    domain_size: int                      # cells per domain side
    n_dc: int                             # number of domain columns (for cell→domain mapping)

    def domain_id_for_cell(self, cell: tuple[int, int]) -> int:
        r, c = cell
        return (r // self.domain_size) * self.n_dc + (c // self.domain_size)

    def edge(self, src: int, tgt: int) -> Optional[DomainEdge]:
        for e in self.adj.get(src, []):
            if e.target == tgt:
                return e
        return None


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_correlation_graph(
    w_field: np.ndarray,
    w_by_variable: dict[str, np.ndarray],
    gp: "IGNISGPPrior",
    terrain: TerrainData,
    resolution_m: float,
    correlation_length_m: float,
) -> CorrelationGraph:
    """Partition grid into domains ~1 correlation length in diameter and build adjacency graph."""
    rows, cols = w_field.shape
    domain_size = max(1, int(round(correlation_length_m / resolution_m)))

    n_dr = (rows + domain_size - 1) // domain_size
    n_dc = (cols + domain_size - 1) // domain_size

    domains: list[CorrelationDomain] = []
    for dr in range(n_dr):
        for dc in range(n_dc):
            r0 = dr * domain_size
            r1 = min(r0 + domain_size, rows)
            c0 = dc * domain_size
            c1 = min(c0 + domain_size, cols)

            cells = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
            w_vals = [float(w_field[r, c]) for r, c in cells]
            best_idx = int(np.argmax(w_vals))
            rep = cells[best_idx]

            centroid_r = (r0 + r1) / 2.0 * resolution_m
            centroid_c = (c0 + c1) / 2.0 * resolution_m

            dom_var = max(
                w_by_variable,
                key=lambda v: float(w_by_variable[v][rep[0], rep[1]]),
            )

            domains.append(CorrelationDomain(
                domain_id=len(domains),
                cells=cells,
                representative_cell=rep,
                centroid_m=np.array([centroid_r, centroid_c], dtype=np.float64),
                info_value=w_vals[best_idx],
                dominant_variable=dom_var,
            ))

    # Build adjacency (domains within 1.5× domain footprint of each other)
    max_adj_dist = 1.5 * domain_size * resolution_m
    adj: dict[int, list[DomainEdge]] = {d.domain_id: [] for d in domains}

    for i, di in enumerate(domains):
        for j, dj in enumerate(domains):
            if j <= i:
                continue
            dist = float(np.linalg.norm(di.centroid_m - dj.centroid_m))
            if dist > max_adj_dist:
                continue

            # Cross-correlation: approximate as exp(-dist / length_scale)
            cross_corr = float(np.exp(-dist / (correlation_length_m + 1e-6)))

            # Edge info: crossing to a dissimilar domain reveals new information
            edge_info = (1.0 - cross_corr) * min(di.info_value, dj.info_value)

            e_fwd = DomainEdge(
                source=i, target=j,
                cross_correlation=cross_corr,
                information_gain=edge_info,
                real_distance_m=dist,
            )
            e_rev = DomainEdge(
                source=j, target=i,
                cross_correlation=cross_corr,
                information_gain=edge_info,
                real_distance_m=dist,
            )
            adj[i].append(e_fwd)
            adj[j].append(e_rev)

    return CorrelationGraph(domains=domains, adj=adj, domain_size=domain_size, n_dc=n_dc)


# ---------------------------------------------------------------------------
# Dijkstra return costs from staging area
# ---------------------------------------------------------------------------

def _precompute_return_costs(
    graph: CorrelationGraph,
    staging_area: tuple[int, int],
    resolution_m: float,
) -> np.ndarray:
    """
    Dijkstra from the staging-area domain outward.
    Returns float array[n_domains]: min flight distance back to staging.
    """
    n = len(graph.domains)
    dist = np.full(n, np.inf)
    start = graph.domain_id_for_cell(staging_area)
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


# ---------------------------------------------------------------------------
# Single-drone path planning
# ---------------------------------------------------------------------------

def _plan_drone_path(
    graph: CorrelationGraph,
    staging_domain: int,
    max_range_m: float,
    return_costs: np.ndarray,
    current_w: np.ndarray,    # float[n_domains], mutable working copy
    current_var: np.ndarray,  # float[rows, cols], GP variance — updated in-place externally
    orig_var: np.ndarray,     # float[rows, cols], original variance (denominator for ratio)
    gp: "IGNISGPPrior",
    resolution_m: float,
) -> list[int]:
    """
    Greedy path through domain graph from staging_domain.
    Returns list of domain IDs (including staging start; no return appended).
    """
    path_domains = [staging_domain]
    visited = {staging_domain}
    remaining_range = max_range_m
    current_domain = staging_domain

    while True:
        best_domain: Optional[int] = None
        best_score = -np.inf
        best_travel = 0.0
        best_info = 0.0

        for edge in graph.adj.get(current_domain, []):
            nxt = edge.target
            if nxt in visited:
                continue
            travel_cost = edge.real_distance_m
            return_cost = return_costs[nxt]
            if travel_cost + return_cost > remaining_range:
                continue

            w_domain = current_w[nxt]
            total_info = w_domain + edge.information_gain
            efficiency = total_info / (travel_cost + 1.0)

            if efficiency > best_score:
                best_score = efficiency
                best_domain = nxt
                best_travel = travel_cost
                best_info = total_info

        if best_domain is None:
            break

        # Visit best domain: update GP variance and discount all domain w values
        rep = graph.domains[best_domain].representative_cell
        updated_var = gp.conditional_variance(current_var, rep)

        # Scale current_w by the variance reduction ratio
        for d in graph.domains:
            if d.domain_id in visited:
                continue
            r, c = d.representative_cell
            old_v = float(orig_var[r, c])
            if old_v > 1e-12:
                ratio = float(updated_var[r, c]) / old_v
                current_w[d.domain_id] *= ratio

        np.copyto(current_var, updated_var)

        path_domains.append(best_domain)
        visited.add(best_domain)
        remaining_range -= best_travel
        current_domain = best_domain

    return path_domains


# ---------------------------------------------------------------------------
# Convert domain path → DronePlan
# ---------------------------------------------------------------------------

def _domains_to_drone_plan(
    drone_id: int,
    domain_ids: list[int],
    graph: CorrelationGraph,
    staging_area: tuple[int, int],
    shape: tuple[int, int],
    camera_footprint_cells: int,
) -> DronePlan:
    """Build DronePlan from a list of visited domain IDs."""
    if len(domain_ids) <= 1:
        return DronePlan(drone_id=drone_id, waypoints=[], cells_observed=[])

    # Waypoints: staging → domain reps → staging
    target_wps = [
        graph.domains[did].representative_cell
        for did in domain_ids[1:]  # skip staging domain
    ]
    waypoints = [staging_area] + target_wps + [staging_area]
    observed = cells_along_path(waypoints, shape, camera_footprint_cells)

    return DronePlan(drone_id=drone_id, waypoints=waypoints, cells_observed=observed)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class CorrelationPathSelector:
    """
    Path selector implementing the correlation-domain graph algorithm.

    Plans drone paths directly from the information field — no separate
    point-selection step.  Uses GP conditional variance updates to ensure
    sequential drones cover non-redundant terrain.
    """

    name = "correlation_path"
    kind = "paths"

    def __init__(
        self,
        correlation_length_m: float = GP_CORRELATION_LENGTH_FMC_M,
        drone_range_m: float = DRONE_RANGE_M,
        camera_footprint_cells: int = CAMERA_FOOTPRINT_CELLS,
    ) -> None:
        self.correlation_length_m = correlation_length_m
        self.drone_range_m = drone_range_m
        self.camera_footprint_cells = camera_footprint_cells

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
        **_,
    ) -> PathSelectionResult:
        t0 = time.perf_counter()
        shape = terrain.shape

        # Build correlation-domain graph
        graph = _build_correlation_graph(
            w_field=info_field.w,
            w_by_variable=info_field.w_by_variable,
            gp=gp,
            terrain=terrain,
            resolution_m=resolution_m,
            correlation_length_m=self.correlation_length_m,
        )

        # Precompute return costs from staging
        return_costs = _precompute_return_costs(graph, staging_area, resolution_m)

        staging_domain = graph.domain_id_for_cell(staging_area)
        n_domains = len(graph.domains)

        # Initial per-domain w values and GP variance
        current_w = np.array([d.info_value for d in graph.domains], dtype=np.float64)
        gp_var = info_field.gp_variance.get("fmc")
        if gp_var is None:
            # Fall back to first available variable
            gp_var = next(iter(info_field.gp_variance.values()))
        orig_var = gp_var.astype(np.float64)
        current_var = orig_var.copy()

        # Plan fleet: each drone's path updates the shared variance
        drone_plans: list[DronePlan] = []
        marginal_gains: list[float] = []
        total_info = 0.0

        for drone_id in range(k):
            domain_ids = _plan_drone_path(
                graph=graph,
                staging_domain=staging_domain,
                max_range_m=self.drone_range_m,
                return_costs=return_costs,
                current_w=current_w,
                current_var=current_var,
                orig_var=orig_var,
                gp=gp,
                resolution_m=resolution_m,
            )

            plan = _domains_to_drone_plan(
                drone_id=drone_id,
                domain_ids=domain_ids,
                graph=graph,
                staging_area=staging_area,
                shape=shape,
                camera_footprint_cells=self.camera_footprint_cells,
            )
            drone_plans.append(plan)

            # Score this drone's contribution as sum of w over its target waypoints
            drone_info = sum(
                float(info_field.w[r, c])
                for r, c in plan.waypoints[1:-1]  # exclude staging endpoints
            ) if len(plan.waypoints) > 2 else 0.0
            marginal_gains.append(drone_info)
            total_info += drone_info

        return PathSelectionResult(
            kind="paths",
            drone_plans=drone_plans,
            strategy_name=self.name,
            compute_time_s=time.perf_counter() - t0,
            total_info=total_info,
            marginal_gains=marginal_gains,
        )
