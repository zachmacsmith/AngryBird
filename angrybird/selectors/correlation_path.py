"""
Correlation-Domain Path Selector (Part 1 of docs/Path Optimization.md).

Domains are computed via Felzenszwalb terrain-adaptive segmentation (see
docs/Terrain adaptive domains.md) so boundaries follow ridgelines, fuel type
transitions, and aspect breaks rather than an arbitrary regular grid.

Pipeline:
  1. build_terrain_domains()  — Felzenszwalb on multi-channel terrain features
  2. build_correlation_graph() — adjacency via 4-connected pixel scan, cross-
                                  correlation via feature dissimilarity
  3. _precompute_return_costs() — Dijkstra from staging-area domain
  4. _plan_fleet()             — sequential greedy paths with GP variance updates
  5. _domains_to_drone_plan()  — domain path → DronePlan (Bresenham + footprint)

Fallback to regular-grid partition if Felzenszwalb produces pathological results
(single giant domain or trivial fragmentation).
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..config import (
    CAMERA_FOOTPRINT_CELLS,
    DRONE_RANGE_M,
    GP_CORRELATION_LENGTH_FMC_M,
    GRID_RESOLUTION_M,
)
from ..path_planner import cells_along_path
from ..types import DronePlan, EnsembleResult, InformationField, PathSelectionResult, TerrainData

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
    centroid_m: np.ndarray          # (y_m, x_m) — row × resolution, col × resolution
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
# Terrain feature extraction (shared by domain building and cross-correlation)
# ---------------------------------------------------------------------------

def _terrain_features(terrain: TerrainData) -> np.ndarray:
    """
    Build normalised multi-channel feature array (rows, cols, 6).
    Channel weights reflect their importance as FMC correlation drivers.
    Aspect is encoded as (cos, sin) to handle circularity.
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
        norm(aspect_cos)                           * 1.5,  # primary FMC driver
        norm(aspect_sin)                           * 1.5,
        norm(terrain.fuel_model.astype(np.float64)) * 2.0,  # highest: discontinuous
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
    """Run Felzenszwalb and return a dense int[rows,cols] label map."""
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
    """Fallback: plain grid partition returns domain IDs as a label map."""
    rows, cols = shape
    domain_size = max(1, int(round(correlation_length_m / resolution_m)))
    n_dc = (cols + domain_size - 1) // domain_size
    labels = np.zeros((rows, cols), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            labels[r, c] = (r // domain_size) * n_dc + (c // domain_size)
    return labels


def _is_pathological(label_map: np.ndarray) -> bool:
    """Return True if Felzenszwalb produced degenerate segmentation."""
    sizes = np.bincount(label_map.ravel())
    n_cells = label_map.size
    # One domain covers > 50 % of the grid, or > 25 % of cells are singletons
    if sizes.max() > 0.5 * n_cells:
        return True
    if (sizes == 1).sum() > 0.25 * n_cells:
        return True
    return False


# ---------------------------------------------------------------------------
# Graph construction from label map
# ---------------------------------------------------------------------------

def _build_correlation_graph(
    label_map: np.ndarray,
    features: np.ndarray,
    w_field: np.ndarray,
    w_by_variable: dict[str, np.ndarray],
    resolution_m: float,
) -> CorrelationGraph:
    """Build CorrelationGraph from a dense label map and terrain features."""
    rows, cols = label_map.shape
    unique_ids = np.unique(label_map)

    # --- Build domain objects ---
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

    # --- Find adjacent domain pairs via 4-connected scan ---
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
    # Last column (vertical adjacency only)
    for r in range(rows - 1):
        here  = int(label_map[r,     cols - 1])
        below = int(label_map[r + 1, cols - 1])
        if here != below:
            adjacency.add((min(here, below), max(here, below)))
    # Last row (horizontal adjacency only)
    for c in range(cols - 1):
        here  = int(label_map[rows - 1, c])
        right = int(label_map[rows - 1, c + 1])
        if here != right:
            adjacency.add((min(here, right), max(here, right)))

    # --- Build edges ---
    adj: dict[int, list[DomainEdge]] = {int(did): [] for did in unique_ids}

    for d_i, d_j in adjacency:
        di = domain_dict[d_i]
        dj = domain_dict[d_j]

        real_dist = float(np.linalg.norm(di.centroid_m - dj.centroid_m))

        # Cross-correlation via feature dissimilarity at representative cells
        ri, ci = di.representative_cell
        rj, cj = dj.representative_cell
        feat_diff = features[ri, ci] - features[rj, cj]
        dissimilarity = float(np.sqrt((feat_diff ** 2).sum()))
        cross_corr = float(np.exp(-dissimilarity))

        edge_info = (1.0 - cross_corr) * min(di.info_value, dj.info_value)

        adj[d_i].append(DomainEdge(d_i, d_j, cross_corr, edge_info, real_dist))
        adj[d_j].append(DomainEdge(d_j, d_i, cross_corr, edge_info, real_dist))

    domains = list(domain_dict.values())
    return CorrelationGraph(domains=domains, adj=adj, label_map=label_map)


# ---------------------------------------------------------------------------
# Dijkstra return costs from staging-area domain
# ---------------------------------------------------------------------------

def _precompute_return_costs(
    graph: CorrelationGraph,
    staging_area: tuple[int, int],
) -> dict[int, float]:
    """
    Dijkstra from the staging-area domain outward.
    Returns {domain_id: min_flight_distance_back_to_staging}.
    """
    start = graph.domain_id_for_cell(staging_area)
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
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return dist


# ---------------------------------------------------------------------------
# Single-drone greedy path
# ---------------------------------------------------------------------------

def _plan_drone_path(
    graph: CorrelationGraph,
    staging_domain: int,
    max_range_m: float,
    return_costs: dict[int, float],
    current_w: dict[int, float],  # mutable; shared across drones — visited domains zeroed on exit
    current_var: np.ndarray,      # GP variance, updated in-place after each visit
    gp: "IGNISGPPrior",
) -> list[int]:
    """
    Greedy path through the domain graph from staging_domain.
    Returns visited domain IDs (staging first; no return node appended).

    Mutates current_w and current_var so subsequent drones see reduced information
    at already-covered terrain.

    Two invariants maintained:
      - current_w[d] tracks marginal information remaining at d, discounted by
        the variance reduction from all prior observations (ratio uses current_var
        as denominator so each visit contributes exactly once to the discount).
      - visited domains are zeroed in current_w so later drones skip them.
    """
    path_domains = [staging_domain]
    visited = {staging_domain}
    remaining_range = max_range_m
    current_domain = staging_domain

    # Precompute rep_cell lookup to avoid O(n) linear scan per step
    rep_by_id = {d.domain_id: d.representative_cell for d in graph.domains}

    while True:
        best_domain: Optional[int] = None
        best_score = -np.inf
        best_travel = 0.0

        for edge in graph.adj.get(current_domain, []):
            nxt = edge.target
            if nxt in visited:
                continue
            w_domain = current_w.get(nxt, 0.0)
            if w_domain <= 0.0:
                continue  # already visited by a prior drone; nothing left to gain
            return_cost = return_costs.get(nxt, np.inf)
            if edge.real_distance_m + return_cost > remaining_range:
                continue

            total_info = w_domain + edge.information_gain
            efficiency = total_info / (edge.real_distance_m + 1.0)

            if efficiency > best_score:
                best_score = efficiency
                best_domain = nxt
                best_travel = edge.real_distance_m

        if best_domain is None:
            break

        # Zero the selected domain immediately so later drones won't revisit it.
        # (Bug fix: the update loop runs before visited.add, so without this the
        # domain self-discounts to a small positive value rather than 0.)
        current_w[best_domain] = 0.0

        # Update GP variance after hypothetical observation at representative cell.
        rep_cell = rep_by_id[best_domain]
        updated_var = gp.conditional_variance(current_var, rep_cell)

        # Discount current_w for unvisited domains by the MARGINAL variance
        # reduction from this visit.  Use current_var (not orig_var) as the
        # denominator so each visit contributes exactly once — avoids compounding
        # discounts on domains near previously visited areas.
        for d in graph.domains:
            if d.domain_id in visited or d.domain_id == best_domain:
                continue
            r, c = rep_by_id[d.domain_id]
            old_v = float(current_var[r, c])
            if old_v > 1e-12:
                ratio = float(updated_var[r, c]) / old_v
                current_w[d.domain_id] = current_w.get(d.domain_id, 0.0) * ratio

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
    if len(domain_ids) <= 1:
        return DronePlan(drone_id=drone_id, waypoints=[], cells_observed=[])

    domain_by_id = {d.domain_id: d for d in graph.domains}
    target_wps = [
        domain_by_id[did].representative_cell
        for did in domain_ids[1:]    # skip staging domain
    ]
    waypoints = [staging_area] + target_wps + [staging_area]
    observed = cells_along_path(waypoints, shape, camera_footprint_cells)

    return DronePlan(drone_id=drone_id, waypoints=waypoints, cells_observed=observed)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class CorrelationPathSelector:
    """
    Path selector: terrain-adaptive Felzenszwalb domains + greedy GP-variance
    fleet planning.  Bypasses the point-selection step entirely.

    Domain segmentation follows ridgelines, fuel boundaries, and aspect breaks
    (see docs/Terrain adaptive domains.md).  Falls back to regular-grid domains
    if Felzenszwalb produces pathological output.
    """

    name = "correlation_path"
    kind = "paths"

    def __init__(
        self,
        correlation_length_m: float = GP_CORRELATION_LENGTH_FMC_M,
        drone_range_m: float = DRONE_RANGE_M,
        camera_footprint_cells: int = CAMERA_FOOTPRINT_CELLS,
        min_domain_cells: int = 10,
    ) -> None:
        self.correlation_length_m = correlation_length_m
        self.drone_range_m = drone_range_m
        self.camera_footprint_cells = camera_footprint_cells
        self.min_domain_cells = min_domain_cells

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

        # 1. Terrain features (shared by segmentation and cross-correlation)
        features = _terrain_features(terrain)

        # 2. Terrain-adaptive label map, with fallback to regular grid
        try:
            label_map = _felzenszwalb_label_map(
                terrain, features, self.correlation_length_m, resolution_m,
                self.min_domain_cells,
            )
            if _is_pathological(label_map):
                raise ValueError("Felzenszwalb produced degenerate segmentation")
            n_domains = int(label_map.max()) + 1
            logger.debug("Felzenszwalb: %d terrain-adaptive domains", n_domains)
        except Exception as exc:
            logger.warning(
                "Felzenszwalb failed (%s); falling back to regular grid", exc
            )
            label_map = _regular_grid_label_map(
                shape, self.correlation_length_m, resolution_m
            )

        # 3. Build correlation graph
        graph = _build_correlation_graph(
            label_map=label_map,
            features=features,
            w_field=info_field.w,
            w_by_variable=info_field.w_by_variable,
            resolution_m=resolution_m,
        )

        # 4. Return costs (Dijkstra from staging)
        return_costs = _precompute_return_costs(graph, staging_area)
        staging_domain = graph.domain_id_for_cell(staging_area)

        # 5. Initial per-domain w values and GP variance
        current_w: dict[int, float] = {d.domain_id: d.info_value for d in graph.domains}
        gp_var = info_field.gp_variance.get("fmc")
        if gp_var is None:
            gp_var = next(iter(info_field.gp_variance.values()))
        current_var = gp_var.astype(np.float64)

        # 6. Plan fleet: each drone updates the shared variance so subsequent
        #    drones cover non-redundant terrain
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
                gp=gp,
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

            drone_info = sum(
                float(info_field.w[r, c])
                for r, c in plan.waypoints[1:-1]
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
