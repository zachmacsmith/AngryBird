Path-Based Information Optimization & Scaling Analysis

---

## Part 1: Correlation-Domain Graph Path Optimization

### Concept

Replace point selection with path optimization over a reduced graph. Cells within one correlation length are grouped into domains — measuring one cell in a domain captures ~90% of the information the others would provide. The optimization operates on this coarser graph, selecting paths through domains that maximize non-redundant information per unit travel cost, subject to range and depot constraints.

### Step 1: Build Correlation-Domain Graph

Partition the grid into regions of approximately one correlation length diameter. Each region becomes a node. Edges connect adjacent regions.

```python
@dataclass
class CorrelationDomain:
    domain_id: int
    cells: list[tuple[int, int]]           # all grid cells in this domain
    representative_cell: tuple[int, int]    # cell with highest w_i
    centroid: np.ndarray                    # (x_m, y_m) in meters, for real distance
    info_value: float                       # max w_i within domain
    dominant_variable: str                  # "fmc" | "wind_speed" | "wind_direction"

@dataclass  
class DomainEdge:
    source: int                            # domain ID
    target: int                            # domain ID
    cross_correlation: float               # correlation between domains (0-1)
    information_gain: float                # info from crossing this boundary
    real_distance_m: float                 # flight distance between centroids
    travel_time_s: float                   # real_distance / drone_speed

def build_correlation_graph(w_field, gp, terrain, config):
    """
    Reduce full grid to correlation-domain graph.
    
    w_field: (rows, cols) information value at every cell
    gp: fitted GP (for cross-domain correlation computation)
    terrain: TerrainData
    config: includes correlation_length, resolution
    """
    rows, cols = w_field.shape
    domain_size = int(config.correlation_length / config.resolution_m)
    # e.g., 500m / 50m = 10 cells per domain side
    
    n_dr = rows // domain_size + (1 if rows % domain_size else 0)
    n_dc = cols // domain_size + (1 if cols % domain_size else 0)
    
    # Build domains
    domains = []
    domain_id = 0
    for dr in range(n_dr):
        for dc in range(n_dc):
            r_start = dr * domain_size
            r_end = min(r_start + domain_size, rows)
            c_start = dc * domain_size
            c_end = min(c_start + domain_size, cols)
            
            cells = [(r, c) for r in range(r_start, r_end) 
                            for c in range(c_start, c_end)]
            
            # Representative cell: highest w_i in domain
            w_values = [w_field[r, c] for r, c in cells]
            best_idx = np.argmax(w_values)
            rep_cell = cells[best_idx]
            
            centroid_r = (r_start + r_end) / 2 * config.resolution_m
            centroid_c = (c_start + c_end) / 2 * config.resolution_m
            
            domains.append(CorrelationDomain(
                domain_id=domain_id,
                cells=cells,
                representative_cell=rep_cell,
                centroid=np.array([centroid_r, centroid_c]),
                info_value=w_values[best_idx],
                dominant_variable=get_dominant_variable(rep_cell, w_by_variable)
            ))
            domain_id += 1
    
    # Build edges between adjacent domains
    edges = []
    for i, d_i in enumerate(domains):
        for j, d_j in enumerate(domains):
            if i >= j:
                continue
            
            # Check adjacency (within 1.5× domain size)
            dist = np.linalg.norm(d_i.centroid - d_j.centroid)
            if dist > 1.5 * domain_size * config.resolution_m:
                continue
            
            # Cross-domain correlation from GP kernel
            cross_corr = gp.kernel_(
                _obs_features([d_i.representative_cell], terrain, config.resolution_m),
                _obs_features([d_j.representative_cell], terrain, config.resolution_m)
            )[0, 0]
            
            # Information gain from crossing: high when correlation is low
            # (different terrain/fuel → new information at boundary)
            edge_info = (1.0 - cross_corr) * min(d_i.info_value, d_j.info_value)
            
            # Real flight distance (could account for terrain avoidance)
            real_dist = np.linalg.norm(d_i.centroid - d_j.centroid)
            
            edges.append(DomainEdge(
                source=i, target=j,
                cross_correlation=cross_corr,
                information_gain=edge_info,
                real_distance_m=real_dist,
                travel_time_s=real_dist / config.drone_speed
            ))
    
    return CorrelationGraph(domains, edges)
```

**Scale:** For a 200×200 grid with 500m correlation length at 50m resolution: ~400 domains, ~1,600 edges. For a 2000×2000 grid (100×100 km): ~40,000 domains, ~160,000 edges. Still tractable for graph algorithms.

### Step 2: Precompute Station Return Costs

Before path planning, run Dijkstra from every ground station to every domain using real_distance as edge weights. This gives a lookup table: "from any domain, how far is the nearest reachable station?"

```python
def precompute_return_costs(graph, stations):
    """
    For each domain, compute minimum real distance to any station.
    Uses Dijkstra on the domain graph with real_distance edge weights.
    """
    min_return_cost = np.full(len(graph.domains), np.inf)
    nearest_station = np.full(len(graph.domains), -1, dtype=int)
    
    for station in stations:
        # Station maps to its containing domain
        station_domain = graph.domain_containing(station.location)
        
        # Dijkstra from this station's domain
        distances = dijkstra(graph, source=station_domain, weight='real_distance_m')
        
        for d_id, dist in enumerate(distances):
            if dist < min_return_cost[d_id]:
                min_return_cost[d_id] = dist
                nearest_station[d_id] = station.id
    
    return min_return_cost, nearest_station
```

### Step 3: Greedy Path Planning with Exact GP Updates

For each drone, select a path through the domain graph that maximizes non-redundant information subject to range and depot constraints.

```python
def plan_drone_path(graph, start_station, stations, max_range_m,
                    gp_variance, sensitivity, observability,
                    min_return_cost, nearest_station):
    """
    Plan a single drone's path through the correlation-domain graph.
    
    Uses exact GP conditional variance updates (not heuristic discounts)
    to track non-redundant information along the path.
    """
    start_domain = graph.domain_containing(start_station.location)
    
    path = [start_domain]
    remaining_range = max_range_m
    current = start_domain
    current_variance = gp_variance.copy()
    total_info = 0.0
    marginal_gains = []
    
    while True:
        # Recompute w at each unvisited domain using CURRENT variance
        candidates = []
        for domain in graph.domains:
            if domain.domain_id in [d.domain_id for d in path]:
                continue
            
            rep = domain.representative_cell
            w = current_variance[rep] * abs(sensitivity[rep]) * observability[rep]
            
            # Travel cost to reach this domain
            travel_cost = graph.real_distance(current.domain_id, domain.domain_id)
            if travel_cost is None:
                continue  # not reachable in one hop (could use Dijkstra for multi-hop)
            
            # Can we afford to go there AND return to a station?
            return_cost = min_return_cost[domain.domain_id]
            if travel_cost + return_cost > remaining_range:
                continue
            
            # Information efficiency: non-redundant info per meter of travel
            # Include edge information gain (crossing terrain boundary)
            edge = graph.edge(current.domain_id, domain.domain_id)
            edge_info = edge.information_gain if edge else 0.0
            
            total_domain_info = w + edge_info
            efficiency = total_domain_info / (travel_cost + 1e-6)
            
            candidates.append((domain, efficiency, travel_cost, total_domain_info))
        
        if not candidates:
            break
        
        # Select best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_domain, _, travel_cost, info_gained = candidates[0]
        
        # Update GP variance (exact conditional update at representative cell)
        current_variance = gp_conditional_variance(
            current_variance, best_domain.representative_cell, gp)
        
        # Update path state
        path.append(best_domain)
        remaining_range -= travel_cost
        current = best_domain
        total_info += info_gained
        marginal_gains.append(info_gained)
    
    # Return to nearest station
    return_station_domain = nearest_station[current.domain_id]
    path.append(graph.domains[return_station_domain])
    
    return DronePath(
        domains=path,
        total_info=total_info,
        marginal_gains=marginal_gains,
        total_distance=max_range_m - remaining_range,
        start_station=start_station,
        end_station=stations[nearest_station[current.domain_id]]
    )

def plan_fleet(graph, stations, n_drones, max_range_m,
               gp_variance, sensitivity, observability):
    """
    Plan paths for K drones sequentially.
    After each drone's path is planned, update the GP variance
    to reflect what that drone will observe — subsequent drones
    avoid redundancy with earlier drones' paths.
    """
    paths = []
    current_variance = gp_variance.copy()
    
    for k in range(n_drones):
        # Pick starting station (round-robin or nearest to highest remaining w)
        start = select_start_station(stations, current_variance, sensitivity, k)
        
        path = plan_drone_path(
            graph, start, stations, max_range_m,
            current_variance, sensitivity, observability,
            min_return_cost, nearest_station
        )
        paths.append(path)
        
        # Update variance for all cells along this drone's path
        for domain in path.domains:
            current_variance = gp_conditional_variance(
                current_variance, domain.representative_cell, gp)
    
    return paths
```

### Step 4: Score Full Cell-Level Path

Convert the domain-level path to cell-level waypoints and score with sequential GP conditional variance to get the exact non-redundant information along the actual drone trajectory.

```python
def score_path(domain_path, gp, gp_variance, sensitivity, resolution):
    """
    Exact information along the full cell-level trajectory.
    
    For each cell the drone overflies (including transit between
    domain representative cells), compute marginal information
    conditioned on all previous cells along the path.
    """
    # Convert domain path to cell-level waypoints
    waypoints = [d.representative_cell for d in domain_path]
    
    # Generate all cells along the flight path (Bresenham between waypoints)
    all_cells = []
    for i in range(len(waypoints) - 1):
        segment = bresenham_line(waypoints[i], waypoints[i+1])
        # Add camera footprint (3 cells wide)
        for cell in segment:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    c = (cell[0]+dr, cell[1]+dc)
                    if in_bounds(c, gp_variance.shape) and c not in all_cells:
                        all_cells.append(c)
    
    # Sequential GP scoring
    current_var = gp_variance.copy()
    total_info = 0.0
    per_cell_info = []
    
    for cell in all_cells:
        # Marginal info at this cell given everything observed so far
        w_before = current_var[cell] * abs(sensitivity[cell])
        
        # Update variance
        current_var = gp_conditional_variance(current_var, cell, gp)
        
        w_after = current_var[cell] * abs(sensitivity[cell])
        marginal = w_before - w_after
        total_info += marginal
        per_cell_info.append((cell, marginal))
    
    return PathScore(
        total_info=total_info,
        n_cells_observed=len(all_cells),
        per_cell_info=per_cell_info,
        info_per_meter=total_info / path_length_m
    )
```

### Step 5: Local Refinement (Optional)

After the greedy selects domains and the path scorer evaluates the trajectory, locally perturb waypoints within their domains to find a better path:

```python
def refine_path(domain_path, gp, gp_variance, sensitivity, n_iterations=20):
    """
    Stochastic local search: perturb waypoints within domains,
    rescore, keep if better.
    """
    current_score = score_path(domain_path, gp, gp_variance, sensitivity)
    
    for _ in range(n_iterations):
        # Pick a random domain in the path
        idx = np.random.randint(1, len(domain_path) - 1)  # skip start/end stations
        domain = domain_path[idx]
        
        # Pick a random alternative cell within the domain
        alt_cell = random.choice(domain.cells)
        
        # Create modified path
        modified = domain_path.copy()
        modified[idx] = modified[idx]._replace(representative_cell=alt_cell)
        
        new_score = score_path(modified, gp, gp_variance, sensitivity)
        if new_score.total_info > current_score.total_info:
            domain_path = modified
            current_score = new_score
    
    return domain_path, current_score
```

Cost: 20 iterations × ~10ms per score = 200ms. Marginal improvement ~5-10%.

---
