# Terrain-Adaptive Correlation Domains

---

## Problem

The current correlation-domain graph uses a regular grid partition — blocks of 10×10 cells regardless of terrain. This misaligns with actual correlation structure: a domain straddling a ridgeline contains cells from two different microclimates (south-facing dry slope and north-facing wet slope) that have low FMC correlation with each other. The path planner treats them as one domain, potentially skipping an important boundary or wasting an observation on a domain that should have been two.

Terrain-adaptive domains follow ridgelines, fuel boundaries, and aspect transitions, producing domains where cells within a domain are genuinely correlated and boundaries between domains correspond to real correlation breaks.

---

## Approaches

### 1. Watershed Segmentation on Terrain Features

**Theory:** Watershed segmentation treats a 2D field as a topographic surface and finds the "catchment basins" — regions where all gradient flows converge to the same local minimum. Applied to the DEM, it partitions terrain into drainage basins. Applied to derived terrain features (slope, curvature), it identifies regions of homogeneous terrain morphology.

**For IGNIS:** Apply watershed not to raw elevation but to a composite "terrain dissimilarity" surface. Compute the gradient magnitude of the multi-channel terrain feature space (elevation, aspect, fuel model). High gradient = terrain transition = domain boundary. Watershed basins = correlation domains.

**Algorithm:**

```python
from scipy.ndimage import label, watershed_ift
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def watershed_domains(elevation, aspect, fuel_model, min_domain_size):
    # Compute terrain gradient magnitude (edge strength)
    elev_grad = np.sqrt(np.gradient(elevation, axis=0)**2 + 
                         np.gradient(elevation, axis=1)**2)
    
    # Aspect discontinuity (circular gradient)
    aspect_rad = np.radians(aspect)
    ax = np.gradient(np.cos(aspect_rad), axis=0)**2 + np.gradient(np.cos(aspect_rad), axis=1)**2
    ay = np.gradient(np.sin(aspect_rad), axis=0)**2 + np.gradient(np.sin(aspect_rad), axis=1)**2
    aspect_grad = np.sqrt(ax + ay)
    
    # Fuel model boundary (binary: different fuel = high gradient)
    fuel_edge_x = (np.diff(fuel_model, axis=1) != 0).astype(float)
    fuel_edge_y = (np.diff(fuel_model, axis=0) != 0).astype(float)
    fuel_grad = np.zeros_like(elevation)
    fuel_grad[:, :-1] += fuel_edge_x
    fuel_grad[:-1, :] += fuel_edge_y
    
    # Composite edge map
    edge_map = (elev_grad / (elev_grad.max() + 1e-10) + 
                aspect_grad / (aspect_grad.max() + 1e-10) +
                fuel_grad * 2.0)  # fuel boundaries weighted higher
    
    # Find seeds: local minima in edge map = interior of homogeneous regions
    markers = peak_local_max(-edge_map, min_distance=min_domain_size,
                              labels=np.ones_like(edge_map, dtype=bool))
    marker_array = np.zeros_like(edge_map, dtype=int)
    for i, (r, c) in enumerate(markers):
        marker_array[r, c] = i + 1
    
    # Watershed from seeds
    domains = watershed(edge_map, markers=marker_array)
    
    return domains  # int array (rows, cols), each value = domain ID
```

**Pros:** Naturally follows ridgelines and drainage boundaries. Theoretically motivated — watershed basins are the natural segmentation of terrain into distinct hydrological (and microclimatic) units. Handles irregular domain shapes.

**Cons:** Over-segments in flat terrain (many spurious local minima). Requires tuning `min_distance` parameter. Can produce very small domains at complex terrain intersections. No direct control over domain count.

**Complexity:** O(D log D) for the watershed. Scipy implementation handles 40K cells in ~10ms, 4M cells in ~1 second.

**Available in:** `skimage.segmentation.watershed`, `scipy.ndimage.watershed_ift`

---

### 2. Felzenszwalb Graph-Based Segmentation

**Theory:** Constructs a graph where nodes are cells and edge weights are dissimilarity between adjacent cells. Sorts edges by weight. Merges regions greedily, but with a threshold that adapts to the internal variation of each region. A merge happens only if the edge weight (inter-region dissimilarity) is less than the internal variation of at least one of the two regions, plus a regularization term k/|C| that prevents large regions from absorbing everything.

The key property: "preserve detail in low-variability regions while ignoring detail in high-variability regions." In terrain terms: distinguish fine-scale aspect changes in a complex mountain (preserve detail where terrain varies slowly) while merging across minor undulations in a flat valley (ignore detail where terrain varies rapidly but uniformly).

**For IGNIS:** Build a 4-connected or 8-connected grid graph. Edge weight between adjacent cells = terrain distance (same as the GP kernel's distance function: Euclidean + α × elevation_diff + β × aspect_diff). This directly encodes the correlation structure — cells that the GP kernel considers similar get low edge weights and merge into the same domain.

**Algorithm:**

```python
from skimage.segmentation import felzenszwalb

def felzenszwalb_domains(elevation, aspect, fuel_model, canopy_cover,
                          scale=500, min_size=20):
    """
    Felzenszwalb segmentation on multi-channel terrain features.
    
    scale: controls domain size (higher = larger domains).
           ~500 produces domains of ~correlation-length size.
    min_size: minimum cells per domain (prevents tiny fragments).
    """
    # Stack terrain features into a multi-channel "image"
    # Normalize each to [0, 1] range for equal weighting
    def norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / (r + 1e-10)
    
    # Aspect as two channels (cos, sin) to handle circularity
    aspect_cos = norm(np.cos(np.radians(aspect)))
    aspect_sin = norm(np.sin(np.radians(aspect)))
    
    features = np.stack([
        norm(elevation) * 1.0,         # elevation weight
        aspect_cos * 1.5,              # aspect weight (higher — major FMC driver)
        aspect_sin * 1.5,
        norm(fuel_model.astype(float)) * 2.0,  # fuel type weight (highest — discontinuous)
        norm(canopy_cover) * 0.5,      # canopy cover weight
    ], axis=-1).astype(np.float64)
    
    # Felzenszwalb operates on the stacked feature channels
    domains = felzenszwalb(features, scale=scale, sigma=0.5, min_size=min_size)
    
    return domains
```

**Pros:** Built into scikit-image (one function call). Runs in O(E log E) where E is number of edges — nearly linear. Automatically adapts domain size: fine domains in complex terrain, coarse domains in uniform terrain. The `scale` parameter maps intuitively to correlation length. Handles multi-channel features natively.

**Cons:** The `scale` and `sigma` parameters need calibration. Not specifically designed for geospatial data — treats the grid as an image. Doesn't natively handle circular variables (aspect needs cos/sin encoding). The channel weights affect results significantly and are heuristic.

**Complexity:** O(D log D) for 8-connected grid. ~5ms for 40K cells, ~50ms for 400K cells.

**Available in:** `skimage.segmentation.felzenszwalb` (C implementation, very fast)

---

### 3. Extended SLIC Superpixels (Geospatial SLIC)

**Theory:** SLIC (Simple Linear Iterative Clustering) is a superpixel algorithm that performs local k-means clustering. Seed cluster centers on a regular grid. Iteratively assign each cell to the nearest center in a combined spatial+feature distance, then update centers. Converges in 5-10 iterations.

Nowosad & Stepinski (2022) extended SLIC to work with non-imagery geospatial rasters — arbitrary multi-channel data with custom distance measures. This directly addresses our use case: segment terrain features (elevation, aspect, fuel type) into superpixels that respect both spatial proximity and feature similarity.

**For IGNIS:** Use the terrain-distance function from the GP kernel as the feature distance in SLIC. This guarantees that SLIC domains align with the GP's correlation structure — cells grouped into the same superpixel have high kernel value (high correlation).

**Algorithm:**

```python
def slic_domains(elevation, slope, aspect, fuel_model, canopy_cover,
                  n_domains=400, compactness=10, n_iterations=10):
    """
    SLIC superpixels on terrain features.
    
    n_domains: target number of domains (≈ grid_area / correlation_length²)
    compactness: spatial regularity (higher = more compact, lower = more adaptive)
    """
    # Feature vector per cell
    aspect_cos = np.cos(np.radians(aspect))
    aspect_sin = np.sin(np.radians(aspect))
    
    features = np.stack([
        elevation / elevation.std(),
        slope / (slope.std() + 1e-6),
        aspect_cos * 2.0,    # higher weight on aspect
        aspect_sin * 2.0,
        fuel_model.astype(float) / 13.0,  # normalize to ~[0,1]
        canopy_cover,
    ], axis=-1)
    
    rows, cols = elevation.shape
    n_cells = rows * cols
    cells_per_domain = n_cells / n_domains
    grid_spacing = int(np.sqrt(cells_per_domain))
    
    # Initialize centers on regular grid
    centers = []
    for r in range(grid_spacing // 2, rows, grid_spacing):
        for c in range(grid_spacing // 2, cols, grid_spacing):
            centers.append((r, c, features[r, c]))
    
    labels = np.full((rows, cols), -1, dtype=int)
    distances = np.full((rows, cols), np.inf)
    
    for iteration in range(n_iterations):
        # Assign cells to nearest center
        for k, (cr, cc, cf) in enumerate(centers):
            # Search window: 2S × 2S around center
            r_lo = max(0, cr - grid_spacing)
            r_hi = min(rows, cr + grid_spacing)
            c_lo = max(0, cc - grid_spacing)
            c_hi = min(cols, cc + grid_spacing)
            
            for r in range(r_lo, r_hi):
                for c in range(c_lo, c_hi):
                    # Spatial distance
                    d_spatial = np.sqrt((r - cr)**2 + (c - cc)**2) / grid_spacing
                    # Feature distance
                    d_feature = np.sqrt(((features[r, c] - cf)**2).sum())
                    # Combined distance
                    d = d_feature + compactness * d_spatial
                    
                    if d < distances[r, c]:
                        distances[r, c] = d
                        labels[r, c] = k
        
        # Update centers
        for k in range(len(centers)):
            mask = labels == k
            if mask.any():
                coords = np.argwhere(mask)
                cr_new = int(coords[:, 0].mean())
                cc_new = int(coords[:, 1].mean())
                cf_new = features[mask].mean(axis=0)
                centers[k] = (cr_new, cc_new, cf_new)
    
    return labels
```

**Pros:** Direct control over domain count via `n_domains`. Produces compact, regular-ish domains (good for path planning — no long thin slivers). Naturally handles multi-channel features. The `compactness` parameter trades off spatial regularity vs feature adherence — tunable for your use case. Extended version (Nowosad & Stepinski 2022) is validated on geospatial rasters specifically.

**Cons:** Iterative (5-10 iterations over the full grid). The naive Python implementation above is O(n_domains × grid_spacing² × n_iterations) — slow. But scikit-image has a C implementation: `skimage.segmentation.slic`. The compactness parameter is hard to set without trial-and-error. Produces roughly equal-sized domains, which may not match the actual correlation structure (correlation lengths may vary spatially).

**Complexity:** O(D × n_iterations) for the C implementation. ~20ms for 40K cells, ~200ms for 400K cells.

**Available in:** `skimage.segmentation.slic` (use `channel_axis=-1` for multi-channel)

---

### 4. Kernel-Informed Spectral Clustering

**Theory:** Build a similarity graph where edge weights are the GP kernel values between adjacent cells. Apply spectral clustering: compute the Laplacian eigenvectors, then k-means in the eigenvector space. Cells that the kernel considers highly correlated end up in the same cluster.

**For IGNIS:** This directly uses the GP kernel to define domains — guaranteeing that domains align with the GP's correlation model. Cells in the same domain have high kernel value. Cells in different domains have low kernel value.

**Algorithm:**

```python
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def spectral_domains(terrain, kernel, n_domains, resolution_m):
    """
    Spectral clustering using the GP kernel as the similarity measure.
    """
    rows, cols = terrain.elevation.shape
    D = rows * cols
    
    # Build sparse similarity matrix from kernel (8-connected)
    W = lil_matrix((D, D))
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nidx = nr * cols + nc
                    # Kernel value between adjacent cells
                    x1 = np.array([[r * resolution_m, c * resolution_m,
                                    terrain.elevation[r,c], terrain.aspect[r,c]]])
                    x2 = np.array([[nr * resolution_m, nc * resolution_m,
                                    terrain.elevation[nr,nc], terrain.aspect[nr,nc]]])
                    w = kernel(x1, x2)[0, 0]
                    W[idx, nidx] = w
    
    # Normalized Laplacian
    W = W.tocsr()
    D_diag = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = 1.0 / np.sqrt(D_diag + 1e-10)
    # L_norm = I - D^{-1/2} W D^{-1/2}
    
    # Compute smallest k eigenvectors of L_norm
    eigvals, eigvecs = eigsh(L_norm, k=n_domains, which='SM')
    
    # K-means on eigenvectors
    labels = KMeans(n_clusters=n_domains).fit_predict(eigvecs)
    
    return labels.reshape(rows, cols)
```

**Pros:** Theoretically the most principled — domains are defined by the actual GP kernel, guaranteeing alignment with the correlation model. Captures non-local similarity patterns.

**Cons:** By far the most expensive. Building the sparse similarity matrix is O(8D) kernel evaluations. Computing k eigenvectors of a D×D sparse matrix is O(D × k²) at best. For D=40K and k=400: ~minutes. For D=400K: infeasible. Also requires choosing n_domains in advance.

**Complexity:** O(D × k² × eigenvector_iterations). ~30 seconds for 40K cells with k=400. Impractical beyond 1× scale.

**Available in:** `sklearn.cluster.SpectralClustering` (but doesn't scale well)

---

### 5. Hydrological Watershed (D8 Flow Direction)

**Theory:** Compute flow direction (D8 algorithm) at every cell — the steepest downhill neighbor. Trace flow paths to accumulation sinks. Every cell belongs to exactly one watershed — the set of all cells whose water flows to the same outlet.

**For IGNIS:** Watersheds are natural terrain units where microclimate is relatively uniform (same aspect, similar elevation range, similar exposure). FMC within a watershed is more correlated than FMC across watersheds because the same aspect and elevation profile apply.

**Algorithm:**

```python
# Using pysheds or richdem
import richdem as rd

def hydrological_domains(elevation, min_domain_size):
    dem = rd.rdarray(elevation, no_data=-9999)
    
    # Fill depressions
    rd.FillDepressions(dem, epsilon=True, in_place=True)
    
    # Flow direction
    flow_dir = rd.FlowDir(dem, method='D8')
    
    # Flow accumulation
    accum = rd.FlowAccumulation(flow_dir, method='D8')
    
    # Label watersheds
    labels = rd.Basins(flow_dir)
    
    # Merge small domains
    labels = merge_small_domains(labels, min_domain_size)
    
    return labels
```

**Pros:** Physically the most meaningful segmentation for terrain — watersheds are real geomorphological units. Well-studied algorithms with robust implementations (richdem, pysheds, GRASS GIS). Handles complex terrain correctly (ridgelines become boundaries, valleys are unified).

**Cons:** Only uses elevation — ignores fuel type, canopy cover, and other features that affect FMC correlation. Over-segments in mountainous terrain (many small drainage basins). Under-segments in flat terrain (one huge basin). Requires depression filling, which modifies the DEM. Doesn't directly target correlation-length-sized domains.

**Complexity:** O(D) for D8 + accumulation + basin labeling. Very fast — ~5ms for 40K cells.

**Available in:** `richdem`, `pysheds`, `whitebox-tools`

---

## Comparison

|Method|Complexity|Multi-feature?|Domain shape|Size control|GP alignment|Implementation|
|---|---|---|---|---|---|---|
|Regular grid|O(1)|No|Square blocks|Exact|Poor|Trivial|
|Watershed (terrain gradient)|O(D log D)|Yes (via composite edge map)|Follows ridgelines|Indirect (min_distance)|Moderate|20 lines + skimage|
|Felzenszwalb|O(D log D)|Yes (multi-channel native)|Adaptive to local variation|Via `scale` param|Good|5 lines + skimage|
|Extended SLIC|O(D × iters)|Yes (custom distance)|Compact, semi-regular|Direct (n_domains)|Good|10 lines + skimage|
|Spectral (kernel-based)|O(D × k²)|Yes (via kernel)|Arbitrary|Direct (k)|Perfect|30 lines, slow|
|Hydrological watershed|O(D)|No (elevation only)|Natural drainage basins|Indirect|Moderate|5 lines + richdem|

---

## Recommendation: Felzenszwalb

**Why Felzenszwalb wins for IGNIS:**

**1. The edge weight maps directly to the GP kernel distance.** The Felzenszwalb edge weight between two adjacent cells is their dissimilarity in the terrain feature space. The GP kernel's terrain distance IS this dissimilarity. By using the same terrain features with the same relative weights, Felzenszwalb domains align with the GP correlation structure without explicitly computing kernel values (which is what makes spectral clustering expensive).

**2. Adaptive domain size is physically correct.** In complex terrain (Santa Monica Mountains: steep slopes, many aspect changes, mixed fuels), Felzenszwalb produces many small domains — correctly capturing the high spatial variability. In uniform terrain (flat grassland), it produces few large domains — correctly reflecting the smooth correlation structure. A regular grid can't do this. SLIC's equal-sized domains can't do this.

**3. It's one function call in scikit-image.** The C implementation handles 40K cells in ~5ms. No parameter tuning beyond `scale` (which maps to correlation length) and `min_size` (which prevents tiny fragments). Compare to spectral clustering which requires building a sparse matrix, computing eigenvectors, and running k-means.

**4. Fuel type boundaries are handled correctly.** By including fuel model as a high-weight channel, Felzenszwalb places domain boundaries at fuel type transitions. A ridge where chaparral meets timber has a natural domain boundary because the fuel model changes abruptly. This is physically important — FMC in chaparral and FMC in timber are poorly correlated even at short distances.

---

## Implementation Specification

### Build Correlation Domains via Felzenszwalb

```python
from skimage.segmentation import felzenszwalb
from scipy.ndimage import find_objects, label

def build_terrain_domains(terrain: TerrainData, 
                           target_correlation_length: float,
                           resolution_m: float,
                           min_domain_cells: int = 10) -> np.ndarray:
    """
    Segment terrain into correlation domains using Felzenszwalb.
    
    Returns: int array (rows, cols) where each value is a domain ID.
    """
    # Map scale parameter to correlation length
    # Empirically: scale ≈ correlation_length / resolution × 0.5
    scale = target_correlation_length / resolution_m * 0.5
    
    # Build multi-channel feature array
    # Normalize each feature to comparable ranges
    def norm(x):
        std = x.std()
        return (x - x.mean()) / (std + 1e-10)
    
    # Aspect as cos/sin to handle circularity
    aspect_cos = np.cos(np.radians(terrain.aspect))
    aspect_sin = np.sin(np.radians(terrain.aspect))
    
    # Channel weights reflect importance for FMC correlation:
    # Aspect is the primary driver of microclimate (solar exposure)
    # Fuel type determines fuel response characteristics
    # Elevation affects temperature and thus moisture
    # Canopy cover affects shading and wind exposure
    # Slope affects drainage and solar angle
    features = np.stack([
        norm(terrain.elevation) * 1.0,
        norm(terrain.slope) * 0.5,
        norm(aspect_cos) * 1.5,
        norm(aspect_sin) * 1.5,
        norm(terrain.fuel_model.astype(float)) * 2.0,
        norm(terrain.canopy_cover) * 0.8,
    ], axis=-1).astype(np.float64)
    
    # Run Felzenszwalb
    # sigma: Gaussian smoothing before segmentation (prevents noise-driven splits)
    domains = felzenszwalb(features, scale=scale, sigma=0.8, 
                            min_size=min_domain_cells,
                            channel_axis=-1)
    
    return domains


def build_correlation_graph(domains, terrain, w_field, resolution_m):
    """
    From Felzenszwalb domain labels, build the weighted graph
    for path planning.
    """
    unique_ids = np.unique(domains)
    n_domains = len(unique_ids)
    
    # Compute domain properties
    domain_list = []
    for d_id in unique_ids:
        mask = domains == d_id
        cells = list(zip(*np.where(mask)))
        
        # Representative cell: highest w_i
        w_values = w_field[mask]
        best_idx = np.argmax(w_values)
        rep_cell = cells[best_idx]
        
        # Centroid in meters
        rs, cs = np.where(mask)
        centroid = np.array([rs.mean() * resolution_m, 
                             cs.mean() * resolution_m])
        
        domain_list.append(CorrelationDomain(
            domain_id=d_id,
            cells=cells,
            representative_cell=rep_cell,
            centroid=centroid,
            info_value=w_values.max(),
            area_cells=len(cells),
        ))
    
    # Find adjacent domain pairs
    edges = []
    # Check all 4-connected neighbor pairs across the grid
    adjacency = set()
    for r in range(domains.shape[0] - 1):
        for c in range(domains.shape[1] - 1):
            here = domains[r, c]
            right = domains[r, c + 1]
            below = domains[r + 1, c]
            if here != right:
                pair = (min(here, right), max(here, right))
                adjacency.add(pair)
            if here != below:
                pair = (min(here, below), max(here, below))
                adjacency.add(pair)
    
    # Build edges with real distance and cross-domain info gain
    domain_dict = {d.domain_id: d for d in domain_list}
    for d_i, d_j in adjacency:
        di = domain_dict[d_i]
        dj = domain_dict[d_j]
        
        real_dist = np.linalg.norm(di.centroid - dj.centroid)
        
        # Cross-domain correlation: use terrain feature dissimilarity
        # at the representative cells as proxy
        # (avoids expensive kernel evaluation)
        feat_i = features[di.representative_cell]
        feat_j = features[dj.representative_cell]
        dissimilarity = np.sqrt(((feat_i - feat_j)**2).sum())
        cross_corr = np.exp(-dissimilarity)  # approximate kernel value
        
        edge_info = (1.0 - cross_corr) * min(di.info_value, dj.info_value)
        
        edges.append(DomainEdge(
            source=d_i, target=d_j,
            cross_correlation=cross_corr,
            information_gain=edge_info,
            real_distance_m=real_dist,
        ))
    
    return CorrelationGraph(domain_list, edges)
```

### Validation

|Test|Expected result|
|---|---|
|Flat uniform terrain (control)|Few large domains, regular-ish shapes|
|Mountain ridgeline|Domain boundary follows the ridge — domains don't straddle it|
|Fuel type transition (chaparral → timber)|Domain boundary at fuel change|
|South-facing vs north-facing slope|Separate domains even at close distance|
|Canyon mouth|Narrow domain or domain boundary at the entrance|
|Urban/non-burnable area|Separate domain (fuel model drives segmentation)|

### Calibrating the Scale Parameter

The `scale` parameter in Felzenszwalb controls how willing the algorithm is to merge regions. Higher scale = larger domains. The mapping to correlation length is approximate:

```python
# Target: domains of approximately correlation_length diameter
# Felzenszwalb scale is in feature-space dissimilarity units
# Empirically calibrate by running once and checking:
domains = build_terrain_domains(terrain, target_correlation_length=500)
sizes = np.bincount(domains.ravel())
median_size = np.median(sizes)
target_size = (target_correlation_length / resolution_m) ** 2  # ~100 cells for 500m at 50m
print(f"Median domain: {median_size} cells, target: {target_size}")
# Adjust scale up if median is too small, down if too large
```

For the hackathon: run once, eyeball the domain map, adjust scale if domains are obviously too small or too large. Takes 30 seconds.

---

## Fallback

If for any reason the terrain-adaptive segmentation produces pathological results (too many tiny domains, one giant domain, domains that don't make physical sense), fall back to the regular grid partition. The path planner works with either — it just consumes a `CorrelationGraph` regardless of how the domains were constructed. The regular grid is always correct if suboptimal. The adaptive segmentation is better when it works but has failure modes (bad parameter tuning, unusual terrain).

```python
def build_domains(terrain, w_field, config):
    try:
        domains = build_terrain_domains(terrain, config.correlation_length,
                                         config.resolution_m)
        # Sanity check
        sizes = np.bincount(domains.ravel())
        if sizes.max() > 0.5 * domains.size:  # one domain > 50% of grid
            raise ValueError("Single domain too large — falling back to grid")
        if len(sizes) > domains.size / 4:  # more than 25% of cells are their own domain
            raise ValueError("Too many tiny domains — falling back to grid")
    except Exception:
        domains = regular_grid_domains(terrain.shape, config.correlation_length,
                                        config.resolution_m)
    
    return build_correlation_graph(domains, terrain, w_field, config.resolution_m)
```