# WISP: Comprehensive System Design

---

## 1. What the System Does

WISP takes the data currently available to fire managers (sparse weather stations, satellite imagery, terrain maps), quantifies exactly where and why fire predictions are uncertain, and directs drones to the locations where measurement most reduces that uncertainty. It then assimilates the drone observations and repeats — each cycle sharpening the prediction in the regions that matter most.

The system answers three questions every ~20 minutes:

1. Where is the fire model most uncertain about what will happen next?
2. Which of those uncertainties actually affect the predicted fire trajectory?
3. Where should drones fly to resolve the uncertainties that matter?

---

## 2. Data Flow Overview

```
    EXTERNAL DATA                      WISP PIPELINE
    
    LANDFIRE (30m)                 ┌──────────────────┐
    ├─ Elevation/Slope/Aspect      │  1. TERRAIN       │
    ├─ Fuel models (Anderson 13)───┤     (static,      │
    └─ Canopy cover                │      loaded once)  │
                                   └────────┬─────────┘
    RAWS stations (~50km apart)             │
    ├─ Temperature                          │
    ├─ Humidity                    ┌────────▼─────────┐
    ├─ Wind speed/direction────────┤  2. GP PRIOR      │
    └─ 10-hr fuel moisture stick   │     Posterior mean │
                                   │     + variance at  │
    Weather forecast (HRRR)────────┤     every cell     │
                                   └────────┬─────────┘
    Satellite (MODIS/VIIRS)                 │
    └─ Fire perimeter detection    ┌────────▼─────────┐
          │                        │  3. ENSEMBLE       │
          │                        │     N members,     │
          └────────────────────────┤     perturbed by   │
                                   │     GP variance    │
                                   └────────┬─────────┘
                                            │
                                   ┌────────▼─────────┐
                                   │  4. INFORMATION   │
                                   │     FIELD          │
                                   │     w_i at every   │
                                   │     grid cell      │
                                   └───┬─────────┬────┘
                                       │         │
                              ┌────────▼──┐ ┌────▼───────┐
                              │ 5a. GREEDY│ │5b. QUBO    │
                              │  SELECTOR │ │  SELECTOR  │
                              └────┬──────┘ └────┬───────┘
                                   │             │
                              ┌────▼─────────────▼────┐
                              │  6. EVALUATION &       │
                              │     COMPARISON         │
                              └────────────┬──────────┘
                                           │
                              ┌────────────▼──────────┐
      MISSION QUEUE ◄─────────┤  7. PATH PLANNING     │
      (to UTM layer)          │     + OBSERVATION SIM  │
                              └────────────┬──────────┘
                                           │
                              ┌────────────▼──────────┐
                              │  8. DATA ASSIMILATION  │
                              │     EnKF + GP update   │
                              └────────────┬──────────┘
                                           │
                                    RETURN TO STEP 2
```

---

## 3. Component Specifications

### 3.1 Terrain Manager (static, loaded once)

**Purpose:** Load and serve terrain and fuel data. Immutable after initialization.

**Source:** LANDFIRE via `landfire-python` library. One API call downloads elevation, slope, aspect, and fuel models for any bounding box at 30m resolution. Resample to 50m for computational tractability.

**Data:**

```python
@dataclass(frozen=True)
class TerrainData:
    elevation: np.ndarray       # float32[rows, cols], meters
    slope: np.ndarray           # float32[rows, cols], degrees
    aspect: np.ndarray          # float32[rows, cols], degrees from north
    fuel_model: np.ndarray      # int8[rows, cols], Anderson 13 fuel model IDs
    resolution_m: float         # 50.0
    origin: tuple[float, float] # (lat, lon) of NW corner
    shape: tuple[int, int]      # (rows, cols)
```

**Fuel model parameters:** Each Anderson fuel model ID maps to a set of Rothermel parameters (fuel load, surface-area-to-volume ratio, fuel bed depth, moisture of extinction, dead fuel heat content). These are published constants from Andrews (2018, RMRS-GTR-371). Store as a lookup dict:

```python
FUEL_PARAMS = {
    1: {"load": 0.034, "sav": 3500, "depth": 0.305, "mx": 0.12, ...},
    2: {"load": 0.092, "sav": 3000, "depth": 0.305, "mx": 0.15, ...},
    # ... through fuel model 13
}
```

**Implementation:** ~30 lines. Download with `landfire-python`, load with `rasterio`, resample with `scipy.ndimage.zoom`, build lookup arrays.

---

### 3.2 Gaussian Process Prior

**Purpose:** Compute a spatially resolved estimate of FMC and wind fields with calibrated uncertainty at every grid cell, conditioned on all available observations (RAWS stations, weather forecast, and any previous drone measurements).

**Mathematical foundation:** Gaussian process regression (Rasmussen & Williams 2006). Given observations y at locations X_obs:

```
Posterior mean:     μ(x) = k(x, X_obs) [K(X_obs, X_obs) + σ²_n I]⁻¹ y
Posterior variance: σ²(x) = k(x,x) - k(x, X_obs) [K(X_obs, X_obs) + σ²_n I]⁻¹ k(X_obs, x)
```

The posterior variance σ²(x) depends only on observation geometry and the kernel — not on observed values. This means the uncertainty map can be computed before seeing data, and updated cheaply when new observations arrive.

**Kernel choice:** Matérn 3/2 with terrain-aware distance. Two points on the same aspect and elevation band are more correlated than two points at equal Euclidean distance on opposite sides of a ridge.

```python
def terrain_distance(x1, x2, terrain):
    geo_dist = euclidean(x1, x2)
    elev_diff = abs(terrain.elevation[x1] - terrain.elevation[x2])
    aspect_diff = angular_diff(terrain.aspect[x1], terrain.aspect[x2])
    return geo_dist + alpha * elev_diff + beta * aspect_diff
```

Hyperparameters (correlation length, alpha, beta) are fitted from RAWS data or set to physically reasonable defaults (correlation length ~1-2 km for FMC, ~5 km for wind).

**Conditional variance update (used by greedy selector):** When a new observation is added at x_new, the variance at every other point updates as:

```
σ²_updated(x) = σ²(x) - k(x, x_new)² / (k(x_new, x_new) + σ²_noise)
```

This is one vectorized operation — no matrix inversion, no refitting. The greedy selector calls this K times (once per drone). Cost: microseconds per call.

**Output:**

```python
@dataclass(frozen=True)
class GPPrior:
    fmc_mean: np.ndarray         # float32[rows, cols]
    fmc_variance: np.ndarray     # float32[rows, cols]
    wind_speed_mean: np.ndarray  # float32[rows, cols]
    wind_speed_variance: np.ndarray
    wind_dir_mean: np.ndarray
    wind_dir_variance: np.ndarray
```

**Implementation:** Use scikit-learn `GaussianProcessRegressor` for fitting. For the conditional variance update, implement the closed-form equation directly in NumPy (faster than refitting the full GP each time). ~60 lines total.

---

### 3.3 Ensemble Fire Engine

**Purpose:** Run N fire spread simulations with parameters drawn from the GP posterior, producing N arrival time maps whose spread quantifies prediction uncertainty.

**Perturbation generation:** Each ensemble member gets a spatially correlated perturbation field for each uncertain variable, scaled by the GP posterior variance at each cell:

```python
def generate_member(gp_prior, correlation_length):
    # Draw spatially correlated noise with unit variance
    noise = draw_correlated_field(shape, correlation_length)  # standard GP draw
    # Scale by local GP uncertainty
    fmc = gp_prior.fmc_mean + noise * np.sqrt(gp_prior.fmc_variance)
    return fmc
```

Cells near RAWS stations (low GP variance) get small perturbations. Cells far from any observation (high GP variance) get large perturbations. The correlation structure ensures adjacent cells on the same terrain feature get similar perturbations.

**Correlated field generation:** Drawing from a multivariate normal with a full covariance matrix is O(D³) for D grid cells — infeasible for D = 40,000. Use circulant embedding with FFT instead: generate a correlated field by convolving white noise with the kernel's square root in Fourier space. O(D log D). Standard technique.

```python
def draw_correlated_field(shape, correlation_length, resolution):
    # Generate white noise
    white = np.random.randn(*shape)
    # Build kernel in Fourier space
    freqs = np.fft.fftfreq(shape[0], d=resolution)
    freq_grid = np.sqrt(freqs[:,None]**2 + freqs[None,:]**2)
    kernel_fft = np.exp(-2 * (np.pi * correlation_length * freq_grid)**2)
    # Convolve
    return np.real(np.fft.ifft2(np.fft.fft2(white) * np.sqrt(kernel_fft)))
```

**Fire spread model:** Cellular automata with Rothermel rate-of-spread as the transition kernel. Each cell's ignition probability depends on:

- Whether any neighbor is burning
- The Rothermel ROS at this cell (from its fuel model, FMC, wind, slope)
- The time step

```
P(ignite) = 1 - exp(-ROS × dt / cell_size)  per burning neighbor
P(remain unburned) = ∏(1 - P(ignite)) over all burning neighbors
```

The Rothermel ROS equation:

```
ROS = (I_R × ξ × (1 + φ_w + φ_s)) / (ρ_b × ε × Q_ig)
```

- I_R: reaction intensity (from fuel load, SAV ratio, packing ratio, mineral content)
- ξ: propagating flux ratio
- φ_w: wind factor = C × (3.281 × wind_speed)^B × (β/β_op)^(-E)
- φ_s: slope factor = 5.275 × β^(-0.3) × tan(slope)²
- Q_ig: heat of preignition = 250 + 1116 × FMC

The FMC dependence enters through Q_ig (denominator) and through the moisture damping coefficient η_M that multiplies I_R. Both make ROS decrease as FMC increases.

**Wind directionality:** The ROS above gives the maximum spread rate (in the wind direction). Spread in other directions follows an elliptical model: ROS(θ) = ROS_max × (1 - ε) / (1 - ε × cos(θ)), where θ is the angle from the wind direction and ε is the eccentricity. Each of the 8 CA neighbors gets a different effective ROS based on its angular relationship to wind direction.

**Ensemble execution:** All N members are independent — run in parallel.

- **CPU:** `multiprocessing.Pool(n_cores).map(run_member, parameter_sets)` — 8× speedup on 8-core machine
- **GPU:** Implement the Rothermel equation and CA step as PyTorch tensor operations. Stack all N members along a batch dimension. All members execute in a single kernel launch. Expected speedup: 100×+ over single-core CPU.

**Output:**

```python
@dataclass(frozen=True)
class EnsembleResult:
    member_arrival_times: np.ndarray      # float32[N, rows, cols], NaN = unburned
    member_fmc_fields: np.ndarray         # float32[N, rows, cols], perturbed FMC used
    member_wind_fields: np.ndarray        # float32[N, rows, cols], perturbed wind used
    burn_probability: np.ndarray          # float32[rows, cols], fraction burned
    mean_arrival_time: np.ndarray         # float32[rows, cols]
    arrival_time_variance: np.ndarray     # float32[rows, cols]
    n_members: int
```

**Implementation:** The Rothermel parameter computation is ~40 lines of arithmetic. The CA step is ~30 lines. The ensemble wrapper with parallelization is ~40 lines. The correlated field generation is ~20 lines. Total: ~130 lines for the fire engine.

---

### 3.4 Information Field Computation

**Purpose:** Compute, at every grid cell, the information value of a measurement — combining uncertainty (from the GP), sensitivity (from the ensemble), and observability (from drone sensor capabilities).

**Step 1: Sensitivity computation.** For each variable v and each cell c, compute the correlation between the ensemble's arrival time at c and the ensemble's input perturbation for v:

```python
# Vectorized over all cells simultaneously
# A: (N, D) arrival times, D = rows*cols
# p_v: (N,) perturbation magnitudes for variable v
A_centered = A - A.mean(axis=0)
p_centered = p_v - p_v.mean()
cov = (p_centered @ A_centered) / (N - 1)          # (D,)
sensitivity_v = cov / (A_centered.std(axis=0) * p_centered.std() + 1e-10)  # (D,)
sensitivity_v = sensitivity_v.reshape(rows, cols)
```

This is one matrix-vector multiply per variable — milliseconds total.

**Subtlety with structured perturbations:** The above assumes each member has a scalar perturbation per variable (δ_fmc_n applied uniformly). But our perturbations are spatially varying fields — each member has a different FMC perturbation at each cell. To compute per-cell sensitivity with spatially varying perturbations, use the _local_ perturbation:

```python
# member_fmc_fields: (N, rows, cols) — actual FMC used by each member
# gp_prior.fmc_mean: (rows, cols) — baseline
perturbation_fields = member_fmc_fields - gp_prior.fmc_mean  # (N, rows, cols)

# Per-cell correlation between arrival time and LOCAL perturbation
for each cell c:
    sensitivity_fmc[c] = corr(arrival_times[:, c], perturbation_fields[:, c])
```

This is more expensive (per-cell correlation) but captures the actual local relationship. Vectorize as:

```python
# Both shape (N, D)
A_flat = member_arrival_times.reshape(N, -1)
P_flat = perturbation_fields.reshape(N, -1)
A_c = A_flat - A_flat.mean(0)
P_c = P_flat - P_flat.mean(0)
# Per-cell correlation: elementwise operations
cov = (A_c * P_c).sum(0) / (N - 1)                        # (D,)
sensitivity = cov / (A_c.std(0) * P_c.std(0) + 1e-10)     # (D,)
```

One line of NumPy broadcasting. Milliseconds.

**Step 2: Observability.** Which variables can the drone measure at each location?

```python
# FMC: measurable everywhere by multispectral camera (R² ~ 0.86)
D_fmc = np.full((rows, cols), 0.86)

# Wind: measurable by anemometer but accuracy degrades at altitude
# and in complex terrain
D_wind = np.full((rows, cols), 0.9)

# Optionally degrade near active fire (smoke, turbulence)
fire_proximity = distance_to_fire(ensemble.burn_probability)
degradation = np.clip(fire_proximity / degradation_radius, 0, 1)
D_fmc *= degradation
D_wind *= degradation
```

**Step 3: Information value.**

```python
w = (gp_prior.fmc_variance * np.abs(sensitivity_fmc) * D_fmc +
     gp_prior.wind_speed_variance * np.abs(sensitivity_wind) * D_wind +
     gp_prior.wind_dir_variance * np.abs(sensitivity_wd) * D_wd)
```

One line. Elementwise multiply and sum. The result is a (rows, cols) heatmap — the information field.

**Step 4: Priority and exclusion overlays.** If operator has specified priority regions or exclusion zones:

```python
w *= priority_weight_field    # >1 in priority regions, 1 elsewhere
w[exclusion_mask] = 0         # zero in excluded regions
```

**Output:**

```python
@dataclass(frozen=True)
class InformationField:
    w: np.ndarray                           # float32[rows, cols], total info value
    w_by_variable: dict[str, np.ndarray]    # per-variable breakdown
    sensitivity: dict[str, np.ndarray]      # S_v per variable
    gp_variance: dict[str, np.ndarray]      # σ²_v per variable
```

**Implementation:** ~40 lines. The entire computation is elementwise array operations on outputs that already exist from steps 3.2 and 3.3.

---

### 3.5a Greedy Selector

**Purpose:** Select K measurement locations by iteratively choosing the highest-value cell, then updating the GP variance to account for the information that observation would provide, thus naturally handling redundancy.

**Algorithm:**

```
1. Start with current GP posterior variance
2. Compute w = σ² × sensitivity × observability at all cells
3. Select cell with highest w (with minimum spacing constraint)
4. Update GP variance field: σ²_new(x) = σ²(x) - k(x, x_sel)² / (k(x_sel, x_sel) + σ²_n)
5. Recompute w with updated σ²
6. Repeat from step 3 until K locations selected
```

The GP update (step 4) is the key: it reduces variance at cells correlated with the selected location, ensuring the next selection favors a different region. No explicit redundancy encoding needed — the physics of spatial correlation handles it through the GP kernel.

**Theoretical guarantee:** Under Gaussian assumptions, mutual information is submodular (Krause et al. 2008). Greedy maximization of a submodular function achieves ≥ (1 - 1/e) ≈ 63% of optimal. This guarantee holds regardless of problem size.

**Output:**

```python
@dataclass(frozen=True)
class SelectionResult:
    selected_locations: list[tuple[int, int]]
    marginal_gains: list[float]       # w_i at time of each selection
    cumulative_gain: list[float]      # running total — the drone value curve
    strategy_name: str
    compute_time_s: float
```

**Implementation:** ~50 lines. The GP variance update is one vectorized operation per selection.

---

### 3.5b QUBO Selector

**Purpose:** Encode the same information-gain-with-redundancy problem as a QUBO matrix and solve via quantum annealing or classical fallback. This is the quantum computing connection.

**Step 1: Candidate selection.** Take the top M cells by w_i with a minimum spacing filter. Since w_i is already computed over the full grid (step 3.4), this is just a sort-and-filter:

```python
candidates = []
for idx in np.argsort(w.ravel())[::-1]:
    loc = np.unravel_index(idx, w.shape)
    if all(distance(loc, c) > min_spacing for c in candidates):
        candidates.append(loc)
    if len(candidates) >= M:
        break
```

M = 200-400. The candidates are the locations the QUBO will choose among. They are pre-filtered to high-value regions, so the QUBO's combinatorial search is focused where it matters.

**Step 2: QUBO construction.** Linear terms from w_i at candidates. Quadratic terms from spatial correlation in the ensemble:

```python
# Linear terms (M,)
w_cand = np.array([w[c] for c in candidates])

# Quadratic terms (M, M) from ensemble covariance at candidates
# For each variable, extract ensemble values at candidate locations
# and compute pairwise correlation
J = np.zeros((M, M))
for var in variables:
    values = ensemble_perturbation_at_candidates[var]  # (N, M)
    rho = np.corrcoef(values.T)                        # (M, M)
    w_var = np.array([w_by_variable[var][c] for c in candidates])  # (M,)
    J -= rho * np.sqrt(np.outer(w_var, w_var))

# Assemble QUBO with cardinality penalty
lam = np.max(np.abs(w_cand))
Q = np.zeros((M, M))
for i in range(M):
    Q[i,i] = -w_cand[i] + lam * (1 - 2*K)
for i in range(M):
    for j in range(i+1, M):
        Q[i,j] = -J[i,j] + 2*lam
```

**Step 3: Solve with fallback chain.**

```
1. D-Wave QPU via Ocean SDK (EmbeddingComposite + DWaveSampler)
2. Simulated annealing (neal.SimulatedAnnealingSampler) 
3. Greedy fallback (always succeeds)
```

Log which solver was used. The comparison framework evaluates solution quality regardless of solver.

**Physical meaning of the QUBO terms:**

- Diagonal Q_ii: how valuable is location i alone? (high w_i = more valuable)
- Off-diagonal Q_ij: how redundant are locations i and j? (high correlation in same variable = redundant, negative Q_ij penalizes selecting both)
- Penalty terms: enforce exactly K selections

**Output:** Same `SelectionResult` dataclass as greedy, plus solver metadata (energy, chain breaks, solver name).

**Implementation:** ~100 lines for QUBO construction + solver wrapper.

---

### 3.5c Baseline Selectors

**Uniform:** K locations on a regular grid. ~5 lines.

**Fire-front following:** K locations evenly spaced along the predicted fire perimeter (cells with burn probability between 0.2 and 0.8). ~10 lines.

Both use the same `SelectionResult` output format.

---

### 3.6 Counterfactual Evaluation

**Purpose:** Compare all selection strategies on equal footing. For each strategy's selected locations, simulate what would happen if drones observed there, and measure how much prediction uncertainty would decrease.

**Method:** For each strategy:

1. Generate synthetic observations at selected locations from a hidden ground truth (the "true" FMC and wind fields, known only to this component).
    
2. Compute the GP posterior variance that would result from adding those observations.
    
3. Measure entropy reduction: how much does total w (integrated over the grid) decrease?
    

```python
def evaluate_strategy(selected_locs, ground_truth, gp, info_field, noise):
    # Simulate observations
    observations = []
    for loc in selected_locs:
        obs_fmc = ground_truth.fmc[loc] + np.random.normal(0, noise.fmc_sigma)
        observations.append((loc, obs_fmc))
    
    # Compute updated GP variance after these observations
    gp_updated_variance = gp.fmc_variance.copy()
    for loc, _ in observations:
        gp_updated_variance = gp.conditional_variance(gp_updated_variance, loc)
    
    # Recompute w with updated variance (sensitivity unchanged)
    w_after = gp_updated_variance * np.abs(info_field.sensitivity["fmc"]) * D_fmc
    # ... add other variables
    
    entropy_before = info_field.w.sum()
    entropy_after = w_after.sum()
    reduction = entropy_before - entropy_after
    perr = reduction / len(selected_locs)  # per-drone metric
    
    return {"entropy_reduction": reduction, "perr": perr}
```

**Path-integrated observations:** In reality, a drone doesn't teleport to K points — it flies paths and observes every cell along the way. For a more realistic evaluation, compute the cells each drone would overfly during transit between selected locations and include those as observations too:

```python
def cells_along_path(path, grid_resolution):
    """Return all grid cells the drone flies over."""
    cells = set()
    for i in range(len(path) - 1):
        # Bresenham's line algorithm between consecutive waypoints
        for cell in bresenham(path[i], path[i+1]):
            # Camera footprint: 3 cells wide at flight altitude
            for offset in [(-1,0), (0,0), (1,0), (0,-1), (0,1)]:
                cells.add((cell[0]+offset[0], cell[1]+offset[1]))
    return list(cells)
```

This makes the comparison more favorable to targeted strategies — they choose paths through terrain transitions where each new cell provides non-redundant information, while uniform paths waste coverage on homogeneous areas.

**Only the primary strategy's observations update actual system state.** The other strategies are evaluated counterfactually — "what would have happened if." This ensures the fire state evolves consistently across cycles while still producing fair comparisons.

**Output:**

```python
@dataclass(frozen=True)
class CycleReport:
    cycle_id: int
    info_field: InformationField               # the w heatmap for this cycle
    evaluations: dict[str, StrategyEvaluation]  # per-strategy metrics
    ensemble_summary: dict                      # burn probability, variance stats
    placement_stability: float                  # Jaccard with previous cycle's primary
```

**Implementation:** ~60 lines.

---

### 3.7 Path Planner

**Purpose:** Convert selected target locations into feasible drone flight plans. Produce the cell list that each drone would observe along its path.

**For the hackathon:** simple nearest-neighbor routing from a staging area through assigned targets and back. No airspace constraints.

```python
def plan_paths(selected_locations, staging_area, n_drones, drone_range):
    # Assign targets to drones (split by proximity)
    assignments = assign_targets_to_drones(selected_locations, n_drones)
    
    plans = []
    for drone_id, targets in assignments.items():
        # Order targets by nearest-neighbor from staging area
        path = [staging_area]
        remaining = list(targets)
        while remaining:
            nearest = min(remaining, key=lambda t: distance(path[-1], t))
            path.append(nearest)
            remaining.remove(nearest)
        path.append(staging_area)  # return
        
        # Compute all cells observed along path
        observed = cells_along_path(path, resolution)
        plans.append(DronePlan(waypoints=path, cells_observed=observed))
    
    return plans
```

**Implementation:** ~40 lines.

---

### 3.8 Observation Simulation

**Purpose:** Generate synthetic drone observations from a hidden ground truth. Swap for real observations in a deployed system.

**Interface:**

```python
class ObservationSource(Protocol):
    def observe(self, cells: list[tuple[int,int]]) -> list[DroneObservation]: ...

class SimulatedSource:
    def __init__(self, ground_truth, noise_config):
        self._truth = ground_truth
        self._noise = noise_config
    
    def observe(self, cells):
        observations = []
        for cell in cells:
            obs = DroneObservation(
                location=cell,
                fmc=self._truth.fmc[cell] + np.random.normal(0, self._noise.fmc_sigma),
                fmc_sigma=self._noise.fmc_sigma,
                wind_speed=self._truth.wind[cell] + np.random.normal(0, self._noise.ws_sigma),
                wind_speed_sigma=self._noise.ws_sigma
            )
            observations.append(obs)
        return observations
```

The noise parameters should reflect real drone sensor accuracy: FMC sigma ~0.05 (from UAV multispectral R² ~ 0.86), wind speed sigma ~1 m/s.

**Implementation:** ~20 lines.

---

### 3.9 Data Assimilation

**Purpose:** Update the ensemble state using drone observations, so the next cycle's predictions reflect what was learned.

**Two parallel updates happen:**

**GP update:** Add drone observation locations and values to the GP's conditioning set. The posterior variance field drops near observed locations. This directly affects the next cycle's perturbation scales — cells that were just observed get smaller perturbations.

```python
gp.add_observations(new_locations, new_values, new_noise)
# Next call to gp.predict() returns updated mean and variance
```

**EnKF update:** Adjust each ensemble member's state to be consistent with observations. The Kalman gain propagates information from observed cells to unobserved cells via ensemble covariance.

```python
def enkf_update(ensemble_states, observations, obs_locations, obs_noise):
    N = ensemble_states.shape[0]      # number of members
    D = ensemble_states.shape[1]      # state dimension (rows*cols)
    n_obs = len(observations)
    
    # Ensemble mean and anomalies
    x_mean = ensemble_states.mean(axis=0)                    # (D,)
    A = ensemble_states - x_mean                              # (N, D)
    
    # Observation operator: extract state at obs locations
    H_indices = [r * cols + c for r, c in obs_locations]
    HX = ensemble_states[:, H_indices]                        # (N, n_obs)
    HA = HX - HX.mean(axis=0)                                # (N, n_obs)
    
    # Covariances
    PHT = (A.T @ HA) / (N - 1)                               # (D, n_obs)
    HPHT = (HA.T @ HA) / (N - 1)                             # (n_obs, n_obs)
    R = np.diag(obs_noise ** 2)                               # (n_obs, n_obs)
    
    # Kalman gain
    K = PHT @ np.linalg.inv(HPHT + R)                        # (D, n_obs)
    
    # Localization: taper K beyond correlation radius
    for j, obs_loc in enumerate(obs_locations):
        distances = compute_distances(obs_loc, all_grid_cells)
        taper = gaspari_cohn(distances, localization_radius)  # (D,)
        K[:, j] *= taper
    
    # Update each member with perturbed observations
    y = np.array(observations)                                # (n_obs,)
    for n in range(N):
        y_perturbed = y + np.random.multivariate_normal(np.zeros(n_obs), R)
        ensemble_states[n] += K @ (y_perturbed - HX[n])
    
    return ensemble_states
```

**Replan triggers:** After assimilation, compute two diagnostics:

- Variance reduction: if total posterior variance dropped by >20%, flag for the orchestrator
- Wind shift: if observed wind differs from prior mean by >30° at any location, flag for immediate full re-solve

These are returned as flags — the assimilator does not decide whether to replan.

**Implementation:** ~60 lines including localization.

---

### 3.10 Orchestrator

**Purpose:** Sequence all components. Manage cycle state. The only component that mutates state.

```python
class Orchestrator:
    def run_cycle(self) -> CycleReport:
        # 1. GP prior from current observations
        gp_prior = self.gp.predict(self.grid)
        
        # 2. Ensemble with GP-scaled perturbations
        ensemble = self.fire_engine.run(self.terrain, gp_prior, self.fire_state,
                                         n_members=self.N, horizon=self.T)
        
        # 3. Information field over full grid
        info_field = compute_information_field(ensemble, gp_prior, self.observability)
        
        # 4. Run all selectors
        greedy_result = self.greedy.select(info_field, gp_prior, self.K)
        qubo_result = self.qubo.select(info_field, ensemble, self.K, M=300)
        uniform_result = self.uniform.select(self.terrain.shape, self.K)
        firefront_result = self.firefront.select(ensemble, self.K)
        
        # 5. Evaluate all counterfactually
        evaluations = {}
        for name, result in [("greedy", greedy_result), ("qubo", qubo_result),
                              ("uniform", uniform_result), ("fire_front", firefront_result)]:
            plan = self.path_planner.plan(result.selected_locations)
            evaluations[name] = self.evaluator.evaluate(
                name, plan.cells_observed, self.ground_truth, gp_prior, info_field)
        
        # 6. Primary strategy updates actual state
        primary = evaluations[self.primary_strategy]
        primary_plan = self.path_planner.plan(
            getattr(self, f"{self.primary_strategy}_result").selected_locations)
        observations = self.obs_source.observe(primary_plan.cells_observed)
        
        # 7. Assimilate
        self.gp.add_observations(
            [o.location for o in observations],
            [o.fmc for o in observations],
            [o.fmc_sigma for o in observations])
        
        updated_ensemble = enkf_update(
            ensemble.member_fmc_fields.reshape(self.N, -1),
            [o.fmc for o in observations],
            [o.location for o in observations],
            np.array([o.fmc_sigma for o in observations]))
        
        # 8. Build report
        stability = jaccard(greedy_result.selected_locations,
                           self.previous_selections)
        self.previous_selections = greedy_result.selected_locations
        self.cycle_count += 1
        
        return CycleReport(self.cycle_count, info_field, evaluations, stability)
```

**Implementation:** ~80 lines.

---

## 4. External Interface

### WISP → UTM (Mission Queue)

```python
@dataclass(frozen=True)
class MissionRequest:
    target: tuple[float, float]        # (lat, lon)
    information_value: float           # w_i
    dominant_variable: str             # "fmc" | "wind_speed"
    substitutes: list[tuple[float,float]]  # fallback locations
    expiry_minutes: float              # after this, re-solve needed

class MissionQueue:
    requests: list[MissionRequest]     # sorted by info value, descending
```

### UTM → WISP

```python
def ingest_observation(obs: DroneObservation) -> Optional[MissionQueue]:
    """Assimilate observation, optionally trigger replan."""

def add_exclusion_zone(polygon, reason) -> None:
    """Exclude region from future targeting."""

def add_priority_region(polygon, weight) -> None:
    """Boost information value in region."""
```

---

## 5. Degradation Contracts

|Component|Failure|Behavior|
|---|---|---|
|LANDFIRE API|Down|Use synthetic terrain (fractal DEM + random fuel)|
|GP fitting|Insufficient RAWS data|Use constant prior variance everywhere|
|Fire engine|Zero variance (all members agree)|Return empty mission queue — model is confident|
|QUBO solver (D-Wave)|Unreachable|Fall back to simulated annealing|
|QUBO solver (SA)|Timeout|Fall back to greedy|
|EnKF|No observations|Pass through prior unchanged|
|EnKF|Wild outlier observation|Log warning, apply (localization limits damage)|

---

## 6. Build Order

|Day|Priority|What to Build|Validates|
|---|---|---|---|
|1 AM|Highest|Shared data types. TerrainData, GPPrior, EnsembleResult, InformationField, SelectionResult. All five people agree before writing code.|Everyone can import and use the same structures|
|1 PM|Highest|GP prior (scikit-learn). Fire engine (Rothermel CA, single-threaded).|GP produces variance field. CA produces arrival times.|
|2 AM|Highest|Ensemble wrapper (multiprocessing). Information field computation.|w_i heatmap looks physically sensible|
|2 PM|High|Greedy selector. Uniform + fire-front baselines.|Greedy selects different locations than uniform|
|3 AM|High|Counterfactual evaluator. Orchestrator loop (batch).|Full cycle runs. Comparison numbers come out.|
|3 PM|High|QUBO construction + SA solver. EnKF update.|QUBO matches or approaches greedy quality|
|4 AM|Medium|D-Wave integration. Multiple cycles with assimilation.|Loop closes. Entropy decreases across cycles.|
|4 PM|Medium|Path-integrated observations. Multiple scenarios.|Targeted > uniform across scenarios|
|5 AM|Medium|Visualization. Presentation.|Demo works|
|5 PM|—|Present|—|

---

## 7. What Success Looks Like

**Minimum viable result:** The information field heatmap is physically sensible (high values at terrain transitions ahead of the fire, low near RAWS, zero behind the fire). Greedy and QUBO both outperform uniform and fire-front on PERR. Entropy decreases across 5+ cycles.

**Strong result:** The above, plus: QUBO and greedy select measurably different locations with QUBO slightly outperforming on at least some scenarios. The drone value curve shows a clear knee. The comparison holds across 2-3 fire scenarios with different terrain complexity.

**Exceptional result:** The above, plus: a less accurate fire model with WISP-targeted data outpredicts a more accurate fire model with only RAWS data. This validates the thesis that wildfire prediction is data-limited, not model-limited.

---

## 8. Code Estimate

| Component                   | Lines    | Dependencies               |
| --------------------------- | -------- | -------------------------- |
| Data types                  | 80       | numpy                      |
| Terrain loader              | 30       | landfire, rasterio         |
| GP prior                    | 60       | scikit-learn               |
| Correlated field generation | 20       | numpy (FFT)                |
| Fire engine (Rothermel CA)  | 130      | numpy (or pytorch for GPU) |
| Information field           | 40       | numpy                      |
| Greedy selector             | 50       | numpy                      |
| QUBO selector               | 100      | dwave-ocean-sdk            |
| Baselines                   | 15       | numpy                      |
| Evaluator                   | 60       | numpy                      |
| EnKF                        | 60       | numpy                      |
| Path planner                | 40       | numpy                      |
| Observation sim             | 20       | numpy                      |
| Orchestrator                | 80       | —                          |
| Visualization               | 80       | matplotlib                 |
| **Total**                   | **~865** |                            |


Potential directory organisation (make it a well organised package up to your discretion): 
ignis/
├── __init__.py          # empty
├── types.py             # shared dataclasses
├── config.py            # constants + fuel params
├── utils.py             # UTM projection, distances, thinning
├── terrain.py           # LANDFIRE loader
├── gp.py                # GP prior + conditional variance
├── fire_engine.py       # Rothermel CA + ensemble
├── information.py       # sensitivity + info field computation
├── selectors/
│   ├── __init__.py
│   ├── greedy.py
│   ├── qubo.py
│   └── baselines.py
├── assimilation.py      # EnKF + observation source
├── orchestrator.py      # cycle loop
├── evaluation.py        # counterfactual comparison
└── visualization.py     # plots

scripts/
├── run_cycle.py         # python -m scripts.run_cycle
└── run_comparison.py    # python -m scripts.run_comparison
