# IGNIS: Fire State Estimation Design Specification

---

## Problem

The ensemble currently starts every member from an identical, deterministic fire front. All ensemble spread comes from FMC/wind uncertainty. Fire location uncertainty — which is substantial between satellite passes — is ignored. This means the system underestimates total prediction uncertainty and never routes drones to confirm fire position, even when perimeter uncertainty is the dominant source of prediction error.

## Solution

Maintain per-member fire states across cycles. Reconstruct the arrival time field from sparse fire observations. Create ensemble diversity in fire state by perturbing the reconstructed arrival times at the uncertain perimeter. Use particle filter reweighting when new fire observations arrive. Hard-reset the ensemble when observations reveal significant drift from the ensemble consensus.

---

## New Components

### 1. FireStateEstimator

Maintains the best estimate of fire arrival time, tracks observation recency, and decides when to hard-reset vs continue.

Lives in: `ignis/fire_state.py`

```python
class FireStateEstimator:
    """
    Manages fire state estimation from sparse observations.
    Produces arrival time fields for ensemble initialization.
    """
    
    def __init__(self, grid_shape, dx, max_arrival):
        self.grid_shape = grid_shape
        self.dx = dx
        self.max_arrival = max_arrival
        
        # Best-estimate arrival time field
        # Initialized to MAX_ARRIVAL everywhere (no fire)
        self.arrival_time = np.full(grid_shape, max_arrival, dtype=np.float32)
        
        # Confidence: how reliable is the arrival time at each cell?
        # 1.0 = directly observed, decays toward 0.0 over time
        # 0.0 = purely model-extrapolated
        self.confidence = np.zeros(grid_shape, dtype=np.float32)
        
        # Last observation time per cell
        self.last_observed = np.full(grid_shape, -np.inf, dtype=np.float32)
        
        # Uncertainty in arrival time (seconds) at each cell
        # Low at observed cells, grows with distance from observations
        # and with time since last observation
        self.arrival_uncertainty = np.full(grid_shape, np.inf, dtype=np.float32)
        
        # Track whether a hard reset occurred this cycle
        self.reset_this_cycle = False
    
    def set_ignition(self, ignition_cell, ignition_time=0.0):
        """Initialize with known ignition point."""
        self.arrival_time[ignition_cell] = ignition_time
        self.confidence[ignition_cell] = 1.0
        self.last_observed[ignition_cell] = ignition_time
        self.arrival_uncertainty[ignition_cell] = 0.0
```

### 2. Arrival Time Reconstruction

When fire observations arrive (satellite pass, drone thermal survey), reconstruct the full arrival time surface using fast marching from observed fire cells with Rothermel ROS.

```python
def reconstruct_arrival_time(self, fire_observations, current_time,
                              terrain, gp_prior):
    """
    Reconstruct a complete arrival time field from sparse fire observations
    using fast marching with Rothermel ROS.
    
    fire_observations: list of FireDetectionObservation
        Each has: location, is_fire, confidence, timestamp
    terrain: TerrainData
    gp_prior: GPPrior (for FMC/wind mean fields used in ROS computation)
    
    Returns: reconstructed arrival_time field (rows, cols)
    """
    rows, cols = self.grid_shape
    arrival = np.full((rows, cols), self.max_arrival, dtype=np.float32)
    
    # Step 1: Anchor arrival times at observed cells
    # Cells confirmed burning: arrival_time ≈ observation time
    # (the fire arrived sometime before the observation)
    for obs in fire_observations:
        r, c = obs.location
        if obs.is_fire:
            # Best estimate: fire arrived at observation time
            # Could refine: fire arrived between last no-fire obs and this obs
            arrival[r, c] = obs.timestamp
            self.confidence[r, c] = obs.confidence
            self.last_observed[r, c] = current_time
    
    # Cells confirmed NOT burning: arrival_time = MAX_ARRIVAL
    for obs in fire_observations:
        r, c = obs.location
        if not obs.is_fire:
            arrival[r, c] = self.max_arrival
            self.confidence[r, c] = obs.confidence
            self.last_observed[r, c] = current_time
    
    # Step 2: Compute ROS at every cell from GP-mean environmental parameters
    ros_field = compute_ros_field(
        terrain=terrain,
        fmc=gp_prior.fmc_mean,
        wind_speed=gp_prior.wind_speed_mean,
        wind_direction=gp_prior.wind_dir_mean
    )
    # ros_field: (rows, cols) in m/s
    
    # Step 3: Fast marching from observed-fire cells
    # Treat observed fire cells as sources. Propagate outward using
    # travel time = distance / ROS at each cell.
    # This fills in arrival times at unobserved cells based on how
    # fast fire would spread through them given current best-estimate
    # environmental conditions.
    arrival = self._fast_march(arrival, ros_field)
    
    # Step 4: Compute arrival time uncertainty
    # Cells directly observed: low uncertainty (observation noise only)
    # Cells near observations: moderate (propagation from nearby observed cells)
    # Cells far from any observation: high (pure model extrapolation)
    self._compute_uncertainty(fire_observations, current_time, ros_field)
    
    # Update stored estimate
    self.arrival_time = arrival
    return arrival

def _fast_march(self, arrival, ros_field):
    """
    Fast marching method to fill arrival times at unobserved cells.
    
    Uses observed cells (where arrival < MAX_ARRIVAL) as sources.
    Propagates outward: for each neighbor, arrival = min(arrival,
    source_arrival + distance / ros).
    
    This is Dijkstra's algorithm on the grid with edge weights = dx / ros.
    """
    import heapq
    
    rows, cols = arrival.shape
    visited = np.zeros_like(arrival, dtype=bool)
    
    # Initialize priority queue with all observed fire cells
    pq = []
    for r in range(rows):
        for c in range(cols):
            if arrival[r, c] < self.max_arrival * 0.9:
                heapq.heappush(pq, (arrival[r, c], r, c))
    
    # 8-connected neighbors with distance factors
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
    ]
    
    while pq:
        t, r, c = heapq.heappop(pq)
        if visited[r, c]:
            continue
        visited[r, c] = True
        arrival[r, c] = t
        
        for dr, dc, dist_factor in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                # Travel time from (r,c) to (nr,nc)
                ros = ros_field[nr, nc]
                if ros < 1e-6:
                    continue  # non-burnable cell
                travel_time = (self.dx * dist_factor) / ros
                new_arrival = t + travel_time
                
                if new_arrival < arrival[nr, nc]:
                    arrival[nr, nc] = new_arrival
                    heapq.heappush(pq, (new_arrival, nr, nc))
    
    return arrival

def _compute_uncertainty(self, fire_observations, current_time, ros_field):
    """
    Compute arrival time uncertainty at each cell.
    
    Uncertainty sources:
    1. Observation noise at directly observed cells (~30s for satellite,
       ~5s for drone thermal)
    2. Propagation uncertainty: grows with distance from nearest observation
       because ROS is uncertain (driven by FMC/wind uncertainty)
    3. Temporal uncertainty: grows with time since last observation
       because fire may have deviated from model prediction
    """
    # Distance to nearest fire observation (in cells)
    obs_mask = np.zeros(self.grid_shape, dtype=bool)
    for obs in fire_observations:
        if obs.is_fire:
            obs_mask[obs.location] = True
    
    if not obs_mask.any():
        # No fire observations — maximum uncertainty everywhere
        self.arrival_uncertainty = np.full(
            self.grid_shape, self.max_arrival, dtype=np.float32)
        return
    
    from scipy.ndimage import distance_transform_edt
    dist_to_obs = distance_transform_edt(~obs_mask) * self.dx  # meters
    
    # Uncertainty grows with distance from observation
    # At the observation: ~30 seconds (satellite temporal resolution)
    # Per 100m from observation: +60 seconds (ROS uncertainty accumulates)
    obs_noise_s = 30.0  # seconds
    propagation_rate = 0.6  # seconds of uncertainty per meter of distance
    self.arrival_uncertainty = obs_noise_s + propagation_rate * dist_to_obs
    
    # Time since observation adds uncertainty (fire may have deviated)
    time_since = current_time - self.last_observed
    time_since = np.clip(time_since, 0, 7200)  # cap at 2 hours
    # Maximum possible deviation: max_ros × time_since
    max_ros = np.percentile(ros_field[ros_field > 0], 95) if (ros_field > 0).any() else 1.0
    temporal_uncertainty = max_ros * time_since  # meters of possible fire advance
    temporal_uncertainty_s = temporal_uncertainty / (max_ros + 1e-6)  # convert back to seconds
    
    self.arrival_uncertainty = np.sqrt(
        self.arrival_uncertainty**2 + temporal_uncertainty_s**2
    ).astype(np.float32)
```

### 3. Ensemble Fire State Manager

Manages per-member fire states across cycles. Handles carry-forward, reinitialization from reconstruction, and resampling after particle filter.

```python
class EnsembleFireState:
    """
    Maintains per-member fire state across IGNIS cycles.
    
    Two modes:
    - CONTINUE: each member carries its fire state forward from previous cycle
    - RESET: all members reinitialize from a reconstructed arrival time field
             with perturbations at the uncertain perimeter
    """
    
    def __init__(self, n_members, grid_shape, dx, max_arrival):
        self.n_members = n_members
        self.grid_shape = grid_shape
        self.dx = dx
        self.max_arrival = max_arrival
        
        # Per-member arrival times from previous cycle
        # None until first cycle completes
        self.member_arrival_times = None
        
        # Per-member phi (SDF) for level-set engine
        self.member_phi = None
        
        # Track whether ensemble has been initialized
        self.initialized = False
    
    def initialize_from_ignition(self, ignition_cell):
        """First cycle: all members start from same ignition point."""
        sdf = self._compute_sdf_from_point(ignition_cell)
        self.member_phi = np.tile(sdf, (self.n_members, 1, 1)).copy()
        self.member_arrival_times = np.full(
            (self.n_members, *self.grid_shape), self.max_arrival, dtype=np.float32)
        self.member_arrival_times[:, ignition_cell[0], ignition_cell[1]] = 0.0
        self.initialized = True
    
    def initialize_from_reconstruction(self, arrival_time_field, 
                                         arrival_uncertainty):
        """
        Hard reset: create fresh ensemble around reconstructed arrival times.
        
        Each member gets the reconstructed arrival time perturbed by
        correlated noise scaled by local uncertainty. Members differ
        at the uncertain perimeter and agree in well-observed regions.
        
        arrival_time_field: (rows, cols) reconstructed arrival times
        arrival_uncertainty: (rows, cols) uncertainty in seconds per cell
        """
        self.member_arrival_times = np.zeros(
            (self.n_members, *self.grid_shape), dtype=np.float32)
        
        for n in range(self.n_members):
            # Correlated noise field (unit variance, spatial structure)
            noise = draw_correlated_field(
                self.grid_shape,
                correlation_length=500.0,  # meters — fire front correlation
                resolution=self.dx
            )
            
            # Scale by local uncertainty
            # Well-observed cells: near-zero perturbation
            # Uncertain perimeter cells: perturbation of ±arrival_uncertainty seconds
            perturbation = noise * arrival_uncertainty
            
            member_arrival = arrival_time_field + perturbation
            
            # Clamp: don't perturb deep interior or far exterior
            # Cells that burned long ago (>1 hour before current time) stay fixed
            deep_burned = arrival_time_field < (arrival_time_field.min() + 3600)
            member_arrival[deep_burned] = arrival_time_field[deep_burned]
            
            # Cells that are definitely unburned stay MAX_ARRIVAL
            far_exterior = arrival_time_field > self.max_arrival * 0.9
            member_arrival[far_exterior] = self.max_arrival
            
            self.member_arrival_times[n] = member_arrival
        
        # Convert to SDF for level-set engine
        self._recompute_phi_from_arrival_times(current_time=arrival_time_field.max())
        self.initialized = True
    
    def carry_forward(self, ensemble_result):
        """
        Continue mode: store each member's fire state from the completed cycle.
        Called after each cycle's ensemble run.
        """
        self.member_arrival_times = ensemble_result.member_arrival_times.copy()
    
    def get_initial_phi(self, current_time):
        """
        Return per-member SDF for the fire engine.
        Shape: (N, rows, cols) — each member has a different fire front.
        """
        self._recompute_phi_from_arrival_times(current_time)
        return self.member_phi
    
    def resample(self, indices):
        """
        After particle filter reweighting, resample members.
        indices: (N,) array of member indices to keep (with repetition).
        """
        self.member_arrival_times = self.member_arrival_times[indices]
        if self.member_phi is not None:
            self.member_phi = self.member_phi[indices]
    
    def _recompute_phi_from_arrival_times(self, current_time):
        """Convert arrival times to SDF for each member."""
        from scipy.ndimage import distance_transform_edt
        
        self.member_phi = np.zeros(
            (self.n_members, *self.grid_shape), dtype=np.float32)
        
        for n in range(self.n_members):
            burned = self.member_arrival_times[n] < current_time
            if not burned.any():
                self.member_phi[n] = np.ones(self.grid_shape) * self.dx
                continue
            if burned.all():
                self.member_phi[n] = np.ones(self.grid_shape) * -self.dx
                continue
            d_in = distance_transform_edt(burned) * self.dx
            d_out = distance_transform_edt(~burned) * self.dx
            self.member_phi[n] = np.where(burned, -d_in, d_out)
    
    def _compute_sdf_from_point(self, cell):
        """SDF for a single ignition point."""
        rows, cols = self.grid_shape
        y, x = np.mgrid[0:rows, 0:cols]
        dist = np.sqrt((y - cell[0])**2 + (x - cell[1])**2) * self.dx
        ignition_radius = 2.0 * self.dx
        return (dist - ignition_radius).astype(np.float32)
```

### 4. Consistency Checker

Decides whether to hard-reset or continue based on how well the ensemble matches new fire observations.

```python
class ConsistencyChecker:
    """
    Compares new fire observations against ensemble consensus.
    Triggers hard reset when disagreement exceeds threshold.
    """
    
    def __init__(self, disagreement_threshold=0.2, min_observations=5):
        self.disagreement_threshold = disagreement_threshold
        self.min_observations = min_observations
    
    def check(self, fire_observations, ensemble_result, 
              current_time) -> tuple[bool, float]:
        """
        Compare fire observations against ensemble.
        
        Returns: (should_reset, disagreement_fraction)
        
        should_reset = True if the ensemble has drifted significantly
        from observed fire state.
        """
        if len(fire_observations) < self.min_observations:
            return False, 0.0
        
        # Ensemble consensus: P(burned) at each observed cell
        ensemble_burned = (
            ensemble_result.member_arrival_times < current_time
        )  # (N, rows, cols) bool
        burn_probability = ensemble_burned.mean(axis=0)  # (rows, cols)
        
        disagreements = 0
        total = 0
        
        for obs in fire_observations:
            r, c = obs.location
            p_burned = burn_probability[r, c]
            
            if obs.is_fire and p_burned < 0.3:
                # Observation says fire, ensemble says probably no fire
                disagreements += 1
            elif not obs.is_fire and p_burned > 0.7:
                # Observation says no fire, ensemble says probably fire
                disagreements += 1
            total += 1
        
        disagreement_fraction = disagreements / total if total > 0 else 0.0
        should_reset = disagreement_fraction > self.disagreement_threshold
        
        return should_reset, disagreement_fraction
```

### 5. Particle Filter Reweighting

For individual fire observations (not a full satellite pass), reweight ensemble members based on consistency with observations.

```python
def particle_filter_fire(ensemble_result, fire_observations, 
                          current_time, n_members):
    """
    Reweight and resample ensemble members based on fire observations.
    
    Members whose fire state is consistent with observations get higher weight.
    Members that disagree get downweighted.
    
    Returns: resampled indices (N,)
    """
    weights = np.ones(n_members, dtype=np.float64)
    
    ensemble_burned = (
        ensemble_result.member_arrival_times < current_time
    )  # (N, rows, cols)
    
    for obs in fire_observations:
        r, c = obs.location
        
        for n in range(n_members):
            member_burned = ensemble_burned[n, r, c]
            
            if obs.is_fire:
                if member_burned:
                    weights[n] *= obs.confidence
                else:
                    weights[n] *= (1.0 - obs.confidence)
            else:  # no fire observed
                if not member_burned:
                    weights[n] *= obs.confidence
                else:
                    weights[n] *= (1.0 - obs.confidence)
    
    # Normalize
    weights /= weights.sum() + 1e-30
    
    # Effective sample size: measure of ensemble diversity after reweighting
    n_eff = 1.0 / (weights**2).sum()
    
    # Only resample if effective sample size drops below threshold
    # (avoids unnecessary resampling that destroys diversity)
    if n_eff < n_members * 0.5:
        indices = systematic_resample(weights, n_members)
    else:
        indices = np.arange(n_members)  # no resampling needed
    
    return indices, n_eff

def systematic_resample(weights, n):
    """Standard particle filter systematic resampling."""
    positions = (np.arange(n) + np.random.uniform()) / n
    cumsum = np.cumsum(weights)
    return np.searchsorted(cumsum, positions)
```

---

## Integration with Orchestrator

### Modified Cycle Logic

```python
class Orchestrator:
    def __init__(self, ...):
        # ... existing init ...
        self.fire_state_estimator = FireStateEstimator(
            grid_shape, dx, max_arrival)
        self.ensemble_fire_state = EnsembleFireState(
            n_members, grid_shape, dx, max_arrival)
        self.consistency_checker = ConsistencyChecker(
            disagreement_threshold=0.2, min_observations=5)
        
        self._last_reset_time = 0.0
    
    def run_cycle(self):
        self.obs_store.lock_for_cycle()
        try:
            # 1. GP prior (FMC, wind — unchanged)
            gp_prior = self.gp.predict(self.shape)
            
            # 2. Check for new fire observations
            fire_obs = self.obs_store.get_fire_detections(
                since=self._last_cycle_time)
            
            # 3. Decide: hard reset or continue?
            if fire_obs and self.ensemble_fire_state.initialized:
                should_reset, disagreement = self.consistency_checker.check(
                    fire_obs, self._last_ensemble_result, self.current_time)
                
                if should_reset:
                    # Reconstruct arrival time from observations
                    arrival_field = self.fire_state_estimator.reconstruct_arrival_time(
                        fire_obs, self.current_time, self.terrain, gp_prior)
                    
                    # Create fresh ensemble around reconstruction
                    self.ensemble_fire_state.initialize_from_reconstruction(
                        arrival_field, 
                        self.fire_state_estimator.arrival_uncertainty)
                    
                    self._last_reset_time = self.current_time
                    log.info(f"Hard reset: {disagreement:.0%} disagreement, "
                             f"{len(fire_obs)} observations")
                else:
                    # Particle filter reweight with individual observations
                    indices, n_eff = particle_filter_fire(
                        self._last_ensemble_result, fire_obs,
                        self.current_time, self.n_members)
                    self.ensemble_fire_state.resample(indices)
                    log.info(f"Particle filter: {disagreement:.0%} disagreement, "
                             f"N_eff={n_eff:.0f}")
            
            # 4. Get per-member initial fire state
            initial_phi = self.ensemble_fire_state.get_initial_phi(
                self.current_time)
            
            # 5. Run ensemble (per-member phi + GP-sampled FMC/wind)
            ensemble_result = self.fire_engine.run(
                initial_phi=initial_phi,  # (N, rows, cols) — per member
                gp_prior=gp_prior,
                config=self.ensemble_config
            )
            
            # 6. Carry forward fire state for next cycle
            self.ensemble_fire_state.carry_forward(ensemble_result)
            self._last_ensemble_result = ensemble_result
            
            # 7. Information field, selection, etc. (unchanged)
            info_field = compute_information_field(ensemble_result, gp_prior)
            # ... rest of cycle ...
            
        finally:
            self.obs_store.unlock_cycle()
```

### Fire Engine Interface Change

The fire engine must accept per-member initial phi instead of broadcasting a single SDF:

```python
# BEFORE (current):
phi = torch.tensor(sdf).unsqueeze(0).expand(n_members, -1, -1).clone()

# AFTER:
# initial_phi is already (N, rows, cols) — one SDF per member
phi = torch.tensor(initial_phi, device=self.device, dtype=torch.float32)
```

This is a one-line change in the fire engine. The `initial_phi` parameter replaces the internally-computed SDF. All downstream computation (ROS, level-set update, CFL) is unchanged — it already operates on (N, rows, cols) tensors.

---

## Data Flow

```
Fire observations (satellite/drone thermal)
        │
        ▼
ConsistencyChecker
  compare obs vs ensemble consensus
        │
        ├── disagreement > 20%: HARD RESET
        │       │
        │       ▼
        │   FireStateEstimator.reconstruct_arrival_time()
        │     fast march from observed fire cells using GP-mean ROS
        │       │
        │       ▼
        │   EnsembleFireState.initialize_from_reconstruction()
        │     perturb arrival times at uncertain boundary
        │     each member gets different fire front
        │       │
        │       ▼
        │   Fresh ensemble with fire state diversity
        │
        └── disagreement ≤ 20%: CONTINUE
                │
                ├── Individual fire obs: particle_filter_fire()
                │     reweight members, resample if N_eff drops
                │
                └── No fire obs: carry forward unchanged
                        │
                        ▼
                EnsembleFireState.get_initial_phi()
                  per-member SDF from carried/reset/resampled state
                        │
                        ▼
                Fire engine runs with per-member phi + GP-sampled FMC/wind
                        │
                        ▼
                EnsembleFireState.carry_forward(result)
                  store each member's fire state for next cycle
```

---

## What Changes in the Information Field

With per-member fire states, the ensemble now disagrees about where the fire currently IS. This creates a new source of high w_i:

```python
# Burn probability from per-member fire states
burn_prob = (ensemble.member_arrival_times < current_time).mean(axis=0)

# Binary entropy: high where members disagree about current fire state
fire_entropy = binary_entropy(burn_prob)

# Information field now includes fire state uncertainty
w_total = w_fmc + w_wind_speed + w_wind_dir + alpha * fire_entropy
```

Cells at the uncertain fire boundary have high `fire_entropy`. A drone with a thermal camera flying over these cells resolves the binary question (fire or no fire), which triggers particle filter reweighting that kills inconsistent members. This is the mechanism for routing drones to confirm fire position when that's the most valuable measurement.

The system now automatically trades off: "Should I send this drone to measure FMC 5 km ahead (prognostic value) or confirm the fire perimeter at this ridge (diagnostic value)?" The information field weights both based on their respective uncertainties and sensitivities.

---

## Interaction with Existing Architecture

### ObservationStore

FireDetectionObservation objects are already stored. The new code reads them via `obs_store.get_fire_detections(since=...)`. No change to the store.

### GP Prior

Unchanged. The GP models FMC and wind, not fire state. The reconstruction uses GP-mean fields to compute ROS for the fast march, but doesn't modify the GP.

### EnKF

The standard EnKF still handles FMC/wind assimilation. Fire state assimilation uses the particle filter, not the EnKF. These are separate mechanisms on separate state variables, running in the same cycle.

### Fire Engine

One interface change: accept `initial_phi: (N, rows, cols)` instead of computing and broadcasting internally. Everything else unchanged.

---

## Computational Cost

|Component|Cost|When|
|---|---|---|
|Fast marching (Dijkstra on grid)|O(D log D), ~50ms at 38K cells|On hard reset only|
|Uncertainty computation|O(D) distance transform, ~5ms|On hard reset only|
|Ensemble perturbation (N correlated fields)|N × O(D log D) FFT, ~200ms for N=200|On hard reset only|
|SDF recomputation from arrival times|N × O(D) distance transform, ~1s for N=200|Every cycle|
|Consistency check|O(n_obs × N), <1ms|Every cycle with fire obs|
|Particle filter reweighting|O(n_obs × N), <1ms|Every cycle with fire obs|

The SDF recomputation (N distance transforms per cycle) is the most expensive addition. At N=200 and D=38K, each distance_transform_edt takes ~5ms, so 200 transforms take ~1 second. This is noticeable but within budget.

**Optimization:** Only recompute SDF for members whose fire state actually changed. In carry-forward mode, only the fire-front cells changed (interior and far exterior are identical). Compute SDF only in a narrow band around each member's fire front. This reduces from O(N × D) to O(N × perimeter_length), which is typically 10-100× cheaper.

---

## Testing

|Test|What it validates|Expected|
|---|---|---|
|Fast march from single point, flat terrain, uniform ROS|Reconstruction produces circular fire|Arrival times form concentric circles|
|Fast march from scattered observations|Reconstruction fills gaps between observations|Smooth arrival time surface connecting observed points|
|Ensemble initialization: all members agree at observed cells|Perturbation is zero where uncertainty is zero|arrival_times[:, observed_r, observed_c] identical across members|
|Ensemble initialization: members disagree at uncertain perimeter|Perturbation is large where uncertainty is large|std across members > 0 at perimeter cells|
|Consistency check: ensemble matches observations|Returns should_reset=False|disagreement < threshold|
|Consistency check: ensemble contradicts observations|Returns should_reset=True|disagreement > threshold|
|Particle filter: observation kills inconsistent members|After resampling, all members agree at observed cell|burn state at observed cell is unanimous|
|Full cycle: hard reset then forward propagation|Fire advances from reconstructed state|Arrival times extend beyond reconstruction|
|Full cycle: continue then particle filter|Incremental correction without full reset|Small adjustments, ensemble diversity preserved|

---

## Files to Create/Modify

|File|Action|Scope|
|---|---|---|
|`ignis/fire_state.py`|CREATE|FireStateEstimator, EnsembleFireState, ConsistencyChecker, particle_filter_fire|
|`ignis/fire_engine.py` or `ignis/simulation/gpu_fire_engine.py`|MODIFY|Accept initial_phi as (N, rows, cols) parameter instead of computing internally|
|`ignis/orchestrator.py`|MODIFY|Add fire state management to cycle logic (steps 2-4 above)|
|`ignis/information.py`|MODIFY|Add fire_entropy term to information field (3 lines)|
|`ignis/tests/test_fire_state.py`|CREATE|All tests listed above|

---

## Known Limitations

**Fast march assumes static ROS.** The reconstruction computes ROS from current GP-mean fields and marches using those values. In reality, FMC and wind changed as the fire spread — the ROS at cell X when the fire arrived there hours ago was different from the current ROS at cell X. Mandel's variational approach accounts for this by incorporating time-varying ROS into the reconstruction. The fast march is an approximation that works well when conditions haven't changed dramatically since the fire started. For multi-day fires with diurnal cycles, this approximation degrades.

**FMC/wind inconsistency after hard reset.** The reconstruction used GP-mean parameters. After ensemble perturbation, each member has different FMC/wind. The fire state is momentarily inconsistent with the member's environmental parameters. This self-corrects within one forward propagation but introduces a short transient. Acceptable for the hackathon.

**Satellite observation temporal ambiguity.** A VIIRS detection says "this cell is burning at observation time." But the fire arrived at that cell sometime between the last no-fire observation and now. The reconstruction assigns arrival_time = observation_time, which is an upper bound. The true arrival could have been hours earlier. This introduces systematic bias in the reconstructed arrival time surface. The uncertainty field accounts for this partially but doesn't fully resolve it.