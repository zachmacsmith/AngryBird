# GPU Fire Engine: Implementation Specification

Target: PyTorch CUDA implementation of Rothermel surface fire + Van Wagner crown fire + level-set propagation + CFL adaptive stepping + ensemble batching.

---

## 1. Interface Contract

```python
class GPUFireEngine:
    def __init__(self, terrain: TerrainData, device: str = "cuda"):
        """Load terrain data to GPU once. Reuse across cycles."""
        ...
    
    def run(self, snapshot: CycleSnapshot, config: EnsembleConfig) -> EnsembleResult:
        """Run N-member ensemble. Returns numpy arrays."""
        ...
```

Input: perturbed FMC and wind fields as numpy arrays (N, rows, cols). Output: EnsembleResult with numpy arrays (N, rows, cols) for arrival times, plus burn probability, variance, etc.

All GPU allocation and computation is internal. The boundary is numpy in, numpy out.

---

## 2. Memory Layout

### 2.1 Terrain Tensors (loaded once, shared across all members)

```python
# All shape: (rows, cols), dtype: float32, device: cuda
self.elevation     # meters
self.slope         # radians (convert from degrees at load time)
self.aspect        # radians
self.fuel_model    # int32 — index into fuel parameter table
self.cbh           # canopy base height, meters
self.cbd           # canopy bulk density, kg/m³
self.canopy_cover  # fraction 0-1

# Derived at load time:
self.tan_slope_sq  # tan(slope)² — precomputed, used every timestep
self.slope_factor  # 5.275 * beta^(-0.3) * tan(slope)² per cell — precomputed if beta is per-fuel-model
```

### 2.2 Fuel Parameter Table (small, constant)

Anderson 13 fuel models as a 2D tensor. Each row is one fuel model. Columns are Rothermel parameters.

```python
# Shape: (14, N_PARAMS), dtype: float32, device: cuda
# Row 0 is unused (no fuel model 0). Rows 1-13 are Anderson models.
# Index with fuel_model tensor directly.

FUEL_COLS = {
    "w0": 0,       # fuel load, kg/m²
    "sigma": 1,    # surface-area-to-volume ratio, 1/m
    "delta": 2,    # fuel bed depth, m
    "mx": 3,       # moisture of extinction, fraction
    "heat": 4,     # heat content, kJ/kg
    "rho_p": 5,    # particle density, kg/m³ (constant 513 for all)
    "st": 6,       # total mineral content, fraction (0.0555)
    "se": 7,       # effective mineral content, fraction (0.010)
}

# Example: fuel model 1 (short grass)
# [0.034, 3500, 0.305, 0.12, 18607, 513, 0.0555, 0.010]
```

Lookup is a gather operation: `params = fuel_table[fuel_model_grid]` produces a (rows, cols, N_PARAMS) tensor.

### 2.3 Per-Ensemble-Member State (batched)

```python
# All shape: (N, rows, cols), dtype: float32, device: cuda
self.fmc           # 1-hr dead fuel moisture, fraction
self.wind_speed    # m/s at 10m
self.wind_direction # degrees

# Fire state per member:
self.phi           # level-set function: <0 burned, >0 unburned
self.arrival_time  # seconds since ignition, MAX_ARRIVAL if unburned
self.intensity     # fireline intensity at each cell, kW/m (for crown fire check)
self.fire_type     # 0 = unburned, 1 = surface, 2 = crown
```

Total GPU memory per member: 7 tensors × 40,000 cells × 4 bytes = 1.12 MB. For 1000 members: 1.12 GB. Well within 8-16 GB GPU memory.

### 2.4 Working Tensors (allocated once, reused each timestep)

```python
# Shape: (N, rows, cols)
self.ros           # rate of spread, m/s
self.ros_head      # head fire ROS (max, in wind direction)
self.ros_flank     # flank fire ROS
self.ros_back      # backing fire ROS
self.dphi_dt       # level-set time derivative
self.max_speed     # per-member max wavespeed for CFL (shape: (N,))
```

---

## 3. Rothermel Surface ROS Kernel

All operations are elementwise across (N, rows, cols). One kernel launch computes ROS for all cells of all members simultaneously.

### 3.1 Fuel Parameter Gather

```python
def gather_fuel_params(fuel_model, fuel_table):
    """
    fuel_model: int32 (rows, cols)
    fuel_table: float32 (14, N_PARAMS)
    Returns: float32 (rows, cols, N_PARAMS)
    """
    return fuel_table[fuel_model.long()]  # PyTorch advanced indexing

# Broadcast to ensemble: (rows, cols, N_PARAMS) → works with (N, rows, cols) state
# via unsqueeze and expand
```

### 3.2 Rothermel Equation

```python
def rothermel_ros(fmc, wind_speed, fuel_params, tan_slope_sq):
    """
    All inputs: (N, rows, cols) except fuel_params (rows, cols, N_PARAMS)
    fuel_params is broadcast over N via unsqueeze(0).
    Returns: ros (N, rows, cols) in m/s
    """
    # Extract fuel parameters — all (rows, cols), broadcast over N
    w0    = fuel_params[..., 0]   # fuel load, kg/m²
    sigma = fuel_params[..., 1]   # SAV ratio, 1/m
    delta = fuel_params[..., 2]   # fuel bed depth, m
    mx    = fuel_params[..., 3]   # moisture of extinction, fraction
    heat  = fuel_params[..., 4]   # heat content, kJ/kg
    rho_p = fuel_params[..., 5]   # particle density, kg/m³
    st    = fuel_params[..., 6]   # total mineral content
    se    = fuel_params[..., 7]   # effective mineral content
    
    # Packing ratio
    beta = w0 / (delta * rho_p + 1e-10)
    beta_op = 3.348 * sigma.pow(-0.8189)
    beta_ratio = beta / (beta_op + 1e-10)
    
    # Optimum reaction velocity (1/min)
    A = 133.0 * sigma.pow(-0.7913)
    gamma_max = sigma.pow(1.5) / (495.0 + 0.0594 * sigma.pow(1.5))
    gamma = gamma_max * beta_ratio.pow(A) * torch.exp(A * (1.0 - beta_ratio))
    
    # Net fuel load
    wn = w0 * (1.0 - st)
    
    # Mineral damping
    eta_s = 0.174 * se.pow(-0.19)
    eta_s = torch.clamp(eta_s, max=1.0)
    
    # Moisture damping
    rm = fmc / (mx + 1e-10)
    eta_m = 1.0 - 2.59 * rm + 5.11 * rm.pow(2) - 3.52 * rm.pow(3)
    eta_m = torch.clamp(eta_m, min=0.0, max=1.0)
    
    # Reaction intensity (kW/m²)
    I_R = gamma * wn * heat * eta_m * eta_s
    
    # Propagating flux ratio
    xi = torch.exp((0.792 + 0.681 * sigma.pow(0.5)) * (beta + 0.1)) \
         / (192.0 + 0.2595 * sigma)
    
    # Wind factor
    C = 7.47 * torch.exp(-0.133 * sigma.pow(0.55))
    B = 0.02526 * sigma.pow(0.54)
    E = 0.715 * torch.exp(-3.59e-4 * sigma)
    # Wind speed in m/min for Rothermel (input is m/s)
    ws_mmin = wind_speed * 60.0
    # 10m wind to midflame: divide by wind adjustment factor
    # Simplified: midflame ≈ 0.4 × 10m for timber, 0.6 for grass
    # Use 0.5 as default, or compute from canopy cover
    ws_midflame = ws_mmin * 0.5
    phi_w = C * ws_midflame.pow(B) * beta_ratio.pow(-E)
    
    # Slope factor
    phi_s = 5.275 * beta.pow(-0.3) * tan_slope_sq
    
    # Heat of preignition (kJ/kg)
    Q_ig = 250.0 + 1116.0 * fmc
    
    # Effective heating number
    epsilon = torch.exp(-138.0 / (sigma + 1e-10))
    
    # Bulk density
    rho_b = w0 / (delta + 1e-10)
    
    # Rate of spread (m/min)
    ros_mmin = (I_R * xi * (1.0 + phi_w + phi_s)) / (rho_b * epsilon * Q_ig + 1e-10)
    
    # Convert to m/s
    ros = ros_mmin / 60.0
    ros = torch.clamp(ros, min=0.0)
    
    return ros, I_R
```

### 3.3 Fireline Intensity

```python
def fireline_intensity(ros, I_R, w0):
    """
    Byram's fireline intensity (kW/m).
    ros: m/s
    I_R: kW/m² (reaction intensity)
    w0: kg/m² (fuel load)
    """
    heat_content = 18000.0  # kJ/kg
    # I_fire = heat_content * w0 * ros
    return heat_content * w0 * ros
```

---

## 4. Crown Fire Kernel

Runs after surface ROS computation. Checks Van Wagner criterion at every cell of every member. Where crown fire initiates, overrides ROS with crown fire value.

```python
def crown_fire_check(surface_intensity, cbh, cbd, wind_speed, ros_surface):
    """
    All inputs: (N, rows, cols)
    cbh, cbd: (rows, cols), broadcast over N
    Returns: ros_final (N, rows, cols), fire_type (N, rows, cols)
    """
    # Van Wagner 1977: critical intensity for crown fire initiation
    fmc_foliar = 1.0  # fraction, typical value
    # I_critical = (0.01 * cbh * (460 + 25.9 * FMC_foliar_pct))^1.5
    I_critical = (0.01 * cbh * (460.0 + 25.9 * fmc_foliar * 100.0)).pow(1.5)
    
    # Does surface intensity exceed critical?
    initiates = surface_intensity > I_critical
    
    # Rothermel 1991 crown fire ROS
    # ROS_crown = 11.02 * wind_kmh^0.854 * cbd^0.19  (m/min)
    wind_kmh = wind_speed * 3.6
    ros_crown_mmin = 11.02 * wind_kmh.pow(0.854) * cbd.pow(0.19)
    ros_crown = ros_crown_mmin / 60.0  # m/s
    
    # Final ROS: max of surface and crown where initiated
    ros_final = torch.where(initiates, torch.maximum(ros_surface, ros_crown), ros_surface)
    
    # Fire type: 0 = unburned (handled elsewhere), 1 = surface, 2 = crown
    fire_type = torch.where(initiates, 
                           torch.full_like(ros_surface, 2),
                           torch.full_like(ros_surface, 1))
    
    return ros_final, fire_type
```

---

## 5. Directional ROS (Elliptical Spread Model)

Rothermel gives head fire ROS (maximum, in wind direction). Fire spreads in an ellipse:

```python
def directional_ros(ros_head, wind_direction, aspect):
    """
    Compute ROS in 8 cardinal/intercardinal directions relative to each cell.
    
    ros_head: (N, rows, cols) — maximum ROS
    wind_direction: (N, rows, cols) — degrees, direction wind blows FROM
    
    Returns: ros_8dir (N, 8, rows, cols) — ROS toward each neighbor
    """
    # Length-to-breadth ratio (Anderson 1983)
    # LB = 0.936 * exp(0.2566 * wind_speed_kmh) + 0.461 * exp(-0.1548 * wind_speed_kmh) - 0.397
    # Simplified: use effective wind speed to compute eccentricity
    # For now, use fixed LB from wind speed (precomputed)
    
    # Eccentricity of the spread ellipse
    # LB = (1 + e) / (1 - e), so e = (LB - 1) / (LB + 1)
    # Backing ROS = ros_head * (1 - e) / (1 + e)
    
    LB = 1.0 + 0.25 * wind_speed_kmh  # simplified linear approximation
    LB = torch.clamp(LB, min=1.0, max=8.0)
    eccentricity = (LB - 1.0) / (LB + 1.0)
    
    # 8 neighbor directions (degrees, clockwise from north)
    # N=0, NE=45, E=90, SE=135, S=180, SW=225, W=270, NW=315
    neighbor_angles = torch.tensor([0, 45, 90, 135, 180, 225, 270, 315],
                                    device=ros_head.device, dtype=torch.float32)
    
    # Angle between spread direction and each neighbor
    # Wind blows FROM wind_direction, fire spreads in direction (wind_direction + 180)
    spread_dir = (wind_direction + 180.0) % 360.0  # (N, rows, cols)
    
    # For each neighbor direction, compute angle difference
    # theta = angle between neighbor direction and head fire direction
    # Shape manipulation: neighbor_angles is (8,), spread_dir is (N, rows, cols)
    # Result: (N, 8, rows, cols)
    spread_dir_expanded = spread_dir.unsqueeze(1)  # (N, 1, rows, cols)
    angles_expanded = neighbor_angles.view(1, 8, 1, 1)  # (1, 8, 1, 1)
    theta = torch.abs(angles_expanded - spread_dir_expanded)
    theta = torch.minimum(theta, 360.0 - theta)  # shortest angular distance
    theta_rad = theta * (3.14159 / 180.0)
    
    # Elliptical ROS model: ROS(theta) = ros_head * (1 - e) / (1 - e * cos(theta))
    ros_head_expanded = ros_head.unsqueeze(1)  # (N, 1, rows, cols)
    ecc_expanded = eccentricity.unsqueeze(1)    # (N, 1, rows, cols)
    
    ros_8 = ros_head_expanded * (1.0 - ecc_expanded) / (1.0 - ecc_expanded * torch.cos(theta_rad) + 1e-10)
    
    return ros_8  # (N, 8, rows, cols)
```

---

## 6. Level-Set Propagation

The fire front is tracked as a level-set function φ where φ < 0 means burned and φ > 0 means unburned. The front is at φ = 0.

### 6.1 Spatial Gradients

```python
def compute_phi_gradient(phi, dx):
    """
    Upwind finite differences for |∇φ|.
    phi: (N, rows, cols)
    dx: grid spacing in meters (scalar)
    Returns: grad_mag (N, rows, cols)
    """
    # Forward and backward differences with zero-padding at boundaries
    # x-direction (columns)
    phi_xp = torch.roll(phi, -1, dims=2)  # phi(i, j+1)
    phi_xm = torch.roll(phi, 1, dims=2)   # phi(i, j-1)
    
    # y-direction (rows)
    phi_yp = torch.roll(phi, -1, dims=1)  # phi(i+1, j)
    phi_ym = torch.roll(phi, 1, dims=1)   # phi(i-1, j)
    
    # Zero-flux boundary conditions
    phi_xp[:, :, -1] = phi[:, :, -1]
    phi_xm[:, :, 0] = phi[:, :, 0]
    phi_yp[:, -1, :] = phi[:, -1, :]
    phi_ym[:, 0, :] = phi[:, 0, :]
    
    # Upwind scheme (Godunov): select appropriate one-sided difference
    # based on sign of speed (always positive for fire — fire always expands)
    Dxm = (phi - phi_xm) / dx   # backward difference, x
    Dxp = (phi_xp - phi) / dx   # forward difference, x
    Dym = (phi - phi_ym) / dx   # backward difference, y
    Dyp = (phi_yp - phi) / dx   # forward difference, y
    
    # Godunov upwind for expanding front (speed > 0):
    # Use max(Dxm, 0)² + min(Dxp, 0)² for x-component
    grad_x_sq = torch.maximum(Dxm, torch.zeros_like(Dxm)).pow(2) + \
                torch.minimum(Dxp, torch.zeros_like(Dxp)).pow(2)
    grad_y_sq = torch.maximum(Dym, torch.zeros_like(Dym)).pow(2) + \
                torch.minimum(Dyp, torch.zeros_like(Dyp)).pow(2)
    
    grad_mag = torch.sqrt(grad_x_sq + grad_y_sq + 1e-10)
    
    return grad_mag
```

### 6.2 Level-Set Update

```python
def level_set_step(phi, ros, dx, dt):
    """
    Advance level set by one timestep.
    dφ/dt + F|∇φ| = 0, where F = ROS (speed of the front)
    
    phi: (N, rows, cols)
    ros: (N, rows, cols) — speed at each cell (directionally averaged or head-fire)
    dx: grid spacing, meters
    dt: timestep, seconds
    Returns: phi_new (N, rows, cols)
    """
    grad_mag = compute_phi_gradient(phi, dx)
    
    # Level-set equation: φ_new = φ - dt * F * |∇φ|
    phi_new = phi - dt * ros * grad_mag
    
    return phi_new
```

**Note on directional ROS:** The basic level-set above uses a scalar ROS per cell. For the elliptical spread model, you need to project the directional ROS onto the gradient direction. The more accurate approach:

```python
def level_set_step_directional(phi, ros_8, dx, dt):
    """
    Level-set with directional speed.
    For each cell, ROS depends on the direction the front is locally moving.
    """
    # Compute gradient components (not just magnitude)
    grad_x = central_diff_x(phi, dx)  # (N, rows, cols)
    grad_y = central_diff_y(phi, dx)  # (N, rows, cols)
    
    # Front propagation direction at each cell
    # (normal to the level set, pointing outward = into unburned fuel)
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-10)
    normal_x = grad_x / grad_mag
    normal_y = grad_y / grad_mag
    
    # Angle of propagation direction
    prop_angle = torch.atan2(normal_x, normal_y)  # radians, 0=north
    prop_angle_deg = (prop_angle * 180.0 / 3.14159) % 360.0
    
    # Interpolate ROS from the 8 directional values
    # ros_8: (N, 8, rows, cols), neighbor_angles: [0,45,90,...,315]
    # Find which two of the 8 directions bracket the propagation angle
    # and linearly interpolate
    sector = prop_angle_deg / 45.0  # 0-8 continuous
    sector_lo = sector.long() % 8
    sector_hi = (sector_lo + 1) % 8
    frac = sector - sector.float().floor()
    
    # Gather from ros_8 using sector indices
    ros_lo = torch.gather(ros_8, 1, sector_lo.unsqueeze(1)).squeeze(1)
    ros_hi = torch.gather(ros_8, 1, sector_hi.unsqueeze(1)).squeeze(1)
    ros_interp = ros_lo * (1.0 - frac) + ros_hi * frac
    
    # Apply upwind gradient magnitude (not the central diff used for direction)
    grad_mag_upwind = compute_phi_gradient(phi, dx)
    
    phi_new = phi - dt * ros_interp * grad_mag_upwind
    return phi_new
```

For the hackathon, start with the scalar (non-directional) level-set. Add directional ROS interpolation as a refinement. The scalar version already captures terrain and FMC effects on spread speed — it just doesn't capture the elliptical shape. The elliptical shape matters for asymmetric spread under wind, but the scalar version still produces reasonable fire perimeters.

---

## 7. CFL Adaptive Timestepping

The level-set is numerically stable only when the CFL condition is satisfied: dt ≤ dx / max(ROS). If any cell has a very high ROS (crown fire at 50+ m/min), dt must shrink to prevent the front from jumping over cells.

```python
def compute_dt(ros, dx, target_cfl=0.4):
    """
    Compute stable timestep from CFL condition.
    ros: (N, rows, cols)
    dx: grid spacing, meters
    target_cfl: safety factor (0.4 = conservative, 0.8 = aggressive)
    Returns: dt in seconds (scalar — same for all members)
    """
    # Max ROS across ALL members and ALL cells
    max_ros = ros.max().item()  # scalar — requires GPU→CPU sync
    
    if max_ros < 1e-10:
        return 300.0  # no fire spreading, use max dt
    
    dt = target_cfl * dx / max_ros
    
    # Clamp to reasonable range
    dt = max(0.1, min(dt, 300.0))  # 0.1s to 5min
    
    return dt
```

**GPU sync note:** `ros.max().item()` triggers a GPU-to-CPU synchronization (the GPU computes the max, then transfers the scalar to CPU). This is ~10 μs overhead per timestep. Unavoidable because the CPU needs the scalar dt to schedule the next kernel launch. At 500 timesteps, total sync overhead is ~5 ms. Negligible.

**Adaptive stepping loop:**

```python
def run_simulation(phi, fmc, wind_speed, wind_direction, 
                   fuel_params, terrain, total_time, dx, target_cfl=0.4):
    """
    Run from t=0 to total_time with adaptive dt.
    """
    t = 0.0
    arrival_time = torch.full_like(phi, MAX_ARRIVAL)
    
    while t < total_time:
        # 1. Compute ROS at all cells of all members
        ros_surface, I_R = rothermel_ros(fmc, wind_speed, fuel_params, terrain.tan_slope_sq)
        
        # 2. Fireline intensity
        intensity = fireline_intensity(ros_surface, I_R, fuel_params[..., 0])
        
        # 3. Crown fire check
        ros_final, fire_type = crown_fire_check(
            intensity, terrain.cbh, terrain.cbd, wind_speed, ros_surface)
        
        # 4. CFL timestep
        dt = compute_dt(ros_final, dx, target_cfl)
        dt = min(dt, total_time - t)  # don't overshoot
        
        # 5. Level-set update
        phi_old = phi.clone()
        phi = level_set_step(phi, ros_final, dx, dt)
        
        # 6. Record arrival times
        # Cells that just crossed φ = 0 (were positive, now negative)
        just_arrived = (phi_old > 0) & (phi <= 0)
        arrival_time[just_arrived] = t
        
        t += dt
    
    return arrival_time, fire_type
```

---

## 8. Ensemble Batching

The key GPU advantage: all N members execute in the same kernel launches. The batch dimension N is just another axis in the tensor. PyTorch broadcasts all operations over it automatically.

### 8.1 Initialization

```python
def initialize_ensemble(terrain, perturbations, ignition_cell, device="cuda"):
    """
    Set up batched state for N ensemble members.
    
    perturbations: dict with keys "fmc_1hr", "wind_speed", "wind_dir"
        each value: numpy (N, rows, cols)
    ignition_cell: (row, col)
    """
    N = perturbations["fmc_1hr"].shape[0]
    rows, cols = terrain.shape
    
    # Transfer perturbations to GPU
    fmc = torch.tensor(perturbations["fmc_1hr"], dtype=torch.float32, device=device)
    ws = torch.tensor(perturbations["wind_speed"], dtype=torch.float32, device=device)
    wd = torch.tensor(perturbations["wind_dir"], dtype=torch.float32, device=device)
    
    # Initialize level set: +1 everywhere, -1 at ignition
    phi = torch.ones((N, rows, cols), dtype=torch.float32, device=device)
    phi[:, ignition_cell[0], ignition_cell[1]] = -1.0
    
    # Smooth the ignition point to avoid sharp discontinuity
    # (optional but improves numerical behavior)
    # Use signed distance from ignition point
    y_grid, x_grid = torch.meshgrid(
        torch.arange(rows, device=device, dtype=torch.float32),
        torch.arange(cols, device=device, dtype=torch.float32),
        indexing='ij'
    )
    dist = torch.sqrt((y_grid - ignition_cell[0])**2 + (x_grid - ignition_cell[1])**2)
    ignition_radius = 2.0  # cells
    phi_init = dist - ignition_radius  # negative inside circle, positive outside
    phi = phi_init.unsqueeze(0).expand(N, -1, -1).clone()
    
    arrival_time = torch.full((N, rows, cols), MAX_ARRIVAL, dtype=torch.float32, device=device)
    # Cells inside ignition radius arrive at t=0
    arrival_time[:, phi_init <= 0] = 0.0
    
    return phi, fmc, ws, wd, arrival_time
```

### 8.2 Full Engine

```python
class GPUFireEngine:
    def __init__(self, terrain: TerrainData, device: str = "cuda"):
        self.device = torch.device(device)
        self.dx = terrain.resolution_m
        
        # Load terrain to GPU (once)
        self.slope = torch.tensor(np.radians(terrain.slope), 
                                   dtype=torch.float32, device=self.device)
        self.tan_slope_sq = torch.tan(self.slope).pow(2)
        self.aspect = torch.tensor(np.radians(terrain.aspect),
                                    dtype=torch.float32, device=self.device)
        self.cbh = torch.tensor(terrain.canopy_base_height,
                                 dtype=torch.float32, device=self.device)
        self.cbd = torch.tensor(terrain.canopy_bulk_density,
                                 dtype=torch.float32, device=self.device)
        
        # Fuel parameter table
        self.fuel_table = torch.tensor(build_fuel_table(),
                                        dtype=torch.float32, device=self.device)
        
        # Fuel params per cell (gather once)
        fuel_model = torch.tensor(terrain.fuel_model, dtype=torch.long, device=self.device)
        self.fuel_params = self.fuel_table[fuel_model]  # (rows, cols, N_PARAMS)
        
        self.shape = terrain.shape
    
    def run(self, snapshot: CycleSnapshot, config: EnsembleConfig) -> EnsembleResult:
        N = config.n_members
        total_time = config.horizon_hours * 3600.0
        
        # Initialize ensemble on GPU
        phi, fmc, ws, wd, arrival_time = initialize_ensemble(
            self, config.perturbations, snapshot.ignition_cell, self.device)
        
        # Expand terrain params to broadcast with (N, rows, cols)
        fuel_params = self.fuel_params.unsqueeze(0)  # (1, rows, cols, N_PARAMS)
        tan_slope_sq = self.tan_slope_sq.unsqueeze(0)  # (1, rows, cols)
        cbh = self.cbh.unsqueeze(0)
        cbd = self.cbd.unsqueeze(0)
        
        # Simulation loop
        t = 0.0
        while t < total_time:
            # Surface ROS
            ros_surface, I_R = rothermel_ros(fmc, ws, fuel_params, tan_slope_sq)
            
            # Crown fire
            intensity = fireline_intensity(ros_surface, I_R, fuel_params[..., 0])
            ros_final, fire_type = crown_fire_check(intensity, cbh, cbd, ws, ros_surface)
            
            # Adaptive dt
            dt = compute_dt(ros_final, self.dx)
            dt = min(dt, total_time - t)
            
            # Level-set update
            phi_old = phi.clone()
            phi = level_set_step(phi, ros_final, self.dx, dt)
            
            # Record arrival times
            just_arrived = (phi_old > 0) & (phi <= 0)
            arrival_time[just_arrived] = t
            
            t += dt
        
        # Transfer results to CPU as numpy
        at_np = arrival_time.cpu().numpy()
        ft_np = fire_type.cpu().numpy()
        
        burned = at_np < (MAX_ARRIVAL * 0.9)
        burn_prob = burned.mean(axis=0).astype(np.float32)
        
        masked = np.where(burned, at_np, np.nan)
        with np.errstate(invalid='ignore'):
            mean_at = np.nanmean(masked, axis=0).astype(np.float32)
        mean_at = np.nan_to_num(mean_at, nan=MAX_ARRIVAL)
        
        variance = at_np.var(axis=0).astype(np.float32)
        
        return EnsembleResult(
            member_arrival_times=at_np,
            burn_probability=burn_prob,
            mean_arrival_time=mean_at,
            arrival_time_variance=variance,
            member_fmc_fields=config.perturbations["fmc_1hr"],
            member_wind_fields=config.perturbations["wind_speed"],
            n_members=N
        )
```

---

## 9. Unit Conversion Reference

All internal computation uses SI units. Convert at boundaries only.

|Quantity|Rothermel internal|SI (our convention)|Conversion|
|---|---|---|---|
|Wind speed|ft/min (midflame)|m/s (10m)|× 60 × 0.3048 × wind_adj_factor|
|ROS|m/min (chains/hr in some refs)|m/s|÷ 60|
|Fuel load|lb/ft² (some refs)|kg/m²|× 4.88243|
|Heat content|BTU/lb (some refs)|kJ/kg|× 2.326|
|CBH|m|m|none|
|CBD|kg/m³|kg/m³|none|
|Temperature|—|—|not used in Rothermel ROS|

**The wind adjustment factor** converts 10m open wind to midflame wind height. This depends on fuel type and canopy:

```python
# Simplified: use Baughman & Albini 1980
# For timber with canopy cover > 0: WAF ≈ 0.4
# For grass/shrub with no canopy: WAF ≈ 0.6
# For open: WAF = 1.0
def wind_adjustment_factor(canopy_cover, fuel_model):
    if canopy_cover > 0.5:
        return 0.4
    elif canopy_cover > 0.1:
        return 0.5
    else:
        return 0.6
```

For GPU: precompute WAF as a (rows, cols) tensor at initialization.

---

## 10. Validation

### 10.1 Single-Cell ROS Check

Before running any ensemble, verify the Rothermel implementation:

```python
# Fuel model 1 (short grass), FMC = 6%, wind = 5 mph midflame, flat terrain
# Expected ROS ≈ 0.67 chains/hr ≈ 0.22 m/s (from BehavePlus)
ros = rothermel_ros(fmc=0.06, wind_speed=2.24, fuel_params=FM1, slope=0)
assert abs(ros - 0.22) / 0.22 < 0.10, f"ROS mismatch: {ros} vs expected 0.22 m/s"

# Fuel model 4 (chaparral), FMC = 10%, wind = 10 mph midflame, 30% slope
# Expected ROS ≈ 4.5 chains/hr ≈ 1.5 m/s
ros = rothermel_ros(fmc=0.10, wind_speed=4.47, fuel_params=FM4, slope=30)
assert abs(ros - 1.5) / 1.5 < 0.15, f"ROS mismatch: {ros} vs expected 1.5 m/s"
```

### 10.2 Fire Shape Check

Run on flat terrain, uniform fuel, constant wind. The fire perimeter should be elliptical with the major axis aligned downwind. Measure the length-to-breadth ratio and compare to the Anderson 1983 formula.

### 10.3 Crown Fire Check

Set up a cell with fuel model 8 (timber), low CBH (2m), high wind (20 mph). Surface fire intensity should exceed Van Wagner's critical intensity. Verify that ROS jumps to crown fire values (~30-50 m/min).

Then increase FMC to 15%. Surface intensity drops below threshold. Crown fire should NOT initiate. Verify ROS stays at surface values.

### 10.4 Ensemble Consistency

Run 100 members with identical parameters (zero perturbation). All members should produce identical arrival times. Any divergence indicates a race condition or non-deterministic operation (check random number seeds).

### 10.5 Cross-Validate Against ELMFIRE

Run a single ELMFIRE simulation with specific parameters. Run the GPU engine with the same parameters. Compare arrival time fields. They won't match exactly (different numerical methods), but the fire perimeter shape and area burned should agree within ~20%.