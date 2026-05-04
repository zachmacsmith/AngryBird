# WISP Simulation Harness: Design Specification

---
## Overview

The simulation harness wraps the core WISP pipeline in a synthetic environment where ground truth is known, drones are simulated, and the entire system can be visualized as an animated video. It serves two purposes: validating that the system works, and producing the demo that wins the hackathon.

The simulation runs on a unified clock. At each simulation timestep, the ground truth fire advances, drones move along their paths collecting measurements, and at defined intervals the WISP core runs a cycle (ensemble → information field → selection → assimilation). The visualization renders every timestep as a video frame.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SIMULATION CLOCK                             │
│                     t = 0, 1, 2, ... T_max                         │
│                     dt = 10 seconds (configurable)                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────────┐
          │                │                    │
          ▼                ▼                    ▼
   ┌──────────────┐ ┌──────────────┐  ┌────────────────┐
   │ GROUND TRUTH │ │    DRONE     │  │  WISP CORE    │
   │              │ │  SIMULATOR   │  │  (runs every   │
   │ • FMC field  │ │              │  │   N minutes)   │
   │ • Wind field │ │ • Positions  │  │                │
   │ • Fire state │ │ • Paths      │  │ • Ensemble     │
   │              │ │ • Sensors    │  │ • Info field   │
   │ Advances     │ │ Move &       │  │ • Selection    │
   │ every dt     │ │ collect      │  │ • Assimilate   │
   │              │ │ every dt     │  │                │
   └──────┬───────┘ └──────┬───────┘  └───────┬────────┘
          │                │                   │
          │         ┌──────▼───────┐           │
          │         │ OBSERVATION  │           │
          │         │ BUFFER       ├──────────▶│
          │         │ Thin & pass  │  (on cycle)
          │         │ to WISP     │           │
          │         └──────────────┘           │
          │                                    │
          ▼                                    ▼
   ┌──────────────────────────────────────────────────┐
   │              FRAME RENDERER                       │
   │  Captures state from all three systems            │
   │  Renders one frame per simulation timestep        │
   │  Outputs: MP4 video or frame sequence             │
   └──────────────────────────────────────────────────┘
```

---

## Component 1: Ground Truth

### 1.1 Static Ground Truth: FMC Field

Generated once at initialization. Terrain-structured, spatially correlated. Does not change during the simulation (dead fuel moisture changes slowly — hours — relative to simulation duration).

```python
@dataclass
class GroundTruth:
    fmc: np.ndarray              # float32[rows, cols] — the "real" FMC
    wind_speed: np.ndarray       # float32[rows, cols] — current wind speed, m/s
    wind_direction: np.ndarray   # float32[rows, cols] — current wind direction, degrees
    fire_arrival_time: np.ndarray  # float32[rows, cols] — ground truth fire
    
    # Wind field evolution parameters
    wind_events: list[WindEvent]  # scheduled wind changes
```

FMC generation follows the terrain-structured approach discussed previously (aspect effect, elevation effect, canopy shading, smooth correlated noise at ~500m scale).

### 1.2 Dynamic Ground Truth: Wind Field

Wind evolves over the simulation. Two components:

**Smooth drift:** The base wind field rotates and varies in speed slowly over hours, simulating synoptic weather evolution.

```python
def update_wind_smooth(wind_speed, wind_direction, t, drift_rate=5.0):
    """Drift wind direction by drift_rate degrees per hour."""
    dt_hours = t / 3600.0
    return wind_speed, wind_direction + drift_rate * dt_hours
```

**Discrete events:** Scheduled wind shifts (e.g., at t=2 hours, wind shifts 40° and speed increases by 3 m/s). These create the operational scenarios where the system's value is most visible.

```python
@dataclass
class WindEvent:
    time_s: float              # when the shift occurs
    direction_change: float    # degrees
    speed_change: float        # m/s
    ramp_duration_s: float     # how long the transition takes (0 = instant)

# Example: afternoon wind shift at hour 3
events = [
    WindEvent(time_s=10800, direction_change=45, speed_change=3.0, ramp_duration_s=600)
]
```

The wind field also has spatial structure (terrain channeling):

```python
def compute_wind_field(base_speed, base_direction, terrain, t, events):
    """Compute spatially varying wind at time t."""
    # Apply events
    speed, direction = apply_events(base_speed, base_direction, t, events)
    
    # Terrain modulation
    tpi = terrain_position_index(terrain.elevation)
    speed_field = speed * (1.0 + 0.3 * np.clip(tpi, -1, 2))  # ridges windier
    direction_field = direction + terrain_channeling(terrain)   # valleys steer wind
    
    # Small-scale turbulence (changes each timestep)
    speed_field += np.random.normal(0, 0.3, terrain.shape)
    direction_field += np.random.normal(0, 3.0, terrain.shape)
    
    return speed_field, direction_field % 360
```

### 1.3 Ground Truth Fire

A single fire simulation running on the ground truth FMC and dynamic wind fields. This is the "real" fire that the system is trying to predict. It advances at every simulation timestep.

**Implementation:** Use the same fire engine (ELMFIRE or custom CA) but running on ground truth parameters with no perturbation. The fire advances incrementally — at each timestep dt, the CA steps forward one increment using the current wind field.

```python
class GroundTruthFire:
    def __init__(self, terrain, fmc_truth, ignition_point):
        self.state = initialize_fire(terrain, ignition_point)
        self.arrival_times = np.full(terrain.shape, MAX_ARRIVAL)
        self.current_time = 0.0
    
    def step(self, dt, wind_speed, wind_direction, fmc):
        """Advance fire by dt seconds using current ground truth conditions."""
        ros = rothermel_ros(self.terrain, fmc, wind_speed, wind_direction)
        # CA step
        newly_ignited = ca_step(self.state, ros, dt)
        self.arrival_times[newly_ignited] = self.current_time
        self.current_time += dt
```

The ground truth fire produces the "real" fire perimeter that the WISP ensemble is trying to predict. The gap between WISP's prediction and this ground truth is the error the system works to reduce.

---

## Component 2: Drone Simulator

### 2.1 Drone State

Each drone has a position, a queue of waypoints, and a sensor state.

```python
@dataclass
class DroneState:
    drone_id: str
    position: np.ndarray          # [x, y] in meters (UTM)
    altitude: float               # meters AGL
    speed: float                  # m/s cruise speed
    status: str                   # "idle" | "transit" | "loiter" | "returning"
    waypoint_queue: list[np.ndarray]  # remaining waypoints
    current_target: np.ndarray | None
    path_history: list[np.ndarray]    # positions visited (for visualization trail)
    observations_buffer: list[DroneObservation]  # collected this sortie
    endurance_remaining_s: float  # seconds of flight time left
    base_position: np.ndarray     # home location for return
```

### 2.2 Drone Movement

At each simulation timestep, each drone moves toward its current target at cruise speed.

```python
def move_drone(drone: DroneState, dt: float):
    if drone.status == "idle" or drone.current_target is None:
        if drone.waypoint_queue:
            drone.current_target = drone.waypoint_queue.pop(0)
            drone.status = "transit"
        else:
            return  # nothing to do
    
    # Direction to target
    direction = drone.current_target - drone.position
    dist = np.linalg.norm(direction)
    
    if dist < drone.speed * dt:
        # Arrived at target
        drone.position = drone.current_target.copy()
        drone.path_history.append(drone.position.copy())
        
        if drone.waypoint_queue:
            drone.current_target = drone.waypoint_queue.pop(0)
        else:
            # Return to base
            drone.current_target = drone.base_position.copy()
            drone.status = "returning"
    else:
        # Move toward target
        step = direction / dist * drone.speed * dt
        drone.position += step
        drone.path_history.append(drone.position.copy())
    
    drone.endurance_remaining_s -= dt
```

### 2.3 Sensor Simulation

At each timestep, the drone collects measurements from the ground truth at its current location and surrounding cells within the camera footprint.

**FMC measurement:** Multispectral camera footprint covers a radius around the drone's nadir point, depending on altitude and lens. At 50m altitude with a typical FOV, the footprint is roughly 3×3 to 5×5 cells.

**Wind measurement:** Anemometer measures at the drone's exact position only (point measurement).

```python
def collect_observations(drone: DroneState, ground_truth: GroundTruth,
                          grid_resolution: float, noise: NoiseConfig,
                          current_time: float) -> list[DroneObservation]:
    observations = []
    
    # Convert drone position to grid cell
    center_cell = position_to_cell(drone.position, grid_resolution, origin)
    
    # FMC: observe cells within camera footprint
    fmc_radius_cells = int(noise.camera_footprint_m / grid_resolution)
    for dr in range(-fmc_radius_cells, fmc_radius_cells + 1):
        for dc in range(-fmc_radius_cells, fmc_radius_cells + 1):
            cell = (center_cell[0] + dr, center_cell[1] + dc)
            if not in_bounds(cell, ground_truth.fmc.shape):
                continue
            dist_cells = np.sqrt(dr**2 + dc**2)
            if dist_cells > fmc_radius_cells:
                continue
            
            # Noise increases toward edge of footprint
            edge_factor = 1.0 + 0.5 * (dist_cells / fmc_radius_cells)
            fmc_noise = noise.fmc_sigma * edge_factor
            
            obs = DroneObservation(
                location=cell,
                fmc=ground_truth.fmc[cell] + np.random.normal(0, fmc_noise),
                fmc_sigma=fmc_noise,
                wind_speed=np.nan,       # FMC-only observation at non-center cells
                wind_speed_sigma=np.nan,
                wind_direction=np.nan,
                wind_direction_sigma=np.nan,
                timestamp=current_time,
                drone_id=drone.drone_id
            )
            observations.append(obs)
    
    # Wind: observe at center cell only
    ws_obs = DroneObservation(
        location=center_cell,
        fmc=ground_truth.fmc[center_cell] + np.random.normal(0, noise.fmc_sigma),
        fmc_sigma=noise.fmc_sigma,
        wind_speed=ground_truth.wind_speed[center_cell] + np.random.normal(0, noise.ws_sigma),
        wind_speed_sigma=noise.ws_sigma,
        wind_direction=ground_truth.wind_direction[center_cell] + np.random.normal(0, noise.wd_sigma),
        wind_direction_sigma=noise.wd_sigma,
        timestamp=current_time,
        drone_id=drone.drone_id
    )
    observations.append(ws_obs)
    
    drone.observations_buffer.extend(observations)
    return observations
```

### 2.4 Observation Buffer and Thinning

Drones accumulate observations continuously. At each WISP cycle boundary, the buffer is thinned and passed to the core model.

```python
class ObservationBuffer:
    def __init__(self, min_spacing_m: float = 200.0):
        self._buffer: list[DroneObservation] = []
        self._min_spacing = min_spacing_m
    
    def add(self, observations: list[DroneObservation]):
        self._buffer.extend(observations)
    
    def flush_thinned(self) -> list[DroneObservation]:
        """Return thinned observations and clear buffer."""
        thinned = thin_observations(self._buffer, self._min_spacing)
        self._buffer.clear()
        return thinned
```

---

## Component 3: Simulation Loop

### 3.1 Clock and Cycle Timing

```python
@dataclass
class SimulationConfig:
    dt: float = 10.0                  # simulation timestep (seconds)
    total_time_s: float = 21600.0     # 6 hours
    ignis_cycle_interval_s: float = 1200.0  # WISP runs every 20 minutes
    n_drones: int = 5
    drone_speed: float = 15.0        # m/s
    drone_endurance_s: float = 1800.0  # 30 minutes per sortie
    camera_footprint_m: float = 100.0  # radius of FMC observation footprint
    
    # Visualization
    render_fps: int = 30
    playback_speed: float = 60.0      # 1 second of video = 60 seconds of simulation
    output_path: str = "simulation.mp4"
```

### 3.2 Main Loop

```python
class SimulationRunner:
    def __init__(self, config: SimulationConfig, terrain: TerrainData,
                 ground_truth: GroundTruth, orchestrator: Orchestrator):
        self.config = config
        self.terrain = terrain
        self.truth = ground_truth
        self.orchestrator = orchestrator
        self.obs_buffer = ObservationBuffer()
        self.renderer = FrameRenderer(terrain, config)
        
        # Initialize drones at base
        self.drones = [
            DroneState(
                drone_id=f"drone_{i}",
                position=config.base_position.copy(),
                altitude=50.0,
                speed=config.drone_speed,
                status="idle",
                waypoint_queue=[],
                current_target=None,
                path_history=[config.base_position.copy()],
                observations_buffer=[],
                endurance_remaining_s=config.drone_endurance_s,
                base_position=config.base_position.copy()
            )
            for i in range(config.n_drones)
        ]
        
        # State tracking
        self.current_time = 0.0
        self.last_cycle_time = -config.ignis_cycle_interval_s  # trigger cycle at t=0
        self.cycle_count = 0
        self.cycle_reports = []
        
        # Current WISP outputs for visualization
        self.current_info_field = None
        self.current_gp_mean = None
        self.current_gp_variance = None
        self.current_mission_queue = None
    
    def run(self):
        """Main simulation loop."""
        n_steps = int(self.config.total_time_s / self.config.dt)
        
        for step in range(n_steps):
            self.current_time = step * self.config.dt
            
            # 1. Update ground truth wind field
            self.truth.wind_speed, self.truth.wind_direction = compute_wind_field(
                self.truth.base_wind_speed,
                self.truth.base_wind_direction,
                self.terrain,
                self.current_time,
                self.truth.wind_events
            )
            
            # 2. Advance ground truth fire
            self.truth.fire.step(
                self.config.dt,
                self.truth.wind_speed,
                self.truth.wind_direction,
                self.truth.fmc
            )
            
            # 3. Move drones and collect observations
            for drone in self.drones:
                move_drone(drone, self.config.dt)
                if drone.status in ("transit", "loiter"):
                    obs = collect_observations(
                        drone, self.truth,
                        self.terrain.resolution_m,
                        self.config.noise,
                        self.current_time
                    )
                    self.obs_buffer.add(obs)
            
            # 4. Check if WISP cycle is due
            if self.current_time - self.last_cycle_time >= self.config.ignis_cycle_interval_s:
                self._run_ignis_cycle()
                self.last_cycle_time = self.current_time
            
            # 5. Render frame
            self.renderer.render_frame(
                step=step,
                time=self.current_time,
                ground_truth=self.truth,
                drones=self.drones,
                info_field=self.current_info_field,
                gp_mean=self.current_gp_mean,
                gp_variance=self.current_gp_variance,
                mission_queue=self.current_mission_queue,
                cycle_reports=self.cycle_reports
            )
        
        # Finalize video
        self.renderer.finalize()
        return self.cycle_reports
    
    def _run_ignis_cycle(self):
        """Execute one WISP cycle: assimilate observations, replan."""
        self.cycle_count += 1
        
        # Flush and thin observations from buffer
        observations = self.obs_buffer.flush_thinned()
        
        # Run core WISP cycle
        report = self.orchestrator.run_cycle(observations=observations)
        self.cycle_reports.append(report)
        
        # Cache WISP outputs for visualization
        self.current_info_field = report.info_field
        self.current_gp_mean = report.gp_mean
        self.current_gp_variance = report.gp_variance
        self.current_mission_queue = report.mission_queue
        
        # Assign new waypoints to drones from mission queue
        self._assign_drone_waypoints(report.mission_queue)
    
    def _assign_drone_waypoints(self, queue: MissionQueue):
        """Distribute mission requests to available drones."""
        available = [d for d in self.drones 
                     if d.status == "idle" or d.endurance_remaining_s < 60]
        
        # Reset endurance for drones that returned to base
        for drone in available:
            if np.linalg.norm(drone.position - drone.base_position) < 50:
                drone.endurance_remaining_s = self.config.drone_endurance_s
        
        # Assign top-K requests to available drones
        for i, drone in enumerate(available):
            if i < len(queue.requests):
                target = queue.requests[i]
                target_pos = cell_to_position(
                    target.grid_cell, 
                    self.terrain.resolution_m, 
                    self.terrain.origin
                )
                drone.waypoint_queue = [target_pos]
                drone.current_target = target_pos
                drone.status = "transit"
                drone.observations_buffer.clear()
```

---

## Component 4: Frame Renderer

### 4.1 Layout

Six-panel layout plus a timeline bar:

```
┌──────────────────────────────────────────────────────────┐
│  SIMULATION TIME: 02:34:10    CYCLE: 7/18    DRONES: 5   │
├───────────────────┬───────────────────┬──────────────────┤
│                   │                   │                  │
│  GROUND TRUTH     │  WISP ESTIMATE   │  UNCERTAINTY     │
│  FMC + WIND       │  FMC + WIND       │  MAP             │
│                   │                   │                  │
│  [terrain base]   │  [terrain base]   │  [terrain base]  │
│  [FMC colormap]   │  [GP mean FMC]    │  [GP variance]   │
│  [wind vectors]   │  [est. wind vecs] │  [info field w]  │
│  [true fire]      │  [predicted fire] │  [drone targets] │
│  [drone sprites]  │  [burn prob.]     │  [high-w cells]  │
│                   │                   │                  │
├───────────────────┴───────────────────┴──────────────────┤
│                                                          │
│  ENTROPY CONVERGENCE                                     │
│  [running plot of total entropy per cycle]               │
│  [lines for greedy / qubo / uniform / fire-front]        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 4.2 Layer Rendering

Each map panel is a stack of layers rendered bottom-to-top:

```python
class MapPanel:
    def __init__(self, ax, terrain, title):
        self.ax = ax
        self.terrain = terrain
        self.title = title
        
        # Base layer: terrain hillshade (static, rendered once)
        self.hillshade = render_hillshade(terrain.elevation)
        self.base_img = ax.imshow(self.hillshade, cmap='gray', alpha=0.4)
    
    def update(self, **layers):
        """Update dynamic layers for this frame."""
        # Clear previous dynamic artists
        self._clear_dynamic()
        
        if "fmc" in layers:
            self.ax.imshow(layers["fmc"], cmap='YlOrBr_r', 
                          alpha=0.6, vmin=0.02, vmax=0.25)
        
        if "wind" in layers:
            ws, wd = layers["wind"]
            # Subsample for arrow visibility
            step = max(1, ws.shape[0] // 20)
            Y, X = np.mgrid[0:ws.shape[0]:step, 0:ws.shape[1]:step]
            U = ws[::step, ::step] * np.sin(np.radians(wd[::step, ::step]))
            V = ws[::step, ::step] * np.cos(np.radians(wd[::step, ::step]))
            self.ax.quiver(X, Y, U, -V, color='white', alpha=0.7, 
                          scale=100, width=0.003)
        
        if "fire" in layers:
            fire_mask = layers["fire"] < layers.get("current_time", 1e9)
            self.ax.contour(fire_mask.astype(float), levels=[0.5], 
                          colors='red', linewidths=2)
            self.ax.contourf(fire_mask.astype(float), levels=[0.5, 1.5],
                           colors=['red'], alpha=0.3)
        
        if "burn_probability" in layers:
            bp = layers["burn_probability"]
            self.ax.imshow(bp, cmap='YlOrRd', alpha=0.5, vmin=0, vmax=1)
        
        if "uncertainty" in layers:
            self.ax.imshow(layers["uncertainty"], cmap='inferno',
                          alpha=0.7)
        
        if "drone_targets" in layers:
            targets = layers["drone_targets"]
            for i, t in enumerate(targets):
                self.ax.plot(t[1], t[0], '^', color='cyan', 
                           markersize=10, markeredgecolor='white')
                self.ax.annotate(str(i+1), (t[1], t[0]), 
                               color='white', fontsize=8,
                               ha='center', va='bottom')
```

### 4.3 Drone Rendering

Drones are rendered as moving sprites with trails and projected paths.

```python
def render_drones(ax, drones, grid_resolution, origin):
    for drone in drones:
        # Current position
        cell = position_to_cell(drone.position, grid_resolution, origin)
        
        # Color by status
        color = {"idle": "gray", "transit": "lime", 
                 "loiter": "cyan", "returning": "orange"}[drone.status]
        
        # Drone marker
        ax.plot(cell[1], cell[0], 'o', color=color, 
               markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        
        # Trail (last 60 seconds of positions)
        trail = drone.path_history[-60:]
        if len(trail) > 1:
            trail_cells = [position_to_cell(p, grid_resolution, origin) for p in trail]
            trail_y = [c[0] for c in trail_cells]
            trail_x = [c[1] for c in trail_cells]
            ax.plot(trail_x, trail_y, '-', color=color, alpha=0.4, linewidth=1)
        
        # Projected path to current target
        if drone.current_target is not None:
            target_cell = position_to_cell(drone.current_target, grid_resolution, origin)
            ax.plot([cell[1], target_cell[1]], [cell[0], target_cell[0]],
                   '--', color=color, alpha=0.6, linewidth=1)
        
        # Camera footprint circle
        footprint_cells = drone.altitude / grid_resolution  # rough
        circle = plt.Circle((cell[1], cell[0]), footprint_cells, 
                           fill=False, color=color, alpha=0.3, linewidth=0.5)
        ax.add_patch(circle)
        
        # Label
        ax.annotate(drone.drone_id[-1], (cell[1]+1, cell[0]-1),
                   color=color, fontsize=7)
```

### 4.4 Video Output

```python
class FrameRenderer:
    def __init__(self, terrain, config):
        self.fig, self.axes = plt.subplots(2, 3, figsize=(20, 14))
        self.panels = {
            "truth_fmc": MapPanel(self.axes[0,0], terrain, "Ground Truth: FMC + Wind"),
            "estimate":  MapPanel(self.axes[0,1], terrain, "WISP Estimate"),
            "uncertainty": MapPanel(self.axes[0,2], terrain, "Information Field"),
        }
        self.entropy_ax = self.fig.add_subplot(2, 1, 2)  # bottom spanning row
        
        # Video writer
        self.writer = FFMpegWriter(fps=config.render_fps)
        self.writer.setup(self.fig, config.output_path, dpi=150)
        
        # Frame skip: render every Nth simulation step
        # At dt=10s and 60× playback, 30fps video needs 1 frame per 20s sim time
        self.frame_interval = int(config.playback_speed / config.render_fps / config.dt)
        self.frame_count = 0
    
    def render_frame(self, step, time, ground_truth, drones,
                     info_field, gp_mean, gp_variance,
                     mission_queue, cycle_reports):
        """Render one frame if it's time."""
        if step % self.frame_interval != 0:
            return
        
        # Update title
        hours = int(time // 3600)
        mins = int((time % 3600) // 60)
        secs = int(time % 60)
        self.fig.suptitle(
            f"Time: {hours:02d}:{mins:02d}:{secs:02d}  |  "
            f"Cycle: {len(cycle_reports)}  |  "
            f"Observations: {sum(r.n_observations for r in cycle_reports)}",
            fontsize=14, fontweight='bold'
        )
        
        # Panel 1: Ground truth
        self.panels["truth_fmc"].update(
            fmc=ground_truth.fmc,
            wind=(ground_truth.wind_speed, ground_truth.wind_direction),
            fire=ground_truth.fire.arrival_times,
            current_time=time
        )
        render_drones(self.axes[0,0], drones, 
                     self.terrain.resolution_m, self.terrain.origin)
        
        # Panel 2: WISP estimate
        if gp_mean is not None:
            self.panels["estimate"].update(
                fmc=gp_mean,
                burn_probability=cycle_reports[-1].burn_probability if cycle_reports else None
            )
            render_drones(self.axes[0,1], drones,
                         self.terrain.resolution_m, self.terrain.origin)
        
        # Panel 3: Information field + drone targets
        if info_field is not None:
            targets = [r.grid_cell for r in mission_queue.requests] if mission_queue else []
            self.panels["uncertainty"].update(
                uncertainty=info_field.w,
                drone_targets=targets
            )
            render_drones(self.axes[0,2], drones,
                         self.terrain.resolution_m, self.terrain.origin)
        
        # Bottom panel: entropy convergence
        if cycle_reports:
            self.entropy_ax.clear()
            cycles = range(1, len(cycle_reports) + 1)
            for strategy in ["greedy", "qubo", "uniform", "fire_front"]:
                values = [r.evaluations[strategy].entropy_reduction 
                         for r in cycle_reports if strategy in r.evaluations]
                if values:
                    cumulative = np.cumsum(values)
                    self.entropy_ax.plot(cycles[:len(cumulative)], cumulative, 
                                       label=strategy, linewidth=2)
            self.entropy_ax.set_xlabel("Cycle")
            self.entropy_ax.set_ylabel("Cumulative Entropy Reduction")
            self.entropy_ax.legend(loc='upper left')
            self.entropy_ax.grid(True, alpha=0.3)
        
        # Write frame
        self.writer.grab_frame()
        self.frame_count += 1
    
    def finalize(self):
        self.writer.finish()
        print(f"Video saved: {self.frame_count} frames")
```

---

## Component 5: Noise Configuration

```python
@dataclass
class NoiseConfig:
    fmc_sigma: float = 0.03           # FMC measurement noise (fraction)
    ws_sigma: float = 1.0             # wind speed noise (m/s)
    wd_sigma: float = 10.0            # wind direction noise (degrees)
    camera_footprint_m: float = 100.0 # radius of FMC observation
    degrade_near_fire: bool = True
    fire_degradation_radius_m: float = 500.0
    fire_degradation_factor: float = 3.0  # multiply sigma by this near fire
```

When `degrade_near_fire` is True, observations within `fire_degradation_radius_m` of the ground truth fire front have their noise multiplied by `fire_degradation_factor` (smoke, thermal interference).

---

## Package Structure

```
ignis/
└── simulation/
    ├── __init__.py
    ├── ground_truth.py       # GroundTruth, FMC generation, wind evolution
    ├── fire_oracle.py        # Ground truth fire simulation
    ├── drone_sim.py          # DroneState, movement, sensor simulation
    ├── observation_buffer.py # Thinning and buffering
    ├── runner.py             # SimulationRunner main loop
    ├── renderer.py           # FrameRenderer, MapPanel, video output
    └── scenarios.py          # Pre-defined scenarios (flat, hilly, wind-shift)
```

---

## Pre-Defined Scenarios

```python
# scenarios.py

def flat_homogeneous():
    """Control scenario. Flat terrain, uniform fuel, constant wind.
    Targeted placement should show minimal advantage over uniform."""
    ...

def hilly_heterogeneous():
    """Primary demo. Complex terrain, mixed fuel types, variable FMC.
    Targeted placement should significantly outperform uniform."""
    ...

def wind_shift():
    """Stress test. Complex terrain + wind shift event at hour 3.
    Tests system response to sudden change. Shows re-routing."""
    ...

def crown_fire_risk():
    """Scenario with stands of low CBH where crown fire transition
    is uncertain. Tests bimodal detection and binary entropy boost."""
    ...
```

---

## Performance Considerations

**Frame rendering is the bottleneck.** Each frame involves clearing and redrawing matplotlib artists. At 30 fps output with 6 hours of simulation at 60× playback, that's ~10,800 frames. At ~0.1 seconds per frame render, total rendering time is ~18 minutes. This is acceptable as a post-processing step but too slow for live playback.

**Options to speed up:**

1. Render every 3rd frame (10 fps output) — 6 minutes render time, slightly choppy but fine for demo
2. Pre-render static layers (hillshade, colorbar) once, only update dynamic layers (fire, drones, vectors)
3. Use blitting in matplotlib (only redraw changed artists)
4. Switch to a faster renderer (Pillow + manual compositing, or Pygame for live preview)

**Recommendation:** Render at 10 fps during development. Switch to 30 fps for the final demo video only. Add a `--live` flag that displays frames in a matplotlib window without saving to video (slower but interactive).

---

## Build Priority

|Component|Lines|Priority|Notes|
|---|---|---|---|
|Ground truth FMC generation|40|Day 2|Terrain-structured, use existing code from earlier discussion|
|Ground truth wind (smooth + events)|30|Day 2|Simple evolution model|
|Ground truth fire|20|Day 2|Single run of fire engine on truth params|
|Drone movement|40|Day 2-3|Point-to-point at constant speed|
|Sensor simulation|50|Day 3|FMC footprint + wind point measurement|
|Observation buffer + thinning|20|Day 3|Exists from earlier spec, adapt|
|Main simulation loop|60|Day 3-4|Wires everything together|
|Frame renderer|120|Day 4|Most code but mostly matplotlib boilerplate|
|Video output|20|Day 4|FFMpegWriter wrapper|
|Scenarios|40|Day 4|Pre-defined terrain + wind configurations|
|**Total**|**~440**|||