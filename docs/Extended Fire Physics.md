# IGNIS: Extended Fire Physics & Discrete Event Handling

Additions to the core architecture. Each section is independent and can be implemented separately.

---

## 1. Nelson Fuel Moisture Model as GP Prior Mean

### Problem

The GP currently interpolates FMC from sparse RAWS point observations, treating the space between stations as smooth. This ignores terrain-driven variation: a south-facing slope at 1,500m elevation dries much faster than a north-facing slope at the same elevation 200m away. The GP mean between stations is effectively a flat estimate that the variance field corrects after drone observations.

### Fix

Use the Nelson dead fuel moisture model to compute a physics-informed FMC estimate at every grid cell, then use this as the GP prior mean instead of interpolated RAWS values.

Nelson's model (2000) estimates FMC from weather and terrain. The core computation per cell:

```python
def nelson_fmc(temperature, humidity, solar_radiation, 
               slope, aspect, elevation, canopy_cover, hour_of_day):
    """
    Nelson 2000 fuel moisture model.
    Computes equilibrium moisture content + timelag response.
    """
    # Equilibrium moisture content (where FMC is heading)
    if humidity < 0.1:
        emc = 0.03229 + 0.281073 * humidity - 0.000578 * humidity * temperature
    elif humidity < 0.5:
        emc = 2.22749 + 0.160107 * humidity - 0.01478 * temperature
    else:
        emc = 21.0606 + 0.005565 * humidity**2 - 0.00035 * humidity * temperature
    
    emc /= 100.0  # convert to fraction
    
    # Solar radiation correction based on slope, aspect, canopy
    # South-facing exposed slopes get more radiation → lower FMC
    solar_factor = compute_solar_factor(slope, aspect, canopy_cover, hour_of_day)
    emc *= (1.0 - 0.25 * solar_factor)  # simplified radiation correction
    
    # Elevation correction (higher = cooler = moister, roughly)
    emc *= (1.0 + 0.0001 * (elevation - reference_elevation))
    
    return emc

# Apply to full grid
fmc_prior_mean = np.vectorize(nelson_fmc)(
    temperature, humidity, solar_radiation,
    terrain.slope, terrain.aspect, terrain.elevation,
    canopy_cover, hour_of_day
)
```

The NWCG publishes lookup tables for this computation (Fosberg & Deeming 1971, Nelson 2000). For the hackathon, a simplified version that captures the slope/aspect/elevation effects is sufficient — the exact Nelson parameterization can be refined later.

### Integration with GP

```python
# GP prior mean: Nelson model (terrain-aware, physics-based)
# GP prior variance: from RAWS observation geometry + temporal decay
gp.fit(
    X=raws_locations,                    # station coordinates
    y=raws_fmc_observations,             # measured FMC at stations
    prior_mean=nelson_fmc_field,         # Nelson model as mean function
    noise=raws_noise
)
# GP corrects Nelson model where RAWS data disagrees
# GP uncertainty reflects distance from any observation source
```

The GP's residual (observation minus Nelson prediction at RAWS locations) tells you how wrong the Nelson model is locally. This residual is what the GP actually interpolates — not raw FMC, but the correction to Nelson's estimate. This is standard in geostatistics: use a physical model as the trend, GP on the residuals.

### Implementation cost

~30 lines for simplified Nelson model. ~5 lines to integrate with GP as prior mean function.

---

## 2. Crown Fire Modeling

### What to Add

Two equations from the FARSITE/FlamMap physics stack:

**Van Wagner (1977) crown fire initiation:** Does surface fire intensity exceed the threshold for igniting the canopy?

**Rothermel (1991) crown fire spread:** If crown fire initiates, how fast does it spread?

### Additional Data Required

Two LANDFIRE layers, downloaded alongside elevation and fuel models:

- **Canopy Base Height (CBH):** Height from ground to bottom of live canopy (meters). Determines how intense the surface fire must be to reach the canopy.
- **Canopy Bulk Density (CBD):** Mass of canopy fuel per unit volume (kg/m³). Determines whether crown fire sustains once initiated.

```python
# In terrain loader, add to LANDFIRE download
@dataclass(frozen=True)
class TerrainData:
    elevation: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray
    fuel_model: np.ndarray
    canopy_base_height: np.ndarray   # NEW — meters
    canopy_bulk_density: np.ndarray  # NEW — kg/m³
    canopy_cover: np.ndarray         # NEW — fraction, also used by Nelson model
    resolution_m: float
    origin: tuple[float, float]
    shape: tuple[int, int]
```

### Implementation

```python
def crown_fire_initiation(I_surface, cbh, fmc_foliar):
    """
    Van Wagner 1977.
    I_surface: surface fireline intensity (kW/m)
    cbh: canopy base height (m)
    fmc_foliar: foliar moisture content (fraction, typically ~1.0)
    Returns: bool array, True where crown fire initiates
    """
    # Critical intensity to ignite canopy
    I_critical = (0.01 * cbh * (460 + 25.9 * fmc_foliar * 100))**1.5
    return I_surface > I_critical

def crown_fire_ros(wind_speed_10m, cbd):
    """
    Rothermel 1991 crown fire spread model.
    wind_speed_10m: 10-meter open wind speed (m/s)
    cbd: canopy bulk density (kg/m³)
    Returns: crown fire rate of spread (m/min)
    """
    # Convert wind to km/h for Rothermel's equation
    wind_kmh = wind_speed_10m * 3.6
    return 11.02 * wind_kmh**0.854 * cbd**0.19

def surface_fire_intensity(ros_surface, I_R, fuel_load):
    """
    Byram's fireline intensity.
    ros_surface: surface rate of spread (m/min)
    I_R: reaction intensity (kW/m²)
    fuel_load: available fuel (kg/m²)
    Returns: fireline intensity (kW/m)
    """
    heat_content = 18000  # kJ/kg, standard
    return heat_content * fuel_load * ros_surface / 60.0
```

### Modified CA Step

```python
def compute_ros_with_crown(fuel, fmc, wind_speed, slope, cbh, cbd):
    # Standard surface fire
    ros_surface = rothermel_surface_ros(fuel, fmc, wind_speed, slope)
    I_R = reaction_intensity(fuel, fmc)
    intensity = surface_fire_intensity(ros_surface, I_R, fuel.load)
    
    # Check crown fire initiation
    fmc_foliar = 1.0  # typical foliar moisture, could vary seasonally
    initiates = crown_fire_initiation(intensity, cbh, fmc_foliar)
    
    # If crown fire initiates, use the faster of surface and crown ROS
    ros_crown = crown_fire_ros(wind_speed, cbd)
    ros = np.where(initiates, np.maximum(ros_surface, ros_crown), ros_surface)
    
    # Also flag the fire type for downstream use
    fire_type = np.where(initiates, CROWN_FIRE, SURFACE_FIRE)
    
    return ros, fire_type
```

### What This Creates in the Ensemble

With FMC perturbed across members, some members will have low enough FMC that surface intensity exceeds the Van Wagner threshold, triggering crown fire. Others won't. The ensemble splits:

- Members with crown fire: ROS jumps to 30-100 m/min. Fire arrives at downstream cells in minutes.
- Members without: ROS stays at 2-5 m/min. Fire arrives in hours.

Arrival time distributions at downstream cells become sharply bimodal. The bimodal detector (Section 3) flags these cells. The binary entropy term boosts their w_i. Drones are sent to measure FMC in the specific stands where crown fire transition is uncertain.

A drone measuring FMC = 0.07 in a stand with CBH = 3m might definitively resolve whether crown fire initiates — collapsing the bimodal distribution to one mode. That's the most consequential single measurement the system can make.

### Computational Cost

Two extra arithmetic operations per cell per timestep: one intensity comparison, one conditional ROS selection. No new state variables, no structural change to the CA. The ensemble runs at the same speed. The additional LANDFIRE layers add ~160 KB to the terrain data for a 200×200 grid.

---

## 3. Bimodal Detection & Binary Entropy Layer

### Detection

After each ensemble run, classify each cell by the distribution shape of its arrival times across members:

```python
def detect_bimodality(arrival_times, max_arrival):
    """
    arrival_times: (N, rows, cols)
    Returns: burn_fraction, bimodal_score, fire_type_disagreement
    """
    N = arrival_times.shape[0]
    
    # Burn/no-burn bimodality
    burns = arrival_times < (max_arrival * 0.9)
    burn_fraction = burns.mean(axis=0)          # (rows, cols)
    bimodal_score = 1.0 - 2.0 * np.abs(burn_fraction - 0.5)
    # 0 = all agree, 1 = 50/50 split
    
    return burn_fraction, bimodal_score

def detect_regime_split(fire_types):
    """
    fire_types: (N, rows, cols), values SURFACE_FIRE or CROWN_FIRE
    Detects disagreement about fire regime.
    """
    crown_fraction = (fire_types == CROWN_FIRE).mean(axis=0)
    regime_bimodal = 1.0 - 2.0 * np.abs(crown_fraction - 0.5)
    return crown_fraction, regime_bimodal
```

### Binary Entropy Augmentation of Information Field

```python
def augmented_information_field(w_continuous, burn_fraction, 
                                 regime_bimodal, alpha=0.5):
    """
    Blend continuous information value with binary entropy
    for discrete events.
    """
    # Binary entropy: max at p=0.5, zero at p=0 or p=1
    eps = 1e-10
    p = burn_fraction
    binary_entropy = -(p * np.log2(p + eps) + (1-p) * np.log2(1-p + eps))
    # Max value = 1.0 at p=0.5
    
    # Crown fire regime entropy (same formula, different input)
    p_crown = regime_bimodal  # reuse bimodal_score from regime detection
    regime_entropy = -(p_crown * np.log2(p_crown + eps) + 
                       (1-p_crown) * np.log2(1-p_crown + eps))
    
    # Combined information field
    # alpha controls weight of binary considerations
    # beta controls extra weight for crown fire regime transitions
    beta = 0.3
    w_total = w_continuous + alpha * binary_entropy + beta * regime_entropy
    
    return w_total
```

The alpha and beta weights control how aggressively the system targets bimodal cells. At alpha=0, pure continuous variance drives placement (original behavior). At alpha=1, binary entropy contributes equally to w_i. For the hackathon, alpha=0.5 is a reasonable default — test sensitivity to this parameter.

### Bimodal-Aware EnKF Update

At cells where the ensemble is bimodal, the standard Kalman update averages across modes, producing a meaningless mean. Use selective member reweighting instead:

```python
def enkf_update_adaptive(ensemble_states, observations, 
                          bimodal_score, threshold=0.6):
    """
    Standard EnKF for unimodal cells, particle-filter reweighting for bimodal.
    """
    # Identify which observations fall in bimodal regions
    bimodal_obs = []
    unimodal_obs = []
    for obs in observations:
        if bimodal_score[obs.location] > threshold:
            bimodal_obs.append(obs)
        else:
            unimodal_obs.append(obs)
    
    # Standard EnKF for unimodal observations
    if unimodal_obs:
        ensemble_states = standard_enkf_update(
            ensemble_states, unimodal_obs)
    
    # Particle-filter reweighting for bimodal observations
    if bimodal_obs:
        weights = np.ones(N)
        for obs in bimodal_obs:
            for n in range(N):
                # How consistent is this member with the observation?
                predicted = ensemble_states[n, state_index(obs.location)]
                residual = obs.value - predicted
                # Gaussian likelihood
                likelihood = np.exp(-0.5 * residual**2 / obs.sigma**2)
                weights[n] *= likelihood
        
        # Normalize weights
        weights /= weights.sum()
        
        # Resample: duplicate high-weight members, drop low-weight
        # Systematic resampling (standard particle filter technique)
        indices = systematic_resample(weights, N)
        ensemble_states = ensemble_states[indices]
        
        # Add small perturbation to resampled members to maintain diversity
        ensemble_states += np.random.normal(0, 0.001, ensemble_states.shape)
    
    return ensemble_states

def systematic_resample(weights, N):
    """Standard particle filter resampling."""
    positions = (np.arange(N) + np.random.uniform()) / N
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    return indices
```

### Implementation Cost

- Bimodal detection: ~15 lines, runs in microseconds (one mean + absolute value over ensemble axis)
- Binary entropy augmentation: ~10 lines, elementwise operations
- Adaptive EnKF: ~40 lines, adds particle filter branch for bimodal observations
- Total: ~65 lines

---

## 4. Spotting Risk Overlay

### Why Include It

Full stochastic spotting simulation (ember generation, lofting, transport, landing, ignition) is too complex for the hackathon and structurally changes the CA. But the spotting _risk zone_ — where embers are likely to land if spotting occurs — is a simple function of wind speed, fire intensity, and terrain. Drone wind measurements directly improve this estimate.

### Albini Spotting Distance (Simplified)

```python
def max_spotting_distance(intensity, wind_speed_10m, terrain_elevation_diff):
    """
    Albini 1979, simplified.
    Estimates maximum distance embers can travel from the fire front.
    
    intensity: fireline intensity (kW/m)
    wind_speed_10m: 10-meter wind speed (m/s)
    terrain_elevation_diff: elevation difference between fire and landing point (m)
    Returns: maximum spotting distance (m)
    """
    # Flame height (m) — Byram 1959
    flame_height = 0.0775 * intensity**0.46
    
    # Maximum lofting height (m) — simplified from Albini
    z_max = 12.2 * flame_height**0.5
    
    # Ember travel distance under wind (simplified)
    # Terminal velocity of typical ember ~5-10 m/s
    v_terminal = 7.0  # m/s
    fall_time = z_max / v_terminal  # seconds
    
    # Horizontal transport = wind × fall time + terrain effect
    spot_distance = wind_speed_10m * fall_time + 0.5 * terrain_elevation_diff
    
    return max(spot_distance, 0)
```

This is a gross simplification of Albini's full model but captures the key dependencies: spotting distance scales with intensity (taller flames loft embers higher), wind speed (embers travel further), and terrain (downhill spotting goes further).

### Integration as Risk Overlay

Don't modify the CA. Instead, compute a spotting risk map as a separate output:

```python
def compute_spotting_risk(ensemble, terrain, wind_field):
    """
    For each cell at the fire front, compute the zone where
    spot fires could ignite. Combine across ensemble members.
    """
    spotting_risk = np.zeros(terrain.shape)
    
    for n in range(ensemble.n_members):
        # Find fire front cells in this member
        front = find_fire_front(ensemble.member_arrival_times[n])
        
        for cell in front:
            # Intensity at this cell
            intensity = ensemble.member_intensity[n, cell[0], cell[1]]
            wind = wind_field[n, cell[0], cell[1]]
            
            # Max spotting distance
            d_max = max_spotting_distance(intensity, wind, 0)
            
            # Wind direction determines spotting direction
            wind_dir = wind_direction[n, cell[0], cell[1]]
            
            # Mark cells within spotting cone
            for target in cells_in_cone(cell, wind_dir, d_max, cone_angle=30):
                spotting_risk[target] += 1.0 / ensemble.n_members
    
    return spotting_risk  # 0 to 1: fraction of members that could spot to each cell
```

### How Drone Wind Data Helps

The spotting distance equation is dominated by wind_speed (linear) and intensity (sub-linear via flame height). Better wind measurements from drones directly tighten the spotting distance prediction. If the ensemble has wide wind uncertainty, the spotting risk zone is a broad smear. After a drone measures wind at a specific location, the ensemble wind spread narrows, and the spotting risk zone sharpens — potentially shrinking from a 3 km uncertainty band to a 500m band.

This doesn't go through the QUBO — the spotting risk map is a separate output for situational awareness. But it creates a feedback loop: the information field may route drones to measure wind at locations where spotting risk is high AND wind uncertainty is high, because narrowing the wind estimate there simultaneously improves both the spread prediction and the spotting risk assessment.

To formalize this feedback, add spotting risk as a consequence weight in the information field:

```python
# Cells in the spotting risk zone get boosted w_i
# because reducing wind uncertainty there also reduces spotting uncertainty
w_total = w_continuous + alpha * binary_entropy + gamma * spotting_risk * wind_variance
```

gamma controls how much the system prioritizes measuring wind in spotting-threatened areas.

### Implementation Cost

- Spotting distance function: ~15 lines
- Risk overlay computation: ~25 lines (can be slow due to per-front-cell loop — optimize or subsample)
- Integration with information field: ~3 lines
- Total: ~45 lines

### What This Does NOT Do

It does not simulate actual ember ignition in the CA. The fire model still has no spotting. The risk overlay is a diagnostic output — it shows where spotting COULD happen and how uncertain that prediction is. It informs drone routing (measure wind in spotting-threatened areas) and situational awareness (warn incident commander about ember risk downwind). It does not change the fire spread prediction.

The honest framing: "The model does not simulate spotting dynamics, but it estimates the spotting risk zone from fire intensity and wind conditions. Drone wind measurements narrow this estimate. Full stochastic spotting simulation is a future extension."

---

## 5. Integration Map

How these additions connect to the existing pipeline:

```
LANDFIRE download
├── elevation, slope, aspect, fuel_model        (existing)
├── canopy_base_height, canopy_bulk_density      (NEW — crown fire)
└── canopy_cover                                 (NEW — Nelson model)
        │
        ▼
Nelson FMC Model ──────────────────────────────── NEW
├── Terrain-aware FMC estimate at every cell
└── Feeds GP as prior mean function
        │
        ▼
GP Prior (existing, improved)
├── Mean: Nelson model (was: RAWS interpolation)
├── Variance: from observation geometry (unchanged)
└── Residuals: GP interpolates corrections to Nelson
        │
        ▼
Ensemble Fire Engine (existing, extended)
├── Surface ROS: Rothermel 1972 (existing)
├── Crown fire check: Van Wagner 1977              NEW
├── Crown fire ROS: Rothermel 1991                  NEW
├── Output: arrival_times + fire_type per member
└── Intensity per member (for spotting overlay)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
Bimodal Detection ── NEW           Spotting Risk Overlay ── NEW
├── burn_fraction                  ├── per-front-cell distance calc
├── bimodal_score                  ├── risk map (0-1)
└── regime_bimodal (crown)         └── feeds situational awareness
        │                                  │
        ▼                                  │
Information Field (existing, augmented)    │
├── w_continuous (unchanged)               │
├── + alpha × binary_entropy        ◄─────┘
├── + beta × regime_entropy          (wind uncertainty × spotting risk)
└── + gamma × spotting_risk × wind_variance
        │
        ▼
Selection + QUBO (unchanged — consumes augmented w_i)
        │
        ▼
EnKF (existing, extended)
├── Unimodal cells: standard Kalman update (existing)
└── Bimodal cells: particle-filter reweighting    NEW
```

---

## 6. Build Priority

|Addition|Lines|Hackathon priority|Why|
|---|---|---|---|
|Nelson FMC prior mean|~35|High|Improves GP physically, easy to implement, makes FMC field terrain-aware|
|Crown fire (Van Wagner + Rothermel 91)|~25|High|Creates the most important discrete event, minimal code, high demo impact|
|Bimodal detection + binary entropy|~25|High|15 lines detection + 10 lines entropy boost. Visible in information field heatmap.|
|Adaptive EnKF (particle filter branch)|~40|Medium|Correct handling of bimodal updates. Can skip if time-pressed — standard EnKF still works, just suboptimal at bimodal cells.|
|Spotting risk overlay|~45|Low|Nice situational awareness output but doesn't change the core sensing loop. Add if time permits on day 4-5.|

Crown fire + bimodal detection together are ~50 lines and create the most compelling demo scenario: the system identifies stands where crown fire transition is uncertain and routes drones to resolve it. This is the headline result beyond the basic information field.