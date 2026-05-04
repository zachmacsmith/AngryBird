# WISP: Implementation Considerations

Companion to the comprehensive design document. Covers pitfalls that will cause silent bugs if not addressed.

---

## 1. Coordinate Reference System

**Problem:** The system ingests lat/lon coordinates (LANDFIRE, RAWS, satellite) but all internal computation — GP kernels, localization taper, spacing filters, path planning — requires distances in meters. Operating on raw lat/lon produces distorted distances (1° longitude ≠ 1° latitude, and both vary with location) and triggers expensive haversine calculations where Euclidean arithmetic suffices.

**Fix:** Project to local UTM at ingestion. Compute the UTM zone from the input bounding box. All internal math operates in meters on a flat grid. Convert back to lat/lon only at the mission queue output boundary.

```python
def get_utm_crs(lon, lat):
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"

# In terrain loader, computed from bounding box center:
utm_crs = get_utm_crs(center_lon, center_lat)
transformer = Transformer.from_crs("EPSG:4326", utm_crs)
```

**Rule:** No component except the terrain loader and mission queue builder touches lat/lon. Everything internal is (x_meters, y_meters).

---

## 2. EnKF Shape Reference

Print this and keep it visible during implementation. Most EnKF bugs are shape mismatches.

```
N       = ensemble size (200)
D       = state dimension (rows × cols, or rows × cols × n_variables)
n_obs   = number of observations this cycle (20-50)

X:           (N, D)           ensemble state matrix
A:           (N, D)           anomalies = X - X.mean(axis=0)
HX:          (N, n_obs)       states at observation locations = X[:, obs_indices]
HA:          (N, n_obs)       anomalies at obs locations = HX - HX.mean(axis=0)
PHT:         (D, n_obs)       = A.T @ HA / (N-1)
HPHT:        (n_obs, n_obs)   = HA.T @ HA / (N-1)
R:           (n_obs, n_obs)   observation noise covariance (diagonal)
K:           (D, n_obs)       Kalman gain = PHT @ inv(HPHT + R)
innovation:  (n_obs,)         = y - HX[n] (per member)
update:      (D,)             = K @ innovation (per member)
```

**H is never materialized as a matrix.** The observation operator is an index gather, not a matrix multiply:

```python
# Correct — gather operation
obs_indices = [row * cols + col for row, col in obs_locations]
HX = X[:, obs_indices]

# Wrong — wasteful sparse matrix
H = np.zeros((n_obs, D))
for j, idx in enumerate(obs_indices):
    H[j, idx] = 1.0
HX = (H @ X.T).T
```

Both produce identical results. The gather is faster and eliminates the possibility of shape errors from a malformed H.

---

## 3. Unburned Cell Representation

**Problem:** Cells that never burn in a simulation have no arrival time. Representing this as NaN or 0 corrupts every downstream computation.

- NaN propagates: `np.corrcoef` on arrays containing NaN returns NaN. Sensitivity becomes NaN. QUBO coefficients become NaN. Solver crashes or produces garbage.
- Zero is worse: the GP interprets arrival_time = 0 as "fire arrived at t=0," i.e., already burning at initialization.

**Fix:** Use a finite sentinel value:

```python
MAX_ARRIVAL = 2.0 * horizon_hours  # e.g., 12.0 for a 6-hour forecast
arrival_times = np.full((rows, cols), MAX_ARRIVAL, dtype=np.float32)
```

**Why 2× horizon, not 1× or infinity:**

If some ensemble members burn a cell at hour 5.5 and others don't, the disagreement is real — the ensemble genuinely doesn't know if this cell burns. Members that don't burn it have arrival_time = MAX_ARRIVAL = 12.0. The variance between 5.5 and 12.0 is large, correctly reflecting the uncertainty. If MAX_ARRIVAL = 6.01, the variance between 5.5 and 6.01 is near zero, hiding the disagreement. If MAX_ARRIVAL = inf, the variance is inf, overwhelming all other signals.

**Additional NaN guard in sensitivity computation:**

When all ensemble members agree on arrival time at a cell (zero variance), `corrcoef` returns NaN via division by zero. Guard explicitly:

```python
std_A = A_centered.std(axis=0)
std_A[std_A < 1e-10] = 1e-10  # prevent div-by-zero, not just numerical convenience
sensitivity = cov / (std_A * std_p + 1e-10)
```

---

## 4. GPU/CPU Swappability

**Design principle:** Isolate the compute backend inside the fire engine. Everything else stays NumPy. The interface between components is always numpy arrays.

```python
class FireEngine(Protocol):
    def run(self, snapshot: CycleSnapshot, config: EnsembleConfig) -> EnsembleResult:
        """Returns EnsembleResult with numpy arrays, regardless of backend."""
        ...

class NumpyFireEngine:
    """CPU backend. Parallelized via multiprocessing.Pool."""
    def run(self, snapshot, config) -> EnsembleResult:
        with Pool(n_cores) as pool:
            results = pool.map(self._run_member, parameter_sets)
        return EnsembleResult(member_arrival_times=np.stack(results), ...)

class TorchFireEngine:
    """GPU backend. All members batched along dim 0."""
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
    
    def run(self, snapshot, config) -> EnsembleResult:
        # ... batched computation on GPU ...
        return EnsembleResult(member_arrival_times=result.cpu().numpy(), ...)
```

**The conversion boundary:** `result.cpu().numpy()` is one memory copy per cycle. At 200 members × 200×200 grid × 4 bytes = 32 MB, this takes ~10 ms. Negligible relative to the ensemble compute.

**Orchestrator selects at configuration time:**

```python
if torch.cuda.is_available() and config.use_gpu:
    engine = TorchFireEngine(device="cuda")
else:
    engine = NumpyFireEngine(n_cores=os.cpu_count())
```

**No other component needs to know.** The uncertainty decomposition, QUBO builder, EnKF, GP, evaluator — all receive numpy arrays from EnsembleResult and operate in NumPy. Swapping the fire engine is one line in config.

**Pragmatic dual-backend approach for the Rothermel CA:**

Most of the Rothermel equation is elementwise arithmetic that works identically in NumPy and PyTorch:

```python
class RothermelCA:
    def __init__(self, backend="numpy"):
        self.xp = np if backend == "numpy" else torch
    
    def compute_ros(self, fmc, wind_speed, slope):
        xp = self.xp
        rm = fmc / self.mx
        eta_m = xp.clip(1 - 2.59*rm + 5.11*rm**2 - 3.52*rm**3, 0, 1)
        Q_ig = 250 + 1116 * fmc
        # ... identical arithmetic
```

This works for ~80% of the code. The remaining 20% (convolutions for neighbor counting, random number generation, advanced indexing) needs backend-specific paths.

**Build order:** NumPy first. Get the full pipeline working on CPU. Port to PyTorch only if ensemble speed is a bottleneck (day 4). The Protocol interface means nothing else changes.

---

## 5. Temporal Kernel Calibration

**Problem:** The GP temporal kernel decay rate τ determines how fast old measurements lose relevance. Arbitrary values produce either overconfident estimates (τ too large — stale measurements treated as current) or wasteful re-measurement (τ too small — drone revisits locations unnecessarily).

**Fix:** τ is not a tunable hyperparameter. It's a physical constant of the fuel class:

```python
TEMPORAL_DECAY = {
    "fmc_1hr":  1.0,    # hours — fine dead fuels equilibrate in ~1 hour
    "fmc_10hr": 10.0,   # hours — small branches
    "wind_speed": 0.5,  # hours — conservative for complex terrain
    "wind_direction": 1.0  # hours — stable synoptic conditions
}
```

These values come from the Nelson fuel moisture model (2000) and the NFDRS timelag classification system. They can be refined empirically by fitting the temporal kernel to RAWS time series (stations report hourly, giving direct autocorrelation measurements), but the published timelags are defensible defaults.

**Fire arrival time observations do not decay.** Once a cell has burned, that's permanent. Set τ = ∞ (or simply exclude fire arrival observations from temporal decay) for fire state variables.

---

## 6. Multi-Modal Observation Handling

All observation types enter the same EnKF update. They differ only in observation operator and noise:

```python
OBSERVATION_SPECS = {
    "drone_fmc": {
        "state_variable": "fmc",
        "noise_sigma": 0.05,
        "spatial_footprint": 1,      # cells — point measurement
        "temporal_decay": True,       # subject to τ decay
    },
    "drone_wind": {
        "state_variable": "wind_speed",
        "noise_sigma": 1.0,          # m/s
        "spatial_footprint": 1,
        "temporal_decay": True,
    },
    "satellite_fire": {
        "state_variable": "fire_arrival_time",
        "noise_sigma": 0.2,          # binary detection uncertainty
        "spatial_footprint": 8,       # ~375m VIIRS pixel / 50m grid ≈ 8 cells
        "temporal_decay": False,      # fire state is permanent
        "update_method": "reweight",  # particle-filter-like, not Gaussian
    },
    "satellite_fmc": {
        "state_variable": "fmc",
        "noise_sigma": 0.15,
        "spatial_footprint": 20,      # ~1km MODIS pixel / 50m grid
        "temporal_decay": True,
    },
    "raws_fmc": {
        "state_variable": "fmc",
        "noise_sigma": 0.03,
        "spatial_footprint": 1,
        "temporal_decay": True,
    },
}
```

For satellite observations that span multiple cells, the observation operator averages over the footprint:

```python
if footprint > 1:
    # obs_indices includes all cells within the satellite pixel
    HX = X[:, obs_indices].mean(axis=1)  # (N,) — average over footprint
else:
    HX = X[:, obs_index]                  # (N,) — point measurement
```

The EnKF handles the rest — noisier, coarser observations produce smaller Kalman gain and thus smaller updates, automatically weighting them appropriately against precise drone measurements.

---

## 7. Burned Cell Exclusion

The EnKF automatically ignores burned cells (zero ensemble variance → zero Kalman gain → zero update). No explicit masking required at hackathon scale.

For reference, the masking approach (reducing D to D_active) saves computation proportional to the fraction of burned cells but requires an index remapping:

```python
full_to_active = np.full(D, -1, dtype=int)
active = burn_prob.ravel() < 0.95
full_to_active[active] = np.arange(active.sum())
obs_indices_active = full_to_active[obs_indices_full]
assert np.all(obs_indices_active >= 0), "Observation falls on burned cell"
```

**Skip this for the hackathon.** The zero-variance mechanism handles it correctly. The masking saves ~10-20ms per update on a 200×200 grid — invisible in a 20-minute cycle. Add it when scaling to 1000×1000+ grids.

---

## 8. Random Seed Management

Reproducibility matters for the comparison framework. If greedy and QUBO produce different results, you need to know it's because of the selection algorithm, not because different ensemble members were generated.

```python
# At cycle start, set seed from cycle count
rng = np.random.default_rng(seed=config.base_seed + cycle_id)

# Use this rng for ALL random operations in the cycle:
# - Ensemble perturbation generation
# - Stochastic CA ignition
# - Simulated observation noise
# - EnKF perturbed observations

# Pass rng explicitly, never use np.random.* global functions
perturbation = rng.normal(0, sigma, size=grid_shape)
```

All four placement strategies evaluate against the same ensemble from the same seed. The comparison is clean.

---

## 9. Common Silent Failure Modes

|Bug|Symptom|Cause|Fix|
|---|---|---|---|
|Sensitivity is NaN at many cells|QUBO produces garbage or crashes|Unburned cells stored as NaN, or zero-variance cells hit div-by-zero|Use MAX_ARRIVAL sentinel + epsilon guard in denominator|
|All drones sent to same corner of domain|QUBO clusters selections despite J_ij|Lat/lon distances used instead of meters — spatial correlation is distorted|Project to UTM at ingestion|
|EnKF update makes things worse|Posterior variance increases after observation|Localization radius too large — spurious long-range correlations in small ensemble|Reduce localization radius or increase ensemble size|
|Fire spreads at wrong speed|ROS off by 10-100×|Unit mismatch — Rothermel uses mixed imperial/metric internally|Validate single-cell ROS against BehavePlus for same fuel model + conditions|
|GP variance is uniform everywhere|Information field shows no spatial structure|Terrain covariates not included in kernel, or kernel length scale too large|Add terrain distance to kernel, fit length scale to RAWS data|
|Greedy and QUBO give identical results every cycle|No information about QUBO value-add|K too small for complementary pairs to matter, or QUBO penalty λ miscalibrated|Test at K=10-20, verify λ ≈ max(|
|Ensemble collapses (all members converge)|Zero variance everywhere after a few cycles|No inflation — EnKF systematically underestimates posterior variance|Apply inflation factor >1.0 to ensemble anomalies after each update|

---

## 10. Validation Checkpoints

Before integrating components, validate each in isolation:

| Component         | Validation                                                     | Expected Result                                                                                       |
| ----------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Rothermel ROS     | Compute ROS for fuel model 1, FMC=0.06, wind=5 m/s, slope=20°  | Compare to BehavePlus output (±5%)                                                                    |
| CA fire spread    | Run on flat terrain, uniform fuel, no wind                     | Circular spread pattern                                                                               |
| CA fire spread    | Run on flat terrain, uniform fuel, 10 m/s wind                 | Elliptical spread elongated downwind                                                                  |
| GP prior          | Place 3 RAWS-like stations on grid, compute variance           | Variance near zero at stations, increases smoothly with distance, follows terrain kernel              |
| Ensemble          | Run 200 members, perturb FMC ±20%                              | Arrival time variance highest at cells where FMC matters for spread (ahead of fire in dry fuel)       |
| Sensitivity       | Correlate arrival time with FMC perturbation per cell          | High sensitivity ahead of fire, zero behind, highest in cells where fire path crosses fuel transition |
| Information field | Overlay w_i heatmap on fire prediction                         | High values at terrain transitions ahead of fire, low near RAWS, zero behind fire                     |
| Greedy selector   | Select K=5 from synthetic info field with two distinct peaks   | Selects from both peaks, not all 5 from the highest peak                                              |
| EnKF update       | Observe FMC at one cell, check variance reduction at neighbors | Neighbors with same terrain features see largest reduction, distant/dissimilar cells unchanged        |