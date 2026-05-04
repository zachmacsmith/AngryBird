# GP Temporal Decay Specification

## Problem

Observations accumulate with full weight forever. After 2-5 cycles the GP posterior variance is near-zero everywhere. The information field collapses. Drones stop being routed.

## Solution

Each observation's influence decays exponentially with age. The GP observation noise grows over time:

```python
def effective_sigma(obs_sigma, obs_time, current_time, tau):
    """
    obs_sigma: original measurement noise
    obs_time: when the observation was taken (seconds)
    current_time: current simulation time (seconds)
    tau: decay timescale (seconds)
    Returns: inflated sigma that reflects staleness
    """
    age = current_time - obs_time
    decay = np.exp(age / tau)  # grows exponentially with age
    return obs_sigma * decay
```

Fresh observation: decay ≈ 1.0, sigma unchanged. After one tau: sigma × 2.7. After two tau: sigma × 7.4. The GP naturally treats stale observations as less informative.

## Decay Constants

From fuel moisture timelag physics (Nelson 2000):

```python
TAU = {
    "fmc_1hr":  3600.0,    # 1 hour in seconds
    "fmc_10hr": 36000.0,   # 10 hours
    "wind":     1800.0,    # 30 minutes — wind decorrelates fast
}
```

## Implementation

In the GP fitting step, before each cycle:

```python
def prepare_observations(all_observations, current_time, tau_dict):
    prepared = []
    for obs in all_observations:
        tau = tau_dict[obs.variable_type]
        sigma_eff = effective_sigma(obs.sigma, obs.timestamp, current_time, tau)
        
        # Drop observations that have decayed beyond usefulness
        if sigma_eff > 10 * obs.sigma:  # 10× original noise = negligible information
            continue
        
        prepared.append(obs._replace(sigma=sigma_eff))
    return prepared
```

This is applied every cycle before GP fitting. Old observations either contribute with inflated noise or get pruned entirely.

## Three Floors

Temporal decay solves the accumulation problem. Two additional floors prevent edge-case collapse:

```python
AGGREGATION_SIGMA_FLOOR = 0.015   # even a perfect average of 1000 readings has this much systematic error
PROCESS_NOISE_FLOOR = 0.01        # minimum FMC uncertainty per cycle, injected into ensemble perturbations

# In aggregation:
sigma_agg = max(sigma_individual / sqrt(n_readings), AGGREGATION_SIGMA_FLOOR)

# In ensemble generation:
perturbation_scale = np.maximum(np.sqrt(gp_variance), PROCESS_NOISE_FLOOR)
```

Temporal decay handles the normal case. The floors handle the pathological cases (huge observation counts, perfectly observed regions with changing conditions).