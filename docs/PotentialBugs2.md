# IGNIS: Implementation Addendum

Three items not adequately covered in existing documentation.

---

## 1. Observation Thinning

A drone swath at 50m grid resolution produces ~1,080 cell observations per 20-minute sortie. Most are within one GP correlation length (~1-2 km) of each other and therefore nearly redundant. Feeding all 1,080 into the EnKF means inverting a 1,080×1,080 HPHT matrix (slow, numerically unstable). Feeding them into the GP risks covariance matrix singularity.

Add a thinning step between observation collection and assimilation:

```python
def thin_observations(obs_list, min_spacing_m=200):
    thinned = []
    for obs in sorted(obs_list, key=lambda o: o.fmc_sigma):  # keep lowest noise
        if all(distance(obs.location, t.location) > min_spacing_m for t in thinned):
            thinned.append(obs)
    return thinned
```

1,080 raw observations → ~50 thinned observations. HPHT drops to 50×50. Also add jitter to manual GP variance updates (the greedy selector's conditional update bypasses scikit-learn's built-in nugget):

```python
sigma2_updated = sigma2 - k_xnew**2 / (k_newnew + sigma_noise**2 + 1e-6)
sigma2_updated = np.maximum(sigma2_updated, 1e-10)
```

---

## 2. Ensemble Inflation

The implementation considerations doc mentions covariance collapse but doesn't specify where inflation goes in the orchestrator. Make it explicit — inflate after every EnKF update, before the next ensemble generation:

```python
# In orchestrator, immediately after enkf_update returns:
mean = updated_states.mean(axis=0)
anomalies = updated_states - mean
updated_states = mean + config.inflation_factor * anomalies  # 1.02-1.10
```

This works in tandem with the GP temporal decay (which re-expands perturbation scales between cycles). Inflation prevents intra-cycle collapse; temporal decay prevents inter-cycle collapse. Without inflation, the system will appear to work for 3-4 cycles then stop sending drones as the information field goes to zero everywhere.

---

## 3. Temporal Smearing

A drone flying a 5 km path takes ~20 minutes. Early and late observations from the same sortie are assimilated simultaneously as if they occurred at the same instant.

At moderate fire ROS (~2 m/min), the fire advances ~40 meters in 20 minutes — less than one 50m grid cell. Wind can shift faster, but the anemometer reads local conditions at measurement time regardless of when it's assimilated. The smearing is sub-grid-scale under moderate conditions.

No action needed for the hackathon. Acknowledge the approximation. If challenged: "temporal smearing within a 20-minute cycle is sub-grid at moderate spread rates." For fast-moving fires or longer cycles, the fix is batching observations into 5-minute temporal windows and assimilating each against the ensemble state at that window's midpoint — requiring ensemble state checkpoints at ~4 points per cycle rather than storing the full trajectory.