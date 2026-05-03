# IGNIS: Build Order & Development Timeline

## Principle

Build bottom-up along the data flow. Each step produces a testable output before the next begins. Never write code that depends on a component that doesn't exist yet.

---
## Phase 0: Shared Foundation (2-3 hours, everyone)

**Before anyone writes component code.**

1. `types.py` — All frozen dataclasses: TerrainData, GPPrior, EnsembleResult, InformationField, SelectionResult, DroneObservation, MissionRequest. Agree on array shapes and field names.
    
2. `config.py` — All constants: grid resolution, ensemble size, perturbation ranges, fuel parameters lookup table, noise configs, inflation factor.
    
3. `utils.py` — UTM projection (auto zone detection from bounding box), distance computation, Gaspari-Cohn taper function, observation thinning.
    

**Test:** Everyone can `from ignis.types import *` and `from ignis.config import *`.

---

## Phase 1: Fire Engine (1-1.5 days)

Build the thing that everything else consumes.

1. **Rothermel ROS function.** Pure arithmetic: fuel params + FMC + wind + slope → rate of spread. ~30 lines. **Validate immediately** against BehavePlus for fuel model 1 at known conditions.
    
2. **CA stepper.** Single timestep: given current burn state + ROS grid → updated burn state + arrival times. Handle wind directionality (8-neighbor elliptical weighting). ~40 lines.
    
3. **Single simulation.** Chain CA steps over full time horizon. Input: terrain + FMC field + wind field. Output: arrival time grid. ~20 lines. **Test:** flat terrain, no wind → circular spread. Add wind → elliptical spread.
    
4. **Ensemble wrapper.** Generate N perturbation fields (spatially correlated, for now use fixed perturbation magnitudes — GP-scaled perturbations come in Phase 2). Run N simulations via multiprocessing. Pack into EnsembleResult. ~30 lines. **Test:** 100 members on 100×100 grid runs in <60 seconds.
    

**Deliverable:** `fire_engine.run(terrain, fmc, wind, config) → EnsembleResult`

---

## Phase 2: GP + Information Field (0.5-1 day)

This is where the science lives.

1. **GP prior.** Fit scikit-learn GP to synthetic RAWS locations. Predict mean + variance over grid. ~30 lines. **Test:** variance is near-zero at station locations, increases with distance, respects terrain kernel if implemented.
    
2. **GP-scaled perturbation generation.** Replace fixed perturbation magnitudes in the ensemble with GP variance-scaled correlated fields (circulant embedding / FFT method). ~20 lines. Re-run ensemble — verify variance structure changes (higher spread in data-sparse regions).
    
3. **Sensitivity computation.** Vectorized correlation between arrival times and perturbation fields across ensemble members. ~15 lines. **Test:** sensitivity is high ahead of fire, zero behind.
    
4. **Information field.** `w = gp_variance × sensitivity × observability`. One line of NumPy. **Test:** w_i heatmap overlaid on fire prediction looks physically sensible — high at terrain transitions ahead of fire, low near RAWS, zero behind.
    

**Deliverable:** `compute_information_field(ensemble, gp_prior) → InformationField` with a compelling heatmap. This is a presentable result on its own.

---

## Phase 3: Selection Strategies (0.5-1 day)

Everything here consumes the information field.

1. **Baselines first.** Uniform grid selector and fire-front selector. 15 lines total. Trivial, done in minutes.
    
2. **Greedy selector.** Iterative selection with GP conditional variance update. ~50 lines. **Test:** selects from distinct high-value regions, not all from one cluster.
    
3. **QUBO selector.** Candidate extraction (top-M by w_i with spacing filter). Correlation matrix from ensemble. QUBO matrix assembly. SA solver via Ocean SDK. ~100 lines. **Test:** QUBO solution overlaps substantially with greedy solution (they should agree on most locations at small K). If they completely disagree, debug the QUBO coefficients.
    
4. **D-Wave integration.** Replace SA with QPU submission. Same QUBO matrix, different solver call. ~15 lines on top of SA. May require cloud access setup.
    

**Deliverable:** Four SelectionResult objects per cycle, ready for comparison.

---
### Phase 4a: Core Assimilation (0.5 day)
 
These components are part of the deployed system. In a real deployment with real drones, you ship this code.
 
Lives in `ignis/assimilation.py` and `ignis/orchestrator.py`.
 
1. **EnKF.** Implement the update with localization and inflation. Accepts `list[DroneObservation]` from any source (real or simulated — it doesn't know or care). ~60 lines. **Test:** observe FMC at one cell, verify variance drops at correlated neighbors and is unchanged at distant/dissimilar cells.
2. **GP update.** Add observations to GP conditioning set. Recompute posterior variance. ~5 lines. **Test:** variance drops near observed location, unchanged far away.
3. **Observation thinning.** Reduce dense swath observations to one per correlation length before passing to EnKF. ~15 lines.
4. **Core orchestrator.** The operational cycle: GP → ensemble → info field → select → assimilate incoming observations → repeat. Accepts observations as input, does not generate them. ~60 lines. **Test:** given a list of synthetic observations passed in manually, run one full cycle.
```python
# Core orchestrator interface — observation-source agnostic
queue, situation = orchestrator.run_cycle(observations=drone_data)
```
 
**Deliverable:** `orchestrator.run_cycle(observations) → MissionQueue` works end-to-end with manually provided test observations.
 
### Phase 4b: Simulation Harness (0.5 day)
 
These components exist only for the hackathon. They substitute for real drones and enable the four-way strategy comparison. In a real deployment, none of this ships.
 
Lives in `ignis/simulation/`.
 
1. **Ground truth manager.** Generate and hold the hidden "true" FMC and wind fields that the simulated observer samples from. ~20 lines.
2. **Simulated observer.** Generate synthetic observations at specified locations by sampling ground truth + calibrated noise. Implements the same `ObservationSource` interface that real drone telemetry would. ~30 lines.
3. **Counterfactual evaluator.** For each strategy's selections, simulate what observations would have been collected and how much GP variance would have decreased. Computes PERR and entropy metrics. ~40 lines. **Test:** targeted strategies show higher entropy reduction than uniform.
4. **Simulation runner.** Wraps the core orchestrator with the simulated observer and comparison logic. Runs all four strategies counterfactually each cycle, but only the primary strategy's observations feed into the real state update. ~30 lines.
```python
# Simulation wrapper — hackathon evaluation
from ignis.simulation import SimulationRunner
runner = SimulationRunner(orchestrator, ground_truth, strategies=["greedy", "qubo", "uniform", "fire_front"])
results = runner.run_comparison(n_cycles=10)
```
 
**Deliverable:** Multi-cycle comparison data. PERR numbers for all four strategies.
 
**The test for which folder a file belongs in:** would you ship it with real drones? EnKF → yes → `ignis/`. Simulated observer → no → `ignis/simulation/`.
 
---
---

## Phase 5: Polish (0.5-1 day)

1. **Visualization.** Information field heatmap, drone placement overlay, entropy reduction curves across cycles, drone value curve. Matplotlib is sufficient.
    
2. **Multiple scenarios.** Run comparison on 2-3 fire scenarios: flat homogeneous terrain (control), hilly heterogeneous terrain (where targeting should shine), wind-shift event (stress test).
    
3. **LANDFIRE integration.** If time permits, replace synthetic terrain with real terrain for one scenario.
    
4. **Presentation.** Structure: observation gap → concept → information field → QUBO formulation → results → future extensions.
    

---

## Dependency Graph

```
types.py + config.py + utils.py
         │
         ▼
    Fire Engine  ──────────────────────────────┐
         │                                      │
         ▼                                      │
    GP Prior                                    │
         │                                      │
         ▼                                      │
    Information Field                           │
         │                                      │
    ┌────┴─────────┬──────────┐                 │
    ▼              ▼          ▼                 │
  Greedy        QUBO      Baselines             │
    │              │          │                 │
    └──────┬───────┘──────────┘                 │
           ▼                                    │
    Counterfactual Evaluator                    │
           │                                    │
           ▼                                    │
    Simulated Observer                          │
           │                                    │
           ▼                                    │
    EnKF + GP Update  ◄────────────────────────┘
           │
           ▼
    Orchestrator (wires everything)
           │
           ▼
    Visualization + Presentation
```

No component depends on something below it. Build top-to-bottom in this graph and you never block.

---

## Team Parallelism

After Phase 0, work splits:

- **Person A:** Fire engine (Phase 1) → GPU port if needed (Phase 5)
- **Person B:** GP + information field (Phase 2, starts day 1 PM with synthetic ensemble data, integrates real ensemble on day 2)
- **Person C:** QUBO selector + D-Wave (Phase 3, starts day 2 with synthetic info field, integrates real info field when ready)
- **Person D:** Observer + path planner + UTM interface (Phase 4 partial, starts day 2)
- **Person E:** Orchestrator + EnKF + evaluator + visualization (Phase 4, starts day 2-3 as components become available)

**Critical path:** Fire engine → information field → everything else. Person A must deliver a working ensemble by end of day 1. Everything else can develop against synthetic/mock data until the real components arrive, but the full pipeline can't close until the fire engine works.

---

## What to Cut (if behind schedule)

Cut from the bottom of each phase, in this order:

1. ~~LANDFIRE real terrain~~ → synthetic terrain throughout
2. ~~D-Wave QPU~~ → SA only, show QUBO structure
3. ~~Multiple scenarios~~ → one well-tuned scenario
4. ~~Path-integrated observations~~ → point observations only
5. ~~Wind directionality in CA~~ → isotropic spread (simpler, still demonstrates the concept)
6. ~~Terrain-aware GP kernel~~ → isotropic Matérn (still produces spatial variance structure from station geometry alone)

**Never cut:** The four-way comparison across multiple cycles with entropy reduction plots. This is the result.