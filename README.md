# IGNIS — Intelligent Guidance for Networked Information Systems

Autonomous drone fleet planning for wildfire situational awareness. IGNIS quantifies where fire models are most uncertain, directs drones to resolve those uncertainties, and assimilates observations every ~20 minutes to sharpen predictions in real time.

---

## What it does

Every IGNIS cycle:
1. **GP Prior** — fuses RAWS station readings, drone observations, and Nelson FMC physics into spatially varying estimates of fuel moisture and wind (with uncertainty).
2. **Fire Ensemble** — runs N perturbed fire spread members to produce burn probability and arrival-time variance across the grid.
3. **Information Field** — computes where GP variance × fire sensitivity × drone observability is highest — the cells where a measurement would most reduce prediction uncertainty.
4. **Selection** — greedy or QUBO solver picks the k cell locations that maximize cumulative information gain.
5. **Assimilation** — drone observations update the GP posterior and EnKF ensemble; the cycle repeats.

---

## Repository layout

```
angrybird/          Core package (ships to production)
  gp.py               Gaussian Process prior — FMC, wind speed, wind direction
  orchestrator.py     Sequences the full IGNIS cycle
  information.py      Information field computation
  assimilation.py     EnKF + GP observation ingestion
  observations.py     Centralized ObservationStore (RAWS, drone, satellite)
  fire_state.py       FireStateEstimator, EnsembleFireState, ConsistencyChecker
  nelson.py           Nelson dead fuel moisture model
  raws.py             RAWS station types and observer
  tif_getter.py       LANDFIRE GeoTIFF download → TerrainData
  selectors/          Greedy and QUBO placement strategies
  visualization/      Operational and evaluation plots

simulation/         Simulation harness (dev/eval only — not production)
  runner.py           SimulationRunner (clock-based) and CycleRunner (cycle-based)
  scenarios.py        Built-in scenarios: hilly_heterogeneous, dual_ignition, etc.
  simple_fire.py      Huygens elliptical CPU fire engine (lightweight fallback)
  gpu_fire_engine.py  PyTorch GPU fire engine with crown fire and terrain slope
  ground_truth.py     Synthetic terrain, FMC, and wind field generation
  renderer.py         6-panel frame renderer → MP4 video

scripts/
  run_sim.py          PRIMARY entry point — clock-based simulation with video output

archived/           Deprecated scripts (do not run)
docs/               Architecture and design specifications
changelogs/         Per-agent change logs
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Run the default scenario (hilly terrain, 6 h, 5 drones)
python scripts/run_sim.py

# 1-hour dual-ignition stress test
python scripts/run_sim.py --scenario dual_ignition --hours 1

# Faster iteration: fewer members, shorter run
python scripts/run_sim.py --members 10 --hours 1 --drones 3
```

Output frames and an MP4 video are written to `out/<scenario_name>/`.

### Available scenarios

| Name | Description |
|------|-------------|
| `hilly_heterogeneous` | Ridge/valley terrain, mixed fuels, SW wind + 30° shift at hour 2 |
| `wind_shift` | Same terrain, 45° wind shift at hour 3 |
| `dual_ignition` | Two simultaneous fires, 45° wind shift at t=30 min, 1-hour run |
| `flat_homogeneous` | Flat control baseline — minimal drone advantage expected |
| `crown_fire_risk` | Dense timber, high FMC, crown fire conditions |

New scenarios are auto-discovered: add a function to `simulation/scenarios.py` and it appears in `--scenario` choices automatically.

---

## Download real terrain

```python
from angrybird.tif_getter import download_terrain

terrain = download_terrain(
    bbox=(-122.7, 37.6, -122.2, 38.0),  # (min_lon, min_lat, max_lon, max_lat)
    resolution_m=50.0,
)
```

Uses the LANDFIRE Product Service API. Requires `rasterio` and `pyproj`.

---

## Key dependencies

| Package | Purpose |
|---------|---------|
| `numpy`, `scipy` | Array math, KD-tree fire spread |
| `scikit-learn` | Gaussian Process regression |
| `torch` | GPU fire engine (optional — falls back to SimpleFire) |
| `rasterio`, `pyproj` | GeoTIFF I/O and reprojection |
| `matplotlib` | Visualization and video frames |
| `requests`, `tqdm` | LANDFIRE tile download |
