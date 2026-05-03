# IGNIS: Visualization Design Specification

---

## Design Principle

Every visualization answers a specific question that a specific person needs answered. No decorative plots. Each visualization is tagged with who needs it and what decision it informs.

---

## 1. Core System Visualizations

These would exist in a deployed system. They communicate system state to operators and decision-makers.

### 1.1 Fire Prediction Map

**Question:** Where is the fire going?

**Content:** Terrain base layer (hillshade from DEM) with overlaid:

- Current fire perimeter (from ensemble consensus, burn_probability > 0.9)
- Burn probability gradient (0.0 to 1.0, yellow → red colormap)
- Mean predicted arrival time contours (1-hour isochrones)

**Audience:** Incident commander, fire behavior analyst.

```
┌──────────────────────────────────┐
│ ░░░░░░░▓▓▓████████░░░░░░░░░░░░░ │
│ ░░░░░░▓▓██████████▓░░░░░░░░░░░░ │
│ ░░░░░▓████████████▓▓░░░░░░░░░░░ │
│ ░░░░▓██████████████▓░░░░░░░░░░░ │
│ ░░░░▓█████████████▓▓▓░░░░░░░░░░ │
│ ░░░░░▓████████████▓░░░░░░░░░░░░ │
│                                  │
│  █ burned  ▓ P>0.5  ░ P<0.2     │
│  --- 1hr  ─── 2hr  ··· 4hr      │
└──────────────────────────────────┘
```

**Implementation:** `matplotlib.contourf` for burn probability, `matplotlib.contour` for isochrones, hillshade from `matplotlib.colors.LightSource`.

### 1.2 Information Field Heatmap

**Question:** Where should drones go and why?

**Content:** The w_i field overlaid on terrain, showing where measurement is most valuable. This is the central visualization of the entire system.

**Two views:**

**a) Total information value.** Single heatmap of w_total. Hot spots are where drones should go.

**b) Attribution breakdown.** Three-panel or RGB composite showing which variable dominates the information value at each cell:

- Red channel = FMC-driven uncertainty
- Blue channel = wind-driven uncertainty
- Green channel = bimodal/crown fire entropy

A cell that's pure red needs FMC measurement. Pure blue needs wind measurement. Purple (red + blue) needs both. Green needs a binary-resolving measurement near a regime transition.

**Audience:** System operator (for verification), judges (for understanding the concept).

```
┌─────────────────┐ ┌─────────────────┐
│ Total w_i        │ │ Attribution      │
│                  │ │                  │
│   ░░░░░▓▓░░░░   │ │   ░░░░░RR░░░░   │
│   ░░░▓▓██▓░░░   │ │   ░░░RRBB░░░░   │
│   ░░▓████▓░░░   │ │   ░░BBBBB░░░░   │
│   ░░░▓▓█▓░░░░   │ │   ░░░GG░░░░░░   │
│   ░░░░░▓░░░░░   │ │   ░░░GG░░░░░░   │
│                  │ │                  │
│  ▓ high  ░ low   │ │  R=FMC B=wind   │
│                  │ │  G=regime shift  │
└─────────────────┘ └─────────────────┘
```

### 1.3 GP Uncertainty Field

**Question:** Where do we lack data?

**Content:** The GP posterior standard deviation for FMC and wind, overlaid with observation locations (RAWS as circles, drone measurements as triangles, satellite footprints as squares). Shows data coverage gaps.

**Key feature:** This updates visually after each assimilation cycle. Cells near new drone observations drop in uncertainty. Cells where observations have aged (temporal decay) gradually increase.

**Audience:** System operator — verifies the GP is behaving correctly. Judges — demonstrates that drone observations actually reduce uncertainty.

### 1.4 Drone Placement Map

**Question:** Where are drones going and why these locations?

**Content:** Fire prediction map with drone target locations overlaid. Each target is annotated with:

- Rank (1st, 2nd, ... Kth priority)
- Dominant variable (FMC or wind icon)
- Information value (size of marker proportional to w_i)

Connect targets with planned flight paths (lines). Color-code by drone assignment.

Show substitutes as smaller markers with dotted connections to their primary target.

**Audience:** Drone operators (in deployment), judges (to see the selection logic).

### 1.5 Mission Queue Display

**Question:** What should the UTM act on?

**Content:** Ranked table of mission requests with columns:

- Rank
- Location (lat/lon or grid cell)
- Information value
- Dominant variable
- Expiry time
- Status (pending / in-progress / completed / rejected)

**Audience:** UTM operator. This is the text-based counterpart to the placement map.

### 1.6 Crown Fire Risk Overlay

**Question:** Where might the fire blow up?

**Content:** Cells colored by crown fire transition probability (fraction of ensemble members where Van Wagner criterion is met). Overlaid on terrain with canopy base height contours.

**Key feature:** Cells with 20-80% crown fire probability (the bimodal zone) are highlighted with a distinct outline — these are where a single measurement could resolve whether catastrophic transition occurs.

**Audience:** Fire behavior analyst, incident commander.

### 1.7 Spotting Risk Overlay

**Question:** Where might embers land?

**Content:** Spotting risk probability map (from the ensemble-averaged Albini calculation) overlaid on terrain. Wind vectors shown as arrows. High-risk downwind zones highlighted.

**Audience:** Operations section chief, structure protection specialists.

---

## 2. Simulation & Evaluation Visualizations

These exist only for the hackathon to validate the system and demonstrate the comparison results. Not part of a deployed system.

### 2.1 Ensemble Spread Visualization

**Question:** Is the ensemble producing meaningful variance?

**Content:** Grid of 4-9 individual ensemble member fire perimeters overlaid on the same terrain map. Shows how different perturbations produce different fire trajectories.

**Key feature:** Visually demonstrates why ensemble methods are necessary — a single simulation gives one perimeter, but the truth could be any of these.

**Audience:** Judges — immediate visual understanding of uncertainty.

```
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Member 1 │ │ Member 2 │ │ Member 3 │
│   ╱▓▓╲   │ │  ╱▓▓▓╲  │ │   ╱▓╲    │
│  ╱████╲  │ │ ╱█████╲ │ │  ╱███╲   │
│ ╱██████╲ │ │╱███████╲│ │ ╱█████╲  │
└──────────┘ └──────────┘ └──────────┘
  FMC=0.08     FMC=0.06     FMC=0.12
  Crown: No    Crown: Yes   Crown: No
```

### 2.2 Arrival Time Distribution at Key Cells

**Question:** Is the ensemble bimodal where we think it is?

**Content:** Histogram of arrival times across ensemble members for 3-4 selected cells:

- A cell well behind the fire (all members agree — sharp peak)
- A cell at the fire boundary (bimodal — some burn, some don't)
- A cell where crown fire transition is uncertain (bimodal with widely separated modes)
- A cell well ahead with high FMC uncertainty (broad unimodal)

**Audience:** Technical reviewers. Demonstrates that the bimodal detection is identifying real distributional structure, not noise.

### 2.3 Four-Way Strategy Comparison

**Question:** Does targeted placement outperform uniform?

**Content:** Four side-by-side maps showing drone placements for each strategy (greedy, QUBO, uniform, fire-front) on the same fire scenario, same cycle.

Below: bar chart of entropy reduction per strategy for this cycle.

**Key feature:** Visual makes the argument instantly — targeted strategies place drones at terrain transitions and high-uncertainty regions, while uniform scatters them evenly including over irrelevant areas.

```
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Greedy   │ │ QUBO     │ │ Uniform  │ │FireFront │
│  ·       │ │  ·       │ │ ·  ·  ·  │ │          │
│     ·    │ │    ·     │ │          │ │ ····     │
│       ·  │ │      ·   │ │ ·  ·  ·  │ │ ····     │
│  ·    ·  │ │  ·   ·   │ │          │ │          │
│          │ │          │ │ ·  ·  ·  │ │          │
│ PERR:670 │ │ PERR:698 │ │ PERR:426 │ │ PERR:469 │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 2.4 Entropy Convergence Curve

**Question:** Does the system improve over multiple cycles?

**Content:** Line plot of total predictive entropy (y-axis) vs. cycle number (x-axis), one line per strategy. Targeted strategies should converge faster (steeper descent) than baselines.

**Key feature:** This is the primary quantitative result. If the greedy/QUBO lines drop below uniform/fire-front, the concept is validated. The gap between the lines is the value-add of information-theoretic targeting.

```
Entropy
  │╲
  │ ╲·····  uniform
  │  ╲
  │   ╲····· fire-front
  │    ╲
  │     ╲···· greedy
  │      ╲
  │       ╲·· QUBO
  │
  └──────────── Cycle
  1  2  3  4  5  6  7  8
```

### 2.5 Drone Value Curve

**Question:** How many drones are worth deploying?

**Content:** Plot of marginal information gain (y-axis) vs. drone number (x-axis) from the greedy selector's sequential gains. Shows diminishing returns. The knee of the curve is the operationally optimal fleet size.

**Audience:** Resource managers (in deployment) and judges (demonstrates the system provides actionable resource allocation guidance).

### 2.6 QUBO vs Greedy Overlap Analysis

**Question:** Are greedy and QUBO finding different solutions?

**Content:** Per-cycle Jaccard similarity between greedy and QUBO selections. Plotted over cycles. When Jaccard is high (>0.8), they agree — QUBO adds little. When Jaccard is low (<0.5) and QUBO outperforms on PERR, the QUBO is finding complementary placements that greedy misses.

Side-by-side maps for cycles where they disagree most, highlighting the locations QUBO selects that greedy doesn't.

### 2.7 Placement Stability Plot

**Question:** How fast does the optimal placement change?

**Content:** Jaccard similarity between the primary strategy's selections at consecutive cycles. High stability (>0.7) means the system doesn't need to completely recompute every cycle — operationally feasible. Low stability (<0.3) means the fire is evolving faster than the sensing loop.

**Audience:** Technical reviewers assessing operational feasibility.

### 2.8 Ground Truth Reveal

**Question:** Was the system measuring the right things?

**Content:** Side-by-side: the hidden ground truth FMC field vs. the system's GP estimate after N cycles of drone observation. Overlay drone observation locations. Show residual map (ground truth minus estimate).

**Key feature:** Regions where drones observed should have low residual. Regions drones never visited may have high residual — but if those regions had low sensitivity (fire never reaches them), the high residual doesn't matter. This demonstrates that the system correctly prioritized measurement where it mattered, not where uncertainty was highest per se.

### 2.9 Innovation Tracking

**Question:** Is the model learning from observations?

**Content:** Plot of innovation magnitude (|observation - prediction|) at drone measurement locations over cycles. If innovations decrease over time, the model is learning — predictions at newly observed locations are closer to truth because nearby observations have already constrained the field.

If innovations remain high, the model is either wrong (systematic bias) or the fire environment is changing faster than the system learns.

### 2.10 Data-Limited vs Model-Limited Test

**Question:** Does better data beat a better model?

**Content:** If implemented — comparison plot showing prediction accuracy (vs ground truth) for:

- Simple CA model + IGNIS-targeted observations (5 drones)
- Simple CA model + RAWS-only data (no drones)

If the first line is below the second, wildfire prediction is data-limited. This is a single plot that validates the entire thesis.

---

## 3. Presentation-Specific Visualizations

Not part of the software — static slides built from system outputs.

### 3.1 The Observation Gap Slide

**Content:** Map of western US showing RAWS station locations (~2,200 dots spaced ~50 km apart). Zoom to a specific fire perimeter. Count RAWS stations within 50 km. Show the gap.

**Punchline:** "Within this fire perimeter, there are zero ground-truth measurements of fuel moisture. Everything the model uses is interpolated from 40 km away."

### 3.2 Architecture Diagram

**Content:** Simplified version of the data flow from the comprehensive design doc. No code, no equations — boxes and arrows showing the loop: predict → identify uncertainty → route drones → measure → update → repeat.

### 3.3 Before/After Information Field

**Content:** Side-by-side information field heatmaps from cycle 1 (before any drone observations) and cycle 5 (after four rounds of targeted sensing). The progressive darkening of the heatmap (uncertainty reduction) is the visual proof that the system works.

---

## 4. Implementation

### Package Structure

```
ignis/
└── visualization/
    ├── __init__.py
    ├── core.py          # 1.1-1.7: operational visualizations
    ├── evaluation.py    # 2.1-2.10: simulation comparison plots
    └── presentation.py  # 3.1-3.3: static slide generation
```

### Shared Style

```python
VIZ_CONFIG = {
    "terrain_cmap": "terrain",
    "burn_prob_cmap": "YlOrRd",
    "info_field_cmap": "inferno",
    "uncertainty_cmap": "viridis",
    "crown_risk_cmap": "PuRd",
    "drone_marker": "^",
    "raws_marker": "o",
    "satellite_marker": "s",
    "figure_dpi": 150,
    "font_size": 10,
}
```

All visualizations use consistent colormaps and marker styles. Fire-related quantities use warm colormaps (YlOrRd). Uncertainty/information uses perceptual colormaps (inferno, viridis). Terrain uses hillshade.

### Priority

|Visualization|Priority|Effort|Impact|
|---|---|---|---|
|1.2 Information field heatmap|Highest|Low|The single most important image|
|2.4 Entropy convergence curve|Highest|Low|The primary quantitative result|
|2.3 Four-way strategy comparison|Highest|Medium|Visual proof of concept|
|1.1 Fire prediction map|High|Medium|Context for everything else|
|2.1 Ensemble spread|High|Low|Explains why ensembles matter|
|1.4 Drone placement map|High|Low|Shows what the system does|
|2.5 Drone value curve|Medium|Low|Answers "how many drones"|
|2.8 Ground truth reveal|Medium|Medium|Validates measurement targeting|
|1.3 GP uncertainty field|Medium|Low|Shows data assimilation effect|
|3.1 Observation gap slide|Medium|Low|Opens the presentation|
|2.2 Arrival time distributions|Medium|Low|Validates bimodal detection|
|1.6 Crown fire risk overlay|Low|Low|Nice to have with crown fire model|
|2.6 QUBO vs greedy overlap|Low|Low|For technical reviewers|
|2.9 Innovation tracking|Low|Low|Model learning diagnostic|
|2.10 Data-limited test|Low|Medium|Thesis validation if time permits|
|1.7 Spotting risk overlay|Lowest|Medium|Only if spotting overlay is built|