# WISP and AB Protocol: Framework Overview and Paper Outline

## 1. Executive Summary
The **Wildfire Intelligence Surveillance Package (WISP)** is an autonomous drone fleet planning framework designed for real-time wildfire situational awareness. At its core is the **AB Protocol (AngryBird)**, an algorithmic routing and planning system that dynamically quantifies where fire models are most uncertain and directs drone fleets to resolve those uncertainties. 

By assimilating sensor observations continuously (e.g., every 20 minutes), WISP sharpens predictive models of fire spread in real time. It effectively bridges the gap between physical fire spread models, Gaussian Process (GP) regression, and advanced combinatorial optimization (including Quantum-ready QUBO formulations) to maximize information gain in dynamic, hazardous environments.

---

## 2. Comprehensive Algorithm Architecture (The WISP Cycle)
The framework operates in a continuous, multi-stage cycle. Each stage is designed to seamlessly flow into the next, maintaining an updated state of the fire and the environment.

### 2.1 Gaussian Process (GP) Prior Estimation
- **Functionality:** Fuses incoming data from RAWS (Remote Automated Weather Stations), drone telemetry, and Nelson fuel moisture physics.
- **Output:** Spatially varying estimates of Fuel Moisture Content (FMC) and wind fields (speed/direction) along with their associated uncertainties (variance).
- **Key Detail:** The GP uses an observation store that continually updates.

### 2.2 Fire Ensemble Simulation
- **Functionality:** Runs $N$ perturbed fire spread members based on the GP priors. Supports both a lightweight CPU engine (Huygens elliptical) and a PyTorch GPU engine (supporting crown fire and terrain slopes).
- **Outputs:**
  - Per-cell burn probability.
  - Arrival-time variance across the grid.
  - Fire state estimation (tracking active perimeters and resolving disagreements between predicted and observed fire locations).

### 2.3 Information Field Computation (The Core Novelty)
- **Functionality:** Calculates a spatially-explicit "Information Value" ($w$) map to determine where a drone measurement would most drastically reduce model uncertainty.
- **Mathematical Formulation:** 
  - $w = (\text{GP Variance}) \times |\text{Sensitivity}| \times (\text{Observability})$
  - **Sensitivity:** The per-cell correlation between fire arrival times and environmental perturbation fields (FMC, Wind).
  - **Observability:** Represents sensor degradation near the active fire front (accounting for smoke and turbulence).
- **Entropy Augmentation:** Incorporates binary entropy terms to highlight regions where ensemble members disagree strongly (e.g., bimodality in burn vs. no-burn predictions, or surface vs. crown fire regimes).

### 2.4 Selection and Routing (AB Protocol)
- **Functionality:** Selects the optimal $K$ cell locations for $K$ drones to maximize cumulative information gain.
- **Strategies:**
  - **Greedy:** A fast, sub-optimal baseline.
  - **QUBO (Quadratic Unconstrained Binary Optimization):** Encodes the selection as a QUBO problem. It maximizes information value while penalizing redundancy (using ensemble covariance to prevent drones from sampling highly correlated areas) and enforcing cardinality (exactly $K$ drones).
- **Solver Fallback Chain:**
  1. **D-Wave QPU:** Cloud-based quantum annealing.
  2. **Simulated Annealing:** Classical heuristic fallback.
  3. **Pure-Greedy:** Emergency failsafe.

### 2.5 Data Assimilation (EnKF)
- **Functionality:** Ingests the new drone observations to update both the GP posterior and the Fire Ensemble.
- **Key Techniques:**
  - **Ensemble Kalman Filter (EnKF):** Performs state updates on the ensemble members.
  - **Observation Aggregation:** Uses inverse-variance weighting to combine dense spatial observations into single representative cells, preventing the GP from being over-constrained.
  - **Localization & Inflation:** Uses Gaspari-Cohn tapers and covariance inflation to prevent ensemble collapse and spurious long-range correlations.

---

## 3. Key Benefits and Novel Contributions
*Ideal sections to highlight in a research paper.*

1. **Closed-Loop Uncertainty Reduction:** Unlike traditional static routing, WISP dynamically adjusts flight paths based on the *current* uncertainty of the predictive ensemble, creating a tight feedback loop between physics-based models and robotic sampling.
2. **Quantum-Ready Routing (QUBO):** The AB Protocol is uniquely formulated to leverage near-term quantum hardware (D-Wave). The explicit handling of sensor redundancy via covariance matrices is a significant step forward for multi-agent path planning.
3. **Robust Data Assimilation:** The system gracefully handles dense, noisy data through precision-weighted aggregation and EnKF localization, ensuring the simulation remains stable and informative over long horizons.
4. **Multi-Modal Uncertainty Tracking:** By calculating binary entropy on fire regime disagreements (e.g., surface vs. crown fire transitions), the system actively targets complex physical phenomena, not just spatial spread.

---

## 4. Evaluation and Simulation Framework (WISPsim)
To validate the framework, the repository includes `WISPsim`, a clock/cycle-based simulation harness. Relevant evaluation scenarios include:
- **Hilly Heterogeneous:** Tests routing across ridge/valley terrain with mixed fuels and dynamic wind shifts.
- **Dual Ignition:** Stress-tests the EnKF and path planner with multiple simultaneous fire fronts.
- **Crown Fire Risk:** Evaluates the bimodal entropy logic in dense timber scenarios.
- **Flat Homogeneous:** A control baseline to quantify the informational advantage of the AB Protocol over blind patrols.
