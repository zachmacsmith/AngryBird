Addendum to the IGNIS system architecture. Replaces the ground-station-to-ground-station orienteering formulation with a cycle-aligned, mode-based path planner that treats the drone as a persistent sensing asset.

---

## 1. Problem Statement

The original path formulation modeled each planning cycle as an independent sortie: depart ground station, collect observations, return to ground station, all within one cycle. This is wrong. A drone with 45 minutes of battery should not return to base every 20 minutes. The drone should fly continuously, replanning at cycle boundaries when the fire model and information field update, and only return when battery requires it.

The correct formulation decouples three concerns:

|Concern|Cadence|Governed by|
|---|---|---|
|Path optimization|Every Δt_replan (~30s)|Information field + budget|
|Fire model update|Every T_cycle (~20 min)|EnKF assimilation|
|Battery return|Once per sortie|Remaining range vs. distance to GS|

---

## 2. Core Concepts

### 2.1 Cycle Budget

In normal operation, the drone plans a path covering one cycle of flight:

```
d_cycle = v_drone × T_cycle
```

At drone speed ~15 m/s and T_cycle = 20 min, d_cycle ≈ 18 km. The drone aims to reach its planned endpoint at the moment the next cycle begins, at which point it receives a fresh information field and replans from its new position.

The drone does not return to a ground station between cycles. Its terminal position from cycle N becomes the start position for cycle N+1.

### 2.2 Remaining Range

```
r = remaining_range_m        # from battery telemetry, in meters of flight
d_max = max_range_m           # full-charge range
d_safety = α × d_max          # safety margin (α = 0.10)
```

The safety margin accounts for headwinds, GPS error, obstacle avoidance, and battery degradation. It is never spent on information gathering.

### 2.3 Reserve on Arrival

The key quantity governing mode transitions:

```
d_return = flight_distance(p_current, nearest_GS)
reserve_on_arrival = (r - d_return) / d_max
```

This is the fraction of total battery the drone would have left if it flew straight home right now. When this drops below a threshold R_threshold, the drone commits to returning.

---

## 3. Operating Modes

### 3.1 Mode Definitions

```
┌──────────┐           ┌──────────┐           ┌───────────┐
│  NORMAL  │──trigger──▶│  RETURN  │──trigger──▶│ EMERGENCY │
└──────────┘           └──────────┘           └───────────┘
     │                      │                       │
     │ free exploration     │ info-gathering         │ direct flight
     │ budget = d_cycle     │ with reachability      │ to GS, no
     │ no endpoint          │ invariant              │ optimization
     │ constraint           │ target GS locked       │
     └──────────────────────┴───────────────────────┘
              all transitions are one-way
```

**NORMAL**

- Budget: d_cycle
- Endpoint constraint: none
- Objective: maximize information along path
- Drone ends each cycle wherever is optimal, receives new plan

**RETURN**

- Budget: min(d_cycle, r - d_safety)
- Endpoint constraint: reachability invariant (Section 4)
- Objective: maximize information along path, subject to making progress toward GS
- Target GS locked at transition
- May span multiple cycles

**EMERGENCY**

- Budget: r (everything remaining)
- Endpoint: target GS, direct flight
- Objective: reach GS safely
- No information gathering, no orienteering solver
- Fallback for invariant violation (unexpected battery drain, headwinds)

### 3.2 Transition Conditions

**NORMAL → RETURN**

```
Trigger: reserve_on_arrival ≤ R_threshold

where:
    R_threshold = 0.35
    reserve_on_arrival = (r - d_return) / d_max
    d_return = flight_distance(p_current, nearest_reachable_GS)
```

At trigger:

- Lock target_gs = nearest_reachable_GS (does not change for remainder of sortie)
- Sticky: drone never returns to NORMAL mode

R_threshold = 0.35 means the drone begins returning when a direct flight home would leave 35% battery remaining. This reserve is available for detours and information gathering during the return leg.

**RETURN → EMERGENCY**

```
Trigger: r ≤ d_return(target_gs) + d_safety
```

The drone can no longer reach its target GS with safety margin while doing any exploration. Switch to direct flight.

### 3.3 Mode Check Frequency

Mode transitions are evaluated at every replan interval (Δt_replan), not just at cycle boundaries. Battery state changes continuously; a mode check every 30 seconds ensures the drone doesn't overshoot the trigger by a significant margin.

---

## 4. Reachability Invariant (RETURN Mode)

### 4.1 Definition

During RETURN mode, the drone plans one cycle of flight at a time. The endpoint of each cycle must satisfy:

```
flight_distance(p_end, target_gs) ≤ (r_at_cycle_end) - d_safety
```

where:

```
r_at_cycle_end = r - d_planned   (remaining range after flying this cycle's path)
d_planned ≤ d_cycle              (distance of this cycle's planned path)
```

In words: after this cycle's flight, the drone must still be able to reach the locked target GS with safety margin using only its remaining battery.

### 4.2 Geometric Interpretation

The invariant defines a **feasible endpoint region**: a disk centered on the target GS with radius:

```
R_feasible = (r - d_planned) - d_safety
```

This disk shrinks each cycle as battery depletes. Early return cycles have a large feasible region (lots of room for detours). Late return cycles have a small feasible region (near the GS). On the final return cycle, the feasible region collapses to the GS itself, and the solver produces a path terminating at the station.

### 4.3 Invariant Enforcement in the Solver

The orienteering solver must reject any candidate path whose terminal node falls outside the feasible region. Implementation approaches:

**Option A: Pre-prune the graph.** Before solving, remove all correlation-domain nodes with `flight_distance(node, target_gs) > R_feasible` from the candidate set. Solve orienteering on the reduced graph. Fast, simple, but loses nodes that could be visited mid-path (not as terminal node).

**Option B: Terminal constraint.** Allow all nodes mid-path but constrain the terminal node to lie within R_feasible. Standard orienteering solvers support endpoint constraints. Preferred — captures more information-gathering opportunities.

### 4.4 Multi-Cycle Return Planning

The drone does NOT plan the entire return trip upfront. It plans one cycle at a time, replanning at each cycle boundary with updated info field and battery telemetry. The invariant guarantees that each cycle's plan is independently safe.

Rationale: planning the full multi-cycle return requires predicting future information fields, which depend on future fire model outputs. The information field changes at each cycle boundary. Planning ahead would produce a path optimized for stale information. The receding-horizon single-cycle approach adapts to new information automatically.

---

## 5. Replanning

### 5.1 Replan Triggers

The drone replans its path under three conditions, checked at every Δt_replan interval:

1. **Cycle boundary:** the fire model has been re-run, a new information field is available. This is the primary replan trigger. The drone discards its remaining planned waypoints and generates a new path from p_current.
    
2. **Significant deviation:** the drone's actual position deviates from the planned path by more than ε_deviation (e.g., 200m), due to wind, obstacle avoidance, or autopilot drift. Replan from actual position.
    
3. **Mode transition:** NORMAL → RETURN or RETURN → EMERGENCY. Immediate replan with new constraints.
    

Between these triggers, the drone follows its planned waypoint sequence without modification. Replanning every 30 seconds is a check, not a forced recomputation — if no trigger condition is met, the current plan persists.

### 5.2 Replan Input

```
DroneState:
    position          (x, y) in UTM meters
    heading           radians
    remaining_range   meters, from battery telemetry
    mode              NORMAL | RETURN | EMERGENCY
    target_gs         GS position (set at RETURN transition, null in NORMAL)
    visited_domains   set of correlation-domain IDs observed this sortie

EnvironmentState:
    info_field        (R, C) array of w_i values, updated at cycle boundaries
    correlation_graph InfoGraph with node w_i and edge real distances
    ground_stations   list of GS positions
    T_cycle           seconds until next cycle boundary
```

### 5.3 Replan Output

```
PlannedPath:
    waypoints         ordered list of (x, y) UTM positions
    expected_distance total path length in meters
    expected_info     total information gain (Σ w_i of visited domains)
    terminal_domain   correlation domain ID where path ends
```

### 5.4 Visited Domain Tracking

When the drone passes through a correlation domain, that domain is added to `visited_domains`. On replan, visited domains are removed from the information graph (their w_i is set to zero). This prevents the drone from revisiting already-observed regions.

At each cycle boundary, the fire model re-runs and produces a new info field. Some previously visited domains may have new uncertainty (fire has advanced, conditions changed). Whether to re-admit them depends on whether the new w_i exceeds a revisit threshold:

```
if new_w_i(domain) > w_revisit_threshold:
    remove domain from visited_domains
```

This allows the drone to re-observe locations where conditions have changed significantly since the last pass. Default: set w_revisit_threshold high enough that revisits are rare (e.g., top 5% of w_i values).

---

## 6. Solver Interface

### 6.1 Unified Solver Contract

```
solve_orienteering(
    start:               (x, y)            # drone's current position
    budget:              float             # meters of flight for this plan
    graph:               InfoGraph         # correlation domains + edges
    visited:             set[int]          # domains to exclude
    endpoint:            (x, y) | None     # hard endpoint (EMERGENCY, final RETURN cycle)
    endpoint_max_radius: float | None      # feasible region radius (RETURN, non-final)
    endpoint_center:     (x, y) | None     # center of feasible region (= target_gs)
) -> list[Waypoint]
```

Mode-specific invocation:

|Mode|budget|endpoint|endpoint_max_radius|
|---|---|---|---|
|NORMAL|d_cycle|None|None|
|RETURN (non-final)|d_cycle|None|R_feasible|
|RETURN (final)|r - d_safety|target_gs|None|
|EMERGENCY|r|target_gs|None|

"Final RETURN cycle" is detected when R_feasible < d_cycle — the feasible region is small enough that the drone must reach GS within this cycle.

### 6.2 Solver Requirements

The solver must:

1. Find a path through the correlation-domain graph starting at or near `start`.
2. Maximize Σ w_i of visited domains along the path.
3. Respect the budget constraint: Σ real_edge_distance ≤ budget.
4. If `endpoint` is set, terminate at that position (within tolerance ε_endpoint).
5. If `endpoint_max_radius` is set, terminate at a position within that radius of `endpoint_center`.
6. Exclude all domains in `visited` from contributing to the objective.
7. Return an ordered waypoint list that can be fed to the drone autopilot.

The solver does NOT need to:

- Plan beyond one cycle (receding horizon handles multi-cycle planning)
- Account for other drones (deconfliction happens via info field — other drones' observations reduce w_i, which the solver sees as low-value regions)
- Model drone dynamics (turn radius, acceleration — these are handled by the autopilot layer that interpolates between waypoints)

### 6.3 Solver Implementation Recommendation

The correlation-domain graph has ~400 nodes and ~1,600 edges. At this scale, greedy insertion heuristics solve orienteering in <10ms. The recommended approach:

1. Find the shortest path from start to endpoint (or to the highest-w_i reachable node if no endpoint) using Dijkstra on real distances.
2. Greedily insert the highest-value detour nodes that fit within the remaining budget without violating the endpoint constraint.
3. Apply 2-opt local search: swap adjacent waypoint pairs, keep if Σ w_i improves without exceeding budget.
4. Score the final path using sequential GP conditional variance (exact information gain accounting for observation correlation).

Total solver time: <50ms per replan. Well within the 30-second replan interval.

---

## 7. Ground Station Management

### 7.1 GS Locking

At NORMAL → RETURN transition, the drone locks its target GS:

```
target_gs = argmin over reachable GS: flight_distance(p_current, gs)
```

A GS is reachable if `flight_distance(p_current, gs) < r - d_safety`.

The lock persists for the remainder of the sortie. Rationale:

- Prevents oscillation between equidistant stations.
- Allows ground crew to anticipate arrival and prepare battery swap.
- Simplifies the solver (single fixed endpoint vs. multi-option).

### 7.2 GS Lock Override

The lock is overridden only if the target GS becomes unreachable:

```
if flight_distance(p_current, target_gs) > r - d_safety:
    # Target GS unreachable — relock to nearest reachable
    target_gs = argmin over reachable GS: flight_distance(p_current, gs)
    
    if no GS reachable:
        mode = EMERGENCY
        # land at nearest safe location (open field, road)
        # outside scope of this spec — handled by autopilot safety layer
```

Causes of unreachability: unexpected headwinds, battery degradation faster than modeled, airspace closure forcing a detour.

### 7.3 GS Availability

The system maintains a list of active ground stations with operational status. A GS may go offline (crew rotation, equipment failure, fire proximity). Offline GS positions are excluded from nearest-GS calculations and lock candidates.

If the locked target GS goes offline during a RETURN sortie, the drone relocks to the nearest available GS using the same override logic.

---

## 8. Loiter Behavior

### 8.1 When to Loiter

In NORMAL mode, the orienteering solver may return a path shorter than d_cycle if there is insufficient information value within range. This occurs when:

- The drone has already observed all nearby correlation domains.
- The remaining high-value domains are beyond d_cycle reach.
- The fire model has not yet updated (mid-cycle, info field is stale).

In this case, the drone should loiter at its current position and wait for the next cycle boundary to reveal new uncertainty. Loiter consumes minimal battery (hover power) and preserves range for future cycles.

### 8.2 Loiter Detection

```
if solver returns path with expected_info < w_min_useful:
    loiter at p_current until next cycle boundary
```

w_min_useful is a configurable threshold. If the best available path collects less information than this threshold, flying it is not worth the battery cost.

### 8.3 Loiter Battery Accounting

Hover consumes power but covers zero distance. The battery model must track energy consumption (in joules or watt-hours), not just distance. During loiter:

```
r_after_loiter = r - (P_hover / P_cruise) × v_drone × t_loiter
```

where P_hover / P_cruise is the hover-to-cruise power ratio (typically 0.7-0.9 for multirotors). This correction ensures the mode transition check uses accurate remaining range even after extended loiter periods.

---

## 9. Multi-Drone Coordination

### 9.1 Implicit Deconfliction via Information Field

Multiple drones operate on the same information field. When drone A observes a correlation domain, the GP assimilates the observation, conditional variance drops, and w_i for that domain decreases. When drone B replans at the next cycle boundary, it sees the reduced w_i and is naturally steered elsewhere.

This requires that drone observations are communicated to the base station and assimilated before other drones replan. The cycle boundary serves as the synchronization point: all observations from all drones are assimilated, the fire model re-runs, and all drones receive the same updated info field.

### 9.2 Intra-Cycle Deconfliction

Between cycle boundaries, two drones could independently navigate toward the same high-value domain. The implicit deconfliction only works at cycle boundaries.

For the hackathon: accept this as a known limitation. The information loss from occasional double-observation is small compared to the complexity of real-time inter-drone communication.

For production: broadcast each drone's planned path to all other drones at replan time. Each drone marks domains on other drones' planned paths as "claimed" and excludes them from its own solver. This is a soft reservation system — if a drone deviates from its plan, the reservation expires at the next replan.

### 9.3 Staggered Mode Transitions

Drones will enter RETURN mode at different times depending on when they launched and their individual battery consumption. This is a feature: it ensures continuous coverage. While one drone returns for battery swap, others continue observing.

The base station should stagger launches so that RETURN mode transitions are spread across cycles, avoiding a coverage gap where all drones return simultaneously.

---

## 10. Integration with Existing Architecture

### 10.1 Information Field (Unchanged)

The information field w_i is computed from ensemble variance, sensitivity, and observability exactly as specified in the existing architecture. The path planner consumes it; it does not modify how it is produced.

### 10.2 QUBO Point Selection (Unchanged)

The QUBO selects the K highest-value, lowest-redundancy measurement locations from the information field. This is a separate optimization from path planning. The QUBO output can inform the path planner (prioritize paths that visit QUBO-selected locations), but the path planner operates on the full correlation-domain graph, not the QUBO output.

Interaction: QUBO-selected points receive a bonus weight in the info graph, biasing the orienteering solver toward them without hard-constraining the path.

### 10.3 Correlation-Domain Graph (Minor Change)

The graph construction adds one new field per node:

```
CorrelationDomain:
    id:                int
    centroid:          (x, y) UTM
    w_i:              float          # information value (existing)
    flight_distances:  dict[int, float]  # edge costs to neighbors (existing)
    gs_distances:      dict[int, float]  # NEW: flight distance to each GS
```

The GS distances are precomputed at graph construction time. They are used for the reachability invariant check without calling flight_distance() repeatedly during solving. Updated only when GS availability changes.

### 10.4 Cycle Update Sequence (Modified)

```
1. Receive drone observations (telemetry stream)          [existing]
2. At cycle boundary:
   a. EnKF update: assimilate all observations             [existing]
   b. Update m_10h: exponential smoothing                  [from fuel spec]
   c. Run fire ensemble                                    [existing]
   d. Compute information field w_i                        [existing]
   e. Rebuild correlation-domain graph                     [existing]
   f. FOR EACH active drone:                               [MODIFIED]
      i.   Check mode transition                           [NEW]
      ii.  Compute budget and constraints per mode         [NEW]
      iii. Solve orienteering from drone's current pos     [MODIFIED start point]
      iv.  Transmit waypoints to drone                     [existing]
3. Between cycle boundaries:
   a. Every Δt_replan: check mode transitions              [NEW]
   b. On mode change: immediate replan with current info   [NEW]
   c. On significant deviation: replan from actual pos     [NEW]
```

### 10.5 Communication Requirements

|Data|Direction|Frequency|Size|
|---|---|---|---|
|Drone telemetry (pos, battery)|Drone → Base|Continuous (1 Hz)|~50 bytes|
|Sensor observations|Drone → Base|Continuous (1 Hz)|~200 bytes|
|Waypoint plan|Base → Drone|At replan (~30s)|~500 bytes|
|Mode change notification|Base → Drone|On transition|~20 bytes|

Total bandwidth: <5 kbps per drone. Easily within typical UHF/WiFi datalink.

For disconnected operation (mountain terrain, comms blackout): the drone carries a snapshot of the info field and runs the orienteering solver onboard. See Section 11.

---

## 11. Disconnected Operation

### 11.1 Onboard Autonomy

If communication with the base station is lost, the drone must operate autonomously. It carries:

- Last received info field (stale but usable)
- Correlation-domain graph
- GS positions
- Its own battery state

The drone continues executing its current plan. At the next cycle boundary (by its own clock), it replans using the stale info field minus its own visited domains. Mode transitions proceed using local battery telemetry.

### 11.2 Reconnection

When communication is restored, the drone uploads all stored observations to the base station. The base assimilates them in the next cycle. The drone receives the current info field and resumes normal coordinated operation.

### 11.3 Comms Loss During RETURN Mode

If comms are lost during RETURN mode, the drone continues returning to its locked target GS using onboard navigation. This is the simplest case — no info field updates needed, the drone just flies its return path with whatever detours the stale info field suggests.

---

## 12. Parameters and Configuration

|Parameter|Symbol|Default|Description|
|---|---|---|---|
|Cycle duration|T_cycle|1200 s (20 min)|Fire model update interval|
|Replan interval|Δt_replan|30 s|Mode check and deviation check frequency|
|Drone cruise speed|v_drone|15 m/s|Nominal flight speed|
|Drone max range|d_max|40,000 m|Full-charge flight distance|
|Safety margin fraction|α|0.10|Fraction of d_max reserved as buffer|
|Return threshold|R_threshold|0.35|reserve_on_arrival triggering RETURN|
|Deviation tolerance|ε_deviation|200 m|Replan if actual pos deviates this far|
|Endpoint tolerance|ε_endpoint|50 m|Close enough counts as reaching endpoint|
|Minimum useful info|w_min_useful|configurable|Below this, loiter instead of fly|
|Revisit threshold|w_revisit|top 5% of w_i|Re-admit visited domain if w_i exceeds|
|QUBO bonus weight|λ_qubo|1.5|Multiplier on w_i for QUBO-selected nodes|
|Hover power ratio|P_hover/P_cruise|0.85|Battery drain during loiter vs flight|

### 12.1 Tuning Guidance

R_threshold = 0.35 is conservative. Lower values (0.25) give more exploration time but less information-gathering budget during return. Higher values (0.45) give more return-leg information but trigger return earlier. The right value depends on the spatial density of ground stations relative to the fire area.

Rule of thumb: R_threshold should be set so that the exploration budget during return (R_threshold × d_max) is roughly equal to one d_cycle. This means the return leg has approximately one cycle's worth of detour budget, which keeps the return path informationally productive.

```
R_threshold ≈ d_cycle / d_max = (v × T_cycle) / d_max
```

At v=15 m/s, T_cycle=1200s, d_max=40km: R_threshold ≈ 18/40 = 0.45. This is slightly higher than the default 0.35, suggesting the default is conservative. Adjust based on operational risk tolerance.

---

## 13. Validation Requirements

### 13.1 Unit Tests

1. **Mode transition logic**: given a sequence of (position, remaining_range) values, verify NORMAL → RETURN triggers at correct reserve_on_arrival, and RETURN → EMERGENCY triggers when range equals d_return + d_safety.
    
2. **Reachability invariant**: for a RETURN mode plan, verify that every candidate path's terminal node satisfies the feasible-region constraint. Test with R_feasible = 0 (must terminate at GS).
    
3. **Budget calculation**: verify budget = d_cycle in NORMAL, budget = min(d_cycle, r - d_safety) in RETURN, budget = r in EMERGENCY.
    
4. **GS locking**: verify target_gs is set at NORMAL → RETURN and does not change on subsequent replans. Verify override triggers when target becomes unreachable.
    

### 13.2 Integration Tests

5. **Full sortie simulation**: simulate a drone launching from GS, flying 3-4 cycles in NORMAL, transitioning to RETURN, returning over 2 cycles, arriving at GS. Verify total distance ≤ d_max - d_safety, and total information collected is higher than a naive strategy (fly out to max range, return directly).
    
6. **Multi-drone coverage**: simulate 3 drones with staggered launches. Verify that the implicit deconfliction (via info field updates) prevents all three from visiting the same domains. Measure coverage (fraction of high-value domains observed) vs. a single drone.
    
7. **Loiter behavior**: create a scenario where the drone has observed all nearby domains. Verify it loiters until the next cycle boundary rather than flying to low-value regions.
    
8. **Disconnected operation**: simulate comms loss for 2 cycles. Verify the drone continues on stale info field, replans independently, and re-syncs observations on reconnection.
    

### 13.3 Performance Tests

9. **Solver latency**: the orienteering solver must complete in <50ms for a 400-node graph. Measure across all three modes. EMERGENCY mode should be <1ms (direct path, no optimization).
    
10. **Mode check overhead**: the mode transition check runs every 30s per drone. With 10 drones, this is 10 evaluations of flight_distance + arithmetic. Must complete in <1ms total.
    

---

## 14. Risks and Open Questions

### 14.1 Battery Model Accuracy

The system assumes remaining_range is accurately reported by battery telemetry. In practice, battery voltage curves are nonlinear and temperature-dependent. A 35% state-of-charge reading at -10°C may deliver substantially less range than at 20°C. Mitigation: use a conservative battery model that maps voltage × temperature to worst-case remaining range. The safety margin α = 0.10 provides additional buffer but may not be sufficient in extreme cold.

### 14.2 Wind Effects on Range

The system uses flight_distance (terrain-aware Euclidean or A* through no-fly zones) as the range cost. Wind is not modeled. A 10 m/s headwind halves effective range in one direction. Mitigation: the weather model provides wind fields. flight_distance could be replaced with flight_time × v_ground where v_ground = v_drone ± v_wind_component. This adds a wind-field lookup to every distance computation. Not implemented in v1; flag for v2 if operational testing reveals range prediction errors.

### 14.3 Optimal R_threshold

The default R_threshold = 0.35 is a heuristic. The optimal value depends on GS density, fire area size, and information field spatial structure. An adaptive threshold that adjusts based on observed information yield during the sortie would be better but adds complexity. Defer to v2.

### 14.4 Solver Feasibility in Constrained RETURN

When R_feasible is small and the feasible endpoint region contains few correlation-domain nodes, the orienteering solver may find no valid path that collects meaningful information. In this case the solver degenerates to shortest-path-to-GS, which is correct behavior. Ensure the solver handles this gracefully (returns a direct path rather than failing).