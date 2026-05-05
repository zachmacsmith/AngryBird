"""
Unit tests for docs/Drone Path.md: mode transitions, budget calculation,
reachability invariant, GS locking, full sortie simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from angrybird.gp import IGNISGPPrior
from angrybird.observations import ObservationStore
from angrybird.selectors.correlation_path import (
    CorrelationPathSelector,
    _build_correlation_graph,
    _check_mode_transitions,
    _compute_all_gs_distances,
    _felzenszwalb_label_map,
    _min_gs_return_costs,
    _pos_to_domain,
    _terrain_features,
)
from angrybird.types import (
    DroneFlightState,
    DroneMode,
    EnsembleResult,
    InformationField,
    TerrainData,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_terrain(rows=30, cols=30, seed=0):
    rng = np.random.default_rng(seed)
    fuel = np.full((rows, cols), 101, dtype=np.int16)
    fuel[rows // 2:, :] = 161  # fuel boundary in middle
    return TerrainData(
        elevation=rng.uniform(0, 500, (rows, cols)).astype(np.float32),
        slope=rng.uniform(0, 30, (rows, cols)).astype(np.float32),
        aspect=rng.uniform(0, 360, (rows, cols)).astype(np.float32),
        fuel_model=fuel,
        canopy_cover=rng.uniform(0, 0.5, (rows, cols)).astype(np.float32),
        canopy_height=np.ones((rows, cols), dtype=np.float32) * 10,
        canopy_base_height=np.ones((rows, cols), dtype=np.float32) * 3,
        canopy_bulk_density=np.ones((rows, cols), dtype=np.float32) * 0.05,
        shape=(rows, cols),
        resolution_m=50.0,
    )


def _make_info_field(terrain, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = terrain.shape
    w = rng.uniform(0, 5, (rows, cols)).astype(np.float32)
    return InformationField(
        w=w,
        w_by_variable={"fmc": w},
        sensitivity={"fmc": np.ones((rows, cols), dtype=np.float32)},
        gp_variance={"fmc": rng.uniform(0, 0.04, (rows, cols)).astype(np.float32)},
    )


def _make_ensemble(terrain, n_members=5):
    rows, cols = terrain.shape
    zero = np.zeros((rows, cols), dtype=np.float32)
    return EnsembleResult(
        member_arrival_times=np.full((n_members, rows, cols), np.nan, dtype=np.float32),
        member_fmc_fields=np.full((n_members, rows, cols), 0.1, dtype=np.float32),
        member_wind_fields=np.full((n_members, rows, cols), 5.0, dtype=np.float32),
        burn_probability=zero,
        mean_arrival_time=zero,
        arrival_time_variance=zero,
        n_members=n_members,
    )


@pytest.fixture(scope="module")
def base_scene():
    terrain = _make_terrain(seed=42)
    info = _make_info_field(terrain, seed=42)
    ensemble = _make_ensemble(terrain)
    obs_store = ObservationStore()
    gp = IGNISGPPrior(obs_store=obs_store, terrain=terrain, resolution_m=50.0)

    features = _terrain_features(terrain)
    label_map = _felzenszwalb_label_map(terrain, features, 1500.0, 50.0, 5)
    graph = _build_correlation_graph(label_map, features, info.w, {"fmc": info.w}, 50.0)

    gses_m = [np.array([0.0, 0.0])]
    gs_dists = _compute_all_gs_distances(graph, gses_m, 50.0)
    min_costs = _min_gs_return_costs(gs_dists)

    rows, cols = terrain.shape
    far_m = np.array([(rows - 1) * 50.0, (cols - 1) * 50.0])
    far_domain = _pos_to_domain(far_m, graph, 50.0)
    d_ret_far = gs_dists[0].get(far_domain, np.inf)

    return dict(
        terrain=terrain,
        info=info,
        ensemble=ensemble,
        gp=gp,
        graph=graph,
        gses_m=gses_m,
        gs_dists=gs_dists,
        min_costs=min_costs,
        far_m=far_m,
        far_domain=far_domain,
        d_ret_far=d_ret_far,
    )


# ---------------------------------------------------------------------------
# 1. Mode transition tests (spec §3.2, §13.1 items 1-4)
# ---------------------------------------------------------------------------

class TestModeTransitions:
    D_MAX = 20_000.0
    D_SAFETY = 2_000.0
    R_THRESH = 0.35

    def _transition(self, state, scene):
        return _check_mode_transitions(
            state, scene["graph"], scene["gses_m"], scene["gs_dists"],
            50.0, self.D_MAX, self.D_SAFETY, self.R_THRESH,
        )

    def test_normal_full_battery_stays_normal(self, base_scene):
        s = DroneFlightState(0, np.array([0., 0.]), self.D_MAX, DroneMode.NORMAL, -1)
        out = self._transition(s, base_scene)
        assert out.mode == DroneMode.NORMAL

    def test_normal_triggers_return_below_threshold(self, base_scene):
        d_ret = base_scene["d_ret_far"]
        far_m = base_scene["far_m"]
        # reserve = (r - d_ret) / D_MAX = 0.20 < 0.35 → RETURN
        r = 0.20 * self.D_MAX + d_ret
        s = DroneFlightState(0, far_m.copy(), r, DroneMode.NORMAL, -1)
        out = self._transition(s, base_scene)
        assert out.mode == DroneMode.RETURN
        assert out.target_gs_idx == 0

    def test_normal_stays_normal_above_threshold(self, base_scene):
        d_ret = base_scene["d_ret_far"]
        far_m = base_scene["far_m"]
        # reserve = 0.50 > 0.35
        r = 0.50 * self.D_MAX + d_ret
        s = DroneFlightState(0, far_m.copy(), r, DroneMode.NORMAL, -1)
        out = self._transition(s, base_scene)
        assert out.mode == DroneMode.NORMAL

    def test_return_triggers_emergency(self, base_scene):
        d_ret = base_scene["d_ret_far"]
        far_m = base_scene["far_m"]
        # r = d_ret + d_safety - 10  →  r ≤ d_ret + d_safety → EMERGENCY
        r = d_ret + self.D_SAFETY - 10
        s = DroneFlightState(0, far_m.copy(), r, DroneMode.RETURN, 0)
        out = self._transition(s, base_scene)
        assert out.mode == DroneMode.EMERGENCY

    def test_return_stays_return_with_margin(self, base_scene):
        d_ret = base_scene["d_ret_far"]
        far_m = base_scene["far_m"]
        # r = d_ret + d_safety + 500 → stays RETURN
        r = d_ret + self.D_SAFETY + 500
        s = DroneFlightState(0, far_m.copy(), r, DroneMode.RETURN, 0)
        out = self._transition(s, base_scene)
        assert out.mode == DroneMode.RETURN

    def test_emergency_is_terminal(self, base_scene):
        s = DroneFlightState(0, np.array([0., 0.]), 5000., DroneMode.EMERGENCY, 0)
        out = self._transition(s, base_scene)
        assert out.mode == DroneMode.EMERGENCY

    def test_gs_lock_preserved_across_return(self, base_scene):
        d_ret = base_scene["d_ret_far"]
        far_m = base_scene["far_m"]
        r = d_ret + self.D_SAFETY + 1000
        s = DroneFlightState(0, far_m.copy(), r, DroneMode.RETURN, 0)
        out = self._transition(s, base_scene)
        assert out.target_gs_idx == 0  # lock not overridden


# ---------------------------------------------------------------------------
# 2. Budget calculation (spec §13.1 item 3)
# ---------------------------------------------------------------------------

class TestBudget:
    D_CYCLE = 18_000.0
    D_SAFETY = 2_000.0

    def test_normal_budget_equals_d_cycle_with_buffer(self, base_scene):
        """NORMAL mode uses d_cycle × 1.10 as budget; plan distance must not exceed it."""
        from angrybird.selectors.correlation_path import BUDGET_BUFFER
        sel = CorrelationPathSelector(
            min_domain_cells=5,
            drone_range_m=40_000,
            d_cycle_m=self.D_CYCLE,
        )
        result = sel.select(
            base_scene["info"], base_scene["gp"], base_scene["ensemble"], k=1,
            terrain=base_scene["terrain"], staging_area=(0, 0), resolution_m=50.0,
        )
        assert len(result.drone_plans) == 1
        plan = result.drone_plans[0]
        assert plan.plan_distance_m <= self.D_CYCLE * BUDGET_BUFFER + 1e-3

    def test_return_budget_limited_by_remaining_range(self, base_scene):
        """RETURN mode budget = min(d_cycle, r - d_safety)."""
        sel = CorrelationPathSelector(min_domain_cells=5, drone_range_m=40_000, d_cycle_m=self.D_CYCLE)
        r = 5_000.0  # less than d_cycle → budget = r - d_safety = 3_000
        state = DroneFlightState(
            drone_id=0,
            position_m=np.array([0., 0.]),
            remaining_range_m=r,
            mode=DroneMode.RETURN,
            target_gs_idx=0,
        )
        result = sel.select(
            base_scene["info"], base_scene["gp"], base_scene["ensemble"], k=1,
            terrain=base_scene["terrain"], staging_area=(0, 0), resolution_m=50.0,
            drone_states=[state],
        )
        plan = result.drone_plans[0]
        expected_budget = min(self.D_CYCLE, r - self.D_SAFETY)
        assert plan.plan_distance_m <= expected_budget + 1e-3


# ---------------------------------------------------------------------------
# 3. Reachability invariant (spec §4)
# ---------------------------------------------------------------------------

class TestReachabilityInvariant:
    def test_return_path_endpoint_reachable_from_gs(self, base_scene):
        """
        In RETURN mode, the terminal domain's GS distance must satisfy the
        invariant: dist(endpoint, GS) ≤ (r - d_planned) - d_safety.
        """
        d_safety = 2_000.0
        d_max = 40_000.0
        sel = CorrelationPathSelector(
            min_domain_cells=5,
            drone_range_m=d_max,
            d_cycle_m=18_000.0,
            safety_fraction=d_safety / d_max,
        )
        d_ret = base_scene["d_ret_far"]
        r = d_ret + d_safety + 5_000.0  # 5km extra budget to detour
        state = DroneFlightState(
            drone_id=0,
            position_m=base_scene["far_m"].copy(),
            remaining_range_m=r,
            mode=DroneMode.RETURN,
            target_gs_idx=0,
        )
        result = sel.select(
            base_scene["info"], base_scene["gp"], base_scene["ensemble"], k=1,
            terrain=base_scene["terrain"], staging_area=(0, 0), resolution_m=50.0,
            drone_states=[state],
            ground_stations_m=base_scene["gses_m"],
        )
        plan = result.drone_plans[0]
        updated = result.updated_drone_states[0]

        # Invariant: dist(endpoint, GS) ≤ (r - d_planned) - d_safety
        endpoint_m = updated.position_m
        dist_endpoint_to_gs = np.linalg.norm(endpoint_m - base_scene["gses_m"][0])
        r_at_end = r - plan.plan_distance_m
        R_feasible = r_at_end - d_safety
        assert dist_endpoint_to_gs <= R_feasible + 500, (
            f"Reachability violated: dist={dist_endpoint_to_gs:.0f} > R_feasible={R_feasible:.0f}"
        )


# ---------------------------------------------------------------------------
# 4. EMERGENCY mode: direct path (spec §3.1)
# ---------------------------------------------------------------------------

class TestEmergencyMode:
    def test_emergency_returns_plan(self, base_scene):
        sel = CorrelationPathSelector(min_domain_cells=5, drone_range_m=40_000)
        d_ret = base_scene["d_ret_far"]
        state = DroneFlightState(
            drone_id=0,
            position_m=base_scene["far_m"].copy(),
            remaining_range_m=d_ret + 500,
            mode=DroneMode.EMERGENCY,
            target_gs_idx=0,
        )
        result = sel.select(
            base_scene["info"], base_scene["gp"], base_scene["ensemble"], k=1,
            terrain=base_scene["terrain"], staging_area=(0, 0), resolution_m=50.0,
            drone_states=[state],
            ground_stations_m=base_scene["gses_m"],
        )
        plan = result.drone_plans[0]
        updated = result.updated_drone_states[0]
        assert plan.drone_mode == "EMERGENCY"
        assert len(plan.waypoints) >= 2
        assert updated.returned is True

    def test_emergency_mode_marks_landed(self, base_scene):
        sel = CorrelationPathSelector(min_domain_cells=5, drone_range_m=40_000)
        d_ret = base_scene["d_ret_far"]
        state = DroneFlightState(
            drone_id=0,
            position_m=base_scene["far_m"].copy(),
            remaining_range_m=d_ret + 1,
            mode=DroneMode.EMERGENCY,
            target_gs_idx=0,
        )
        result = sel.select(
            base_scene["info"], base_scene["gp"], base_scene["ensemble"], k=1,
            terrain=base_scene["terrain"], staging_area=(0, 0), resolution_m=50.0,
            drone_states=[state],
            ground_stations_m=base_scene["gses_m"],
        )
        assert result.updated_drone_states[0].returned is True


# ---------------------------------------------------------------------------
# 5. Cross-drone deconfliction via shared GP variance
# ---------------------------------------------------------------------------

class TestCrossDroneDeconfliction:
    def test_no_cross_drone_waypoint_overlap(self, base_scene):
        """
        After drone 0 plans, domains it selected are zeroed in current_w via
        GP variance updates — drone 1 naturally avoids them without any
        persistent visited set.
        """
        sel = CorrelationPathSelector(min_domain_cells=5, drone_range_m=40_000)
        result = sel.select(
            base_scene["info"], base_scene["gp"], base_scene["ensemble"], k=2,
            terrain=base_scene["terrain"], staging_area=(0, 0), resolution_m=50.0,
        )
        wps0 = set(result.drone_plans[0].waypoints)
        wps1 = set(result.drone_plans[1].waypoints)
        # Allow start waypoint overlap (both begin at staging) but not exploration waypoints
        overlap = wps0 & wps1 - {(0, 0)}
        assert len(overlap) == 0, f"Cross-drone overlap at: {overlap}"


# ---------------------------------------------------------------------------
# 6. Full sortie simulation (spec §13.2 item 5)
# ---------------------------------------------------------------------------

class TestFullSortie:
    def test_multi_cycle_sortie_within_range(self, base_scene):
        """
        Simulate up to 5 cycles. Total distance ≤ d_max - d_safety.
        """
        D_MAX = 15_000.0
        D_SAFETY = 1_500.0
        D_CYCLE = 5_000.0

        sel = CorrelationPathSelector(
            min_domain_cells=5,
            drone_range_m=D_MAX,
            d_cycle_m=D_CYCLE,
            safety_fraction=D_SAFETY / D_MAX,
            R_threshold=0.35,
        )
        terrain = base_scene["terrain"]
        info = base_scene["info"]
        ensemble = base_scene["ensemble"]
        gp = base_scene["gp"]

        states = None
        total_dist = 0.0

        for cycle in range(5):
            result = sel.select(
                info, gp, ensemble, k=1,
                terrain=terrain, staging_area=(0, 0), resolution_m=50.0,
                drone_states=states,
            )
            plan = result.drone_plans[0]
            total_dist += plan.plan_distance_m
            states = result.updated_drone_states

            if states[0].returned:
                break

        usable = D_MAX - D_SAFETY
        assert total_dist <= usable + 100, (
            f"Sortie used {total_dist:.0f} m > usable {usable:.0f} m"
        )

    def test_drone_mode_sequence_one_way(self, base_scene):
        """Mode transitions are strictly one-way within a sortie."""
        D_MAX = 10_000.0
        D_SAFETY = 1_000.0
        D_CYCLE = 3_000.0
        sel = CorrelationPathSelector(
            min_domain_cells=5,
            drone_range_m=D_MAX,
            d_cycle_m=D_CYCLE,
            safety_fraction=D_SAFETY / D_MAX,
            R_threshold=0.40,
        )
        terrain = base_scene["terrain"]
        info = base_scene["info"]
        ensemble = base_scene["ensemble"]
        gp = base_scene["gp"]

        states = None
        mode_history = []

        for _ in range(8):
            result = sel.select(
                info, gp, ensemble, k=1,
                terrain=terrain, staging_area=(0, 0), resolution_m=50.0,
                drone_states=states,
            )
            states = result.updated_drone_states
            mode_history.append(states[0].mode.value)
            if states[0].returned:
                break

        mode_rank = {"NORMAL": 0, "RETURN": 1, "EMERGENCY": 2}
        ranks = [mode_rank[m] for m in mode_history]
        for i in range(len(ranks) - 1):
            assert ranks[i + 1] >= ranks[i], (
                f"Mode went backwards at step {i}: {mode_history[i]} → {mode_history[i+1]}"
            )
