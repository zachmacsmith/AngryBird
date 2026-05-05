"""
Fire state estimation for IGNIS.

Maintains per-member fire states across forecast cycles, reconstructs
arrival time fields from sparse fire observations using fast marching,
and manages ensemble fire state diversity through controlled perturbations
at uncertain boundaries.

All internal times are in **seconds** (matching observation timestamps and
the ``start_time`` passed to the orchestrator).  EnsembleResult arrival times
are in **minutes**; conversion is handled explicitly at the boundary.

See docs/Fire State Estimation.md for the full design specification.
"""

from __future__ import annotations

import heapq
import logging
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from .config import (
    CANOPY_COVER_FRACTION,
    FUEL_MINERAL_CONTENT,
    FUEL_MINERAL_SILICA_FREE,
    FUEL_PARAMS,
    FUEL_PARTICLE_DENSITY,
)
from .observations import FireDetectionObservation
from .types import EnsembleResult, GPPrior, TerrainData

logger = logging.getLogger(__name__)

# SAV in config is 1/ft; Rothermel SI uses 1/m
_SAV_FT_TO_M: float = 0.3048


# ---------------------------------------------------------------------------
# NumPy Rothermel ROS (CPU, single-member, for fast march reconstruction)
# ---------------------------------------------------------------------------

def compute_ros_field(
    terrain: TerrainData,
    fmc: np.ndarray,           # float32[rows, cols], fraction
    wind_speed: np.ndarray,    # float32[rows, cols], m/s at 10 m
    wind_direction: np.ndarray,  # float32[rows, cols], degrees (unused in scalar ROS)
) -> np.ndarray:
    """
    Compute Rothermel surface ROS (m/s) at every grid cell from GP-mean fields.

    NumPy port of the PyTorch kernel in gpu_fire_engine._rothermel_ros.
    Used for fast march reconstruction — no ensemble batching, CPU only.
    Wind direction is accepted for interface consistency but not used by the
    isotropic scalar Rothermel model (same convention as the GPU engine).

    Returns: float32[rows, cols] ROS in m/s.  Non-burnable cells → 0.0.
    """
    rows, cols = terrain.shape
    fm = terrain.fuel_model.astype(np.int32)
    eps = 1e-10

    # ── Fuel parameter lookup (per-cell) ──────────────────────────────────
    w0    = np.zeros((rows, cols), dtype=np.float64)
    sigma = np.zeros((rows, cols), dtype=np.float64)
    delta = np.zeros((rows, cols), dtype=np.float64)
    mx    = np.zeros((rows, cols), dtype=np.float64)
    heat  = np.zeros((rows, cols), dtype=np.float64)

    for fid, p in FUEL_PARAMS.items():
        mask = fm == fid
        if not mask.any():
            continue
        w0[mask]    = p["load"]
        sigma[mask] = p["sav"] * _SAV_FT_TO_M
        delta[mask] = p["depth"]
        mx[mask]    = p["mx"]
        heat[mask]  = p["h"]

    rho_p = float(FUEL_PARTICLE_DENSITY)
    st    = float(FUEL_MINERAL_CONTENT)
    se    = float(FUEL_MINERAL_SILICA_FREE)

    burnable = w0 > eps

    # ── Wind adjustment factor: 10 m → midflame height ────────────────────
    cc = np.zeros((rows, cols), dtype=np.float64)
    for fid, val in CANOPY_COVER_FRACTION.items():
        cc[fm == fid] = float(val)
    wind_adj = np.where(cc > 0.5, 0.4, np.where(cc > 0.1, 0.5, 0.6))

    # ── Slope factor ───────────────────────────────────────────────────────
    slope_rad    = np.radians(terrain.slope.astype(np.float64))
    tan_slope_sq = np.tan(slope_rad) ** 2

    # ── Rothermel (1972) surface ROS ──────────────────────────────────────
    fmc_d  = fmc.astype(np.float64)
    ws_d   = np.clip(wind_speed.astype(np.float64), 0.0, None)

    beta       = w0 / (delta * rho_p + eps)
    beta_op    = 3.348 * np.power(sigma + eps, -0.8189)
    beta_ratio = beta / (beta_op + eps)

    A         = 133.0 * np.power(sigma + eps, -0.7913)
    s15       = np.power(sigma + eps, 1.5)
    gamma_max = s15 / (495.0 + 0.0594 * s15)
    gamma     = (gamma_max
                 * np.power(np.clip(beta_ratio, eps, None), A)
                 * np.exp(np.clip(A * (1.0 - beta_ratio), -100.0, 100.0)))

    wn    = w0 * (1.0 - st)
    eta_s = np.clip(0.174 * (se + eps) ** -0.19, 0.0, 1.0)
    rm    = np.clip(fmc_d / (mx + eps), 0.0, 1.0)
    eta_m = np.clip(1.0 - 2.59 * rm + 5.11 * rm**2 - 3.52 * rm**3, 0.0, 1.0)

    I_R = gamma * wn * heat * eta_m * eta_s

    xi_exp = np.clip(
        (0.792 + 0.681 * np.sqrt(sigma + eps)) * (beta + 0.1), -100.0, 100.0)
    xi = np.exp(xi_exp) / (192.0 + 0.2595 * sigma + eps)

    C = 7.47 * np.exp(-0.133 * np.power(sigma + eps, 0.55))
    B = 0.02526 * np.power(sigma + eps, 0.54)
    E = 0.715 * np.exp(-3.59e-4 * sigma)

    ws_midflame = ws_d * 60.0 * wind_adj        # m/min
    phi_w = (C
             * np.power(np.clip(ws_midflame, eps, None), B)
             * np.power(np.clip(beta_ratio,  eps, None), -E))
    phi_s = 5.275 * np.power(np.clip(beta, eps, None), -0.3) * tan_slope_sq

    Q_ig      = 250.0 + 1116.0 * fmc_d
    eps_field = np.exp(-138.0 / (sigma + eps))
    rho_b     = w0 / (delta + eps)

    ros = ((I_R * xi * (1.0 + phi_w + phi_s))
           / (rho_b * eps_field * Q_ig + eps)
           / 60.0)           # m/s
    ros = np.clip(ros, 0.0, None)
    ros[~burnable] = 0.0

    return ros.astype(np.float32)


# ---------------------------------------------------------------------------
# Correlated random field (FFT spectral method)
# ---------------------------------------------------------------------------

def draw_correlated_field(
    grid_shape: tuple[int, int],
    correlation_length: float,                # metres
    resolution: float,                        # metres per cell
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw a spatially correlated unit-variance Gaussian random field.

    Uses FFT spectral filtering with a Gaussian kernel:
        S(k) ∝ exp(-0.5 · k² · L²)

    Returns: float32[rows, cols], zero mean, unit variance.
    """
    if rng is None:
        rng = np.random.default_rng()

    rows, cols = grid_shape
    noise = rng.standard_normal(grid_shape)

    noise_fft = np.fft.fft2(noise)

    kr = np.fft.fftfreq(rows, d=resolution)   # cycles/metre
    kc = np.fft.fftfreq(cols, d=resolution)
    KR, KC = np.meshgrid(kr, kc, indexing="ij")
    k2 = KR**2 + KC**2

    spectrum = np.exp(-0.5 * k2 * correlation_length**2)
    filtered = np.real(np.fft.ifft2(noise_fft * spectrum))

    std = float(filtered.std())
    if std > 1e-10:
        filtered /= std

    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# FireStateEstimator
# ---------------------------------------------------------------------------

class FireStateEstimator:
    """
    Manages fire state estimation from sparse observations.
    Produces arrival time fields (seconds) for ensemble initialization.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        dx: float,           # metres per cell
        max_arrival: float,  # sentinel value in seconds (e.g. 2 × horizon_s)
    ) -> None:
        self.grid_shape  = grid_shape
        self.dx          = dx
        self.max_arrival = max_arrival

        # Best-estimate arrival time field (seconds).  MAX_ARRIVAL = no fire.
        self.arrival_time = np.full(grid_shape, max_arrival, dtype=np.float32)

        # Confidence: 1.0 = directly observed, 0.0 = purely model-extrapolated.
        self.confidence = np.zeros(grid_shape, dtype=np.float32)

        # Last time a cell was directly observed (seconds since sim start).
        self.last_observed = np.full(grid_shape, -np.inf, dtype=np.float32)

        # Arrival time uncertainty (seconds) at each cell.
        self.arrival_uncertainty = np.full(grid_shape, np.inf, dtype=np.float32)

        # Whether a hard reset occurred this cycle.
        self.reset_this_cycle = False

    def set_ignition(
        self,
        ignition_cell: tuple[int, int],
        ignition_time: float = 0.0,
    ) -> None:
        """Initialize with known ignition point (time in seconds)."""
        r, c = ignition_cell
        self.arrival_time[r, c]        = ignition_time
        self.confidence[r, c]          = 1.0
        self.last_observed[r, c]       = ignition_time
        self.arrival_uncertainty[r, c] = 0.0

    def reconstruct_arrival_time(
        self,
        fire_observations: list[FireDetectionObservation],
        current_time: float,
        terrain: TerrainData,
        gp_prior: GPPrior,
    ) -> np.ndarray:
        """
        Reconstruct a complete arrival time field (seconds) from sparse fire
        observations using fast marching with Rothermel ROS.

        Observed fire cells anchor the march; unobserved cells are filled
        using travel time = dx / ROS propagated outward from sources.

        Returns: float32[rows, cols] arrival time in seconds.
        """
        rows, cols = self.grid_shape
        arrival = np.full((rows, cols), self.max_arrival, dtype=np.float32)

        # Step 1: anchor arrival times at observed cells
        for obs in fire_observations:
            r, c = obs.location
            if obs.is_fire:
                arrival[r, c]               = obs.timestamp
                self.confidence[r, c]       = obs.confidence
                self.last_observed[r, c]    = current_time

        # Confirmed no-fire cells stay at MAX_ARRIVAL
        for obs in fire_observations:
            r, c = obs.location
            if not obs.is_fire:
                arrival[r, c]            = self.max_arrival
                self.confidence[r, c]    = obs.confidence
                self.last_observed[r, c] = current_time

        # Step 2: ROS field from GP-mean environmental parameters
        ros_field = compute_ros_field(
            terrain=terrain,
            fmc=gp_prior.fmc_mean,
            wind_speed=gp_prior.wind_speed_mean,
            wind_direction=gp_prior.wind_dir_mean,
        )

        # Step 3: fast march from observed fire cells
        arrival = self._fast_march(arrival, ros_field)

        # Step 4: compute arrival time uncertainty
        self._compute_uncertainty(fire_observations, current_time, ros_field)

        self.arrival_time = arrival
        return arrival

    def _fast_march(
        self,
        arrival: np.ndarray,
        ros_field: np.ndarray,
    ) -> np.ndarray:
        """
        Dijkstra-based fast marching to fill arrival times at unobserved cells.

        Observed fire cells are sources.  Propagates outward: for each
        neighbour, candidate = source_arrival + (dx × dist_factor) / ROS.
        Non-burnable cells (ROS < 1e-6) are skipped.
        """
        rows, cols = arrival.shape
        visited    = np.zeros((rows, cols), dtype=bool)
        threshold  = self.max_arrival * 0.9

        pq: list[tuple[float, int, int]] = []
        for r in range(rows):
            for c in range(cols):
                if arrival[r, c] < threshold:
                    heapq.heappush(pq, (float(arrival[r, c]), r, c))

        # 8-connected neighbourhood with Euclidean distance factors
        neighbours = [
            (-1,  0, 1.000), ( 1,  0, 1.000),
            ( 0, -1, 1.000), ( 0,  1, 1.000),
            (-1, -1, 1.414), (-1,  1, 1.414),
            ( 1, -1, 1.414), ( 1,  1, 1.414),
        ]

        while pq:
            t, r, c = heapq.heappop(pq)
            if visited[r, c]:
                continue
            visited[r, c] = True
            arrival[r, c] = t

            for dr, dc, dist_factor in neighbours:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                    ros = float(ros_field[nr, nc])
                    if ros < 1e-6:
                        continue
                    travel_time = (self.dx * dist_factor) / ros
                    new_arrival = t + travel_time
                    if new_arrival < arrival[nr, nc]:
                        arrival[nr, nc] = new_arrival
                        heapq.heappush(pq, (float(new_arrival), nr, nc))

        return arrival

    def _compute_uncertainty(
        self,
        fire_observations: list[FireDetectionObservation],
        current_time: float,
        ros_field: np.ndarray,
    ) -> None:
        """
        Compute arrival time uncertainty (seconds) at each cell.

        Three sources:
          1. Observation noise (~30 s satellite, ~5 s drone thermal)
          2. Propagation uncertainty grows with distance from nearest observation
          3. Temporal uncertainty grows with time since last observation
        """
        obs_mask = np.zeros(self.grid_shape, dtype=bool)
        for obs in fire_observations:
            if obs.is_fire:
                obs_mask[obs.location] = True

        if not obs_mask.any():
            self.arrival_uncertainty = np.full(
                self.grid_shape, self.max_arrival, dtype=np.float32)
            return

        dist_to_obs = distance_transform_edt(~obs_mask) * self.dx   # metres

        obs_noise_s      = 30.0   # satellite temporal resolution
        propagation_rate = 0.6    # seconds per metre
        spatial_unc      = obs_noise_s + propagation_rate * dist_to_obs

        time_since  = np.clip(current_time - self.last_observed, 0.0, 7200.0)
        burnable    = ros_field > 0
        max_ros     = float(np.percentile(ros_field[burnable], 95)) if burnable.any() else 1.0
        temporal_unc = time_since * max_ros / (max_ros + 1e-6)

        self.arrival_uncertainty = np.sqrt(
            spatial_unc.astype(np.float64)**2 + temporal_unc.astype(np.float64)**2
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# EnsembleFireState
# ---------------------------------------------------------------------------

class EnsembleFireState:
    """
    Maintains per-member fire state across IGNIS cycles.

    Member arrival times are stored in **seconds**.  Conversion from
    EnsembleResult minutes (carry_forward) and to the fire engine (get_initial_phi)
    is handled internally.

    Two modes:
      CONTINUE — carry each member's fire state forward unchanged.
      RESET    — reinitialize all members from a reconstructed arrival time
                 field with correlated perturbations at the uncertain perimeter.
    """

    def __init__(
        self,
        n_members: int,
        grid_shape: tuple[int, int],
        dx: float,           # metres per cell
        max_arrival: float,  # sentinel in seconds (e.g. 2 × horizon_min × 60)
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.n_members   = n_members
        self.grid_shape  = grid_shape
        self.dx          = dx
        self.max_arrival = max_arrival
        self.rng         = rng or np.random.default_rng()

        # Per-member arrival times in seconds.  None until first initialization.
        self.member_arrival_times: Optional[np.ndarray] = None

        # Per-member SDF in metres (negative inside fire, positive outside).
        self.member_phi: Optional[np.ndarray] = None

        self.initialized = False

        # Simulation time (seconds from t=0) at which carry_forward was last
        # called.  Arrival times stored after carry_forward are in seconds
        # *relative to this epoch*, so burned-by-T requires elapsed = T - epoch.
        # Initialisation paths (from ignition / fire_state / reconstruction)
        # store absolute arrival times, so epoch stays 0.
        self._carry_forward_time_s: float = 0.0

    # ------------------------------------------------------------------
    # Initialization paths
    # ------------------------------------------------------------------

    def initialize_from_ignition(self, ignition_cell: tuple[int, int]) -> None:
        """First cycle: all members start from the same known ignition point."""
        sdf = self._compute_sdf_from_point(ignition_cell)
        self.member_phi = np.tile(sdf, (self.n_members, 1, 1)).copy()
        self.member_arrival_times = np.full(
            (self.n_members, *self.grid_shape), self.max_arrival, dtype=np.float32)
        r, c = ignition_cell
        self.member_arrival_times[:, r, c] = 0.0
        self.initialized = True

    def initialize_from_fire_state(self, fire_state: np.ndarray) -> None:
        """
        Initialize from a 2D burn mask (backward-compat path for cycle 1 when
        no ignition cell is known).  All members receive the same SDF derived
        from the burn mask.

        fire_state: bool/float32[rows, cols], 1 = burned, 0 = unburned.
        """
        burned = fire_state > 0.5
        if not burned.any():
            r0, c0 = self.grid_shape[0] // 2, self.grid_shape[1] // 2
            burned = burned.copy()
            burned[r0, c0] = True
        d_in  = distance_transform_edt(burned).astype(np.float32)
        d_out = distance_transform_edt(~burned).astype(np.float32)
        sdf   = np.where(burned, -d_in, d_out) * self.dx
        self.member_phi = np.tile(sdf, (self.n_members, 1, 1)).copy()
        self.member_arrival_times = np.full(
            (self.n_members, *self.grid_shape), self.max_arrival, dtype=np.float32)
        self.member_arrival_times[:, burned] = 0.0
        self.initialized = True

    def initialize_from_reconstruction(
        self,
        arrival_time_field: np.ndarray,   # float32[rows, cols], seconds
        arrival_uncertainty: np.ndarray,  # float32[rows, cols], seconds
        current_time: float,              # seconds
    ) -> None:
        """
        Hard reset: create fresh ensemble around reconstructed arrival times.

        Each member gets the reconstructed arrival time perturbed by a
        spatially correlated noise field scaled by local uncertainty.
        Members agree in well-observed regions and diverge at the uncertain
        perimeter.
        """
        self.member_arrival_times = np.zeros(
            (self.n_members, *self.grid_shape), dtype=np.float32)

        # Shared masks — computed once
        deep_burned  = arrival_time_field < (float(arrival_time_field.min()) + 3600.0)
        far_exterior = arrival_time_field > self.max_arrival * 0.9

        for n in range(self.n_members):
            noise = draw_correlated_field(
                self.grid_shape,
                correlation_length=500.0,   # metres
                resolution=self.dx,
                rng=self.rng,
            )
            perturbation   = noise * arrival_uncertainty
            member_arrival = arrival_time_field + perturbation

            # Deep interior: keep exact (fire already well-observed there)
            member_arrival[deep_burned]  = arrival_time_field[deep_burned]
            # Far exterior: definitely unburned — force to sentinel
            member_arrival[far_exterior] = self.max_arrival

            self.member_arrival_times[n] = member_arrival.astype(np.float32)

        self._recompute_phi_from_arrival_times(current_time)
        self.initialized = True

    # ------------------------------------------------------------------
    # Cycle interface
    # ------------------------------------------------------------------

    def carry_forward(
        self,
        member_arrival_times_min: np.ndarray,
        carry_forward_time_s: float = 0.0,
    ) -> None:
        """
        Continue mode: store each member's fire state from the completed cycle.

        member_arrival_times_min: float32[N, rows, cols] in **minutes** relative
          to the cycle that just completed (i.e. minutes from carry_forward_time_s).
          Converted to seconds internally.

        carry_forward_time_s: absolute simulation time (seconds from t=0) at
          which this ensemble was generated.  Stored so _recompute_phi_from_
          arrival_times can compute the correct elapsed time on the next call.

        The caller's ensemble may have a different N than self.n_members was
        initialised with (e.g. CycleRunner passes n_members=15 while the
        orchestrator defaults to ENSEMBLE_SIZE=100).  We update n_members to
        track the actual ensemble size so _recompute_phi_from_arrival_times
        iterates the correct range.
        """
        self.member_arrival_times = (
            member_arrival_times_min * 60.0).astype(np.float32)
        self._carry_forward_time_s = carry_forward_time_s
        # Keep n_members in sync with the actual stored array size.
        self.n_members = self.member_arrival_times.shape[0]

    def get_initial_phi(self, current_time_s: float) -> np.ndarray:
        """
        Return per-member SDF (metres) for the fire engine.
        Shape: (N, rows, cols).  Recomputed from stored arrival times.
        """
        self._recompute_phi_from_arrival_times(current_time_s)
        assert self.member_phi is not None
        return self.member_phi

    def resample(self, indices: np.ndarray) -> None:
        """
        After particle filter reweighting, resample members.
        indices: int array of length N with member indices (with repetition).
        """
        if self.member_arrival_times is not None:
            self.member_arrival_times = self.member_arrival_times[indices]
        if self.member_phi is not None:
            self.member_phi = self.member_phi[indices]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _recompute_phi_from_arrival_times(self, current_time_s: float) -> None:
        """Convert per-member arrival times (seconds) to SDF (metres).

        Stored arrival times are in seconds *relative to _carry_forward_time_s*
        (the simulation time of the ensemble that was carried forward).
        Initialisation paths (ignition / fire_state / reconstruction) store
        absolute seconds, keeping _carry_forward_time_s = 0.
        """
        assert self.member_arrival_times is not None
        # elapsed_s: how much simulated time has passed since the epoch at which
        # the stored arrival times were computed.  A cell with stored arrival
        # <= elapsed_s has actually burned by current_time_s.
        elapsed_s = current_time_s - self._carry_forward_time_s
        self.member_phi = np.zeros(
            (self.n_members, *self.grid_shape), dtype=np.float32)
        for n in range(self.n_members):
            # Use <= so cells at arrival == 0 (ignition / already burned) are
            # treated as burned even when elapsed_s == 0.
            burned = self.member_arrival_times[n] <= elapsed_s
            if not burned.any():
                self.member_phi[n] = np.full(self.grid_shape, self.dx, dtype=np.float32)
                continue
            if burned.all():
                self.member_phi[n] = np.full(self.grid_shape, -self.dx, dtype=np.float32)
                continue
            d_in  = distance_transform_edt(burned).astype(np.float32)
            d_out = distance_transform_edt(~burned).astype(np.float32)
            self.member_phi[n] = np.where(burned, -d_in, d_out) * self.dx

    def _compute_sdf_from_point(self, cell: tuple[int, int]) -> np.ndarray:
        """SDF centred on a single ignition point (metres)."""
        rows, cols = self.grid_shape
        y, x = np.mgrid[0:rows, 0:cols]
        dist = np.sqrt((y - cell[0])**2 + (x - cell[1])**2) * self.dx
        return (dist - 2.0 * self.dx).astype(np.float32)


# ---------------------------------------------------------------------------
# ConsistencyChecker
# ---------------------------------------------------------------------------

class ConsistencyChecker:
    """
    Compares new fire observations against ensemble consensus.
    Triggers hard reset when disagreement exceeds threshold.
    """

    def __init__(
        self,
        disagreement_threshold: float = 0.2,
        min_observations: int = 5,
    ) -> None:
        self.disagreement_threshold = disagreement_threshold
        self.min_observations       = min_observations

    def check(
        self,
        fire_observations: list[FireDetectionObservation],
        ensemble_result: EnsembleResult,
        current_time_s: float,
        last_cycle_time_s: float = 0.0,
    ) -> tuple[bool, float]:
        """
        Compare fire observations against ensemble consensus.

        EnsembleResult.member_arrival_times are in **minutes from the previous
        cycle start** (last_cycle_time_s); current_time_s is in **absolute
        seconds**.  We evaluate burned-state at the elapsed time between cycles.

        Returns: (should_reset, disagreement_fraction)
          should_reset = True when the ensemble has drifted significantly.
        """
        if len(fire_observations) < self.min_observations:
            return False, 0.0

        elapsed_min = (current_time_s - last_cycle_time_s) / 60.0
        ensemble_burned  = ensemble_result.member_arrival_times < elapsed_min
        burn_probability = ensemble_burned.mean(axis=0)   # (rows, cols)

        disagreements = 0
        total         = 0
        for obs in fire_observations:
            r, c     = obs.location
            p_burned = float(burn_probability[r, c])
            if obs.is_fire and p_burned < 0.3:
                disagreements += 1
            elif not obs.is_fire and p_burned > 0.7:
                disagreements += 1
            total += 1

        disagreement_fraction = disagreements / total if total > 0 else 0.0
        should_reset          = disagreement_fraction > self.disagreement_threshold
        return should_reset, disagreement_fraction


# ---------------------------------------------------------------------------
# Particle filter
# ---------------------------------------------------------------------------

def particle_filter_fire(
    ensemble_result: EnsembleResult,
    fire_observations: list[FireDetectionObservation],
    current_time_s: float,
    n_members: int,
    last_cycle_time_s: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Reweight and resample ensemble members based on fire observations.

    Members consistent with observations receive higher weight; contradicting
    members are downweighted.  Resampling occurs only when ESS drops below
    50 % of N (avoids unnecessary diversity loss).

    EnsembleResult arrival times are in **minutes from last_cycle_time_s** (the
    previous cycle start).  We evaluate burned-state at the elapsed time between
    cycles, not at absolute simulation time, to avoid a systematic reference-
    point mismatch that causes ensemble collapse.

    Returns: (resampled_indices: int[N], n_eff: float)
    """
    weights      = np.ones(n_members, dtype=np.float64)
    elapsed_min  = (current_time_s - last_cycle_time_s) / 60.0
    ensemble_burned  = ensemble_result.member_arrival_times < elapsed_min

    for obs in fire_observations:
        r, c = obs.location
        conf = float(obs.confidence)
        for n in range(n_members):
            member_burned = bool(ensemble_burned[n, r, c])
            if obs.is_fire:
                weights[n] *= conf if member_burned else (1.0 - conf)
            else:
                weights[n] *= conf if not member_burned else (1.0 - conf)

    weight_sum = weights.sum()
    if weight_sum < 1e-30:
        weights = np.ones(n_members, dtype=np.float64) / n_members
    else:
        weights /= weight_sum

    n_eff = 1.0 / float((weights**2).sum())

    if n_eff < n_members * 0.5:
        indices = systematic_resample(weights, n_members)
    else:
        indices = np.arange(n_members)

    return indices, n_eff


def systematic_resample(weights: np.ndarray, n: int) -> np.ndarray:
    """Standard particle filter systematic resampling."""
    positions = (np.arange(n) + np.random.uniform()) / n
    cumsum    = np.cumsum(weights)
    return np.searchsorted(cumsum, positions)
