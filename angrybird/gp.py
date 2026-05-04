"""
GP Prior: spatial estimation of FMC and wind fields with calibrated uncertainty.

Public surface:
  IGNISGPPrior          — main class: fit, predict, conditional_variance, add_observations
  draw_correlated_field — FFT-based spatially correlated noise (used by fire engine)
  draw_gp_scaled_field  — correlated noise scaled by GP posterior std
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Hyperparameter,
    Kernel,
)

from .config import (
    GP_CORRELATION_LENGTH_FMC_M,
    GP_CORRELATION_LENGTH_WIND_M,
    GP_NOISE_FMC,
    GP_NOISE_WIND_DIR,
    GP_NOISE_WIND_SPEED,
    GP_OBS_DECAY_DROP_FACTOR,
    GP_TERRAIN_ALPHA,
    GP_TERRAIN_BETA,
    GRID_RESOLUTION_M,
    TAU_FMC_S,
    TAU_WIND_DIR_S,
    TAU_WIND_SPEED_S,
)
from .types import GPPrior, TerrainData


# ---------------------------------------------------------------------------
# Terrain-aware Matérn 3/2 kernel
# ---------------------------------------------------------------------------

class _TerrainMatern32(Kernel):
    """
    Matérn ν=3/2 kernel where distance is augmented by terrain differences.

    Feature layout for X rows: [northing_m, easting_m, elevation_m, aspect_deg]
    Terrain distance: d = geo_dist + alpha*elev_diff + beta*aspect_diff
    """

    def __init__(
        self,
        length_scale: float = 1500.0,
        alpha: float = GP_TERRAIN_ALPHA,
        beta: float = GP_TERRAIN_BETA,
        length_scale_bounds: tuple[float, float] = (100.0, 50_000.0),
    ):
        self.length_scale = length_scale
        self.alpha = alpha
        self.beta = beta
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self) -> Hyperparameter:
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def _terrain_dist(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        geo = cdist(X[:, :2], Y[:, :2])
        elev_diff = np.abs(X[:, 2:3] - Y[:, 2:3].T)
        raw = np.abs(X[:, 3:4] - Y[:, 3:4].T) % 360
        aspect_diff = np.minimum(raw, 360.0 - raw)
        return geo + self.alpha * elev_diff + self.beta * aspect_diff

    def __call__(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None, eval_gradient: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        d = self._terrain_dist(X, Y) / self.length_scale
        K = (1.0 + np.sqrt(3.0) * d) * np.exp(-np.sqrt(3.0) * d)
        if eval_gradient:
            dK_dl = 3.0 * d ** 2 * np.exp(-np.sqrt(3.0) * d) / self.length_scale
            return K, dK_dl[:, :, np.newaxis]
        return K

    def diag(self, X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[0], dtype=np.float64)

    def is_stationary(self) -> bool:
        return True

    def __repr__(self) -> str:
        return (
            f"TerrainMatern32(length_scale={self.length_scale:.1f}, "
            f"alpha={self.alpha}, beta={self.beta})"
        )


# ---------------------------------------------------------------------------
# Feature construction helpers
# ---------------------------------------------------------------------------

def _obs_features(
    locations: list[tuple[int, int]],
    terrain: Optional[TerrainData],
    resolution_m: float,
) -> np.ndarray:
    """Convert observation (row, col) list to (n, 4) feature matrix."""
    locs = np.array(locations, dtype=np.float64)
    northing = locs[:, 0] * resolution_m
    easting = locs[:, 1] * resolution_m
    if terrain is not None:
        rows = locs[:, 0].astype(int)
        cols = locs[:, 1].astype(int)
        elevation = terrain.elevation[rows, cols].astype(np.float64)
        aspect = terrain.aspect[rows, cols].astype(np.float64)
    else:
        elevation = np.zeros(len(locs))
        aspect = np.zeros(len(locs))
    return np.column_stack([northing, easting, elevation, aspect])


def _grid_features(
    shape: tuple[int, int],
    terrain: Optional[TerrainData],
    resolution_m: float,
) -> np.ndarray:
    """Build (rows*cols, 4) feature matrix for a full grid."""
    rows, cols = shape
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    northing = rr.ravel() * resolution_m
    easting = cc.ravel() * resolution_m
    if terrain is not None:
        elevation = terrain.elevation[rr, cc].ravel().astype(np.float64)
        aspect = terrain.aspect[rr, cc].ravel().astype(np.float64)
    else:
        elevation = np.zeros(rows * cols)
        aspect = np.zeros(rows * cols)
    return np.column_stack([northing, easting, elevation, aspect])


# ---------------------------------------------------------------------------
# Main GP class
# ---------------------------------------------------------------------------

class IGNISGPPrior:
    """
    Manages GP-based estimation of FMC and wind fields.

    Typical usage:
        gp = IGNISGPPrior(terrain=terrain_data, resolution_m=50.0)
        gp.add_raws(raws_locs, fmc_vals, ws_vals, wd_vals)
        prior = gp.predict(shape=(200, 200))  # -> GPPrior dataclass
        # After drone observations:
        gp.add_observations(drone_locs, fmc_vals, fmc_sigmas)
        prior = gp.predict(shape=(200, 200))  # updated prior
    """

    def __init__(
        self,
        terrain: Optional[TerrainData] = None,
        resolution_m: float = GRID_RESOLUTION_M,
        length_scale_fmc: float = GP_CORRELATION_LENGTH_FMC_M,
        length_scale_wind: float = GP_CORRELATION_LENGTH_WIND_M,
        alpha: float = GP_TERRAIN_ALPHA,
        beta: float = GP_TERRAIN_BETA,
        noise_fmc: float = GP_NOISE_FMC,
        noise_wind_speed: float = GP_NOISE_WIND_SPEED,
        noise_wind_dir: float = GP_NOISE_WIND_DIR,
        tau_fmc: float = TAU_FMC_S,
        tau_wind_speed: float = TAU_WIND_SPEED_S,
        tau_wind_dir: float = TAU_WIND_DIR_S,
    ):
        self.terrain = terrain
        self.resolution_m = resolution_m
        self.length_scale_fmc = length_scale_fmc
        self.length_scale_wind = length_scale_wind
        self.alpha = alpha
        self.beta = beta
        self.noise_fmc = noise_fmc
        self.noise_wind_speed = noise_wind_speed
        self.noise_wind_dir = noise_wind_dir

        # RAWS observation stores — replaced entirely on each add_raws() call.
        # No timestamps: RAWS readings are always treated as current; calling
        # add_raws() again (e.g. each IGNIS cycle) supersedes the previous reading.
        self._raws_fmc_locs:   list[tuple[int, int]] = []
        self._raws_fmc_vals:   list[float] = []
        self._raws_fmc_sigmas: list[float] = []

        self._raws_ws_locs:   list[tuple[int, int]] = []
        self._raws_ws_vals:   list[float] = []
        self._raws_ws_sigmas: list[float] = []

        self._raws_wd_locs:   list[tuple[int, int]] = []
        self._raws_wd_vals:   list[float] = []
        self._raws_wd_sigmas: list[float] = []

        # Drone observation stores — accumulate over time, subject to temporal decay.
        self._fmc_locs: list[tuple[int, int]] = []
        self._fmc_vals: list[float] = []
        self._fmc_sigmas: list[float] = []

        self._ws_locs: list[tuple[int, int]] = []
        self._ws_vals: list[float] = []
        self._ws_sigmas: list[float] = []

        self._wd_locs: list[tuple[int, int]] = []
        self._wd_vals: list[float] = []
        self._wd_sigmas: list[float] = []

        # Observation timestamps — parallel to drone stores only.
        # Used to compute effective sigma = orig_sigma * exp(age / tau).
        self._fmc_times: list[float] = []
        self._ws_times:  list[float] = []
        self._wd_times:  list[float] = []

        # GP clock — updated once per simulation cycle.
        self._current_time: float = 0.0
        self._tau_fmc:        float = tau_fmc
        self._tau_wind_speed: float = tau_wind_speed
        self._tau_wind_dir:   float = tau_wind_dir

        # Cached fitted regressors — invalidated when new observations arrive
        self._gp_fmc: Optional[GaussianProcessRegressor] = None
        self._gp_ws: Optional[GaussianProcessRegressor] = None
        self._gp_wd: Optional[GaussianProcessRegressor] = None
        self._dirty: bool = True

        # Cached grid features (rebuilt when shape changes)
        self._cached_shape: Optional[tuple[int, int]] = None
        self._cached_X_grid: Optional[np.ndarray] = None

        # Nelson (2000) physics-informed FMC prior mean.
        # When set, FMC obs are fitted as residuals from this field and
        # predictions add it back — GP corrects Nelson where data disagrees.
        self._nelson_mean: Optional[np.ndarray] = None

        # Scenario/weather-model wind prior mean fields.
        # When set, wind obs are fitted as residuals and predictions add back
        # the prior — same pattern as Nelson for FMC.  Corrects the hardcoded
        # 270° / 5 m/s fallback to the actual scenario baseline.
        self._wind_speed_mean: Optional[np.ndarray] = None
        self._wind_dir_mean:   Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Nelson mean function
    # ------------------------------------------------------------------

    def set_nelson_mean(self, field: np.ndarray) -> None:
        """
        Set the Nelson (2000) terrain-aware FMC field as the GP prior mean.

        After this call, RAWS/drone FMC observations are fitted as residuals
        (observed - Nelson).  Predictions add Nelson back, so gp.predict()
        returns physics-corrected FMC estimates everywhere — not just near
        stations.  The GP variance field is unaffected (still reflects distance
        from observations).

        Call once per IGNIS cycle before gp.predict(), passing the Nelson
        field recomputed for the current hour / temperature / humidity.
        """
        self._nelson_mean = np.asarray(field, dtype=np.float32).copy()
        self._dirty = True   # residuals changed → refit on next predict()

    # ------------------------------------------------------------------
    # Wind prior mean function
    # ------------------------------------------------------------------

    def set_wind_prior_mean(
        self, ws_field: np.ndarray, wd_field: np.ndarray
    ) -> None:
        """
        Set spatially varying prior mean fields for wind speed and direction.

        Mirrors set_nelson_mean for FMC: drone wind observations are fitted as
        residuals (observed − prior), predictions add the prior back.  The GP
        then corrects where drones disagree with the background.

        Call once at simulation startup (and again after a wind-shift event)
        with the scenario's base wind fields.  Without this, predict() falls
        back to a hardcoded 270° west / 5 m/s prior regardless of scenario.

        Wind direction residuals use circular arithmetic so wrap-around at 0°/
        360° does not introduce a ~360° spike.
        """
        self._wind_speed_mean = np.asarray(ws_field, dtype=np.float32).copy()
        self._wind_dir_mean   = np.asarray(wd_field, dtype=np.float32).copy()
        self._dirty = True

    def update_time(self, t: float) -> None:
        """
        Advance the GP clock to simulation time t (seconds).

        Triggers a refit on the next predict() call so that effective observation
        sigmas (which grow with age) and any stale-observation pruning are applied.
        No-op if t has not changed.
        """
        if t != self._current_time:
            self._current_time = t
            self._dirty = True

    # ------------------------------------------------------------------
    # Observation ingestion
    # ------------------------------------------------------------------

    def add_raws(
        self,
        locations: list[tuple[int, int]],
        fmc_vals: list[float],
        ws_vals: list[float],
        wd_vals: list[float],
        fmc_sigmas: Optional[list[float]] = None,
        ws_sigmas: Optional[list[float]] = None,
        wd_sigmas: Optional[list[float]] = None,
        obs_time: Optional[float] = None,  # retained for API compatibility; unused
    ) -> None:
        """Replace RAWS station observations for all three variables.

        Each call replaces any previous RAWS readings — the latest reading from a
        fixed ground station supersedes the old one.  Call this every IGNIS cycle
        with fresh telemetry and the GP will always anchor to current conditions.

        RAWS observations are never subject to temporal decay.  They represent
        the instantaneous station reading at the moment of the call, which is
        always "now" from the GP's perspective.
        """
        n = len(locations)
        fmc_sigmas = fmc_sigmas or [self.noise_fmc] * n
        ws_sigmas  = ws_sigmas  or [self.noise_wind_speed] * n
        wd_sigmas  = wd_sigmas  or [self.noise_wind_dir] * n

        self._raws_fmc_locs   = list(locations)
        self._raws_fmc_vals   = list(fmc_vals)
        self._raws_fmc_sigmas = list(fmc_sigmas)

        self._raws_ws_locs   = list(locations)
        self._raws_ws_vals   = list(ws_vals)
        self._raws_ws_sigmas = list(ws_sigmas)

        self._raws_wd_locs   = list(locations)
        self._raws_wd_vals   = list(wd_vals)
        self._raws_wd_sigmas = list(wd_sigmas)

        self._dirty = True

    def add_observations(
        self,
        locations: list[tuple[int, int]],
        fmc_vals: list[float],
        fmc_sigmas: list[float],
        ws_vals: Optional[list[float]] = None,
        ws_sigmas: Optional[list[float]] = None,
        ws_locs: Optional[list[tuple[int, int]]] = None,
        wd_vals: Optional[list[float]] = None,
        wd_sigmas: Optional[list[float]] = None,
        wd_locs: Optional[list[tuple[int, int]]] = None,
        obs_times: Optional[list[float]] = None,
        ws_times:  Optional[list[float]] = None,
        wd_times:  Optional[list[float]] = None,
    ) -> None:
        """Add drone FMC (and optionally wind speed + direction) observations.

        Triggers refit on next predict().  Wind direction is now accepted so that
        the live-estimator and assimilation pipeline can update the posterior wind
        direction field from drone nadir measurements.
        """
        self._fmc_locs.extend(locations)
        self._fmc_vals.extend(fmc_vals)
        self._fmc_sigmas.extend(fmc_sigmas)
        _fmc_t = obs_times if obs_times is not None else [self._current_time] * len(locations)
        self._fmc_times.extend(_fmc_t)

        if ws_vals is not None:
            _ws_locs = ws_locs if ws_locs is not None else locations
            ws_sigmas = ws_sigmas or [self.noise_wind_speed] * len(_ws_locs)
            self._ws_locs.extend(_ws_locs)
            self._ws_vals.extend(ws_vals)
            self._ws_sigmas.extend(ws_sigmas)
            _ws_t = ws_times if ws_times is not None else [self._current_time] * len(_ws_locs)
            self._ws_times.extend(_ws_t)

        if wd_vals is not None:
            _wd_locs = wd_locs if wd_locs is not None else locations
            wd_sigmas = wd_sigmas or [self.noise_wind_dir] * len(_wd_locs)
            self._wd_locs.extend(_wd_locs)
            self._wd_vals.extend(wd_vals)
            self._wd_sigmas.extend(wd_sigmas)
            _wd_t = wd_times if wd_times is not None else [self._current_time] * len(_wd_locs)
            self._wd_times.extend(_wd_t)

        self._dirty = True

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_variable(
        self,
        locs: list[tuple[int, int]],
        vals: list[float],
        sigmas: list[float],
        length_scale: float,
        noise_sigma: float,
        previous_gp: Optional[GaussianProcessRegressor] = None,
    ) -> Optional[GaussianProcessRegressor]:
        if not locs:
            return None
        X = _obs_features(locs, self.terrain, self.resolution_m)
        y = np.array(vals, dtype=np.float64)
        # Per-observation alpha: uses effective (age-inflated) sigmas for drone obs
        # and original sigmas for RAWS obs.  This is the mechanism that makes decay
        # gradual rather than binary — stale observations have inflated alpha and thus
        # lower weight in the GP posterior, not just a hard prune at the drop threshold.
        alpha = np.array(sigmas, dtype=np.float64) ** 2
        if previous_gp is not None:
            # Lock hyperparameters to the first-fit values; only update the posterior.
            # Re-optimizing every cycle lets the kernel amplitude grow when new obs span
            # a wider range, reversing variance reductions from earlier cycles.
            gp = GaussianProcessRegressor(
                kernel=previous_gp.kernel_,
                alpha=alpha,
                n_restarts_optimizer=0,
                normalize_y=False,
                copy_X_train=True,
                optimizer=None,
            )
        else:
            ls_bounds = (length_scale * 0.5, length_scale * 5.0)
            kernel = C(1.0, (1e-3, 1e3)) * _TerrainMatern32(
                length_scale, self.alpha, self.beta,
                length_scale_bounds=ls_bounds,
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                n_restarts_optimizer=2,
                normalize_y=False,
                copy_X_train=True,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                gp.fit(X, y)
            except np.linalg.LinAlgError:
                if previous_gp is not None:
                    # Locked kernel became ill-conditioned with accumulated obs.
                    # Fall back to a fresh fit with hyperparameter optimization.
                    logger.warning(
                        "Locked-kernel Cholesky failed (%d obs); "
                        "re-fitting with optimization.", len(locs)
                    )
                    ls_bounds = (length_scale * 0.5, length_scale * 5.0)
                    kernel = C(1.0, (1e-3, 1e3)) * _TerrainMatern32(
                        length_scale, self.alpha, self.beta,
                        length_scale_bounds=ls_bounds,
                    )
                    gp = GaussianProcessRegressor(
                        kernel=kernel, alpha=alpha,
                        n_restarts_optimizer=2, normalize_y=False, copy_X_train=True,
                    )
                    gp.fit(X, y)
                else:
                    raise
        return gp

    def _prune_and_decay(
        self,
        locs:   list,
        vals:   list,
        sigmas: list,
        times:  list,
        tau:    float,
    ) -> tuple[list, list, list, list, list[float], bool]:
        """
        Apply temporal decay and prune observations whose effective sigma
        exceeds GP_OBS_DECAY_DROP_FACTOR × the original sigma.

        Returns
        -------
        (filtered_locs, filtered_vals, filtered_sigmas, filtered_times,
         effective_sigmas, pruned)

        Original sigmas are retained in storage (never inflated in-place).
        Effective sigmas are returned for use in GP fitting only.
        pruned is True when at least one observation was dropped.
        """
        if not locs:
            return [], [], [], [], [], False

        keep: list[bool] = []
        eff_sigmas: list[float] = []
        for sigma, t in zip(sigmas, times):
            age = max(0.0, self._current_time - t)
            sigma_eff = float(sigma * np.exp(age / tau))
            if sigma_eff <= GP_OBS_DECAY_DROP_FACTOR * sigma:
                keep.append(True)
                eff_sigmas.append(sigma_eff)
            else:
                keep.append(False)

        pruned = not all(keep)
        f_locs   = [l for l, k in zip(locs,   keep) if k]
        f_vals   = [v for v, k in zip(vals,   keep) if k]
        f_sigs   = [s for s, k in zip(sigmas, keep) if k]
        f_times  = [t for t, k in zip(times,  keep) if k]
        return f_locs, f_vals, f_sigs, f_times, eff_sigmas, pruned

    def fit(self) -> None:
        """Fit GPs to all current observations.

        RAWS observations are always included at their original sigmas (no decay).
        Drone observations are subject to temporal decay: effective sigma grows
        exponentially with age, and observations are pruned when sigma_eff > 10×
        original.  Merged RAWS + drone training data is passed to the GP so the
        posterior reflects both the stable ground-station anchor and the decaying
        flight observations.
        """
        # --- FMC: decay drone obs, then merge with RAWS ---
        (self._fmc_locs, self._fmc_vals, self._fmc_sigmas, self._fmc_times,
         fmc_eff_sigmas, fmc_pruned) = self._prune_and_decay(
            self._fmc_locs, self._fmc_vals, self._fmc_sigmas, self._fmc_times,
            self._tau_fmc,
        )
        if fmc_pruned:
            self._gp_fmc = None  # force full refit when training set changes

        # Merged locs/vals/sigmas: RAWS at original sigma, drones at effective sigma
        all_fmc_locs  = self._raws_fmc_locs + self._fmc_locs
        all_fmc_vals  = self._raws_fmc_vals  + list(self._fmc_vals)
        all_fmc_sigs  = self._raws_fmc_sigmas + (fmc_eff_sigmas if fmc_eff_sigmas else self._fmc_sigmas)

        # Nelson prior: subtract from combined locs
        fmc_vals_fit = list(all_fmc_vals)
        if self._nelson_mean is not None and all_fmc_locs:
            ri = [int(r) for r, _ in all_fmc_locs]
            ci = [int(c) for _, c in all_fmc_locs]
            nelson_at_obs = self._nelson_mean[ri, ci].astype(np.float64)
            fmc_vals_fit = [float(v) - float(n)
                            for v, n in zip(all_fmc_vals, nelson_at_obs)]

        self._gp_fmc = self._fit_variable(
            all_fmc_locs, fmc_vals_fit, all_fmc_sigs,
            self.length_scale_fmc, self.noise_fmc,
            previous_gp=self._gp_fmc,
        )

        # --- Wind speed: decay drone obs, merge with RAWS ---
        (self._ws_locs, self._ws_vals, self._ws_sigmas, self._ws_times,
         ws_eff_sigmas, ws_pruned) = self._prune_and_decay(
            self._ws_locs, self._ws_vals, self._ws_sigmas, self._ws_times,
            self._tau_wind_speed,
        )
        if ws_pruned:
            self._gp_ws = None

        all_ws_locs = self._raws_ws_locs + self._ws_locs
        all_ws_vals = self._raws_ws_vals  + list(self._ws_vals)
        all_ws_sigs = self._raws_ws_sigmas + (ws_eff_sigmas if ws_eff_sigmas else self._ws_sigmas)

        ws_vals_fit = list(all_ws_vals)
        if self._wind_speed_mean is not None and all_ws_locs:
            ri = [int(r) for r, _ in all_ws_locs]
            ci = [int(c) for _, c in all_ws_locs]
            ws_prior_at_obs = self._wind_speed_mean[ri, ci].astype(np.float64)
            ws_vals_fit = [float(v) - float(p)
                           for v, p in zip(all_ws_vals, ws_prior_at_obs)]
        self._gp_ws = self._fit_variable(
            all_ws_locs, ws_vals_fit, all_ws_sigs,
            self.length_scale_wind, self.noise_wind_speed,
            previous_gp=self._gp_ws,
        )

        # --- Wind direction: decay drone obs, merge with RAWS ---
        (self._wd_locs, self._wd_vals, self._wd_sigmas, self._wd_times,
         wd_eff_sigmas, wd_pruned) = self._prune_and_decay(
            self._wd_locs, self._wd_vals, self._wd_sigmas, self._wd_times,
            self._tau_wind_dir,
        )
        if wd_pruned:
            self._gp_wd = None

        all_wd_locs = self._raws_wd_locs + self._wd_locs
        all_wd_vals = self._raws_wd_vals  + list(self._wd_vals)
        all_wd_sigs = self._raws_wd_sigmas + (wd_eff_sigmas if wd_eff_sigmas else self._wd_sigmas)

        wd_vals_fit = list(all_wd_vals)
        if self._wind_dir_mean is not None and all_wd_locs:
            ri = [int(r) for r, _ in all_wd_locs]
            ci = [int(c) for _, c in all_wd_locs]
            wd_prior_at_obs = self._wind_dir_mean[ri, ci].astype(np.float64)
            wd_vals_fit = [float(((v - p + 180.0) % 360.0) - 180.0)
                           for v, p in zip(all_wd_vals, wd_prior_at_obs)]
        self._gp_wd = self._fit_variable(
            all_wd_locs, wd_vals_fit, all_wd_sigs,
            self.length_scale_wind, self.noise_wind_dir,
            previous_gp=self._gp_wd,
        )

        self._dirty = False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_variable(
        self,
        gp: Optional[GaussianProcessRegressor],
        X_grid: np.ndarray,
        shape: tuple[int, int],
        default_mean: float,
        default_variance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, variance) arrays of shape (rows, cols)."""
        rows, cols = shape
        if gp is None:
            mean = np.full((rows, cols), default_mean, dtype=np.float32)
            var = np.full((rows, cols), default_variance, dtype=np.float32)
            return mean, var

        mu, sigma = gp.predict(X_grid, return_std=True)
        mean = mu.reshape(rows, cols).astype(np.float32)
        var = (sigma ** 2).reshape(rows, cols).astype(np.float32)
        return mean, var

    def _get_grid_features(self, shape: tuple[int, int]) -> np.ndarray:
        if shape != self._cached_shape:
            self._cached_X_grid = _grid_features(shape, self.terrain, self.resolution_m)
            self._cached_shape = shape
        return self._cached_X_grid

    def predict(self, shape: tuple[int, int]) -> GPPrior:
        """
        Predict FMC and wind mean+variance over a (rows, cols) grid.
        Auto-fits if new observations have been added since last fit.
        """
        if self._dirty:
            self.fit()

        X_grid = self._get_grid_features(shape)

        # When Nelson is active, GP predicts residuals; add Nelson mean back.
        # default_mean=0 → no-obs fallback is pure Nelson (residual = 0).
        # Without Nelson, fall back to 0.10 flat prior.
        _nelson = (self._nelson_mean
                   if self._nelson_mean is not None
                   and self._nelson_mean.shape == tuple(shape)
                   else None)
        fmc_mean, fmc_var = self._predict_variable(
            self._gp_fmc, X_grid, shape,
            default_mean=0.0 if _nelson is not None else 0.10,
            default_variance=0.04,
        )
        if _nelson is not None:
            fmc_mean = np.clip(fmc_mean + _nelson, 0.02, 0.40).astype(np.float32)
        _ws_prior = (self._wind_speed_mean
                     if self._wind_speed_mean is not None
                     and self._wind_speed_mean.shape == tuple(shape)
                     else None)
        _wd_prior = (self._wind_dir_mean
                     if self._wind_dir_mean is not None
                     and self._wind_dir_mean.shape == tuple(shape)
                     else None)

        ws_mean, ws_var = self._predict_variable(
            self._gp_ws, X_grid, shape,
            default_mean=0.0 if _ws_prior is not None else 5.0,
            default_variance=4.0,
        )
        if _ws_prior is not None:
            ws_mean = np.clip(ws_mean + _ws_prior, 0.5, 25.0).astype(np.float32)

        wd_mean, wd_var = self._predict_variable(
            self._gp_wd, X_grid, shape,
            default_mean=0.0 if _wd_prior is not None else 270.0,
            default_variance=900.0,
        )
        if _wd_prior is not None:
            wd_mean = ((wd_mean + _wd_prior) % 360.0).astype(np.float32)

        return GPPrior(
            fmc_mean=fmc_mean,
            fmc_variance=np.clip(fmc_var, 0.0, None),
            wind_speed_mean=ws_mean,
            wind_speed_variance=np.clip(ws_var, 0.0, None),
            wind_dir_mean=wd_mean,
            wind_dir_variance=np.clip(wd_var, 0.0, None),
        )

    # ------------------------------------------------------------------
    # Conditional variance update (for greedy selector)
    # ------------------------------------------------------------------

    def conditional_variance(
        self,
        current_variance: np.ndarray,
        new_loc: tuple[int, int],
        noise_sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Closed-form posterior variance update after a hypothetical FMC observation
        at `new_loc` (greedy selector workhorse).

            σ²_new(x) = σ²(x) - k_post(x, x_new)² / (k_post(x_new, x_new) + σ²_noise)

        Uses the GP posterior kernel (not the prior kernel) so the update is
        consistent with the variance already conditioned on RAWS observations.
        Posterior cross-covariance is computed from sklearn's stored Cholesky L_.

        `current_variance` may be (rows*cols,) flat or (rows, cols).
        Returns same shape as input.
        """
        if self._gp_fmc is None:
            return current_variance

        flat = current_variance.ravel().astype(np.float64)
        shape_in = current_variance.shape
        noise_sigma = noise_sigma or self.noise_fmc

        X_grid = self._cached_X_grid
        if X_grid is None:
            return current_variance

        X_new = _obs_features([new_loc], self.terrain, self.resolution_m)
        gp = self._gp_fmc
        kernel = gp.kernel_

        # sklearn with normalize_y=True fits the kernel in the normalized y-scale.
        # All kernel evaluations below are in normalized scale.
        # Multiply by y_train_std^2 to recover original-scale covariances.
        y_std = float(getattr(gp, "_y_train_std", 1.0))
        y_var = y_std ** 2  # scale factor for converting normalized → original

        # Prior cross-covariance in normalized scale: k_norm(X_grid, x_new)
        K_prior_cross_norm = kernel(X_grid, X_new).ravel()          # (D,)
        K_prior_self_norm  = float(kernel(X_new, X_new)[0, 0])

        # Posterior correction via stored Cholesky L_ of (K_train + alpha*I)
        K_train_new  = kernel(gp.X_train_, X_new)         # (n_train, 1)
        K_train_grid = kernel(gp.X_train_, X_grid)        # (n_train, D)
        V_new  = solve_triangular(gp.L_, K_train_new,  lower=True)  # (n_train, 1)
        V_grid = solve_triangular(gp.L_, K_train_grid, lower=True)  # (n_train, D)

        # Posterior cross-covariance in NORMALIZED scale, then rescale to original
        k_post_cross = (K_prior_cross_norm - (V_grid * V_new).sum(axis=0)) * y_var  # (D,)
        k_post_self  = (K_prior_self_norm  - float((V_new ** 2).sum()))              * y_var

        # +1e-6 jitter (PotentialBugs2 §1)
        denominator = max(k_post_self, 0.0) + noise_sigma ** 2 + 1e-6
        updated = flat - k_post_cross ** 2 / denominator
        # Floor at 1e-10 — variance must stay positive
        return np.maximum(updated, 1e-10).reshape(shape_in).astype(np.float32)


# ---------------------------------------------------------------------------
# Correlated field generation (used by fire engine for perturbations)
# ---------------------------------------------------------------------------

def draw_correlated_field(
    shape: tuple[int, int],
    correlation_length: float,
    resolution: float,
) -> np.ndarray:
    """
    Draw one spatially correlated field with unit variance using circulant
    embedding (FFT convolution). O(D log D) for D = rows*cols.

    Args:
        shape:              (rows, cols)
        correlation_length: spatial correlation length in metres
        resolution:         grid cell size in metres

    Returns:
        float32[rows, cols] with zero mean and approximately unit variance.
    """
    white = np.random.randn(*shape)
    freqs_r = np.fft.fftfreq(shape[0], d=resolution)
    freqs_c = np.fft.fftfreq(shape[1], d=resolution)
    freq_grid = np.sqrt(freqs_r[:, None] ** 2 + freqs_c[None, :] ** 2)
    # Squared-exponential spectral density (approximates Gaussian kernel)
    kernel_fft = np.exp(-2.0 * (np.pi * correlation_length * freq_grid) ** 2)
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * np.sqrt(kernel_fft)))
    # Normalise to unit variance
    std = field.std()
    if std > 1e-10:
        field /= std
    return field.astype(np.float32)


def draw_gp_scaled_field(
    shape: tuple[int, int],
    correlation_length: float,
    resolution: float,
    gp_variance: np.ndarray,
) -> np.ndarray:
    """
    Draw a spatially correlated perturbation field scaled by GP posterior std.

    Cells with high GP variance (far from observations) get large perturbations.
    Cells with low GP variance (near RAWS stations) get small perturbations.

    Returns float32[rows, cols].
    """
    unit_field = draw_correlated_field(shape, correlation_length, resolution)
    return (unit_field * np.sqrt(np.clip(gp_variance, 0.0, None))).astype(np.float32)
