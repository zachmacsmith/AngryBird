"""
GP Prior: spatial estimation of FMC and wind fields with calibrated uncertainty.

Public surface:
  IGNISGPPrior          — main class: fit, predict, conditional_variance
  draw_correlated_field — FFT-based spatially correlated noise (used by fire engine)
  draw_gp_scaled_field  — correlated noise scaled by GP posterior std

Observation management has moved to ObservationStore. IGNISGPPrior reads from
the store each cycle via get_decayed_for_type(); it no longer stores observations
internally. To add observations, call obs_store.add_raws() / add_drone_observations().
"""

from __future__ import annotations

import logging
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
    FMC_MAX_FRACTION,
    FMC_MIN_FRACTION,
    GP_CORRELATION_LENGTH_FMC_M,
    GP_CORRELATION_LENGTH_WIND_M,
    GP_DEFAULT_FMC_MEAN,
    GP_DEFAULT_FMC_VARIANCE,
    GP_DEFAULT_WIND_DIR_MEAN,
    GP_DEFAULT_WIND_DIR_VARIANCE,
    GP_DEFAULT_WIND_SPEED_MEAN,
    GP_DEFAULT_WIND_SPEED_VARIANCE,
    GP_NOISE_FMC,
    GP_NOISE_WIND_DIR,
    GP_NOISE_WIND_SPEED,
    GP_TERRAIN_ALPHA,
    GP_TERRAIN_BETA,
    GRID_RESOLUTION_M,
    WIND_SPEED_MAX_MS,
    WIND_SPEED_MIN_MS,
)
from .observations import (
    DataPoint, DroneObservation as DroneObs, ObservationStore,
    ObservationType, RAWSObservation, VariableType,
)
from .types import GPPrior, TerrainData

logger = logging.getLogger(__name__)


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
    GP-based estimation of FMC and wind fields over a terrain grid.

    Observations are managed externally by ObservationStore. This class reads
    from the store on every fit() call and produces a GP posterior. It never
    stores observations itself.

    Typical usage:
        obs_store = ObservationStore(decay_config)
        obs_store.add_raws("STA1", raws_obs)
        gp = IGNISGPPrior(obs_store, terrain=terrain_data)
        prior = gp.predict(shape=(200, 200))        # -> GPPrior
        # After drone observations arrive:
        obs_store.add_drone_observations(drone_obs)
        prior = gp.predict(shape=(200, 200))        # refits automatically
    """

    def __init__(
        self,
        obs_store: Optional[ObservationStore] = None,
        terrain: Optional[TerrainData] = None,
        resolution_m: float = GRID_RESOLUTION_M,
        length_scale_fmc: float = GP_CORRELATION_LENGTH_FMC_M,
        length_scale_wind: float = GP_CORRELATION_LENGTH_WIND_M,
        alpha: float = GP_TERRAIN_ALPHA,
        beta: float = GP_TERRAIN_BETA,
        noise_fmc: float = GP_NOISE_FMC,
        noise_wind_speed: float = GP_NOISE_WIND_SPEED,
        noise_wind_dir: float = GP_NOISE_WIND_DIR,
    ):
        if obs_store is None:
            obs_store = ObservationStore()
        self._store = obs_store
        self.terrain = terrain
        self.resolution_m = resolution_m
        self.length_scale_fmc = length_scale_fmc
        self.length_scale_wind = length_scale_wind
        self.alpha = alpha
        self.beta = beta
        self.noise_fmc = noise_fmc
        self.noise_wind_speed = noise_wind_speed
        self.noise_wind_dir = noise_wind_dir

        # Cached fitted regressors. Reset to None to force a fresh kernel
        # optimization (e.g. after residual basis changes). Kept non-None to
        # lock the kernel (only update posterior) when the training set is stable.
        self._gp_fmc: Optional[GaussianProcessRegressor] = None
        self._gp_ws:  Optional[GaussianProcessRegressor] = None
        self._gp_wd:  Optional[GaussianProcessRegressor] = None

        # Track obs count from the last fit per variable.
        # If the count changes (new obs or pruning), force a fresh kernel fit.
        self._last_fmc_count: int = -1
        self._last_ws_count:  int = -1
        self._last_wd_count:  int = -1

        # Current time — updated by fit(); predict() uses stored value
        self._current_time: float = 0.0

        # Cached grid features (rebuilt when shape changes)
        self._cached_shape:  Optional[tuple[int, int]] = None
        self._cached_X_grid: Optional[np.ndarray] = None

        # V_grid cache for conditional_variance — the expensive triangular solve
        # solve_triangular(L_, K_train_grid) depends only on the GP's training set
        # and the fixed grid, so it is valid for all greedy selector iterations
        # within a cycle.  Invalidated by id(gp.X_train_) when fit() replaces the
        # training array.
        self._cv_V_grid:     Optional[np.ndarray] = None  # (n_train, n_grid) float64
        self._cv_V_grid_key: Optional[int]        = None  # id(gp.X_train_) at cache fill

        # Physics-informed prior means.
        # When set, observations are fitted as residuals (obs - prior) and
        # predictions add the prior back. See set_nelson_mean() and
        # set_wind_prior_mean() for details.
        self._nelson_mean:     Optional[np.ndarray] = None
        self._wind_speed_mean: Optional[np.ndarray] = None
        self._wind_dir_mean:   Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Prior mean setters
    # ------------------------------------------------------------------

    def set_nelson_mean(self, field: np.ndarray) -> None:
        """
        Set the Nelson (2000) terrain-aware FMC field as the GP prior mean.

        After this call observations are fitted as residuals (observed - Nelson).
        Predictions add Nelson back, so gp.predict() returns physics-corrected
        FMC estimates everywhere — not just near stations. The GP variance field
        is unaffected (still reflects distance from observations).

        Call once per IGNIS cycle before gp.predict(), passing the Nelson field
        recomputed for the current hour / temperature / humidity.
        """
        self._nelson_mean = np.asarray(field, dtype=np.float32).copy()
        self._gp_fmc = None  # residuals changed — force fresh kernel optimization

    def set_wind_prior_mean(
        self, ws_field: np.ndarray, wd_field: np.ndarray
    ) -> None:
        """
        Set spatially varying prior mean fields for wind speed and direction.

        Mirrors set_nelson_mean for FMC: wind observations are fitted as residuals
        (observed − prior), predictions add the prior back. The GP then corrects
        where drones disagree with the background.

        Call once at startup (and again after a wind-shift event) with the
        scenario's base wind fields. Without this, predict() falls back to a
        hardcoded 270° west / 5 m/s prior.

        Wind direction residuals use circular arithmetic so wrap-around at 0°/
        360° does not introduce a ~360° spike.
        """
        self._wind_speed_mean = np.asarray(ws_field, dtype=np.float32).copy()
        self._wind_dir_mean   = np.asarray(wd_field, dtype=np.float32).copy()
        self._gp_ws = None  # force fresh kernel — residuals changed
        self._gp_wd = None

    # ------------------------------------------------------------------
    # Backward-compatibility wrappers (delegate to ObservationStore)
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
        timestamp: float = 0.0,
        station_id: str = "COMPAT",
    ) -> None:
        """Compatibility shim: converts lists to RAWSObservation and adds to store."""
        n = len(locations)
        fmc_sigmas = fmc_sigmas or [self.noise_fmc] * n
        ws_sigmas  = ws_sigmas  or [self.noise_wind_speed] * n
        wd_sigmas  = wd_sigmas  or [self.noise_wind_dir] * n
        for i, (loc, fmc, ws, wd, sf, ss, sd) in enumerate(zip(
            locations, fmc_vals, ws_vals, wd_vals, fmc_sigmas, ws_sigmas, wd_sigmas
        )):
            self._store.add_raws(RAWSObservation(
                _source_id=f"{station_id}_{i}",
                _timestamp=timestamp,
                location=loc,
                fmc=fmc, fmc_sigma=sf,
                wind_speed=ws, wind_speed_sigma=ss,
                wind_direction=wd, wind_direction_sigma=sd,
            ))

    def add_observations(
        self,
        locations: list[tuple[int, int]],
        fmc_vals: list[float],
        fmc_sigmas: list[float],
        ws_vals: Optional[list[float]] = None,
        ws_sigmas: Optional[list[float]] = None,
        wd_vals: Optional[list[float]] = None,
        wd_sigmas: Optional[list[float]] = None,
        timestamp: float = 0.0,
    ) -> None:
        """Compatibility shim: converts lists to DroneObservation and adds to store."""
        _ws_s = ws_sigmas  or ([self.noise_wind_speed] * len(locations) if ws_vals else None)
        _wd_s = wd_sigmas  or ([self.noise_wind_dir]   * len(locations) if wd_vals else None)
        new_obs = []
        for i, (loc, fmc, sf) in enumerate(zip(locations, fmc_vals, fmc_sigmas)):
            new_obs.append(DroneObs(
                _source_id=f"compat_drone_{i}",
                _timestamp=timestamp,
                location=loc,
                fmc=fmc, fmc_sigma=sf,
                wind_speed=ws_vals[i]  if ws_vals else None,
                wind_speed_sigma=_ws_s[i] if _ws_s else None,
                wind_direction=wd_vals[i] if wd_vals else None,
                wind_direction_sigma=_wd_s[i] if _wd_s else None,
            ))
        self._store.add_batch(new_obs)

    def update_time(self, current_time: float) -> None:
        """Compatibility shim: updates stored time (used by predict())."""
        self._current_time = current_time

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
        prior_variance: float = 1.0,
    ) -> Optional[GaussianProcessRegressor]:
        if not locs:
            return None
        X = _obs_features(locs, self.terrain, self.resolution_m)
        y = np.array(vals, dtype=np.float64)
        # Per-observation alpha: effective (age-inflated) sigmas for drone obs,
        # original sigmas for RAWS obs. Stale observations get inflated alpha
        # and lower weight — decay is gradual, not binary.
        alpha = np.array(sigmas, dtype=np.float64) ** 2
        if previous_gp is not None:
            # Lock hyperparameters to the first-fit values; only update posterior.
            # Re-optimizing every cycle lets kernel amplitude grow when new obs span
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
            # Bound C to [1% of prior, prior]. C is the kernel amplitude and
            # equals the posterior variance at cells far from all observations.
            # Capping at prior_variance ensures RAWS observations can only reduce
            # total grid variance — never increase it by projecting a larger kernel
            # amplitude into unobserved regions (which caused negative "RAWS entropy
            # reduction" when diverse RAWS wind speeds drove C >> GP_DEFAULT_*_VARIANCE).
            c_bounds = (prior_variance * 0.01, prior_variance)
            kernel = C(prior_variance, c_bounds) * _TerrainMatern32(
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

    def fit(self, current_time: float = 0.0) -> None:
        """
        Fit GPs to all observations from the store at current_time.

        Calls store.get_data_points(current_time, variable) which returns DataPoints
        with decay already applied. RAWS points carry original sigma; drone/satellite
        carry age-inflated sigma. Thinning (one point per correlation-length cell) is
        applied inside the store query.

        Kernel locking: if the observation count for a variable hasn't changed since
        the last fit, hyperparameters are held fixed (only the posterior is updated).
        A changed count triggers a fresh kernel optimization.
        """
        self._current_time = current_time
        thin_fmc = int(self.length_scale_fmc  / self.resolution_m)
        thin_ws  = int(self.length_scale_wind / self.resolution_m)

        # --- FMC ---
        fmc_pts = self._store.get_data_points(
            current_time, VariableType.FMC, min_spacing_cells=thin_fmc)
        n_fmc = len(fmc_pts)
        if n_fmc != self._last_fmc_count:
            self._gp_fmc = None
            self._last_fmc_count = n_fmc

        fmc_locs   = [p.location for p in fmc_pts]
        fmc_vals   = [p.value    for p in fmc_pts]
        fmc_sigmas = [p.sigma    for p in fmc_pts]
        fmc_vals_fit = list(fmc_vals)
        if self._nelson_mean is not None and fmc_locs:
            ri = [int(r) for r, _ in fmc_locs]
            ci = [int(c) for _, c in fmc_locs]
            nelson_at_obs = self._nelson_mean[ri, ci].astype(np.float64)
            fmc_vals_fit = [float(v) - float(n)
                            for v, n in zip(fmc_vals, nelson_at_obs)]

        self._gp_fmc = self._fit_variable(
            fmc_locs, fmc_vals_fit, fmc_sigmas,
            self.length_scale_fmc, self.noise_fmc,
            previous_gp=self._gp_fmc,
            prior_variance=GP_DEFAULT_FMC_VARIANCE,
        )

        # --- Wind speed ---
        ws_pts = self._store.get_data_points(
            current_time, VariableType.WIND_SPEED, min_spacing_cells=thin_ws)
        n_ws = len(ws_pts)
        if n_ws != self._last_ws_count:
            self._gp_ws = None
            self._last_ws_count = n_ws

        ws_locs   = [p.location for p in ws_pts]
        ws_vals   = [p.value    for p in ws_pts]
        ws_sigmas = [p.sigma    for p in ws_pts]
        ws_vals_fit = list(ws_vals)
        if self._wind_speed_mean is not None and ws_locs:
            ri = [int(r) for r, _ in ws_locs]
            ci = [int(c) for _, c in ws_locs]
            ws_prior_at_obs = self._wind_speed_mean[ri, ci].astype(np.float64)
            ws_vals_fit = [float(v) - float(p)
                           for v, p in zip(ws_vals, ws_prior_at_obs)]

        self._gp_ws = self._fit_variable(
            ws_locs, ws_vals_fit, ws_sigmas,
            self.length_scale_wind, self.noise_wind_speed,
            previous_gp=self._gp_ws,
            prior_variance=GP_DEFAULT_WIND_SPEED_VARIANCE,
        )

        # --- Wind direction ---
        wd_pts = self._store.get_data_points(
            current_time, VariableType.WIND_DIRECTION, min_spacing_cells=thin_ws)
        n_wd = len(wd_pts)
        if n_wd != self._last_wd_count:
            self._gp_wd = None
            self._last_wd_count = n_wd

        wd_locs   = [p.location for p in wd_pts]
        wd_vals   = [p.value    for p in wd_pts]
        wd_sigmas = [p.sigma    for p in wd_pts]
        wd_vals_fit = list(wd_vals)
        if self._wind_dir_mean is not None and wd_locs:
            ri = [int(r) for r, _ in wd_locs]
            ci = [int(c) for _, c in wd_locs]
            wd_prior_at_obs = self._wind_dir_mean[ri, ci].astype(np.float64)
            wd_vals_fit = [float(((v - p + 180.0) % 360.0) - 180.0)
                           for v, p in zip(wd_vals, wd_prior_at_obs)]

        self._gp_wd = self._fit_variable(
            wd_locs, wd_vals_fit, wd_sigmas,
            self.length_scale_wind, self.noise_wind_dir,
            previous_gp=self._gp_wd,
            prior_variance=GP_DEFAULT_WIND_DIR_VARIANCE,
        )

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
            var  = np.full((rows, cols), default_variance, dtype=np.float32)
            return mean, var

        mu, sigma = gp.predict(X_grid, return_std=True)
        mean = mu.reshape(rows, cols).astype(np.float32)
        var  = (sigma ** 2).reshape(rows, cols).astype(np.float32)
        return mean, var

    def _get_grid_features(self, shape: tuple[int, int]) -> np.ndarray:
        if shape != self._cached_shape:
            self._cached_X_grid = _grid_features(shape, self.terrain, self.resolution_m)
            self._cached_shape = shape
        return self._cached_X_grid

    def predict(self, shape: tuple[int, int]) -> GPPrior:
        """
        Predict FMC and wind mean+variance over a (rows, cols) grid.
        Always refits from the store at the stored current_time (set by the
        last fit() call, or 0.0 if fit() has never been called explicitly).
        """
        self.fit(self._current_time)

        X_grid = self._get_grid_features(shape)

        # Nelson active: GP predicts residuals; add Nelson mean back.
        # default_mean=0 → no-obs fallback is pure Nelson (residual = 0).
        _nelson = (self._nelson_mean
                   if self._nelson_mean is not None
                   and self._nelson_mean.shape == tuple(shape)
                   else None)
        fmc_mean, fmc_var = self._predict_variable(
            self._gp_fmc, X_grid, shape,
            default_mean    = 0.0 if _nelson is not None else GP_DEFAULT_FMC_MEAN,
            default_variance= GP_DEFAULT_FMC_VARIANCE,
        )
        if _nelson is not None:
            fmc_mean = np.clip(fmc_mean + _nelson, FMC_MIN_FRACTION, FMC_MAX_FRACTION).astype(np.float32)

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
            default_mean    = 0.0 if _ws_prior is not None else GP_DEFAULT_WIND_SPEED_MEAN,
            default_variance= GP_DEFAULT_WIND_SPEED_VARIANCE,
        )
        if _ws_prior is not None:
            ws_mean = np.clip(ws_mean + _ws_prior, WIND_SPEED_MIN_MS, WIND_SPEED_MAX_MS).astype(np.float32)

        wd_mean, wd_var = self._predict_variable(
            self._gp_wd, X_grid, shape,
            default_mean    = 0.0 if _wd_prior is not None else GP_DEFAULT_WIND_DIR_MEAN,
            default_variance= GP_DEFAULT_WIND_DIR_VARIANCE,
        )
        if _wd_prior is not None:
            wd_mean = ((wd_mean + _wd_prior) % 360.0).astype(np.float32)

        return GPPrior(
            fmc_mean          = fmc_mean,
            fmc_variance      = np.clip(fmc_var, 0.0, None),
            wind_speed_mean   = ws_mean,
            wind_speed_variance = np.clip(ws_var, 0.0, None),
            wind_dir_mean     = wd_mean,
            wind_dir_variance = np.clip(wd_var, 0.0, None),
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

        Uses the GP posterior kernel so the update is consistent with the variance
        already conditioned on existing observations.
        Posterior cross-covariance is computed from sklearn's stored Cholesky L_.

        `current_variance` may be (rows*cols,) flat or (rows, cols).
        Returns same shape as input.
        """
        if self._gp_fmc is None:
            return current_variance

        flat     = current_variance.ravel().astype(np.float64)
        shape_in = current_variance.shape
        noise_sigma = noise_sigma or self.noise_fmc

        X_grid = self._cached_X_grid
        if X_grid is None:
            return current_variance

        X_new  = _obs_features([new_loc], self.terrain, self.resolution_m)
        gp     = self._gp_fmc
        kernel = gp.kernel_

        y_std_raw = getattr(gp, "_y_train_std", 1.0)
        y_std_arr = np.asarray(y_std_raw, dtype=np.float64).ravel()

        if y_std_arr.size == 0:
            y_std = 1.0
        else:
            y_std = float(y_std_arr[0])
        y_var = y_std ** 2

        K_prior_cross_norm = kernel(X_grid, X_new).ravel()
        K_prior_self_norm  = float(kernel(X_new, X_new)[0, 0])

        K_train_new = kernel(gp.X_train_, X_new)
        V_new = solve_triangular(gp.L_, K_train_new, lower=True)

        # V_grid = solve_triangular(L_, K_train_grid) is the dominant cost
        # (300×80k triangular solve).  It depends only on the training set and
        # fixed grid, so cache it and reuse across all greedy selector iterations
        # within a cycle.  Invalidate when fit() replaces X_train_ (new id).
        x_train_id = id(gp.X_train_)
        if self._cv_V_grid is None or self._cv_V_grid_key != x_train_id:
            K_train_grid = kernel(gp.X_train_, X_grid)
            self._cv_V_grid     = solve_triangular(gp.L_, K_train_grid, lower=True)
            self._cv_V_grid_key = x_train_id
        V_grid = self._cv_V_grid

        k_post_cross = (K_prior_cross_norm - (V_grid * V_new).sum(axis=0)) * y_var
        k_post_self  = (K_prior_self_norm  - float((V_new ** 2).sum()))     * y_var

        denominator = max(k_post_self, 0.0) + noise_sigma ** 2 + 1e-6
        updated = flat - k_post_cross ** 2 / denominator
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
    kernel_fft = np.exp(-2.0 * (np.pi * correlation_length * freq_grid) ** 2)
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * np.sqrt(kernel_fft)))
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
    Cells with low GP variance (near stations) get small perturbations.

    Returns float32[rows, cols].
    """
    unit_field = draw_correlated_field(shape, correlation_length, resolution)
    return (unit_field * np.sqrt(np.clip(gp_variance, 0.0, None))).astype(np.float32)
