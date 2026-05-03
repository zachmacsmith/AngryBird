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
    GP_TERRAIN_ALPHA,
    GP_TERRAIN_BETA,
    GRID_RESOLUTION_M,
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

        # Observation stores — separate for each variable
        self._fmc_locs: list[tuple[int, int]] = []
        self._fmc_vals: list[float] = []
        self._fmc_sigmas: list[float] = []

        self._ws_locs: list[tuple[int, int]] = []
        self._ws_vals: list[float] = []
        self._ws_sigmas: list[float] = []

        self._wd_locs: list[tuple[int, int]] = []
        self._wd_vals: list[float] = []
        self._wd_sigmas: list[float] = []

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
    ) -> None:
        """Add RAWS station observations for all three variables."""
        n = len(locations)
        fmc_sigmas = fmc_sigmas or [self.noise_fmc] * n
        ws_sigmas = ws_sigmas or [self.noise_wind_speed] * n
        wd_sigmas = wd_sigmas or [self.noise_wind_dir] * n

        self._fmc_locs.extend(locations)
        self._fmc_vals.extend(fmc_vals)
        self._fmc_sigmas.extend(fmc_sigmas)

        self._ws_locs.extend(locations)
        self._ws_vals.extend(ws_vals)
        self._ws_sigmas.extend(ws_sigmas)

        self._wd_locs.extend(locations)
        self._wd_vals.extend(wd_vals)
        self._wd_sigmas.extend(wd_sigmas)

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
    ) -> None:
        """Add drone FMC (and optionally wind speed + direction) observations.

        Triggers refit on next predict().  Wind direction is now accepted so that
        the live-estimator and assimilation pipeline can update the posterior wind
        direction field from drone nadir measurements.
        """
        self._fmc_locs.extend(locations)
        self._fmc_vals.extend(fmc_vals)
        self._fmc_sigmas.extend(fmc_sigmas)

        if ws_vals is not None:
            _ws_locs = ws_locs if ws_locs is not None else locations
            ws_sigmas = ws_sigmas or [self.noise_wind_speed] * len(_ws_locs)
            self._ws_locs.extend(_ws_locs)
            self._ws_vals.extend(ws_vals)
            self._ws_sigmas.extend(ws_sigmas)

        if wd_vals is not None:
            _wd_locs = wd_locs if wd_locs is not None else locations
            wd_sigmas = wd_sigmas or [self.noise_wind_dir] * len(_wd_locs)
            self._wd_locs.extend(_wd_locs)
            self._wd_vals.extend(wd_vals)
            self._wd_sigmas.extend(wd_sigmas)

        self._dirty = True

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _build_gp(self, length_scale: float, noise_sigma: float) -> GaussianProcessRegressor:
        # Use alpha (fixed noise) instead of WhiteKernel so the optimizer cannot
        # collapse all variance into noise. With sparse RAWS, WhiteKernel absorbs
        # all signal and the posterior variance becomes spatially flat (PotentialBugs1 §9).
        ls_bounds = (length_scale * 0.5, length_scale * 5.0)
        kernel = C(1.0, (1e-3, 1e3)) * _TerrainMatern32(
            length_scale, self.alpha, self.beta, length_scale_bounds=ls_bounds
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise_sigma ** 2,   # fixed known measurement noise
            n_restarts_optimizer=2,
            normalize_y=True,
            copy_X_train=True,
        )

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
        if previous_gp is not None:
            # Lock hyperparameters to the first-fit values; only update the posterior.
            # Re-optimizing every cycle lets the kernel amplitude grow when new obs span
            # a wider range, reversing variance reductions from earlier cycles.
            gp = GaussianProcessRegressor(
                kernel=previous_gp.kernel_,
                alpha=noise_sigma ** 2,
                n_restarts_optimizer=0,
                normalize_y=True,
                copy_X_train=True,
                optimizer=None,
            )
        else:
            gp = self._build_gp(length_scale, noise_sigma)
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
                    gp = self._build_gp(length_scale, noise_sigma)
                    gp.fit(X, y)
                else:
                    raise
        return gp

    def fit(self) -> None:
        """Fit GPs to all current observations."""
        # FMC: fit residuals from Nelson prior mean so the GP corrects the
        # physics model rather than interpolating raw values.
        fmc_vals = self._fmc_vals
        if self._nelson_mean is not None and self._fmc_locs:
            ri = [int(r) for r, _ in self._fmc_locs]
            ci = [int(c) for _, c in self._fmc_locs]
            nelson_at_obs = self._nelson_mean[ri, ci].astype(np.float64)
            fmc_vals = [float(v) - float(n)
                        for v, n in zip(self._fmc_vals, nelson_at_obs)]

        self._gp_fmc = self._fit_variable(
            self._fmc_locs, fmc_vals, self._fmc_sigmas,
            self.length_scale_fmc, self.noise_fmc,
            previous_gp=self._gp_fmc,
        )
        self._gp_ws = self._fit_variable(
            self._ws_locs, self._ws_vals, self._ws_sigmas,
            self.length_scale_wind, self.noise_wind_speed,
            previous_gp=self._gp_ws,
        )
        self._gp_wd = self._fit_variable(
            self._wd_locs, self._wd_vals, self._wd_sigmas,
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
        ws_mean, ws_var = self._predict_variable(
            self._gp_ws, X_grid, shape,
            default_mean=5.0,    # 5 m/s fallback
            default_variance=4.0,
        )
        wd_mean, wd_var = self._predict_variable(
            self._gp_wd, X_grid, shape,
            default_mean=270.0,  # westerly fallback
            default_variance=900.0,
        )

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
