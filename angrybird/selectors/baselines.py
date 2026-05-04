"""
Baseline selectors for comparison:

  UniformSelector    — K locations on a regular grid (ignores information field)
  FireFrontSelector  — K locations evenly spaced along the predicted fire perimeter
"""

from __future__ import annotations

import time

import numpy as np

from ..config import FIRE_FRONT_HI_PROB, FIRE_FRONT_LO_PROB, GRID_RESOLUTION_M, MIN_SELECTION_SPACING_M
from ..gp import IGNISGPPrior
from ..types import EnsembleResult, InformationField, SelectionResult
from .base import spacing_mask


class UniformSelector:
    """K locations on a regular grid across the full domain."""

    name = "uniform"
    kind = "points"

    def __init__(self, resolution_m: float = GRID_RESOLUTION_M) -> None:
        self.resolution_m = resolution_m

    def select(
        self,
        info_field: InformationField,
        gp: IGNISGPPrior,
        ensemble: EnsembleResult,
        k: int,
    ) -> SelectionResult:
        t0 = time.perf_counter()
        rows, cols = info_field.w.shape

        # Find the grid spacing that yields at least K points
        # spacing s → n_rows = rows//s, n_cols = cols//s → n = n_rows*n_cols ≥ k
        spacing = max(1, int(np.sqrt(rows * cols / k)))
        while True:
            r_pts = np.arange(spacing // 2, rows, spacing)
            c_pts = np.arange(spacing // 2, cols, spacing)
            grid = [(int(r), int(c)) for r in r_pts for c in c_pts]
            if len(grid) >= k or spacing == 1:
                break
            spacing -= 1

        # Sort by info value (descending) so the "best K uniform" are returned
        grid.sort(key=lambda rc: float(info_field.w[rc[0], rc[1]]), reverse=True)
        selected = grid[:k]

        return SelectionResult(
            selected_locations=selected,
            marginal_gains=[float(info_field.w[r, c]) for r, c in selected],
            cumulative_gain=list(np.cumsum([float(info_field.w[r, c])
                                            for r, c in selected])),
            strategy_name=self.name,
            compute_time_s=time.perf_counter() - t0,
        )


class FireFrontSelector:
    """
    K locations evenly spaced along the predicted fire perimeter.

    Perimeter = cells with burn_probability in [lo_thresh, hi_thresh].
    Falls back to all cells with burn_prob > lo_thresh if the perimeter band
    has fewer than K cells.
    """

    name = "fire_front"
    kind = "points"

    def __init__(
        self,
        lo_thresh: float = FIRE_FRONT_LO_PROB,
        hi_thresh: float = FIRE_FRONT_HI_PROB,
        min_spacing_m: float = MIN_SELECTION_SPACING_M,
        resolution_m: float = GRID_RESOLUTION_M,
    ) -> None:
        self.lo_thresh = lo_thresh
        self.hi_thresh = hi_thresh
        self.min_spacing_m = min_spacing_m
        self.resolution_m = resolution_m

    def select(
        self,
        info_field: InformationField,
        gp: IGNISGPPrior,
        ensemble: EnsembleResult,
        k: int,
    ) -> SelectionResult:
        t0 = time.perf_counter()
        shape = info_field.w.shape
        bp = ensemble.burn_probability
        min_cells = self.min_spacing_m / self.resolution_m

        # Find perimeter cells; widen band if too few
        perimeter_mask = (bp >= self.lo_thresh) & (bp <= self.hi_thresh)
        if perimeter_mask.sum() < k:
            perimeter_mask = bp >= self.lo_thresh
        if not perimeter_mask.any():
            # Degenerate case: no fire yet — fall back to high-w cells
            perimeter_mask = info_field.w > 0

        cand_rows, cand_cols = np.where(perimeter_mask)
        if len(cand_rows) == 0:
            # No valid cells at all — return empty result
            return SelectionResult(
                selected_locations=[],
                marginal_gains=[],
                cumulative_gain=[],
                strategy_name=self.name,
                compute_time_s=time.perf_counter() - t0,
            )

        # Sort candidates by angle around the fire centroid to get
        # evenly-spaced coverage along the perimeter
        cy = float(cand_rows.mean())
        cx = float(cand_cols.mean())
        angles = np.arctan2(cand_rows - cy, cand_cols - cx)
        order = np.argsort(angles)
        cand_rows = cand_rows[order]
        cand_cols = cand_cols[order]

        # Pick every (n/k)-th candidate, respecting min spacing
        selected: list[tuple[int, int]] = []
        n_cands = len(cand_rows)
        step = max(1, n_cands // k)
        idx = 0
        while len(selected) < k and idx < n_cands * 2:
            r, c = int(cand_rows[idx % n_cands]), int(cand_cols[idx % n_cands])
            if not spacing_mask(shape, selected, min_cells)[r, c]:
                selected.append((r, c))
            idx += step

        # If we couldn't fill K with spacing constraint, relax and fill greedily
        if len(selected) < k:
            for i in range(n_cands):
                r, c = int(cand_rows[i]), int(cand_cols[i])
                if (r, c) not in selected:
                    if not spacing_mask(shape, selected, min_cells)[r, c]:
                        selected.append((r, c))
                    if len(selected) == k:
                        break

        return SelectionResult(
            selected_locations=selected,
            marginal_gains=[float(info_field.w[r, c]) for r, c in selected],
            cumulative_gain=list(np.cumsum([float(info_field.w[r, c])
                                            for r, c in selected])),
            strategy_name=self.name,
            compute_time_s=time.perf_counter() - t0,
        )
