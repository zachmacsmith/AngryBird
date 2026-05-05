"""
QUBO selector: encode measurement selection as a Quadratic Unconstrained Binary
Optimisation problem and solve via a fallback chain:

  1. D-Wave QPU  (dwave-ocean-sdk, requires cloud credentials)
  2. Simulated Annealing  (neal, part of ocean-sdk or standalone)
  3. Pure-greedy fallback  (always succeeds, same result as GreedySelector)

The QUBO structure:
  - Diagonal Q_ii = -w_i + λ(1 - 2K)   (information value minus cardinality penalty)
  - Off-diagonal Q_ij = -J_ij + 2λ      (redundancy penalty from ensemble covariance)
  - Penalty λ = max|w_i| enforces the cardinality constraint ∑x_i = K

References: Krause et al. 2008 (submodular), D-Wave Ocean SDK docs.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from ..config import (
    BURNED_PROBABILITY_THRESHOLD,
    GRID_RESOLUTION_M,
    MIN_SELECTION_SPACING_M,
    QUBO_DWAVE_NUM_READS,
    QUBO_LAMBDA_INFLATION,
    QUBO_MAX_CANDIDATES,
    QUBO_SA_NUM_READS,
)
from ..gp import IGNISGPPrior
from ..types import EnsembleResult, InformationField, SelectionResult
from .base import spacing_mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate extraction (shared with QUBO construction)
# ---------------------------------------------------------------------------

def extract_candidates(
    w: np.ndarray,
    max_candidates: int,
    min_spacing_m: float,
    resolution_m: float,
    exclude_mask: Optional[np.ndarray] = None,
) -> list[tuple[int, int]]:
    """
    Return up to `max_candidates` cells sorted by w descending, with minimum
    spacing enforced so they are spread across the domain.
    """
    shape = w.shape
    min_cells = min_spacing_m / resolution_m
    w_work = w.copy()
    if exclude_mask is not None:
        w_work[exclude_mask] = 0.0

    candidates: list[tuple[int, int]] = []
    for flat_idx in np.argsort(w_work.ravel())[::-1]:
        if w_work.ravel()[flat_idx] <= 0.0:
            break
        loc = (int(flat_idx // shape[1]), int(flat_idx % shape[1]))
        if not spacing_mask(shape, candidates, min_cells)[loc]:
            candidates.append(loc)
        if len(candidates) >= max_candidates:
            break
    return candidates


# ---------------------------------------------------------------------------
# QUBO matrix construction
# ---------------------------------------------------------------------------

def build_qubo(
    candidates: list[tuple[int, int]],
    info_field: InformationField,
    ensemble: EnsembleResult,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build and return:
      Q  — (M, M) QUBO matrix (upper triangular convention)
      w_cand — (M,) information values at candidates

    Diagonal:   Q_ii = -w_i + λ(1 - 2K)
    Off-diag:   Q_ij = -J_ij + 2λ    where J encodes ensemble redundancy
    """
    M = len(candidates)
    if M == 0:
        return np.zeros((0, 0)), np.zeros(0)

    # Linear terms
    w_cand = np.array([float(info_field.w[r, c]) for r, c in candidates],
                      dtype=np.float64)

    # Quadratic redundancy terms from ensemble correlations
    J = np.zeros((M, M), dtype=np.float64)
    N = ensemble.n_members

    for var_name, w_var_field in info_field.w_by_variable.items():
        if var_name == "fmc":
            pert_field = ensemble.member_fmc_fields
            prior_mean = info_field.gp_variance["fmc"]   # used as proxy
        elif var_name == "wind_speed":
            pert_field = ensemble.member_wind_fields
            prior_mean = info_field.gp_variance.get("wind_speed",
                         np.zeros_like(info_field.w))
        else:
            continue

        # Extract ensemble perturbations at candidate locations: (N, M)
        vals = np.stack(
            [pert_field[:, r, c] for r, c in candidates], axis=1
        ).astype(np.float64)

        # Pairwise correlation (M, M) — guards against zero-std columns
        std = vals.std(axis=0)
        safe = std > 1e-10
        if not safe.any():
            continue
        rho = np.corrcoef(vals[:, safe].T)   # (M_safe, M_safe)

        # Per-candidate information weight for this variable
        w_var = np.array([float(w_var_field[r, c]) for r, c in candidates])

        # Outer product weight — measures joint importance of pairs
        outer = np.sqrt(np.outer(w_var[safe], w_var[safe]))
        J_sub = rho * outer

        idx = np.where(safe)[0]
        for ii, i in enumerate(idx):
            for jj, j in enumerate(idx):
                J[i, j] -= J_sub[ii, jj]

    # Assemble QUBO (upper triangular, minimisation convention)
    lam = float(np.max(np.abs(w_cand))) if w_cand.max() > 0 else 1.0
    # Slightly inflate λ to make cardinality constraint robust
    lam *= QUBO_LAMBDA_INFLATION

    Q = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        Q[i, i] = -w_cand[i] + lam * (1 - 2 * k)
    for i in range(M):
        for j in range(i + 1, M):
            Q[i, j] = -J[i, j] + 2 * lam

    return Q, w_cand


# ---------------------------------------------------------------------------
# Solver fallback chain
# ---------------------------------------------------------------------------

def _qubo_to_dict(Q: np.ndarray) -> dict[tuple[int, int], float]:
    """Convert upper-triangular matrix to Ocean SDK dict format."""
    M = Q.shape[0]
    return {
        (i, j): float(Q[i, j])
        for i in range(M)
        for j in range(i, M)
        if Q[i, j] != 0.0
    }


def _repair_to_k(sample: dict[int, int], k: int, w_cand: np.ndarray) -> list[int]:
    """Force a binary sample to have exactly k ones, favouring high-w indices."""
    selected = [i for i, v in sample.items() if v == 1]
    M = len(w_cand)
    while len(selected) > k:
        worst = min(selected, key=lambda i: w_cand[i])
        selected.remove(worst)
    while len(selected) < k:
        pool = [i for i in range(M) if i not in selected]
        if not pool:
            break
        selected.append(max(pool, key=lambda i: w_cand[i]))
    return selected


def _greedy_fallback(Q: np.ndarray, k: int, w_cand: np.ndarray) -> list[int]:
    """Pure greedy solve of the QUBO (ignores quadratic terms — emergency fallback)."""
    order = np.argsort(w_cand)[::-1]
    return list(map(int, order[:k]))


def solve_qubo(
    Q: np.ndarray,
    w_cand: np.ndarray,
    k: int,
    n_sa_reads: int = QUBO_SA_NUM_READS,
) -> tuple[list[int], str, float]:
    """
    Try the solver fallback chain. Returns (selected_indices, solver_name, energy).
    `selected_indices` are indices into the candidates list.
    """
    M = Q.shape[0]
    if M == 0:
        return [], "empty", float("nan")

    Q_dict = _qubo_to_dict(Q)

    # ── Attempt 1: D-Wave QPU ────────────────────────────────────────────
    try:
        from dwave.system import DWaveSampler, EmbeddingComposite  # type: ignore
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample_qubo(Q_dict, num_reads=QUBO_DWAVE_NUM_READS, label="ignis")
        best = response.first
        selected = _repair_to_k(dict(best.sample), k, w_cand)
        logger.info("QUBO solved on D-Wave QPU (energy=%.4f, chain_breaks=%s)",
                    best.energy, best.chain_break_fraction)
        return selected, "dwave_qpu", float(best.energy)
    except Exception as exc:
        logger.debug("D-Wave QPU unavailable: %s", exc)

    # ── Attempt 2: Simulated Annealing (neal / ocean) ────────────────────
    try:
        import neal  # type: ignore
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample_qubo(Q_dict, num_reads=n_sa_reads)
        best = response.first
        selected = _repair_to_k(dict(best.sample), k, w_cand)
        logger.info("QUBO solved by simulated annealing (energy=%.4f)", best.energy)
        return selected, "simulated_annealing", float(best.energy)
    except Exception as exc:
        logger.debug("SA solver unavailable: %s", exc)

    # ── Attempt 3: Pure-greedy fallback ─────────────────────────────────
    logger.warning("All QUBO solvers failed — falling back to pure greedy.")
    selected = _greedy_fallback(Q, k, w_cand)
    return selected, "greedy_fallback", float("nan")


# ---------------------------------------------------------------------------
# Selector class
# ---------------------------------------------------------------------------

class QUBOSelector:
    """
    QUBO-based selector with D-Wave QPU → SA → greedy fallback chain.

    Construction parameters are held on the object so the selector is
    stateless between cycles (safe to reuse across the registry).
    """

    name = "qubo"
    kind = "points"

    def __init__(
        self,
        max_candidates: int = QUBO_MAX_CANDIDATES,
        min_spacing_m: float = MIN_SELECTION_SPACING_M,
        resolution_m: float = GRID_RESOLUTION_M,
        n_sa_reads: int = 1000,
    ) -> None:
        self.max_candidates = max_candidates
        self.min_spacing_m = min_spacing_m
        self.resolution_m = resolution_m
        self.n_sa_reads = n_sa_reads

    def select(
        self,
        info_field: InformationField,
        gp: IGNISGPPrior,
        ensemble: EnsembleResult,
        k: int,
    ) -> SelectionResult:
        t0 = time.perf_counter()

        burned_mask = ensemble.burn_probability > 0.95

        # Step 1: extract top-M candidates from info field
        candidates = extract_candidates(
            info_field.w,
            max_candidates=self.max_candidates,
            min_spacing_m=self.min_spacing_m,
            resolution_m=self.resolution_m,
            exclude_mask=burned_mask,
        )

        if len(candidates) <= k:
            # Fewer candidates than drones — just take all of them
            selected = candidates[:k]
            solver_name = "trivial"
            energy = float("nan")
        else:
            # Step 2: build QUBO
            Q, w_cand = build_qubo(candidates, info_field, ensemble, k)

            # Step 3: solve
            selected_idx, solver_name, energy = solve_qubo(
                Q, w_cand, k, self.n_sa_reads
            )
            selected = [candidates[i] for i in selected_idx]

        gains = [float(info_field.w[r, c]) for r, c in selected]

        return SelectionResult(
            kind="points",
            selected_locations=selected,
            marginal_gains=gains,
            strategy_name=self.name,
            compute_time_s=time.perf_counter() - t0,
            solver_metadata={
                "solver": solver_name,
                "energy": energy,
                "n_candidates": len(candidates),
            },
        )
