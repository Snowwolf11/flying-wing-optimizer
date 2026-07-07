"""Hierarchical (coarse-to-fine) search: the default optimizer.

The spec calls for: coarse grid -> evaluate all -> retain best N -> refine
around each -> repeat until desired resolution. A literal full-factorial
grid is exponential in the number of dimensions (Stage 1 alone has ~15 --
3 airfoil parameters x 5 span stations -- so even 3 points/dim would be
3^15 = 14 million evaluations). "Grid" here is instead a Latin Hypercube
space-filling sample at each stage/scale: still deterministic (fixed seed),
still coarse-to-fine with elitist retention, still embarrassingly parallel,
but tractable at any dimensionality.

This is one `Optimizer` implementation (see base.py) -- Stage 1/2/multi-
cycle drivers depend only on that interface, so CMA-ES / Bayesian
optimization / differential evolution / particle swarm can replace this
later without touching them.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import qmc

from .base import Optimizer, ObjectiveFn, EvaluatedCandidate, OptimizationResult


@dataclass
class HierarchicalGridSearch(Optimizer):
    n_stages: int = 4
    n_samples_per_stage: int = 40
    retain_best_n: int = 5
    shrink_factor: float = 0.4  # each stage's local search range = previous stage's range * shrink_factor
    seed: int = 0
    n_jobs: int = 1  # >1 evaluates each stage's candidates in parallel worker processes

    def optimize(
        self, objective_fn: ObjectiveFn, bounds: list[tuple[float, float]], x0: np.ndarray | None = None,
    ) -> OptimizationResult:
        bounds_arr = np.array(bounds, dtype=float)
        lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
        n_dims = len(bounds)

        history: list[list[EvaluatedCandidate]] = []

        coarse_unit = qmc.LatinHypercube(d=n_dims, seed=self.seed).random(n=self.n_samples_per_stage)
        candidates = qmc.scale(coarse_unit, lo, hi)
        if x0 is not None:
            candidates = np.vstack([np.asarray(x0, dtype=float)[None, :], candidates])

        evaluated = self._evaluate_batch(objective_fn, candidates)
        history.append(evaluated)
        retained = self._retain_best(evaluated)

        search_range = hi - lo
        for stage in range(1, self.n_stages):
            search_range = search_range * self.shrink_factor
            n_per_elite = max(1, self.n_samples_per_stage // max(1, len(retained)))

            stage_candidates = []
            for i, elite in enumerate(retained):
                local_lo = np.clip(elite.x - search_range / 2.0, lo, hi)
                local_hi = np.clip(elite.x + search_range / 2.0, lo, hi)
                local_unit = qmc.LatinHypercube(d=n_dims, seed=self.seed + stage * 1000 + i).random(n=n_per_elite)
                stage_candidates.append(qmc.scale(local_unit, local_lo, local_hi))
            stage_candidates = np.vstack(stage_candidates)

            evaluated = self._evaluate_batch(objective_fn, stage_candidates)
            history.append(evaluated)
            retained = self._retain_best(retained + evaluated)  # never lose the best-so-far

        best = max(retained, key=lambda c: c.score)
        return OptimizationResult(best_candidate=best, history=history)

    def _evaluate_batch(self, objective_fn: ObjectiveFn, candidates: np.ndarray) -> list[EvaluatedCandidate]:
        if self.n_jobs <= 1:
            return [objective_fn(x) for x in candidates]
        with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
            return list(ex.map(objective_fn, candidates))

    def _retain_best(self, candidates: list[EvaluatedCandidate]) -> list[EvaluatedCandidate]:
        return sorted(candidates, key=lambda c: c.score, reverse=True)[: self.retain_best_n]
