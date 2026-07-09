"""Hierarchical (coarse-to-fine) search: the original optimizer, kept as a
selectable alternative to CMA-ES (cmaes.py, the default -- see its module
docstring for why).

The spec calls for: coarse grid -> evaluate all -> retain best N -> refine
around each -> repeat until desired resolution. A literal full-factorial
grid is exponential in the number of dimensions (Stage 1 alone has ~15 --
3 airfoil parameters x 5 span stations -- so even 3 points/dim would be
3^15 = 14 million evaluations). "Grid" here is instead a Latin Hypercube
space-filling sample at each stage/scale: still deterministic (fixed seed),
still coarse-to-fine with elitist retention, still embarrassingly parallel,
but tractable at any dimensionality. Its main weakness relative to CMA-ES:
the shrink is isotropic and axis-aligned, so it can't learn correlated
directions between parameters (e.g. "chord and twist should move together
here") the way CMA-ES's adapted covariance matrix does.

This is one `Optimizer` implementation (see base.py) -- Stage 1/2/multi-
cycle drivers depend only on that interface, so any algorithm can be
selected without touching them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import qmc

from .base import Optimizer, ObjectiveFn, EvaluatedCandidate, OptimizationResult, evaluate_batch

ProgressCallback = Callable[[dict], None]


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
        progress_cb: ProgressCallback | None = None,
    ) -> OptimizationResult:
        bounds_arr = np.array(bounds, dtype=float)
        lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
        n_dims = len(bounds)

        # Rough estimate only (the exact per-stage count can drift slightly
        # from integer division across retained elites) -- good enough for a
        # progress bar, not used for anything else.
        evals_total_estimate = self.n_samples_per_stage * self.n_stages + (1 if x0 is not None else 0)
        evals_done = 0

        history: list[list[EvaluatedCandidate]] = []

        coarse_unit = qmc.LatinHypercube(d=n_dims, seed=self.seed).random(n=self.n_samples_per_stage)
        candidates = qmc.scale(coarse_unit, lo, hi)
        if x0 is not None:
            # Clip: x0 (typically a ParameterSet's baseline-derived default)
            # can legitimately fall outside the current search bounds -- e.g.
            # someone tightened a bound below the baseline design's actual
            # value. Left unclipped, an out-of-bounds x0 that becomes an
            # elite can collapse a later refinement stage's local bounds to
            # zero width (clip(x0 +/- range/2, lo, hi) saturates both ends to
            # the same edge), which crashes qmc.scale.
            x0_clipped = np.clip(np.asarray(x0, dtype=float), lo, hi)
            candidates = np.vstack([x0_clipped[None, :], candidates])

        evaluated = evaluate_batch(objective_fn, candidates, self.n_jobs)
        history.append(evaluated)
        retained = self._retain_best(evaluated)
        evals_done += len(evaluated)
        if progress_cb is not None:
            progress_cb({
                "stage": 1, "n_stages": self.n_stages,
                "evals_done": evals_done, "evals_total": evals_total_estimate,
                "best_score": retained[0].score,
            })

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

            evaluated = evaluate_batch(objective_fn, stage_candidates, self.n_jobs)
            history.append(evaluated)
            retained = self._retain_best(retained + evaluated)  # never lose the best-so-far
            evals_done += len(evaluated)
            if progress_cb is not None:
                progress_cb({
                    "stage": stage + 1, "n_stages": self.n_stages,
                    "evals_done": evals_done, "evals_total": evals_total_estimate,
                    "best_score": retained[0].score,
                })

        best = max(retained, key=lambda c: c.score)
        return OptimizationResult(best_candidate=best, history=history)

    def _retain_best(self, candidates: list[EvaluatedCandidate]) -> list[EvaluatedCandidate]:
        return sorted(candidates, key=lambda c: c.score, reverse=True)[: self.retain_best_n]
