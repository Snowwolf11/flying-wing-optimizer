"""Optimizer interface.

Every algorithm -- the hierarchical grid search implemented now, and CMA-ES /
Bayesian optimization / differential evolution / particle swarm that could
replace or supplement it later -- implements this same interface. Stage 1 /
Stage 2 / multi-cycle drivers only ever talk to an `Optimizer`, never to a
specific algorithm, so swapping the algorithm doesn't touch the drivers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class EvaluatedCandidate:
    x: np.ndarray
    score: float
    valid: bool
    extra: dict = field(default_factory=dict)


@dataclass
class OptimizationResult:
    best_candidate: EvaluatedCandidate
    history: list[list[EvaluatedCandidate]]  # one list of evaluated candidates per stage/iteration -- for convergence plots

    @property
    def best_x(self) -> np.ndarray:
        return self.best_candidate.x

    @property
    def best_score(self) -> float:
        return self.best_candidate.score


ObjectiveFn = Callable[[np.ndarray], EvaluatedCandidate]


class Optimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        objective_fn: ObjectiveFn,
        bounds: list[tuple[float, float]],
        x0: np.ndarray | None = None,
    ) -> OptimizationResult:
        """Maximize `objective_fn` (higher EvaluatedCandidate.score is better) over the given bounds."""


def evaluate_batch(objective_fn: ObjectiveFn, candidates: np.ndarray, n_jobs: int = 1) -> list[EvaluatedCandidate]:
    """Evaluate a batch of candidate vectors, in parallel worker processes if
    n_jobs > 1. Shared by every batch-oriented Optimizer (hierarchical.py,
    cmaes.py) so they evaluate a generation/stage's candidates the same way."""
    if n_jobs <= 1:
        return [objective_fn(x) for x in candidates]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(objective_fn, candidates))
