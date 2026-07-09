"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy): the default
optimizer, chosen over the original hierarchical.py Latin Hypercube search
(kept as a selectable alternative) because it fits this problem's actual
shape well:

  - Moderate, continuous dimensionality (Stage 1 ~15, Stage 2 ~30) -- squarely
    in CMA-ES's well-evidenced sweet spot (roughly 2 to a few hundred dims).
  - No gradient available through AeroSandbox/NeuralFoil -- needs a
    derivative-free method.
  - Non-separable: Stage 2's own parameterization builds chord/twist/LE/Z
    offset as a root value plus *cumulative* per-segment deltas/slopes (see
    stage2.py), so one variable's change cascades through every downstream
    station by construction -- and the structural safety-factor term is a
    cumulative tip-to-root bending-moment integral, coupling every station's
    chord/twist together. Methods that treat dimensions independently (e.g.
    coordinate-wise search, or the previous method's axis-aligned isotropic
    shrink) fight this structure; CMA-ES's adapted covariance matrix can
    represent it.
  - Rugged/ill-conditioned: hard constraint penalties (objective/objective.py's
    invalid_penalty, squared-threshold terms) create sharp value changes at
    constraint boundaries -- exactly the kind of landscape CMA-ES is reported
    to handle better than local/gradient-based methods.

CMA-ES itself is fundamentally a *local* refinement method once it commits to
one basin, so multi-modality (this is an aircraft design landscape; expect
multiple distinct local optima -- e.g. different sweep/twist combinations
reaching similar L/D) is handled here via IPOP-style restarts: each restart
after the first starts from a fresh random point with a doubled population
size, and the best candidate across all restarts is kept.

Runs entirely in a [0, 1]^n normalized coordinate space, not the raw
(physically-unit'd, wildly different-scale -- chord in meters vs. sweep in
degrees vs. slope in 1/span) bounds -- CMA-ES's covariance adaptation would
eventually learn the true per-dimension scale on its own, but starting from a
uniform initial step size across axes that differ by orders of magnitude
wastes early generations relearning something already known from the bounds.
Candidates are mapped back to real units only for `objective_fn` calls, so
this is entirely internal -- the Optimizer contract (bounds/x0/best_x all in
real units) is unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import cma

from .base import Optimizer, ObjectiveFn, EvaluatedCandidate, OptimizationResult, evaluate_batch

ProgressCallback = Callable[[dict], None]


def _normalize(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    scale = np.where(hi > lo, hi - lo, 1.0)
    return (x - lo) / scale


def _denormalize(x_norm: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    # hi - lo (not the zero-width-safe `scale` above) is deliberate here: a
    # degenerate (hi == lo) dimension collapses any normalized value back to
    # exactly `lo`, its one valid value, regardless of what CMA-ES proposed.
    return lo + np.clip(x_norm, 0.0, 1.0) * (hi - lo)


@dataclass
class CMAESOptimizer(Optimizer):
    sigma0: float = 0.25  # initial step size, as a fraction of the normalized [0, 1] domain width (the standard ~1/4-of-domain recommendation)
    population_size: int | None = None  # None = pycma's own default heuristic (4 + floor(3*ln(n))); set explicitly to make full use of many n_jobs workers per generation
    max_generations: int = 100  # outer cap; CMA-ES's own convergence criteria (tolfun/tolx) usually stop it sooner
    n_restarts: int = 2  # IPOP-style: each restart after the first doubles the population and starts from a fresh random point, guarding against getting stuck in one basin on this multi-modal landscape
    seed: int = 0
    n_jobs: int = 1  # >1 evaluates each generation's candidates in parallel worker processes

    def optimize(
        self, objective_fn: ObjectiveFn, bounds: list[tuple[float, float]], x0: np.ndarray | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> OptimizationResult:
        bounds_arr = np.array(bounds, dtype=float)
        lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
        n_dims = len(bounds)

        n_restarts = max(1, self.n_restarts)
        base_popsize = self.population_size  # None until the first es tells us its own default
        evals_total_estimate = 0  # filled in incrementally as each restart's actual popsize becomes known
        evals_done = 0
        history: list[list[EvaluatedCandidate]] = []
        best_overall: EvaluatedCandidate | None = None

        rng = np.random.default_rng(self.seed)

        for restart in range(n_restarts):
            if restart == 0 and x0 is not None:
                x0_norm = _normalize(np.clip(np.asarray(x0, dtype=float), lo, hi), lo, hi)
            else:
                # IPOP restarts begin from a fresh random point, not x0 --
                # reusing the same start would just re-explore the same basin.
                x0_norm = rng.uniform(0.0, 1.0, size=n_dims)

            popsize = None if base_popsize is None else base_popsize * (2 ** restart)
            options = {
                "bounds": [0.0, 1.0],
                "seed": int(rng.integers(1, 2**31 - 1)),
                "maxiter": self.max_generations,
                "verbose": -9,  # silence pycma's own stdout logging
            }
            if popsize is not None:
                options["popsize"] = popsize

            es = cma.CMAEvolutionStrategy(x0_norm, self.sigma0, options)
            if restart == 0:
                if base_popsize is None:
                    base_popsize = es.popsize  # lock in pycma's auto-chosen popsize so restarts double *that*, not re-derive it each time
                # Now that restart 0's popsize is known, every later restart's
                # is deterministic (base_popsize * 2**r) -- compute the whole
                # multi-restart total upfront so the progress bar's percentage
                # doesn't jump backwards when a new restart's budget is added.
                evals_total_estimate = sum(base_popsize * (2 ** r) * self.max_generations for r in range(n_restarts))

            generation = 0
            while not es.stop() and generation < self.max_generations:
                candidates_norm = es.ask()
                if generation == 0 and restart == 0 and x0 is not None:
                    # Elitist injection: es.ask() draws samples *around* the
                    # seed mean (x0_norm), essentially never x0 itself, so
                    # without this, best_overall's very first value could be
                    # worse than what the caller started from -- e.g.
                    # multi-cycle's Stage1<->Stage2 driver feeds each stage's
                    # best design forward as the next stage's x0 (cycle.py);
                    # a real run showed the reported best score visibly *drop*
                    # right after every stage switch. Guaranteeing x0 itself
                    # gets evaluated in generation 0 means best_overall (and
                    # therefore result.best_score) can never come out below
                    # the seed's own score.
                    candidates_norm[0] = x0_norm
                candidates_real = np.array([_denormalize(c, lo, hi) for c in candidates_norm])
                evaluated = evaluate_batch(objective_fn, candidates_real, self.n_jobs)
                es.tell(candidates_norm, [-c.score for c in evaluated])  # pycma minimizes; our score is maximize-better

                history.append(evaluated)
                evals_done += len(evaluated)
                generation += 1

                gen_best = max(evaluated, key=lambda c: c.score)
                if best_overall is None or gen_best.score > best_overall.score:
                    best_overall = gen_best

                if progress_cb is not None:
                    # "stage"/"n_stages" reused as generation/max_generations
                    # (an upper bound -- es.stop() usually ends a run sooner)
                    # so the GUI's existing "Stage X/Y" label needs no
                    # CMA-ES-specific handling; "restart"/"n_restarts" let it
                    # additionally show which IPOP restart is running.
                    progress_cb({
                        "stage": generation, "n_stages": self.max_generations,
                        "restart": restart + 1, "n_restarts": n_restarts,
                        "evals_done": evals_done, "evals_total": evals_total_estimate,
                        "best_score": best_overall.score,
                    })

        return OptimizationResult(best_candidate=best_overall, history=history)
