"""Multi-cycle driver: alternate Stage 1 (airfoil schedule) and Stage 2
(planform) optimization, each stage starting from wherever the previous one
left off, for a configurable number of cycles -- optionally stopping early
once the improvement per cycle falls below a tolerance ("until convergence").
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..geometry.params import DesignParameters
from ..objective.objective import ObjectiveWeights, NormalizationConstants
from .base import OptimizationResult
from .hierarchical import HierarchicalGridSearch, ProgressCallback
from .cmaes import CMAESOptimizer
from .stage1 import run_stage1
from .stage2 import run_stage2


@dataclass
class CycleRecord:
    cycle: int
    stage: str  # "stage1" or "stage2"
    result: OptimizationResult
    params: DesignParameters


@dataclass
class MultiCycleResult:
    records: list[CycleRecord] = field(default_factory=list)

    @property
    def best_record(self) -> CycleRecord:
        return max(self.records, key=lambda r: r.result.best_score)

    @property
    def best_params(self) -> DesignParameters:
        return self.best_record.params

    @property
    def score_history(self) -> list[float]:
        """Best score after each stage, in run order."""
        return [r.result.best_score for r in self.records]


def run_multi_cycle(
    initial_params: DesignParameters,
    n_cycles: int = 2,
    weights: ObjectiveWeights | None = None,
    normalization: NormalizationConstants | None = None,
    stage1_optimizer: HierarchicalGridSearch | None = None,
    stage2_optimizer: HierarchicalGridSearch | None = None,
    start_with: str = "stage1",
    convergence_tol: float | None = None,
    progress_cb: ProgressCallback | None = None,
) -> MultiCycleResult:
    """Run `n_cycles` Stage1<->Stage2 cycles (2 stages per cycle). If
    `convergence_tol` is set, stop early once a full cycle's best score
    improves by less than that amount over the previous cycle's.
    """
    weights = weights or ObjectiveWeights()
    normalization = normalization or NormalizationConstants()
    stage1_optimizer = stage1_optimizer or CMAESOptimizer()
    stage2_optimizer = stage2_optimizer or CMAESOptimizer()

    stage_order = ["stage1", "stage2"] if start_with == "stage1" else ["stage2", "stage1"]

    records: list[CycleRecord] = []
    current_params = initial_params
    best_params_so_far = initial_params
    best_score_so_far: float | None = None
    previous_cycle_score: float | None = None

    for cycle in range(n_cycles):
        for stage_name in stage_order:
            stage_progress_cb = None
            if progress_cb is not None:
                stage_progress_cb = lambda info, cycle=cycle, stage_name=stage_name: progress_cb(
                    {**info, "cycle": cycle, "n_cycles": n_cycles, "stage_name": stage_name}
                )
            # A fresh seed per cycle, not the same one every time -- with the
            # "always continue from the true best" rule below, two
            # consecutive cycles can easily feed the *same* x0 into the same
            # stage (e.g. the other stage made no improvement in between);
            # reusing one fixed seed would then replay an identical,
            # already-seen search instead of exploring anything new.
            if stage_name == "stage1":
                optimizer = replace(stage1_optimizer, seed=stage1_optimizer.seed + cycle) if hasattr(stage1_optimizer, "seed") else stage1_optimizer
                result, stage_params = run_stage1(current_params, weights=weights, normalization=normalization, optimizer=optimizer, progress_cb=stage_progress_cb)
            else:
                optimizer = replace(stage2_optimizer, seed=stage2_optimizer.seed + cycle) if hasattr(stage2_optimizer, "seed") else stage2_optimizer
                result, stage_params = run_stage2(current_params, weights=weights, normalization=normalization, optimizer=optimizer, progress_cb=stage_progress_cb)
            records.append(CycleRecord(cycle=cycle, stage=stage_name, result=result, params=stage_params))

            # Always continue from the best design found *anywhere* so far,
            # not just whatever this stage's own optimizer returned -- a
            # stage's result can legitimately score worse than what it
            # started from (e.g. a baseline value that falls outside a
            # since-tightened bounds_overrides.yaml range gets silently
            # reprojected onto a different point when re-encoded as that
            # stage's search-space default, before the optimizer even runs;
            # cmaes.py's x0-injection only guarantees *that* possibly-altered
            # point gets evaluated, not that it matches the true previous
            # best). Never letting current_params regress means a stage that
            # fails to improve just wastes its own budget instead of
            # corrupting every later stage.
            if best_score_so_far is None or result.best_score > best_score_so_far:
                best_score_so_far = result.best_score
                best_params_so_far = stage_params
            current_params = best_params_so_far

        if convergence_tol is not None and previous_cycle_score is not None:
            if (best_score_so_far - previous_cycle_score) < convergence_tol:
                break
        previous_cycle_score = best_score_so_far

    return MultiCycleResult(records=records)
