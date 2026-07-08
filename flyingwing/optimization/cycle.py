"""Multi-cycle driver: alternate Stage 1 (airfoil schedule) and Stage 2
(planform) optimization, each stage starting from wherever the previous one
left off, for a configurable number of cycles -- optionally stopping early
once the improvement per cycle falls below a tolerance ("until convergence").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..geometry.params import DesignParameters
from ..objective.objective import ObjectiveWeights, NormalizationConstants
from .base import OptimizationResult
from .hierarchical import HierarchicalGridSearch, ProgressCallback
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
    stage1_optimizer = stage1_optimizer or HierarchicalGridSearch()
    stage2_optimizer = stage2_optimizer or HierarchicalGridSearch()

    stage_order = ["stage1", "stage2"] if start_with == "stage1" else ["stage2", "stage1"]

    records: list[CycleRecord] = []
    current_params = initial_params
    previous_cycle_score: float | None = None

    for cycle in range(n_cycles):
        for stage_name in stage_order:
            stage_progress_cb = None
            if progress_cb is not None:
                stage_progress_cb = lambda info, cycle=cycle, stage_name=stage_name: progress_cb(
                    {**info, "cycle": cycle, "n_cycles": n_cycles, "stage_name": stage_name}
                )
            if stage_name == "stage1":
                result, current_params = run_stage1(current_params, weights=weights, normalization=normalization, optimizer=stage1_optimizer, progress_cb=stage_progress_cb)
            else:
                result, current_params = run_stage2(current_params, weights=weights, normalization=normalization, optimizer=stage2_optimizer, progress_cb=stage_progress_cb)
            records.append(CycleRecord(cycle=cycle, stage=stage_name, result=result, params=current_params))

        cycle_end_score = records[-1].result.best_score
        if convergence_tol is not None and previous_cycle_score is not None:
            if (cycle_end_score - previous_cycle_score) < convergence_tol:
                break
        previous_cycle_score = cycle_end_score

    return MultiCycleResult(records=records)
