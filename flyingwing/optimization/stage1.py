"""Stage 1 driver: optimize the airfoil schedule (thickness/camber/reflex
scale at each control station), planform held fixed at the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _overrides as _ov
from ..config import BOUNDS_OVERRIDES_YAML
from ..geometry.params import DesignParameters
from ..geometry.airfoil_family import MIN_THICKNESS_RATIO, MAX_THICKNESS_RATIO, max_thickness_ratio
from ..objective.metrics import evaluate_design
from ..objective.objective import ObjectiveWeights, score
from .base import EvaluatedCandidate, OptimizationResult
from .vector import ParameterSet, Var, resolve_per_station_bounds
from .hierarchical import HierarchicalGridSearch, ProgressCallback

_BASE_MAX_THICKNESS_RATIO = max_thickness_ratio(1.0)
THICKNESS_SCALE_BOUNDS = (MIN_THICKNESS_RATIO / _BASE_MAX_THICKNESS_RATIO, MAX_THICKNESS_RATIO / _BASE_MAX_THICKNESS_RATIO)
CAMBER_SCALE_BOUNDS = (0.0, 1.6)
REFLEX_SCALE_BOUNDS = (0.0, 1.6)

# Each of these may be overridden as either a single (lo, hi) pair (applied
# to every airfoil control station) or a list of n (lo, hi) pairs (one per
# station, n = len(baseline.airfoil_schedule.y_control)) -- see
# vector.resolve_per_station_bounds. Not in config.py because
# THICKNESS_SCALE_BOUNDS depends on airfoil_family.max_thickness_ratio,
# which itself imports config -- applying overrides locally here avoids that
# circular import.
_ov.apply_overrides(globals(), BOUNDS_OVERRIDES_YAML, {"THICKNESS_SCALE_BOUNDS", "CAMBER_SCALE_BOUNDS", "REFLEX_SCALE_BOUNDS"})


def _stage1_build(x: np.ndarray, baseline: DesignParameters) -> DesignParameters:
    n = len(baseline.airfoil_schedule.y_control)
    thickness = tuple(x[0:n])
    camber = tuple(x[n : 2 * n])
    reflex = tuple(x[2 * n : 3 * n])
    return baseline.with_airfoil_schedule(thickness_scale=thickness, camber_scale=camber, reflex_scale=reflex)


def make_stage1_parameter_set(baseline: DesignParameters) -> ParameterSet:
    n = len(baseline.airfoil_schedule.y_control)
    thickness_bounds = resolve_per_station_bounds(THICKNESS_SCALE_BOUNDS, n)
    camber_bounds = resolve_per_station_bounds(CAMBER_SCALE_BOUNDS, n)
    reflex_bounds = resolve_per_station_bounds(REFLEX_SCALE_BOUNDS, n)

    variables = []
    for i in range(n):
        variables.append(Var(f"thickness_scale_{i}", *thickness_bounds[i], default=baseline.airfoil_schedule.thickness_scale[i]))
    for i in range(n):
        variables.append(Var(f"camber_scale_{i}", *camber_bounds[i], default=baseline.airfoil_schedule.camber_scale[i]))
    for i in range(n):
        variables.append(Var(f"reflex_scale_{i}", *reflex_bounds[i], default=baseline.airfoil_schedule.reflex_scale[i]))

    return ParameterSet(variables=variables, build_fn=_stage1_build, baseline=baseline)


@dataclass
class Stage1Objective:
    """vector -> EvaluatedCandidate. A real (picklable) class rather than a
    closure, since ProcessPoolExecutor needs to pickle it under Windows'
    spawn start method when n_jobs > 1."""

    parameter_set: ParameterSet
    weights: ObjectiveWeights

    def __call__(self, x: np.ndarray) -> EvaluatedCandidate:
        params = self.parameter_set.build(x)
        metrics = evaluate_design(params)
        result = score(metrics, self.weights)
        return EvaluatedCandidate(
            x=np.asarray(x, dtype=float), score=result.score, valid=metrics.valid,
            extra={"contributions": result.contributions, "metrics": metrics},
        )


def run_stage1(
    baseline: DesignParameters,
    weights: ObjectiveWeights | None = None,
    optimizer: HierarchicalGridSearch | None = None,
    progress_cb: ProgressCallback | None = None,
) -> tuple[OptimizationResult, DesignParameters]:
    """Optimize the airfoil schedule; returns the optimization result plus
    the resulting full DesignParameters (planform unchanged from baseline)."""
    weights = weights or ObjectiveWeights()
    optimizer = optimizer or HierarchicalGridSearch()

    parameter_set = make_stage1_parameter_set(baseline)
    objective_fn = Stage1Objective(parameter_set, weights)

    result = optimizer.optimize(objective_fn, parameter_set.bounds, x0=parameter_set.default_vector, progress_cb=progress_cb)
    best_params = parameter_set.build(result.best_x)
    return result, best_params
