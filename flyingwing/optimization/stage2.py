"""Stage 2 driver: optimize the planform (global span/sweep + spanwise
chord/twist/LE-offset-deviation/z-offset), airfoil schedule held fixed at
the baseline (typically wherever Stage 1 converged to).

Chord and twist are parameterized as a root value plus non-negative
per-segment deltas (chord decreasing, twist decreasing i.e. washout
increasing), rather than as independent per-station values -- see
config.py's comment on CHORD_DECREMENT_M_BOUNDS for why: with independent
per-station bounds, a random sample has almost no chance of landing on a
monotonic sequence by chance, which was making the hierarchical search
waste nearly every candidate on a constraint violation instead of
exploring genuinely different designs. This reparameterization makes
monotonicity hold by construction instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.params import DesignParameters
from ..objective.metrics import evaluate_design
from ..objective.objective import ObjectiveWeights, score
from .base import EvaluatedCandidate, OptimizationResult
from .vector import ParameterSet, Var
from .hierarchical import HierarchicalGridSearch
from ..config import (
    WINGSPAN_MIN_M, WINGSPAN_MAX_M, SWEEP_DEG_BOUNDS, CHORD_M_BOUNDS, TWIST_DEG_BOUNDS,
    CHORD_DECREMENT_M_BOUNDS, CHORD_ROOT_M_BOUNDS, TWIST_ROOT_DEG_BOUNDS, WASHOUT_INCREMENT_DEG_BOUNDS,
    LE_OFFSET_DEVIATION_M_BOUNDS, Z_OFFSET_M_BOUNDS, LE_OFFSET_SLOPE_BOUNDS, Z_OFFSET_SLOPE_BOUNDS,
)


def _cumulative_decrease(root: float, deltas: np.ndarray, floor: float) -> np.ndarray:
    """[root, root-deltas[0], root-deltas[0]-deltas[1], ...], floored --
    monotonically non-increasing by construction since deltas >= 0."""
    values = root - np.cumsum(np.concatenate([[0.0], deltas]))
    return np.maximum(values, floor)


def _slope_chain(root: float, slopes: np.ndarray, y_control: np.ndarray) -> np.ndarray:
    """[root, root + slopes[0]*dy[0], ...] -- a free-form (non-monotonic)
    curve built from a bounded slope per segment rather than an independent
    value per station, so a random sample can't put a huge jump across a
    tiny span-fraction gap (which is what was blowing up LE curvature)."""
    dy = np.diff(y_control)
    return root + np.concatenate([[0.0], np.cumsum(slopes * dy)])


def _stage2_build(x: np.ndarray, baseline: DesignParameters) -> DesignParameters:
    y_control = np.asarray(baseline.planform.y_control, dtype=float)
    n = len(y_control)
    span_m, sweep_deg = float(x[0]), float(x[1])
    i = 2

    chord_root = x[i]; i += 1
    chord_deltas = x[i : i + n - 1]; i += n - 1
    chord_m = tuple(_cumulative_decrease(chord_root, chord_deltas, CHORD_M_BOUNDS[0]))

    twist_root = x[i]; i += 1
    washout_deltas = x[i : i + n - 1]; i += n - 1
    twist_deg = tuple(_cumulative_decrease(twist_root, washout_deltas, TWIST_DEG_BOUNDS[0]))

    le_root = x[i]; i += 1
    le_slopes = x[i : i + n - 1]; i += n - 1
    le_offset_deviation_m = tuple(_slope_chain(le_root, le_slopes, y_control))

    z_root = x[i]; i += 1
    z_slopes = x[i : i + n - 1]; i += n - 1
    z_offset_m = tuple(_slope_chain(z_root, z_slopes, y_control))

    return baseline.with_planform(
        span_m=span_m, sweep_deg=sweep_deg,
        chord_m=chord_m, twist_deg=twist_deg,
        le_offset_deviation_m=le_offset_deviation_m, z_offset_m=z_offset_m,
    )


def make_stage2_parameter_set(baseline: DesignParameters) -> ParameterSet:
    y_control = np.asarray(baseline.planform.y_control, dtype=float)
    n = len(y_control)
    dy = np.diff(y_control)
    planform = baseline.planform

    chord_deltas_default = -np.diff(planform.chord_m)
    twist_deltas_default = -np.diff(planform.twist_deg)
    le_slopes_default = np.diff(planform.le_offset_deviation_m) / dy
    z_slopes_default = np.diff(planform.z_offset_m) / dy

    variables = [
        Var("span_m", WINGSPAN_MIN_M, WINGSPAN_MAX_M, default=planform.span_m),
        Var("sweep_deg", *SWEEP_DEG_BOUNDS, default=planform.sweep_deg),
        Var("chord_root_m", *CHORD_ROOT_M_BOUNDS, default=planform.chord_m[0]),
    ]
    for i in range(n - 1):
        variables.append(Var(f"chord_decrement_{i}", *CHORD_DECREMENT_M_BOUNDS, default=max(chord_deltas_default[i], 0.0)))
    variables.append(Var("twist_root_deg", *TWIST_ROOT_DEG_BOUNDS, default=planform.twist_deg[0]))
    for i in range(n - 1):
        variables.append(Var(f"washout_increment_{i}", *WASHOUT_INCREMENT_DEG_BOUNDS, default=max(twist_deltas_default[i], 0.0)))
    variables.append(Var("le_offset_deviation_root_m", *LE_OFFSET_DEVIATION_M_BOUNDS, default=planform.le_offset_deviation_m[0]))
    for i in range(n - 1):
        variables.append(Var(f"le_offset_slope_{i}", *LE_OFFSET_SLOPE_BOUNDS, default=float(le_slopes_default[i])))
    variables.append(Var("z_offset_root_m", *Z_OFFSET_M_BOUNDS, default=planform.z_offset_m[0]))
    for i in range(n - 1):
        variables.append(Var(f"z_offset_slope_{i}", *Z_OFFSET_SLOPE_BOUNDS, default=float(z_slopes_default[i])))

    return ParameterSet(variables=variables, build_fn=_stage2_build, baseline=baseline)


@dataclass
class Stage2Objective:
    """vector -> EvaluatedCandidate. A real (picklable) class, not a
    closure -- see Stage1Objective in stage1.py for why."""

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


def run_stage2(
    baseline: DesignParameters,
    weights: ObjectiveWeights | None = None,
    optimizer: HierarchicalGridSearch | None = None,
) -> tuple[OptimizationResult, DesignParameters]:
    """Optimize the planform; returns the optimization result plus the
    resulting full DesignParameters (airfoil schedule unchanged from baseline)."""
    weights = weights or ObjectiveWeights()
    optimizer = optimizer or HierarchicalGridSearch()

    parameter_set = make_stage2_parameter_set(baseline)
    objective_fn = Stage2Objective(parameter_set, weights)

    result = optimizer.optimize(objective_fn, parameter_set.bounds, x0=parameter_set.default_vector)
    best_params = parameter_set.build(result.best_x)
    return result, best_params
