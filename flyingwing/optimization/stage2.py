"""Stage 2 driver: optimize the planform (global span/sweep + spanwise
chord/twist/LE-offset-deviation/z-offset), airfoil schedule held fixed at
the baseline (typically wherever Stage 1 converged to).

Chord and twist are parameterized as a root value plus non-negative
per-segment deltas (chord decreasing, twist decreasing i.e. washout
increasing), and LE offset deviation / Z offset as a root value plus a
per-segment slope, rather than as independent per-station values -- see
config.py's comment on CHORD_STATION_M_BOUNDS for why: with independent
per-station bounds, a random sample has almost no chance of landing on a
monotonic sequence by chance, which was making the hierarchical search
waste nearly every candidate on a constraint violation instead of
exploring genuinely different designs. This reparameterization makes
monotonicity (chord/twist) and bounded curvature (LE/Z offset) hold by
construction instead.

The *bounds* on this parameterization, however, are specified per-station
in directly-interpretable absolute units (config.py's CHORD_STATION_M_BOUNDS
etc.) -- make_stage2_parameter_set derives each segment's decrement/slope
Var bounds from the two stations it connects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.params import DesignParameters
from ..objective.metrics import evaluate_design
from ..objective.objective import ObjectiveWeights, NormalizationConstants, score
from .base import EvaluatedCandidate, OptimizationResult
from .vector import ParameterSet, Var, resolve_per_station_bounds
from .hierarchical import HierarchicalGridSearch, ProgressCallback
from ..config import (
    WINGSPAN_MIN_M, WINGSPAN_MAX_M, SWEEP_DEG_BOUNDS,
    CHORD_STATION_M_BOUNDS, TWIST_STATION_DEG_BOUNDS, LE_OFFSET_ROOT_M_BOUNDS, Z_OFFSET_ROOT_M_BOUNDS,
    LE_OFFSET_SLOPE_M_PER_SPAN_BOUNDS, Z_OFFSET_SLOPE_M_PER_SPAN_BOUNDS,
)


def _station_bounds_arrays(y_control: np.ndarray) -> dict[str, np.ndarray]:
    """Resolve each per-station bound config value to (lo, hi) arrays of
    length n -- chord/twist only (LE/Z offset's root is a flat (lo, hi) pair,
    not per-station; their slope bounds are per-segment, resolved
    separately in make_stage2_parameter_set). Shared between
    make_stage2_parameter_set and _stage2_build so both always agree on the
    same bounds without passing them explicitly through the BuildFn
    signature (which optimization/vector.ParameterSet fixes as
    (x, baseline) -> DesignParameters)."""
    n = len(y_control)
    chord = resolve_per_station_bounds(CHORD_STATION_M_BOUNDS, n)
    twist = resolve_per_station_bounds(TWIST_STATION_DEG_BOUNDS, n)
    return {
        "chord_lo": np.array([b[0] for b in chord]), "chord_hi": np.array([b[1] for b in chord]),
        "twist_lo": np.array([b[0] for b in twist]), "twist_hi": np.array([b[1] for b in twist]),
    }


def _cumulative_decrease(root: float, deltas: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """[root, root-deltas[0], root-deltas[0]-deltas[1], ...], floored --
    monotonically non-increasing by construction since deltas >= 0. `floor`
    may be a scalar or a per-station array."""
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
    bounds = _station_bounds_arrays(y_control)
    span_m, sweep_deg = float(x[0]), float(x[1])
    i = 2

    chord_root = x[i]; i += 1
    chord_deltas = x[i : i + n - 1]; i += n - 1
    chord_m = tuple(_cumulative_decrease(chord_root, chord_deltas, bounds["chord_lo"]))

    twist_root = x[i]; i += 1
    washout_deltas = x[i : i + n - 1]; i += n - 1
    twist_deg = tuple(_cumulative_decrease(twist_root, washout_deltas, bounds["twist_lo"]))

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
    b = _station_bounds_arrays(y_control)

    # Segment i's decrement Var bounds, derived from the two stations it
    # connects -- e.g. for chord, the largest possible decrement is the drop
    # from station i's ceiling to station i+1's floor.
    chord_decrement_bounds = [(0.0, max(0.0, b["chord_hi"][i] - b["chord_lo"][i + 1])) for i in range(n - 1)]
    washout_increment_bounds = [(0.0, max(0.0, b["twist_hi"][i] - b["twist_lo"][i + 1])) for i in range(n - 1)]
    # LE/Z offset's slope bounds are set directly per segment, not derived
    # from a per-station absolute range -- see config.py's
    # LE_OFFSET_SLOPE_M_PER_SPAN_BOUNDS/Z_OFFSET_SLOPE_M_PER_SPAN_BOUNDS.
    le_offset_slope_bounds = resolve_per_station_bounds(LE_OFFSET_SLOPE_M_PER_SPAN_BOUNDS, n - 1)
    z_offset_slope_bounds = resolve_per_station_bounds(Z_OFFSET_SLOPE_M_PER_SPAN_BOUNDS, n - 1)

    chord_deltas_default = -np.diff(planform.chord_m)
    twist_deltas_default = -np.diff(planform.twist_deg)
    le_slopes_default = np.diff(planform.le_offset_deviation_m) / dy
    z_slopes_default = np.diff(planform.z_offset_m) / dy

    variables = [
        Var("span_m", WINGSPAN_MIN_M, WINGSPAN_MAX_M, default=planform.span_m),
        Var("sweep_deg", *SWEEP_DEG_BOUNDS, default=planform.sweep_deg),
        Var("chord_root_m", float(b["chord_lo"][0]), float(b["chord_hi"][0]), default=planform.chord_m[0]),
    ]
    for i in range(n - 1):
        lo, hi = chord_decrement_bounds[i]
        variables.append(Var(f"chord_decrement_{i}", lo, hi, default=float(np.clip(chord_deltas_default[i], lo, hi))))
    variables.append(Var("twist_root_deg", float(b["twist_lo"][0]), float(b["twist_hi"][0]), default=planform.twist_deg[0]))
    for i in range(n - 1):
        lo, hi = washout_increment_bounds[i]
        variables.append(Var(f"washout_increment_{i}", lo, hi, default=float(np.clip(twist_deltas_default[i], lo, hi))))
    variables.append(Var("le_offset_deviation_root_m", *LE_OFFSET_ROOT_M_BOUNDS, default=planform.le_offset_deviation_m[0]))
    for i in range(n - 1):
        lo, hi = le_offset_slope_bounds[i]
        variables.append(Var(f"le_offset_slope_{i}", lo, hi, default=float(np.clip(le_slopes_default[i], lo, hi))))
    variables.append(Var("z_offset_root_m", *Z_OFFSET_ROOT_M_BOUNDS, default=planform.z_offset_m[0]))
    for i in range(n - 1):
        lo, hi = z_offset_slope_bounds[i]
        variables.append(Var(f"z_offset_slope_{i}", lo, hi, default=float(np.clip(z_slopes_default[i], lo, hi))))

    return ParameterSet(variables=variables, build_fn=_stage2_build, baseline=baseline)


@dataclass
class Stage2Objective:
    """vector -> EvaluatedCandidate. A real (picklable) class, not a
    closure -- see Stage1Objective in stage1.py for why."""

    parameter_set: ParameterSet
    weights: ObjectiveWeights
    normalization: NormalizationConstants = field(default_factory=NormalizationConstants)

    def __call__(self, x: np.ndarray) -> EvaluatedCandidate:
        params = self.parameter_set.build(x)
        metrics = evaluate_design(params)
        result = score(metrics, self.weights, self.normalization)
        return EvaluatedCandidate(
            x=np.asarray(x, dtype=float), score=result.score, valid=metrics.valid,
            extra={"contributions": result.contributions, "metrics": metrics},
        )


def run_stage2(
    baseline: DesignParameters,
    weights: ObjectiveWeights | None = None,
    normalization: NormalizationConstants | None = None,
    optimizer: HierarchicalGridSearch | None = None,
    progress_cb: ProgressCallback | None = None,
) -> tuple[OptimizationResult, DesignParameters]:
    """Optimize the planform; returns the optimization result plus the
    resulting full DesignParameters (airfoil schedule unchanged from baseline)."""
    weights = weights or ObjectiveWeights()
    normalization = normalization or NormalizationConstants()
    optimizer = optimizer or HierarchicalGridSearch()

    parameter_set = make_stage2_parameter_set(baseline)
    objective_fn = Stage2Objective(parameter_set, weights, normalization)

    result = optimizer.optimize(objective_fn, parameter_set.bounds, x0=parameter_set.default_vector, progress_cb=progress_cb)
    best_params = parameter_set.build(result.best_x)
    return result, best_params
