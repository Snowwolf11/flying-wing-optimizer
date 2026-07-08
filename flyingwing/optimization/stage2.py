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

from dataclasses import dataclass

import numpy as np

from ..geometry.params import DesignParameters
from ..objective.metrics import evaluate_design
from ..objective.objective import ObjectiveWeights, score
from .base import EvaluatedCandidate, OptimizationResult
from .vector import ParameterSet, Var, resolve_per_station_bounds
from .hierarchical import HierarchicalGridSearch, ProgressCallback
from ..config import (
    WINGSPAN_MIN_M, WINGSPAN_MAX_M, SWEEP_DEG_BOUNDS,
    CHORD_STATION_M_BOUNDS, TWIST_STATION_DEG_BOUNDS, LE_OFFSET_STATION_M_BOUNDS, Z_OFFSET_STATION_M_BOUNDS,
    Z_OFFSET_TIP_MIN_M, Z_OFFSET_TIP_SEGMENT_Y_THRESHOLD,
    MAX_LE_OFFSET_SLOPE_M_PER_SPAN, MAX_Z_OFFSET_SLOPE_M_PER_SPAN,
)


def _station_bounds_arrays(y_control: np.ndarray) -> dict[str, np.ndarray]:
    """Resolve each per-station bound config value to (lo, hi) arrays of
    length n. Shared between make_stage2_parameter_set and _stage2_build so
    both always agree on the same bounds without passing them explicitly
    through the BuildFn signature (which optimization/vector.ParameterSet
    fixes as (x, baseline) -> DesignParameters)."""
    n = len(y_control)
    chord = resolve_per_station_bounds(CHORD_STATION_M_BOUNDS, n)
    twist = resolve_per_station_bounds(TWIST_STATION_DEG_BOUNDS, n)
    le = resolve_per_station_bounds(LE_OFFSET_STATION_M_BOUNDS, n)
    z = resolve_per_station_bounds(Z_OFFSET_STATION_M_BOUNDS, n)
    return {
        "chord_lo": np.array([b[0] for b in chord]), "chord_hi": np.array([b[1] for b in chord]),
        "twist_lo": np.array([b[0] for b in twist]), "twist_hi": np.array([b[1] for b in twist]),
        "le_lo": np.array([b[0] for b in le]), "le_hi": np.array([b[1] for b in le]),
        "z_lo": np.array([b[0] for b in z]), "z_hi": np.array([b[1] for b in z]),
    }


def _clip_slope(raw_lo: float, raw_hi: float, cap: float) -> tuple[float, float]:
    """Clip (raw_lo, raw_hi) to [-cap, cap], collapsing to a single point at
    the cap boundary instead of producing an inverted (lo > hi) bound if
    raw_lo/raw_hi both fall on the same side, beyond the cap."""
    lo, hi = max(raw_lo, -cap), min(raw_hi, cap)
    if lo > hi:
        pinned = cap if raw_lo > 0 else -cap
        return pinned, pinned
    return lo, hi


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
    # The tip station's Z offset lower bound is floored (winglet bias) --
    # see config.py's Z_OFFSET_TIP_MIN_M comment.
    b["z_lo"][-1] = max(b["z_lo"][-1], Z_OFFSET_TIP_MIN_M)
    b["z_hi"][-1] = max(b["z_hi"][-1], b["z_lo"][-1])

    # Segment i's decrement/slope Var bounds, derived from the two stations
    # it connects -- e.g. for chord, the largest possible decrement is the
    # drop from station i's ceiling to station i+1's floor.
    chord_decrement_bounds = [(0.0, max(0.0, b["chord_hi"][i] - b["chord_lo"][i + 1])) for i in range(n - 1)]
    washout_increment_bounds = [(0.0, max(0.0, b["twist_hi"][i] - b["twist_lo"][i + 1])) for i in range(n - 1)]
    # Slope bounds are capped at MAX_*_OFFSET_SLOPE_M_PER_SPAN regardless of
    # station spacing -- deriving purely from (station bound range)/dy
    # blows up for closely-spaced stations (e.g. the 0.08/0.12/0.14
    # fuselage-break cluster), exactly the curvature blowup this
    # parameterization exists to prevent. Clipping raw_lo and raw_hi
    # independently against the cap can invert them (lo > hi) if a segment's
    # required slope to bridge two very different, asymmetric per-station
    # ranges exceeds the cap in a single direction -- _clip_slope collapses
    # that degenerate case to a single valid point instead of producing an
    # invalid Var bound.
    le_offset_slope_bounds = [
        _clip_slope((b["le_lo"][i + 1] - b["le_hi"][i]) / dy[i], (b["le_hi"][i + 1] - b["le_lo"][i]) / dy[i], MAX_LE_OFFSET_SLOPE_M_PER_SPAN)
        for i in range(n - 1)
    ]
    z_offset_slope_bounds = [
        _clip_slope((b["z_lo"][i + 1] - b["z_hi"][i]) / dy[i], (b["z_hi"][i + 1] - b["z_lo"][i]) / dy[i], MAX_Z_OFFSET_SLOPE_M_PER_SPAN)
        for i in range(n - 1)
    ]
    # Every segment ENTERING the winglet region (y_control[i+1] >=
    # threshold) additionally gets its slope lower bound floored at 0 --
    # not just the segment ending at the tip -- so a random sample is
    # biased toward an upturned tip across the whole winglet region, not
    # just its last segment (station-level floors alone only bias the
    # segment leading INTO the region; two consecutive floored stations
    # with the same range don't by themselves force a rise between them).
    def _floor_nonneg(lo: float, hi: float) -> tuple[float, float]:
        lo2 = max(lo, 0.0)
        return lo2, max(hi, lo2)

    z_offset_slope_bounds = [
        _floor_nonneg(lo, hi) if y_control[i + 1] >= Z_OFFSET_TIP_SEGMENT_Y_THRESHOLD else (lo, hi)
        for i, (lo, hi) in enumerate(z_offset_slope_bounds)
    ]

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
    variables.append(Var("le_offset_deviation_root_m", float(b["le_lo"][0]), float(b["le_hi"][0]), default=planform.le_offset_deviation_m[0]))
    for i in range(n - 1):
        lo, hi = le_offset_slope_bounds[i]
        variables.append(Var(f"le_offset_slope_{i}", lo, hi, default=float(np.clip(le_slopes_default[i], lo, hi))))
    variables.append(Var("z_offset_root_m", float(b["z_lo"][0]), float(b["z_hi"][0]), default=planform.z_offset_m[0]))
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
    progress_cb: ProgressCallback | None = None,
) -> tuple[OptimizationResult, DesignParameters]:
    """Optimize the planform; returns the optimization result plus the
    resulting full DesignParameters (airfoil schedule unchanged from baseline)."""
    weights = weights or ObjectiveWeights()
    optimizer = optimizer or HierarchicalGridSearch()

    parameter_set = make_stage2_parameter_set(baseline)
    objective_fn = Stage2Objective(parameter_set, weights)

    result = optimizer.optimize(objective_fn, parameter_set.bounds, x0=parameter_set.default_vector, progress_cb=progress_cb)
    best_params = parameter_set.build(result.best_x)
    return result, best_params
