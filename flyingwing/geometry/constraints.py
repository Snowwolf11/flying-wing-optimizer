"""Design validity constraints.

Stage 1 (airfoil schedule) and Stage 2 (planform) each have their own
constraint set, but a fully-generated aircraft always has both an airfoil
schedule and a planform -- so a design is only truly valid if it satisfies
both sets simultaneously, regardless of which stage produced it. Use
`check_all_constraints` for that combined check (what `evaluate_design`
uses); the individual `check_stage*_constraints` functions remain useful on
their own for isolated debugging.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aircraft import Aircraft
from .params import DesignParameters
from .airfoil_family import MIN_THICKNESS_RATIO, max_thickness_ratio
from ..config import (
    MIN_ABSOLUTE_THICKNESS_M, MIN_SPAR_DEPTH_M, MAX_LE_CURVATURE_PER_M, MAX_Z_CURVATURE_PER_M,
    SPAR_DEPTH_FRACTION_THICKNESS, FUSELAGE_MIN_INTERNAL_HEIGHT_M, FUSELAGE_MIN_INTERNAL_LENGTH_M,
)


@dataclass
class ConstraintReport:
    valid: bool
    violations: list[str]


def check_stage1_constraints(aircraft: Aircraft) -> ConstraintReport:
    violations: list[str] = []

    # thickness ratio must not increase anywhere along the span
    d = np.diff(aircraft.thickness_ratio)
    if np.any(d > 1e-9):
        violations.append(f"thickness ratio increases somewhere along span (max increase {float(np.max(d)):.4f})")

    # realistic minimum tip thickness
    tip_ratio = float(aircraft.thickness_ratio[-1])
    if tip_ratio < MIN_THICKNESS_RATIO:
        violations.append(f"tip thickness ratio {tip_ratio:.3f} below minimum {MIN_THICKNESS_RATIO:.3f}")

    # fuselage internal box must fit inside the centre body
    violations.extend(aircraft.fuselage_fit.violations)

    return ConstraintReport(valid=(len(violations) == 0), violations=violations)


def curve_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Curvature (1/radius) of x as a function of y, via finite differences."""
    dx = np.gradient(x, y)
    d2x = np.gradient(dx, y)
    return np.abs(d2x) / (1.0 + dx ** 2) ** 1.5


def check_stage2_constraints(aircraft: Aircraft) -> ConstraintReport:
    violations: list[str] = []

    chord_diff = np.diff(aircraft.chord_m)
    if np.any(chord_diff > 1e-9):
        violations.append(f"chord increases somewhere along span (max increase {float(np.max(chord_diff)) * 1000:.2f} mm)")

    twist_diff = np.diff(aircraft.twist_deg)
    if np.any(twist_diff > 1e-9):
        violations.append(f"twist (washout) increases somewhere along span (max increase {float(np.max(twist_diff)):.3f} deg)")

    local_thickness_m = aircraft.thickness_ratio * aircraft.chord_m
    min_local_thickness = float(np.min(local_thickness_m))
    if min_local_thickness < MIN_ABSOLUTE_THICKNESS_M:
        violations.append(f"local thickness {min_local_thickness * 1000:.2f} mm below minimum {MIN_ABSOLUTE_THICKNESS_M * 1000:.2f} mm")

    spar_depth_m = SPAR_DEPTH_FRACTION_THICKNESS * local_thickness_m
    min_spar_depth = float(np.min(spar_depth_m))
    if min_spar_depth < MIN_SPAR_DEPTH_M:
        violations.append(f"available spar depth {min_spar_depth * 1000:.2f} mm below minimum {MIN_SPAR_DEPTH_M * 1000:.2f} mm")

    if np.any(aircraft.chord_m <= 0):
        violations.append("chord is zero or negative somewhere along the span (self-intersecting geometry)")

    le_curvature = curve_curvature(aircraft.x_le_m, aircraft.span_station_m)
    max_le_curvature = float(np.max(le_curvature))
    if max_le_curvature > MAX_LE_CURVATURE_PER_M:
        violations.append(f"leading-edge curvature {max_le_curvature:.1f} /m exceeds maximum {MAX_LE_CURVATURE_PER_M:.1f} /m (not smoothly manufacturable)")

    z_curvature = curve_curvature(aircraft.z_le_m, aircraft.span_station_m)
    max_z_curvature = float(np.max(z_curvature))
    if max_z_curvature > MAX_Z_CURVATURE_PER_M:
        violations.append(f"vertical (winglet) curvature {max_z_curvature:.1f} /m exceeds maximum {MAX_Z_CURVATURE_PER_M:.1f} /m (not smoothly manufacturable)")

    return ConstraintReport(valid=(len(violations) == 0), violations=violations)


def check_all_constraints(aircraft: Aircraft) -> ConstraintReport:
    stage1 = check_stage1_constraints(aircraft)
    stage2 = check_stage2_constraints(aircraft)
    violations = stage1.violations + stage2.violations
    return ConstraintReport(valid=(len(violations) == 0), violations=violations)


def quick_reject_reason(params: DesignParameters) -> str | None:
    """Cheap validity pre-check directly on the raw control points, before
    paying for build_aircraft() + AeroBuildup + NeuralFoil (the dominant
    per-candidate cost, measured around 10s/eval on Stage 2 -- see the
    CMA-ES-vs-LHS benchmark). Returns a human-readable rejection reason if
    the design is DEFINITELY invalid, else None.

    Deliberately conservative -- every check here is chosen so it can only
    reject a candidate check_all_constraints() (the real, post-build check)
    would ALSO reject, never one it would have accepted:

    - Chord/twist monotonicity: exact, not approximate. chord_m/twist_deg
      are PCHIP-interpolated (spanwise.py) between these exact control
      points, and PCHIP is shape-preserving -- monotonic control points
      imply a monotonic full curve and vice versa, so checking the ~7
      control points is mathematically equivalent to checking the full
      ~200-station mesh, just far cheaper.
    - LE/Z offset curvature: same formula and threshold as the real check
      (curve_curvature vs MAX_LE_CURVATURE_PER_M/MAX_Z_CURVATURE_PER_M),
      evaluated on the sparse control points rather than the full mesh. This
      is only an approximation, but a coarse finite-difference estimate
      SMooths over sharp *local* bends rather than exaggerating them, so it
      is biased toward *underestimating* curvature relative to the fine
      mesh -- i.e. biased toward missing a rejection (a false negative,
      caught downstream anyway), not toward a false rejection. (This is why
      a raw slope-vs-search-bounds check isn't used instead: a segment can
      have a large average slope while still being perfectly straight --
      zero curvature -- so bounding slope directly risks rejecting a
      genuinely valid, merely steep, design.)
    - Fuselage fit: the real check searches for a placement across every
      footprint station simultaneously (geometry/fuselage.py) -- too
      expensive to replicate here. But chord is guaranteed monotonically
      non-increasing (checked above), so the root (y=0) station has the
      provably largest chord of any station -- if even the root's own
      thickness/chord can't contain the required box, no placement search
      over the more constrained full footprint can succeed either. Purely a
      one-directional guarantee (fails here => fails for real; passes here
      => still needs the real search), which is exactly what's needed for a
      reject-only pre-filter.

    Not exhaustive -- e.g. min local thickness/spar depth isn't checked here
    since it depends on the full thickness_ratio(y) x chord(y) product,
    which doesn't reduce to a cheap control-point check the same way. Any
    invalid candidate that slips past this still gets caught by the real,
    always-correct check_all_constraints() downstream -- this function only
    exists to skip the expensive build for the definitely-invalid case.
    """
    planform = params.planform
    chord = np.asarray(planform.chord_m, dtype=float)
    twist = np.asarray(planform.twist_deg, dtype=float)

    if np.any(np.diff(chord) > 1e-9):
        return "chord increases somewhere between control points (quick pre-check)"
    if np.any(np.diff(twist) > 1e-9):
        return "twist (washout) increases somewhere between control points (quick pre-check)"

    y_control = np.asarray(planform.y_control, dtype=float)
    if len(y_control) >= 3:
        span_station_control_m = y_control * (planform.span_m / 2.0)
        x_le_control = y_control * (planform.span_m / 2.0) * np.tan(np.radians(planform.sweep_deg)) + np.asarray(
            planform.le_offset_deviation_m, dtype=float
        )
        z_le_control = np.asarray(planform.z_offset_m, dtype=float)

        le_curvature = curve_curvature(x_le_control, span_station_control_m)
        if np.any(le_curvature > MAX_LE_CURVATURE_PER_M):
            return "leading-edge curvature exceeds maximum between control points (quick pre-check)"

        z_curvature = curve_curvature(z_le_control, span_station_control_m)
        if np.any(z_curvature > MAX_Z_CURVATURE_PER_M):
            return "vertical (winglet) curvature exceeds maximum between control points (quick pre-check)"

    chord_root = chord[0]
    thickness_scale_root = params.airfoil_schedule.thickness_scale[0]
    max_local_thickness_m = max_thickness_ratio(thickness_scale_root) * chord_root
    if max_local_thickness_m < FUSELAGE_MIN_INTERNAL_HEIGHT_M:
        return "root station's own best-case thickness can't fit the required fuselage height (quick pre-check)"
    if chord_root < FUSELAGE_MIN_INTERNAL_LENGTH_M:
        return "root chord shorter than the required fuselage length (quick pre-check)"

    return None
