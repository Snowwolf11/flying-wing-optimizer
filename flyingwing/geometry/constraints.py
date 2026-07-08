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
from .airfoil_family import MIN_THICKNESS_RATIO
from ..config import (
    MIN_ABSOLUTE_THICKNESS_M, MIN_SPAR_DEPTH_M, MAX_LE_CURVATURE_PER_M, MAX_Z_CURVATURE_PER_M,
    SPAR_DEPTH_FRACTION_THICKNESS,
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
