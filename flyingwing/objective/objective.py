"""Objective function: combine a `DesignMetrics` into one scalar score.

Every metric maps to exactly one weighted contribution, of one of four
shapes:

- "maximize":       contribution = +weight * value
- "minimize":        contribution = -weight * value
- "threshold" (one-sided soft constraint, e.g. safety factor): contribution
  = -weight * max(0, threshold - value)^2 -- no reward for exceeding the
  threshold, since (for safety factor especially) the mass term already
  prices in the cost of extra strength.
- "target range" (e.g. static margin): quadratic penalty outside [lo, hi],
  zero inside.

Hard geometric constraint violations (fuselage fit, thickness monotonicity)
are added as a separate, much larger penalty term scaled by how far out of
bounds the design is -- smooth enough to give a hierarchical search a
gradient toward feasibility, but large enough that a feasible design always
outscores an infeasible one.

To add a new metric: add a field to `DesignMetrics`, add a weight (and any
threshold/target) to `ObjectiveWeights` below, add one line in `score()`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

from .metrics import DesignMetrics
from ..config import (
    FUSELAGE_MIN_INTERNAL_HEIGHT_M, FUSELAGE_MIN_INTERNAL_LENGTH_M,
    MIN_ABSOLUTE_THICKNESS_M, MIN_SPAR_DEPTH_M, MAX_LE_CURVATURE_PER_M,
)

# Reference scales used only to make constraint-violation terms of very
# different units (meters, degrees, 1/meters, dimensionless ratios)
# comparable before summing them into one penalty -- "violation as a
# fraction of a characteristic threshold," not a physical unit conversion.
_CHORD_VIOLATION_REFERENCE_M = 0.05
_TWIST_VIOLATION_REFERENCE_DEG = 2.0


@dataclass
class ObjectiveWeights:
    # Aerodynamics (maximize)
    w_cruise_L_over_D: float = 1.0
    w_fast_L_over_D: float = 0.4
    w_root_cl_max: float = 2.0

    # Structure / mass
    w_mass: float = 3.0  # minimize total_structural_mass_kg
    w_safety_factor: float = 0.05  # threshold, penalizes only if below safety_factor_min
    safety_factor_min: float = 1.5

    # Stability. Disabled (weight 0) by default: static margin is computed
    # relative to a placeholder 25%-MAC reference point, not a real CG, so
    # it isn't meaningful to optimize against until mass-distribution/CG
    # modeling exists. Turn it on once that's in place.
    w_static_margin: float = 0.0
    static_margin_target: tuple[float, float] = (0.02, 0.15)

    w_cm0: float = 5.0  # threshold, penalizes root zero-lift Cm more negative than cm0_min
    cm0_min: float = -0.02

    # Payload / geometry (maximize)
    w_payload_volume: float = 1000.0  # payload_volume_margin_m3 is O(1e-3 m^3); scaled to be comparable to the L/D terms

    # Hard geometric constraints (fuselage fit, thickness/chord/twist
    # monotonicity, min thickness/spar depth, LE curvature)
    invalid_penalty: float = 1000.0  # flat penalty applied whenever DesignMetrics.valid is False
    constraint_penalty_scale: float = 1000.0  # multiplies the summed (normalized, dimensionless) constraint violations

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ObjectiveWeights":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if "static_margin_target" in data:
            data["static_margin_target"] = tuple(data["static_margin_target"])
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        data = asdict(self)
        data["static_margin_target"] = list(data["static_margin_target"])
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)


@dataclass
class ObjectiveResult:
    score: float
    contributions: dict[str, float]


def _constraint_penalty(metrics: DesignMetrics) -> float:
    """Sum of constraint violations, each normalized to a dimensionless
    "fraction of a characteristic threshold" so meters/degrees/1-per-meter/
    ratio quantities are comparable before being summed and scaled."""
    penalty = 0.0
    penalty += max(0.0, -metrics.fuselage_height_margin_m) / FUSELAGE_MIN_INTERNAL_HEIGHT_M
    penalty += max(0.0, -metrics.fuselage_length_margin_m) / FUSELAGE_MIN_INTERNAL_LENGTH_M
    penalty += max(0.0, -metrics.tip_thickness_margin)  # already a dimensionless thickness ratio
    penalty += metrics.thickness_monotonic_violation  # already a dimensionless thickness ratio
    penalty += metrics.chord_monotonic_violation / _CHORD_VIOLATION_REFERENCE_M
    penalty += metrics.twist_monotonic_violation / _TWIST_VIOLATION_REFERENCE_DEG
    penalty += max(0.0, -metrics.min_local_thickness_margin) / MIN_ABSOLUTE_THICKNESS_M
    penalty += max(0.0, -metrics.min_spar_depth_margin) / MIN_SPAR_DEPTH_M
    penalty += metrics.le_curvature_violation / MAX_LE_CURVATURE_PER_M
    return penalty


def score(metrics: DesignMetrics, weights: ObjectiveWeights = None) -> ObjectiveResult:
    weights = weights or ObjectiveWeights()
    c: dict[str, float] = {}

    c["cruise_L_over_D"] = weights.w_cruise_L_over_D * metrics.cruise_L_over_D
    c["fast_L_over_D"] = weights.w_fast_L_over_D * metrics.fast_L_over_D
    c["root_cl_max"] = weights.w_root_cl_max * metrics.root_cl_max

    c["mass"] = -weights.w_mass * metrics.total_structural_mass_kg
    c["safety_factor"] = -weights.w_safety_factor * max(0.0, weights.safety_factor_min - metrics.min_safety_factor) ** 2

    lo, hi = weights.static_margin_target
    sm = metrics.static_margin
    c["static_margin"] = -weights.w_static_margin * (max(0.0, lo - sm) ** 2 + max(0.0, sm - hi) ** 2)

    c["cm0"] = -weights.w_cm0 * max(0.0, weights.cm0_min - metrics.root_cm_zero_lift) ** 2

    c["payload_volume"] = weights.w_payload_volume * metrics.payload_volume_margin_m3

    c["constraint_penalty"] = -weights.constraint_penalty_scale * _constraint_penalty(metrics)

    total = sum(c.values())
    if not metrics.valid:
        total -= weights.invalid_penalty
        c["invalid_penalty"] = -weights.invalid_penalty

    return ObjectiveResult(score=total, contributions=c)
