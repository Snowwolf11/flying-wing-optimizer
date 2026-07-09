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

import numpy as np
import yaml

from .metrics import DesignMetrics
from ..config import (
    FUSELAGE_MIN_INTERNAL_HEIGHT_M, FUSELAGE_MIN_INTERNAL_LENGTH_M,
    MIN_ABSOLUTE_THICKNESS_M, MIN_SPAR_DEPTH_M, MAX_LE_CURVATURE_PER_M, MAX_Z_CURVATURE_PER_M,
)

# Reference scales used only to make constraint-violation terms of very
# different units (meters, degrees, 1/meters, dimensionless ratios)
# comparable before summing them into one penalty -- "violation as a
# fraction of a characteristic threshold," not a physical unit conversion.
_CHORD_VIOLATION_REFERENCE_M = 0.05
_TWIST_VIOLATION_REFERENCE_DEG = 2.0


@dataclass
class ObjectiveWeights:
    # Aerodynamics (maximize). Each of these is divided by the matching
    # NormalizationConstants entry before being weighted (see score()) --
    # weight values below were rescaled (old_weight * default-baseline value
    # of the metric) when normalization was introduced, so a design
    # identical to the default baseline scores exactly as it did before.
    # Going forward, weight alone controls relative importance.
    w_cruise_L_over_D: float = 7.558
    w_fast_L_over_D: float = 3.837
    w_root_cl_max: float = 2.761

    # Structure / mass
    w_mass: float = 2.889  # minimize total_structural_mass_kg
    w_safety_factor: float = 0.05  # threshold, penalizes only if below safety_factor_min -- not normalized (see NormalizationConstants docstring)
    safety_factor_min: float = 1.5

    # Stability. static_margin now uses a real (though assumption-heavy)
    # component-mass-and-position CG estimate (objective/cg.py), not a
    # placeholder reference point -- it's meaningful to optimize against.
    # Still disabled (weight 0) by default so existing tuned weight files
    # don't silently change behavior; enable deliberately.
    w_static_margin: float = 0.0
    static_margin_target: tuple[float, float] = (0.02, 0.15)

    w_cm0: float = 5.0  # threshold, penalizes root zero-lift Cm more negative than cm0_min
    cm0_min: float = -0.02

    # Payload / geometry (maximize)
    w_payload_volume: float = 5.727  # payload_volume_margin_m3, normalized (see above)

    # Soaring performance at the cruise trim condition (cheap proxies, see
    # objective/metrics.py -- NOT the more precise best-glide-alpha sweep in
    # objective/performance.py). Disabled (weight 0) by default -- enable
    # deliberately.
    w_soaring_power: float = 0.0  # minimize soaring_power_w (power dissipated while gliding -- lower means it stays aloft in weaker lift)
    w_flight_angle: float = 0.0  # minimize cruise_glide_angle_deg (shallower glide)

    # Roll (lateral) stability -- "dihedral effect" (see objective/metrics.py
    # and analysis/aero_3d.py). cruise_Clb_per_rad is negative for a roll-
    # stable design, so this term is scored on -Clb (a "the more, the
    # better" quantity, same maximize pattern as cruise_L_over_D) -- more
    # negative Clb increases the score, a positive (roll-unstable) Clb
    # decreases it. Disabled (weight 0) by default -- enable deliberately.
    w_roll_stability: float = 0.0

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
        # A null entry (e.g. from a hand-edited file, or a GUI number input
        # that silently emptied out -- see gui/config_tab.py's save
        # callbacks for the primary fix) would otherwise construct a field
        # as None and crash score() far away from the actual cause; drop it
        # instead so the field falls back to its dataclass default.
        data = {k: v for k, v in data.items() if v is not None}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        data = asdict(self)
        data["static_margin_target"] = list(data["static_margin_target"])
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)


@dataclass
class NormalizationConstants:
    """Reference scale for each "maximize"/"minimize" objective term (the
    "threshold"/"target range" terms -- safety_factor, static_margin, cm0 --
    already compare against a physically meaningful threshold in native
    units, so a design at the threshold scores exactly 0 there regardless of
    units; normalizing them "to 1 at the default design" doesn't fit the
    same pattern and isn't done here).

    Without this, a term's importance is mostly determined by its raw
    physical magnitude rather than its weight -- e.g. payload volume margin
    is O(1e-3 m^3) while L/D is O(10), so before this existed
    w_payload_volume had to be ~1000x w_cruise_L_over_D just to be visible
    at all, and that scale factor was entangled with (and hid) the actual
    relative importance the weights were supposed to express.

    Each metric is divided by its normalization constant before being
    weighted -- the defaults below are that metric's own value computed for
    this project's default baseline design (`geometry.params.
    default_design_parameters()`), so for that design every normalized term
    evaluates to ~1.0 and the weight alone sets its contribution to the
    score. Regenerate them (e.g. via evaluate_design on a new baseline) if
    the baseline design changes materially.
    """

    norm_cruise_L_over_D: float = 7.558
    norm_fast_L_over_D: float = 9.591
    norm_root_cl_max: float = 1.381
    norm_mass: float = 0.963
    norm_payload_volume: float = 0.005727
    norm_soaring_power: float = 33.84
    norm_flight_angle: float = 7.537
    norm_roll_stability: float = 0.126  # magnitude of cruise_Clb_per_rad for the default baseline (see score() -- the term is scored on -Clb, so this stays positive like the other norm_* constants)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NormalizationConstants":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # See ObjectiveWeights.from_yaml -- same null-entry guard.
        data = {k: v for k, v in data.items() if v is not None}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


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
    penalty += metrics.z_curvature_violation / MAX_Z_CURVATURE_PER_M
    return penalty


def score(
    metrics: DesignMetrics, weights: ObjectiveWeights = None, normalization: NormalizationConstants = None,
) -> ObjectiveResult:
    weights = weights or ObjectiveWeights()
    n = normalization or NormalizationConstants()
    c: dict[str, float] = {}

    c["cruise_L_over_D"] = weights.w_cruise_L_over_D * metrics.cruise_L_over_D / n.norm_cruise_L_over_D
    c["fast_L_over_D"] = weights.w_fast_L_over_D * metrics.fast_L_over_D / n.norm_fast_L_over_D
    c["root_cl_max"] = weights.w_root_cl_max * metrics.root_cl_max / n.norm_root_cl_max

    c["mass"] = -weights.w_mass * metrics.total_structural_mass_kg / n.norm_mass
    c["safety_factor"] = -weights.w_safety_factor * max(0.0, weights.safety_factor_min - metrics.min_safety_factor) ** 2

    lo, hi = weights.static_margin_target
    sm = metrics.static_margin
    c["static_margin"] = -weights.w_static_margin * (max(0.0, lo - sm) ** 2 + max(0.0, sm - hi) ** 2)

    c["cm0"] = -weights.w_cm0 * max(0.0, weights.cm0_min - metrics.root_cm_zero_lift) ** 2

    c["payload_volume"] = weights.w_payload_volume * metrics.payload_volume_margin_m3 / n.norm_payload_volume

    # soaring_power_w/cruise_glide_angle_deg can be NaN for a pathological
    # (near-zero-or-negative-L/D) candidate -- guard the weight==0 default so
    # an occasional undefined value on an otherwise-irrelevant term doesn't
    # poison every score with NaN.
    c["soaring_power"] = 0.0 if weights.w_soaring_power == 0 else -weights.w_soaring_power * metrics.soaring_power_w / n.norm_soaring_power
    c["flight_angle"] = 0.0 if weights.w_flight_angle == 0 else -weights.w_flight_angle * metrics.cruise_glide_angle_deg / n.norm_flight_angle

    # Scored on -Clb (positive = roll-stable), so this is a "maximize"-style
    # term like cruise_L_over_D, not a "minimize" one like mass -- see
    # ObjectiveWeights.w_roll_stability. Same NaN guard as soaring_power/
    # flight_angle above.
    c["roll_stability"] = 0.0 if weights.w_roll_stability == 0 else weights.w_roll_stability * (-metrics.cruise_Clb_per_rad) / n.norm_roll_stability

    c["constraint_penalty"] = -weights.constraint_penalty_scale * _constraint_penalty(metrics)

    total = sum(c.values())
    if not metrics.valid:
        total -= weights.invalid_penalty
        c["invalid_penalty"] = -weights.invalid_penalty

    # A NaN can still reach here for a pathological candidate with a nonzero
    # soaring_power/flight_angle/roll_stability weight (the weight==0 guards
    # above only cover the *disabled* case) -- e.g. a wildly invalid,
    # negative-L/D design makes cruise_glide_angle_deg undefined. Left as
    # NaN, `max(candidates, key=lambda c: c.score)` elsewhere (hierarchical.py,
    # cmaes.py) has a real footgun: Python's max() never replaces a NaN
    # "current best" with a later, better real number (`5 > nan` is False),
    # so a single NaN candidate can silently "win" over every valid one.
    # -inf sorts as worst-possible instead, which is what a degenerate
    # candidate should be.
    if not np.isfinite(total):
        total = -np.inf

    return ObjectiveResult(score=total, contributions=c)
