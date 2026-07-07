"""Design parameter vector: the single object the optimizer proposes and the
geometry generator consumes.

`DesignParameters` is a complete, self-consistent description of one flying
wing. It has no knowledge of NeuralFoil, AVL, structural proxies, or any
optimizer -- it is pure data. The geometry generator (`aircraft.py`) turns it
into a full 3D model; nothing else needs to know how the vector is laid out.

Stage 1 (airfoil optimization) only varies `airfoil_schedule`.
Stage 2 (planform optimization) only varies `planform`.
Both stages hold the other half fixed at whatever the previous stage
converged to (or at the defaults below, on a first pass).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np

from .spanwise import SpanwiseDistribution

# Airfoil-schedule control stations: root | inner centre-body edge |
# outer-wing start | mid-span | tip. The 0.09/0.14 pair brackets the
# fuselage-to-wing transition (matching DEFAULT_PLANFORM_Y_CONTROL's
# 0.08/0.12/0.14 break) so the airfoil schedule can change quickly there
# without needing a discontinuity.
DEFAULT_AIRFOIL_Y_CONTROL = (0.00, 0.09, 0.14, 0.60, 1.00)

# Planform control stations. Interpolated with PCHIP (see spanwise.py), so
# the planform is one smooth curve through these stations rather than a
# piecewise-linear, kinked one. The 0.08/0.12/0.14 cluster is where the
# fuselage ends and the outer wing begins.
DEFAULT_PLANFORM_Y_CONTROL = (0.00, 0.08, 0.12, 0.14, 0.60, 0.85, 1.00)


@dataclass(frozen=True)
class AirfoilSchedule:
    """thickness / camber / reflex scale distributions along the span.

    Each scale multiplies (thickness, camber) or adds an aft-loading bump
    (reflex) to the baseline MH64 shape -- see `airfoil_family.py`.
    """

    y_control: tuple = DEFAULT_AIRFOIL_Y_CONTROL
    thickness_scale: tuple = (1.7, 1.4, 1.1, 0.95, 0.85)
    camber_scale: tuple = (1.00, 1.00, 0.95, 0.85, 0.70)
    reflex_scale: tuple = (0.30, 0.30, 0.45, 0.65, 0.85)

    def distributions(self) -> dict[str, SpanwiseDistribution]:
        y = np.asarray(self.y_control, dtype=float)
        return {
            "thickness_scale": SpanwiseDistribution(y, np.asarray(self.thickness_scale, dtype=float)),
            "camber_scale": SpanwiseDistribution(y, np.asarray(self.camber_scale, dtype=float)),
            "reflex_scale": SpanwiseDistribution(y, np.asarray(self.reflex_scale, dtype=float)),
        }


@dataclass(frozen=True)
class Planform:
    """Global + spanwise planform description.

    Winglets are NOT parameterized separately. `le_offset_deviation_m`,
    `z_offset_m` and a tapering `chord_m` can bend the tip up and back on
    their own, so a winglet (or a gull wing, or a blended centre body) can
    emerge from the same continuous distributions used for the rest of the
    wing, without a discontinuity where a separate winglet part would begin.

    `sweep_deg` sets the reference straight-line sweep of the leading edge;
    `le_offset_deviation_m` is a local deviation *from* that reference line,
    so the optimizer can bend/curve the planform while still having a single
    global sweep scalar to search over. All four spanwise distributions
    (chord, twist, LE offset, z offset) are interpolated with PCHIP (smooth,
    shape-preserving -- see spanwise.py), unlike the airfoil schedule, which
    stays piecewise-linear.

    The default control points describe a forward-projecting nose that
    sweeps back fairly steeply into the wing root, a shallow mid-span sweep,
    and a stronger raked/blended sweep back into the tip.
    """

    span_m: float = 1.6
    sweep_deg: float = 30.0
    y_control: tuple = DEFAULT_PLANFORM_Y_CONTROL
    chord_m: tuple = (0.58, 0.50, 0.37, 0.35, 0.20, 0.15, 0.055)
    twist_deg: tuple = (0.0, 0.0, -0.3, -0.4, -2.0, -3.0, -5.0)
    le_offset_deviation_m: tuple = (0.000, 0.00, 0.06, 0.07, 0.1, 0.12, 0.16)
    z_offset_m: tuple = (0.00, 0.00, 0.01, 0.015, 0.07, 0.08, 0.18)

    def chord_distribution(self) -> SpanwiseDistribution:
        y = np.asarray(self.y_control, dtype=float)
        return SpanwiseDistribution(y, np.asarray(self.chord_m, dtype=float), kind="pchip")

    def twist_distribution(self) -> SpanwiseDistribution:
        y = np.asarray(self.y_control, dtype=float)
        return SpanwiseDistribution(y, np.asarray(self.twist_deg, dtype=float), kind="pchip")

    def z_offset_distribution(self) -> SpanwiseDistribution:
        y = np.asarray(self.y_control, dtype=float)
        return SpanwiseDistribution(y, np.asarray(self.z_offset_m, dtype=float), kind="pchip")

    def le_offset_distribution(self) -> SpanwiseDistribution:
        y = np.asarray(self.y_control, dtype=float)
        reference = y * (self.span_m / 2.0) * np.tan(np.radians(self.sweep_deg))
        deviation = np.asarray(self.le_offset_deviation_m, dtype=float)
        return SpanwiseDistribution(y, reference + deviation, kind="pchip")

    def distributions(self) -> dict[str, SpanwiseDistribution]:
        return {
            "chord_m": self.chord_distribution(),
            "twist_deg": self.twist_distribution(),
            "le_offset_m": self.le_offset_distribution(),
            "z_offset_m": self.z_offset_distribution(),
        }


@dataclass(frozen=True)
class DesignParameters:
    """A complete description of one flying-wing design.

    This is the only object the geometry generator accepts. Optimizers build
    one of these per candidate (typically by copying a fixed baseline and
    overriding either `planform` or `airfoil_schedule`) via
    `flyingwing.optimization.vector.ParameterSet`.
    """

    planform: Planform = field(default_factory=Planform)
    airfoil_schedule: AirfoilSchedule = field(default_factory=AirfoilSchedule)
    n_span_stations: int = 200

    def with_planform(self, **kwargs) -> "DesignParameters":
        return replace(self, planform=replace(self.planform, **kwargs))

    def with_airfoil_schedule(self, **kwargs) -> "DesignParameters":
        return replace(self, airfoil_schedule=replace(self.airfoil_schedule, **kwargs))


def default_design_parameters() -> DesignParameters:
    """A reasonable, valid starting point for a ~1.6 m FPV flying wing."""
    return DesignParameters()
