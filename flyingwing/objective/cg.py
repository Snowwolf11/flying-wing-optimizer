"""Center-of-gravity estimate and the derived, physically meaningful static
margin.

Longitudinal (x, chordwise) placement only -- the aircraft is symmetric
left-right, so spanwise mass moments cancel and only chordwise position
affects pitch (longitudinal) stability.

Each fixed component (motor/ESC, avionics, servos, shell, spar) gets an
assumed x-position; the battery's mass is known but its position is treated
as the free variable, since that's the one component a builder actually
chooses the placement of. `estimate_cg` reports both a concrete "assumed"
CG/static margin (battery at a configurable default position) and the
x-range the battery could occupy while keeping static margin inside a
target band -- a physically actionable answer ("mount the battery between
x=... and x=... from the nose") rather than an abstract number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import _overrides as _ov
from ..geometry.aircraft import Aircraft
from ..analysis.structures import StructuralProxyResult
from ..config import BOUNDS_OVERRIDES_YAML
from .mass import MassEstimate, FIXED_EQUIPMENT_MASS_KG

# Fractions of FIXED_EQUIPMENT_MASS_KG. Motor/ESC and servos both sit aft
# (rear-mounted pusher prop -- the common flying-wing FPV layout -- and
# elevon servos embedded near the trailing edge); avionics (FC/receiver/
# wiring) sits more centrally. Sum to 1.0.
MOTOR_ESC_MASS_FRACTION = 0.30
AVIONICS_MASS_FRACTION = 0.35
SERVO_MASS_FRACTION = 0.35

# x-position of each fixed component, as a fraction of ROOT chord measured
# from the root leading edge -- these components live in the centre body
# near the root, not spread along the span.
MOTOR_ESC_X_FRACTION_CHORD = 0.95
AVIONICS_X_FRACTION_CHORD = 0.35
SERVO_X_FRACTION_CHORD = 0.85

# Chordwise centroid assumptions for mass that IS distributed along the
# span, as a fraction of local chord at each station.
SHELL_CENTROID_X_FRACTION_CHORD = 0.40
SPAR_X_FRACTION_CHORD = 0.30  # near max thickness, standard spar placement

# The battery's mass is fixed (should match whatever battery is assumed in
# objective/performance.py's endurance estimate) but its position is
# estimate_cg's main output, not an input -- BATTERY_X_FRACTION_CHORD is
# only used to report one concrete "assumed" CG/static margin value,
# defaulting to a typical payload-bay location.
BATTERY_MASS_KG = 0.15
BATTERY_X_FRACTION_CHORD = 0.35

# Matches ObjectiveWeights.static_margin_target's default. Used to compute
# the battery position range inside estimate_cg (called from evaluate_design,
# which has no access to whatever ObjectiveWeights a caller might use later)
# -- call battery_position_range directly with a different target for an
# up-to-date range under different weights.
DEFAULT_STATIC_MARGIN_TARGET = (0.02, 0.15)

_ov.apply_overrides(globals(), BOUNDS_OVERRIDES_YAML, {
    "MOTOR_ESC_MASS_FRACTION", "AVIONICS_MASS_FRACTION", "SERVO_MASS_FRACTION",
    "MOTOR_ESC_X_FRACTION_CHORD", "AVIONICS_X_FRACTION_CHORD", "SERVO_X_FRACTION_CHORD",
    "SHELL_CENTROID_X_FRACTION_CHORD", "SPAR_X_FRACTION_CHORD",
    "BATTERY_MASS_KG", "BATTERY_X_FRACTION_CHORD",
})


@dataclass
class MassComponent:
    name: str
    mass_kg: float
    x_m: float


@dataclass
class CGEstimate:
    components: list[MassComponent] = field(default_factory=list)  # fixed components only, not the battery
    fixed_mass_kg: float = float("nan")
    fixed_moment_kgm: float = float("nan")  # sum(mass_i * x_i) over fixed components

    battery_mass_kg: float = float("nan")
    battery_x_assumed_m: float = float("nan")
    cg_x_assumed_m: float = float("nan")  # CG with the battery at its assumed position
    static_margin_assumed: float = float("nan")

    battery_x_min_m: float = float("nan")  # valid range keeping static margin in DEFAULT_STATIC_MARGIN_TARGET
    battery_x_max_m: float = float("nan")
    battery_range_feasible: bool = False


def battery_position_range(
    fixed_mass_kg: float, fixed_moment_kgm: float, battery_mass_kg: float,
    neutral_point_x_m: float, mac_m: float,
    static_margin_target: tuple[float, float] = DEFAULT_STATIC_MARGIN_TARGET,
) -> tuple[float, float, bool]:
    """The battery x-range that keeps static_margin = (x_np - x_cg)/MAC
    inside `static_margin_target`, given every other component's mass is
    fixed at `fixed_mass_kg` with combined moment `fixed_moment_kgm`.
    Returns (x_min, x_max, feasible) -- feasible is False if x_min > x_max
    (no battery position satisfies the target, e.g. not enough fixed mass
    aft/forward of the neutral point to trim into range)."""
    lo, hi = static_margin_target
    total_mass = fixed_mass_kg + battery_mass_kg
    # static_margin = (x_np - x_cg)/MAC  =>  x_cg = x_np - static_margin*MAC
    x_cg_forward_limit = neutral_point_x_m - hi * mac_m  # highest static margin -> most-forward CG bound
    x_cg_aft_limit = neutral_point_x_m - lo * mac_m  # lowest static margin -> most-aft CG bound

    if battery_mass_kg <= 0 or total_mass <= 0:
        return float("nan"), float("nan"), False

    def _x_battery_for(x_cg_bound: float) -> float:
        return (x_cg_bound * total_mass - fixed_moment_kgm) / battery_mass_kg

    x_min = _x_battery_for(x_cg_forward_limit)
    x_max = _x_battery_for(x_cg_aft_limit)
    return x_min, x_max, x_min <= x_max


def _fixed_components(aircraft: Aircraft, structures: StructuralProxyResult, mass: MassEstimate) -> list[MassComponent]:
    x_le_root = float(aircraft.x_le_m[0])
    chord_root = float(aircraft.chord_m[0])

    components = [
        MassComponent("motor_esc", FIXED_EQUIPMENT_MASS_KG * MOTOR_ESC_MASS_FRACTION, x_le_root + MOTOR_ESC_X_FRACTION_CHORD * chord_root),
        MassComponent("avionics", FIXED_EQUIPMENT_MASS_KG * AVIONICS_MASS_FRACTION, x_le_root + AVIONICS_X_FRACTION_CHORD * chord_root),
        MassComponent("servos", FIXED_EQUIPMENT_MASS_KG * SERVO_MASS_FRACTION, x_le_root + SERVO_X_FRACTION_CHORD * chord_root),
    ]

    # Shell mass isn't concentrated at one station -- distribute it along
    # the span weighted by local chord (a wetted-area proxy, consistent
    # with how the *total* shell mass is itself derived from wing area),
    # each station's x-centroid at a fixed chord fraction.
    chord_integral = np.trapezoid(aircraft.chord_m, aircraft.span_station_m)
    shell_x_per_station = aircraft.x_le_m + SHELL_CENTROID_X_FRACTION_CHORD * aircraft.chord_m
    if chord_integral > 0:
        shell_x_centroid = float(np.trapezoid(aircraft.chord_m * shell_x_per_station, aircraft.span_station_m) / chord_integral)
    else:
        shell_x_centroid = x_le_root
    components.append(MassComponent("shell", mass.shell_mass_kg, shell_x_centroid))

    # Spar mass is already distributed per station (spar_material_area_m2);
    # its x-centroid is the area-weighted average station position.
    spar_x_per_station = aircraft.x_le_m + SPAR_X_FRACTION_CHORD * aircraft.chord_m
    spar_area_total = np.trapezoid(structures.spar_material_area_m2, structures.span_station_m)
    if spar_area_total > 0:
        spar_x_centroid = float(np.trapezoid(structures.spar_material_area_m2 * spar_x_per_station, structures.span_station_m) / spar_area_total)
    else:
        spar_x_centroid = x_le_root
    components.append(MassComponent("spar", mass.spar_mass_kg, spar_x_centroid))

    return components


def estimate_cg(
    aircraft: Aircraft, structures: StructuralProxyResult, mass: MassEstimate,
    neutral_point_x_m: float, mac_m: float,
    battery_mass_kg: float = BATTERY_MASS_KG,
) -> CGEstimate:
    components = _fixed_components(aircraft, structures, mass)
    fixed_mass = sum(c.mass_kg for c in components)
    fixed_moment = sum(c.mass_kg * c.x_m for c in components)

    x_le_root = float(aircraft.x_le_m[0])
    chord_root = float(aircraft.chord_m[0])
    battery_x_assumed = x_le_root + BATTERY_X_FRACTION_CHORD * chord_root

    total_mass = fixed_mass + battery_mass_kg
    cg_x_assumed = (fixed_moment + battery_mass_kg * battery_x_assumed) / total_mass if total_mass > 0 else x_le_root
    static_margin_assumed = (neutral_point_x_m - cg_x_assumed) / mac_m if mac_m > 0 else float("nan")

    battery_x_min, battery_x_max, math_feasible = battery_position_range(
        fixed_mass, fixed_moment, battery_mass_kg, neutral_point_x_m, mac_m,
    )
    # battery_position_range only checks the target-range math is
    # self-consistent (x_min <= x_max) -- also clip against the root
    # chord's footprint (a reasonable proxy for "inside the airframe"), so a
    # mathematically valid but physically unbuildable range (e.g. requiring
    # the battery ahead of the nose) is reported honestly as infeasible.
    physical_lo, physical_hi = x_le_root, x_le_root + chord_root
    if math_feasible:
        battery_x_min = max(battery_x_min, physical_lo)
        battery_x_max = min(battery_x_max, physical_hi)
    feasible = math_feasible and battery_x_min <= battery_x_max

    return CGEstimate(
        components=components, fixed_mass_kg=fixed_mass, fixed_moment_kgm=fixed_moment,
        battery_mass_kg=battery_mass_kg, battery_x_assumed_m=battery_x_assumed,
        cg_x_assumed_m=cg_x_assumed, static_margin_assumed=static_margin_assumed,
        battery_x_min_m=battery_x_min, battery_x_max_m=battery_x_max,
        battery_range_feasible=feasible,
    )
