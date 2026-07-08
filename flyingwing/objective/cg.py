"""Center-of-gravity estimate and the derived, physically meaningful static
margin.

Longitudinal (x, chordwise) placement drives static margin -- the aircraft is
symmetric left-right, so spanwise mass moments cancel and only chordwise
position affects pitch (longitudinal) stability. Each component's position is
now placed by a quick, cheap analysis of the actual generated geometry rather
than an assumed constant fraction of root chord:

  - motor/ESC: rear-mounted pusher prop, near the trailing edge at the root
    (y=0) -- MOTOR_ESC_X_FRACTION_CHORD of root chord.
  - servos: near the elevons, not the root -- SERVO_X_FRACTION_CHORD of the
    LOCAL chord at SERVO_Y_STATION (evaluated on the aircraft's own
    chord(y)/x_le(y) distributions, so it tracks whatever planform this
    design actually has).
  - avionics + battery: no single correct chordwise fraction the way a rear
    motor or elevon-adjacent servos have -- they're freely placeable inside
    the fuselage, so they default to the centroid of the REAL internal
    fuselage box (`aircraft.fuselage_fit`, the actual best-fit placement
    search in geometry/fuselage.py, not a root-chord-footprint proxy).
  - shell/spar: distributed mass, each getting an x-centroid derived from the
    MH64 profile's own geometry (wetted-perimeter centroid for the shell,
    max-thickness location for the spar) -- see geometry/airfoil_family.py.

The battery's mass is known but its position is still treated as the free
variable, since that's the one component a builder actually chooses the
placement of. `estimate_cg` reports both a concrete "assumed" CG/static
margin (battery at the fuselage box centroid) and the x-range the battery
could occupy while keeping static margin inside a target band -- a
physically actionable answer ("mount the battery between x=... and x=...
from the nose") rather than an abstract number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import _overrides as _ov
from ..geometry.aircraft import Aircraft
from ..geometry.airfoil_family import max_thickness_x_over_c, shell_centroid_x_over_c
from ..analysis.structures import StructuralProxyResult
from ..config import BOUNDS_OVERRIDES_YAML, FUSELAGE_MIN_INTERNAL_LENGTH_M
from .mass import MassEstimate, MOTOR_ESC_MASS_KG, AVIONICS_MASS_KG, SERVO_MASS_KG

# Placement rules -- not exposed as separate GUI-tunable "fraction of root
# chord" scalars (gui/config_tab.py) since positions are now derived by
# evaluating the aircraft's own geometry, not assumed as an independent
# constant. Still overridable via bounds_overrides.yaml for advanced tuning.
MOTOR_ESC_X_FRACTION_CHORD = 0.95  # rear pusher prop, near the TE at the root
SERVO_Y_STATION = 0.4  # normalized span station where the elevon servos sit
SERVO_X_FRACTION_CHORD = 0.5  # fraction of LOCAL chord at SERVO_Y_STATION

# The battery's mass is fixed (should match whatever battery is assumed in
# objective/performance.py's endurance estimate) but its position is
# estimate_cg's main output, not an input.
BATTERY_MASS_KG = 0.15

# Matches ObjectiveWeights.static_margin_target's default. Used to compute
# the battery position range inside estimate_cg (called from evaluate_design,
# which has no access to whatever ObjectiveWeights a caller might use later)
# -- call battery_position_range directly with a different target for an
# up-to-date range under different weights.
DEFAULT_STATIC_MARGIN_TARGET = (0.02, 0.15)

_ov.apply_overrides(globals(), BOUNDS_OVERRIDES_YAML, {
    "MOTOR_ESC_X_FRACTION_CHORD", "SERVO_Y_STATION", "SERVO_X_FRACTION_CHORD",
    "BATTERY_MASS_KG",
})


@dataclass
class MassComponent:
    name: str
    mass_kg: float
    x_m: float
    y_m: float = 0.0  # spanwise position, m -- 0 unless the component isn't on the centerline (e.g. servos)
    z_m: float = 0.0  # vertical position, m


@dataclass
class CGEstimate:
    components: list[MassComponent] = field(default_factory=list)  # fixed components only, not the battery
    fixed_mass_kg: float = float("nan")
    fixed_moment_kgm: float = float("nan")  # sum(mass_i * x_i) over fixed components

    battery_mass_kg: float = float("nan")
    battery_x_assumed_m: float = float("nan")
    battery_z_assumed_m: float = float("nan")
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


def _interp_at_y(y_stations: np.ndarray, values: np.ndarray, y: float) -> float:
    return float(np.interp(y, y_stations, values))


def fuselage_box_placement(aircraft: Aircraft) -> tuple[float, float, float, float]:
    """(x_center_m, z_center_m, x_min_m, x_max_m) of the real internal
    fuselage box -- avionics and the battery are "freely placeable" within
    it, so they default to its centroid. Falls back to a root-chord-centered
    box if no valid placement was found (same fallback
    viz/geometry_plots.py._fuselage_box_bounds uses, so the CG model and the
    fuselage-box visualization stay consistent even for an otherwise-invalid
    design)."""
    fit = aircraft.fuselage_fit
    if np.isnan(fit.box_x_min_m):
        x_center = float(aircraft.x_le_m[0] + aircraft.chord_m[0] / 2.0)
        z_center = float(aircraft.z_le_m[0])
        half_length = FUSELAGE_MIN_INTERNAL_LENGTH_M / 2.0
        return x_center, z_center, x_center - half_length, x_center + half_length
    x_center = (fit.box_x_min_m + fit.box_x_max_m) / 2.0
    z_center = (fit.box_z_min_m + fit.box_z_max_m) / 2.0
    return x_center, z_center, fit.box_x_min_m, fit.box_x_max_m


def _fixed_components(aircraft: Aircraft, structures: StructuralProxyResult, mass: MassEstimate) -> list[MassComponent]:
    x_le_root = float(aircraft.x_le_m[0])
    chord_root = float(aircraft.chord_m[0])
    z_le_root = float(aircraft.z_le_m[0])

    motor_x = x_le_root + MOTOR_ESC_X_FRACTION_CHORD * chord_root

    servo_chord = _interp_at_y(aircraft.y_stations, aircraft.chord_m, SERVO_Y_STATION)
    servo_x_le = _interp_at_y(aircraft.y_stations, aircraft.x_le_m, SERVO_Y_STATION)
    servo_x = servo_x_le + SERVO_X_FRACTION_CHORD * servo_chord
    servo_y = SERVO_Y_STATION * aircraft.half_span_m
    servo_z = _interp_at_y(aircraft.y_stations, aircraft.z_le_m, SERVO_Y_STATION)

    avionics_x, avionics_z, _, _ = fuselage_box_placement(aircraft)

    components = [
        MassComponent("motor_esc", MOTOR_ESC_MASS_KG, motor_x, 0.0, z_le_root),
        MassComponent("avionics", AVIONICS_MASS_KG, avionics_x, 0.0, avionics_z),
        MassComponent("servos", SERVO_MASS_KG, servo_x, servo_y, servo_z),
    ]

    # Shell mass isn't concentrated at one station -- distribute it along
    # the span weighted by local chord (a wetted-area proxy, consistent
    # with how the *total* shell mass is itself derived from wing area),
    # each station's x-centroid at the MH64 profile's own wetted-perimeter
    # centroid (computed once from the real airfoil geometry, not assumed).
    shell_centroid_frac = shell_centroid_x_over_c()
    chord_integral = np.trapezoid(aircraft.chord_m, aircraft.span_station_m)
    shell_x_per_station = aircraft.x_le_m + shell_centroid_frac * aircraft.chord_m
    if chord_integral > 0:
        shell_x_centroid = float(np.trapezoid(aircraft.chord_m * shell_x_per_station, aircraft.span_station_m) / chord_integral)
        shell_z_centroid = float(np.trapezoid(aircraft.chord_m * aircraft.z_le_m, aircraft.span_station_m) / chord_integral)
    else:
        shell_x_centroid = x_le_root
        shell_z_centroid = z_le_root
    components.append(MassComponent("shell", mass.shell_mass_kg, shell_x_centroid, 0.0, shell_z_centroid))

    # Spar mass is already distributed per station (spar_material_area_m2);
    # its x-centroid uses the MH64 profile's own max-thickness location --
    # the standard, physically sensible spar placement -- instead of an
    # assumed fraction.
    spar_x_frac = max_thickness_x_over_c()
    spar_x_per_station = aircraft.x_le_m + spar_x_frac * aircraft.chord_m
    spar_area_total = np.trapezoid(structures.spar_material_area_m2, structures.span_station_m)
    if spar_area_total > 0:
        spar_x_centroid = float(np.trapezoid(structures.spar_material_area_m2 * spar_x_per_station, structures.span_station_m) / spar_area_total)
        spar_z_centroid = float(np.trapezoid(structures.spar_material_area_m2 * aircraft.z_le_m, structures.span_station_m) / spar_area_total)
    else:
        spar_x_centroid = x_le_root
        spar_z_centroid = z_le_root
    components.append(MassComponent("spar", mass.spar_mass_kg, spar_x_centroid, 0.0, spar_z_centroid))

    return components


def estimate_cg(
    aircraft: Aircraft, structures: StructuralProxyResult, mass: MassEstimate,
    neutral_point_x_m: float, mac_m: float,
    battery_mass_kg: float = BATTERY_MASS_KG,
) -> CGEstimate:
    components = _fixed_components(aircraft, structures, mass)
    fixed_mass = sum(c.mass_kg for c in components)
    fixed_moment = sum(c.mass_kg * c.x_m for c in components)

    battery_x_assumed, battery_z_assumed, box_x_min, box_x_max = fuselage_box_placement(aircraft)

    total_mass = fixed_mass + battery_mass_kg
    cg_x_assumed = (fixed_moment + battery_mass_kg * battery_x_assumed) / total_mass if total_mass > 0 else battery_x_assumed
    static_margin_assumed = (neutral_point_x_m - cg_x_assumed) / mac_m if mac_m > 0 else float("nan")

    battery_x_min, battery_x_max, math_feasible = battery_position_range(
        fixed_mass, fixed_moment, battery_mass_kg, neutral_point_x_m, mac_m,
    )
    # battery_position_range only checks the target-range math is
    # self-consistent (x_min <= x_max) -- also clip against the real
    # internal fuselage box (geometry.fuselage.check_fuselage_fit's actual
    # placement search), not a rough root-chord-footprint proxy, so a
    # mathematically valid but physically unbuildable range is reported
    # honestly as infeasible.
    if math_feasible:
        battery_x_min = max(battery_x_min, box_x_min)
        battery_x_max = min(battery_x_max, box_x_max)
    feasible = math_feasible and battery_x_min <= battery_x_max

    return CGEstimate(
        components=components, fixed_mass_kg=fixed_mass, fixed_moment_kgm=fixed_moment,
        battery_mass_kg=battery_mass_kg,
        battery_x_assumed_m=battery_x_assumed, battery_z_assumed_m=battery_z_assumed,
        cg_x_assumed_m=cg_x_assumed, static_margin_assumed=static_margin_assumed,
        battery_x_min_m=battery_x_min, battery_x_max_m=battery_x_max,
        battery_range_feasible=feasible,
    )
