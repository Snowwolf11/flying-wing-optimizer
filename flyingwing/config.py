"""Global constants and defaults shared across the framework.

Units: SI throughout (meters, kg, seconds, Pascals) unless a name says otherwise.
Speeds are also exposed in km/h for convenience since that's how the target
performance envelope is specified.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "configs"

# ---------------------------------------------------------------------------
# Target flight envelope
# ---------------------------------------------------------------------------
KMH_TO_MS = 1.0 / 3.6

CRUISE_SPEED_RANGE_KMH = (70.0, 80.0)
CRUISE_SPEED_KMH = 75.0
CRUISE_SPEED_MS = CRUISE_SPEED_KMH * KMH_TO_MS

TOP_SPEED_KMH = 150.0
TOP_SPEED_MS = TOP_SPEED_KMH * KMH_TO_MS

# ---------------------------------------------------------------------------
# Wingspan bounds (m)
# ---------------------------------------------------------------------------
WINGSPAN_MIN_M = 1.4
WINGSPAN_MAX_M = 1.8

# ---------------------------------------------------------------------------
# Required internal fuselage box (m) -- must fit inside the centre body
# ---------------------------------------------------------------------------
FUSELAGE_MIN_INTERNAL_WIDTH_M = 0.140
FUSELAGE_MIN_INTERNAL_HEIGHT_M = 0.055
FUSELAGE_MIN_INTERNAL_LENGTH_M = 0.300

# ---------------------------------------------------------------------------
# Atmosphere (ISA sea level, used unless an analysis overrides it)
# ---------------------------------------------------------------------------
AIR_DENSITY_KG_M3 = 1.225
KINEMATIC_VISCOSITY_M2_S = 1.460e-5
GRAVITY_M_S2 = 9.80665

# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------
N_SPAN_STATIONS = 200

# ---------------------------------------------------------------------------
# Base airfoil family
# ---------------------------------------------------------------------------
BASE_AIRFOIL_NAME = "mh64"

# ---------------------------------------------------------------------------
# Structural proxy (fast ranking estimate, NOT certification-grade -- see
# analysis/structures.py). A simple thin-walled rectangular spar box is
# assumed inside the airfoil's own thickness/chord envelope.
# ---------------------------------------------------------------------------
SPAR_WIDTH_FRACTION_CHORD = 0.12
SPAR_DEPTH_FRACTION_THICKNESS = 0.85  # leaves margin for skin/D-box above and below the spar
SPAR_WALL_THICKNESS_M = 0.0015
ALLOWABLE_SPAR_STRESS_PA = 250e6  # generic unidirectional carbon fiber spar cap, margin already included
DESIGN_LOAD_FACTOR_G = 8.0  # target limit load factor for loops/flips/rolls/high-g turns

# Torsion + tip-deflection deep-analysis extras (analysis/structures.py's
# analyze_torsion_and_deflection) -- final-design-only, not evaluated per
# optimizer candidate. The elastic/torsion axis location defaults to the
# MH64 profile's own max-thickness point (geometry.airfoil_family.
# max_thickness_x_over_c), not a fixed fraction here; AERODYNAMIC_CENTER_X_
# FRACTION_CHORD (quarter-chord, thin-airfoil-theory default) combined with
# it gives the AC-to-spar moment arm that's the dominant torsion driver on a
# swept wing. SPAR_YOUNG_MODULUS_PA matches a realistic standard-modulus
# unidirectional carbon fiber/epoxy spar cap (~120 GPa along the fiber
# direction), consistent with ALLOWABLE_SPAR_STRESS_PA's "generic
# unidirectional carbon fiber" assumption.
AERODYNAMIC_CENTER_X_FRACTION_CHORD = 0.25
SPAR_YOUNG_MODULUS_PA = 120e9
ALLOWABLE_SPAR_SHEAR_STRESS_PA = 100e6

# ---------------------------------------------------------------------------
# Stage 2 (planform) constraints and search bounds
# ---------------------------------------------------------------------------
MIN_ABSOLUTE_THICKNESS_M = 0.003  # thinnest realistic local thickness anywhere along the span
MIN_SPAR_DEPTH_M = 0.0025  # thinnest realistic available spar depth anywhere along the span
MAX_LE_CURVATURE_PER_M = 50.0  # bounds how tight the leading-edge curve may bend (1/radius of curvature)
MAX_Z_CURVATURE_PER_M = 50.0  # same idea as MAX_LE_CURVATURE_PER_M, but for the vertical (winglet) curve z_offset(y)

SWEEP_DEG_BOUNDS = (10.0, 45.0)

# Search bounds on the ACTUAL chord (m) / twist (deg) value at each planform
# control station -- one (lo, hi) pair per station, in the same units as the
# quantity itself, so they're directly interpretable ("chord at this station
# must be between X and Y") rather than a rate-of-change value. Each accepts
# either a single (lo, hi) pair (broadcast to every station) or a list of
# one (lo, hi) pair per station for asymmetric control (e.g. a
# tighter/larger root) -- see vector.resolve_per_station_bounds.
#
# Internally (optimization/stage2.py), chord and twist are still built as a
# root value + non-negative per-segment deltas, so monotonicity holds by
# construction instead of needing independent per-station sampling to land
# on a monotonic sequence by chance (7 independent draws are correctly
# ordered with probability ~1/7!, which was wasting nearly every
# hierarchical-search candidate on a constraint-penalty violation instead of
# exploring genuinely different, valid designs). make_stage2_parameter_set
# derives each segment's decrement Var bounds from the two stations it
# connects: e.g. for chord, the maximum possible decrement over a segment is
# max(0, chord_hi[i] - chord_lo[i+1]) -- the largest drop reachable from
# station i's ceiling to station i+1's floor. (LE/Z offset use the same
# root-plus-per-segment-slope construction, but their slope bounds are set
# directly -- see LE_OFFSET_SLOPE_M_PER_SPAN_BOUNDS below -- rather than
# derived from a per-station absolute range.)
#
# Chord/twist default bounds are explicit per-station lists (not a single
# broadcast pair) shaped roughly around the default Planform's own taper, so
# the *derived* decrement bounds stay reasonable by default -- a single wide
# bound broadcast to every station (e.g. tip and root both (0.03, 0.9))
# would let a single segment's decrement span nearly the whole chord range.
# Tied to len(DEFAULT_PLANFORM_Y_CONTROL) = 7; override via
# bounds_overrides.yaml (a list of 7 pairs, or a single broadcast pair) if
# the station count changes.
CHORD_STATION_M_BOUNDS = [
    (0.4, 0.9), (0.2, 0.75), (0.15, 0.6), (0.15, 0.55), (0.08, 0.35), (0.05, 0.28), (0.03, 0.15),
]
TWIST_STATION_DEG_BOUNDS = [
    (-3.0, 3.0), (-5.0, 3.0), (-6.0, 2.0), (-7.0, 2.0), (-9.0, 1.0), (-11.0, 1.0), (-15.0, 1.0),
]
# LE/Z offset's ROOT (station 0) value only -- everywhere else along the
# span, LE/Z offset is governed directly by *_OFFSET_SLOPE_M_PER_SPAN_BOUNDS
# below, not by a per-station absolute range (unlike chord/twist above,
# which really do have a meaningful bound at every station).
LE_OFFSET_ROOT_M_BOUNDS = (-0.15, 0.5)
Z_OFFSET_ROOT_M_BOUNDS = (-0.05, 0.35)

# Explicit, directly-editable per-segment (not per-station -- n-1 of them)
# bounds on LE/Z offset's slope (m of offset per unit of normalized span y,
# "m/span"), one pair per gap between consecutive y_control stations.
# Replaces an earlier version that *derived* these from per-station absolute
# ranges plus a single global slope cap plus a special-cased non-negative
# floor for whichever segments fell in an assumed "winglet region" -- that
# mechanism was only a soft bias (verified: sampling every segment at its
# own lower bound could still produce a fully drooped tip, contradicting
# what "min winglet height" implied) and its single global slope cap applied
# uniformly to every segment, including ones meant to form a tight winglet
# bend, capping the local bend angle at ~50 deg regardless of intent. Direct
# per-segment control lets a design (or a search) explicitly permit a much
# steeper bend in just the outboard segments without loosening every other
# segment's smoothness at the same time.
# Defaults below are a snapshot of what the old derivation produced for the
# default 7-station baseline (so a fresh checkout searches essentially the
# same space as before) -- override via bounds_overrides.yaml (a list of 6
# pairs, or a single broadcast pair) if the station count changes.
LE_OFFSET_SLOPE_M_PER_SPAN_BOUNDS = [
    (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), (-1.413, 1.413), (-1.5, 1.5), (-1.5, 1.5),
]
Z_OFFSET_SLOPE_M_PER_SPAN_BOUNDS = [
    (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-0.87, 0.87), (0.0, 1.0), (0.0, 1.0),
]

# ---------------------------------------------------------------------------
# Optional overrides: if configs/bounds_overrides.yaml exists (e.g. saved by
# the GUI's Bounds & Weights tab), it replaces any of the constants named
# below. Absent by default, so a fresh checkout uses exactly the hardcoded
# values above. See _overrides.py.
# ---------------------------------------------------------------------------
from . import _overrides as _ov

BOUNDS_OVERRIDES_YAML = CONFIG_DIR / "bounds_overrides.yaml"

_CONFIG_OVERRIDABLE_NAMES = {
    "FUSELAGE_MIN_INTERNAL_WIDTH_M", "FUSELAGE_MIN_INTERNAL_HEIGHT_M", "FUSELAGE_MIN_INTERNAL_LENGTH_M",
    "SPAR_WIDTH_FRACTION_CHORD", "SPAR_DEPTH_FRACTION_THICKNESS", "SPAR_WALL_THICKNESS_M",
    "ALLOWABLE_SPAR_STRESS_PA", "DESIGN_LOAD_FACTOR_G",
    "AERODYNAMIC_CENTER_X_FRACTION_CHORD", "SPAR_YOUNG_MODULUS_PA", "ALLOWABLE_SPAR_SHEAR_STRESS_PA",
    "MIN_ABSOLUTE_THICKNESS_M", "MIN_SPAR_DEPTH_M", "MAX_LE_CURVATURE_PER_M", "MAX_Z_CURVATURE_PER_M",
    "WINGSPAN_MIN_M", "WINGSPAN_MAX_M",
    "SWEEP_DEG_BOUNDS",
    "CHORD_STATION_M_BOUNDS", "TWIST_STATION_DEG_BOUNDS", "LE_OFFSET_ROOT_M_BOUNDS", "Z_OFFSET_ROOT_M_BOUNDS",
    "LE_OFFSET_SLOPE_M_PER_SPAN_BOUNDS", "Z_OFFSET_SLOPE_M_PER_SPAN_BOUNDS",
}
_ov.apply_overrides(globals(), BOUNDS_OVERRIDES_YAML, _CONFIG_OVERRIDABLE_NAMES)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
