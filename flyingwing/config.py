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
# optimizer candidate. SPAR_X_FRACTION_CHORD is the assumed elastic/torsion
# axis location; AERODYNAMIC_CENTER_X_FRACTION_CHORD (quarter-chord, thin-
# airfoil-theory default) combined with it gives the AC-to-spar moment arm
# that's the dominant torsion driver on a swept wing. SPAR_YOUNG_MODULUS_PA
# is a generic composite spar-cap laminate stiffness (consistent in spirit
# with ALLOWABLE_SPAR_STRESS_PA's "generic unidirectional carbon fiber"
# assumption).
SPAR_X_FRACTION_CHORD = 0.30
AERODYNAMIC_CENTER_X_FRACTION_CHORD = 0.25
SPAR_YOUNG_MODULUS_PA = 70e9
ALLOWABLE_SPAR_SHEAR_STRESS_PA = 100e6

# ---------------------------------------------------------------------------
# Stage 2 (planform) constraints and search bounds
# ---------------------------------------------------------------------------
MIN_ABSOLUTE_THICKNESS_M = 0.003  # thinnest realistic local thickness anywhere along the span
MIN_SPAR_DEPTH_M = 0.0025  # thinnest realistic available spar depth anywhere along the span
MAX_LE_CURVATURE_PER_M = 50.0  # bounds how tight the leading-edge curve may bend (1/radius of curvature)
MAX_Z_CURVATURE_PER_M = 50.0  # same idea as MAX_LE_CURVATURE_PER_M, but for the vertical (winglet) curve z_offset(y)

SWEEP_DEG_BOUNDS = (10.0, 45.0)

# Search bounds on the ACTUAL value (chord in m, twist in deg, LE offset
# deviation in m, Z offset in m) at each planform control station -- one
# (lo, hi) pair per station, in the same units as the quantity itself, so
# they're directly interpretable ("chord at this station must be between X
# and Y") rather than the rate-of-change values previously used here
# (chord decrement/washout increment per segment, LE/Z offset slope per
# segment). Each accepts either a single (lo, hi) pair (broadcast to every
# station, the default below) or a list of one (lo, hi) pair per station
# for asymmetric control (e.g. a tighter/larger root) -- see
# vector.resolve_per_station_bounds.
#
# Internally (optimization/stage2.py), chord and twist are still built as a
# root value + non-negative per-segment deltas, and LE/Z offset as a root
# value + a per-segment slope, so monotonicity (chord/twist) and bounded
# curvature (LE/Z offset) still hold by construction -- independent
# per-station sampling was measured to have almost no chance of landing on
# a monotonic sequence (7 independent draws are correctly ordered with
# probability ~1/7!), wasting nearly every hierarchical-search candidate on
# a constraint-penalty violation instead of exploring genuinely different,
# valid designs. make_stage2_parameter_set derives each segment's
# decrement/slope Var bounds from the two stations it connects: e.g. for
# chord, the maximum possible decrement over a segment is
# max(0, chord_hi[i] - chord_lo[i+1]) -- the largest drop reachable from
# station i's ceiling to station i+1's floor.
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
LE_OFFSET_STATION_M_BOUNDS = (-0.15, 0.5)
Z_OFFSET_STATION_M_BOUNDS = (-0.05, 0.35)

# LE/Z offset are slope-based (see above), and a slope bound *derived*
# purely from two stations' absolute ranges blows up when those stations
# are close together in y (e.g. the 0.08/0.12/0.14 fuselage-break cluster)
# -- exactly the "huge jump over a tiny span fraction" that blows up LE
# curvature. These cap the derived slope regardless of station spacing,
# same role the old LE_OFFSET_SLOPE_BOUNDS/Z_OFFSET_SLOPE_BOUNDS played.
MAX_LE_OFFSET_SLOPE_M_PER_SPAN = 1.5
MAX_Z_OFFSET_SLOPE_M_PER_SPAN = 1.0

# The tip station's Z offset lower bound is floored at this value
# (regardless of Z_OFFSET_STATION_M_BOUNDS above) for every station at or
# beyond Z_OFFSET_TIP_SEGMENT_Y_THRESHOLD, so Stage 2 stays biased toward an
# upturned, winglet-like tip instead of being equally likely to droop --
# same intent as the old Z_OFFSET_TIP_SLOPE_BOUNDS, just expressed as a
# directly-interpretable minimum tip height instead of a minimum slope.
Z_OFFSET_TIP_MIN_M = 0.05
Z_OFFSET_TIP_SEGMENT_Y_THRESHOLD = 0.85

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
    "SPAR_X_FRACTION_CHORD", "AERODYNAMIC_CENTER_X_FRACTION_CHORD", "SPAR_YOUNG_MODULUS_PA", "ALLOWABLE_SPAR_SHEAR_STRESS_PA",
    "MIN_ABSOLUTE_THICKNESS_M", "MIN_SPAR_DEPTH_M", "MAX_LE_CURVATURE_PER_M", "MAX_Z_CURVATURE_PER_M",
    "WINGSPAN_MIN_M", "WINGSPAN_MAX_M",
    "SWEEP_DEG_BOUNDS",
    "CHORD_STATION_M_BOUNDS", "TWIST_STATION_DEG_BOUNDS", "LE_OFFSET_STATION_M_BOUNDS", "Z_OFFSET_STATION_M_BOUNDS",
    "Z_OFFSET_TIP_MIN_M", "Z_OFFSET_TIP_SEGMENT_Y_THRESHOLD",
    "MAX_LE_OFFSET_SLOPE_M_PER_SPAN", "MAX_Z_OFFSET_SLOPE_M_PER_SPAN",
}
_ov.apply_overrides(globals(), BOUNDS_OVERRIDES_YAML, _CONFIG_OVERRIDABLE_NAMES)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
