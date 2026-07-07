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

# ---------------------------------------------------------------------------
# Stage 2 (planform) constraints and search bounds
# ---------------------------------------------------------------------------
MIN_ABSOLUTE_THICKNESS_M = 0.003  # thinnest realistic local thickness anywhere along the span
MIN_SPAR_DEPTH_M = 0.0025  # thinnest realistic available spar depth anywhere along the span
MAX_LE_CURVATURE_PER_M = 50.0  # bounds how tight the leading-edge curve may bend (1/radius of curvature)

SWEEP_DEG_BOUNDS = (10.0, 45.0)
CHORD_M_BOUNDS = (0.03, 0.9)
TWIST_DEG_BOUNDS = (-15.0, 3.0)
LE_OFFSET_DEVIATION_M_BOUNDS = (-0.15, 0.35)
Z_OFFSET_M_BOUNDS = (-0.05, 0.30)

# chord and twist are optimized as root value + non-negative per-segment
# deltas (see optimization/stage2.py) so monotonicity holds by
# construction, rather than as independent per-station values -- with
# independent bounds, a random sample has almost no chance of landing on a
# monotonic sequence by chance (7 independent draws are correctly ordered
# with probability ~1/7! for 7 stations), which was making the hierarchical
# search waste nearly every candidate on a constraint-penalty violation
# instead of exploring genuinely different, valid designs.
CHORD_DECREMENT_M_BOUNDS = (0.0, 0.2)  # per span-station segment
WASHOUT_INCREMENT_DEG_BOUNDS = (0.0, 6.0)  # per span-station segment
TWIST_ROOT_DEG_BOUNDS = (-3.0, 3.0)

# The root chord specifically needs to be large enough to have any chance of
# satisfying the fuselage box (given a typical root thickness ratio) --
# CHORD_M_BOUNDS' 0.03 m floor is appropriate for the tip, not the root.
CHORD_ROOT_M_BOUNDS = (0.4, 0.9)

# LE offset deviation and z offset are free-form (non-monotonic) curves, so
# they can't use the same "root + non-negative decrement" trick as chord/
# twist. But independent per-station values have the same underlying
# problem: some planform control stations are very close together (the
# 0.08/0.12/0.14 fuselage-break cluster), so an independent random draw at
# each one creates huge slope changes over a tiny span fraction -- which
# is exactly what blows up leading-edge curvature. Parameterizing as a root
# value + a bounded *slope* per segment (value[i] = value[i-1] +
# slope[i]*(y[i]-y[i-1])) bounds the slope directly, which keeps curvature
# in check by construction instead of by chance.
LE_OFFSET_SLOPE_BOUNDS = (-1.5, 1.5)  # m per unit normalized span
Z_OFFSET_SLOPE_BOUNDS = (-1.0, 1.0)  # m per unit normalized span

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
    "MIN_ABSOLUTE_THICKNESS_M", "MIN_SPAR_DEPTH_M", "MAX_LE_CURVATURE_PER_M",
    "WINGSPAN_MIN_M", "WINGSPAN_MAX_M",
    "SWEEP_DEG_BOUNDS", "CHORD_M_BOUNDS", "TWIST_DEG_BOUNDS",
    "LE_OFFSET_DEVIATION_M_BOUNDS", "Z_OFFSET_M_BOUNDS",
    "CHORD_DECREMENT_M_BOUNDS", "WASHOUT_INCREMENT_DEG_BOUNDS", "TWIST_ROOT_DEG_BOUNDS", "CHORD_ROOT_M_BOUNDS",
    "LE_OFFSET_SLOPE_BOUNDS", "Z_OFFSET_SLOPE_BOUNDS",
}
_ov.apply_overrides(globals(), BOUNDS_OVERRIDES_YAML, _CONFIG_OVERRIDABLE_NAMES)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
