# Flying Wing Aircraft Optimization Framework

A Python framework for automated design and optimization of a high-performance
FPV flying wing (1.4-1.8 m span), built around a continuously varying
MH64-derived airfoil family. Given a starting design, it can evaluate its
aerodynamics/structure/mass, score it, and search for better designs —
either the airfoil schedule (Stage 1), the planform (Stage 2), or both in
alternation (multi-cycle).

This README is meant to be enough on its own to run the framework, understand
what it's doing, change what it optimizes for, and know what's still rough.

## Table of contents

1. [Status](#status)
2. [Quick start](#quick-start)
3. [How it works (pipeline overview)](#how-it-works-pipeline-overview)
4. [The aircraft representation](#the-aircraft-representation)
5. [Geometry generation](#geometry-generation)
6. [Analysis modules](#analysis-modules)
7. [Stability, center of gravity, and performance estimates](#stability-center-of-gravity-and-performance-estimates)
8. [The objective function](#the-objective-function)
9. [Optimization](#optimization)
10. [Visualization](#visualization)
11. [The interactive GUI](#the-interactive-gui)
12. [How to change parameters, bounds, and weights](#how-to-change-parameters-bounds-and-weights)
13. [Code structure](#code-structure)
14. [Simplifications and known limitations](#simplifications-and-known-limitations)
15. [What's not implemented / next steps](#whats-not-implemented--next-steps)

---

## Status

Every module in the original project spec is implemented, plus several
follow-on rounds covering CG/stability, endurance estimates, a consolidated
deep-analysis report, Cp-based flow visualization, STL export, a structural
torsion/deflection check, a geometry-derived (not fixed-fraction) CG/mass
model, objective-function normalization, three new optional objective terms
(soaring power, glide angle, roll stability), a second optimization algorithm
(CMA-ES, now the default), and a cheap pre-build validity filter. Everything
has been run end-to-end at least once, including real (non-trivial)
optimization runs and a head-to-head CMA-ES-vs-Latin-Hypercube benchmark.

**What's still not done is tuning** — the default objective weights and
search bounds are reasonable first guesses, not calibrated values. A real
correctness bug in the fuselage-fit check was found and fixed early on (see
[Simplifications and known limitations](#simplifications-and-known-limitations)):
the check now does a real box-placement search rather than an earlier,
buggier version, and under that corrected check, the built-in default
`DesignParameters()` baseline (`geometry/params.py`) currently comes up short
of the required internal height — a valid design *is* reachable within the
current bounds (a real CMA-ES run from this baseline finds one — see
[Optimization](#optimization)), but the shipped default itself is not valid
out of the box. Expect (and plan for) a weight/bound/default-baseline tuning
pass.

| Module | Status |
|---|---|
| Geometry generation (airfoil family, aircraft generator, mesh, fuselage fit) | Done |
| Geometry visualization (3D, orthographic, airfoil distribution, fuselage box) | Done |
| 2D aero analysis (NeuralFoil) | Done |
| 2D aero validation (XFoil) | Wired, optional, not required (no XFoil binary needed) |
| 3D aero analysis (AeroBuildup, VLM) | Done, VLM has a resolution-robust fallback (see below) |
| 3D aero validation (AVL) | Wired, optional, not required |
| Structural proxy (Schrenk + spar sizing + torsion/deflection) | Done |
| CG / mass model | Done -- component positions geometry-derived (motor/ESC/servo placement rules, avionics/battery in the real fuselage box, shell/spar centroids from the actual MH64 profile), not fixed chord fractions |
| Performance estimates (glide ratio/angle, endurance/range) | Done |
| Aero + structural + flow visualization | Done |
| Objective function | Done, normalized (see below), weights not tuned |
| Optimizers: CMA-ES (default) + Latin-Hypercube hierarchical search (selectable) | Both done, benchmarked against each other |
| Cheap pre-build validity filter (`quick_reject_reason`) | Done -- skips build_aircraft()+AeroBuildup+NeuralFoil for definitely-invalid candidates |
| Stage 1 (airfoil) optimization | Done, verified |
| Stage 2 (planform) optimization | Done, verified; bounds are per-station absolute values (chord/twist) and per-segment slopes (LE/Z offset) |
| Multi-cycle Stage1↔Stage2 driver | Done, verified |
| Interactive Dash GUI (6 tabs) | Done, verified via real HTTP callback round-trips |
| STL export | Done |

---

## Quick start

### Get the code

```
git clone https://github.com/Snowwolf11/flying-wing-optimizer.git
cd flying-wing-optimizer
```

No `git`, or just want a copy? Use GitHub's green **Code → Download ZIP**
button on the repo page instead, then unzip and `cd` into the resulting
folder.

### Install

Requires Python 3.11+. Everything here is pure Python — no OS-specific code
— but the virtual environment itself is *not* portable between machines, so
create a fresh one rather than copying someone else's `.venv` folder:

**Windows (PowerShell or cmd):**
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux:**
```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If any exact pinned version in `requirements.txt` fails to resolve on your
platform (it's a `pip freeze` snapshot taken on Windows), fall back to the
unpinned dependency list instead, which lets pip pick current
platform-compatible versions:

```
.venv/bin/python -m pip install -e .          # macOS/Linux
.venv\Scripts\python.exe -m pip install -e .  # Windows
```

Every command shown elsewhere in this README uses the Windows
`.venv\Scripts\python.exe` form for concreteness — on macOS/Linux, substitute
`.venv/bin/python` (and `/` for any `\` in a script path, e.g.
`scripts/run_gui.py`).

### Run an optimization demo

```
.venv\Scripts\python.exe scripts\run_stage1.py        # optimize airfoil schedule
.venv\Scripts\python.exe scripts\run_stage2.py         # optimize planform
.venv\Scripts\python.exe scripts\run_multi_cycle.py    # alternate both
```

Each writes plots to `outputs/<name>_run/*.html` (open directly in a browser)
and a `result.pkl` containing the full optimization result + best design, so
you can reload and re-plot without re-running the optimization (see
[Code structure](#code-structure)). Every argument is optional — e.g.
`--optimizer` (`cma`, the default, or `lhs`), `--cma-max-generations`,
`--cma-population-size`, `--cma-n-restarts`, `--n-stages`/
`--n-samples-per-stage` (the `lhs` equivalents), `--n-jobs`, `--seed`,
`--output-dir-name`, `--weights-yaml`, `--baseline-yaml` — run with `--help`
to see a script's full list.

**On timing**: one `evaluate_design()` call costs ~10 s on a typical machine
(AeroBuildup + NeuralFoil dominate; see
[How it works](#how-it-works-pipeline-overview)), *unless* the cheap
pre-build filter catches an obviously-invalid candidate first, which takes
~0.1 ms (see [Optimization](#optimization)). A full run's wall time is
therefore mostly `(unrejected candidates) × ~10 s / n_jobs`, which varies a
lot by design/bounds/machine — use the GUI Run tab's **Estimate Time**
button (or just start with a small `--cma-max-generations`/`--n-samples-per-stage`)
rather than trusting a fixed number here.

### Run the interactive GUI

```
.venv\Scripts\python.exe scripts\run_gui.py
```

Then open **http://127.0.0.1:8050** in a browser. It's a full control panel,
not just a design viewer — see [The interactive GUI](#the-interactive-gui):
edit any design parameter (including adding/removing control stations) or
search bound/weight, launch Stage 1/2/multi-cycle runs as background jobs,
browse past results, run a consolidated deep-analysis report (score
breakdown, mass/CG, structural detail, Cp flow visualization) on any past run,
export an STL, and read this README in-app — all from one page.

### Run a single design evaluation from Python

```python
from flyingwing.geometry.params import default_design_parameters
from flyingwing.objective.metrics import evaluate_design
from flyingwing.objective.objective import score

params = default_design_parameters()
metrics = evaluate_design(params)   # ~10 seconds (or ~0.1 ms if quick_reject_reason catches it first)
result = score(metrics)
print(result.score, metrics.cruise_L_over_D, metrics.min_safety_factor)
```

---

## How it works (pipeline overview)

```
DesignParameters  --(geometry)-->  Aircraft  --(analysis)-->  metrics  --(objective)-->  score
       ^                                                                                     |
       |                                                                                     v
       +----------------------------------  Optimizer proposes a new vector  <---------------+
```

- **`DesignParameters`** (`geometry/params.py`) is the complete, self-contained
  description of one aircraft: a `Planform` + an `AirfoilSchedule`. It's pure
  data — no knowledge of analysis or optimization.
- **`build_aircraft()`** (`geometry/aircraft.py`) turns `DesignParameters` into
  an `Aircraft`: a ~200-station spanwise mesh, an AeroSandbox `Airplane`, and
  derived geometric properties (wing area, aspect ratio, MAC, ...).
- **`evaluate_design()`** (`objective/metrics.py`) is the one function that
  wires together geometry → 2D/3D aerodynamics → structural proxy → mass
  estimate → CG/static-margin estimate, and returns a flat `DesignMetrics`
  object. This is also the one function called on every optimizer candidate.
  Its first step is a **cheap pre-build validity filter**
  (`geometry.constraints.quick_reject_reason`): a handful of checks on the
  raw `DesignParameters` control points alone (chord/twist monotonicity,
  LE/Z curvature, a fuselage-fit proxy) that can only reject a candidate the
  full check below would *also* reject, never one it would have accepted —
  a candidate caught here returns in ~0.1 ms instead of paying for
  `build_aircraft()` + AeroBuildup + NeuralFoil (~10 s, the dominant cost for
  everything that isn't caught early). Higher-fidelity checks (finer VLM,
  hybrid viscous drag, torsion/deflection) run separately, only on-demand for
  a finished design (see
  [Stability, center of gravity, and performance estimates](#stability-center-of-gravity-and-performance-estimates)
  and the GUI's Deep Analysis tab).
- **`score()`** (`objective/objective.py`) combines `DesignMetrics` into one
  scalar via `ObjectiveWeights`.
- **Optimizers** (`optimization/`) never see `DesignParameters` directly —
  they see a flat numeric vector and bounds (via `ParameterSet`, see
  [Optimization](#optimization)), propose new vectors, and call
  `evaluate_design` + `score` on each. This keeps geometry, analysis, and
  optimization fully decoupled, per the original design goal.

---

## The aircraft representation

The aircraft is always a symmetric flying wing, described along a normalized
span coordinate `y ∈ [0, 1]` (`y=0` = symmetry plane/root, `y=1` = wing tip).
Every spanwise quantity — chord, twist, thickness, camber, reflex, leading-edge
position, vertical offset — is a function of `y`, built by interpolating a
handful of **control points** (`geometry/spanwise.py`):

- **Airfoil schedule** (`thickness_scale`, `camber_scale`, `reflex_scale`):
  interpolated **linearly**. This is deliberate — robust, predictable, cheap,
  and sufficient given the ~200-station resolution used downstream.
- **Planform** (`chord`, `twist`, LE offset, Z offset): interpolated with
  **PCHIP** (shape-preserving cubic Hermite). This gives one smooth curve
  through the control points instead of a piecewise-linear, kinked one, and
  — because it's shape-preserving — a PCHIP curve through monotonic control
  points is guaranteed to stay monotonic (no overshoot between points).

Default control stations:
- Airfoil: `y = (0.00, 0.09, 0.14, 0.60, 1.00)` — root, fuselage/wing
  transition (paired stations), mid-span, tip.
- Planform: `y = (0.00, 0.08, 0.12, 0.14, 0.60, 0.85, 1.00)` — nose apex,
  fuselage-end/wing-start cluster, mid-span, and two tip stations for the
  raked/blended tip region.

**Winglets are not a separate part.** There's no winglet parameter. A winglet
(or gull wing, or blended centre body) emerges from `chord(y)`, `z_offset(y)`,
and the leading-edge curve tapering/curving near the tip — the same
continuous functions used for the rest of the wing, so there's never a
discontinuity where a "winglet" would begin.

**Leading edge parameterization**: `le_offset(y) = reference(y) + deviation(y)`,
where `reference(y) = y · (span/2) · tan(sweep_deg)` is a straight sweep line
from the single global `sweep_deg` scalar, and `deviation(y)` is a per-station
local curve on top of it. This is why `Planform` has both a `sweep_deg` field
(matching the spec's "global sweep angle" variable) and a full
`le_offset_deviation_m` curve (the spec's "leading edge offset(y)").

---

## Geometry generation

### Airfoil family (`geometry/airfoil_family.py`)

Only one airfoil family is used, everywhere: MH64, decomposed into a camber
line and a thickness envelope (via a cosine-spaced resample of the base
AeroSandbox `mh64` coordinates). Each section airfoil is:

```
thickness(x) = thickness_base(x) · thickness_scale
camber(x)    = camber_base(x) · camber_scale + reflex_scale · REFLEX_NOMINAL_AMPLITUDE · bump(x)
```

`bump(x)` is a smoothstep function, zero below `x=0.55` (55% chord) and
ramping to 1 at the trailing edge — `reflex_scale` controls an *additional*
aft-loading/reflex deflection on top of whatever reflex the base MH64 shape
already has, so the amount of self-stabilizing reflex can be tuned
independently per span station. `REFLEX_NOMINAL_AMPLITUDE = 0.02` (2% chord)
is the deflection at `reflex_scale = 1.0`.

### Aircraft generator (`geometry/aircraft.py`)

`build_aircraft(params)`:
1. Evaluates all spanwise distributions at ~200 cosine-spaced stations
   (concentrated near the root and tip, where geometry changes fastest).
2. Generates each station's 3D airfoil surface (scaled by chord, rotated by
   twist, translated to its LE/Z position) → the fine visualization mesh.
3. Separately builds a **coarser** (41-station, linearly-spaced) AeroSandbox
   `Airplane` for analysis. This is intentionally a different resolution
   than the visualization mesh — see the note on VLM below.
4. Runs the fuselage-box fit check and returns everything bundled as an
   `Aircraft` dataclass, along with derived properties (`wing_area_m2`,
   `aspect_ratio`, `mean_aerodynamic_chord_m`, ...).

### Watertight mesh (`geometry/mesh.py`)

The visualization mesh mirrors the half-wing, welds the leading/trailing edge
seams (upper and lower surfaces are numerically coincident there, so they're
welded to one shared vertex instead of left as separate-but-coincident
points), and caps both wingtips. The result is a genuine closed 2-manifold —
verified by checking every edge is shared by exactly 2 faces.

### Fuselage fit (`geometry/fuselage.py`)

The fuselage isn't modeled as separate geometry — it's a required internal
box (`FUSELAGE_MIN_INTERNAL_WIDTH/HEIGHT/LENGTH_M` in `config.py`, currently
140 × 55 × 300 mm) that must fit inside the wing's own upper/lower surface
envelope near the root, centered on the symmetry plane (width is fixed by
symmetry, not searched).

`check_fuselage_fit()` actually **searches for a valid box placement**: it
sweeps the box's chordwise window `[x0, x0+length]` across a grid of
candidate `x0` values and, at each candidate, requires that window to fall
entirely within *every* footprint span station's actual chord **and** that
the local upper/lower surface gap within that window is at least the
required height at *every* one of those stations simultaneously — not just
"enough thickness somewhere along the chord" checked independently of "chord
long enough somewhere," which was an earlier, buggier version of this check
(see [Simplifications and known limitations](#simplifications-and-known-limitations)).
The best-margin placement found is reused directly by the geometry
visualization to actually draw the box in the right place (previously the
plot guessed a position independently of the check, which is part of why it
used to render outside the aircraft for a lot of designs). `fuselage_fit`
(a `FuselageFitResult`) carries both the resulting margins and the winning
`box_x_min_m`/`box_x_max_m`/`box_z_min_m`/`box_z_max_m` — also reused
directly by `objective/cg.py` to place avionics and the battery (see
[Stability, center of gravity, and performance estimates](#stability-center-of-gravity-and-performance-estimates))
and by `geometry/constraints.py::quick_reject_reason`'s cheap root-station
proxy for this same check (see [Optimization](#optimization)).

---

## Analysis modules

### 2D airfoil analysis (`analysis/airfoil_2d.py`)

Primary tool: **NeuralFoil** (via AeroSandbox's `Airfoil.get_aero_from_neuralfoil`).
`evaluate_section()` runs an alpha sweep and reduces it to: `CLmax` (+ its
alpha), lift-curve slope, zero-lift alpha, zero-lift `Cm` (important for a
flying wing's self-trim behavior), drag bucket, minimum `CD`, and stall
sharpness (slope just past `CLmax` — very negative = abrupt stall).

`N_CRIT_ASSUMPTIONS` (clean/moderate/rough, `n_crit` 9/7/4) are available for
sweeping different transition/turbulence assumptions, matching the spec's
"several transition assumptions," though `evaluate_design` currently only
evaluates at the default (`n_crit=9`, clean).

**XFoil** validation (`validate_with_xfoil`) is wired via AeroSandbox's `XFoil`
wrapper but requires an XFoil executable on the system; if it's not present,
the function catches the error, warns, and returns `None` rather than
crashing. Not required for anything else to work.

### 3D aircraft analysis (`analysis/aero_3d.py`)

**AeroBuildup** (strip theory + NeuralFoil per station) is primary — fast
(~2 s per call) and used for everything the objective function needs:
trimmed CL/CD/L-over-D at cruise and top speed, stability derivatives
(`CLa`, `Cma`), neutral point, and a linearized (single Newton step) trim
solve.

**VLM** (vortex lattice) is a slower cross-check, and — important — it builds
its **own separate, much coarser** (13-station) `Airplane` rather than
reusing the 41-station one AeroBuildup uses. Feeding VLM the same
finer/cosine-spaced geometry used elsewhere was found to pack
near-duplicate, near-zero-width panels close to the root/tip, making the
AIC matrix ill-conditioned enough that the solution blew up to nonsense
(~1e22-magnitude coefficients) and took 3+ minutes. The dedicated coarse
geometry converges in ~5 s and its CL/CLa/Cma/neutral-point agree closely
with AeroBuildup's — a useful independent cross-check.

**Panel-count safety turned out to be geometry-dependent, not a fixed
threshold.** For deep-analysis use (see below), `analyze_vlm_robust()` tries
a short list of progressively coarser candidate resolutions (21 → 17 → 13
stations) and keeps the first one whose result passes a physical-plausibility
check (`|CL| < 3`, `0 < CD < 1`), falling back to the original 13-station
default (empirically the most robust) if every finer candidate fails —
verified against a real optimizer-produced design where 13 stations gave a
sane result but 17 and 21 both blew up (finer was *worse* for that specific
geometry, even though 21 stations tested fine on a different, more
conservative design). Returns `None` (not a silently-wrong number) if even
the safest candidate fails. `analyze_hybrid_drag()` combines this robust VLM
call's induced drag with AeroBuildup's profile drag (VLM alone is inviscid)
for an independent total-drag cross-check, also `None`-safe.

**AVL** (`validate_with_avl`) is wired the same optional way as XFoil.

### Structural proxy (`analysis/structures.py`)

Not a substitute for FEA — a fast, ranking-only estimate:

1. **Schrenk's approximation** for the spanwise lift distribution: the
   average of the actual chord distribution and the elliptical distribution
   with the same span and planform area. Evaluated at `DESIGN_LOAD_FACTOR_G`
   (default 8g) by scaling the trimmed cruise `CL` by the load factor
   (`L = n·W = q·S·CL`, so at fixed speed `CL_maneuver = n · CL_trim` —
   this avoids needing a mass estimate to size the load).
2. Shear force and bending moment follow from integrating that load
   outboard-to-inboard.
3. A simple thin-walled rectangular spar box (`SPAR_WIDTH_FRACTION_CHORD` =
   12% of local chord, `SPAR_DEPTH_FRACTION_THICKNESS` = 85% of local
   thickness, `SPAR_WALL_THICKNESS_M` = 1.5 mm) converts bending moment to a
   stress proxy and a safety factor against `ALLOWABLE_SPAR_STRESS_PA`
   (250 MPa, a generic unidirectional carbon fiber spar cap allowable).

The same Schrenk lift distribution is reused (at 1g/cruise trim CL) for the
aerodynamic "lift distribution" and "CL distribution" plots, so the aero and
structural visualizations are consistent with each other rather than using
two different approximations.

**Torsion + deflection (`analyze_torsion_and_deflection`, deep-analysis-only,
not evaluated per optimizer candidate)** extends the same proxy with two more
checks:
1. **Torque** from the offset between the local aerodynamic center (assumed
   quarter-chord, `AERODYNAMIC_CENTER_X_FRACTION_CHORD`) and the assumed
   spar/elastic axis (`SPAR_X_FRACTION_CHORD`) — the dominant torsion driver
   on a swept wing. (Each section's own pitching moment is a smaller,
   secondary contributor and isn't included, to avoid extra per-station
   NeuralFoil calls in what's meant to stay a cheap proxy.)
2. **Bredt-Batho thin-walled shear stress** (treating the spar box itself as
   the torsion cell) → a torsion safety factor against
   `ALLOWABLE_SPAR_SHEAR_STRESS_PA`, and **Euler-Bernoulli bending
   deflection** via double integration of `M/EI` (using
   `SPAR_YOUNG_MODULUS_PA`) — integrated *from the clamped root outward*
   (unlike shear/moment, which integrate from the free tip inward; this
   distinction mattered — an early version integrated deflection the same
   direction as shear/moment and got the boundary condition backwards).

### Mass estimation (`objective/mass.py`)

A rough parametric estimate feeding the objective function only:
- **Shell mass** = `2 × wing_area_m2 × 0.55 kg/m²` (wetted-area proxy for
  foam-core + glass/film skin).
- **Spar mass** = spar material cross-sectional area (from the structural
  proxy's spar box — the same box, so a heavier/thicker spar shows up as
  both higher safety factor *and* higher mass, giving the objective function
  a real trade-off instead of "more strength for free") integrated over
  span × `1600 kg/m³` (generic glass/carbon laminate density).
- **Motor/ESC + avionics + servo masses** are three independent, realistic
  "high quality component" defaults — `MOTOR_ESC_MASS_KG = 0.10` (a
  ~2807-3110-size outrunner + 40-60A ESC), `AVIONICS_MASS_KG = 0.03` (FC +
  receiver + wiring), `SERVO_MASS_KG = 0.03` (2× quality digital metal-gear
  elevon servos) — replacing an earlier single `FIXED_EQUIPMENT_MASS_KG`
  lump (0.45 kg) split by tunable fractions. `MassEstimate` still exposes a
  `fixed_equipment_mass_kg` property (their sum) for anything that only
  needs the total.
- **Payload volume margin** = an ellipse-proxy internal volume of the
  centre body (integrated over the fuselage's spanwise footprint) minus the
  required fuselage box volume.

---

## Stability, center of gravity, and performance estimates

### CG / static margin (`objective/cg.py`)

`estimate_cg()` builds a real component mass+position model, evaluated on
every `evaluate_design()` call (cheap — pure array arithmetic, no extra
solver calls). Every component's *position* is now derived from the
design's own actual geometry rather than an assumed constant fraction of
root chord:

- **Motor/ESC**: fixed placement rule — `x = 0.95 × root chord`, `y = 0`
  (a rear-mounted pusher prop, the common flying-wing FPV layout). Not a
  computed position, but the *chord it's a fraction of* is this design's
  own actual root chord, so it still tracks whatever planform the design
  has.
- **Servos**: `x = 50%` of the **local** chord at `y = 0.4` (40% span) —
  interpolated from the aircraft's own `chord_m(y)`/`x_le_m(y)` arrays, not
  the root. This is a deliberate change from an earlier root-based
  placement: elevon servos physically sit near the control surfaces, not
  the center body. Displayed mirrored at ±y in the CG diagram (there are
  physically two).
- **Avionics and the battery**: no single "correct" chordwise fraction the
  way a rear motor or elevon-adjacent servos have, so both default to the
  centroid of the **real internal fuselage box**
  (`aircraft.fuselage_fit` — the actual best-fit placement search from
  [Fuselage fit](#geometry-generation), not a root-chord-footprint
  approximation).
- **Shell and spar mass centroids** come from the MH64 profile's own
  geometry, computed once in `geometry/airfoil_family.py` and cached:
  - `shell_centroid_x_over_c()` — the arc-length-weighted x/c centroid of
    the base profile's upper+lower wetted perimeter (≈0.496). Shell mass
    follows *surface length*, not enclosed area, so this — not an assumed
    fraction — is the physically correct chordwise centroid for it.
  - `max_thickness_x_over_c()` — the x/c of the profile's own maximum
    thickness (≈0.273), the standard, physically sensible spar location
    (best bending stiffness per unit spar mass). Also reused as the default
    spar/elastic axis for the torsion calculation below, so the CG model
    and the structural torsion analysis agree on where the spar actually
    is, instead of each having its own independently-tunable fraction.
  - Both are exact per-station (chord-weighted for the shell, spar-area-
    weighted for the spar) integrals along the span, not single-point
    values.
- **The battery's position is treated as the unknown**, not an input — its
  mass is fixed (`BATTERY_MASS_KG`), but `estimate_cg()` solves for the x
  *range* the battery could occupy while keeping `static_margin = (x_np -
  x_cg) / MAC` inside the target band (`ObjectiveWeights.static_margin_target`),
  clipped against the real fuselage box's x-extent — a directly actionable
  answer ("mount the battery between x=... and x=... from the nose") instead
  of an abstract number.

None of the position-fraction constants this used to expose
(`MOTOR_ESC_X_FRACTION_CHORD`, `AVIONICS_X_FRACTION_CHORD`,
`SERVO_X_FRACTION_CHORD`, `BATTERY_X_FRACTION_CHORD`,
`SHELL_CENTROID_X_FRACTION_CHORD`, `SPAR_X_FRACTION_CHORD`) are GUI-editable
any more — they're either a fixed rule (motor/servo) or computed directly
from the design's own geometry (shell/spar/fuselage), so there's nothing
meaningful left to tune per-design. `BATTERY_MASS_KG` remains editable
(Bounds & Weights tab).

`ObjectiveWeights.w_static_margin` is still 0 by default (so existing tuned
weight files don't silently change behavior) even though the metric itself
is now meaningful — worth turning on deliberately.

### Roll (lateral) stability (`analysis/aero_3d.py`, `objective/metrics.py`)

`cruise_Clb_per_rad` — the roll-due-to-sideslip ("dihedral effect")
stability derivative — comes from the same `AeroBuildup(...).run_with_stability_derivatives()`
call already made for `Cma`/`CLa`/neutral point, so it's free (no extra
solver call). Negative means roll-stable (a gust-induced sideslip rolls the
aircraft back toward wings-level); the default baseline's winglet-shaped
tips give it `Clb ≈ -0.126`. Not scored by default (see
[The objective function](#the-objective-function)).

### Performance estimates (`objective/performance.py`)

`estimate_performance()` is **not** called from `evaluate_design()`/the
optimizer loop — finding best-L/D needs its own small alpha sweep (several
extra AeroBuildup calls), so it's invoked on-demand only (GUI Design tab's
"Run Evaluation" button, Results/Deep Analysis tabs, CLI scripts' post-run
summary):

- **Best glide ratio/angle**: a small alpha sweep at cruise speed to find
  peak L/D (L/D-vs-alpha is ~speed-independent to first order), converted to
  glide angle (`atan(1/L_over_D)`) and sink rate.
- **Endurance/range**: cruise power `P = mass·g·V / (L/D · η)` against a
  battery-capacity energy budget (`BATTERY_CAPACITY_MAH`, `BATTERY_VOLTAGE_V`,
  `BATTERY_USABLE_FRACTION`, `PROPULSIVE_EFFICIENCY`) — a rough parametric
  estimate, not a full discharge-curve simulation.

---

## The objective function

### Metrics (`objective/metrics.py` — `DesignMetrics`)

One `evaluate_design(params)` call returns all of these:

| Category | Fields |
|---|---|
| Validity | `valid`, `constraint_violations` (list of human-readable strings) |
| Constraint margins (≥0 = compliant) | `fuselage_height_margin_m`, `fuselage_length_margin_m`, `tip_thickness_margin`, `thickness_monotonic_violation`, `chord_monotonic_violation`, `twist_monotonic_violation`, `min_local_thickness_margin`, `min_spar_depth_margin`, `le_curvature_violation`, `z_curvature_violation` |
| Geometry | `wing_area_m2`, `aspect_ratio`, `span_m` |
| Aero — cruise (75 km/h) | `cruise_trim_alpha_deg`, `cruise_CL`, `cruise_CD`, `cruise_L_over_D` |
| Aero — fast (150 km/h) | `fast_trim_alpha_deg`, `fast_CL`, `fast_CD`, `fast_L_over_D` |
| Soaring/glide (cruise trim, cheap proxy) | `cruise_glide_angle_deg` (`atan(1/cruise_L_over_D)`, shallower=better), `soaring_power_w` (weight × sink rate, lower=stays aloft in weaker lift) — *not* `objective/performance.py`'s more precise best-glide-alpha sweep, kept cheap since this runs on every candidate |
| Stability / CG | `cruise_Cm`, `neutral_point_x_m`, `mean_aerodynamic_chord_m`, `cg_x_m`, `static_margin` (real, CG-based — see [Stability, center of gravity, and performance estimates](#stability-center-of-gravity-and-performance-estimates)), `battery_x_min_m`, `battery_x_max_m`, `battery_range_feasible` |
| Roll stability | `cruise_Clb_per_rad` (dihedral effect; negative = stable) |
| 2D root-section | `root_cl_max`, `root_cm_zero_lift` |
| Structure (at 8g) | `min_safety_factor`, `root_bending_moment_nm` |
| Mass / payload | `total_structural_mass_kg`, `payload_volume_margin_m3` |

(Glide ratio/angle and endurance/range are deliberately *not* in `DesignMetrics`
— they're computed separately, on-demand, by `objective/performance.py`; see
above.)

### Normalization (`objective/objective.py` — `NormalizationConstants`)

Every "maximize"/"minimize" term is divided by a matching normalization
constant *before* being weighted. Without this, a term's importance is
mostly determined by its raw physical magnitude rather than its weight — e.g.
payload volume margin is `O(1e-3 m³)` while L/D is `O(10)`, so before
normalization existed, `w_payload_volume` had to be ~1000× `w_cruise_L_over_D`
just to be visible at all, and that scale factor was entangled with (and
hid) the actual relative importance the weights were supposed to express.

Each normalization constant's default is that metric's own value computed
for the project's default baseline design, so for that design every
normalized term evaluates to ~1.0 and **the weight alone sets its
contribution to the score**. (Threshold/target-range terms — safety factor,
static margin, Cm0 — already compare against a physically meaningful
threshold in native units, so a design at the threshold scores exactly 0
there regardless of units; normalizing "to 1 at the default design" doesn't
fit that pattern and isn't done for them.) Editable from the GUI's Bounds &
Weights tab (a "Normalization constants" section right below the objective
weights) or `configs/normalization.yaml` directly.

A practical consequence: if you already have a hand-tuned
`objective_weights.yaml` from before normalization existed, its weight
*numbers* need rescaling (`old_weight × the metric's own normalization
constant`) to keep the same relative behavior — a bare weight of `1000` for
payload volume, for instance, now means something wildly different than it
used to.

### Weights and scoring (`objective/objective.py` — `ObjectiveWeights`, `score()`)

Every metric maps to exactly one weighted contribution, of one of four shapes:

| Shape | Formula | Used for |
|---|---|---|
| Maximize (normalized) | `+weight × value / norm` | `cruise_L_over_D`, `fast_L_over_D`, `root_cl_max`, `payload_volume_margin_m3`, `-cruise_Clb_per_rad` (roll stability, scored on the negated value since more-negative Clb is better) |
| Minimize (normalized) | `-weight × value / norm` | `total_structural_mass_kg`, `soaring_power_w`, `cruise_glide_angle_deg` |
| Threshold (one-sided, not normalized) | `-weight × max(0, threshold - value)²` | `min_safety_factor` (no reward *above* threshold — mass already prices in extra strength), `root_cm_zero_lift` |
| Target range (not normalized) | `-weight × (max(0, lo-v)² + max(0, v-hi)²)` | `static_margin` |

Plus a **constraint penalty**: every entry in the constraint-margins table
above is normalized to a dimensionless "fraction of a characteristic
threshold" (so meters/degrees/1-per-meter/ratio quantities are comparable),
summed, and multiplied by `constraint_penalty_scale` (default 1000). If
`valid` is `False`, a flat `invalid_penalty` (default 1000) is also
subtracted. **Both of these are hard valid/invalid flags, not tunable
trade-off weights**: for a design that satisfies every constraint, both
terms are mathematically exactly zero — not just small — regardless of
`constraint_penalty_scale`'s or `invalid_penalty`'s magnitude, so an
infeasible design is essentially always ranked below a feasible one, but
once a design *is* feasible, these two terms are completely inert and every
bit of the ranking comes from the weighted terms above. `score()` also
guards against `NaN`: a pathological candidate can make an individual
metric (e.g. `cruise_glide_angle_deg` for a negative-L/D design)
mathematically undefined, and Python's `max(candidates, key=lambda c:
c.score)` never replaces a `NaN` "current best" with a later real number —
so a single `NaN`-scored candidate could otherwise silently "win" over every
valid one. `score()` converts a `NaN` total to `-inf` before returning, so
it always loses instead.

Current default weights (`configs/objective_weights.yaml`; already rescaled
for normalization per the note above):

```yaml
w_cruise_L_over_D: 7.558
w_fast_L_over_D: 3.837
w_root_cl_max: 2.761
w_mass: 2.889
w_safety_factor: 0.05
safety_factor_min: 1.5
w_static_margin: 0.0      # disabled by default -- static_margin is real, just not yet weighted
static_margin_target: [0.02, 0.15]
w_cm0: 5.0
cm0_min: -0.02
w_payload_volume: 5.727
w_soaring_power: 0.0      # disabled by default -- enable deliberately
w_flight_angle: 0.0       # disabled by default -- enable deliberately
w_roll_stability: 0.0     # disabled by default -- enable deliberately
invalid_penalty: 1000.0
constraint_penalty_scale: 1000.0
```

`score()` also returns `ObjectiveResult.contributions`, a dict of every
term's individual contribution — useful for seeing *why* one design beat
another, not just that it did.

---

## Optimization

### The `Optimizer` interface (`optimization/base.py`)

Every algorithm implements one interface:

```python
class Optimizer(ABC):
    def optimize(self, objective_fn, bounds, x0=None) -> OptimizationResult: ...
```

`objective_fn` maps a flat numpy vector to an `EvaluatedCandidate(x, score,
valid, extra)`. Two algorithms implement it — **CMA-ES (`CMAESOptimizer`,
the default)** and **the original Latin Hypercube hierarchical search
(`HierarchicalGridSearch`, kept as a selectable alternative)** — and Stage
1/2/multi-cycle drivers depend only on this interface, so either can be
swapped in without touching those drivers. `evaluate_batch()`
(`optimization/base.py`) is the shared parallel-evaluation helper both use
(`n_jobs > 1` runs a generation/stage's candidates in a `ProcessPoolExecutor`).

### `ParameterSet` (`optimization/vector.py`)

The translation layer between an optimizer's flat vector and
`DesignParameters`. A `ParameterSet` has named, bounded `Var`s and a
`build_fn(x, baseline) -> DesignParameters`. This is what keeps the geometry
module fully independent of the optimizer, per the original design goal —
optimizers never see `DesignParameters` at all.

### CMA-ES (`optimization/cmaes.py`) — the default optimizer

**Covariance Matrix Adaptation Evolution Strategy.** Chosen over the
original Latin-Hypercube search because it fits this problem well: moderate,
continuous dimensionality (Stage 1 ~15 dims, Stage 2 ~30 — squarely in
CMA-ES's well-evidenced sweet spot), no gradient available through
AeroSandbox/NeuralFoil, and — importantly — **this problem is genuinely
non-separable**: Stage 2's own parameterization builds chord/twist/LE/Z
offset as a root value plus *cumulative* per-segment deltas/slopes (see
Stage 2 below), so one variable's change cascades through every downstream
station by construction, and the structural safety-factor term is a
cumulative tip-to-root bending-moment integral coupling every station's
chord/twist together. CMA-ES's adapted covariance matrix can represent that
kind of coupling; a method that treats dimensions independently (like the
original isotropic, axis-aligned Latin-Hypercube shrink) structurally can't.
A real benchmark (~330 evaluations each, Stage 2, same baseline/weights)
confirmed this in practice: CMA-ES reached roughly double the final
objective score, and — more tellingly — **kept improving through its entire
evaluation budget while the Latin-Hypercube search plateaued at 50% of its
budget and never improved further**, consistent with the non-separability
argument above.

**What the parameters actually control** (`CMAESOptimizer`, defaults in
parentheses):
- **`sigma0`** (0.25) — the initial search radius, as a fraction of the
  (internally normalized) search space width. Think of it as how big a jump
  candidates make around the starting design before the algorithm has
  learned anything about the landscape's shape. It's not really a
  speed/quality knob the way the next two are — it shapes *where* the search
  goes early on, not *how many* evaluations it costs. Bigger = broader early
  exploration (better chance of finding a different, better region, but can
  waste evaluations wandering if the start point is already decent).
  Smaller = converges faster and more locally around the starting design,
  more likely to settle for a nearby local optimum.
- **`population_size`** (`None` = an automatic `4 + 3·ln(n_dims)` heuristic)
  — how many candidate designs get evaluated *per generation*. This is the
  direct cost lever: cost per generation = `population_size` evaluations.
  Bigger = more information per generation (more robust, less likely to be
  misled by one lucky/unlucky candidate, better for the rugged/multi-modal
  landscape here) but proportionally more expensive per generation. Leaving
  it on auto is usually right; set it explicitly to a multiple of `n_jobs`
  if you want to make full use of many parallel workers.
- **`max_generations`** (100) — a hard cap on generations *per restart*
  (see below). CMA-ES's own internal convergence check usually stops a
  restart earlier once it detects it's stagnated, so this mostly matters
  when the search *hasn't* converged yet by the cap.
- **`n_restarts`** (2) — CMA-ES is fundamentally a *local* refinement method
  once it commits to one basin, so multi-modality (this is an aircraft
  design landscape — expect multiple distinct local optima, e.g. different
  sweep/twist combinations reaching similar L/D) is handled via IPOP-style
  restarts: each restart after the first starts from a **fresh random
  point** (not the previous restart's result) with the population size
  **doubled**, and the best candidate across all restarts is kept. This is
  the single biggest lever on total run length — restart *r* costs up to
  `population_size × 2^r × max_generations`, more than either of the two
  parameters above.

Total evaluation budget is therefore `Σ population_size × 2^r ×
max_generations` for `r` in `0..n_restarts-1` — an **upper bound**, since
`es.stop()`'s own convergence criteria usually end a restart sooner.

**For a longer, more thorough search**: raise `n_restarts` first (biggest
quality lever), then `max_generations`, then consider bumping
`population_size` above auto to a multiple of `n_jobs`. **For a faster,
somewhat-worse search**: drop `n_restarts` to 1 (biggest time saving —
trades away the multi-modality safety net), then lower `max_generations`;
don't push `population_size` *below* auto, since too few candidates per
generation gives a poorly-conditioned covariance estimate rather than just
"less thorough" results. The GUI Run tab's **Estimate Time** button gives a
quick, rough wall-clock estimate for whatever's currently configured,
measured against a real timed batch rather than a guess.

### Latin Hypercube hierarchical search (`optimization/hierarchical.py`) — selectable alternative

The original spec's approach: coarse grid → evaluate all → retain best N →
refine around each → repeat. **A literal full-factorial grid is
intractable** beyond a handful of dimensions — Stage 1 alone has 15 (3
airfoil parameters × 5 span stations); even 3 points/dimension would be
`3^15 ≈ 14 million` evaluations. "Grid" is instead implemented as a **Latin
Hypercube** space-filling sample at each stage/scale: still deterministic
(fixed seed), still coarse-to-fine with elitist retention, still
embarrassingly parallel, but tractable at any dimensionality. Its main
weakness relative to CMA-ES: the shrink is isotropic and axis-aligned, so it
can't learn correlated directions between parameters — see the benchmark
finding above.

Key parameters (`HierarchicalGridSearch`): `n_stages` (default 4),
`n_samples_per_stage` (40), `retain_best_n` (5), `shrink_factor` (0.4 — each
stage's local search range = previous stage's range × this), `seed`, `n_jobs`.

### Cheap pre-build validity filter (`geometry/constraints.py::quick_reject_reason`)

`evaluate_design()`'s first step, applying to *every* candidate from either
optimizer: a handful of checks directly on the raw `DesignParameters`
control points, before paying for `build_aircraft()` + AeroBuildup +
NeuralFoil (the dominant per-candidate cost, ~10 s vs. ~0.1 ms for a
candidate caught here). Deliberately conservative — chosen so each check can
only reject a candidate the real, post-build `check_all_constraints()` would
*also* reject, never one it would have accepted:
- **Chord/twist monotonicity**: exact, not approximate. These curves are
  PCHIP-interpolated between the control points (see
  [The aircraft representation](#the-aircraft-representation)), and PCHIP
  is shape-preserving, so monotonic control points exactly imply a
  monotonic full curve and vice versa — checking the ~7 control points is
  mathematically equivalent to checking the full ~200-station mesh.
- **LE/Z offset curvature**: the *same* formula and threshold as the real
  check, evaluated on the sparse control points rather than the full mesh.
  A coarser finite-difference estimate smooths over sharp *local* bends
  rather than exaggerating them, so it's biased toward under-, not
  over-estimating curvature — i.e. biased toward a missed rejection (caught
  downstream anyway), not a false one. (This is why it doesn't just check
  the search-space slope bounds directly: a segment can have a large
  average slope while being perfectly straight — zero curvature — so
  bounding slope directly risks rejecting a genuinely valid, merely steep,
  design.)
- **Fuselage fit**: the real check searches for a placement across every
  footprint station simultaneously — too expensive to replicate here. But
  chord is guaranteed monotonically non-increasing (checked above), so the
  root station provably has the largest chord of any station — if even the
  root's own best-case thickness/chord can't contain the required box, no
  placement search over the more constrained full footprint can succeed
  either.

Not exhaustive (e.g. min local thickness/spar depth isn't checked, since it
doesn't reduce to a cheap control-point check the same way) — any invalid
candidate that slips past this still gets caught by the real,
always-correct `check_all_constraints()` downstream. Verified against 600
random candidates spanning the full Stage 1/2 search bounds: 5.2% caught
early, zero false rejections.

### Stage 1 (`optimization/stage1.py`)

Varies `thickness_scale`, `camber_scale`, `reflex_scale` at each of the 5
airfoil control stations (15 variables), planform held fixed. Bounds come
from `MIN/MAX_THICKNESS_RATIO` (converted to thickness_scale bounds) and
fixed `(0.0, 1.6)` ranges for camber/reflex scale.

### Stage 2 (`optimization/stage2.py`)

Varies `span_m`, `sweep_deg`, and `chord`/`twist`/`le_offset_deviation`/
`z_offset` at each of the 7 planform control stations (30 variables).

**Important design choice**: chord and twist are **not** optimized as 7
independent per-station values. An early version did that, and across 241
evaluations found *zero* improvement over the baseline — diagnosis: with
independent per-station bounds, a random sample has almost no chance of
landing on a monotonically-decreasing sequence by chance (7 independent
draws are correctly ordered ~1/7! ≈ 0.02% of the time), so nearly every
candidate hit the hard constraint penalty regardless of its aerodynamics.
Fixed by reparameterizing the *optimization variables* as a root value +
non-negative per-segment deltas (chord, twist) or a root value + a bounded
slope per segment (LE offset deviation, Z offset — free-form/non-monotonic
curves can't use the non-negative-delta trick). This makes monotonicity
(chord/twist) hold **by construction** regardless of what bounds are
configured.

**Chord/twist bounds are per-station absolute values** —
`CHORD_STATION_M_BOUNDS` (m), `TWIST_STATION_DEG_BOUNDS` (deg), one `(lo,
hi)` pair per station (or a single pair broadcast to all stations).
`make_stage2_parameter_set()` derives each segment's decrement `Var` bounds
from the two stations it connects, e.g. the largest possible chord decrement
over a segment is `max(0, chord_hi[i] - chord_lo[i+1])` — the drop from
station `i`'s ceiling to station `i+1`'s floor.

**LE/Z offset bounds are a root value plus explicit per-*segment* slope
bounds** — `LE_OFFSET_ROOT_M_BOUNDS`/`Z_OFFSET_ROOT_M_BOUNDS` (a single
`(lo, hi)` pair for the root/station-0 value) and
`LE_OFFSET_SLOPE_M_PER_SPAN_BOUNDS`/`Z_OFFSET_SLOPE_M_PER_SPAN_BOUNDS` (one
`(lo, hi)` pair *per segment* — 6 of them for the default 7-station layout,
m of offset per unit of normalized span y). This replaced an earlier
version that *derived* segment slope bounds from per-station absolute
ranges plus a single global slope cap plus a special-cased non-negative
floor for whichever segments fell in an assumed "winglet region": that
mechanism turned out to be only a soft bias, not a guarantee (sampling every
segment at its own lower bound could still produce a fully drooped tip,
contradicting what a "minimum winglet height" setting implied), and its
single global cap applied uniformly to every segment — including ones
meant to form a tight winglet bend — capping the achievable local bend
angle at ~50° regardless of intent. Direct per-segment bounds give
transparent, fully independent control over every segment, including the
ones spanning the wingtip, with no hidden mechanism.

This parameterization dropped the random-sample constraint-violation rate
from ~100% (fully independent per-station values) to ~20-80% depending on
how tight the configured bounds are, and real optimization runs find genuine
improvement (see Status) — though see
[Simplifications and known limitations](#simplifications-and-known-limitations)
for why a real fuselage-fit fix means the *current* default baseline needs a
multi-cycle run (not Stage 2 alone) to reach full validity.

### Multi-cycle driver (`optimization/cycle.py`)

`run_multi_cycle()` alternates `run_stage1` → `run_stage2` → ... for
`n_cycles` (default 2, 2 stages each), each stage starting from wherever the
previous one left off. Optionally stops early via `convergence_tol` if a
full cycle's improvement falls below that amount. Stage 1 and Stage 2 each
get their own separate optimizer instance (different dimensionality), same
algorithm/hyperparameters for both when using CMA-ES.

---

## Visualization

All in `viz/`, all return `plotly.graph_objects.Figure` (so they can be
displayed inline, saved via `.write_html()`, or embedded):

- **`geometry_plots.py`**: interactive 3D aircraft (watertight mesh, plus the
  required fuselage box drawn at its actual best-fit placement — see
  [Fuselage fit](#geometry-generation)), top/front/side orthographic views
  (with the fuselage box outlined on each), airfoil-section + schedule
  overlay.
- **`aero_plots.py`**: drag polar + CL/Cm-vs-alpha (via an AeroBuildup alpha
  sweep), spanwise lift/local-CL/Reynolds-number distributions.
- **`structures_plots.py`**: bending moment, shear force, spar depth/width,
  bending stress, safety factor vs. span, plus `plot_torsion_and_deflection`
  (torque, torsional shear stress + safety factor, bending deflection vs.
  span — see [Structural proxy](#analysis-modules)).
- **`optimization_plots.py`**: convergence (best-so-far + per-stage
  best/worst), parameter evolution (best candidate's variables across
  stages), and multi-cycle convergence across stage/cycle boundaries.
- **`analysis_plots.py`** (deep-analysis-only): `plot_objective_contributions`
  (why a design scored the way it did, sorted by |contribution|),
  `plot_mass_breakdown` (shell/spar/motor-ESC/avionics/servos/battery, five
  separate bars now that fixed equipment isn't one lump — see
  [Mass estimation](#analysis-modules)), `plot_cg_layout` (top view + front
  view of the actual aircraft, every component drawn at its real computed
  position — see [CG / static margin](#stability-center-of-gravity-and-performance-estimates)
  — with the neutral point, CG, target stability band, and feasible battery
  x-range shown as chordwise lines/bands on the top view; replaced an
  earlier 1D longitudinal-only diagram).
- **`flow_plots.py`** (deep-analysis-only, no CFD solver): pressure
  coefficient (Cp) from NeuralFoil's boundary-layer solution --
  `get_aero_from_neuralfoil()` reports the boundary-layer edge velocity ratio
  (`ue/Vinf`) at 32 fixed panel midpoints on each surface;
  `Cp = 1 - (ue/Vinf)²` (incompressible Bernoulli) turns that into genuine
  boundary-layer-informed pressure data, not a fabrication, just not a full
  3D flow field. `plot_cp_surface_3d` colors the aircraft's own watertight
  mesh per-vertex by Cp (interpolated spanwise from a subset of sampled
  stations and chordwise from NeuralFoil's 32-point grid onto the mesh's own
  161-point cosine grid) — verified vertex-count-exact against the mesh and
  physically sane (suction peak just aft of the leading edge, not at the LE
  itself, matching real airfoil behavior). `plot_cp_sections`/
  `plot_cp_heatmap` (2D Cp-vs-x/c and a spanwise Cp heatmap) are also
  available in the module but no longer wired into the GUI, which shows the
  3D surface instead.

Pareto plots are noted in the original spec as future work (once a
multi-objective algorithm exists) and aren't implemented.

---

## The interactive GUI

`flyingwing/gui/app.py`, launched via `scripts/run_gui.py` →
`http://127.0.0.1:8050`. A 6-tab control panel — all tabs share one page
(Dash's static-tabs pattern), so cross-tab actions (e.g. "send this result
to the Design tab") work directly, without a save/reload step.

**Design tab** — the single-design editor, laid out as a table (not a
stack of labeled input rows):
- Span/sweep are a "Global" row at the top; the planform values (chord,
  twist, LE offset, Z offset) and the airfoil schedule values (thickness,
  camber, reflex) are each a table where **columns are span-control
  stations and rows are quantities** (the first row of each table, `y`, is
  the station's position — editable for interior stations, fixed at 0/1 for
  the root/tip endpoints).
- **Stations are directly addable/removable**: "+ Add station" inserts a
  new column at the midpoint of the largest y-gap, with every quantity's
  value at that new station interpolated from the existing curve (no
  discontinuity); each interior column has its own "×" button to remove
  just that station. `optimization/stage1.py`/`stage2.py` read the station
  count from whatever baseline they're given, so this isn't just a Design
  tab preview feature — a design with a different station count works as an
  optimizer baseline too.
- Changing **any** value immediately regenerates the geometry and updates
  the 3D model, orthographic views, airfoil distribution, and a text panel
  of derived properties (wing area, AR, MAC, fuselage fit, constraint
  validity) — fast (~30 ms), geometry-only, no aero.
- **"Run Full Aerodynamic + Structural Evaluation"** button runs the full
  `evaluate_design` pipeline plus drag-polar/spanwise/structural plots and
  the glide-ratio/endurance performance estimate (~20 s — a real AeroBuildup
  alpha sweep + structural analysis — hence not run on every keystroke; a
  loading spinner shows while it runs).
- **"Export STL"** writes the current live-edited geometry's watertight mesh
  to `outputs/design_tab_export/aircraft.stl` and offers it as a download.
- **"Save as Default Design"** overwrites `configs/default_design.yaml` with
  whatever's currently in the tab — this becomes the baseline every
  script/GUI run uses when none is otherwise specified
  (`geometry.params_io.load_default_design_parameters`, which falls back to
  the hardcoded `default_design_parameters()` if the file doesn't exist),
  replacing an earlier design entirely by hand-editing `geometry/params.py`.
  Same "takes effect on the next run or restart" timing as the Bounds &
  Weights tab below, not immediate for the currently-open tab.

**Bounds & Weights tab** — edits every objective weight and normalization
constant (`configs/objective_weights.yaml`/`configs/normalization.yaml` —
see [Normalization](#the-objective-function)) and every Stage 1/2 search
bound plus structural/mass/CG/performance constant
(`configs/bounds_overrides.yaml`, layered over the hardcoded defaults in
`config.py`/`objective/mass.py`/`objective/cg.py`/`objective/performance.py`
— see [the override mechanism](#how-to-change-parameters-bounds-and-weights)
below). Laid out as collapsible sections (click a header to expand/collapse)
with scalar constants in an auto-wrapping grid rather than one full-width
row each, and per-station/per-segment bounds compacted into a small lo/hi
table (one column per station or segment) inside a collapsed-by-default
section, rather than one row per station. All numeric fields accept either
`.` or `,` as the decimal separator (typed as plain text, parsed in Python —
native `<input type="number">` fields are locale-dependent about which
separator they accept, with no reliable way to detect or work around that
from the server side, so this sidesteps the whole class of problem); every
"Save" button also validates before writing, refusing to save (with a clear
message naming the empty/invalid field) rather than silently writing a
broken value. Saved changes take effect the next time an optimizer run is
launched (each run is a fresh subprocess that re-reads all the YAML files)
or the GUI is restarted — **except** objective weights/normalization, which
the Design tab's "Run Evaluation" button and the Deep Analysis tab both
re-read fresh on every use, no restart needed.

**Run Optimizer tab** — pick Stage 1 / Stage 2 / multi-cycle and a baseline
(the built-in default design / whatever's currently in the Design tab / an
existing result), then a **setup mode**:
- **Simple** (the default) — the only extra input is a target run duration
  in minutes. Everything else — method (always CMA-ES, the best-evidenced
  default), population size, max generations, restarts, and the number of
  parallel workers (most of the machine's cores) — is picked automatically
  to roughly fill that time, using a real timing measurement of the baseline
  design taken just before the run starts (see `optimization/auto_tune.py`).
  For a multi-cycle run the target is split evenly across the (fixed at 2)
  cycles' 4 stage-runs, using Stage 2's (the larger, more expensive problem)
  settings for both stages so the run doesn't overshoot the budget. Meant
  for a first try with no tuning knowledge required; click "Estimate Time"
  to preview exactly what it would pick without starting anything.
- **Advanced** — set every optimizer parameter yourself: an **optimization
  method** (CMA-ES, the default, or Latin Hypercube — see
  [Optimization](#optimization)) with its own settings block (CMA-ES: initial
  step size, population size, max generations, restarts; LHS: stages,
  samples/stage, retain-best-N).

Either way, then either:
- **"Estimate Time"** — times one real batch of evaluations (using the
  currently-configured `n_jobs` and the selected baseline, through the
  actual parallel evaluation path, so it reflects real per-worker overhead,
  not a guess) and extrapolates to the currently-configured total planned
  evaluation count, giving a quick, rough wall-clock estimate before
  committing to a potentially long run. Flags it clearly if the baseline
  itself got caught by the cheap pre-build filter (see
  [Optimization](#optimization)), since that timing wouldn't be
  representative of a real evaluation's cost.
- **"Start Run"** — launches `scripts/run_stage*.py` **as a background
  subprocess** — the exact same script the CLI uses, just with different
  arguments — into a timestamped `outputs/<type>_run_<timestamp>/`
  directory, and polls it every 1 second until it completes or fails: a
  live percentage progress bar + text (stage/generation, restart if using
  CMA-ES, evaluations done, best score so far — parsed from `PROGRESS
  {json}` lines the optimizer prints once per stage/generation) plus the
  raw subprocess log in a scrolling panel below. Only one run at a time (the
  button is disabled while one is active).

**Results tab** — lists every run directory under `outputs/` that has a
`result.pkl` (both the CLI's fixed names and the GUI's timestamped ones),
and on "Load" re-renders its metrics table (re-evaluating the baseline for
comparison takes a few seconds; everything else is instant) and 3D/
orthographic/airfoil/convergence plots by reusing the same `viz/*`
functions used everywhere else — no recomputation of the optimization
itself. "Send to Design tab" loads that run's best design back into the
Design tab (rebuilding its station tables to match, even if the run has a
different station count than what's currently shown). "Export STL" writes
the loaded run's mesh to `outputs/<run>/aircraft.stl`.

**Deep Analysis tab** — pick a past run and "Load" to re-evaluate its best
design from scratch (~30-90 s: several fresh AeroBuildup, VLM, and
NeuralFoil calls, not read from the pickle) and see everything that went
into its score in one place: the objective contribution breakdown, mass
breakdown, a top+front-view CG layout diagram, the full structural proxy plots plus
torsion/deflection, an AeroBuildup-vs-hybrid-VLM L/D cross-check, and the
3D Cp-colored pressure surface. "Export STL" here works the same as the
other two tabs.

**Documentation tab** — renders this README in-app via `dcc.Markdown`, read
fresh from disk on every visit (so edits show up without restarting the
GUI).

---

## How to change parameters, bounds, and weights

Everything below can be done either by hand-editing the named
file/constants, or from the GUI's **Bounds & Weights** tab (which edits the
same YAML files) and **Design**/**Run Optimizer** tabs (for the
design/baseline itself) — pick whichever's more convenient.

**To change the objective function's priorities** (e.g. care more about mass,
less about L/D): edit `configs/objective_weights.yaml` (or the GUI's Bounds
& Weights tab), or in Python:

```python
from flyingwing.objective.objective import ObjectiveWeights
weights = ObjectiveWeights(w_mass=6.0, w_cruise_L_over_D=2.0)
weights.to_yaml("configs/objective_weights.yaml")  # persist it
```

Remember weights are normalized (see [Normalization](#the-objective-function))
— a weight roughly means "this term's contribution to the score, for a
design like the default baseline," so `w_mass=6.0` is a very different
intensity than the pre-normalization `w_mass=6.0` would have been. Then pass
`weights=ObjectiveWeights.from_yaml("configs/objective_weights.yaml")` and
`normalization=NormalizationConstants.from_yaml("configs/normalization.yaml")`
into `run_stage1`/`run_stage2`/`run_multi_cycle`, or `score(metrics, weights,
normalization)` directly. The CLI scripts do this automatically
(`--weights-yaml`/`--normalization-yaml`, defaulting to
`configs/objective_weights.yaml`/`configs/normalization.yaml` if they exist).

**To change the default starting design** (what Stage 1/2 optimize from, and
what the GUI opens with): use the Design tab's **"Save as Default Design"**
button (writes `configs/default_design.yaml`, read by
`geometry.params_io.load_default_design_parameters()` — see
[The interactive GUI](#the-interactive-gui)), or edit the hardcoded fallback
defaults in `geometry/params.py` (`AirfoilSchedule`/`Planform` dataclass
field defaults, used only if `configs/default_design.yaml` doesn't exist),
or construct a custom `DesignParameters`, save it with
`geometry/params_io.save_design_parameters`, and pass its path via
`--baseline-yaml` to any `scripts/run_*.py` (the GUI's Run tab does this for
you when you pick a non-default baseline source).

**To change search bounds** (how far the optimizer is allowed to explore) or
**physical/structural/CG/performance assumptions**: edit the GUI's Bounds &
Weights tab, or `configs/bounds_overrides.yaml` directly, or the hardcoded
defaults in `config.py`/`objective/mass.py`/`objective/cg.py`/
`objective/performance.py` (`*_BOUNDS` constants like `CHORD_STATION_M_BOUNDS`,
`TWIST_STATION_DEG_BOUNDS`, `SWEEP_DEG_BOUNDS`, `LE_OFFSET_ROOT_M_BOUNDS`/
`Z_OFFSET_ROOT_M_BOUNDS` (root value only), `LE_OFFSET_SLOPE_M_PER_SPAN_BOUNDS`/
`Z_OFFSET_SLOPE_M_PER_SPAN_BOUNDS` (per-segment slope — see
[Stage 2](#optimization)); structural constants like `ALLOWABLE_SPAR_STRESS_PA`,
`DESIGN_LOAD_FACTOR_G`, `SPAR_WIDTH_FRACTION_CHORD`, `SPAR_YOUNG_MODULUS_PA`;
fuselage box dimensions `FUSELAGE_MIN_INTERNAL_*_M`; mass constants in
`objective/mass.py` like `SHELL_AREAL_DENSITY_KG_M2`, `MOTOR_ESC_MASS_KG`,
`AVIONICS_MASS_KG`, `SERVO_MASS_KG`; `BATTERY_MASS_KG` in `objective/cg.py`
(the only CG constant still GUI-editable — component *positions* are now
either a fixed rule or computed directly from the design's own geometry, see
[CG / static margin](#stability-center-of-gravity-and-performance-estimates),
so there's nothing left to tune there); and battery/efficiency constants in
`objective/performance.py` like `BATTERY_CAPACITY_MAH`). All these modules
load `configs/bounds_overrides.yaml` at import time via the shared
`_overrides.apply_overrides()` helper and replace any of their constants
named in it — the file is absent by default, so a fresh checkout behaves
exactly like the hardcoded values. **This only takes effect on the next
optimizer subprocess or GUI restart** — unlike objective weights/
normalization, these constants are read once into plain module globals at
import time, not re-read live (see
[Simplifications and known limitations](#simplifications-and-known-limitations)).
Each of `CHORD_STATION_M_BOUNDS` etc. accepts either a single `(lo, hi)`
pair (broadcast to every station/segment) or a list of one `(lo, hi)` pair
per station/segment, for asymmetric control — see
`vector.resolve_per_station_bounds`. These feed directly into
`make_stage1_parameter_set`/`make_stage2_parameter_set`
(`optimization/stage1.py`/`stage2.py`).

> If you have an older `bounds_overrides.yaml` predating either the
> per-station rework or the direct-per-segment-slope-bounds rework, it may
> still contain stale keys from either (`CHORD_DECREMENT_M_BOUNDS`,
> `WASHOUT_INCREMENT_DEG_BOUNDS`, `LE_OFFSET_SLOPE_BOUNDS`,
> `Z_OFFSET_SLOPE_BOUNDS`, `Z_OFFSET_TIP_SLOPE_BOUNDS`, `CHORD_ROOT_M_BOUNDS`,
> `CHORD_M_BOUNDS`, `TWIST_DEG_BOUNDS`, `TWIST_ROOT_DEG_BOUNDS`,
> `LE_OFFSET_DEVIATION_M_BOUNDS`, `Z_OFFSET_M_BOUNDS`, or the later
> `LE_OFFSET_STATION_M_BOUNDS`/`Z_OFFSET_STATION_M_BOUNDS`/
> `MAX_LE_OFFSET_SLOPE_M_PER_SPAN`/`MAX_Z_OFFSET_SLOPE_M_PER_SPAN`/
> `Z_OFFSET_TIP_MIN_M`/`Z_OFFSET_TIP_SEGMENT_Y_THRESHOLD`). These are
> silently ignored now (harmless — `apply_overrides` skips unknown keys) but
> not auto-migrated, since translating one bound representation into another
> is mathematically underdetermined/lossy. Re-tune the current
> `*_ROOT_M_BOUNDS`/`*_SLOPE_M_PER_SPAN_BOUNDS` keys fresh via the GUI, using
> old values as a rough qualitative guide if useful.

**To change the number/position of control stations**: use the Design tab's
"+ Add station" button and each interior station's "×" button (see
[The interactive GUI](#the-interactive-gui)) — this is a live GUI action now,
no code edit needed. To change the *default* station layout the GUI/CLI
scripts open with, change `DEFAULT_AIRFOIL_Y_CONTROL` /
`DEFAULT_PLANFORM_Y_CONTROL` in `geometry/params.py`, and update the
corresponding tuple lengths in `AirfoilSchedule`/`Planform` defaults to
match. The optimizer parameter sets (`make_stage1/2_parameter_set`) read the
control-station count from whatever baseline they're given automatically —
but per-station/per-segment bound overrides sized as a list (not a single
broadcast pair) must match that count exactly, or
`resolve_per_station_bounds` raises a clear error rather than silently
misapplying bounds.

**To change optimizer search effort**: for CMA-ES (the default), adjust
`CMAESOptimizer` arguments (`sigma0`, `population_size`, `max_generations`,
`n_restarts`, `n_jobs` — see [CMA-ES](#optimization) for what each one
actually controls and concrete "longer/better" vs. "faster/worse" presets)
in the `scripts/run_*.py` files or via their `--cma-*` CLI flags, or when
constructing your own `CMAESOptimizer(...)`. For the Latin Hypercube
alternative, the equivalent `HierarchicalGridSearch` arguments are
`n_stages`, `n_samples_per_stage`, `retain_best_n`, `shrink_factor`. The
GUI Run tab's **Estimate Time** button gives a quick before-you-commit
estimate for whatever's currently configured.

**To inspect why a design scored the way it did**: `score(metrics,
weights).contributions` is a dict of every term's individual contribution —
print or plot it, or use the GUI's Deep Analysis tab, which plots it
(`plot_objective_contributions`) alongside the mass/CG breakdown and
structural detail for any past run.

---

## Code structure

```
flyingwing/
  config.py                 Global constants: units, target speeds, wingspan/fuselage/
                             structural bounds, Stage 2 per-station/per-segment search
                             bounds -- see file for full list. Loads
                             configs/bounds_overrides.yaml (if present) at import time.
  _overrides.py              Shared YAML-override-application helper used by config.py,
                              objective/mass.py, objective/cg.py, objective/performance.py,
                              and optimization/stage1.py

  geometry/
    spanwise.py              SpanwiseDistribution (control points -> curve, linear or PCHIP),
                              make_span_stations (cosine spacing)
    params.py                DesignParameters / Planform / AirfoilSchedule -- the complete,
                              optimizer-independent design description
    params_io.py              save_design_parameters()/load_design_parameters(): YAML
                              (de)serialization, used to hand a design across process boundaries.
                              load_default_design_parameters(): reads configs/default_design.yaml
                              if present (see the Design tab's "Save as Default Design"), else
                              falls back to params.py's hardcoded default_design_parameters().
    airfoil_family.py        MH64 decomposition + thickness/camber/reflex modification.
                              max_thickness_x_over_c()/shell_centroid_x_over_c(): spar position
                              and shell mass centroid, computed directly from the base MH64
                              profile's own geometry (cached) -- see objective/cg.py.
    aircraft.py               build_aircraft(): the geometry generator, Aircraft dataclass
    mesh.py                  Watertight triangle mesh construction (mirroring, LE/TE welding, tip caps)
    export.py                 write_stl(): vectorized binary STL export of the watertight mesh
    fuselage.py              Internal fuselage box fit check -- searches for the best-fit
                              chordwise/vertical placement, not just an independent
                              thickness/length check
    constraints.py            Stage 1 + Stage 2 geometric validity constraints, plus
                              quick_reject_reason(): the cheap pre-build validity filter (see
                              Optimization)

  analysis/
    airfoil_2d.py             NeuralFoil (+ optional XFoil) 2D section analysis
    aero_3d.py                 AeroBuildup (primary, now also extracts Clb for roll stability) +
                                VLM (cross-check, with a resolution-robust fallback for deep
                                analysis) + hybrid VLM/viscous drag + optional AVL
    structures.py              Schrenk lift distribution -> shear/bending/spar/stress/safety
                                factor, plus torsion + Euler-Bernoulli deflection (deep-
                                analysis-only); torsion's spar-axis default now comes from
                                airfoil_family.max_thickness_x_over_c(), not a fixed fraction

  objective/
    mass.py                    Parametric mass estimate: shell + spar + three independent
                                component masses (MOTOR_ESC_MASS_KG/AVIONICS_MASS_KG/
                                SERVO_MASS_KG) + payload volume. Loads
                                configs/bounds_overrides.yaml (if present) at import time.
    cg.py                      estimate_cg(): component mass+position CG model -> real static
                                margin + feasible battery x-range. Component positions are
                                geometry-derived (motor/servo placement rules, avionics/battery
                                in the real fuselage box, shell/spar centroids from
                                airfoil_family.py) rather than tunable fractions. Loads
                                configs/bounds_overrides.yaml at import time.
    performance.py             estimate_performance(): glide ratio/angle, endurance/range from a
                                battery-capacity assumption -- deep-analysis-only, not called from
                                evaluate_design(). Loads configs/bounds_overrides.yaml at import time.
    metrics.py                 evaluate_design(): quick_reject_reason() pre-check, then geometry ->
                                analysis -> CG -> DesignMetrics (incl. soaring power, glide angle,
                                roll stability)
    objective.py               ObjectiveWeights + NormalizationConstants + score():
                                DesignMetrics -> scalar score, normalized, NaN-safe

  optimization/
    base.py                    Optimizer interface, EvaluatedCandidate, OptimizationResult,
                                evaluate_batch() (shared parallel-evaluation helper)
    vector.py                   ParameterSet / Var / resolve_per_station_bounds: flat-vector
                                <-> DesignParameters translation, and single-pair-or-per-station
                                bound resolution
    cmaes.py                    CMAESOptimizer -- the default optimizer (CMA-ES via the `cma`
                                package), normalized-space search with IPOP restarts
    hierarchical.py             HierarchicalGridSearch (Latin-Hypercube coarse-to-fine) --
                                selectable alternative to CMA-ES, with an optional progress_cb
                                fired once per stage
    stage1.py                   Stage 1 parameter set + driver (airfoil schedule). Loads
                                configs/bounds_overrides.yaml at import time (THICKNESS_SCALE_BOUNDS
                                etc. live here, not config.py, to avoid a circular import through
                                airfoil_family.py)
    stage2.py                   Stage 2 parameter set + driver (planform): per-station absolute
                                bounds for chord/twist, per-segment slope bounds for LE/Z offset
                                (config.py's CHORD_STATION_M_BOUNDS,
                                LE/Z_OFFSET_SLOPE_M_PER_SPAN_BOUNDS etc.) translated into the
                                underlying monotonic/slope-bounded search variables
    cycle.py                    Multi-cycle Stage1<->Stage2 driver

  viz/
    geometry_plots.py           3D aircraft (incl. the fuselage box at its actual best-fit
                                placement), orthographic views, airfoil distribution
    aero_plots.py                Drag polar, spanwise lift/CL/Reynolds distributions
    structures_plots.py          Bending moment, shear, spar sizing, stress, safety factor,
                                plus torsion/deflection
    optimization_plots.py        Convergence, parameter evolution, multi-cycle convergence --
                                algorithm-agnostic (CMA-ES generations or LHS stages both just
                                look like "one more batch" in OptimizationResult.history)
    analysis_plots.py            Deep-analysis-only: objective contribution breakdown, mass
                                breakdown (5 bars), CG layout (top+front view, real positions)
    flow_plots.py                Deep-analysis-only: Cp from NeuralFoil's boundary-layer output,
                                painted onto the watertight mesh in 3D (plus 2D Cp-vs-x/c and a
                                spanwise Cp heatmap, available but not wired into the GUI)

  gui/
    app.py                     Thin Dash shell combining the 6 tabs below into one page
    numeric.py                  parse_number(): locale-robust text-field parsing (accepts '.' or
                                ',' as the decimal separator) used by every numeric GUI input
    design_tab.py               Design tab -- table-based control-point editor (add/remove
                                stations), live geometry preview, full evaluation, STL export,
                                "Save as Default Design"
    config_tab.py               Bounds & Weights tab -- collapsible sections (incl. a
                                Normalization constants section), compact grids/tables,
                                save-time validation
    run_tab.py                   Run Optimizer tab -- optimizer method selector (CMA-ES/LHS),
                                Estimate Time button, live progress bar + log
    results_tab.py               Results tab -- STL export added here too
    analysis_tab.py              Deep Analysis tab -- consolidated post-run report
    docs_tab.py                  Documentation tab -- renders README.md in-app
    run_manager.py               Subprocess launch/track/poll for the Run Optimizer tab
    results_io.py                Scans outputs/ for result.pkl files, loads them, for the
                                Results/Deep Analysis tabs

configs/
  objective_weights.yaml       Default ObjectiveWeights, editable/reloadable
  normalization.yaml           Default NormalizationConstants, editable/reloadable (see
                                Normalization)
  default_design.yaml          Optional; the baseline every script/GUI run uses when none is
                                otherwise specified, if present (created by the Design tab's
                                "Save as Default Design")
  bounds_overrides.yaml        Optional; overrides named constants across config.py/
                                objective/mass.py/objective/cg.py/objective/performance.py/
                                optimization/stage1.py when present (absent by default --
                                created by the GUI's Save Bounds)

scripts/
  run_stage1.py / run_stage2.py / run_multi_cycle.py    Optimization demos, now with argparse CLI
                                                          flags (all optional, incl. --optimizer
                                                          cma/lhs and the matching --cma-*/legacy
                                                          flags); write plots + result.pkl to
                                                          outputs/<name>_run/, plus a post-run
                                                          performance/battery-range summary. The
                                                          GUI's Run tab invokes these same scripts
                                                          as subprocesses -- it never duplicates
                                                          this logic.
  run_gui.py                                             Launches the Dash GUI (all 6 tabs)

outputs/                       Generated plots + pickled results (gitignored). run_<n>_run/ from
                                the CLI's fixed names, <type>_run_<timestamp>/ from the GUI.
                                aircraft.stl appears here after an "Export STL" click.
```

---

## Simplifications and known limitations

These are documented in code comments where they matter, collected here for
visibility:

- **Objective weights and search bounds are not tuned.** They're reasonable
  first guesses, not a reflection of real design priorities — expect (and
  plan for) an iteration pass here. This is now a *specifically identified*
  gap, not a vague one: see the next bullet.
- **The built-in default baseline design (`DesignParameters()` /
  `geometry/params.py`) currently fails the fuselage-fit constraint.** A real
  bug in the fuselage-fit check (`geometry/fuselage.py`) was found and fixed:
  it used to verify "chord is long enough somewhere along the span" and
  "thickness is enough somewhere along the chord" *independently*, never
  requiring both to hold *at the same chordwise position* for *every* span
  station within the fuselage's width footprint simultaneously. The corrected
  check (a real placement search — see [Fuselage fit](#geometry-generation))
  found the default design short of the required internal height — even
  though each individual footprint station had enough thickness *somewhere*
  along its own chord, no single box window worked for all of them together.
  A valid design **is** reachable within the current Stage 1/2 bounds — a
  real CMA-ES run from this exact baseline reached a valid, meaningfully
  better design (score -0.02 → 1.88, cruise L/D 7.4 → 13.4) in 180
  evaluations (see [CMA-ES](#optimization)) — so the search space itself
  isn't broken, but the shipped default itself is not valid out of the box.
  The default baseline's own parameter values were *deliberately not
  changed* — only the check was fixed — so revisiting the default (via the
  Design tab's "Save as Default Design" once you have a valid one, or just
  always running a real optimization rather than trusting the shipped
  default directly) is the practical next step.
- **`static_margin` and component mass/position are real, geometry-derived
  values now**, not fixed-fraction placeholders (see
  [Stability, center of gravity, and performance estimates](#stability-center-of-gravity-and-performance-estimates))
  — motor/servo placement follows physically-motivated rules evaluated on
  the design's own actual geometry, avionics/battery sit in the real
  fuselage box, and shell/spar centroids come from the actual MH64 profile.
  What's still assumption-heavy: component *masses* are realistic defaults
  for "high quality" parts, not measured for a specific build; there's no 3D
  packing/collision check (a large battery and the avionics could overlap in
  reality even if both fit the fuselage box's 1D x-range independently); and
  the battery itself is treated as a point mass with no shape. Good enough
  to reason about *roughly where the CG needs to be* and to rank designs
  against each other, not to certify a specific build. `w_static_margin` is
  still 0 by default so existing tuned weight files don't silently change
  behavior when this constraint became meaningful — a deliberate choice, not
  an oversight.
- **The structural proxy (including the newer torsion/deflection check) is a
  ranking heuristic, not FEA.** Schrenk's approximation + a simplified
  thin-wall spar box (plus a first-order torque estimate that only accounts
  for the aerodynamic-center-to-spar moment arm, not each section's own
  pitching moment) are fast and good enough to *compare* designs, not to
  certify one.
- **The endurance/range estimate (`objective/performance.py`) is a rough
  parametric model**, not a discharge-curve or propeller-map simulation —
  constant propulsive efficiency and a flat usable-capacity fraction, not
  battery voltage sag or prop-efficiency variation with RPM/airspeed.
- **VLM panel-count safety is geometry-dependent, not a fixed threshold** —
  see [Analysis modules](#analysis-modules) for the specific finding (13
  stations safe on one design, 17 and 21 both ill-conditioned on another).
  `analyze_vlm_robust()`'s fallback list handles this for the deep-analysis
  path, but if you call `analyze_vlm()` directly with a custom resolution,
  there's no guarantee it won't blow up for some geometry — always sanity-
  check `|CL| < ~2` before trusting a result.
- **CMA-ES is fundamentally a local refinement method once it commits to one
  basin.** IPOP restarts (fresh random start + doubled population each time)
  mitigate this but don't guarantee finding the global optimum — a real
  benchmark showed CMA-ES clearly outperforming the Latin-Hypercube search at
  matched evaluation counts (see [CMA-ES](#optimization)), but neither
  algorithm has any formal optimality guarantee on this non-convex,
  constrained landscape. `population_size`/`sigma0` also aren't
  auto-tuned per-problem beyond the standard dimensionality-based heuristic
  — see [CMA-ES](#optimization) for what raising/lowering each one trades off.
- **The Latin-Hypercube search (the selectable alternative) is
  space-filling sampling, not a literal grid.** A full-factorial grid is
  intractable beyond a few dimensions (see [Optimization](#optimization)).
  Results are somewhat sensitive to the random seed and sample count,
  especially in Stage 2's 30-dimensional space, and — per the benchmark
  above — it plateaus well before CMA-ES does at the same evaluation budget,
  since its isotropic, axis-aligned shrink can't represent the coupling
  between Stage 2's parameters the way CMA-ES's adapted covariance can.
- **Stage 2's random-sample validity rate depends heavily on how tight the
  configured per-station/per-segment bounds are** (roughly 20-80% in
  testing) — tighter bounds mean less chance of hitting the
  fuselage-fit/minimum-thickness constraints, at the cost of a smaller
  search space. This is now a knob you control per-station/per-segment (see
  [Stage 2](#optimization)), not a fixed rate. The cheap pre-build filter
  (`quick_reject_reason`, see [Optimization](#optimization)) reduces the
  *cost* of an invalid sample (skips the ~10s build+analysis, catches ~5% of
  fully-random candidates in testing) but doesn't change the underlying
  validity rate itself — it's not exhaustive (e.g. doesn't check minimum
  local thickness/spar depth), so most invalid candidates still have to run
  the full, always-correct check to be caught.
- **2D analysis only evaluates the root section's airfoil**
  (`root_cl_max`, `root_cm_zero_lift` in `DesignMetrics`) at one Reynolds
  number/transition assumption (cruise speed, `n_crit=9`, clean). The
  `N_CRIT_ASSUMPTIONS` dict (clean/moderate/rough) exists in
  `airfoil_2d.py` but isn't swept over in `evaluate_design` — extending to
  multiple span stations and/or transition assumptions would give a fuller
  picture of stall behavior across the wing.
- **XFoil and AVL validation are optional and untested in this environment**
  (no XFoil/AVL binaries were installed here) — the wrapper functions catch
  exceptions and return `None` with a warning if the executables aren't
  found, so the rest of the framework works without them, but the
  validation path itself hasn't been exercised.
- **Bound-name migration**: `configs/bounds_overrides.yaml` files saved before
  the per-station bounds rework may still contain the old per-*segment* keys
  (`CHORD_DECREMENT_M_BOUNDS` etc.) — see the note in
  [How to change parameters, bounds, and weights](#how-to-change-parameters-bounds-and-weights).
  They're silently ignored, not auto-migrated.
- **Only one optimizer run at a time from the GUI.** The Run Optimizer tab
  refuses to start a second run while one is active — a deliberate,
  simple-and-safe default (each run already uses multiprocessing
  internally) rather than a job queue. Starting a run separately from a
  terminal while the GUI has one active isn't tracked by the GUI either
  (it only tracks the one it launched itself).
- **`bounds_overrides.yaml` edits from the GUI don't hot-reload the GUI's own
  in-memory constants** — `config.py`/`objective/mass.py`/`objective/cg.py`/
  `objective/performance.py`/`optimization/stage1.py` all apply this file's
  overrides once, into plain module globals, at import time — a saved change
  takes effect the next time an optimizer subprocess is launched (each one
  re-reads the file fresh) or the GUI itself is restarted. **This is
  different from objective weights/normalization**
  (`objective_weights.yaml`/`normalization.yaml`), which the Design tab's
  "Run Evaluation" button and the Deep Analysis tab both re-read fresh on
  every use — no restart needed for those two. Not a limitation for
  optimizer runs either way, which always pick up the latest saved values of
  everything (each run is a fresh subprocess).
- **STL export is surface-only.** `write_stl()` exports the same watertight
  triangle mesh used for visualization — genuinely manifold, suitable for
  CFD surface meshing (snappyHexMesh, ANSA, etc.), but there's no parametric
  CAD (STEP/IGES) export path.
- **The Cp flow visualization has no real CFD solver behind it** (none is
  installed/available in this environment) — it's derived from NeuralFoil's
  boundary-layer solution (`Cp = 1 - (ue/Vinf)^2`), genuine boundary-layer-
  informed section data, but not a solved 3D flow field (no interference
  effects between stations, no wake, no compressibility beyond what
  NeuralFoil itself models).

---

## What's not implemented / next steps

- **Weight/bound tuning, and a valid default baseline** — still the highest-
  priority next step: a real CMA-ES run finds a valid, meaningfully better
  design from the current default baseline (see
  [Simplifications and known limitations](#simplifications-and-known-limitations)),
  but nobody has fed that result back into `geometry/params.py`'s hardcoded
  defaults (or saved it as `configs/default_design.yaml` via "Save as
  Default Design") yet, and the objective weights are still first guesses,
  not calibrated priorities.
- **Consider enabling `w_static_margin`/`w_soaring_power`/`w_flight_angle`/
  `w_roll_stability`** now that all four are real, meaningful values rather
  than placeholders — all four are still 0 by default.
- **A real (not proxy) fixed-vs-free-form winglet trade study**, now that
  CMA-ES can push the LE/Z per-segment slope bounds (see
  [Stage 2](#optimization)) harder than the old derived-cap mechanism would
  allow — worth checking whether a genuinely sharper winglet bend actually
  earns its roll-stability weight back once one is enabled.
- **Pareto-front visualization / true multi-objective optimization**, once a
  multi-objective algorithm exists (`pymoo`'s NSGA-II is a natural
  candidate) — would let a user see the actual mass-vs-L/D-vs-roll-stability
  trade-off curve and pick a design from it, instead of hand-tuning scalar
  weights beforehand. Noted as future work in the original spec; more
  clearly motivated now that there are 8+ weighted terms to balance.
- **Multi-station / multi-condition 2D analysis** (sweep root/mid/tip
  sections, and the clean/moderate/rough transition assumptions) for a
  fuller picture of stall behavior, fed into the objective function.
- **A less assumption-heavy mass/endurance model** — component masses are
  now realistic per-part defaults rather than one lump-sum estimate,
  and positions are geometry-derived (see
  [CG / static margin](#stability-center-of-gravity-and-performance-estimates)),
  but there's still no 3D packing/collision check between components, and
  `objective/performance.py`'s endurance estimate is still a flat-efficiency
  parametric model, not a discharge-curve/prop-map simulation.
- **A real CFD path**, if the NeuralFoil-boundary-layer-derived Cp
  visualization isn't sufficient — would need an external solver (OpenFOAM,
  SU2, ...) and a meshing step from the now-available STL export.
- **Parametric CAD (STEP/IGES) export**, beyond the current STL surface mesh,
  if a downstream CAD/manufacturing tool needs it.
- **Sensitivity analysis** in the Deep Analysis tab (perturb each Stage 1/2
  variable around the best candidate and re-score, e.g. as a tornado chart)
  — noted as a nice-to-have when the tab was built, not implemented.
- **Fix the `ProcessPoolExecutor`-recreated-every-batch overhead**
  (`evaluate_batch()`, `optimization/base.py`) — a real inefficiency found
  while benchmarking CMA-ES vs. the Latin-Hypercube search: each batch/
  generation spawns a fresh worker pool (re-importing the whole
  AeroSandbox/NeuralFoil stack in every worker each time) rather than
  reusing one long-lived pool across a whole run. Measured parallel
  efficiency with 12 workers was ~27% of the naive `n_jobs`× speedup — CMA-ES
  is hit harder than the Latin-Hypercube search since it does many more,
  smaller batches (generations) for the same total evaluation count.
