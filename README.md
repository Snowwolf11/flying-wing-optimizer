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
7. [The objective function](#the-objective-function)
8. [Optimization](#optimization)
9. [Visualization](#visualization)
10. [The interactive GUI](#the-interactive-gui)
11. [How to change parameters, bounds, and weights](#how-to-change-parameters-bounds-and-weights)
12. [Code structure](#code-structure)
13. [Simplifications and known limitations](#simplifications-and-known-limitations)
14. [What's not implemented / next steps](#whats-not-implemented--next-steps)

---

## Status

Every module in the original project spec is implemented and has been run
end-to-end at least once, including a real (non-trivial) optimization run at
each stage. **What's not done is tuning** — the default objective weights and
search bounds are reasonable first guesses, not calibrated values, so running
Stage 2 (or a multi-cycle run) today tends to converge toward extreme designs
(e.g. oversized wing area, aggressive sweep) rather than something
recognizably "a good FPV wing." The mechanics all work; the *taste* of the
default objective function needs iteration. See
[Simplifications and known limitations](#simplifications-and-known-limitations)
for the full list of what to be aware of.

| Module | Status |
|---|---|
| Geometry generation (airfoil family, aircraft generator, mesh, fuselage check) | Done |
| Geometry visualization (3D, orthographic, airfoil distribution) | Done |
| 2D aero analysis (NeuralFoil) | Done |
| 2D aero validation (XFoil) | Wired, optional, not required (no XFoil binary needed) |
| 3D aero analysis (AeroBuildup, VLM) | Done |
| 3D aero validation (AVL) | Wired, optional, not required |
| Structural proxy (Schrenk + spar sizing) | Done |
| Aero + structural visualization | Done |
| Objective function | Done, weights not tuned |
| Hierarchical optimizer (Latin-Hypercube coarse-to-fine) | Done |
| Stage 1 (airfoil) optimization | Done, verified (score 16.10 → 17.23 in a demo run) |
| Stage 2 (planform) optimization | Done, verified, but produces extreme designs with default weights |
| Multi-cycle Stage1↔Stage2 driver | Done, verified (16.10 → 37.87 over 2 cycles) |
| Interactive Dash GUI | Done, verified via real HTTP callback round-trips |

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
.venv\Scripts\python.exe scripts\run_stage1.py        # optimize airfoil schedule, ~2-3 min
.venv\Scripts\python.exe scripts\run_stage2.py         # optimize planform, ~4-5 min
.venv\Scripts\python.exe scripts\run_multi_cycle.py    # alternate both, ~10-15 min
```

Each writes plots to `outputs/<name>_run/*.html` (open directly in a browser)
and a `result.pkl` containing the full optimization result + best design, so
you can reload and re-plot without re-running the optimization (see
[Code structure](#code-structure)). Every argument is optional — e.g.
`--n-stages`, `--n-samples-per-stage`, `--n-jobs`, `--seed`,
`--output-dir-name`, `--weights-yaml`, `--baseline-yaml` — run with `--help`
to see a script's full list; omitting all of them reproduces exactly the
timings above.

### Run the interactive GUI

```
.venv\Scripts\python.exe scripts\run_gui.py
```

Then open **http://127.0.0.1:8050** in a browser. It's a full control panel,
not just a design viewer — see [The interactive GUI](#the-interactive-gui):
edit any design parameter or search bound/weight, launch Stage 1/2/multi-cycle
runs as background jobs, and browse past results, all from one page.

### Run a single design evaluation from Python

```python
from flyingwing.geometry.params import default_design_parameters
from flyingwing.objective.metrics import evaluate_design
from flyingwing.objective.objective import score

params = default_design_parameters()
metrics = evaluate_design(params)   # ~4-5 seconds
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
  estimate, and returns a flat `DesignMetrics` object.
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
140 × 55 × 300 mm) that must fit inside the wing's own thickness/chord
envelope near the root. The check looks at every span station within the
box's width footprint and requires the local thickness ≥ required height and
local chord ≥ required length everywhere in that footprint.

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

### Mass estimation (`objective/mass.py`)

A rough parametric estimate feeding the objective function only:
- **Shell mass** = `2 × wing_area_m2 × 0.55 kg/m²` (wetted-area proxy for
  foam-core + glass/film skin).
- **Spar mass** = spar material cross-sectional area (from the structural
  proxy's spar box — the same box, so a heavier/thicker spar shows up as
  both higher safety factor *and* higher mass, giving the objective function
  a real trade-off instead of "more strength for free") integrated over
  span × `1600 kg/m³` (generic glass/carbon laminate density).
- **+ `0.45 kg`** fixed allowance for motor/ESC/servos/receiver/FC/wiring.
- **Payload volume margin** = an ellipse-proxy internal volume of the
  centre body (integrated over the fuselage's spanwise footprint) minus the
  required fuselage box volume.

---

## The objective function

### Metrics (`objective/metrics.py` — `DesignMetrics`)

One `evaluate_design(params)` call returns all of these:

| Category | Fields |
|---|---|
| Validity | `valid`, `constraint_violations` (list of human-readable strings) |
| Constraint margins (≥0 = compliant) | `fuselage_height_margin_m`, `fuselage_length_margin_m`, `tip_thickness_margin`, `thickness_monotonic_violation`, `chord_monotonic_violation`, `twist_monotonic_violation`, `min_local_thickness_margin`, `min_spar_depth_margin`, `le_curvature_violation` |
| Geometry | `wing_area_m2`, `aspect_ratio`, `span_m` |
| Aero — cruise (75 km/h) | `cruise_trim_alpha_deg`, `cruise_CL`, `cruise_CD`, `cruise_L_over_D` |
| Aero — fast (150 km/h) | `fast_trim_alpha_deg`, `fast_CL`, `fast_CD`, `fast_L_over_D` |
| Stability | `cruise_Cm`, `static_margin` (⚠ see limitations) |
| 2D root-section | `root_cl_max`, `root_cm_zero_lift` |
| Structure (at 8g) | `min_safety_factor`, `root_bending_moment_nm` |
| Mass / payload | `total_structural_mass_kg`, `payload_volume_margin_m3` |

### Weights and scoring (`objective/objective.py` — `ObjectiveWeights`, `score()`)

Every metric maps to exactly one weighted contribution, of one of four shapes:

| Shape | Formula | Used for |
|---|---|---|
| Maximize | `+weight × value` | `cruise_L_over_D`, `fast_L_over_D`, `root_cl_max`, `payload_volume_margin_m3` |
| Minimize | `-weight × value` | `total_structural_mass_kg` |
| Threshold (one-sided) | `-weight × max(0, threshold - value)²` | `min_safety_factor` (no reward *above* threshold — mass already prices in extra strength), `root_cm_zero_lift` |
| Target range | `-weight × (max(0, lo-v)² + max(0, v-hi)²)` | `static_margin` (weight 0 by default, see limitations) |

Plus a **constraint penalty**: every entry in the constraint-margins table
above is normalized to a dimensionless "fraction of a characteristic
threshold" (so meters/degrees/1-per-meter/ratio quantities are comparable),
summed, and multiplied by `constraint_penalty_scale` (default 1000). If
`valid` is `False`, a flat `invalid_penalty` (default 1000) is also
subtracted — so an infeasible design is essentially always ranked below a
feasible one, while still giving the optimizer a smooth gradient toward
feasibility rather than a flat cliff.

Current default weights (`configs/objective_weights.yaml`):

```yaml
w_cruise_L_over_D: 1.0
w_fast_L_over_D: 0.4
w_root_cl_max: 2.0
w_mass: 3.0
w_safety_factor: 0.05
safety_factor_min: 1.5
w_static_margin: 0.0      # disabled -- see limitations
static_margin_target: [0.02, 0.15]
w_cm0: 5.0
cm0_min: -0.02
w_payload_volume: 1000.0  # payload_volume_margin_m3 is ~1e-3 m^3, scaled up to be comparable to L/D
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
valid, extra)`. Only `HierarchicalGridSearch` is implemented right now, but
Stage 1/2/multi-cycle drivers depend only on this interface — CMA-ES,
Bayesian optimization, differential evolution, or particle swarm could
replace it later without touching those drivers.

### `ParameterSet` (`optimization/vector.py`)

The translation layer between an optimizer's flat vector and
`DesignParameters`. A `ParameterSet` has named, bounded `Var`s and a
`build_fn(x, baseline) -> DesignParameters`. This is what keeps the geometry
module fully independent of the optimizer, per the original design goal —
optimizers never see `DesignParameters` at all.

### Hierarchical grid search (`optimization/hierarchical.py`)

The spec calls for: coarse grid → evaluate all → retain best N → refine
around each → repeat. **A literal full-factorial grid is intractable**
beyond a handful of dimensions — Stage 1 alone has 15 (3 airfoil parameters
× 5 span stations); even 3 points/dimension would be `3^15 ≈ 14 million`
evaluations. "Grid" is instead implemented as a **Latin Hypercube**
space-filling sample at each stage/scale: still deterministic (fixed seed),
still coarse-to-fine with elitist retention, still embarrassingly parallel
(`n_jobs` runs candidates in a `ProcessPoolExecutor`), but tractable at any
dimensionality.

Key parameters (`HierarchicalGridSearch`): `n_stages` (default 4),
`n_samples_per_stage` (40), `retain_best_n` (5), `shrink_factor` (0.4 — each
stage's local search range = previous stage's range × this), `seed`, `n_jobs`.

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
Fixed by reparameterizing:
- **Chord**: `chord_root_m` (bounded `(0.4, 0.9)` — tighter than the general
  chord range, since the root specifically needs to be large enough to have
  any chance of fitting the fuselage box) + 6 non-negative
  `chord_decrement_i` (bounded `(0.0, 0.2)` each). Chord is monotonically
  non-increasing *by construction*.
- **Twist**: `twist_root_deg` + 6 non-negative `washout_increment_i` — same
  idea, washout (twist decreasing outboard) holds by construction.
- **LE offset deviation / Z offset**: these are meant to be free-form
  (non-monotonic) curves, so they use a different fix — `..._root` + a
  **bounded slope** per segment (`value[i] = value[i-1] +
  slope[i]·(y[i]-y[i-1])`), rather than independent per-station values. This
  was needed because some planform stations are very close together (the
  0.08/0.12/0.14 fuselage-break cluster), so independent per-station values
  could jump wildly over a tiny span-fraction gap — exactly what was blowing
  up the leading-edge curvature constraint.

This dropped the random-sample constraint-violation rate from ~100% to
~80%, and real optimization runs now find genuine improvement (see Status).

### Multi-cycle driver (`optimization/cycle.py`)

`run_multi_cycle()` alternates `run_stage1` → `run_stage2` → ... for
`n_cycles` (default 2, 2 stages each), each stage starting from wherever the
previous one left off. Optionally stops early via `convergence_tol` if a
full cycle's improvement falls below that amount.

---

## Visualization

All in `viz/`, all return `plotly.graph_objects.Figure` (so they can be
displayed inline, saved via `.write_html()`, or embedded):

- **`geometry_plots.py`**: interactive 3D aircraft (watertight mesh),
  top/front/side orthographic views, airfoil-section + schedule overlay.
- **`aero_plots.py`**: drag polar + CL/Cm-vs-alpha (via an AeroBuildup alpha
  sweep), spanwise lift/local-CL/Reynolds-number distributions.
- **`structures_plots.py`**: bending moment, shear force, spar depth/width,
  bending stress, safety factor, all vs. span.
- **`optimization_plots.py`**: convergence (best-so-far + per-stage
  best/worst), parameter evolution (best candidate's variables across
  stages), and multi-cycle convergence across stage/cycle boundaries.

Pareto plots are noted in the original spec as future work (once a
multi-objective algorithm exists) and aren't implemented.

---

## The interactive GUI

`flyingwing/gui/app.py`, launched via `scripts/run_gui.py` →
`http://127.0.0.1:8050`. A 4-tab control panel — all tabs share one page
(Dash's static-tabs pattern), so cross-tab actions (e.g. "send this result
to the Design tab") work directly, without a save/reload step.

**Design tab** — the original single-design editor:
- Every design parameter (span, sweep, and all 7 chord/twist/LE/Z + 5
  thickness/camber/reflex values) is a plain numeric input field.
- Changing **any** value immediately regenerates the geometry and updates
  the 3D model, orthographic views, airfoil distribution, and a text panel
  of derived properties (wing area, AR, MAC, fuselage fit, constraint
  validity) — fast (~30 ms), geometry-only, no aero.
- **"Run Full Aerodynamic + Structural Evaluation"** button runs the full
  `evaluate_design` pipeline plus drag-polar/spanwise/structural plots
  (~20 s — a real AeroBuildup alpha sweep + structural analysis — hence
  not run on every keystroke; a loading spinner shows while it runs).

**Bounds & Weights tab** — edits every objective weight
(`configs/objective_weights.yaml`) and every Stage 1/2 search bound plus
structural/mass constant (`configs/bounds_overrides.yaml`, layered over the
hardcoded defaults in `config.py`/`objective/mass.py` — see
[the override mechanism](#how-to-change-parameters-bounds-and-weights)
below). Saved changes take effect the next time an optimizer run is
launched (each run is a fresh subprocess that re-reads both files) or the
GUI is restarted.

**Run Optimizer tab** — pick Stage 1 / Stage 2 / multi-cycle, set the
optimizer's knobs (stages, samples/stage, retain-best-N, seed, parallel
workers), pick a baseline (the built-in default design / whatever's
currently in the Design tab / an existing result), and click Start. This
launches `scripts/run_stage*.py` **as a background subprocess** — the exact
same script the CLI uses, just with different arguments — into a
timestamped `outputs/<type>_run_<timestamp>/` directory, and polls its log
every 2 seconds until it completes or fails. Only one run at a time (the
button is disabled while one is active).

**Results tab** — lists every run directory under `outputs/` that has a
`result.pkl` (both the CLI's fixed names and the GUI's timestamped ones),
and on "Load" re-renders its metrics table (re-evaluating the baseline for
comparison takes a few seconds; everything else is instant) and 3D/
orthographic/airfoil/convergence plots by reusing the same `viz/*`
functions used everywhere else — no recomputation of the optimization
itself. "Send to Design tab" loads that run's best design back into the
Design tab for further hand-tuning.

Note: `y_control` (the span stations each input row corresponds to) is shown
as read-only text above each group — changing the *number* of control
stations isn't exposed in the GUI (see [limitations](#simplifications-and-known-limitations)).

---

## How to change parameters, bounds, and weights

Everything below can be done either by hand-editing the named
file/constants, or from the GUI's **Bounds & Weights** tab (which edits the
same two YAML files) and **Design**/**Run Optimizer** tabs (for the
design/baseline itself) — pick whichever's more convenient.

**To change the objective function's priorities** (e.g. care more about mass,
less about L/D): edit `configs/objective_weights.yaml` (or the GUI's Bounds
& Weights tab), or in Python:

```python
from flyingwing.objective.objective import ObjectiveWeights
weights = ObjectiveWeights(w_mass=6.0, w_cruise_L_over_D=2.0)
weights.to_yaml("configs/objective_weights.yaml")  # persist it
```

Then pass `weights=ObjectiveWeights.from_yaml("configs/objective_weights.yaml")`
into `run_stage1`/`run_stage2`/`run_multi_cycle`, or `score(metrics, weights)`
directly. The CLI scripts do this automatically (`--weights-yaml`, defaulting
to `configs/objective_weights.yaml` if it exists).

**To change the default starting design** (what Stage 1/2 optimize from, and
what the GUI opens with): edit the defaults in `geometry/params.py`
(`AirfoilSchedule` and `Planform` dataclass field defaults), or construct a
custom `DesignParameters`, save it with `geometry/params_io.save_design_parameters`,
and pass its path via `--baseline-yaml` to any `scripts/run_*.py` (the GUI's
Run tab does this for you when you pick a non-default baseline source).

**To change search bounds** (how far the optimizer is allowed to explore) or
**physical/structural assumptions**: edit the GUI's Bounds & Weights tab, or
`configs/bounds_overrides.yaml` directly, or the hardcoded defaults in
`config.py`/`objective/mass.py` (`*_BOUNDS` constants like `CHORD_M_BOUNDS`,
`TWIST_DEG_BOUNDS`, `SWEEP_DEG_BOUNDS`, `CHORD_ROOT_M_BOUNDS`,
`LE_OFFSET_SLOPE_BOUNDS`; structural constants like `ALLOWABLE_SPAR_STRESS_PA`,
`DESIGN_LOAD_FACTOR_G`, `SPAR_WIDTH_FRACTION_CHORD`; fuselage box dimensions
`FUSELAGE_MIN_INTERNAL_*_M`; and mass constants in `objective/mass.py` like
`SHELL_AREAL_DENSITY_KG_M2`). Both `config.py` and `objective/mass.py` load
`configs/bounds_overrides.yaml` at import time via the shared
`_overrides.apply_overrides()` helper and replace any of their constants
named in it — the file is absent by default, so a fresh checkout behaves
exactly like the hardcoded values, and since every optimizer run is a fresh
subprocess, a saved change takes effect on the very next run with no
special reload step needed. These feed directly into
`make_stage1_parameter_set`/`make_stage2_parameter_set`
(`optimization/stage1.py`/`stage2.py`).

**To change the number/position of control stations**: change
`DEFAULT_AIRFOIL_Y_CONTROL` / `DEFAULT_PLANFORM_Y_CONTROL` in
`geometry/params.py`, and update the corresponding tuple lengths in
`AirfoilSchedule`/`Planform` defaults to match. The optimizer parameter sets
(`make_stage1/2_parameter_set`) read the control-station count from the
baseline automatically, so no changes are needed there — but the GUI's input
rows are also generated from the baseline's tuple lengths, so they'll adapt
too.

**To change optimizer search effort**: adjust `HierarchicalGridSearch`
arguments (`n_stages`, `n_samples_per_stage`, `retain_best_n`,
`shrink_factor`, `n_jobs`) in the `scripts/run_*.py` files, or when
constructing your own `HierarchicalGridSearch(...)`.

**To inspect why a design scored the way it did**: `score(metrics,
weights).contributions` is a dict of every term's individual contribution —
print or plot it to see what's driving the total.

---

## Code structure

```
flyingwing/
  config.py                 Global constants: units, target speeds, wingspan/fuselage/
                             structural bounds, Stage 2 search bounds -- see file for full list.
                             Loads configs/bounds_overrides.yaml (if present) at import time.
  _overrides.py              Shared YAML-override-application helper used by config.py and
                              objective/mass.py

  geometry/
    spanwise.py              SpanwiseDistribution (control points -> curve, linear or PCHIP),
                              make_span_stations (cosine spacing)
    params.py                DesignParameters / Planform / AirfoilSchedule -- the complete,
                              optimizer-independent design description
    params_io.py              save_design_parameters()/load_design_parameters(): YAML
                              (de)serialization, used to hand a design across process boundaries
    airfoil_family.py        MH64 decomposition + thickness/camber/reflex modification
    aircraft.py               build_aircraft(): the geometry generator, Aircraft dataclass
    mesh.py                  Watertight triangle mesh construction (mirroring, LE/TE welding, tip caps)
    fuselage.py              Internal fuselage box fit check
    constraints.py            Stage 1 + Stage 2 geometric validity constraints

  analysis/
    airfoil_2d.py             NeuralFoil (+ optional XFoil) 2D section analysis
    aero_3d.py                 AeroBuildup (primary) + VLM (cross-check) + optional AVL
    structures.py              Schrenk lift distribution -> shear/bending/spar/stress/safety factor

  objective/
    mass.py                    Parametric mass estimate (shell + spar + fixed equipment) + payload
                                volume. Loads configs/bounds_overrides.yaml (if present) at import time.
    metrics.py                 evaluate_design(): the one function wiring geometry -> analysis -> DesignMetrics
    objective.py               ObjectiveWeights + score(): DesignMetrics -> scalar score

  optimization/
    base.py                    Optimizer interface, EvaluatedCandidate, OptimizationResult
    vector.py                   ParameterSet / Var: flat-vector <-> DesignParameters translation
    hierarchical.py             HierarchicalGridSearch (Latin-Hypercube coarse-to-fine)
    stage1.py                   Stage 1 parameter set + driver (airfoil schedule)
    stage2.py                   Stage 2 parameter set + driver (planform), with the
                                monotonic/slope-bounded reparameterization
    cycle.py                    Multi-cycle Stage1<->Stage2 driver

  viz/
    geometry_plots.py           3D aircraft, orthographic views, airfoil distribution
    aero_plots.py                Drag polar, spanwise lift/CL/Reynolds distributions
    structures_plots.py          Bending moment, shear, spar sizing, stress, safety factor
    optimization_plots.py        Convergence, parameter evolution, multi-cycle convergence

  gui/
    app.py                     Thin Dash shell combining the 4 tabs below into one page
    design_tab.py               Design tab (live single-design editing -- the original GUI)
    config_tab.py               Bounds & Weights tab
    run_tab.py                   Run Optimizer tab
    results_tab.py               Results tab
    run_manager.py               Subprocess launch/track/poll for the Run Optimizer tab
    results_io.py                Scans outputs/ for result.pkl files, loads them, for the Results tab

configs/
  objective_weights.yaml       Default ObjectiveWeights, editable/reloadable
  bounds_overrides.yaml        Optional; overrides named config.py/objective/mass.py constants
                                when present (absent by default -- created by the GUI's Save Bounds)

scripts/
  run_stage1.py / run_stage2.py / run_multi_cycle.py    Optimization demos, now with argparse CLI
                                                          flags (all optional); write plots +
                                                          result.pkl to outputs/<name>_run/. The
                                                          GUI's Run tab invokes these same scripts
                                                          as subprocesses -- it never duplicates
                                                          this logic.
  run_gui.py                                             Launches the Dash GUI (all 4 tabs)

outputs/                       Generated plots + pickled results (gitignored). run_<n>_run/ from
                                the CLI's fixed names, <type>_run_<timestamp>/ from the GUI.
```

---

## Simplifications and known limitations

These are documented in code comments where they matter, collected here for
visibility:

- **Objective weights and search bounds are not tuned.** They're reasonable
  first guesses calibrated just enough to make the default design valid and
  the optimizer runnable — not a reflection of real design priorities.
  Expect (and plan for) an iteration pass here.
- **`static_margin` is not meaningful yet.** It's computed relative to a
  placeholder reference point (25% of the mean aerodynamic chord), not a
  real center of gravity — there's no mass-distribution/CG model yet. Its
  objective weight defaults to 0 for exactly this reason. Don't trust it as
  a stability indicator until a real CG model exists.
- **The structural proxy is a ranking heuristic, not FEA.** Schrenk's
  approximation + a simplified thin-wall spar box are fast and good enough
  to *compare* designs, not to certify one.
- **No endurance/battery/motor model.** "Long endurance" is only
  approximated indirectly, via the combination of L/D and mass in the
  objective — there's no propulsion or battery-energy model to compute an
  actual flight-time estimate.
- **The hierarchical grid search is Latin-Hypercube sampling, not a literal
  grid.** A full-factorial grid is intractable beyond a few dimensions (see
  [Optimization](#optimization)). This means results are somewhat sensitive
  to the random seed and sample count, especially in Stage 2's 30-dimensional
  space — more samples/stages (or a smarter algorithm) will do better. The
  `Optimizer` interface exists specifically so CMA-ES / Bayesian
  optimization / differential evolution / particle swarm can be dropped in
  later without touching the Stage 1/2/multi-cycle drivers.
- **Even with the monotonic/slope-bounded reparameterization, Stage 2's
  random samples are only valid ~20% of the time** (vs. ~0% before the fix).
  The dominant remaining failure modes are the fuselage-fit and
  minimum-local-thickness constraints interacting with the (fixed) airfoil
  schedule — a genuine, physically real trade-off, not a parameterization
  bug, but it does mean Stage 2 spends a meaningful fraction of its budget
  on infeasible candidates.
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
- **The GUI doesn't expose changing the number/position of control
  stations** (`y_control`) — still a code edit (`geometry/params.py`); only
  the *values* at the existing stations are editable, whether by hand or
  from the GUI.
- **Only one optimizer run at a time from the GUI.** The Run Optimizer tab
  refuses to start a second run while one is active — a deliberate,
  simple-and-safe default (each run already uses multiprocessing
  internally) rather than a job queue. Starting a run separately from a
  terminal while the GUI has one active isn't tracked by the GUI either
  (it only tracks the one it launched itself).
- **Bounds/weights edits from the GUI don't hot-reload the GUI's own
  in-memory constants** — they're written to
  `configs/bounds_overrides.yaml`/`configs/objective_weights.yaml` and take
  effect the next time an optimizer subprocess is launched (each one
  re-reads both files fresh) or the GUI itself is restarted. This only
  matters for the GUI process's own Design-tab preview; it's not a
  limitation for optimizer runs, which always pick up the latest saved values.
- **No STL/CAD export.** The watertight mesh (`geometry/mesh.py`) is built
  for visualization and is genuinely manifold, but there's no export path to
  a file format a slicer/CAD tool would consume.

---

## What's not implemented / next steps

- **Weight/bound tuning** to get Stage 2 and multi-cycle runs converging to
  recognizably sensible FPV-wing designs rather than extreme ones (the
  highest-priority next step given current results).
- **Mass-distribution/CG model**, to make `static_margin` meaningful and
  enable trim/stability-driven optimization.
- **A real endurance model** (battery capacity, motor/prop efficiency,
  Breguet-style range/endurance equation) if endurance is to be optimized
  directly rather than via L/D and mass as proxies.
- **A second optimization algorithm** (CMA-ES is a natural first candidate,
  given the continuous, moderately-high-dimensional, constrained search
  space) implementing the existing `Optimizer` interface, to compare against
  the hierarchical grid search and likely do meaningfully better in Stage 2's
  30-dimensional space.
- **Pareto-front visualization**, once a multi-objective algorithm exists
  (noted as future work in the original spec).
- **Multi-station / multi-condition 2D analysis** (sweep root/mid/tip
  sections, and the clean/moderate/rough transition assumptions) for a
  fuller picture of stall behavior, fed into the objective function.
- **GUI enhancements**: exposing control-station count, search bounds, and
  objective weights as editable GUI controls rather than requiring a code
  edit + restart.
