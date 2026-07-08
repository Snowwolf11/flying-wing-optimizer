"""Bounds & Weights tab: edit objective weights and search-bound/structural
constants, persisted to YAML (`configs/objective_weights.yaml` and
`configs/bounds_overrides.yaml`).

Changes here don't affect the currently-running GUI process's own
constants (those are only read at import time) -- they take effect the
next time an optimizer run is launched (each run is a fresh subprocess that
re-reads both YAML files) or the next time the GUI itself is restarted.
"""
from __future__ import annotations

from dataclasses import asdict

import dash
from dash import dcc, html, Input, Output, State

import flyingwing.config as config
import flyingwing.objective.mass as mass
import flyingwing.objective.cg as cg
import flyingwing.objective.performance as performance
import flyingwing.optimization.stage1 as stage1_module
from ..objective.objective import ObjectiveWeights
from ..optimization.vector import resolve_per_station_bounds
from ..geometry.params import DEFAULT_PLANFORM_Y_CONTROL, DEFAULT_AIRFOIL_Y_CONTROL

WEIGHTS_YAML = "configs/objective_weights.yaml"
BOUNDS_YAML = "configs/bounds_overrides.yaml"

_N_PLANFORM_STATIONS = len(DEFAULT_PLANFORM_Y_CONTROL)
_N_AIRFOIL_STATIONS = len(DEFAULT_AIRFOIL_Y_CONTROL)

# (attribute name, display label, step) -- generated programmatically so the
# layout and the save callback's State list are built from the same list
# and can never drift out of sync with each other.
_WEIGHT_SCALARS = [
    ("w_cruise_L_over_D", "Cruise L/D weight", 0.1),
    ("w_fast_L_over_D", "Fast L/D weight", 0.1),
    ("w_root_cl_max", "Root CLmax weight", 0.1),
    ("w_mass", "Mass weight", 0.1),
    ("w_safety_factor", "Safety factor weight", 0.01),
    ("safety_factor_min", "Safety factor threshold", 0.1),
    ("w_static_margin", "Static margin weight", 0.1),
    ("w_cm0", "Root Cm0 weight", 0.1),
    ("cm0_min", "Root Cm0 threshold", 0.01),
    ("w_payload_volume", "Payload volume weight", 10.0),
    ("invalid_penalty", "Invalid-design flat penalty", 10.0),
    ("constraint_penalty_scale", "Constraint penalty scale", 10.0),
]
_WEIGHT_RANGE = ("static_margin_target", "Static margin target range")

_CONFIG_SCALARS = [
    ("DESIGN_LOAD_FACTOR_G", "Design load factor (g)", 0.5),
    ("ALLOWABLE_SPAR_STRESS_PA", "Allowable spar stress (Pa)", 1e6),
    ("SPAR_WIDTH_FRACTION_CHORD", "Spar width (fraction chord)", 0.01),
    ("SPAR_DEPTH_FRACTION_THICKNESS", "Spar depth (fraction thickness)", 0.01),
    ("SPAR_WALL_THICKNESS_M", "Spar wall thickness (m)", 0.0005),
    ("MIN_ABSOLUTE_THICKNESS_M", "Min local thickness (m)", 0.0005),
    ("MIN_SPAR_DEPTH_M", "Min spar depth (m)", 0.0005),
    ("MAX_LE_CURVATURE_PER_M", "Max LE curvature (1/m)", 5.0),
    ("MAX_Z_CURVATURE_PER_M", "Max vertical (winglet) curvature (1/m)", 5.0),
    ("Z_OFFSET_TIP_SEGMENT_Y_THRESHOLD", "Winglet region start (y, normalized span)", 0.05),
    ("Z_OFFSET_TIP_MIN_M", "Winglet region min Z offset (m)", 0.01),
    ("MAX_LE_OFFSET_SLOPE_M_PER_SPAN", "Max LE offset slope (m/span, advanced)", 0.1),
    ("MAX_Z_OFFSET_SLOPE_M_PER_SPAN", "Max Z offset slope (m/span, advanced)", 0.1),
    ("FUSELAGE_MIN_INTERNAL_WIDTH_M", "Fuselage min width (m)", 0.005),
    ("FUSELAGE_MIN_INTERNAL_HEIGHT_M", "Fuselage min height (m)", 0.005),
    ("FUSELAGE_MIN_INTERNAL_LENGTH_M", "Fuselage min length (m)", 0.01),
    ("WINGSPAN_MIN_M", "Wingspan min (m)", 0.01),
    ("WINGSPAN_MAX_M", "Wingspan max (m)", 0.01),
]
_MASS_SCALARS = [
    ("SHELL_AREAL_DENSITY_KG_M2", "Shell areal density (kg/m^2)", 0.01),
    ("SPAR_MATERIAL_DENSITY_KG_M3", "Spar material density (kg/m^3)", 10.0),
    ("FIXED_EQUIPMENT_MASS_KG", "Fixed equipment mass (kg)", 0.01),
]
_CG_SCALARS = [
    ("MOTOR_ESC_MASS_FRACTION", "Motor/ESC mass fraction (of fixed equipment)", 0.01),
    ("AVIONICS_MASS_FRACTION", "Avionics mass fraction (of fixed equipment)", 0.01),
    ("SERVO_MASS_FRACTION", "Servo mass fraction (of fixed equipment)", 0.01),
    ("MOTOR_ESC_X_FRACTION_CHORD", "Motor/ESC x-position (fraction root chord)", 0.01),
    ("AVIONICS_X_FRACTION_CHORD", "Avionics x-position (fraction root chord)", 0.01),
    ("SERVO_X_FRACTION_CHORD", "Servo x-position (fraction root chord)", 0.01),
    ("SHELL_CENTROID_X_FRACTION_CHORD", "Shell centroid x (fraction local chord)", 0.01),
    ("SPAR_X_FRACTION_CHORD", "Spar x-position (fraction local chord)", 0.01),
    ("BATTERY_MASS_KG", "Battery mass (kg)", 0.01),
    ("BATTERY_X_FRACTION_CHORD", "Battery assumed x-position (fraction root chord)", 0.01),
]
_PERFORMANCE_SCALARS = [
    ("BATTERY_CAPACITY_MAH", "Battery capacity (mAh)", 50.0),
    ("BATTERY_VOLTAGE_V", "Battery voltage (V)", 0.1),
    ("BATTERY_USABLE_FRACTION", "Battery usable fraction", 0.05),
    ("PROPULSIVE_EFFICIENCY", "Propulsive efficiency (motor+ESC+prop)", 0.05),
]
_CONFIG_BOUNDS = [
    ("SWEEP_DEG_BOUNDS", "Sweep (deg)", 0.5),
]

# Bounds editable per control-point/station rather than as one uniform
# range -- one row per Stage 1 airfoil station or Stage 2 planform station,
# sized to the built-in default baseline's station count. (If a Design-tab
# baseline with a *different* station count is used to launch a run,
# vector.resolve_per_station_bounds raises a clear error rather than
# silently misapplying bounds -- see its docstring.) The planform ones are
# bounds on the actual value at each station (chord in m, twist in deg, LE
# offset in m, Z offset in m) -- optimization/stage2.py translates these
# into the underlying decrement/slope search variables that keep chord/
# twist monotonic and LE/Z offset curvature-bounded by construction.
_PER_STATION_BOUNDS = [
    ("THICKNESS_SCALE_BOUNDS", "Thickness scale", 0.01, stage1_module, _N_AIRFOIL_STATIONS),
    ("CAMBER_SCALE_BOUNDS", "Camber scale", 0.01, stage1_module, _N_AIRFOIL_STATIONS),
    ("REFLEX_SCALE_BOUNDS", "Reflex scale", 0.01, stage1_module, _N_AIRFOIL_STATIONS),
    ("CHORD_STATION_M_BOUNDS", "Chord (m)", 0.005, config, _N_PLANFORM_STATIONS),
    ("TWIST_STATION_DEG_BOUNDS", "Twist (deg)", 0.1, config, _N_PLANFORM_STATIONS),
    ("LE_OFFSET_STATION_M_BOUNDS", "LE offset (m)", 0.005, config, _N_PLANFORM_STATIONS),
    ("Z_OFFSET_STATION_M_BOUNDS", "Z offset (m)", 0.005, config, _N_PLANFORM_STATIONS),
]

_INPUT_STYLE = {"width": "100%", "fontSize": "12px", "boxSizing": "border-box"}
_GRID_LABEL_STYLE = {"fontSize": "11px", "display": "block", "color": "#555", "marginBottom": "1px"}
_GRID_STYLE = {"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(150px, 1fr))", "gap": "4px 10px", "marginBottom": "6px"}
_SECTION_STYLE = {"border": "1px solid #ddd", "borderRadius": "4px", "padding": "8px 10px", "marginBottom": "8px"}
_SUMMARY_STYLE = {"fontWeight": "bold", "cursor": "pointer", "fontSize": "14px"}
_TABLE_INPUT_STYLE = {"width": "56px", "fontSize": "11px"}
_TD_STYLE = {"padding": "1px 3px"}
_TH_STYLE = {"padding": "1px 5px", "textAlign": "left", "fontSize": "11px", "color": "#555"}


def _scalar_cell(id_prefix: str, name: str, label: str, value: float, step: float) -> html.Div:
    return html.Div([
        html.Label(label, style=_GRID_LABEL_STYLE),
        dcc.Input(id=f"{id_prefix}-{name}", type="number", value=value, step=step, style=_INPUT_STYLE),
    ])


def _scalar_grid(id_prefix: str, items: list[tuple[str, str, float]], getter) -> html.Div:
    return html.Div(
        [_scalar_cell(id_prefix, name, label, getter(name), step) for name, label, step in items],
        style=_GRID_STYLE,
    )


def _bounds_row(id_prefix: str, name: str, label: str, lo: float, hi: float, step: float) -> html.Div:
    return html.Div([
        html.Label(label, style=_GRID_LABEL_STYLE),
        html.Div([
            dcc.Input(id=f"{id_prefix}-{name}-lo", type="number", value=lo, step=step, style={**_INPUT_STYLE, "width": "70px", "marginRight": "4px"}),
            dcc.Input(id=f"{id_prefix}-{name}-hi", type="number", value=hi, step=step, style={**_INPUT_STYLE, "width": "70px"}),
        ], style={"display": "flex"}),
    ])


def _per_station_bound_table(id_prefix: str, name: str, label: str, module, n: int, step: float) -> html.Details:
    """A compact table (2 rows: lo/hi, one column per station) inside a
    collapsible <details> -- replaces one full-width row per station, which
    made this section by far the tallest part of the tab."""
    pairs = resolve_per_station_bounds(getattr(module, name), n)
    header = html.Tr([html.Th("", style=_TH_STYLE)] + [html.Th(f"st.{i}", style=_TH_STYLE) for i in range(n)])
    lo_row = html.Tr(
        [html.Th("lo", style=_TH_STYLE)]
        + [html.Td(dcc.Input(id=f"{id_prefix}-{name}-lo-{i}", type="number", value=lo, step=step, style=_TABLE_INPUT_STYLE), style=_TD_STYLE) for i, (lo, hi) in enumerate(pairs)]
    )
    hi_row = html.Tr(
        [html.Th("hi", style=_TH_STYLE)]
        + [html.Td(dcc.Input(id=f"{id_prefix}-{name}-hi-{i}", type="number", value=hi, step=step, style=_TABLE_INPUT_STYLE), style=_TD_STYLE) for i, (lo, hi) in enumerate(pairs)]
    )
    table = html.Table(html.Tbody([header, lo_row, hi_row]), style={"borderCollapse": "collapse"})
    return html.Details(
        [html.Summary(label, style={"fontSize": "12px", "cursor": "pointer"}), html.Div(table, style={"overflowX": "auto", "marginTop": "4px"})],
        open=False, style={"marginBottom": "4px"},
    )


def _section(title: str, children: list, open_: bool = True) -> html.Details:
    return html.Details([html.Summary(title, style=_SUMMARY_STYLE), html.Div(children, style={"marginTop": "8px"})], open=open_, style=_SECTION_STYLE)


def _load_current_weights() -> ObjectiveWeights:
    try:
        return ObjectiveWeights.from_yaml(WEIGHTS_YAML)
    except FileNotFoundError:
        return ObjectiveWeights()


def layout() -> html.Div:
    w = _load_current_weights()
    lo, hi = w.static_margin_target

    weights_section = _section("Objective weights", [
        _scalar_grid("weight", _WEIGHT_SCALARS, lambda name: getattr(w, name)),
        _bounds_row("weight", _WEIGHT_RANGE[0], _WEIGHT_RANGE[1], lo, hi, 0.01),
        html.Button("Save Weights", id="save-weights-button", n_clicks=0, style={"marginTop": "8px"}),
        html.Div(id="save-weights-status", style={"marginTop": "4px", "fontFamily": "monospace", "fontSize": "12px"}),
    ])

    structural_section = _section("Structural / mass / CG / performance constants", [
        _scalar_grid("const", _CONFIG_SCALARS, lambda name: getattr(config, name)),
        _scalar_grid("const", _MASS_SCALARS, lambda name: getattr(mass, name)),
        _scalar_grid("const", _CG_SCALARS, lambda name: getattr(cg, name)),
        _scalar_grid("const", _PERFORMANCE_SCALARS, lambda name: getattr(performance, name)),
    ], open_=False)

    bounds_section = _section("Stage 2 search bounds", [
        html.Div([_bounds_row("bound", name, label, *getattr(config, name), step) for name, label, step in _CONFIG_BOUNDS], style=_GRID_STYLE),
        html.Div(
            [_per_station_bound_table("bound", name, label, module, n, step) for name, label, step, module, n in _PER_STATION_BOUNDS],
            style={"marginTop": "6px"},
        ),
    ], open_=False)

    return html.Div(
        [
            html.H2("Bounds & Weights"),
            html.P(
                "Saved changes take effect the next time an optimizer run is launched "
                "(each run re-reads these YAML files fresh) or the next time the GUI is restarted. "
                "Click a section header to expand/collapse it; per-station bound tables are "
                "collapsed by default.",
                style={"fontStyle": "italic", "color": "#666", "fontSize": "13px"},
            ),
            weights_section,
            structural_section,
            bounds_section,
            html.Button("Save Bounds", id="save-bounds-button", n_clicks=0, style={"marginTop": "6px"}),
            html.Div(id="save-bounds-status", style={"marginTop": "4px", "fontFamily": "monospace", "fontSize": "12px"}),
        ],
        style={"padding": "10px", "maxWidth": "1100px"},
    )


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("save-weights-status", "children"),
        Input("save-weights-button", "n_clicks"),
        [State(f"weight-{name}", "value") for name, _, _ in _WEIGHT_SCALARS],
        State(f"weight-{_WEIGHT_RANGE[0]}-lo", "value"),
        State(f"weight-{_WEIGHT_RANGE[0]}-hi", "value"),
        prevent_initial_call=True,
    )
    def save_weights(n_clicks, *values):
        scalar_values = values[: len(_WEIGHT_SCALARS)]
        lo, hi = values[len(_WEIGHT_SCALARS):]

        kwargs = {name: v for (name, _, _), v in zip(_WEIGHT_SCALARS, scalar_values)}
        kwargs["static_margin_target"] = (lo, hi)
        weights = ObjectiveWeights(**kwargs)
        weights.to_yaml(WEIGHTS_YAML)
        return f"Saved to {WEIGHTS_YAML}"

    @app.callback(
        Output("save-bounds-status", "children"),
        Input("save-bounds-button", "n_clicks"),
        [State(f"const-{name}", "value") for name, _, _ in _CONFIG_SCALARS]
        + [State(f"const-{name}", "value") for name, _, _ in _MASS_SCALARS]
        + [State(f"const-{name}", "value") for name, _, _ in _CG_SCALARS]
        + [State(f"const-{name}", "value") for name, _, _ in _PERFORMANCE_SCALARS]
        + [s for name, _, _ in _CONFIG_BOUNDS for s in (State(f"bound-{name}-lo", "value"), State(f"bound-{name}-hi", "value"))]
        + [
            s
            for name, _, _, _, n in _PER_STATION_BOUNDS
            for i in range(n)
            for s in (State(f"bound-{name}-lo-{i}", "value"), State(f"bound-{name}-hi-{i}", "value"))
        ],
        prevent_initial_call=True,
    )
    def save_bounds(n_clicks, *values):
        i = 0
        data = {}
        for name, _, _ in _CONFIG_SCALARS:
            data[name] = values[i]; i += 1
        for name, _, _ in _MASS_SCALARS:
            data[name] = values[i]; i += 1
        for name, _, _ in _CG_SCALARS:
            data[name] = values[i]; i += 1
        for name, _, _ in _PERFORMANCE_SCALARS:
            data[name] = values[i]; i += 1
        for name, _, _ in _CONFIG_BOUNDS:
            lo, hi = values[i], values[i + 1]; i += 2
            data[name] = [lo, hi]
        for name, _, _, _, n in _PER_STATION_BOUNDS:
            pairs = []
            for _station in range(n):
                lo, hi = values[i], values[i + 1]; i += 2
                pairs.append([lo, hi])
            data[name] = pairs

        import yaml
        with open(BOUNDS_YAML, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return f"Saved to {BOUNDS_YAML}"
