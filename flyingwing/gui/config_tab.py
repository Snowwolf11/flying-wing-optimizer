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
from ..objective.objective import ObjectiveWeights

WEIGHTS_YAML = "configs/objective_weights.yaml"
BOUNDS_YAML = "configs/bounds_overrides.yaml"

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
_CONFIG_BOUNDS = [
    ("SWEEP_DEG_BOUNDS", "Sweep (deg)", 0.5),
    ("CHORD_M_BOUNDS", "Chord, general (m)", 0.005),
    ("CHORD_ROOT_M_BOUNDS", "Chord, root (m)", 0.005),
    ("CHORD_DECREMENT_M_BOUNDS", "Chord decrement / segment (m)", 0.005),
    ("TWIST_DEG_BOUNDS", "Twist, general (deg)", 0.1),
    ("TWIST_ROOT_DEG_BOUNDS", "Twist, root (deg)", 0.1),
    ("WASHOUT_INCREMENT_DEG_BOUNDS", "Washout increment / segment (deg)", 0.1),
    ("LE_OFFSET_DEVIATION_M_BOUNDS", "LE offset deviation (m)", 0.005),
    ("LE_OFFSET_SLOPE_BOUNDS", "LE offset slope (m/span)", 0.1),
    ("Z_OFFSET_M_BOUNDS", "Z offset (m)", 0.005),
    ("Z_OFFSET_SLOPE_BOUNDS", "Z offset slope (m/span)", 0.1),
]

_INPUT_STYLE = {"width": "110px", "marginRight": "8px"}
_LABEL_STYLE = {"width": "260px", "display": "inline-block"}


def _scalar_row(id_prefix: str, name: str, label: str, value: float, step: float) -> html.Div:
    return html.Div(
        [
            html.Label(label, style=_LABEL_STYLE),
            dcc.Input(id=f"{id_prefix}-{name}", type="number", value=value, step=step, style=_INPUT_STYLE),
        ],
        style={"marginBottom": "4px"},
    )


def _bounds_row(id_prefix: str, name: str, label: str, lo: float, hi: float, step: float) -> html.Div:
    return html.Div(
        [
            html.Label(label, style=_LABEL_STYLE),
            dcc.Input(id=f"{id_prefix}-{name}-lo", type="number", value=lo, step=step, style=_INPUT_STYLE),
            dcc.Input(id=f"{id_prefix}-{name}-hi", type="number", value=hi, step=step, style=_INPUT_STYLE),
        ],
        style={"marginBottom": "4px"},
    )


def _load_current_weights() -> ObjectiveWeights:
    try:
        return ObjectiveWeights.from_yaml(WEIGHTS_YAML)
    except FileNotFoundError:
        return ObjectiveWeights()


def layout() -> html.Div:
    w = _load_current_weights()

    weight_rows = [
        _scalar_row("weight", name, label, getattr(w, name), step) for name, label, step in _WEIGHT_SCALARS
    ]
    lo, hi = w.static_margin_target
    range_row = _bounds_row("weight", _WEIGHT_RANGE[0], _WEIGHT_RANGE[1], lo, hi, 0.01)

    config_rows = [
        _scalar_row("const", name, label, getattr(config, name), step) for name, label, step in _CONFIG_SCALARS
    ]
    mass_rows = [
        _scalar_row("const", name, label, getattr(mass, name), step) for name, label, step in _MASS_SCALARS
    ]
    bound_rows = [
        _bounds_row("bound", name, label, *getattr(config, name), step) for name, label, step in _CONFIG_BOUNDS
    ]

    return html.Div(
        [
            html.H2("Bounds & Weights"),
            html.P(
                "Saved changes take effect the next time an optimizer run is launched "
                "(each run re-reads these YAML files fresh) or the next time the GUI is restarted.",
                style={"fontStyle": "italic", "color": "#666"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Objective weights"),
                            *weight_rows,
                            range_row,
                            html.Button("Save Weights", id="save-weights-button", n_clicks=0, style={"marginTop": "10px"}),
                            html.Div(id="save-weights-status", style={"marginTop": "6px", "fontFamily": "monospace"}),
                        ],
                        style={"width": "420px", "display": "inline-block", "verticalAlign": "top", "marginRight": "40px"},
                    ),
                    html.Div(
                        [
                            html.H4("Structural / mass constants"),
                            *config_rows,
                            *mass_rows,
                            html.H4("Stage 2 search bounds", style={"marginTop": "16px"}),
                            *bound_rows,
                            html.Button("Save Bounds", id="save-bounds-button", n_clicks=0, style={"marginTop": "10px"}),
                            html.Div(id="save-bounds-status", style={"marginTop": "6px", "fontFamily": "monospace"}),
                        ],
                        style={"width": "480px", "display": "inline-block", "verticalAlign": "top"},
                    ),
                ]
            ),
        ],
        style={"padding": "10px"},
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
        + [s for name, _, _ in _CONFIG_BOUNDS for s in (State(f"bound-{name}-lo", "value"), State(f"bound-{name}-hi", "value"))],
        prevent_initial_call=True,
    )
    def save_bounds(n_clicks, *values):
        i = 0
        data = {}
        for name, _, _ in _CONFIG_SCALARS:
            data[name] = values[i]; i += 1
        for name, _, _ in _MASS_SCALARS:
            data[name] = values[i]; i += 1
        for name, _, _ in _CONFIG_BOUNDS:
            lo, hi = values[i], values[i + 1]; i += 2
            data[name] = [lo, hi]

        import yaml
        with open(BOUNDS_YAML, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return f"Saved to {BOUNDS_YAML}"
