"""Design tab: live single-design parameter editing.

Every design parameter (planform + airfoil schedule) is a numeric input
field; the 3D model, orthographic views, airfoil distribution, and derived
geometric properties update immediately, since geometry generation is fast
(tens of milliseconds). A full aerodynamic + structural evaluation (~20 s)
only runs when explicitly requested via the "Run Evaluation" button -- far
too slow to re-run on every keystroke.

`build_params_from_inputs` and `DESIGN_STATE_INPUTS` are exported for
`run_tab.py`, so a Run can be launched using "whatever's currently in the
Design tab" as its baseline.
"""
from __future__ import annotations

import dash
from dash import dcc, html, Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from ..geometry.params import (
    DesignParameters, Planform, AirfoilSchedule,
    DEFAULT_PLANFORM_Y_CONTROL, DEFAULT_AIRFOIL_Y_CONTROL,
)
from ..geometry.aircraft import build_aircraft
from ..geometry.constraints import check_all_constraints
from ..viz.geometry_plots import plot_3d_aircraft, plot_orthographic_views, plot_airfoil_distribution
from ..viz.aero_plots import plot_drag_polar, plot_spanwise_distributions
from ..viz.structures_plots import plot_structures
from ..analysis.aero_3d import analyze_aerobuildup
from ..analysis.structures import analyze_structures
from ..objective.metrics import evaluate_design
from ..objective.objective import score
from ..config import CRUISE_SPEED_MS

_DEFAULT = DesignParameters()

_INPUT_STYLE = {"width": "70px", "marginRight": "4px"}
_LABEL_STYLE = {"width": "150px", "display": "inline-block", "fontWeight": "bold"}


def _input_row(label: str, id_type: str, values, step: float) -> html.Div:
    """A labeled row of compact numeric inputs, one per span-control station."""
    return html.Div(
        [
            html.Label(label, style=_LABEL_STYLE),
            *[
                dcc.Input(id={"type": id_type, "index": i}, type="number", value=v, step=step, style=_INPUT_STYLE)
                for i, v in enumerate(values)
            ],
        ],
        style={"marginBottom": "6px"},
    )


def layout() -> html.Div:
    p = _DEFAULT.planform
    a = _DEFAULT.airfoil_schedule

    controls = html.Div(
        [
            html.H2("Design"),
            html.H4("Planform"),
            html.Div(
                [
                    html.Label("Span (m)", style=_LABEL_STYLE),
                    dcc.Input(id="span-input", type="number", value=p.span_m, step=0.01, style=_INPUT_STYLE),
                    html.Label("Sweep (deg)", style={**_LABEL_STYLE, "marginLeft": "20px"}),
                    dcc.Input(id="sweep-input", type="number", value=p.sweep_deg, step=0.5, style=_INPUT_STYLE),
                ],
                style={"marginBottom": "10px"},
            ),
            html.Div(f"y control stations: {p.y_control}", style={"fontSize": "12px", "color": "#666", "marginBottom": "8px"}),
            _input_row("Chord (m)", "chord-input", p.chord_m, 0.005),
            _input_row("Twist (deg)", "twist-input", p.twist_deg, 0.1),
            _input_row("LE offset dev. (m)", "le-input", p.le_offset_deviation_m, 0.005),
            _input_row("Z offset (m)", "z-input", p.z_offset_m, 0.005),
            html.H4("Airfoil Schedule", style={"marginTop": "20px"}),
            html.Div(f"y control stations: {a.y_control}", style={"fontSize": "12px", "color": "#666", "marginBottom": "8px"}),
            _input_row("Thickness scale", "thickness-input", a.thickness_scale, 0.02),
            _input_row("Camber scale", "camber-input", a.camber_scale, 0.02),
            _input_row("Reflex scale", "reflex-input", a.reflex_scale, 0.02),
            html.Div(id="properties-display", style={"marginTop": "16px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"}),
            html.Button("Run Full Aerodynamic + Structural Evaluation (~20s)", id="evaluate-button", n_clicks=0, style={"marginTop": "16px"}),
            dcc.Loading(html.Div(id="evaluation-display", style={"marginTop": "10px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"})),
        ],
        style={"width": "420px", "display": "inline-block", "verticalAlign": "top", "padding": "10px", "overflowY": "auto", "maxHeight": "95vh"},
    )

    graphs = html.Div(
        [
            dcc.Graph(id="graph-3d", style={"height": "500px"}),
            dcc.Graph(id="graph-ortho", style={"height": "400px"}),
            dcc.Graph(id="graph-airfoil", style={"height": "600px"}),
            dcc.Loading(dcc.Graph(id="graph-drag-polar", style={"height": "400px"})),
            dcc.Loading(dcc.Graph(id="graph-spanwise", style={"height": "400px"})),
            dcc.Loading(dcc.Graph(id="graph-structures", style={"height": "600px"})),
        ],
        style={"width": "calc(100% - 460px)", "display": "inline-block", "verticalAlign": "top"},
    )

    return html.Div([controls, graphs])


def build_params_from_inputs(span, sweep, chord, twist, le, z, thickness, camber, reflex) -> DesignParameters:
    planform = Planform(
        span_m=span, sweep_deg=sweep, y_control=DEFAULT_PLANFORM_Y_CONTROL,
        chord_m=tuple(chord), twist_deg=tuple(twist),
        le_offset_deviation_m=tuple(le), z_offset_m=tuple(z),
    )
    airfoil_schedule = AirfoilSchedule(
        y_control=DEFAULT_AIRFOIL_Y_CONTROL,
        thickness_scale=tuple(thickness), camber_scale=tuple(camber), reflex_scale=tuple(reflex),
    )
    return DesignParameters(planform=planform, airfoil_schedule=airfoil_schedule)


# Same component IDs, same order as build_params_from_inputs's positional
# args -- used both by this tab's own callbacks (as Input) and by
# run_tab.py (as State, to read "current Design tab values" as a Run
# baseline without needing a Store round-trip, since all tabs share one
# static layout).
_GEOMETRY_IDS = [
    ("span-input", False),
    ("sweep-input", False),
    ({"type": "chord-input", "index": ALL}, True),
    ({"type": "twist-input", "index": ALL}, True),
    ({"type": "le-input", "index": ALL}, True),
    ({"type": "z-input", "index": ALL}, True),
    ({"type": "thickness-input", "index": ALL}, True),
    ({"type": "camber-input", "index": ALL}, True),
    ({"type": "reflex-input", "index": ALL}, True),
]
_GEOMETRY_INPUTS = [Input(id_, "value") for id_, _ in _GEOMETRY_IDS]
DESIGN_STATE_INPUTS = [State(id_, "value") for id_, _ in _GEOMETRY_IDS]


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("graph-3d", "figure"),
        Output("graph-ortho", "figure"),
        Output("graph-airfoil", "figure"),
        Output("properties-display", "children"),
        *_GEOMETRY_INPUTS,
    )
    def update_geometry(span, sweep, chord, twist, le, z, thickness, camber, reflex):
        all_values = [span, sweep, *chord, *twist, *le, *z, *thickness, *camber, *reflex]
        if any(v is None for v in all_values):
            raise PreventUpdate

        params = build_params_from_inputs(span, sweep, chord, twist, le, z, thickness, camber, reflex)
        aircraft = build_aircraft(params)
        constraints = check_all_constraints(aircraft)

        props = (
            f"Wing area:    {aircraft.wing_area_m2:.4f} m^2\n"
            f"Aspect ratio: {aircraft.aspect_ratio:.3f}\n"
            f"MAC:          {aircraft.mean_aerodynamic_chord_m * 1000:.1f} mm\n"
            f"Root chord:   {aircraft.root_chord_m * 1000:.1f} mm\n"
            f"Tip chord:    {aircraft.tip_chord_m * 1000:.1f} mm\n"
            f"Fuselage fit: {'OK' if aircraft.fuselage_fit.fits else 'FAILS'} "
            f"(height margin {aircraft.fuselage_fit.min_height_margin_m * 1000:+.1f} mm, "
            f"length margin {aircraft.fuselage_fit.min_length_margin_m * 1000:+.1f} mm)\n"
            f"Constraints:  {'valid' if constraints.valid else 'INVALID'}\n"
        )
        if not constraints.valid:
            props += "Violations:\n" + "\n".join(f"  - {v}" for v in constraints.violations)

        return (
            plot_3d_aircraft(aircraft),
            plot_orthographic_views(aircraft),
            plot_airfoil_distribution(aircraft),
            props,
        )

    @app.callback(
        Output("evaluation-display", "children"),
        Output("graph-drag-polar", "figure"),
        Output("graph-spanwise", "figure"),
        Output("graph-structures", "figure"),
        Input("evaluate-button", "n_clicks"),
        *DESIGN_STATE_INPUTS,
        prevent_initial_call=True,
    )
    def run_evaluation(n_clicks, span, sweep, chord, twist, le, z, thickness, camber, reflex):
        params = build_params_from_inputs(span, sweep, chord, twist, le, z, thickness, camber, reflex)
        metrics = evaluate_design(params)
        result = score(metrics)

        aircraft = build_aircraft(params)
        cruise_aero = analyze_aerobuildup(aircraft, CRUISE_SPEED_MS, alpha_deg=2.0)
        structures = analyze_structures(aircraft, cruise_trim_cl=cruise_aero.trim_CL, speed_ms=CRUISE_SPEED_MS)

        text = (
            f"Objective score: {result.score:.2f}   (valid={metrics.valid})\n\n"
            f"Cruise L/D:   {metrics.cruise_L_over_D:.2f}   (trim alpha {metrics.cruise_trim_alpha_deg:.2f} deg)\n"
            f"Fast L/D:     {metrics.fast_L_over_D:.2f}\n"
            f"Root CLmax:   {metrics.root_cl_max:.3f}\n"
            f"Static margin (vs placeholder xyz_ref): {metrics.static_margin:.3f}\n"
            f"Min safety factor: {metrics.min_safety_factor:.1f}\n"
            f"Structural mass:   {metrics.total_structural_mass_kg:.3f} kg\n"
            f"Payload volume margin: {metrics.payload_volume_margin_m3 * 1e6:.0f} cm^3\n"
        )
        if not metrics.valid:
            text += "\nConstraint violations:\n" + "\n".join(f"  - {v}" for v in metrics.constraint_violations)

        return (
            text,
            plot_drag_polar(aircraft, CRUISE_SPEED_MS),
            plot_spanwise_distributions(aircraft, CRUISE_SPEED_MS, cruise_aero.trim_CL),
            plot_structures(structures),
        )
