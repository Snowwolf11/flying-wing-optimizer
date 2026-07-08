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

import numpy as np

import dash
from dash import dcc, html, Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from ..geometry.params import DesignParameters, Planform, AirfoilSchedule
from ..geometry.params_io import save_design_parameters, load_default_design_parameters, DEFAULT_DESIGN_YAML
from ..geometry.spanwise import SpanwiseDistribution
from ..geometry.aircraft import build_aircraft
from ..geometry.constraints import check_all_constraints
from ..geometry.mesh import build_watertight_mesh
from ..geometry.export import write_stl
from ..viz.geometry_plots import plot_3d_aircraft, plot_orthographic_views, plot_airfoil_distribution
from ..viz.aero_plots import plot_drag_polar, plot_spanwise_distributions
from ..viz.structures_plots import plot_structures
from ..analysis.aero_3d import analyze_aerobuildup
from ..analysis.structures import analyze_structures
from ..objective.metrics import evaluate_design
from ..objective.objective import score, ObjectiveWeights, NormalizationConstants
from ..objective.performance import estimate_performance
from ..config import CRUISE_SPEED_MS, OUTPUT_DIR
from .numeric import parse_number

WEIGHTS_YAML = "configs/objective_weights.yaml"
NORMALIZATION_YAML = "configs/normalization.yaml"

_DEFAULT = load_default_design_parameters()


def _load_current_weights() -> ObjectiveWeights:
    try:
        return ObjectiveWeights.from_yaml(WEIGHTS_YAML)
    except FileNotFoundError:
        return ObjectiveWeights()


def _load_current_normalization() -> NormalizationConstants:
    try:
        return NormalizationConstants.from_yaml(NORMALIZATION_YAML)
    except FileNotFoundError:
        return NormalizationConstants()

_INPUT_STYLE = {"width": "70px", "marginRight": "4px"}
_LABEL_STYLE = {"width": "150px", "display": "inline-block", "fontWeight": "bold"}


_TD_STYLE = {"padding": "2px 3px", "textAlign": "center"}
_TH_STYLE = {"padding": "2px 6px", "textAlign": "left", "fontWeight": "bold", "fontSize": "12px", "whiteSpace": "nowrap"}
_TABLE_INPUT_STYLE = {"width": "62px", "fontSize": "12px"}


def _cell_input(id_type: str, index: int, value: float, step: float, disabled: bool = False) -> html.Td:
    # type="text", not "number" -- see gui/numeric.py's module docstring for
    # why (native number inputs silently reject one of '.'/',' on a non-US
    # locale, with no way to detect or fix that from Python).
    return html.Td(
        dcc.Input(
            id={"type": id_type, "index": index}, type="text", inputMode="decimal", value=value, disabled=disabled,
            style={**_TABLE_INPUT_STYLE, "backgroundColor": "#eee" if disabled else "white"},
        ),
        style=_TD_STYLE,
    )


def _remove_button_cell(prefix: str, index: int, disabled: bool) -> html.Td:
    if disabled:
        return html.Td("", style=_TD_STYLE)
    return html.Td(
        html.Button(
            "×", id={"type": f"{prefix}-remove-col-button", "index": index}, n_clicks=0,
            title="Remove this station",
            style={"width": "22px", "height": "20px", "fontSize": "12px", "padding": 0, "cursor": "pointer", "lineHeight": "1"},
        ),
        style=_TD_STYLE,
    )


def _station_table(prefix: str, y_control, rows: list[tuple[str, str, list[float], float]]) -> html.Div:
    """rows: list of (label, id_type, values, step) -- one table row per
    quantity, one column per span-control station. Endpoints (y=0 root,
    y=1 tip) are fixed -- SpanwiseDistribution requires them -- so those
    two columns' y-cell and remove-button are disabled; interior columns
    are freely editable/removable. An out-of-order y edit just pauses
    geometry updates rather than crashing (see _strictly_increasing)."""
    n = len(y_control)
    y_row = html.Tr(
        [html.Th("y", style=_TH_STYLE)]
        + [_cell_input(f"{prefix}-y-input", i, y, 0.01, disabled=(i == 0 or i == n - 1)) for i, y in enumerate(y_control)]
    )
    value_rows = [
        html.Tr(
            [html.Th(label, style=_TH_STYLE)]
            + [_cell_input(id_type, i, v, step) for i, v in enumerate(values)]
        )
        for label, id_type, values, step in rows
    ]
    remove_row = html.Tr(
        [html.Th("", style=_TH_STYLE)]
        + [_remove_button_cell(prefix, i, disabled=(i == 0 or i == n - 1)) for i in range(n)]
    )

    table = html.Table(
        html.Tbody([y_row, *value_rows, remove_row]),
        style={"borderCollapse": "collapse"},
    )
    return html.Div(
        [
            html.Div(table, style={"overflowX": "auto", "maxWidth": "100%"}),
            html.Button("+ Add station", id=f"{prefix}-add-button", n_clicks=0, style={"marginTop": "4px", "fontSize": "12px"}),
        ],
        style={"marginBottom": "14px"},
    )


def build_planform_rows(y_control, chord, twist, le, z) -> list:
    return [_station_table("planform", y_control, [
        ("Chord (m)", "chord-input", chord, 0.005),
        ("Twist (deg)", "twist-input", twist, 0.1),
        ("LE offset (m)", "le-input", le, 0.005),
        ("Z offset (m)", "z-input", z, 0.005),
    ])]


def build_airfoil_rows(y_control, thickness, camber, reflex) -> list:
    return [_station_table("airfoil", y_control, [
        ("Thickness scale", "thickness-input", thickness, 0.02),
        ("Camber scale", "camber-input", camber, 0.02),
        ("Reflex scale", "reflex-input", reflex, 0.02),
    ])]


def _strictly_increasing(y) -> bool:
    y = list(y)
    return all(b - a > 1e-6 for a, b in zip(y, y[1:]))


def _interp_at(y_control, values, new_y: float, kind: str) -> float:
    dist = SpanwiseDistribution(np.asarray(y_control, dtype=float), np.asarray(values, dtype=float), kind=kind)
    return float(dist(new_y))


def _add_station(y_control: list[float], value_arrays: list[list[float]], kind: str) -> tuple[list[float], list[list[float]]]:
    """Insert a new station at the midpoint of the largest gap, with each
    value array's new entry interpolated from its own current curve (`kind`
    matches the group: 'pchip' for planform, 'linear' for airfoil) so the
    new station doesn't introduce a discontinuity."""
    gaps = [(y_control[i + 1] - y_control[i], i) for i in range(len(y_control) - 1)]
    _, i = max(gaps)
    new_y = (y_control[i] + y_control[i + 1]) / 2.0
    insert_at = i + 1

    new_y_control = list(y_control)
    new_y_control.insert(insert_at, new_y)
    new_value_arrays = []
    for values in value_arrays:
        new_value = _interp_at(y_control, values, new_y, kind)
        values = list(values)
        values.insert(insert_at, new_value)
        new_value_arrays.append(values)
    return new_y_control, new_value_arrays


def _remove_station_at(y_control: list[float], value_arrays: list[list[float]], index: int) -> tuple[list[float], list[list[float]]] | None:
    """Remove the station at `index` (a specific column's × button),
    keeping the two endpoints. Returns None if index is an endpoint or out
    of range."""
    if index <= 0 or index >= len(y_control) - 1:
        return None

    new_y_control = list(y_control)
    new_y_control.pop(index)
    new_value_arrays = []
    for values in value_arrays:
        values = list(values)
        values.pop(index)
        new_value_arrays.append(values)
    return new_y_control, new_value_arrays


def layout() -> html.Div:
    p = _DEFAULT.planform
    a = _DEFAULT.airfoil_schedule

    controls = html.Div(
        [
            html.H2("Design"),
            html.H4("Planform"),
            html.Div(
                [
                    html.Label("Global:", style={"fontWeight": "bold", "marginRight": "8px"}),
                    html.Label("Span (m)", style={**_LABEL_STYLE, "width": "70px"}),
                    dcc.Input(id="span-input", type="text", inputMode="decimal", value=p.span_m, style=_INPUT_STYLE),
                    html.Label("Sweep (deg)", style={**_LABEL_STYLE, "width": "80px", "marginLeft": "20px"}),
                    dcc.Input(id="sweep-input", type="text", inputMode="decimal", value=p.sweep_deg, style=_INPUT_STYLE),
                ],
                style={"marginBottom": "10px"},
            ),
            html.Div(id="planform-rows-container", children=build_planform_rows(p.y_control, p.chord_m, p.twist_deg, p.le_offset_deviation_m, p.z_offset_m)),
            html.H4("Airfoil Schedule", style={"marginTop": "20px"}),
            html.Div(id="airfoil-rows-container", children=build_airfoil_rows(a.y_control, a.thickness_scale, a.camber_scale, a.reflex_scale)),
            html.Div(id="properties-display", style={"marginTop": "16px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"}),
            html.Button("Run Full Aerodynamic + Structural Evaluation (~20s)", id="evaluate-button", n_clicks=0, style={"marginTop": "16px"}),
            dcc.Loading(html.Div(id="evaluation-display", style={"marginTop": "10px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"})),
            html.Button("Export STL", id="export-stl-button", n_clicks=0, style={"marginTop": "10px"}),
            dcc.Download(id="design-download-stl"),
            html.Hr(style={"marginTop": "16px"}),
            html.Button("Save as Default Design", id="save-default-design-button", n_clicks=0),
            html.P(
                f"Overwrites {DEFAULT_DESIGN_YAML} with whatever's currently in this tab -- becomes "
                "the baseline every script/GUI run uses when none is otherwise specified. Takes effect "
                "the next time an optimizer run is launched or the GUI is restarted (this tab's own "
                "values don't change).",
                style={"fontStyle": "italic", "color": "#666", "fontSize": "12px", "marginTop": "4px"},
            ),
            html.Div(id="save-default-design-status", style={"marginTop": "4px", "fontFamily": "monospace", "fontSize": "12px"}),
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


def build_params_from_inputs(span, sweep, planform_y, chord, twist, le, z, airfoil_y, thickness, camber, reflex) -> DesignParameters:
    """Every arg is a raw dcc.Input(type="text") value (or a list of them)
    -- parse_number()'d here so every caller (this tab's own callbacks and
    run_tab.py, which also calls this directly) gets the '.'/',' robustness
    for free, in one place."""
    span, sweep = parse_number(span), parse_number(sweep)
    planform_y, chord, twist, le, z = (
        [parse_number(v) for v in planform_y], [parse_number(v) for v in chord], [parse_number(v) for v in twist],
        [parse_number(v) for v in le], [parse_number(v) for v in z],
    )
    airfoil_y, thickness, camber, reflex = (
        [parse_number(v) for v in airfoil_y], [parse_number(v) for v in thickness],
        [parse_number(v) for v in camber], [parse_number(v) for v in reflex],
    )
    planform = Planform(
        span_m=span, sweep_deg=sweep, y_control=tuple(planform_y),
        chord_m=tuple(chord), twist_deg=tuple(twist),
        le_offset_deviation_m=tuple(le), z_offset_m=tuple(z),
    )
    airfoil_schedule = AirfoilSchedule(
        y_control=tuple(airfoil_y),
        thickness_scale=tuple(thickness), camber_scale=tuple(camber), reflex_scale=tuple(reflex),
    )
    return DesignParameters(planform=planform, airfoil_schedule=airfoil_schedule)


# Same component IDs, same order as build_params_from_inputs's positional
# args -- used both by this tab's own callbacks (as Input) and by
# run_tab.py (as State, to read "current Design tab values" as a Run
# baseline without needing a Store round-trip, since all tabs share one
# static layout). The y-control arrays resize along with the row
# containers on add/remove, so ALL-pattern matching naturally tracks
# whatever station count is currently displayed.
_GEOMETRY_IDS = [
    ("span-input", False),
    ("sweep-input", False),
    ({"type": "planform-y-input", "index": ALL}, True),
    ({"type": "chord-input", "index": ALL}, True),
    ({"type": "twist-input", "index": ALL}, True),
    ({"type": "le-input", "index": ALL}, True),
    ({"type": "z-input", "index": ALL}, True),
    ({"type": "airfoil-y-input", "index": ALL}, True),
    ({"type": "thickness-input", "index": ALL}, True),
    ({"type": "camber-input", "index": ALL}, True),
    ({"type": "reflex-input", "index": ALL}, True),
]
_GEOMETRY_INPUTS = [Input(id_, "value") for id_, _ in _GEOMETRY_IDS]
DESIGN_STATE_INPUTS = [State(id_, "value") for id_, _ in _GEOMETRY_IDS]


def _real_click_index(triggered_id, triggered_value) -> int | None:
    """`triggered_id` is a per-column remove button's pattern-matching ID
    dict if one was clicked, else None/a different id. Guards against a
    self-referential re-trigger: this callback's own Output regenerates the
    exact remove buttons its Input pattern-matches, so a fresh render (new
    buttons appearing with n_clicks=0) can re-invoke the callback without an
    actual click -- only treat it as a real removal request if the
    triggered button's n_clicks is a genuine, positive click count."""
    if not (isinstance(triggered_id, dict) and triggered_id.get("type", "").endswith("-remove-col-button")):
        return None
    if not triggered_value:
        return None
    return triggered_id["index"]


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("planform-rows-container", "children"),
        Input("planform-add-button", "n_clicks"),
        Input({"type": "planform-remove-col-button", "index": ALL}, "n_clicks"),
        State({"type": "planform-y-input", "index": ALL}, "value"),
        State({"type": "chord-input", "index": ALL}, "value"),
        State({"type": "twist-input", "index": ALL}, "value"),
        State({"type": "le-input", "index": ALL}, "value"),
        State({"type": "z-input", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def edit_planform_stations(n_add, remove_clicks, y_control, chord, twist, le, z):
        y_control, chord, twist, le, z = (
            [parse_number(v) for v in y_control], [parse_number(v) for v in chord], [parse_number(v) for v in twist],
            [parse_number(v) for v in le], [parse_number(v) for v in z],
        )
        if any(v is None for v in [*y_control, *chord, *twist, *le, *z]) or not _strictly_increasing(y_control):
            raise PreventUpdate

        triggered = dash.ctx.triggered_id
        triggered_value = dash.ctx.triggered[0]["value"] if dash.ctx.triggered else None
        remove_index = _real_click_index(triggered, triggered_value)

        if triggered == "planform-add-button":
            new_y, (new_chord, new_twist, new_le, new_z) = _add_station(list(y_control), [chord, twist, le, z], kind="pchip")
        elif remove_index is not None:
            removed = _remove_station_at(list(y_control), [chord, twist, le, z], remove_index)
            if removed is None:
                raise PreventUpdate
            new_y, (new_chord, new_twist, new_le, new_z) = removed
        else:
            raise PreventUpdate

        return build_planform_rows(new_y, new_chord, new_twist, new_le, new_z)

    @app.callback(
        Output("airfoil-rows-container", "children"),
        Input("airfoil-add-button", "n_clicks"),
        Input({"type": "airfoil-remove-col-button", "index": ALL}, "n_clicks"),
        State({"type": "airfoil-y-input", "index": ALL}, "value"),
        State({"type": "thickness-input", "index": ALL}, "value"),
        State({"type": "camber-input", "index": ALL}, "value"),
        State({"type": "reflex-input", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def edit_airfoil_stations(n_add, remove_clicks, y_control, thickness, camber, reflex):
        y_control, thickness, camber, reflex = (
            [parse_number(v) for v in y_control], [parse_number(v) for v in thickness],
            [parse_number(v) for v in camber], [parse_number(v) for v in reflex],
        )
        if any(v is None for v in [*y_control, *thickness, *camber, *reflex]) or not _strictly_increasing(y_control):
            raise PreventUpdate

        triggered = dash.ctx.triggered_id
        triggered_value = dash.ctx.triggered[0]["value"] if dash.ctx.triggered else None
        remove_index = _real_click_index(triggered, triggered_value)

        if triggered == "airfoil-add-button":
            new_y, (new_thickness, new_camber, new_reflex) = _add_station(list(y_control), [thickness, camber, reflex], kind="linear")
        elif remove_index is not None:
            removed = _remove_station_at(list(y_control), [thickness, camber, reflex], remove_index)
            if removed is None:
                raise PreventUpdate
            new_y, (new_thickness, new_camber, new_reflex) = removed
        else:
            raise PreventUpdate

        return build_airfoil_rows(new_y, new_thickness, new_camber, new_reflex)

    @app.callback(
        Output("graph-3d", "figure"),
        Output("graph-ortho", "figure"),
        Output("graph-airfoil", "figure"),
        Output("properties-display", "children"),
        *_GEOMETRY_INPUTS,
    )
    def update_geometry(span, sweep, planform_y, chord, twist, le, z, airfoil_y, thickness, camber, reflex):
        span, sweep = parse_number(span), parse_number(sweep)
        planform_y, chord, twist, le, z = (
            [parse_number(v) for v in planform_y], [parse_number(v) for v in chord], [parse_number(v) for v in twist],
            [parse_number(v) for v in le], [parse_number(v) for v in z],
        )
        airfoil_y, thickness, camber, reflex = (
            [parse_number(v) for v in airfoil_y], [parse_number(v) for v in thickness],
            [parse_number(v) for v in camber], [parse_number(v) for v in reflex],
        )
        all_values = [span, sweep, *planform_y, *chord, *twist, *le, *z, *airfoil_y, *thickness, *camber, *reflex]
        if any(v is None for v in all_values):
            raise PreventUpdate
        if not (_strictly_increasing(planform_y) and _strictly_increasing(airfoil_y)):
            raise PreventUpdate

        params = build_params_from_inputs(span, sweep, planform_y, chord, twist, le, z, airfoil_y, thickness, camber, reflex)
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
    def run_evaluation(n_clicks, span, sweep, planform_y, chord, twist, le, z, airfoil_y, thickness, camber, reflex):
        params = build_params_from_inputs(span, sweep, planform_y, chord, twist, le, z, airfoil_y, thickness, camber, reflex)
        metrics = evaluate_design(params)
        result = score(metrics, _load_current_weights(), _load_current_normalization())

        aircraft = build_aircraft(params)
        cruise_aero = analyze_aerobuildup(aircraft, CRUISE_SPEED_MS, alpha_deg=2.0)
        structures = analyze_structures(aircraft, cruise_trim_cl=cruise_aero.trim_CL, speed_ms=CRUISE_SPEED_MS)
        perf = estimate_performance(aircraft, metrics.total_structural_mass_kg, CRUISE_SPEED_MS)

        text = (
            f"Objective score: {result.score:.2f}   (valid={metrics.valid})\n\n"
            f"Cruise L/D:   {metrics.cruise_L_over_D:.2f}   (trim alpha {metrics.cruise_trim_alpha_deg:.2f} deg)\n"
            f"Fast L/D:     {metrics.fast_L_over_D:.2f}\n"
            f"Root CLmax:   {metrics.root_cl_max:.3f}\n"
            f"Static margin:     {metrics.static_margin:.3f}   (CG assumed at {metrics.cg_x_m * 1000:.0f} mm, NP at {metrics.neutral_point_x_m * 1000:.0f} mm)\n"
            f"Battery x-range for target static margin: "
            f"{f'{metrics.battery_x_min_m * 1000:.0f}-{metrics.battery_x_max_m * 1000:.0f} mm from root LE' if metrics.battery_range_feasible else 'none feasible within the airframe'}\n"
            f"Min safety factor: {metrics.min_safety_factor:.1f}\n"
            f"Structural mass:   {metrics.total_structural_mass_kg:.3f} kg\n"
            f"Payload volume margin: {metrics.payload_volume_margin_m3 * 1e6:.0f} cm^3\n\n"
            f"Best glide ratio: {perf.glide_ratio_max:.1f}   at alpha {perf.glide_alpha_deg:.1f} deg   "
            f"(glide angle {perf.glide_angle_deg:.1f} deg, sink {perf.sink_rate_ms:.2f} m/s)\n"
            f"Cruise power: {perf.cruise_power_w:.1f} W   "
            f"Est. endurance: {perf.estimated_endurance_min:.0f} min   Est. range: {perf.estimated_range_km:.0f} km\n"
        )
        if not metrics.valid:
            text += "\nConstraint violations:\n" + "\n".join(f"  - {v}" for v in metrics.constraint_violations)

        return (
            text,
            plot_drag_polar(aircraft, CRUISE_SPEED_MS),
            plot_spanwise_distributions(aircraft, CRUISE_SPEED_MS, cruise_aero.trim_CL),
            plot_structures(structures),
        )

    @app.callback(
        Output("save-default-design-status", "children"),
        Input("save-default-design-button", "n_clicks"),
        *DESIGN_STATE_INPUTS,
        prevent_initial_call=True,
    )
    def save_default_design(n_clicks, span, sweep, planform_y, chord, twist, le, z, airfoil_y, thickness, camber, reflex):
        params = build_params_from_inputs(span, sweep, planform_y, chord, twist, le, z, airfoil_y, thickness, camber, reflex)
        save_design_parameters(params, DEFAULT_DESIGN_YAML)
        return f"Saved to {DEFAULT_DESIGN_YAML}"

    @app.callback(
        Output("design-download-stl", "data"),
        Input("export-stl-button", "n_clicks"),
        *DESIGN_STATE_INPUTS,
        prevent_initial_call=True,
    )
    def export_stl(n_clicks, span, sweep, planform_y, chord, twist, le, z, airfoil_y, thickness, camber, reflex):
        params = build_params_from_inputs(span, sweep, planform_y, chord, twist, le, z, airfoil_y, thickness, camber, reflex)
        aircraft = build_aircraft(params)
        mesh = build_watertight_mesh(aircraft)
        out_dir = OUTPUT_DIR / "design_tab_export"
        out_dir.mkdir(parents=True, exist_ok=True)
        stl_path = out_dir / "aircraft.stl"
        write_stl(mesh, stl_path)
        return dcc.send_file(str(stl_path))
