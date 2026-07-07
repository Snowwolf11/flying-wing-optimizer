"""Results tab: browse and inspect past optimization runs from `outputs/`.

Loads an existing `result.pkl` (no recomputation of the optimization
itself) and re-renders plots via the same `viz/*` functions used
everywhere else. Re-evaluating the *baseline* for the comparison table is
the one thing that costs a few seconds (it wasn't saved by the run script),
so that's done on demand when a result is selected, same as the Design
tab's "Run Evaluation" button.
"""
from __future__ import annotations

from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

from ..geometry.aircraft import build_aircraft
from ..objective.metrics import evaluate_design
from ..viz.geometry_plots import plot_3d_aircraft, plot_orthographic_views, plot_airfoil_distribution
from ..viz.optimization_plots import plot_convergence, plot_parameter_evolution, plot_multi_cycle_convergence
from . import results_io

_METRIC_FIELDS = [
    "span_m", "aspect_ratio", "wing_area_m2",
    "cruise_L_over_D", "fast_L_over_D", "root_cl_max",
    "min_safety_factor", "total_structural_mass_kg", "payload_volume_margin_m3",
]


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Results"),
            html.Div(
                [
                    dcc.Dropdown(id="results-run-dropdown", options=[], style={"width": "420px", "display": "inline-block"}),
                    html.Button("Refresh list", id="refresh-results-button", n_clicks=0, style={"marginLeft": "10px"}),
                    html.Button("Load", id="load-result-button", n_clicks=0, style={"marginLeft": "10px"}),
                    html.Button("Send to Design tab", id="send-to-design-button", n_clicks=0, style={"marginLeft": "10px"}),
                ],
                style={"marginBottom": "10px"},
            ),
            dcc.Loading(html.Div(id="results-summary-display", style={"fontFamily": "monospace", "whiteSpace": "pre-wrap", "marginBottom": "10px"})),
            dcc.Loading(
                html.Div(
                    [
                        dcc.Graph(id="results-graph-3d", style={"height": "500px"}),
                        dcc.Graph(id="results-graph-ortho", style={"height": "400px"}),
                        dcc.Graph(id="results-graph-airfoil", style={"height": "600px"}),
                        dcc.Graph(id="results-graph-convergence", style={"height": "420px"}),
                    ]
                )
            ),
            dcc.Store(id="results-loaded-store"),
        ],
        style={"padding": "10px"},
    )


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("results-run-dropdown", "options"),
        Input("refresh-results-button", "n_clicks"),
    )
    def refresh(n_clicks):
        return [
            {
                "label": f"{r.output_dir_name}  [{r.run_type}]  score={r.best_score:.2f}  "
                         f"{'valid' if r.valid else 'INVALID'}  ({datetime.fromtimestamp(r.mtime):%Y-%m-%d %H:%M})",
                "value": r.output_dir_name,
            }
            for r in results_io.list_runs()
        ]

    @app.callback(
        Output("results-summary-display", "children"),
        Output("results-graph-3d", "figure"),
        Output("results-graph-ortho", "figure"),
        Output("results-graph-airfoil", "figure"),
        Output("results-graph-convergence", "figure"),
        Output("results-loaded-store", "data"),
        Input("load-result-button", "n_clicks"),
        State("results-run-dropdown", "value"),
        prevent_initial_call=True,
    )
    def load_result(n_clicks, output_dir_name):
        if not output_dir_name:
            empty = go.Figure()
            return "Pick a run first.", empty, empty, empty, empty, None

        data, best_params, best_metrics, aircraft = results_io.load_run_aircraft(output_dir_name)
        run_type = data.get("run_type", "stage_unknown")

        baseline = data.get("baseline")
        baseline_metrics = evaluate_design(baseline) if baseline is not None else None

        lines = [f"Run: {output_dir_name}  [{run_type}]", ""]
        header = f"{'metric':<28}{'optimized':>14}"
        if baseline_metrics is not None:
            header = f"{'metric':<28}{'baseline':>14}{'optimized':>14}"
        lines.append(header)
        for field in _METRIC_FIELDS:
            o = getattr(best_metrics, field)
            if baseline_metrics is not None:
                b = getattr(baseline_metrics, field)
                lines.append(f"{field:<28}{b:>14.4f}{o:>14.4f}")
            else:
                lines.append(f"{field:<28}{o:>14.4f}")
        summary = "\n".join(lines)

        fig_3d = plot_3d_aircraft(aircraft)
        fig_ortho = plot_orthographic_views(aircraft)
        fig_airfoil = plot_airfoil_distribution(aircraft)

        if run_type == "multi_cycle":
            fig_conv = plot_multi_cycle_convergence(data["multi_cycle_result"])
        else:
            fig_conv = plot_convergence(data["result"])

        return summary, fig_3d, fig_ortho, fig_airfoil, fig_conv, output_dir_name

    @app.callback(
        Output("span-input", "value"),
        Output("sweep-input", "value"),
        Output({"type": "chord-input", "index": dash.ALL}, "value"),
        Output({"type": "twist-input", "index": dash.ALL}, "value"),
        Output({"type": "le-input", "index": dash.ALL}, "value"),
        Output({"type": "z-input", "index": dash.ALL}, "value"),
        Output({"type": "thickness-input", "index": dash.ALL}, "value"),
        Output({"type": "camber-input", "index": dash.ALL}, "value"),
        Output({"type": "reflex-input", "index": dash.ALL}, "value"),
        Input("send-to-design-button", "n_clicks"),
        State("results-loaded-store", "data"),
        prevent_initial_call=True,
    )
    def send_to_design(n_clicks, output_dir_name):
        if not output_dir_name:
            raise dash.exceptions.PreventUpdate
        data = results_io.load_run(output_dir_name)
        params = results_io.run_best_params(data)
        p, a = params.planform, params.airfoil_schedule
        return (
            p.span_m, p.sweep_deg,
            list(p.chord_m), list(p.twist_deg), list(p.le_offset_deviation_m), list(p.z_offset_m),
            list(a.thickness_scale), list(a.camber_scale), list(a.reflex_scale),
        )
