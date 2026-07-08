"""Deep Analysis tab: a consolidated post-run report for one finished
design -- objective score breakdown, mass/CG breakdown, structural detail,
performance estimates, and Cp-based flow visualization. Everything here was
already computed somewhere in the pipeline (score().contributions, the mass
and CG models, the structural proxy) but never surfaced together in one
place before.

Like the Results tab, this re-evaluates the selected run's best design from
scratch (build_aircraft + a couple of AeroBuildup calls, ~10-20s) since the
richer intermediate objects (MassEstimate, CGEstimate, StructuralProxyResult)
aren't part of what gets pickled -- only the flat DesignMetrics is.
"""
from __future__ import annotations

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

from ..geometry.aircraft import build_aircraft
from ..geometry.mesh import build_watertight_mesh
from ..geometry.export import write_stl
from ..analysis.aero_3d import analyze_aerobuildup, analyze_hybrid_drag
from ..analysis.structures import analyze_structures, analyze_torsion_and_deflection
from ..objective.mass import estimate_mass
from ..objective.cg import estimate_cg, DEFAULT_STATIC_MARGIN_TARGET
from ..objective.objective import score, ObjectiveWeights, NormalizationConstants
from ..objective.performance import estimate_performance
from ..viz.analysis_plots import plot_objective_contributions, plot_mass_breakdown, plot_cg_layout
from ..viz.structures_plots import plot_structures, plot_torsion_and_deflection
from ..viz.flow_plots import plot_cp_surface_3d
from ..config import CRUISE_SPEED_MS, OUTPUT_DIR
from . import results_io

_GRAPH_STYLE = {"height": "420px"}
WEIGHTS_YAML = "configs/objective_weights.yaml"
NORMALIZATION_YAML = "configs/normalization.yaml"


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


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Deep Analysis"),
            html.P(
                "Pick a run, then Load to re-evaluate its best design and see everything that "
                "went into its score in one place (~30-90s -- runs several fresh AeroBuildup, "
                "VLM, and NeuralFoil calls, not read from the pickle; the VLM cross-check retries "
                "at a coarser resolution if a finer one is ill-conditioned for this geometry, "
                "which can take the longer end of that range).",
                style={"fontStyle": "italic", "color": "#666"},
            ),
            html.Div(
                [
                    dcc.Dropdown(id="analysis-run-dropdown", options=[], style={"width": "420px", "display": "inline-block"}),
                    html.Button("Refresh list", id="analysis-refresh-button", n_clicks=0, style={"marginLeft": "10px"}),
                    html.Button("Load", id="analysis-load-button", n_clicks=0, style={"marginLeft": "10px"}),
                    html.Button("Export STL", id="analysis-export-stl-button", n_clicks=0, style={"marginLeft": "10px"}),
                    dcc.Download(id="analysis-download-stl"),
                ],
                style={"marginBottom": "10px"},
            ),
            dcc.Loading(
                html.Div(
                    [
                        html.Div(id="analysis-summary-display", style={"fontFamily": "monospace", "whiteSpace": "pre-wrap", "marginBottom": "10px"}),
                        dcc.Graph(id="analysis-graph-contributions", style=_GRAPH_STYLE),
                        dcc.Graph(id="analysis-graph-mass", style=_GRAPH_STYLE),
                        dcc.Graph(id="analysis-graph-cg", style=_GRAPH_STYLE),
                        dcc.Graph(id="analysis-graph-structures", style={"height": "650px"}),
                        dcc.Graph(id="analysis-graph-torsion", style={"height": "380px"}),
                        dcc.Graph(id="analysis-graph-cp-surface", style={"height": "600px"}),
                    ]
                )
            ),
        ],
        style={"padding": "10px"},
    )


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("analysis-run-dropdown", "options"),
        Input("analysis-refresh-button", "n_clicks"),
    )
    def refresh(n_clicks):
        return [
            {
                "label": f"{r.output_dir_name}  [{r.run_type}]  score={r.best_score:.2f}  {'valid' if r.valid else 'INVALID'}",
                "value": r.output_dir_name,
            }
            for r in results_io.list_runs()
        ]

    @app.callback(
        Output("analysis-summary-display", "children"),
        Output("analysis-graph-contributions", "figure"),
        Output("analysis-graph-mass", "figure"),
        Output("analysis-graph-cg", "figure"),
        Output("analysis-graph-structures", "figure"),
        Output("analysis-graph-torsion", "figure"),
        Output("analysis-graph-cp-surface", "figure"),
        Input("analysis-load-button", "n_clicks"),
        State("analysis-run-dropdown", "value"),
        prevent_initial_call=True,
    )
    def load_analysis(n_clicks, output_dir_name):
        if not output_dir_name:
            empty = go.Figure()
            return "Pick a run first.", empty, empty, empty, empty, empty, empty

        data, best_params, best_metrics, aircraft = results_io.load_run_aircraft(output_dir_name)

        cruise = analyze_aerobuildup(aircraft, CRUISE_SPEED_MS, alpha_deg=2.0)
        structures = analyze_structures(aircraft, cruise_trim_cl=cruise.trim_CL, speed_ms=CRUISE_SPEED_MS)
        torsion = analyze_torsion_and_deflection(aircraft, structures)
        mass = estimate_mass(aircraft, structures)
        cg = estimate_cg(aircraft, structures, mass, neutral_point_x_m=cruise.neutral_point_x_m, mac_m=aircraft.mean_aerodynamic_chord_m)
        result = score(best_metrics, _load_current_weights(), _load_current_normalization())
        perf = estimate_performance(aircraft, best_metrics.total_structural_mass_kg, CRUISE_SPEED_MS)
        hybrid = analyze_hybrid_drag(aircraft, CRUISE_SPEED_MS, alpha_deg=cruise.trim_alpha_deg)
        hybrid_line = (
            f"vs. hybrid VLM-induced + AeroBuildup-viscous: {hybrid.L_over_D:.2f}"
            if hybrid is not None
            else "vs. hybrid VLM cross-check: unavailable (VLM was ill-conditioned for this geometry at every resolution tried)"
        )

        summary = (
            f"Run: {output_dir_name}\n\n"
            f"Score: {result.score:.2f}   (valid={best_metrics.valid})\n"
            f"Best glide ratio: {perf.glide_ratio_max:.1f}  at alpha {perf.glide_alpha_deg:.1f} deg  "
            f"(glide angle {perf.glide_angle_deg:.1f} deg, sink {perf.sink_rate_ms:.2f} m/s)\n"
            f"Cruise power: {perf.cruise_power_w:.1f} W   Est. endurance: {perf.estimated_endurance_min:.0f} min   Est. range: {perf.estimated_range_km:.0f} km\n\n"
            f"Cruise L/D cross-check -- AeroBuildup (strip theory): {cruise.trim_L_over_D:.2f}   {hybrid_line}\n"
        )
        if not best_metrics.valid:
            summary += "\nConstraint violations:\n" + "\n".join(f"  - {v}" for v in best_metrics.constraint_violations)

        fig_contrib = plot_objective_contributions(result)
        fig_mass = plot_mass_breakdown(mass, cg.battery_mass_kg)
        fig_cg = plot_cg_layout(aircraft, cg, cruise.neutral_point_x_m, aircraft.mean_aerodynamic_chord_m, DEFAULT_STATIC_MARGIN_TARGET)
        fig_structures = plot_structures(structures)
        fig_torsion = plot_torsion_and_deflection(torsion)
        fig_cp_surface = plot_cp_surface_3d(aircraft, alpha_deg=cruise.trim_alpha_deg, speed_ms=CRUISE_SPEED_MS)

        return summary, fig_contrib, fig_mass, fig_cg, fig_structures, fig_torsion, fig_cp_surface

    @app.callback(
        Output("analysis-download-stl", "data"),
        Input("analysis-export-stl-button", "n_clicks"),
        State("analysis-run-dropdown", "value"),
        prevent_initial_call=True,
    )
    def export_stl(n_clicks, output_dir_name):
        if not output_dir_name:
            raise dash.exceptions.PreventUpdate
        _, best_params, _, aircraft = results_io.load_run_aircraft(output_dir_name)
        mesh = build_watertight_mesh(aircraft)
        stl_path = OUTPUT_DIR / output_dir_name / "aircraft.stl"
        write_stl(mesh, stl_path)
        return dcc.send_file(str(stl_path))
