"""Run Optimizer tab: launch Stage 1 / Stage 2 / multi-cycle runs as
background subprocesses (see `run_manager.py`) and poll their progress.
"""
from __future__ import annotations

import time
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, ALL

from ..geometry.params_io import save_design_parameters, load_default_design_parameters
from . import run_manager, results_io
from .design_tab import build_params_from_inputs, DESIGN_STATE_INPUTS

_INPUT_STYLE = {"width": "90px", "marginRight": "10px"}
_LABEL_STYLE = {"width": "220px", "display": "inline-block"}
_PROGRESS_BAR_FILL_BASE_STYLE = {
    "height": "100%", "width": "0%", "backgroundColor": "#2ca02c",
    "transition": "width 0.5s ease", "borderRadius": "4px",
}


def _progress_display() -> tuple[dict, str]:
    """(bar-fill style, readout text) from the running script's most recent
    PROGRESS line (see run_manager.progress()). Empty/zero-width if no
    progress has been reported yet (e.g. still evaluating the first batch)."""
    info = run_manager.progress()
    if not info:
        return dict(_PROGRESS_BAR_FILL_BASE_STYLE), ""

    done = info.get("evals_done", 0)
    total = max(1, info.get("evals_total", 1))
    pct = max(0.0, min(100.0, 100.0 * done / total))

    label = f"Stage {info.get('stage')}/{info.get('n_stages')}"
    if "cycle" in info:
        label = f"Cycle {info['cycle'] + 1}/{info.get('n_cycles', '?')} ({info.get('stage_name', '?')}) -- {label}"

    text = f"{label} -- {done}/{total} evaluations (est.) -- best score so far: {info.get('best_score', float('nan')):.2f}"
    return {**_PROGRESS_BAR_FILL_BASE_STYLE, "width": f"{pct:.0f}%"}, text


def _row(label, id_, value, step=None) -> html.Div:
    kwargs = {"step": step} if step is not None else {}
    return html.Div(
        [html.Label(label, style=_LABEL_STYLE), dcc.Input(id=id_, type="number", value=value, style=_INPUT_STYLE, **kwargs)],
        style={"marginBottom": "4px"},
    )


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Run Optimizer"),
            html.Div(
                [
                    html.Label("Run type", style=_LABEL_STYLE),
                    dcc.Dropdown(
                        id="run-type-dropdown",
                        options=[
                            {"label": "Stage 1 (airfoil schedule)", "value": "stage1"},
                            {"label": "Stage 2 (planform)", "value": "stage2"},
                            {"label": "Multi-cycle (Stage1 <-> Stage2)", "value": "multi_cycle"},
                        ],
                        value="stage1", clearable=False, style={"width": "320px", "display": "inline-block"},
                    ),
                ],
                style={"marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.Label("Baseline design", style=_LABEL_STYLE),
                    dcc.RadioItems(
                        id="baseline-source-radio",
                        options=[
                            {"label": "Default design", "value": "default"},
                            {"label": "Current Design-tab values", "value": "design_tab"},
                            {"label": "An existing result", "value": "existing_result"},
                        ],
                        value="default", inline=False,
                    ),
                ],
                style={"marginBottom": "6px"},
            ),
            html.Div(
                [
                    html.Label("Existing result", style=_LABEL_STYLE),
                    dcc.Dropdown(id="baseline-result-dropdown", options=[], style={"width": "320px", "display": "inline-block"}),
                    html.Button("Refresh list", id="refresh-baseline-results-button", n_clicks=0, style={"marginLeft": "10px"}),
                ],
                style={"marginBottom": "16px"},
            ),

            html.Div(
                id="single-stage-settings",
                children=[
                    html.H4("Optimizer settings"),
                    _row("Number of stages", "opt-n-stages", 4),
                    _row("Samples per stage", "opt-n-samples-per-stage", 40),
                    _row("Retain best N", "opt-retain-best-n", 5),
                    _row("Seed", "opt-seed", 0),
                    _row("Parallel workers (n_jobs)", "opt-n-jobs", 4),
                ],
            ),
            html.Div(
                id="multi-cycle-settings",
                children=[
                    html.H4("Multi-cycle settings"),
                    _row("Number of cycles", "mc-n-cycles", 2),
                    html.Label("Start with", style=_LABEL_STYLE),
                    dcc.Dropdown(
                        id="mc-start-with", options=[{"label": "Stage 1", "value": "stage1"}, {"label": "Stage 2", "value": "stage2"}],
                        value="stage1", clearable=False, style={"width": "150px", "display": "inline-block", "marginBottom": "8px"},
                    ),
                    _row("Stage 1: stages", "mc-stage1-n-stages", 3),
                    _row("Stage 1: samples/stage", "mc-stage1-n-samples-per-stage", 24),
                    _row("Stage 1: retain best N", "mc-stage1-retain-best-n", 5),
                    _row("Stage 2: stages", "mc-stage2-n-stages", 3),
                    _row("Stage 2: samples/stage", "mc-stage2-n-samples-per-stage", 40),
                    _row("Stage 2: retain best N", "mc-stage2-retain-best-n", 6),
                    _row("Seed", "mc-seed", 0),
                    _row("Parallel workers (n_jobs)", "mc-n-jobs", 4),
                ],
                style={"display": "none"},
            ),

            html.Button("Start Run", id="start-run-button", n_clicks=0, style={"marginTop": "14px"}),
            html.Div(id="run-status-display", style={"marginTop": "10px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"}),
            html.Div(
                html.Div(id="run-progress-bar-fill", style=dict(_PROGRESS_BAR_FILL_BASE_STYLE)),
                id="run-progress-bar-track",
                style={
                    "marginTop": "8px", "height": "18px", "width": "100%",
                    "backgroundColor": "#333", "borderRadius": "4px", "overflow": "hidden",
                },
            ),
            html.Div(id="run-progress-text", style={"marginTop": "4px", "fontFamily": "monospace", "fontSize": "13px"}),
            html.Pre(id="run-log-display", style={
                "marginTop": "10px", "height": "300px", "overflowY": "scroll",
                "backgroundColor": "#111", "color": "#0f0", "padding": "8px", "fontSize": "12px",
            }),
            dcc.Interval(id="run-poll-interval", interval=1000, disabled=True),
        ],
        style={"padding": "10px"},
    )


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("single-stage-settings", "style"),
        Output("multi-cycle-settings", "style"),
        Input("run-type-dropdown", "value"),
    )
    def toggle_settings(run_type):
        if run_type == "multi_cycle":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    @app.callback(
        Output("baseline-result-dropdown", "options"),
        Input("refresh-baseline-results-button", "n_clicks"),
    )
    def refresh_baseline_results(n_clicks):
        return [
            {"label": f"{r.output_dir_name}  (score {r.best_score:.2f}, {'valid' if r.valid else 'INVALID'})", "value": r.output_dir_name}
            for r in results_io.list_runs()
        ]

    @app.callback(
        Output("run-status-display", "children"),
        Output("run-poll-interval", "disabled"),
        Output("run-progress-bar-fill", "style"),
        Output("run-progress-text", "children"),
        Output("run-log-display", "children"),
        Input("start-run-button", "n_clicks"),
        Input("run-poll-interval", "n_intervals"),
        State("run-type-dropdown", "value"),
        State("baseline-source-radio", "value"),
        State("baseline-result-dropdown", "value"),
        State("opt-n-stages", "value"), State("opt-n-samples-per-stage", "value"),
        State("opt-retain-best-n", "value"), State("opt-seed", "value"), State("opt-n-jobs", "value"),
        State("mc-n-cycles", "value"), State("mc-start-with", "value"),
        State("mc-stage1-n-stages", "value"), State("mc-stage1-n-samples-per-stage", "value"), State("mc-stage1-retain-best-n", "value"),
        State("mc-stage2-n-stages", "value"), State("mc-stage2-n-samples-per-stage", "value"), State("mc-stage2-retain-best-n", "value"),
        State("mc-seed", "value"), State("mc-n-jobs", "value"),
        *DESIGN_STATE_INPUTS,
        prevent_initial_call=True,
    )
    def start_or_poll(
        n_clicks, n_intervals, run_type, baseline_source, existing_result_name,
        n_stages, n_samples_per_stage, retain_best_n, seed, n_jobs,
        mc_n_cycles, mc_start_with, mc_s1_stages, mc_s1_samples, mc_s1_retain,
        mc_s2_stages, mc_s2_samples, mc_s2_retain, mc_seed, mc_n_jobs,
        *design_state_values,
    ):
        triggered = dash.ctx.triggered_id

        if triggered == "run-poll-interval":
            st = run_manager.status()
            log = run_manager.read_log_tail()
            run = run_manager.get_current_run()
            name = run.output_dir_name if run else "?"
            bar_style, progress_text = _progress_display()
            if st == "running":
                return f"Running ({name})...", False, bar_style, progress_text, log
            if st == "completed":
                return f"Completed ({name}). Open the Results tab and select '{name}' to inspect it.", True, bar_style, progress_text, log
            if st == "failed":
                return f"FAILED ({name}) -- see log below.", True, bar_style, progress_text, log
            return "Idle.", True, dict(_PROGRESS_BAR_FILL_BASE_STYLE), "", ""

        # triggered by the Start Run button
        if run_manager.is_running():
            return "A run is already in progress -- wait for it to finish first.", False, dash.no_update, dash.no_update, dash.no_update

        output_dir_name = f"{run_type}_run_{datetime.now():%Y%m%d_%H%M%S}"

        if baseline_source == "default":
            baseline = load_default_design_parameters()
        elif baseline_source == "design_tab":
            baseline = build_params_from_inputs(*design_state_values)
        else:
            if not existing_result_name:
                return "Pick an existing result to use as the baseline first.", True, dash.no_update, dash.no_update, dash.no_update
            data = results_io.load_run(existing_result_name)
            baseline = results_io.run_best_params(data)

        from ..config import OUTPUT_DIR
        out_dir = OUTPUT_DIR / output_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        baseline_yaml = out_dir / "baseline.yaml"
        save_design_parameters(baseline, baseline_yaml)

        common_args = [
            "--baseline-yaml", str(baseline_yaml),
            "--weights-yaml", "configs/objective_weights.yaml",
            "--normalization-yaml", "configs/normalization.yaml",
        ]

        if run_type == "multi_cycle":
            args = common_args + [
                "--n-cycles", str(mc_n_cycles), "--start-with", mc_start_with,
                "--stage1-n-stages", str(mc_s1_stages), "--stage1-n-samples-per-stage", str(mc_s1_samples),
                "--stage1-retain-best-n", str(mc_s1_retain),
                "--stage2-n-stages", str(mc_s2_stages), "--stage2-n-samples-per-stage", str(mc_s2_samples),
                "--stage2-retain-best-n", str(mc_s2_retain),
                "--seed", str(mc_seed), "--n-jobs", str(mc_n_jobs),
            ]
        else:
            args = common_args + [
                "--n-stages", str(n_stages), "--n-samples-per-stage", str(n_samples_per_stage),
                "--retain-best-n", str(retain_best_n), "--seed", str(seed), "--n-jobs", str(n_jobs),
            ]

        run_manager.launch_run(run_type, args, output_dir_name)
        return f"Started {run_type} run '{output_dir_name}'...", False, dict(_PROGRESS_BAR_FILL_BASE_STYLE), "", ""
