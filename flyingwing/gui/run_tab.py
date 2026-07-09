"""Run Optimizer tab: launch Stage 1 / Stage 2 / multi-cycle runs as
background subprocesses (see `run_manager.py`) and poll their progress.
"""
from __future__ import annotations

import time
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, ALL

import numpy as np

from ..geometry.params_io import save_design_parameters, load_default_design_parameters
from . import run_manager, results_io
from .design_tab import build_params_from_inputs, DESIGN_STATE_INPUTS, _load_current_weights, _load_current_normalization
from .numeric import parse_number
from ..optimization.base import evaluate_batch
from ..optimization.stage1 import make_stage1_parameter_set, Stage1Objective
from ..optimization.stage2 import make_stage2_parameter_set, Stage2Objective
from ..optimization import auto_tune

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
    if "restart" in info:
        label = f"Restart {info['restart']}/{info.get('n_restarts', '?')} -- Gen {label}"
    if "cycle" in info:
        label = f"Cycle {info['cycle'] + 1}/{info.get('n_cycles', '?')} ({info.get('stage_name', '?')}) -- {label}"

    text = f"{label} -- {done}/{total} evaluations (est.) -- best score so far: {info.get('best_score', float('nan')):.2f}"
    return {**_PROGRESS_BAR_FILL_BASE_STYLE, "width": f"{pct:.0f}%"}, text


def _row(label, id_, value, step=None) -> html.Div:
    # type="text", not "number" -- see gui/numeric.py's module docstring for
    # why (native number inputs silently reject one of '.'/',' on a non-US
    # locale, with no way to detect or fix that from Python).
    return html.Div(
        [html.Label(label, style=_LABEL_STYLE), dcc.Input(id=id_, type="text", inputMode="decimal", value=value, style=_INPUT_STYLE)],
        style={"marginBottom": "4px"},
    )


def _method_dropdown(id_: str) -> html.Div:
    return html.Div(
        [
            html.Label("Optimization method", style=_LABEL_STYLE),
            dcc.Dropdown(
                id=id_,
                options=[
                    {"label": "CMA-ES (default)", "value": "cma"},
                    {"label": "Latin Hypercube (legacy)", "value": "lhs"},
                ],
                value="cma", clearable=False, style={"width": "260px", "display": "inline-block"},
            ),
        ],
        style={"marginBottom": "8px"},
    )


def _cma_settings(prefix: str) -> html.Div:
    return html.Div(
        [
            _row("Initial step size (sigma0)", f"{prefix}-cma-sigma0", 0.25, step=0.05),
            _row("Population size (blank = auto)", f"{prefix}-cma-population-size", None),
            _row("Max generations", f"{prefix}-cma-max-generations", 100),
            _row("Restarts (IPOP, multi-modality)", f"{prefix}-cma-n-restarts", 2),
        ],
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
                [
                    html.Label("Setup mode", style=_LABEL_STYLE),
                    dcc.RadioItems(
                        id="setup-mode-radio",
                        options=[
                            {"label": "Simple (just pick a target duration)", "value": "simple"},
                            {"label": "Advanced (set every optimizer parameter myself)", "value": "advanced"},
                        ],
                        value="simple", inline=False,
                    ),
                ],
                style={"marginBottom": "6px"},
            ),
            html.Div(
                id="simple-mode-settings",
                children=[
                    html.H4("Quick setup"),
                    _row("Target run duration (minutes)", "simple-target-minutes", 15),
                    html.Div(
                        "Uses CMA-ES (the recommended method) with every other setting -- population "
                        "size, generations, restarts, parallel workers -- picked automatically to "
                        "roughly fill this much time, based on a quick real timing measurement taken "
                        "from the baseline design just before the run starts. Click \"Estimate Time\" "
                        "first to preview the settings it would pick without starting anything.",
                        style={"maxWidth": "480px", "color": "#888", "fontSize": "13px", "marginTop": "4px"},
                    ),
                ],
                style={"marginBottom": "16px"},
            ),
            html.Div(
                id="single-stage-settings",
                children=[
                    html.H4("Optimizer settings"),
                    _method_dropdown("opt-method-dropdown"),
                    html.Div(id="opt-cma-settings", children=_cma_settings("opt")),
                    html.Div(
                        id="opt-lhs-settings",
                        children=[
                            _row("Number of stages", "opt-n-stages", 4),
                            _row("Samples per stage", "opt-n-samples-per-stage", 40),
                            _row("Retain best N", "opt-retain-best-n", 5),
                        ],
                        style={"display": "none"},
                    ),
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
                    _method_dropdown("mc-method-dropdown"),
                    html.Div(id="mc-cma-settings", children=_cma_settings("mc")),
                    html.Div(
                        id="mc-lhs-settings",
                        children=[
                            _row("Stage 1: stages", "mc-stage1-n-stages", 3),
                            _row("Stage 1: samples/stage", "mc-stage1-n-samples-per-stage", 24),
                            _row("Stage 1: retain best N", "mc-stage1-retain-best-n", 5),
                            _row("Stage 2: stages", "mc-stage2-n-stages", 3),
                            _row("Stage 2: samples/stage", "mc-stage2-n-samples-per-stage", 40),
                            _row("Stage 2: retain best N", "mc-stage2-retain-best-n", 6),
                        ],
                        style={"display": "none"},
                    ),
                    _row("Seed", "mc-seed", 0),
                    _row("Parallel workers (n_jobs)", "mc-n-jobs", 4),
                ],
                style={"display": "none"},
            ),

            html.Div(
                [
                    html.Button("Start Run", id="start-run-button", n_clicks=0),
                    html.Button("Estimate Time", id="estimate-time-button", n_clicks=0, style={"marginLeft": "10px"}),
                ],
                style={"marginTop": "14px"},
            ),
            dcc.Loading(html.Div(id="estimate-time-display", style={"marginTop": "6px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"})),
            # Populated once, right when Simple mode's Start Run auto-picks
            # settings, and left alone for the rest of the run -- unlike
            # run-status-display below (driven by the same callback, but
            # also overwritten every ~1s by the poll interval's "Running..."
            # text, which was wiping this out after a single tick).
            html.Div(id="simple-mode-picked-settings", style={"marginTop": "6px", "fontFamily": "monospace", "whiteSpace": "pre-wrap", "color": "#8ab4f8"}),
            dcc.Loading(html.Div(id="run-status-display", style={"marginTop": "10px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"})),
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


def _resolve_baseline(baseline_source, existing_result_name, design_state_values):
    """(baseline, error_message) -- error_message is None on success. Same
    baseline-resolution logic start_or_poll uses, shared so "Estimate Time"
    measures against the same design a real run would start from."""
    if baseline_source == "default":
        return load_default_design_parameters(), None
    if baseline_source == "design_tab":
        return build_params_from_inputs(*design_state_values), None
    if not existing_result_name:
        return None, "Pick an existing result to use as the baseline first."
    data = results_io.load_run(existing_result_name)
    return results_io.run_best_params(data), None


def _lhs_planned_evals(n_stages, n_samples_per_stage) -> int:
    return int(parse_number(n_stages)) * int(parse_number(n_samples_per_stage))


def _cma_planned_evals(popsize, max_gen, n_restarts, n_dims) -> int:
    """Upper bound -- CMA-ES's own convergence criteria (es.stop()) usually
    end a run well before max_generations is reached."""
    max_gen = int(parse_number(max_gen))
    n_restarts = max(1, int(parse_number(n_restarts)))
    popsize_val = parse_number(popsize)
    base = int(popsize_val) if popsize_val is not None else int(4 + 3 * np.log(n_dims))
    return sum(base * (2 ** r) * max_gen for r in range(n_restarts))


def _time_one_batch(parameter_set, weights, normalization, objective_cls, n_jobs) -> tuple[float, int]:
    """(wall_clock_seconds, batch_size) for evaluating one real batch of
    n_jobs copies of the baseline's own default vector, through the actual
    parallel evaluate_batch() path -- this measures real ProcessPoolExecutor
    overhead, not just a guessed per-eval cost."""
    n_jobs = max(1, int(parse_number(n_jobs)))
    objective_fn = objective_cls(parameter_set, weights, normalization)
    batch = np.tile(parameter_set.default_vector, (n_jobs, 1))
    t0 = time.time()
    evaluated = evaluate_batch(objective_fn, batch, n_jobs)
    dt = time.time() - t0
    # If the baseline itself gets caught by the cheap pre-check (see
    # geometry/constraints.py::quick_reject_reason), this timing is
    # near-instant and NOT representative of a real evaluation's cost --
    # flagged so the caller can warn rather than silently under-estimate.
    quick_rejected = all(
        c.extra["metrics"].constraint_violations and "quick pre-check" in c.extra["metrics"].constraint_violations[0]
        for c in evaluated
    )
    return dt, n_jobs, quick_rejected


def _simple_mode_preview(run_type, ps1, ps2, weights, normalization, target_minutes) -> str:
    """What Simple mode would auto-pick and roughly how long it would take,
    without starting a run -- same real-batch-timing approach as the normal
    (Advanced-mode) estimate, just inverted: given a target duration instead
    of given settings. See optimization/auto_tune.py for the allocation math."""
    target_seconds = max(1.0, parse_number(target_minutes) * 60.0)
    n_jobs = auto_tune.default_n_jobs()

    if run_type == "multi_cycle":
        batch_time, batch_n, quick_rejected = _time_one_batch(ps1, weights, normalization, Stage1Objective, n_jobs)
        per_eval = batch_time / batch_n
        total_evals_budget = target_seconds / per_eval
        # n_cycles has a *weak* dependency on the budget (auto_n_cycles), not
        # a fixed value -- see auto_tune.py. The target is then split evenly
        # across the resulting 2*n_cycles stage-runs.
        n_cycles = auto_tune.auto_n_cycles(total_evals_budget, len(ps2.bounds))
        per_substage_evals = total_evals_budget / (n_cycles * 2)
        s1 = auto_tune.auto_cma_settings(len(ps1.bounds), per_substage_evals)
        s2 = auto_tune.auto_cma_settings(len(ps2.bounds), per_substage_evals)
        total_evals = n_cycles * (s1.planned_evals + s2.planned_evals)
        lines = [
            f"Auto-picked for multi-cycle (CMA-ES, {n_cycles} cycles, n_jobs={n_jobs}):",
            f"  Stage 1: sigma0={s1.sigma0:.3f}, population~{s1.population_size} (auto), max_generations={s1.max_generations}, restarts={s1.n_restarts}",
            f"  Stage 2: sigma0={s2.sigma0:.3f}, population~{s2.population_size} (auto), max_generations={s2.max_generations}, restarts={s2.n_restarts}",
            "  (Stage 2's max_generations/restarts also used for Stage 1, so it doesn't overshoot the budget; sigma0 is identical for both since it only depends on the shared per-substage budget, not dimensionality)",
        ]
    else:
        ps = ps1 if run_type == "stage1" else ps2
        objective_cls = Stage1Objective if run_type == "stage1" else Stage2Objective
        batch_time, batch_n, quick_rejected = _time_one_batch(ps, weights, normalization, objective_cls, n_jobs)
        per_eval = batch_time / batch_n
        s = auto_tune.auto_cma_settings(len(ps.bounds), target_seconds / per_eval)
        total_evals = s.planned_evals
        lines = [
            f"Auto-picked for {run_type} (CMA-ES, n_jobs={n_jobs}):",
            f"  sigma0={s.sigma0:.3f}, population~{s.population_size} (auto), max_generations={s.max_generations}, restarts={s.n_restarts}",
        ]

    total_seconds = total_evals * per_eval
    lines.append(f"Timed {batch_n} real evaluation(s) in parallel just now: {batch_time:.1f}s ({per_eval:.2f}s/eval effective)")
    lines.append(f"Planned ~{total_evals} evaluations -> estimated ~{total_seconds / 60:.1f} min (target was {target_seconds / 60:.1f} min)")
    if quick_rejected:
        lines.append("NOTE: the baseline design itself got caught by the cheap pre-check, so this timing is near-instant and NOT representative -- treat the estimate as a large UNDER-estimate.")
    lines.append("These exact settings will be used automatically when you click Start Run in Simple mode (re-timed fresh at that point).")
    return "\n".join(lines)


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("single-stage-settings", "style"),
        Output("multi-cycle-settings", "style"),
        Output("simple-mode-settings", "style"),
        Input("run-type-dropdown", "value"),
        Input("setup-mode-radio", "value"),
    )
    def toggle_settings(run_type, setup_mode):
        if setup_mode == "simple":
            return {"display": "none"}, {"display": "none"}, {"display": "block"}
        if run_type == "multi_cycle":
            return {"display": "none"}, {"display": "block"}, {"display": "none"}
        return {"display": "block"}, {"display": "none"}, {"display": "none"}

    @app.callback(
        Output("opt-cma-settings", "style"),
        Output("opt-lhs-settings", "style"),
        Output("mc-cma-settings", "style"),
        Output("mc-lhs-settings", "style"),
        Input("opt-method-dropdown", "value"),
        Input("mc-method-dropdown", "value"),
    )
    def toggle_method_settings(opt_method, mc_method):
        def styles(method):
            return ({"display": "block"}, {"display": "none"}) if method == "cma" else ({"display": "none"}, {"display": "block"})
        opt_cma_style, opt_lhs_style = styles(opt_method)
        mc_cma_style, mc_lhs_style = styles(mc_method)
        return opt_cma_style, opt_lhs_style, mc_cma_style, mc_lhs_style

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
        Output("estimate-time-display", "children"),
        Input("estimate-time-button", "n_clicks"),
        State("run-type-dropdown", "value"),
        State("baseline-source-radio", "value"),
        State("baseline-result-dropdown", "value"),
        State("setup-mode-radio", "value"), State("simple-target-minutes", "value"),
        State("opt-method-dropdown", "value"),
        State("opt-cma-sigma0", "value"), State("opt-cma-population-size", "value"),
        State("opt-cma-max-generations", "value"), State("opt-cma-n-restarts", "value"),
        State("opt-n-stages", "value"), State("opt-n-samples-per-stage", "value"), State("opt-n-jobs", "value"),
        State("mc-n-cycles", "value"),
        State("mc-method-dropdown", "value"),
        State("mc-cma-sigma0", "value"), State("mc-cma-population-size", "value"),
        State("mc-cma-max-generations", "value"), State("mc-cma-n-restarts", "value"),
        State("mc-stage1-n-stages", "value"), State("mc-stage1-n-samples-per-stage", "value"),
        State("mc-stage2-n-stages", "value"), State("mc-stage2-n-samples-per-stage", "value"),
        State("mc-n-jobs", "value"),
        *DESIGN_STATE_INPUTS,
        prevent_initial_call=True,
    )
    def estimate_time(
        n_clicks, run_type, baseline_source, existing_result_name,
        setup_mode, simple_target_minutes,
        opt_method, opt_cma_sigma0, opt_cma_popsize, opt_cma_max_gen, opt_cma_restarts,
        n_stages, n_samples_per_stage, n_jobs,
        mc_n_cycles,
        mc_method, mc_cma_sigma0, mc_cma_popsize, mc_cma_max_gen, mc_cma_restarts,
        mc_s1_stages, mc_s1_samples, mc_s2_stages, mc_s2_samples, mc_n_jobs,
        *design_state_values,
    ):
        baseline, err = _resolve_baseline(baseline_source, existing_result_name, design_state_values)
        if err:
            return err

        weights, normalization = _load_current_weights(), _load_current_normalization()
        ps1 = make_stage1_parameter_set(baseline)
        ps2 = make_stage2_parameter_set(baseline)

        if setup_mode == "simple":
            return _simple_mode_preview(run_type, ps1, ps2, weights, normalization, simple_target_minutes)

        lines = []
        if run_type == "multi_cycle":
            batch_time, batch_n, quick_rejected = _time_one_batch(ps1, weights, normalization, Stage1Objective, mc_n_jobs)
            per_eval = batch_time / batch_n

            if mc_method == "lhs":
                s1_evals = _lhs_planned_evals(mc_s1_stages, mc_s1_samples)
                s2_evals = _lhs_planned_evals(mc_s2_stages, mc_s2_samples)
            else:
                s1_evals = _cma_planned_evals(mc_cma_popsize, mc_cma_max_gen, mc_cma_restarts, len(ps1.bounds))
                s2_evals = _cma_planned_evals(mc_cma_popsize, mc_cma_max_gen, mc_cma_restarts, len(ps2.bounds))

            n_cycles = int(parse_number(mc_n_cycles))
            total_evals = n_cycles * (s1_evals + s2_evals)
            lines.append(f"Multi-cycle ({mc_method}): {n_cycles} cycle(s) x (Stage 1 ~{s1_evals} + Stage 2 ~{s2_evals} evals) = ~{total_evals} evaluations")
        else:
            ps = ps1 if run_type == "stage1" else ps2
            objective_cls = Stage1Objective if run_type == "stage1" else Stage2Objective
            batch_time, batch_n, quick_rejected = _time_one_batch(ps, weights, normalization, objective_cls, n_jobs)
            per_eval = batch_time / batch_n

            if opt_method == "lhs":
                total_evals = _lhs_planned_evals(n_stages, n_samples_per_stage)
            else:
                total_evals = _cma_planned_evals(opt_cma_popsize, opt_cma_max_gen, opt_cma_restarts, len(ps.bounds))
            lines.append(f"{run_type} ({opt_method}): ~{total_evals} evaluations")

        total_seconds = total_evals * per_eval
        lines.append(f"Timed {batch_n} real evaluation(s) in parallel just now: {batch_time:.1f}s ({per_eval:.2f}s/eval effective, includes parallel-worker overhead)")
        lines.append(f"Estimated total time: ~{total_seconds / 60:.1f} min ({total_seconds:.0f}s)")
        if quick_rejected:
            lines.append("NOTE: the baseline design itself got caught by the cheap pre-check (see Bounds & Weights), so this timing is near-instant and NOT representative -- a real run's candidates will mostly cost far more than this. Treat the estimate above as a large UNDER-estimate.")
        lines.append("Rough estimate only: assumes every evaluation costs about the same as this one, doesn't know how many candidates a real run will cheaply pre-reject, and CMA-ES's eval count is an upper bound -- it often stops earlier via its own convergence check.")
        return "\n".join(lines)

    @app.callback(
        Output("run-status-display", "children"),
        Output("run-poll-interval", "disabled"),
        Output("run-progress-bar-fill", "style"),
        Output("run-progress-text", "children"),
        Output("run-log-display", "children"),
        Output("simple-mode-picked-settings", "children"),
        Input("start-run-button", "n_clicks"),
        Input("run-poll-interval", "n_intervals"),
        State("run-type-dropdown", "value"),
        State("baseline-source-radio", "value"),
        State("baseline-result-dropdown", "value"),
        State("setup-mode-radio", "value"), State("simple-target-minutes", "value"),
        State("opt-method-dropdown", "value"),
        State("opt-cma-sigma0", "value"), State("opt-cma-population-size", "value"),
        State("opt-cma-max-generations", "value"), State("opt-cma-n-restarts", "value"),
        State("opt-n-stages", "value"), State("opt-n-samples-per-stage", "value"),
        State("opt-retain-best-n", "value"), State("opt-seed", "value"), State("opt-n-jobs", "value"),
        State("mc-n-cycles", "value"), State("mc-start-with", "value"),
        State("mc-method-dropdown", "value"),
        State("mc-cma-sigma0", "value"), State("mc-cma-population-size", "value"),
        State("mc-cma-max-generations", "value"), State("mc-cma-n-restarts", "value"),
        State("mc-stage1-n-stages", "value"), State("mc-stage1-n-samples-per-stage", "value"), State("mc-stage1-retain-best-n", "value"),
        State("mc-stage2-n-stages", "value"), State("mc-stage2-n-samples-per-stage", "value"), State("mc-stage2-retain-best-n", "value"),
        State("mc-seed", "value"), State("mc-n-jobs", "value"),
        *DESIGN_STATE_INPUTS,
        prevent_initial_call=True,
    )
    def start_or_poll(
        n_clicks, n_intervals, run_type, baseline_source, existing_result_name,
        setup_mode, simple_target_minutes,
        opt_method, opt_cma_sigma0, opt_cma_popsize, opt_cma_max_gen, opt_cma_restarts,
        n_stages, n_samples_per_stage, retain_best_n, seed, n_jobs,
        mc_n_cycles, mc_start_with,
        mc_method, mc_cma_sigma0, mc_cma_popsize, mc_cma_max_gen, mc_cma_restarts,
        mc_s1_stages, mc_s1_samples, mc_s1_retain,
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
                return f"Running ({name})...", False, bar_style, progress_text, log, dash.no_update
            if st == "completed":
                return f"Completed ({name}). Open the Results tab and select '{name}' to inspect it.", True, bar_style, progress_text, log, dash.no_update
            if st == "failed":
                return f"FAILED ({name}) -- see log below.", True, bar_style, progress_text, log, dash.no_update
            return "Idle.", True, dict(_PROGRESS_BAR_FILL_BASE_STYLE), "", "", dash.no_update

        # triggered by the Start Run button
        if run_manager.is_running():
            return "A run is already in progress -- wait for it to finish first.", False, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        output_dir_name = f"{run_type}_run_{datetime.now():%Y%m%d_%H%M%S}"

        if baseline_source == "default":
            baseline = load_default_design_parameters()
        elif baseline_source == "design_tab":
            baseline = build_params_from_inputs(*design_state_values)
        else:
            if not existing_result_name:
                return "Pick an existing result to use as the baseline first.", True, dash.no_update, dash.no_update, dash.no_update, dash.no_update
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

        picked_settings_text = ""
        if setup_mode == "simple":
            # Beginner-friendly path: the user only picked a run type,
            # baseline, and target duration. Time one real batch of the
            # baseline (same mechanism as "Estimate Time") and invert it into
            # CMA-ES settings that roughly fill that time -- see
            # optimization/auto_tune.py for the allocation math. Always
            # CMA-ES (the best-evidenced default, see cmaes.py), always uses
            # most of the machine's cores.
            weights, normalization = _load_current_weights(), _load_current_normalization()
            ps1 = make_stage1_parameter_set(baseline)
            ps2 = make_stage2_parameter_set(baseline)
            n_jobs_auto = auto_tune.default_n_jobs()
            target_seconds = max(1.0, parse_number(simple_target_minutes) * 60.0)

            if run_type == "multi_cycle":
                batch_time, batch_n, _ = _time_one_batch(ps1, weights, normalization, Stage1Objective, n_jobs_auto)
                per_eval = batch_time / batch_n
                total_evals_budget = target_seconds / per_eval
                # n_cycles has a *weak* dependency on the budget
                # (auto_n_cycles), not a fixed value -- see auto_tune.py.
                # Stage 1 and Stage 2 have different dimensionality (so
                # different auto population sizes), but the CLI only accepts
                # one shared max_generations/n_restarts pair for both stages
                # -- use Stage 2's (the larger, more expensive problem),
                # which keeps the run from overshooting the target; Stage 1
                # then simply finishes a bit early. sigma0 only depends on
                # the (shared) per-substage budget, so it's identical either way.
                n_cycles = auto_tune.auto_n_cycles(total_evals_budget, len(ps2.bounds))
                per_substage_evals = total_evals_budget / (n_cycles * 2)
                s2 = auto_tune.auto_cma_settings(len(ps2.bounds), per_substage_evals)
                s1 = auto_tune.auto_cma_settings(len(ps1.bounds), per_substage_evals)
                mc_n_cycles, mc_method = n_cycles, "cma"
                mc_cma_sigma0, mc_cma_popsize = s2.sigma0, None
                mc_cma_max_gen, mc_cma_restarts = s2.max_generations, s2.n_restarts
                mc_n_jobs = n_jobs_auto
                picked_settings_text = (
                    f"Simple mode auto-picked (measured {per_eval:.2f}s/eval): CMA-ES, {mc_n_cycles} cycles, n_jobs={n_jobs_auto}\n"
                    f"  Stage 1: sigma0={s1.sigma0:.3f}, population~{s1.population_size} (auto), max_generations={mc_cma_max_gen}, restarts={mc_cma_restarts}\n"
                    f"  Stage 2: sigma0={s2.sigma0:.3f}, population~{s2.population_size} (auto), max_generations={mc_cma_max_gen}, restarts={mc_cma_restarts}  "
                    "(this pair also used for Stage 1, so it doesn't overshoot the budget)"
                )
            else:
                ps = ps1 if run_type == "stage1" else ps2
                objective_cls = Stage1Objective if run_type == "stage1" else Stage2Objective
                batch_time, batch_n, _ = _time_one_batch(ps, weights, normalization, objective_cls, n_jobs_auto)
                per_eval = batch_time / batch_n
                s = auto_tune.auto_cma_settings(len(ps.bounds), target_seconds / per_eval)
                opt_method = "cma"
                opt_cma_sigma0, opt_cma_popsize = s.sigma0, None
                opt_cma_max_gen, opt_cma_restarts = s.max_generations, s.n_restarts
                n_jobs = n_jobs_auto
                picked_settings_text = (
                    f"Simple mode auto-picked (measured {per_eval:.2f}s/eval): CMA-ES, sigma0={s.sigma0:.3f}, "
                    f"population~{s.population_size} (auto), max_generations={s.max_generations}, "
                    f"restarts={s.n_restarts}, n_jobs={n_jobs_auto}"
                )

        def _cma_args(sigma0, popsize, max_gen, restarts) -> list[str]:
            # population_size left unset (None) means "let the script fall
            # back to pycma's own default heuristic" -- omit the flag rather
            # than passing a literal "None" string.
            a = [
                "--cma-sigma0", str(parse_number(sigma0)),
                "--cma-max-generations", str(int(parse_number(max_gen))),
                "--cma-n-restarts", str(int(parse_number(restarts))),
            ]
            popsize_val = parse_number(popsize)
            if popsize_val is not None:
                a += ["--cma-population-size", str(int(popsize_val))]
            return a

        if run_type == "multi_cycle":
            args = common_args + ["--optimizer", mc_method] + [
                "--n-cycles", str(int(parse_number(mc_n_cycles))), "--start-with", mc_start_with,
                "--stage1-n-stages", str(int(parse_number(mc_s1_stages))), "--stage1-n-samples-per-stage", str(int(parse_number(mc_s1_samples))),
                "--stage1-retain-best-n", str(int(parse_number(mc_s1_retain))),
                "--stage2-n-stages", str(int(parse_number(mc_s2_stages))), "--stage2-n-samples-per-stage", str(int(parse_number(mc_s2_samples))),
                "--stage2-retain-best-n", str(int(parse_number(mc_s2_retain))),
                "--seed", str(int(parse_number(mc_seed))), "--n-jobs", str(int(parse_number(mc_n_jobs))),
            ] + (_cma_args(mc_cma_sigma0, mc_cma_popsize, mc_cma_max_gen, mc_cma_restarts) if mc_method == "cma" else [])
        else:
            args = common_args + ["--optimizer", opt_method] + [
                "--n-stages", str(int(parse_number(n_stages))), "--n-samples-per-stage", str(int(parse_number(n_samples_per_stage))),
                "--retain-best-n", str(int(parse_number(retain_best_n))), "--seed", str(int(parse_number(seed))), "--n-jobs", str(int(parse_number(n_jobs))),
            ] + (_cma_args(opt_cma_sigma0, opt_cma_popsize, opt_cma_max_gen, opt_cma_restarts) if opt_method == "cma" else [])

        run_manager.launch_run(run_type, args, output_dir_name)
        return f"Started {run_type} run '{output_dir_name}'...", False, dict(_PROGRESS_BAR_FILL_BASE_STYLE), "", "", picked_settings_text
