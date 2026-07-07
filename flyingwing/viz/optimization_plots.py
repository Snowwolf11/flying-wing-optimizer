"""Optimization visualizations: convergence history and parameter evolution.

Pareto plots are noted in the project spec as future work (once a
multi-objective algorithm replaces the single-scalar-score hierarchical
search) and aren't implemented yet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..optimization.base import OptimizationResult
from ..optimization.cycle import MultiCycleResult


def plot_convergence(result: OptimizationResult) -> go.Figure:
    stages = list(range(len(result.history)))
    best_so_far = []
    stage_best = []
    stage_worst = []
    running_best = -np.inf
    for stage_candidates in result.history:
        scores = [c.score for c in stage_candidates]
        stage_best.append(max(scores))
        stage_worst.append(min(scores))
        running_best = max(running_best, max(scores))
        best_so_far.append(running_best)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stages, y=best_so_far, mode="lines+markers", name="best so far"))
    fig.add_trace(go.Scatter(x=stages, y=stage_best, mode="markers", name="stage best"))
    fig.add_trace(go.Scatter(x=stages, y=stage_worst, mode="markers", name="stage worst", marker=dict(symbol="x")))
    fig.update_layout(
        title="Optimization Convergence", xaxis_title="stage", yaxis_title="objective score", height=420,
    )
    return fig


def plot_parameter_evolution(result: OptimizationResult, variable_names: list[str]) -> go.Figure:
    """How the best candidate's parameters moved across stages."""
    n_vars = len(variable_names)
    best_x_per_stage = []
    running_best_candidate = max(result.history[0], key=lambda c: c.score)
    for stage_candidates in result.history:
        stage_best = max(stage_candidates, key=lambda c: c.score)
        if stage_best.score > running_best_candidate.score:
            running_best_candidate = stage_best
        best_x_per_stage.append(running_best_candidate.x)
    best_x_per_stage = np.array(best_x_per_stage)  # (n_stages, n_vars)

    n_cols = 3
    n_rows = int(np.ceil(n_vars / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=variable_names)

    stages = list(range(len(result.history)))
    for i, name in enumerate(variable_names):
        row, col = i // n_cols + 1, i % n_cols + 1
        fig.add_trace(go.Scatter(x=stages, y=best_x_per_stage[:, i], mode="lines+markers"), row=row, col=col)

    fig.update_layout(title="Best-Candidate Parameter Evolution", showlegend=False, height=250 * n_rows)
    return fig


def plot_multi_cycle_convergence(multi_result: MultiCycleResult) -> go.Figure:
    labels = [f"C{r.cycle}-{r.stage}" for r in multi_result.records]
    scores = multi_result.score_history
    colors = ["#1f77b4" if r.stage == "stage1" else "#ff7f0e" for r in multi_result.records]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=scores, mode="lines+markers", marker=dict(color=colors, size=10)))
    fig.update_layout(
        title="Multi-Cycle Convergence (blue = Stage 1, orange = Stage 2)",
        xaxis_title="cycle-stage", yaxis_title="best objective score", height=420,
    )
    return fig


def save_all(result: OptimizationResult, variable_names: list[str], output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "convergence.html": plot_convergence(result),
        "parameter_evolution.html": plot_parameter_evolution(result, variable_names),
    }
    paths = {}
    for filename, fig in figures.items():
        path = output_dir / filename
        fig.write_html(str(path), include_plotlyjs="cdn")
        paths[filename] = path
    return paths
