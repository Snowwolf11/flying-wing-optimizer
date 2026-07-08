"""Interactive control-panel GUI: a thin Dash shell combining 6 tabs.

- Design: live single-design parameter editing (design_tab.py)
- Bounds & Weights: edit objective weights + search bounds/structural
  constants, persisted to YAML (config_tab.py)
- Run Optimizer: launch Stage 1/2/multi-cycle runs as background
  subprocesses of the existing scripts/run_*.py, and poll their progress
  (run_tab.py, run_manager.py)
- Results: browse/load/plot past runs from outputs/ (results_tab.py,
  results_io.py)
- Deep Analysis: a consolidated post-run report (score breakdown, mass/CG,
  structural detail, Cp-based flow visualization) for one selected run
  (analysis_tab.py)
- Documentation: renders README.md in-app (docs_tab.py)

All 6 tabs' layouts are built once, up front, and are simultaneously
present in the DOM (Dash's static `dcc.Tabs` pattern) -- this keeps every
callback's target components available from the start, so cross-tab
callbacks (e.g. Results' "send to Design tab", Run's "use Design tab
values") work without any dynamic-layout bookkeeping.
"""
from __future__ import annotations

import dash
from dash import dcc, html

from . import design_tab, config_tab, run_tab, results_tab, analysis_tab, docs_tab


def build_layout() -> html.Div:
    return html.Div(
        [
            dcc.Tabs(
                id="main-tabs",
                value="design",
                children=[
                    dcc.Tab(label="Design", value="design", children=design_tab.layout()),
                    dcc.Tab(label="Bounds & Weights", value="config", children=config_tab.layout()),
                    dcc.Tab(label="Run Optimizer", value="run", children=run_tab.layout()),
                    dcc.Tab(label="Results", value="results", children=results_tab.layout()),
                    dcc.Tab(label="Deep Analysis", value="analysis", children=analysis_tab.layout()),
                    dcc.Tab(label="Documentation", value="docs", children=docs_tab.layout()),
                ],
            ),
        ]
    )


def create_app() -> dash.Dash:
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = build_layout()

    design_tab.register_callbacks(app)
    config_tab.register_callbacks(app)
    run_tab.register_callbacks(app)
    results_tab.register_callbacks(app)
    analysis_tab.register_callbacks(app)
    docs_tab.register_callbacks(app)

    return app
