"""Documentation tab: renders the project's README.md directly in the GUI,
so the pipeline/parameter/bounds documentation is available without leaving
the app. Read fresh on every tab switch (not cached at import time) so
README edits show up on next visit without restarting the GUI.
"""
from __future__ import annotations

import dash
from dash import dcc, html, Input, Output

from ..config import PROJECT_ROOT

README_PATH = PROJECT_ROOT / "README.md"


def _read_readme() -> str:
    try:
        return README_PATH.read_text()
    except FileNotFoundError:
        return f"README.md not found at {README_PATH}."


def layout() -> html.Div:
    return html.Div(
        [
            dcc.Markdown(
                id="docs-readme-content",
                children=_read_readme(),
                style={"maxWidth": "900px", "margin": "0 auto", "padding": "20px"},
            ),
        ],
        style={"padding": "10px"},
    )


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("docs-readme-content", "children"),
        Input("main-tabs", "value"),
    )
    def refresh_on_visit(active_tab):
        if active_tab != "docs":
            raise dash.exceptions.PreventUpdate
        return _read_readme()
