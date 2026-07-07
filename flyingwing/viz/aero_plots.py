"""Aerodynamic visualizations: drag polar, spanwise lift/CL/Reynolds
distributions. Spanwise lift/CL reuse the same Schrenk approximation as the
structural proxy (`analysis.structures.schrenk_lift_per_span`), so the
"lift distribution" and "CL distribution" plots here are consistent with
whatever load the structural plots show, rather than a separately invented
approximation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..geometry.aircraft import Aircraft
from ..analysis.aero_3d import analyze_aerobuildup
from ..analysis.structures import schrenk_lift_per_span
from ..config import AIR_DENSITY_KG_M3, KINEMATIC_VISCOSITY_M2_S

DEFAULT_ALPHA_SWEEP_DEG = np.arange(-4.0, 12.01, 2.0)


def compute_drag_polar(
    aircraft: Aircraft, speed_ms: float, alpha_range_deg: np.ndarray = DEFAULT_ALPHA_SWEEP_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    CL, CD, Cm = [], [], []
    for alpha in alpha_range_deg:
        m = analyze_aerobuildup(aircraft, speed_ms, alpha_deg=float(alpha))
        CL.append(m.CL)
        CD.append(m.CD)
        Cm.append(m.Cm)
    return np.asarray(alpha_range_deg), np.asarray(CL), np.asarray(CD), np.asarray(Cm)


def plot_drag_polar(aircraft: Aircraft, speed_ms: float, alpha_range_deg: np.ndarray = DEFAULT_ALPHA_SWEEP_DEG) -> go.Figure:
    alpha, CL, CD, Cm = compute_drag_polar(aircraft, speed_ms, alpha_range_deg)

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Drag polar", "CL vs alpha", "Cm vs alpha"))
    fig.add_trace(go.Scatter(x=CD, y=CL, mode="lines+markers", name="polar"), row=1, col=1)
    fig.update_xaxes(title_text="CD", row=1, col=1)
    fig.update_yaxes(title_text="CL", row=1, col=1)

    fig.add_trace(go.Scatter(x=alpha, y=CL, mode="lines+markers", name="CL(alpha)"), row=1, col=2)
    fig.update_xaxes(title_text="alpha (deg)", row=1, col=2)
    fig.update_yaxes(title_text="CL", row=1, col=2)

    fig.add_trace(go.Scatter(x=alpha, y=Cm, mode="lines+markers", name="Cm(alpha)"), row=1, col=3)
    fig.update_xaxes(title_text="alpha (deg)", row=1, col=3)
    fig.update_yaxes(title_text="Cm", row=1, col=3)

    fig.update_layout(title=f"Drag Polar -- V={speed_ms:.1f} m/s (AeroBuildup)", showlegend=False, height=420)
    return fig


def plot_spanwise_distributions(aircraft: Aircraft, speed_ms: float, trim_cl: float) -> go.Figure:
    q = 0.5 * AIR_DENSITY_KG_M3 * speed_ms ** 2
    lift_per_span = schrenk_lift_per_span(
        aircraft.span_station_m, aircraft.chord_m, aircraft.params.planform.span_m, aircraft.wing_area_m2, trim_cl, q,
    )
    local_cl = lift_per_span / (q * aircraft.chord_m)
    reynolds = speed_ms * aircraft.chord_m / KINEMATIC_VISCOSITY_M2_S

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Lift distribution (Schrenk)", "Local CL distribution", "Reynolds number distribution"),
    )
    fig.add_trace(go.Scatter(x=aircraft.y_stations, y=lift_per_span, mode="lines"), row=1, col=1)
    fig.update_xaxes(title_text="y (normalized span)", row=1, col=1)
    fig.update_yaxes(title_text="lift per span (N/m)", row=1, col=1)

    fig.add_trace(go.Scatter(x=aircraft.y_stations, y=local_cl, mode="lines"), row=1, col=2)
    fig.update_xaxes(title_text="y (normalized span)", row=1, col=2)
    fig.update_yaxes(title_text="local CL", row=1, col=2)

    fig.add_trace(go.Scatter(x=aircraft.y_stations, y=reynolds, mode="lines"), row=1, col=3)
    fig.update_xaxes(title_text="y (normalized span)", row=1, col=3)
    fig.update_yaxes(title_text="Re", row=1, col=3)

    fig.update_layout(title=f"Spanwise Distributions -- V={speed_ms:.1f} m/s, trim CL={trim_cl:.3f}", showlegend=False, height=420)
    return fig


def save_all(aircraft: Aircraft, speed_ms: float, trim_cl: float, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "drag_polar.html": plot_drag_polar(aircraft, speed_ms),
        "spanwise_distributions.html": plot_spanwise_distributions(aircraft, speed_ms, trim_cl),
    }
    paths = {}
    for filename, fig in figures.items():
        path = output_dir / filename
        fig.write_html(str(path), include_plotlyjs="cdn")
        paths[filename] = path
    return paths
