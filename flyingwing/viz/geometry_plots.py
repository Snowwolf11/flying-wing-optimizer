"""Geometry visualizations: interactive 3D aircraft, orthographic views, and
airfoil-distribution plots. All functions return a `plotly.graph_objects.Figure`;
`save_all` writes them to an output directory as standalone HTML files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..geometry.aircraft import Aircraft
from ..geometry.mesh import build_watertight_mesh
from ..geometry.airfoil_family import generate_airfoil_surfaces
from ..config import FUSELAGE_MIN_INTERNAL_WIDTH_M


def _fuselage_box_bounds(aircraft: Aircraft) -> tuple[float, float, float, float, float, float]:
    """(x_min, x_max, y_min, y_max, z_min, z_max) of the required internal
    fuselage box. x/z come directly from geometry.fuselage.check_fuselage_fit's
    search for the best-fitting placement (already computed once per
    build_aircraft() call, stored on aircraft.fuselage_fit) -- width is
    fixed by symmetry about the centerline. If no valid placement was found
    (aircraft.fuselage_fit.fits is False and box_x_min_m is NaN), falls back
    to a simple root-chord-centered box purely so there's still something to
    draw."""
    half_width = FUSELAGE_MIN_INTERNAL_WIDTH_M / 2.0
    fit = aircraft.fuselage_fit

    if np.isnan(fit.box_x_min_m):
        from ..config import FUSELAGE_MIN_INTERNAL_HEIGHT_M, FUSELAGE_MIN_INTERNAL_LENGTH_M
        x_center = float(aircraft.x_le_m[0] + aircraft.chord_m[0] / 2.0)
        z_center = float(aircraft.z_le_m[0])
        return (
            x_center - FUSELAGE_MIN_INTERNAL_LENGTH_M / 2.0, x_center + FUSELAGE_MIN_INTERNAL_LENGTH_M / 2.0,
            -half_width, half_width,
            z_center - FUSELAGE_MIN_INTERNAL_HEIGHT_M / 2.0, z_center + FUSELAGE_MIN_INTERNAL_HEIGHT_M / 2.0,
        )

    return (fit.box_x_min_m, fit.box_x_max_m, -half_width, half_width, fit.box_z_min_m, fit.box_z_max_m)


def _fuselage_box_mesh3d(aircraft: Aircraft) -> go.Mesh3d:
    x0, x1, y0, y1, z0, z1 = _fuselage_box_bounds(aircraft)
    xs = [x0, x0, x1, x1, x0, x0, x1, x1]
    ys = [y0, y1, y1, y0, y0, y1, y1, y0]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]
    # alphahull=0 -> convex hull of the 8 corner points, i.e. exactly a box,
    # without hand-rolling face-index triangulation.
    return go.Mesh3d(
        x=xs, y=ys, z=zs, alphahull=0,
        color="orange", opacity=0.3, name="required fuselage box", showlegend=True,
    )


def plot_3d_aircraft(aircraft: Aircraft) -> go.Figure:
    mesh = build_watertight_mesh(aircraft)
    v, f = mesh.vertices, mesh.faces

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color="lightsteelblue",
                flatshading=False,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5),
                lightposition=dict(x=0, y=1000, z=1000),
                showscale=False,
                name="aircraft",
            ),
            _fuselage_box_mesh3d(aircraft),
        ]
    )
    fig.update_layout(
        title="Flying Wing -- 3D Model",
        scene=dict(
            xaxis_title="x (m, streamwise)",
            yaxis_title="y (m, spanwise)",
            zaxis_title="z (m, vertical)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _planform_outline(aircraft: Aircraft) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leading and trailing edge (x, y, z) polylines for one half of the
    wing, mirrored to show the full planform."""
    y = aircraft.span_station_m
    x_le = aircraft.x_le_m
    x_te = aircraft.x_le_m + aircraft.chord_m
    z_le = aircraft.z_le_m

    y_full = np.concatenate([-y[::-1], y[1:]])
    x_le_full = np.concatenate([x_le[::-1], x_le[1:]])
    x_te_full = np.concatenate([x_te[::-1], x_te[1:]])
    z_le_full = np.concatenate([z_le[::-1], z_le[1:]])
    return y_full, x_le_full, x_te_full, z_le_full


def plot_orthographic_views(aircraft: Aircraft) -> go.Figure:
    y_full, x_le_full, x_te_full, z_le_full = _planform_outline(aircraft)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Top view", "Front view", "Side view"),
    )

    # Top view: y (span) vs x (chordwise), outline of LE + TE
    outline_x = np.concatenate([x_le_full, x_te_full[::-1], [x_le_full[0]]])
    outline_y = np.concatenate([y_full, y_full[::-1], [y_full[0]]])
    fig.add_trace(go.Scatter(x=outline_y, y=outline_x, mode="lines", fill="toself", name="planform"), row=1, col=1)
    fig.update_xaxes(title_text="y (m)", row=1, col=1)
    fig.update_yaxes(title_text="x (m)", autorange="reversed", row=1, col=1, scaleanchor="x1")

    # Front view: y (span) vs z (vertical), leading edge height + a thickness hint
    thickness_full = np.concatenate([aircraft.thickness_ratio[::-1], aircraft.thickness_ratio[1:]]) * np.concatenate(
        [aircraft.chord_m[::-1], aircraft.chord_m[1:]]
    )
    fig.add_trace(go.Scatter(x=y_full, y=z_le_full + thickness_full / 2, mode="lines", name="upper", line=dict(color="steelblue")), row=1, col=2)
    fig.add_trace(go.Scatter(x=y_full, y=z_le_full - thickness_full / 2, mode="lines", name="lower", line=dict(color="steelblue"), fill="tonexty"), row=1, col=2)
    fig.update_xaxes(title_text="y (m)", row=1, col=2)
    fig.update_yaxes(title_text="z (m)", row=1, col=2, scaleanchor="x2")

    # Side view: x (chordwise) vs z (vertical) -- root and tip sections plus LE/TE lines
    fig.add_trace(go.Scatter(x=x_le_full, y=z_le_full, mode="lines", name="LE", line=dict(color="firebrick")), row=1, col=3)
    fig.add_trace(go.Scatter(x=x_te_full, y=z_le_full, mode="lines", name="TE", line=dict(color="darkorange")), row=1, col=3)
    fig.update_xaxes(title_text="x (m)", row=1, col=3)
    fig.update_yaxes(title_text="z (m)", row=1, col=3, scaleanchor="x3")

    # Required fuselage box outline, dashed, on all three views -- same box
    # as plot_3d_aircraft's Mesh3d, projected onto each 2D plane.
    x0, x1, y0, y1, z0, z1 = _fuselage_box_bounds(aircraft)
    box_style = dict(mode="lines", line=dict(color="orange", dash="dash"), name="fuselage box", showlegend=False)

    def _rect(a0, a1, b0, b1):
        return [a0, a1, a1, a0, a0], [b0, b0, b1, b1, b0]

    rx, ry = _rect(y0, y1, x0, x1)
    fig.add_trace(go.Scatter(x=rx, y=ry, **box_style), row=1, col=1)
    rx, ry = _rect(y0, y1, z0, z1)
    fig.add_trace(go.Scatter(x=rx, y=ry, **box_style), row=1, col=2)
    rx, ry = _rect(x0, x1, z0, z1)
    fig.add_trace(go.Scatter(x=rx, y=ry, **box_style), row=1, col=3)

    fit_status = "OK" if aircraft.fuselage_fit.fits else "FAILS"
    fig.update_layout(
        title=f"Flying Wing -- Orthographic Views  (fuselage fit: {fit_status}, "
              f"height margin {aircraft.fuselage_fit.min_height_margin_m * 1000:+.1f} mm, "
              f"length margin {aircraft.fuselage_fit.min_length_margin_m * 1000:+.1f} mm)",
        showlegend=False, height=450,
    )
    return fig


def plot_airfoil_distribution(aircraft: Aircraft, n_sections: int = 7) -> go.Figure:
    """Overlay normalized (x/c, z/c) airfoil shapes at several span stations,
    plus the spanwise thickness/camber/reflex scale distributions."""
    idx = np.linspace(0, len(aircraft.y_stations) - 1, n_sections).astype(int)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Airfoil sections along the span", "Airfoil schedule (thickness / camber / reflex scale)"),
        row_heights=[0.55, 0.45],
        vertical_spacing=0.12,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]
    for n, i in enumerate(idx):
        x, upper, lower = generate_airfoil_surfaces(
            aircraft.thickness_scale[i], aircraft.camber_scale[i], aircraft.reflex_scale[i]
        )
        color = colors[n % len(colors)]
        y_label = aircraft.y_stations[i]
        fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=dict(color=color), name=f"y={y_label:.2f}", legendgroup=f"g{n}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=lower, mode="lines", line=dict(color=color), showlegend=False, legendgroup=f"g{n}"), row=1, col=1)

    fig.update_xaxes(title_text="x/c", row=1, col=1)
    fig.update_yaxes(title_text="z/c", scaleanchor="x", scaleratio=1, row=1, col=1)

    fig.add_trace(go.Scatter(x=aircraft.y_stations, y=aircraft.thickness_scale, mode="lines", name="thickness_scale"), row=2, col=1)
    fig.add_trace(go.Scatter(x=aircraft.y_stations, y=aircraft.camber_scale, mode="lines", name="camber_scale"), row=2, col=1)
    fig.add_trace(go.Scatter(x=aircraft.y_stations, y=aircraft.reflex_scale, mode="lines", name="reflex_scale"), row=2, col=1)
    fig.update_xaxes(title_text="y (normalized span)", row=2, col=1)
    fig.update_yaxes(title_text="scale", row=2, col=1)

    fig.update_layout(title="Airfoil Family Distribution", height=750)
    return fig


def save_all(aircraft: Aircraft, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "aircraft_3d.html": plot_3d_aircraft(aircraft),
        "orthographic_views.html": plot_orthographic_views(aircraft),
        "airfoil_distribution.html": plot_airfoil_distribution(aircraft),
    }
    paths = {}
    for filename, fig in figures.items():
        path = output_dir / filename
        fig.write_html(str(path), include_plotlyjs="cdn")
        paths[filename] = path
    return paths
