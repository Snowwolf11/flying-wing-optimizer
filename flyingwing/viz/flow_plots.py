"""Flow visualization -- pressure coefficient (Cp) distributions, built from
NeuralFoil's boundary-layer solution rather than a real CFD solver (none is
installed/available for this project). NeuralFoil reports the boundary-
layer edge velocity ratio (ue/Vinf) at 32 fixed panel midpoints (uniformly
spaced in x/c -- see neuralfoil's `compute_optimal_x_points`) on each of the
upper/lower surfaces; Cp = 1 - (ue/Vinf)^2 (incompressible Bernoulli) turns
that into a real, boundary-layer-informed pressure distribution. This is
genuine section aerodynamic data, just not a full 3D flow field.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..geometry.aircraft import Aircraft
from ..geometry.airfoil_family import generate_airfoil, _cosine_x_grid, N_SURFACE_POINTS
from ..geometry.mesh import build_watertight_mesh
from ..analysis.airfoil_2d import reynolds_number

N_BL_STATIONS = 32


def _bl_x_over_c(n: int = N_BL_STATIONS) -> np.ndarray:
    """Matches neuralfoil's internal `compute_optimal_x_points`: n uniformly
    spaced panel midpoints in [0, 1]."""
    s = np.linspace(0.0, 1.0, n + 1)
    return (s[1:] + s[:-1]) / 2.0


BL_X_OVER_C = _bl_x_over_c()


def compute_section_cp(
    thickness_scale: float, camber_scale: float, reflex_scale: float,
    chord_m: float, alpha_deg: float, speed_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x_over_c, cp_upper, cp_lower) at one airfoil section."""
    airfoil = generate_airfoil(thickness_scale, camber_scale, reflex_scale)
    re = reynolds_number(chord_m, speed_ms)
    result = airfoil.get_aero_from_neuralfoil(alpha=alpha_deg, Re=re, mach=0.0, n_crit=9.0, model_size="large")
    # NeuralFoil wraps each scalar in a 1-element array; np.ravel(...)[0]
    # (not float(...) directly) extracts it -- matches analysis/aero_3d.py's
    # _scalar() helper, needed since plain float() on a shape-(1,) array is
    # no longer implicitly allowed as of NumPy 2.x.
    ue_upper = np.array([np.ravel(result[f"upper_bl_ue/vinf_{i}"])[0] for i in range(N_BL_STATIONS)])
    ue_lower = np.array([np.ravel(result[f"lower_bl_ue/vinf_{i}"])[0] for i in range(N_BL_STATIONS)])
    return BL_X_OVER_C, 1.0 - ue_upper ** 2, 1.0 - ue_lower ** 2


def plot_cp_sections(aircraft: Aircraft, alpha_deg: float, speed_ms: float, n_sections: int = 5) -> go.Figure:
    """Classic Cp-vs-x/c overlay at a few representative span stations
    (root to tip). Cp axis is inverted (suction plotted upward), the usual
    aerodynamics convention."""
    idx = np.linspace(0, len(aircraft.y_stations) - 1, n_sections).astype(int)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]

    fig = go.Figure()
    for n, i in enumerate(idx):
        x, cp_u, cp_l = compute_section_cp(
            aircraft.thickness_scale[i], aircraft.camber_scale[i], aircraft.reflex_scale[i],
            aircraft.chord_m[i], alpha_deg, speed_ms,
        )
        color = colors[n % len(colors)]
        label = f"y={aircraft.y_stations[i]:.2f}"
        fig.add_trace(go.Scatter(x=x, y=cp_u, mode="lines", line=dict(color=color), name=f"{label} upper", legendgroup=label))
        fig.add_trace(go.Scatter(x=x, y=cp_l, mode="lines", line=dict(color=color, dash="dot"), name=f"{label} lower", legendgroup=label))

    fig.update_xaxes(title_text="x/c")
    fig.update_yaxes(title_text="Cp", autorange="reversed")
    fig.update_layout(title=f"Pressure Coefficient Distribution -- alpha={alpha_deg:.1f} deg, V={speed_ms:.1f} m/s", height=480)
    return fig


def plot_cp_heatmap(aircraft: Aircraft, alpha_deg: float, speed_ms: float, n_span_stations: int = 21) -> go.Figure:
    """Spanwise Cp 'pressure map' (x/c vs span, colored by Cp) for the upper
    and lower surfaces side by side -- a 2D substitute for a full 3D CFD
    surface plot, built from the same per-section NeuralFoil Cp data as
    plot_cp_sections."""
    idx = np.linspace(0, len(aircraft.y_stations) - 1, n_span_stations).astype(int)
    y_vals = aircraft.y_stations[idx]
    cp_upper_grid = np.zeros((n_span_stations, N_BL_STATIONS))
    cp_lower_grid = np.zeros((n_span_stations, N_BL_STATIONS))

    for row, i in enumerate(idx):
        _, cp_u, cp_l = compute_section_cp(
            aircraft.thickness_scale[i], aircraft.camber_scale[i], aircraft.reflex_scale[i],
            aircraft.chord_m[i], alpha_deg, speed_ms,
        )
        cp_upper_grid[row] = cp_u
        cp_lower_grid[row] = cp_l

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Upper surface Cp", "Lower surface Cp"), shared_yaxes=True)
    fig.add_trace(
        go.Heatmap(x=BL_X_OVER_C, y=y_vals, z=cp_upper_grid, colorscale="RdBu_r", zmid=0, colorbar=dict(title="Cp", x=0.44)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(x=BL_X_OVER_C, y=y_vals, z=cp_lower_grid, colorscale="RdBu_r", zmid=0, colorbar=dict(title="Cp", x=1.0)),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="x/c", row=1, col=1)
    fig.update_xaxes(title_text="x/c", row=1, col=2)
    fig.update_yaxes(title_text="y (normalized span)", row=1, col=1)
    fig.update_layout(title=f"Spanwise Pressure Distribution -- alpha={alpha_deg:.1f} deg, V={speed_ms:.1f} m/s", height=480)
    return fig


def _mirror_full_span_scalar(half: np.ndarray) -> np.ndarray:
    """(N, M) half-span scalar field -> (2N-1, M) full span, sharing the
    y=0 row -- same spanwise mirroring geometry.mesh._mirror_full_span uses
    for (x, y, z) coordinate arrays, but without negating anything (a
    scalar field like Cp has no spanwise sign to flip; the left wing's Cp
    at a mirrored station just equals the right wing's)."""
    negative_side = half[1:][::-1]
    return np.concatenate([negative_side, half], axis=0)


def plot_cp_surface_3d(aircraft: Aircraft, alpha_deg: float, speed_ms: float, n_span_samples: int = 50) -> go.Figure:
    """The aircraft's own watertight mesh (see geometry/mesh.py), colored
    per-vertex by Cp -- a true 3D pressure-mapped surface rather than a 2D
    proxy plot. Cp is computed via NeuralFoil at `n_span_samples` stations
    (not all ~200 -- NeuralFoil calls dominate the cost) and interpolated
    spanwise onto the full mesh resolution; chordwise, it's interpolated
    from NeuralFoil's 32-point grid onto the mesh's own 161-point cosine
    x/c grid. Vertex ordering must exactly match
    geometry.mesh.build_watertight_mesh's (upper-then-lower, spanwise-
    mirrored) concatenation -- see that function for the authoritative
    layout this mirrors."""
    mesh = build_watertight_mesh(aircraft)
    y_stations = aircraft.y_stations
    n_span = len(y_stations)

    idx = np.linspace(0, n_span - 1, min(n_span_samples, n_span)).astype(int)
    idx = np.unique(idx)
    cp_upper_sampled = np.zeros((len(idx), N_BL_STATIONS))
    cp_lower_sampled = np.zeros((len(idx), N_BL_STATIONS))
    for row, i in enumerate(idx):
        _, cp_u, cp_l = compute_section_cp(
            aircraft.thickness_scale[i], aircraft.camber_scale[i], aircraft.reflex_scale[i],
            aircraft.chord_m[i], alpha_deg, speed_ms,
        )
        cp_upper_sampled[row] = cp_u
        cp_lower_sampled[row] = cp_l

    # Spanwise: interpolate each of the 32 NeuralFoil x/c columns from the
    # sampled stations onto all n_span stations.
    cp_upper_by_bl_x = np.array([np.interp(y_stations, y_stations[idx], cp_upper_sampled[:, k]) for k in range(N_BL_STATIONS)]).T
    cp_lower_by_bl_x = np.array([np.interp(y_stations, y_stations[idx], cp_lower_sampled[:, k]) for k in range(N_BL_STATIONS)]).T

    # Chordwise: interpolate from the 32-point NeuralFoil grid onto the
    # mesh's own cosine x/c grid (161 points, matching aircraft.upper/
    # lower_surface_m's chordwise resolution).
    mesh_x_over_c = _cosine_x_grid(N_SURFACE_POINTS)
    cp_upper = np.array([np.interp(mesh_x_over_c, BL_X_OVER_C, cp_upper_by_bl_x[j]) for j in range(n_span)])
    cp_lower = np.array([np.interp(mesh_x_over_c, BL_X_OVER_C, cp_lower_by_bl_x[j]) for j in range(n_span)])

    cp_upper_full = _mirror_full_span_scalar(cp_upper)
    cp_lower_full = _mirror_full_span_scalar(cp_lower)
    intensity = np.concatenate([cp_upper_full.reshape(-1), cp_lower_full.reshape(-1)])

    v, f = mesh.vertices, mesh.faces
    fig = go.Figure(data=[
        go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            intensity=intensity, colorscale="RdBu_r", cmid=0.0,
            colorbar=dict(title="Cp"), flatshading=False, showscale=True,
        )
    ])
    fig.update_layout(
        title=f"Pressure-Mapped Surface -- alpha={alpha_deg:.1f} deg, V={speed_ms:.1f} m/s",
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0), height=600,
    )
    return fig
