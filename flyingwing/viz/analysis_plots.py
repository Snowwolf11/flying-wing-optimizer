"""Deep-analysis plots: what's driving a design's score, where its mass and
CG actually come from. All of this data is already computed elsewhere
(`score().contributions`, `objective/mass.py`, `objective/cg.py`) but was
never visualized before -- these functions just make it visible.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..objective.objective import ObjectiveResult
from ..objective.mass import MassEstimate
from ..objective.cg import CGEstimate
from ..geometry.aircraft import Aircraft
from .geometry_plots import _planform_outline, _fuselage_box_bounds


def plot_objective_contributions(result: ObjectiveResult) -> go.Figure:
    """Every term's contribution to the total score, sorted by magnitude --
    shows why a design scored the way it did, not just that it did."""
    items = sorted(result.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]

    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors))
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(
        title=f"Objective Score Breakdown -- total {result.score:.2f}",
        xaxis_title="contribution to score", height=max(300, 30 * len(names)),
        margin=dict(l=140),
    )
    return fig


def plot_mass_breakdown(mass: MassEstimate, battery_mass_kg: float) -> go.Figure:
    """Structural mass breakdown (shell/spar/motor+ESC/avionics/servos) plus
    the battery -- shown separately since it's excluded from
    total_structural_mass_kg (see objective/mass.py)."""
    labels = ["shell", "spar", "motor/ESC", "avionics", "servos", "battery"]
    values = [
        mass.shell_mass_kg, mass.spar_mass_kg,
        mass.motor_esc_mass_kg, mass.avionics_mass_kg, mass.servo_mass_kg,
        battery_mass_kg,
    ]

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#1f77b4"))
    total_flying_mass = mass.total_structural_mass_kg + battery_mass_kg
    fig.update_layout(
        title=f"Mass Breakdown -- {mass.total_structural_mass_kg:.3f} kg structural + {battery_mass_kg:.3f} kg battery = {total_flying_mass:.3f} kg flying",
        yaxis_title="mass (kg)", height=380,
    )
    return fig


_CG_COMPONENT_COLORS = {
    "motor_esc": "#d62728", "avionics": "#2ca02c", "servos": "#9467bd",
    "shell": "#7f7f7f", "spar": "#8c564b", "battery": "#ff7f0e",
}


def plot_cg_layout(
    aircraft: Aircraft, cg: CGEstimate, neutral_point_x_m: float, mac_m: float,
    static_margin_target: tuple[float, float],
) -> go.Figure:
    """Top view (x vs y) + front view (y vs z) of the actual aircraft, with
    every component drawn at its real computed position (fixed marker size,
    color-coded by component, mass shown on hover -- scaling marker size by
    mass made the heavier components like shell/spar/battery swamp the
    plot) -- answers "where does everything go, and is it stable" against
    the plane's real planform and fuselage-box geometry, not an abstract 1D
    axis. Servos are drawn mirrored at +-y (there are physically two, one
    per elevon); every other fixed component sits on the centerline."""
    y_full, x_le_full, x_te_full, z_le_full = _planform_outline(aircraft)
    x0, x1, y0, y1, z0, z1 = _fuselage_box_bounds(aircraft)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Top view", "Front view"))

    def _rect(a0, a1, b0, b1):
        return [a0, a1, a1, a0, a0], [b0, b0, b1, b1, b0]

    # Top view: planform outline + fuselage box.
    outline_x = np.concatenate([x_le_full, x_te_full[::-1], [x_le_full[0]]])
    outline_y = np.concatenate([y_full, y_full[::-1], [y_full[0]]])
    fig.add_trace(go.Scatter(
        x=outline_y, y=outline_x, mode="lines", fill="toself",
        fillcolor="rgba(31,119,180,0.08)", line=dict(color="steelblue"), showlegend=False,
    ), row=1, col=1)
    rx, ry = _rect(y0, y1, x0, x1)
    fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines", line=dict(color="orange", dash="dash"), showlegend=False), row=1, col=1)

    # Front view: thickness envelope + fuselage box.
    thickness_full = np.concatenate([aircraft.thickness_ratio[::-1], aircraft.thickness_ratio[1:]]) * np.concatenate(
        [aircraft.chord_m[::-1], aircraft.chord_m[1:]]
    )
    fig.add_trace(go.Scatter(x=y_full, y=z_le_full + thickness_full / 2, mode="lines", line=dict(color="steelblue"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=y_full, y=z_le_full - thickness_full / 2, mode="lines", line=dict(color="steelblue"), fill="tonexty", fillcolor="rgba(31,119,180,0.08)", showlegend=False), row=1, col=2)
    rx, ry = _rect(y0, y1, z0, z1)
    fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines", line=dict(color="orange", dash="dash"), showlegend=False), row=1, col=2)

    def _add_component(name: str, mass_kg: float, x_m: float, y_m: float, z_m: float) -> None:
        ys = [y_m, -y_m] if y_m != 0 else [0.0]
        size = 14  # fixed -- mass-scaled markers made heavy components (shell/spar/battery) swamp the plot
        color = _CG_COMPONENT_COLORS.get(name, "#1f77b4")
        hover = [f"{name}: {mass_kg * 1000:.0f} g @ x={x_m * 1000:.0f} mm, y={yy * 1000:.0f} mm, z={z_m * 1000:.0f} mm" for yy in ys]
        fig.add_trace(go.Scatter(
            x=ys, y=[x_m] * len(ys), mode="markers+text", text=[name] * len(ys), textposition="top center",
            marker=dict(size=size, color=color), showlegend=False, hovertext=hover, hoverinfo="text",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ys, y=[z_m] * len(ys), mode="markers", marker=dict(size=size, color=color),
            showlegend=False, hovertext=hover, hoverinfo="text",
        ), row=1, col=2)

    for c in cg.components:
        _add_component(c.name, c.mass_kg, c.x_m, c.y_m, c.z_m)
    _add_component("battery", cg.battery_mass_kg, cg.battery_x_assumed_m, 0.0, cg.battery_z_assumed_m)

    # CG / neutral point / target band -- all on the centerline, so drawn as
    # chordwise (x) bands/lines on the top view only.
    lo, hi = static_margin_target
    x_cg_forward = neutral_point_x_m - hi * mac_m
    x_cg_aft = neutral_point_x_m - lo * mac_m
    fig.add_hrect(y0=min(x_cg_forward, x_cg_aft), y1=max(x_cg_forward, x_cg_aft), fillcolor="green", opacity=0.12, line_width=0, row=1, col=1, annotation_text="target CG band", annotation_position="top left")
    fig.add_hline(y=neutral_point_x_m, line_color="purple", line_dash="dash", row=1, col=1, annotation_text="NP")
    fig.add_hline(y=cg.cg_x_assumed_m, line_color="black", line_dash="solid", row=1, col=1, annotation_text=f"CG (SM={cg.static_margin_assumed:.3f})")
    if cg.battery_range_feasible:
        fig.add_hrect(y0=cg.battery_x_min_m, y1=cg.battery_x_max_m, fillcolor="orange", opacity=0.12, line_width=0, row=1, col=1, annotation_text="feasible battery x-range", annotation_position="bottom left")

    fig.update_xaxes(title_text="y (m)", row=1, col=1)
    fig.update_yaxes(title_text="x (m)", autorange="reversed", row=1, col=1, scaleanchor="x1")
    fig.update_xaxes(title_text="y (m)", row=1, col=2)
    fig.update_yaxes(title_text="z (m)", row=1, col=2, scaleanchor="x2")

    fig.update_layout(title="CG Layout -- Top & Front Views (mass shown on hover)", height=420, showlegend=False)
    return fig
