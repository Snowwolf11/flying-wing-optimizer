"""Deep-analysis plots: what's driving a design's score, where its mass and
CG actually come from. All of this data is already computed elsewhere
(`score().contributions`, `objective/mass.py`, `objective/cg.py`) but was
never visualized before -- these functions just make it visible.
"""
from __future__ import annotations

import plotly.graph_objects as go

from ..objective.objective import ObjectiveResult
from ..objective.mass import MassEstimate
from ..objective.cg import CGEstimate


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
    """Structural mass breakdown (shell/spar/fixed equipment) plus the
    battery -- shown separately since it's excluded from
    total_structural_mass_kg (see objective/mass.py)."""
    labels = ["shell", "spar", "fixed equipment", "battery"]
    values = [mass.shell_mass_kg, mass.spar_mass_kg, mass.fixed_equipment_mass_kg, battery_mass_kg]

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#1f77b4"))
    total_flying_mass = mass.total_structural_mass_kg + battery_mass_kg
    fig.update_layout(
        title=f"Mass Breakdown -- {mass.total_structural_mass_kg:.3f} kg structural + {battery_mass_kg:.3f} kg battery = {total_flying_mass:.3f} kg flying",
        yaxis_title="mass (kg)", height=380,
    )
    return fig


def plot_cg_diagram(
    cg: CGEstimate, neutral_point_x_m: float, mac_m: float,
    static_margin_target: tuple[float, float],
) -> go.Figure:
    """A 1D longitudinal diagram: every fixed component's x-position (marker
    size ~ mass), the battery's assumed position, the neutral point, the
    resulting CG, and the target CG band (from static_margin_target) --
    answers "where does everything go, and is it stable" at a glance."""
    fig = go.Figure()

    xs_mm = [c.x_m * 1000 for c in cg.components]
    masses_g = [c.mass_kg * 1000 for c in cg.components]
    names = [c.name for c in cg.components]
    fig.add_trace(go.Scatter(
        x=xs_mm, y=[0] * len(xs_mm), mode="markers+text", text=names, textposition="top center",
        marker=dict(size=[max(10, m / 3) for m in masses_g], color="#1f77b4"), name="fixed components",
        hovertext=[f"{n}: {m:.0f} g @ {x:.0f} mm" for n, m, x in zip(names, masses_g, xs_mm)], hoverinfo="text",
    ))
    fig.add_trace(go.Scatter(
        x=[cg.battery_x_assumed_m * 1000], y=[0], mode="markers+text", text=["battery (assumed)"], textposition="bottom center",
        marker=dict(size=max(10, cg.battery_mass_kg * 1000 / 3), color="#ff7f0e", symbol="square"), name="battery (assumed position)",
    ))

    lo, hi = static_margin_target
    x_cg_forward = (neutral_point_x_m - hi * mac_m) * 1000
    x_cg_aft = (neutral_point_x_m - lo * mac_m) * 1000
    fig.add_vrect(x0=min(x_cg_forward, x_cg_aft), x1=max(x_cg_forward, x_cg_aft), fillcolor="green", opacity=0.15, line_width=0, annotation_text="target CG band", annotation_position="top left")

    fig.add_vline(x=neutral_point_x_m * 1000, line_color="purple", line_dash="dash", annotation_text="neutral point")
    fig.add_vline(x=cg.cg_x_assumed_m * 1000, line_color="black", line_dash="solid", annotation_text=f"CG (SM={cg.static_margin_assumed:.3f})")

    if cg.battery_range_feasible:
        fig.add_vrect(x0=cg.battery_x_min_m * 1000, x1=cg.battery_x_max_m * 1000, fillcolor="orange", opacity=0.15, line_width=0, annotation_text="feasible battery x-range", annotation_position="bottom left")

    fig.update_layout(
        title="Longitudinal CG Diagram (marker size ~ mass)",
        xaxis_title="x (mm from root LE)", yaxis=dict(visible=False, range=[-1, 1]),
        height=320, showlegend=False,
    )
    return fig
