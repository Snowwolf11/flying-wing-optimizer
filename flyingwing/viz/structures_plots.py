"""Structural proxy visualizations: bending moment, shear force, spar depth,
bending stress, and safety factor along the span."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..analysis.structures import StructuralProxyResult, TorsionDeflectionResult


def plot_structures(result: StructuralProxyResult) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Bending moment", "Shear force", "Spar depth (available)",
            "Bending stress", "Safety factor", "Spar width (available)",
        ),
    )

    fig.add_trace(go.Scatter(x=result.y_stations, y=result.bending_moment_nm, mode="lines"), row=1, col=1)
    fig.update_yaxes(title_text="bending moment (N*m)", row=1, col=1)

    fig.add_trace(go.Scatter(x=result.y_stations, y=result.shear_n, mode="lines"), row=1, col=2)
    fig.update_yaxes(title_text="shear (N)", row=1, col=2)

    fig.add_trace(go.Scatter(x=result.y_stations, y=result.spar_depth_available_m * 1000, mode="lines"), row=1, col=3)
    fig.update_yaxes(title_text="spar depth (mm)", row=1, col=3)

    fig.add_trace(go.Scatter(x=result.y_stations, y=result.bending_stress_pa / 1e6, mode="lines"), row=2, col=1)
    fig.update_yaxes(title_text="bending stress (MPa)", row=2, col=1)

    fig.add_trace(go.Scatter(x=result.y_stations, y=result.safety_factor, mode="lines"), row=2, col=2)
    fig.update_yaxes(title_text="safety factor", type="log", row=2, col=2)

    fig.add_trace(go.Scatter(x=result.y_stations, y=result.spar_width_m * 1000, mode="lines"), row=2, col=3)
    fig.update_yaxes(title_text="spar width (mm)", row=2, col=3)

    for col in (1, 2, 3):
        fig.update_xaxes(title_text="y (normalized span)", row=2, col=col)

    fig.update_layout(
        title=f"Structural Proxy -- {result.load_factor_g:.1f}g maneuver (CL={result.cl_maneuver:.3f}), "
              f"min safety factor={result.min_safety_factor:.1f}",
        showlegend=False, height=650,
    )
    return fig


def plot_torsion_and_deflection(result: TorsionDeflectionResult) -> go.Figure:
    """Deep-analysis-only extras on top of plot_structures: torque, thin-
    walled torsional shear stress + safety factor, and bending deflection."""
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=("Torque", "Torsional shear stress", "Torsion safety factor", "Bending deflection"),
    )

    fig.add_trace(go.Scatter(x=result.y_stations, y=result.torque_nm, mode="lines"), row=1, col=1)
    fig.update_yaxes(title_text="torque (N*m)", row=1, col=1)

    fig.add_trace(go.Scatter(x=result.y_stations, y=result.shear_stress_pa / 1e6, mode="lines"), row=1, col=2)
    fig.update_yaxes(title_text="shear stress (MPa)", row=1, col=2)

    fig.add_trace(go.Scatter(x=result.y_stations, y=result.torsion_safety_factor, mode="lines"), row=1, col=3)
    fig.update_yaxes(title_text="safety factor", type="log", row=1, col=3)

    fig.add_trace(go.Scatter(x=result.y_stations, y=result.deflection_m * 1000, mode="lines"), row=1, col=4)
    fig.update_yaxes(title_text="deflection (mm)", row=1, col=4)

    for col in (1, 2, 3, 4):
        fig.update_xaxes(title_text="y (normalized span)", row=1, col=col)

    fig.update_layout(
        title=f"Torsion & Deflection -- min torsion safety factor={result.min_torsion_safety_factor:.1f}, "
              f"tip deflection={result.tip_deflection_m * 1000:.1f} mm",
        showlegend=False, height=350,
    )
    return fig


def save_all(result: StructuralProxyResult, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "structures.html"
    plot_structures(result).write_html(str(path), include_plotlyjs="cdn")
    return path
