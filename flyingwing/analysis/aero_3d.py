"""3D aircraft aerodynamic analysis via AeroSandbox.

AeroBuildup (strip theory + NeuralFoil per station) is the primary, fast
method -- fast enough to call many times per optimizer iteration, using the
same (moderately fine, linearly-spaced) `Aircraft.airplane` the geometry
generator already built.

VLM (vortex lattice) is a slower cross-check. It builds its OWN, much
coarser Airplane rather than reusing `Aircraft.airplane`: VLM refines the
spanwise/chordwise resolution itself between whatever cross-sections it's
given, so handing it the same ~40-station Airplane used for AeroBuildup
packs many near-duplicate, near-zero-width panels close to the root/tip.
That was measured to make the AIC matrix ill-conditioned enough that the
solution blew up to nonsense (~1e22-magnitude coefficients) and took minutes;
a dedicated ~13-station Airplane with modest spanwise/chordwise resolution
converges in ~5 s and agrees closely with AeroBuildup.

AVL (a real external solver, like XFoil) is wired as an optional cross-check
only, following the same "optional, don't require the binary" approach.

Everything here takes a fully-built `Aircraft` (from `geometry.aircraft`) and
an operating condition, and returns plain metrics -- no knowledge of the
optimizer or objective function.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import aerosandbox as asb

from ..geometry.aircraft import Aircraft, build_airplane

N_VLM_STATIONS = 13
VLM_SPANWISE_RESOLUTION = 5
VLM_CHORDWISE_RESOLUTION = 6


@dataclass
class AircraftAeroMetrics:
    speed_ms: float
    alpha_deg: float

    CL: float
    CD: float
    CD_profile: float
    CD_induced: float
    Cm: float
    L_over_D: float
    span_efficiency: float

    CLa_per_rad: float
    Cma_per_rad: float
    neutral_point_x_m: float
    static_margin_vs_xyz_ref: float  # relative to the placeholder 25%-MAC xyz_ref; refine once mass is estimated

    trim_alpha_deg: float
    trim_CL: float
    trim_CD: float
    trim_L_over_D: float


def _scalar(x) -> float:
    """AeroBuildup wraps results in 1-element arrays; VLM returns bare
    scalars for a single alpha. Handle both."""
    return float(np.ravel(x)[0])


def _trim_alpha_deg(alpha_deg: float, Cm: float, Cma_per_rad: float) -> float:
    """Single Newton step toward Cm=0, using the stability derivative from
    the same run -- good enough given the near-linear regime around cruise
    alpha, and avoids a 3rd analysis call."""
    delta_alpha_rad = -Cm / Cma_per_rad if Cma_per_rad != 0 else 0.0
    return alpha_deg + np.degrees(delta_alpha_rad)


def analyze_aerobuildup(aircraft: Aircraft, speed_ms: float, alpha_deg: float = 2.0) -> AircraftAeroMetrics:
    """Primary 3D analysis: fast strip-theory buildup (NeuralFoil per
    station) + stability derivatives, plus a linearized trim solve
    re-evaluated at the trimmed alpha."""
    airplane = aircraft.airplane
    xyz_ref_x = float(airplane.xyz_ref[0])
    mac_m = aircraft.mean_aerodynamic_chord_m

    op = asb.OperatingPoint(velocity=speed_ms, alpha=alpha_deg)
    res = asb.AeroBuildup(airplane=airplane, op_point=op).run_with_stability_derivatives()

    CL = _scalar(res["CL"])
    CD = _scalar(res["CD"])
    Cm = _scalar(res["Cm"])
    D = _scalar(res["D"])
    D_profile = _scalar(res["D_profile"])
    D_induced = _scalar(res["D_induced"])
    CD_profile = CD * (D_profile / D) if D != 0 else CD
    CD_induced = CD * (D_induced / D) if D != 0 else 0.0
    span_efficiency = float(res["wing_aero_components"][0].oswalds_efficiency)

    CLa = _scalar(res["CLa"])
    Cma = _scalar(res["Cma"])
    x_np = _scalar(res["x_np"])
    static_margin = (x_np - xyz_ref_x) / mac_m

    trim_alpha_deg = _trim_alpha_deg(alpha_deg, Cm, Cma)
    op_trim = asb.OperatingPoint(velocity=speed_ms, alpha=trim_alpha_deg)
    res_trim = asb.AeroBuildup(airplane=airplane, op_point=op_trim).run()
    trim_CL = _scalar(res_trim["CL"])
    trim_CD = _scalar(res_trim["CD"])

    return AircraftAeroMetrics(
        speed_ms=speed_ms, alpha_deg=alpha_deg,
        CL=CL, CD=CD, CD_profile=CD_profile, CD_induced=CD_induced, Cm=Cm,
        L_over_D=CL / CD if CD != 0 else float("nan"),
        span_efficiency=span_efficiency,
        CLa_per_rad=CLa, Cma_per_rad=Cma,
        neutral_point_x_m=x_np, static_margin_vs_xyz_ref=static_margin,
        trim_alpha_deg=trim_alpha_deg, trim_CL=trim_CL, trim_CD=trim_CD,
        trim_L_over_D=trim_CL / trim_CD if trim_CD != 0 else float("nan"),
    )


def analyze_vlm(
    aircraft: Aircraft,
    speed_ms: float,
    alpha_deg: float = 2.0,
    n_vlm_stations: int = N_VLM_STATIONS,
    spanwise_resolution: int = VLM_SPANWISE_RESOLUTION,
    chordwise_resolution: int = VLM_CHORDWISE_RESOLUTION,
) -> tuple[AircraftAeroMetrics, dict]:
    """Cross-check / higher-fidelity analysis using the vortex lattice
    method, on a dedicated coarse Airplane (see module docstring for why).
    VLM is inviscid, so all drag is induced drag here; span efficiency is
    computed directly from CL^2 / (pi * AR * CD) rather than taken from an
    AeroBuildup-style component breakdown. Also returns the raw VLM result
    dict (panel-by-panel loading) for visualization.
    """
    airplane = build_airplane(aircraft.params, n_analysis_stations=n_vlm_stations)
    xyz_ref_x = float(airplane.xyz_ref[0])
    mac_m = aircraft.mean_aerodynamic_chord_m
    aspect_ratio = aircraft.aspect_ratio

    op = asb.OperatingPoint(velocity=speed_ms, alpha=alpha_deg)
    vlm = asb.VortexLatticeMethod(
        airplane=airplane, op_point=op,
        spanwise_resolution=spanwise_resolution, chordwise_resolution=chordwise_resolution,
    )
    res = vlm.run_with_stability_derivatives()

    CL = _scalar(res["CL"])
    CD = _scalar(res["CD"])
    Cm = _scalar(res["Cm"])

    CLa = _scalar(res["CLa"])
    Cma = _scalar(res["Cma"])
    x_np = _scalar(res["x_np"])
    static_margin = (x_np - xyz_ref_x) / mac_m

    trim_alpha_deg = _trim_alpha_deg(alpha_deg, Cm, Cma)
    op_trim = asb.OperatingPoint(velocity=speed_ms, alpha=trim_alpha_deg)
    res_trim = asb.VortexLatticeMethod(
        airplane=airplane, op_point=op_trim,
        spanwise_resolution=spanwise_resolution, chordwise_resolution=chordwise_resolution,
    ).run()
    trim_CL = _scalar(res_trim["CL"])
    trim_CD = _scalar(res_trim["CD"])

    # CL^2/(pi*AR*CDi) is numerically unstable right at alpha_deg if that
    # happens to be a near-zero-lift condition (a washed-out flying wing's
    # non-elliptic loading gives it a near-constant induced-drag offset that
    # dominates the CL^2 term there) -- evaluate at the trimmed condition
    # instead, where CL is representative of actual flight.
    span_efficiency = trim_CL ** 2 / (np.pi * aspect_ratio * trim_CD) if trim_CD != 0 else float("nan")

    metrics = AircraftAeroMetrics(
        speed_ms=speed_ms, alpha_deg=alpha_deg,
        CL=CL, CD=CD, CD_profile=0.0, CD_induced=CD, Cm=Cm,
        L_over_D=CL / CD if CD != 0 else float("nan"),
        span_efficiency=span_efficiency,
        CLa_per_rad=CLa, Cma_per_rad=Cma,
        neutral_point_x_m=x_np, static_margin_vs_xyz_ref=static_margin,
        trim_alpha_deg=trim_alpha_deg, trim_CL=trim_CL, trim_CD=trim_CD,
        trim_L_over_D=trim_CL / trim_CD if trim_CD != 0 else float("nan"),
    )
    return metrics, res


def validate_with_avl(aircraft: Aircraft, speed_ms: float, alpha_deg: float = 2.0, avl_command: str = "avl") -> dict | None:
    """Optional AVL cross-check. Returns None (with a warning) if the AVL
    executable isn't available -- AeroBuildup/VLM above are the required
    analysis path; AVL is validation-only, same as XFoil for 2D sections.
    """
    try:
        airplane = build_airplane(aircraft.params, n_analysis_stations=N_VLM_STATIONS)
        op = asb.OperatingPoint(velocity=speed_ms, alpha=alpha_deg)
        avl = asb.AVL(airplane=airplane, op_point=op, avl_command=avl_command)
        return avl.run()
    except Exception as e:  # pragma: no cover -- depends on local AVL install
        warnings.warn(f"AVL validation skipped ({type(e).__name__}: {e})")
        return None
