"""2D airfoil analysis: NeuralFoil (primary) and XFoil (optional validation).

Everything here operates on a single AeroSandbox `Airfoil` at a single
(Re, mach, n_crit) operating point -- this module knows nothing about the
3D aircraft, the span, or the optimizer. `evaluate_section` is the one
entry point most callers need; it runs a full alpha sweep and reduces it to
the representative scalar quantities the project spec asks for (CL/CD,
CLmax, drag bucket, pitching moment, lift curve slope, stall behavior).
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import aerosandbox as asb

from ..config import KINEMATIC_VISCOSITY_M2_S, CRUISE_SPEED_MS, TOP_SPEED_MS

# "Several transition assumptions": NeuralFoil's n_crit is the e^n
# amplification-factor criterion XFoil also uses -- lower means the
# analysis assumes an earlier (more turbulent/rougher) transition.
N_CRIT_ASSUMPTIONS = {"clean": 9.0, "moderate": 7.0, "rough": 4.0}

DEFAULT_ALPHA_SWEEP_DEG = np.arange(-6.0, 16.01, 0.5)
_LINEAR_FIT_ALPHA_RANGE_DEG = (-4.0, 6.0)


@dataclass
class PolarSweep:
    alpha_deg: np.ndarray
    CL: np.ndarray
    CD: np.ndarray
    CM: np.ndarray
    top_xtr: np.ndarray
    bot_xtr: np.ndarray
    confidence: np.ndarray


@dataclass
class AirfoilPolarMetrics:
    Re: float
    mach: float
    n_crit: float

    cl_max: float
    alpha_cl_max_deg: float
    lift_curve_slope_per_rad: float
    zero_lift_alpha_deg: float
    cm_zero_lift: float
    cd_min: float
    cl_at_cd_min: float
    drag_bucket_cl_range: tuple[float, float]
    stall_sharpness_per_rad: float  # dCL/dalpha just past CLmax; very negative = sharp/abrupt stall
    mean_confidence: float

    polar: PolarSweep


def reynolds_number(chord_m: float, speed_ms: float, kinematic_viscosity: float = KINEMATIC_VISCOSITY_M2_S) -> float:
    return speed_ms * chord_m / kinematic_viscosity


def _neuralfoil_sweep(
    airfoil: asb.Airfoil,
    alpha_deg: np.ndarray,
    Re: float,
    mach: float,
    n_crit: float,
    model_size: str,
) -> PolarSweep:
    result = airfoil.get_aero_from_neuralfoil(
        alpha=alpha_deg, Re=Re, mach=mach, n_crit=n_crit, model_size=model_size,
    )
    return PolarSweep(
        alpha_deg=np.asarray(alpha_deg, dtype=float),
        CL=np.asarray(result["CL"]),
        CD=np.asarray(result["CD"]),
        CM=np.asarray(result["CM"]),
        top_xtr=np.asarray(result["Top_Xtr"]),
        bot_xtr=np.asarray(result["Bot_Xtr"]),
        confidence=np.asarray(result["analysis_confidence"]),
    )


def _reduce_polar(polar: PolarSweep, Re: float, mach: float, n_crit: float) -> AirfoilPolarMetrics:
    alpha, CL, CD, CM = polar.alpha_deg, polar.CL, polar.CD, polar.CM

    lo, hi = _LINEAR_FIT_ALPHA_RANGE_DEG
    linear_mask = (alpha >= lo) & (alpha <= hi)
    if np.count_nonzero(linear_mask) < 2:
        linear_mask = np.ones_like(alpha, dtype=bool)
    slope_per_deg, intercept = np.polyfit(alpha[linear_mask], CL[linear_mask], 1)
    lift_curve_slope_per_rad = slope_per_deg * 180.0 / np.pi
    zero_lift_alpha_deg = -intercept / slope_per_deg

    cm_zero_lift = float(np.interp(zero_lift_alpha_deg, alpha, CM))

    cl_max_idx = int(np.argmax(CL))
    cl_max = float(CL[cl_max_idx])
    alpha_cl_max_deg = float(alpha[cl_max_idx])

    if cl_max_idx + 1 < len(alpha):
        d_alpha_rad = np.radians(alpha[cl_max_idx + 1] - alpha[cl_max_idx])
        stall_sharpness_per_rad = float((CL[cl_max_idx + 1] - CL[cl_max_idx]) / d_alpha_rad)
    else:
        stall_sharpness_per_rad = float("nan")

    cd_min_idx = int(np.argmin(CD))
    cd_min = float(CD[cd_min_idx])
    cl_at_cd_min = float(CL[cd_min_idx])

    bucket_mask = CD <= 1.1 * cd_min
    drag_bucket_cl_range = (float(np.min(CL[bucket_mask])), float(np.max(CL[bucket_mask])))

    return AirfoilPolarMetrics(
        Re=Re, mach=mach, n_crit=n_crit,
        cl_max=cl_max, alpha_cl_max_deg=alpha_cl_max_deg,
        lift_curve_slope_per_rad=lift_curve_slope_per_rad,
        zero_lift_alpha_deg=float(zero_lift_alpha_deg),
        cm_zero_lift=cm_zero_lift,
        cd_min=cd_min, cl_at_cd_min=cl_at_cd_min,
        drag_bucket_cl_range=drag_bucket_cl_range,
        stall_sharpness_per_rad=stall_sharpness_per_rad,
        mean_confidence=float(np.mean(polar.confidence)),
        polar=polar,
    )


def evaluate_section(
    airfoil: asb.Airfoil,
    Re: float,
    mach: float = 0.0,
    n_crit: float = 9.0,
    alpha_deg: np.ndarray = DEFAULT_ALPHA_SWEEP_DEG,
    model_size: str = "large",
) -> AirfoilPolarMetrics:
    """Run a NeuralFoil alpha sweep and reduce it to scalar section metrics."""
    polar = _neuralfoil_sweep(airfoil, alpha_deg, Re, mach, n_crit, model_size)
    return _reduce_polar(polar, Re, mach, n_crit)


def validate_with_xfoil(
    airfoil: asb.Airfoil,
    alpha_deg: np.ndarray,
    Re: float,
    mach: float = 0.0,
    n_crit: float = 9.0,
    xfoil_command: str = "xfoil",
) -> PolarSweep | None:
    """Optional XFoil cross-check. Returns None (with a warning) if the
    XFoil executable isn't available or the run fails -- NeuralFoil is the
    primary, required analysis path; XFoil is validation-only and the
    framework must keep working without it installed.
    """
    try:
        xf = asb.XFoil(airfoil=airfoil, Re=Re, mach=mach, n_crit=n_crit, xfoil_command=xfoil_command)
        result = xf.alpha(alpha_deg)
    except Exception as e:  # pragma: no cover -- depends on local XFoil install
        warnings.warn(f"XFoil validation skipped ({type(e).__name__}: {e})")
        return None

    return PolarSweep(
        alpha_deg=np.asarray(result["alpha"], dtype=float),
        CL=np.asarray(result["CL"]),
        CD=np.asarray(result["CD"]),
        CM=np.asarray(result["CM"]),
        top_xtr=np.full_like(np.asarray(result["CL"]), np.nan),
        bot_xtr=np.full_like(np.asarray(result["CL"]), np.nan),
        confidence=np.ones_like(np.asarray(result["CL"])),
    )
