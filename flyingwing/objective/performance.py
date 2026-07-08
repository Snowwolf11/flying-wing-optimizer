"""Interpretable, human-facing performance estimates: best glide ratio/angle
and a rough flight-time (endurance) estimate from a battery assumption.

Deliberately NOT called from evaluate_design()/DesignMetrics -- finding the
best-L/D alpha needs its own small alpha sweep (several extra AeroBuildup
calls), and evaluate_design() is called many thousands of times per
optimizer run (see the project's fast-proxies-in-the-search-loop design).
This module is for inspecting a finished design only: call it once from the
GUI or a CLI script's post-run summary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _overrides as _ov
from ..geometry.aircraft import Aircraft
from ..analysis.aero_3d import analyze_aerobuildup
from ..config import BOUNDS_OVERRIDES_YAML
from .cg import BATTERY_MASS_KG

GRAVITY_M_S2 = 9.80665

BATTERY_CAPACITY_MAH = 1300.0
BATTERY_VOLTAGE_V = 14.8  # nominal 4S LiPo
BATTERY_USABLE_FRACTION = 0.8  # avoid full discharge
PROPULSIVE_EFFICIENCY = 0.65  # combined motor + ESC + prop efficiency

_ov.apply_overrides(globals(), BOUNDS_OVERRIDES_YAML, {
    "BATTERY_CAPACITY_MAH", "BATTERY_VOLTAGE_V", "BATTERY_USABLE_FRACTION", "PROPULSIVE_EFFICIENCY",
})

_GLIDE_ALPHA_SWEEP_DEG = np.arange(-2.0, 10.01, 2.0)


@dataclass
class PerformanceEstimate:
    glide_ratio_max: float
    glide_alpha_deg: float
    glide_angle_deg: float
    sink_rate_ms: float  # at cruise speed, at the best-glide condition

    cruise_power_w: float
    battery_energy_wh: float
    estimated_endurance_min: float
    estimated_range_km: float


def _best_glide(aircraft: Aircraft, speed_ms: float, alpha_range_deg: np.ndarray = _GLIDE_ALPHA_SWEEP_DEG) -> tuple[float, float]:
    """Best (max) L/D and the alpha it occurs at, from a small alpha sweep
    at a fixed speed -- L/D vs. alpha is ~speed-independent to first order
    (same CL/CD regardless of speed, ignoring Reynolds-number drift), so
    cruise speed is a fine speed to sweep at."""
    best_ld, best_alpha = -np.inf, float("nan")
    for alpha in alpha_range_deg:
        m = analyze_aerobuildup(aircraft, speed_ms, alpha_deg=float(alpha))
        if m.L_over_D > best_ld:
            best_ld, best_alpha = m.L_over_D, float(alpha)
    return best_ld, best_alpha


def estimate_performance(
    aircraft: Aircraft, structural_mass_kg: float, cruise_speed_ms: float,
    battery_mass_kg: float = BATTERY_MASS_KG,
    battery_capacity_mah: float = BATTERY_CAPACITY_MAH, battery_voltage_v: float = BATTERY_VOLTAGE_V,
    battery_usable_fraction: float = BATTERY_USABLE_FRACTION, propulsive_efficiency: float = PROPULSIVE_EFFICIENCY,
) -> PerformanceEstimate:
    """`structural_mass_kg` is DesignMetrics.total_structural_mass_kg, which
    excludes the battery (see objective/mass.py) -- battery_mass_kg (should
    match objective/cg.py's assumption) is added here to get the actual
    flying weight for the power/endurance estimate."""
    glide_ratio_max, glide_alpha_deg = _best_glide(aircraft, cruise_speed_ms)
    glide_angle_deg = float(np.degrees(np.arctan2(1.0, glide_ratio_max))) if glide_ratio_max > 0 else float("nan")
    sink_rate_ms = cruise_speed_ms * np.sin(np.radians(glide_angle_deg)) if glide_ratio_max > 0 else float("nan")

    weight_n = (structural_mass_kg + battery_mass_kg) * GRAVITY_M_S2
    cruise_power_w = weight_n * cruise_speed_ms / (glide_ratio_max * propulsive_efficiency) if glide_ratio_max > 0 else float("nan")

    battery_energy_wh = (battery_capacity_mah / 1000.0) * battery_voltage_v * battery_usable_fraction
    estimated_endurance_min = 60.0 * battery_energy_wh / cruise_power_w if cruise_power_w and cruise_power_w > 0 else float("nan")
    estimated_range_km = (
        cruise_speed_ms * (estimated_endurance_min * 60.0) / 1000.0 if not np.isnan(estimated_endurance_min) else float("nan")
    )

    return PerformanceEstimate(
        glide_ratio_max=glide_ratio_max, glide_alpha_deg=glide_alpha_deg, glide_angle_deg=glide_angle_deg,
        sink_rate_ms=sink_rate_ms, cruise_power_w=cruise_power_w,
        battery_energy_wh=battery_energy_wh, estimated_endurance_min=estimated_endurance_min,
        estimated_range_km=estimated_range_km,
    )
