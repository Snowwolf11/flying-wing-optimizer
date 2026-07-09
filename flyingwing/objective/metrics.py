"""Evaluate one design end-to-end: DesignParameters -> DesignMetrics.

This is the one function the optimizer (and the GUI's "run full evaluation"
button) calls per candidate. It owns the wiring between geometry, 2D/3D
aero, structures, and mass -- nothing else needs to know how those modules
fit together.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.params import DesignParameters
from ..geometry.aircraft import build_aircraft
from ..geometry.constraints import check_all_constraints, curve_curvature, quick_reject_reason
from ..geometry.airfoil_family import generate_airfoil, MIN_THICKNESS_RATIO
from ..analysis.airfoil_2d import evaluate_section, reynolds_number
from ..analysis.aero_3d import analyze_aerobuildup
from ..analysis.structures import analyze_structures
from .mass import estimate_mass
from .cg import estimate_cg
from ..config import (
    CRUISE_SPEED_MS, TOP_SPEED_MS, DESIGN_LOAD_FACTOR_G, GRAVITY_M_S2,
    MIN_ABSOLUTE_THICKNESS_M, MIN_SPAR_DEPTH_M, MAX_LE_CURVATURE_PER_M, MAX_Z_CURVATURE_PER_M,
    SPAR_DEPTH_FRACTION_THICKNESS,
)


@dataclass
class DesignMetrics:
    valid: bool
    constraint_violations: list[str] = field(default_factory=list)

    # Numeric constraint margins (>= 0 means compliant), for a smooth
    # constraint penalty in the objective function -- see objective.py.
    fuselage_height_margin_m: float = float("nan")
    fuselage_length_margin_m: float = float("nan")
    tip_thickness_margin: float = float("nan")
    thickness_monotonic_violation: float = float("nan")  # 0 if compliant, > 0 = amount thickness ratio increases somewhere

    # Stage 2 (planform) numeric constraint margins, same convention
    chord_monotonic_violation: float = float("nan")
    twist_monotonic_violation: float = float("nan")
    min_local_thickness_margin: float = float("nan")
    min_spar_depth_margin: float = float("nan")
    le_curvature_violation: float = float("nan")  # 0 if compliant, > 0 = amount max curvature exceeds the bound
    z_curvature_violation: float = float("nan")  # 0 if compliant, > 0 = amount max vertical (winglet) curvature exceeds the bound

    # Geometry
    wing_area_m2: float = float("nan")
    aspect_ratio: float = float("nan")
    span_m: float = float("nan")

    # Aerodynamics -- cruise (70-80 km/h band, evaluated at 75 km/h)
    cruise_trim_alpha_deg: float = float("nan")
    cruise_CL: float = float("nan")
    cruise_CD: float = float("nan")
    cruise_L_over_D: float = float("nan")

    # Aerodynamics -- high speed (150 km/h)
    fast_trim_alpha_deg: float = float("nan")
    fast_CL: float = float("nan")
    fast_CD: float = float("nan")
    fast_L_over_D: float = float("nan")

    # Soaring/glide proxies at the cruise trim condition -- cheap (reuse the
    # already-computed cruise trim L/D, no extra alpha sweep), unlike
    # objective/performance.py's more precise best-glide-alpha sweep, which
    # is deliberately kept out of evaluate_design() since it runs thousands
    # of times per optimizer run.
    cruise_glide_angle_deg: float = float("nan")  # atan(1/cruise_L_over_D) -- shallower (smaller) is better
    soaring_power_w: float = float("nan")  # weight * sink rate at cruise trim -- lower means the design stays aloft in weaker lift

    # Stability (cruise condition)
    cruise_Cm: float = float("nan")
    neutral_point_x_m: float = float("nan")
    mean_aerodynamic_chord_m: float = float("nan")
    cg_x_m: float = float("nan")  # assumed CG (battery at the fuselage box centroid) -- see objective/cg.py
    static_margin: float = float("nan")  # (neutral_point_x_m - cg_x_m) / mean_aerodynamic_chord_m -- a real CG-based value, not a placeholder

    # Roll (lateral) stability -- "dihedral effect," the roll moment per unit
    # sideslip. Negative is roll-stable (a gust-induced sideslip rolls the
    # aircraft back toward wings-level); positive is roll-unstable. Cheap --
    # AeroBuildup's run_with_stability_derivatives() already computes this
    # alongside Cma/CLa at the cruise trim condition, just unused until now.
    cruise_Clb_per_rad: float = float("nan")
    battery_x_min_m: float = float("nan")  # battery x-range keeping static_margin in cg.DEFAULT_STATIC_MARGIN_TARGET
    battery_x_max_m: float = float("nan")
    battery_range_feasible: bool = False

    # Root-section 2D characteristics (stall margin, self-trim behavior)
    root_cl_max: float = float("nan")
    root_cm_zero_lift: float = float("nan")

    # Structure (proxy, at DESIGN_LOAD_FACTOR_G)
    min_safety_factor: float = float("nan")
    root_bending_moment_nm: float = float("nan")

    # Mass + payload
    total_structural_mass_kg: float = float("nan")
    payload_volume_margin_m3: float = float("nan")


def evaluate_design(
    params: DesignParameters,
    cruise_speed_ms: float = CRUISE_SPEED_MS,
    fast_speed_ms: float = TOP_SPEED_MS,
    load_factor_g: float = DESIGN_LOAD_FACTOR_G,
) -> DesignMetrics:
    # Cheap pre-check on the raw control points, before paying for
    # build_aircraft() + AeroBuildup + NeuralFoil (the dominant per-
    # candidate cost) -- see quick_reject_reason's docstring for why this
    # is safe (can only reject what the real check below would also
    # reject). A candidate rejected here just gets a mostly-default
    # DesignMetrics; score() treats a NaN-heavy invalid metrics object the
    # same as any other invalid one (see objective/objective.py's
    # NaN-to-(-inf) guard).
    reject_reason = quick_reject_reason(params)
    if reject_reason is not None:
        return DesignMetrics(valid=False, constraint_violations=[reject_reason])

    aircraft = build_aircraft(params)
    constraints = check_all_constraints(aircraft)

    cruise = analyze_aerobuildup(aircraft, cruise_speed_ms, alpha_deg=2.0)
    fast = analyze_aerobuildup(aircraft, fast_speed_ms, alpha_deg=0.5)

    root_airfoil = generate_airfoil(
        aircraft.thickness_scale[0], aircraft.camber_scale[0], aircraft.reflex_scale[0],
    )
    root_re = reynolds_number(aircraft.chord_m[0], cruise_speed_ms)
    root_section = evaluate_section(root_airfoil, Re=root_re)

    structures = analyze_structures(
        aircraft, cruise_trim_cl=cruise.trim_CL, speed_ms=cruise_speed_ms, load_factor_g=load_factor_g,
    )
    mass = estimate_mass(aircraft, structures)
    cg = estimate_cg(aircraft, structures, mass, neutral_point_x_m=cruise.neutral_point_x_m, mac_m=aircraft.mean_aerodynamic_chord_m)

    thickness_diff = np.diff(aircraft.thickness_ratio)
    thickness_monotonic_violation = float(max(np.max(thickness_diff), 0.0))

    chord_diff = np.diff(aircraft.chord_m)
    chord_monotonic_violation = float(max(np.max(chord_diff), 0.0))

    twist_diff = np.diff(aircraft.twist_deg)
    twist_monotonic_violation = float(max(np.max(twist_diff), 0.0))

    local_thickness_m = aircraft.thickness_ratio * aircraft.chord_m
    min_local_thickness_margin = float(np.min(local_thickness_m) - MIN_ABSOLUTE_THICKNESS_M)
    spar_depth_m = SPAR_DEPTH_FRACTION_THICKNESS * local_thickness_m
    min_spar_depth_margin = float(np.min(spar_depth_m) - MIN_SPAR_DEPTH_M)

    le_curvature = curve_curvature(aircraft.x_le_m, aircraft.span_station_m)
    le_curvature_violation = float(max(np.max(le_curvature) - MAX_LE_CURVATURE_PER_M, 0.0))

    z_curvature = curve_curvature(aircraft.z_le_m, aircraft.span_station_m)
    z_curvature_violation = float(max(np.max(z_curvature) - MAX_Z_CURVATURE_PER_M, 0.0))

    if cruise.trim_L_over_D > 0:
        cruise_glide_angle_deg = float(np.degrees(np.arctan2(1.0, cruise.trim_L_over_D)))
        sink_rate_ms = cruise_speed_ms * float(np.sin(np.radians(cruise_glide_angle_deg)))
        weight_n = (mass.total_structural_mass_kg + cg.battery_mass_kg) * GRAVITY_M_S2
        soaring_power_w = weight_n * sink_rate_ms
    else:
        cruise_glide_angle_deg = float("nan")
        soaring_power_w = float("nan")

    return DesignMetrics(
        valid=constraints.valid,
        constraint_violations=constraints.violations,
        fuselage_height_margin_m=aircraft.fuselage_fit.min_height_margin_m,
        fuselage_length_margin_m=aircraft.fuselage_fit.min_length_margin_m,
        tip_thickness_margin=float(aircraft.thickness_ratio[-1] - MIN_THICKNESS_RATIO),
        thickness_monotonic_violation=thickness_monotonic_violation,
        chord_monotonic_violation=chord_monotonic_violation,
        twist_monotonic_violation=twist_monotonic_violation,
        min_local_thickness_margin=min_local_thickness_margin,
        min_spar_depth_margin=min_spar_depth_margin,
        le_curvature_violation=le_curvature_violation,
        z_curvature_violation=z_curvature_violation,
        wing_area_m2=aircraft.wing_area_m2,
        aspect_ratio=aircraft.aspect_ratio,
        span_m=aircraft.params.planform.span_m,
        cruise_trim_alpha_deg=cruise.trim_alpha_deg,
        cruise_CL=cruise.trim_CL,
        cruise_CD=cruise.trim_CD,
        cruise_L_over_D=cruise.trim_L_over_D,
        fast_trim_alpha_deg=fast.trim_alpha_deg,
        fast_CL=fast.trim_CL,
        fast_CD=fast.trim_CD,
        fast_L_over_D=fast.trim_L_over_D,
        cruise_glide_angle_deg=cruise_glide_angle_deg,
        soaring_power_w=soaring_power_w,
        cruise_Cm=cruise.Cm,
        cruise_Clb_per_rad=cruise.Clb_per_rad,
        neutral_point_x_m=cruise.neutral_point_x_m,
        mean_aerodynamic_chord_m=aircraft.mean_aerodynamic_chord_m,
        cg_x_m=cg.cg_x_assumed_m,
        static_margin=cg.static_margin_assumed,
        battery_x_min_m=cg.battery_x_min_m,
        battery_x_max_m=cg.battery_x_max_m,
        battery_range_feasible=cg.battery_range_feasible,
        root_cl_max=root_section.cl_max,
        root_cm_zero_lift=root_section.cm_zero_lift,
        min_safety_factor=structures.min_safety_factor,
        root_bending_moment_nm=structures.root_bending_moment_nm,
        total_structural_mass_kg=mass.total_structural_mass_kg,
        payload_volume_margin_m3=mass.payload_volume_margin_m3,
    )
