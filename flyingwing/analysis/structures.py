"""Structural proxy: a fast, ranking-only estimate of structural demand.

A full FEA model is intentionally avoided (see project notes). Instead:

1. The spanwise lift distribution is estimated with Schrenk's approximation
   -- the average of the actual chord distribution and the elliptical
   distribution with the same span and area. This is a standard, fast,
   closed-form stand-in for a full lifting-line solve.
2. Shear force and bending moment follow from integrating that load
   outboard-to-inboard.
3. A simple thin-walled rectangular spar box, sized as fractions of the
   local chord and thickness envelope, converts bending moment to a stress
   proxy and a safety factor against a generic allowable stress.

None of this is certification-grade; it only needs to rank competing
designs consistently, which a closed-form Schrenk-based proxy does well
without paying for a numerical lifting-line or full FEA solve on every
candidate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ..geometry.aircraft import Aircraft
from ..config import (
    AIR_DENSITY_KG_M3,
    SPAR_WIDTH_FRACTION_CHORD,
    SPAR_DEPTH_FRACTION_THICKNESS,
    SPAR_WALL_THICKNESS_M,
    ALLOWABLE_SPAR_STRESS_PA,
    DESIGN_LOAD_FACTOR_G,
    SPAR_X_FRACTION_CHORD,
    AERODYNAMIC_CENTER_X_FRACTION_CHORD,
    SPAR_YOUNG_MODULUS_PA,
    ALLOWABLE_SPAR_SHEAR_STRESS_PA,
)


@dataclass
class StructuralProxyResult:
    y_stations: np.ndarray
    span_station_m: np.ndarray

    lift_per_span_n_per_m: np.ndarray
    shear_n: np.ndarray
    bending_moment_nm: np.ndarray

    spar_depth_available_m: np.ndarray
    spar_width_m: np.ndarray
    spar_material_area_m2: np.ndarray  # cross-sectional area of the thin-walled box's material itself (not its interior)
    section_modulus_m3: np.ndarray
    moment_of_inertia_m4: np.ndarray  # second moment of area of the thin-walled box, about the bending axis
    bending_stress_pa: np.ndarray
    safety_factor: np.ndarray

    load_factor_g: float
    cl_maneuver: float

    @property
    def root_bending_moment_nm(self) -> float:
        return float(self.bending_moment_nm[0])

    @property
    def min_safety_factor(self) -> float:
        return float(np.nanmin(self.safety_factor))


def _reverse_cumulative_trapezoid(f: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Integral of f from x[i] to x[-1], for every i -- i.e. running
    "everything outboard of this station" integral, as needed for shear
    (from lift) and bending moment (from shear)."""
    reversed_integral = cumulative_trapezoid(f[::-1], x[::-1], initial=0.0)
    return -reversed_integral[::-1]


def schrenk_lift_per_span(
    span_station_m: np.ndarray, chord_m: np.ndarray, span_m: float, wing_area_m2: float, cl: float, q: float,
) -> np.ndarray:
    """Schrenk's approximation: average of the actual chord distribution and
    the elliptical distribution with the same span and planform area."""
    half_span = span_m / 2.0
    chord_ellipse = (4.0 * wing_area_m2 / (np.pi * span_m)) * np.sqrt(
        np.clip(1.0 - (span_station_m / half_span) ** 2, 0.0, None)
    )
    chord_schrenk = 0.5 * (chord_m + chord_ellipse)
    return q * cl * chord_schrenk


def analyze_structures(
    aircraft: Aircraft,
    cruise_trim_cl: float,
    speed_ms: float,
    load_factor_g: float = DESIGN_LOAD_FACTOR_G,
    air_density_kg_m3: float = AIR_DENSITY_KG_M3,
) -> StructuralProxyResult:
    """Structural proxy at a maneuvering condition: same speed as the given
    trim point, but loaded up to `load_factor_g` by scaling CL (CL scales
    ~linearly with load factor at fixed speed: L = n*W = q*S*CL).
    """
    cl_maneuver = load_factor_g * cruise_trim_cl
    q = 0.5 * air_density_kg_m3 * speed_ms ** 2

    y = aircraft.y_stations
    span_station_m = aircraft.span_station_m
    chord_m = aircraft.chord_m

    lift_per_span = schrenk_lift_per_span(
        span_station_m, chord_m, aircraft.params.planform.span_m, aircraft.wing_area_m2, cl_maneuver, q,
    )

    shear_n = _reverse_cumulative_trapezoid(lift_per_span, span_station_m)
    bending_moment_nm = _reverse_cumulative_trapezoid(shear_n, span_station_m)

    spar_depth = SPAR_DEPTH_FRACTION_THICKNESS * aircraft.thickness_ratio * chord_m
    spar_width = SPAR_WIDTH_FRACTION_CHORD * chord_m

    t = SPAR_WALL_THICKNESS_M
    inner_depth = np.clip(spar_depth - 2 * t, 0.0, None)
    inner_width = np.clip(spar_width - 2 * t, 0.0, None)
    spar_material_area = spar_width * spar_depth - inner_width * inner_depth
    moment_of_inertia = (spar_width * spar_depth ** 3 - inner_width * inner_depth ** 3) / 12.0
    section_modulus = np.divide(
        moment_of_inertia, spar_depth / 2.0,
        out=np.zeros_like(moment_of_inertia), where=spar_depth > 1e-9,
    )

    section_modulus_safe = np.maximum(section_modulus, 1e-12)
    bending_stress = np.abs(bending_moment_nm) / section_modulus_safe
    safety_factor = np.full_like(bending_stress, np.nan)
    stressed = bending_stress > 1e-6
    safety_factor[stressed] = ALLOWABLE_SPAR_STRESS_PA / bending_stress[stressed]

    return StructuralProxyResult(
        y_stations=y, span_station_m=span_station_m,
        lift_per_span_n_per_m=lift_per_span, shear_n=shear_n, bending_moment_nm=bending_moment_nm,
        spar_depth_available_m=spar_depth, spar_width_m=spar_width, spar_material_area_m2=spar_material_area,
        section_modulus_m3=section_modulus, moment_of_inertia_m4=moment_of_inertia,
        bending_stress_pa=bending_stress, safety_factor=safety_factor,
        load_factor_g=load_factor_g, cl_maneuver=cl_maneuver,
    )


@dataclass
class TorsionDeflectionResult:
    """Deep-analysis-only extras (see analysis/structures.py module
    docstring for what's NOT modeled): distributed torque about the assumed
    spar/elastic axis, thin-walled (Bredt-Batho) torsional shear stress, and
    Euler-Bernoulli bending deflection. Final-design-only -- not evaluated
    per optimizer candidate."""

    y_stations: np.ndarray
    span_station_m: np.ndarray

    torque_nm: np.ndarray  # distributed torque, from the AC-to-spar lift moment arm (dominant driver on a swept wing)
    shear_stress_pa: np.ndarray  # thin-walled torsional shear stress, treating the spar box as the torsion cell
    torsion_safety_factor: np.ndarray

    slope_rad: np.ndarray
    deflection_m: np.ndarray  # Euler-Bernoulli bending deflection from the root

    @property
    def min_torsion_safety_factor(self) -> float:
        return float(np.nanmin(self.torsion_safety_factor))

    @property
    def tip_deflection_m(self) -> float:
        return float(self.deflection_m[-1])


def analyze_torsion_and_deflection(
    aircraft: Aircraft, structures: StructuralProxyResult,
    spar_x_fraction_chord: float = SPAR_X_FRACTION_CHORD,
    aerodynamic_center_x_fraction_chord: float = AERODYNAMIC_CENTER_X_FRACTION_CHORD,
    spar_young_modulus_pa: float = SPAR_YOUNG_MODULUS_PA,
) -> TorsionDeflectionResult:
    """First-order torsion + deflection estimate on top of the existing
    bending/shear structural proxy. Torque is driven by the offset between
    the local aerodynamic center (~quarter chord, thin-airfoil-theory
    assumption) and the assumed spar/elastic axis -- the dominant source of
    torsion on a swept wing; each section's own pitching moment is a smaller
    secondary contribution and isn't included here (would need a per-station
    2D analysis call, which this proxy skips to stay cheap)."""
    x_ac = aircraft.x_le_m + aerodynamic_center_x_fraction_chord * aircraft.chord_m
    x_spar = aircraft.x_le_m + spar_x_fraction_chord * aircraft.chord_m
    torque_per_span = structures.lift_per_span_n_per_m * (x_ac - x_spar)
    torque_nm = _reverse_cumulative_trapezoid(torque_per_span, structures.span_station_m)

    enclosed_area = structures.spar_width_m * structures.spar_depth_available_m
    t = SPAR_WALL_THICKNESS_M
    shear_stress = np.divide(
        np.abs(torque_nm), 2.0 * enclosed_area * t,
        out=np.zeros_like(torque_nm), where=enclosed_area > 1e-12,
    )
    torsion_safety_factor = np.full_like(shear_stress, np.nan)
    stressed = shear_stress > 1e-6
    torsion_safety_factor[stressed] = ALLOWABLE_SPAR_SHEAR_STRESS_PA / shear_stress[stressed]

    # Cantilever clamped at the root (y=0): slope and deflection are both
    # zero there, so -- unlike shear/moment, which accumulate from the tip
    # inward -- these integrate forward from the root outward.
    ei = spar_young_modulus_pa * np.maximum(structures.moment_of_inertia_m4, 1e-18)
    curvature = structures.bending_moment_nm / ei
    slope_rad = cumulative_trapezoid(curvature, structures.span_station_m, initial=0.0)
    deflection_m = cumulative_trapezoid(slope_rad, structures.span_station_m, initial=0.0)

    return TorsionDeflectionResult(
        y_stations=structures.y_stations, span_station_m=structures.span_station_m,
        torque_nm=torque_nm, shear_stress_pa=shear_stress, torsion_safety_factor=torsion_safety_factor,
        slope_rad=slope_rad, deflection_m=deflection_m,
    )
