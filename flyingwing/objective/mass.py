"""Rough parametric mass estimation.

This exists to feed the objective function (mass trades off against
strength, payload volume, and performance) -- it is not a substitute for a
real weight & balance takeoff. Shell mass is a wetted-area proxy; spar mass
integrates the same thin-walled spar box already sized by the structural
proxy, so the mass estimate and the structural safety factor stay
consistent with each other (a heavier/thicker spar shows up as both higher
safety factor and higher mass, so the objective function has to trade the
two off rather than "safety factor" being a free lunch).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _overrides as _ov
from ..geometry.aircraft import Aircraft
from ..analysis.structures import StructuralProxyResult
from ..config import (
    FUSELAGE_MIN_INTERNAL_WIDTH_M, FUSELAGE_MIN_INTERNAL_HEIGHT_M, FUSELAGE_MIN_INTERNAL_LENGTH_M,
    BOUNDS_OVERRIDES_YAML,
)

SHELL_AREAL_DENSITY_KG_M2 = 0.55  # foam-core + glass/film skin, both surfaces combined per unit planform area
SPAR_MATERIAL_DENSITY_KG_M3 = 1600.0  # generic glass/carbon spar-cap laminate
FIXED_EQUIPMENT_MASS_KG = 0.45  # motor, ESC, servos, receiver, FC, wiring allowance

_ov.apply_overrides(
    globals(), BOUNDS_OVERRIDES_YAML,
    {"SHELL_AREAL_DENSITY_KG_M2", "SPAR_MATERIAL_DENSITY_KG_M3", "FIXED_EQUIPMENT_MASS_KG"},
)

REQUIRED_FUSELAGE_BOX_VOLUME_M3 = (
    FUSELAGE_MIN_INTERNAL_WIDTH_M * FUSELAGE_MIN_INTERNAL_HEIGHT_M * FUSELAGE_MIN_INTERNAL_LENGTH_M
)


@dataclass
class MassEstimate:
    shell_mass_kg: float
    spar_mass_kg: float
    fixed_equipment_mass_kg: float
    total_structural_mass_kg: float  # shell + spar + fixed equipment; excludes battery/payload mass

    fuselage_internal_volume_m3: float
    payload_volume_margin_m3: float  # fuselage internal volume beyond the minimum required box


def _fuselage_internal_volume_m3(aircraft: Aircraft) -> float:
    """Integrate an ellipse-proxy cross-sectional area (pi/4 * thickness *
    chord, the usual thin-airfoil area approximation) over the fuselage's
    spanwise footprint, both sides."""
    y_fuselage_edge = (FUSELAGE_MIN_INTERNAL_WIDTH_M / 2.0) / aircraft.half_span_m
    footprint = aircraft.y_stations <= y_fuselage_edge

    cross_section_area = (np.pi / 4.0) * aircraft.thickness_ratio * aircraft.chord_m
    one_side_volume = np.trapezoid(cross_section_area[footprint], aircraft.span_station_m[footprint])
    return 2.0 * one_side_volume


def estimate_mass(aircraft: Aircraft, structures: StructuralProxyResult) -> MassEstimate:
    shell_mass_kg = 2.0 * aircraft.wing_area_m2 * SHELL_AREAL_DENSITY_KG_M2

    one_side_spar_volume = np.trapezoid(structures.spar_material_area_m2, structures.span_station_m)
    spar_mass_kg = 2.0 * one_side_spar_volume * SPAR_MATERIAL_DENSITY_KG_M3

    total_structural_mass_kg = shell_mass_kg + spar_mass_kg + FIXED_EQUIPMENT_MASS_KG

    fuselage_volume = _fuselage_internal_volume_m3(aircraft)
    payload_volume_margin_m3 = fuselage_volume - REQUIRED_FUSELAGE_BOX_VOLUME_M3

    return MassEstimate(
        shell_mass_kg=shell_mass_kg,
        spar_mass_kg=spar_mass_kg,
        fixed_equipment_mass_kg=FIXED_EQUIPMENT_MASS_KG,
        total_structural_mass_kg=total_structural_mass_kg,
        fuselage_internal_volume_m3=fuselage_volume,
        payload_volume_margin_m3=payload_volume_margin_m3,
    )
