"""Aircraft geometry generator.

The central component of the project: turns a `DesignParameters` vector into
a complete `Aircraft` -- a fine (~200 station) spanwise-lofted mesh for
visualization/geometric properties, plus an AeroSandbox `Airplane` object for
analysis. Everything here is pure function of `DesignParameters`; nothing in
this module knows anything about optimizers or analysis backends.

Only one wing (the flying wing itself) is generated. Winglets, gull-wing
bends, blended centre bodies etc. are not separate parts -- they emerge from
`chord(y)`, `twist(y)`, `le_offset(y)` and `z_offset(y)` being free-form
curves (see `params.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import aerosandbox as asb

from .params import DesignParameters
from .spanwise import make_span_stations
from .airfoil_family import generate_airfoil, generate_airfoil_surfaces, max_thickness_ratio
from .fuselage import check_fuselage_fit, FuselageFitResult

# Number of chordwise-loft cross sections handed to AeroSandbox for analysis.
# Kept much coarser than the ~200-station visualization mesh: our spanwise
# distributions are themselves piecewise-linear between a handful of control
# points, so a modest number of analysis stations reproduces the same shape
# AeroSandbox's own panel/strip methods can refine spanwise resolution
# further on their own (e.g. VLM's `spanwise_resolution`).
N_ANALYSIS_STATIONS = 41


@dataclass
class Aircraft:
    """A complete generated flying wing (one half modelled, y in [0, 1];
    the aircraft is always symmetric)."""

    params: DesignParameters

    # Fine spanwise arrays, length n_span_stations (default 200).
    y_stations: np.ndarray
    span_station_m: np.ndarray       # physical spanwise coordinate, m (0 at root)
    chord_m: np.ndarray
    twist_deg: np.ndarray
    x_le_m: np.ndarray
    z_le_m: np.ndarray
    thickness_scale: np.ndarray
    camber_scale: np.ndarray
    reflex_scale: np.ndarray
    thickness_ratio: np.ndarray       # actual local t/c

    # Watertight visualization mesh: (n_span_stations, n_surface_points, 3)
    upper_surface_m: np.ndarray
    lower_surface_m: np.ndarray

    # AeroSandbox model, built at a coarser (but consistent) resolution.
    airplane: asb.Airplane

    fuselage_fit: FuselageFitResult

    @property
    def half_span_m(self) -> float:
        return self.params.planform.span_m / 2.0

    @property
    def wing_area_m2(self) -> float:
        """Full (both sides) planform area, trapezoidal integration of chord(y)."""
        one_side = np.trapezoid(self.chord_m, self.span_station_m)
        return 2.0 * one_side

    @property
    def aspect_ratio(self) -> float:
        b = self.params.planform.span_m
        return b ** 2 / self.wing_area_m2

    @property
    def mean_aerodynamic_chord_m(self) -> float:
        one_side_num = np.trapezoid(self.chord_m ** 2, self.span_station_m)
        one_side_den = np.trapezoid(self.chord_m, self.span_station_m)
        return one_side_num / one_side_den

    @property
    def root_chord_m(self) -> float:
        return float(self.chord_m[0])

    @property
    def tip_chord_m(self) -> float:
        return float(self.chord_m[-1])


def _evaluate_stations(params: DesignParameters, y_stations: np.ndarray) -> dict[str, np.ndarray]:
    planform_dists = params.planform.distributions()
    airfoil_dists = params.airfoil_schedule.distributions()

    chord_m = planform_dists["chord_m"](y_stations)
    twist_deg = planform_dists["twist_deg"](y_stations)
    x_le_m = planform_dists["le_offset_m"](y_stations)
    z_le_m = planform_dists["z_offset_m"](y_stations)

    thickness_scale = airfoil_dists["thickness_scale"](y_stations)
    camber_scale = airfoil_dists["camber_scale"](y_stations)
    reflex_scale = airfoil_dists["reflex_scale"](y_stations)

    return dict(
        chord_m=chord_m,
        twist_deg=twist_deg,
        x_le_m=x_le_m,
        z_le_m=z_le_m,
        thickness_scale=thickness_scale,
        camber_scale=camber_scale,
        reflex_scale=reflex_scale,
    )


def _loft_surface_mesh(
    y_stations: np.ndarray,
    span_station_m: np.ndarray,
    station: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 3D upper/lower surface point grids by generating each
    station's airfoil and placing it in 3D (scale by chord, twist about the
    local quarter-chord, translate to its LE position)."""
    n = len(y_stations)
    n_x = None
    upper_pts = None
    lower_pts = None

    for i in range(n):
        x, upper_y, lower_y = generate_airfoil_surfaces(
            station["thickness_scale"][i], station["camber_scale"][i], station["reflex_scale"][i]
        )
        if n_x is None:
            n_x = len(x)
            upper_pts = np.zeros((n, n_x, 3))
            lower_pts = np.zeros((n, n_x, 3))

        chord = station["chord_m"][i]
        twist_rad = np.radians(station["twist_deg"][i])
        x_le = station["x_le_m"][i]
        z_le = station["z_le_m"][i]
        y_span = span_station_m[i]

        for surf_y, out in ((upper_y, upper_pts), (lower_y, lower_pts)):
            xc = x * chord
            zc = surf_y * chord
            # twist about the local leading edge, nose-down positive twist
            # rotates the section such that positive twist_deg increases
            # local incidence... convention: twist rotates (x,z) about LE.
            ct, st = np.cos(twist_rad), np.sin(twist_rad)
            x_rot = xc * ct + zc * st
            z_rot = -xc * st + zc * ct
            out[i, :, 0] = x_le + x_rot
            out[i, :, 1] = y_span
            out[i, :, 2] = z_le + z_rot

    return upper_pts, lower_pts


def build_airplane(params: DesignParameters, n_analysis_stations: int = N_ANALYSIS_STATIONS) -> asb.Airplane:
    # Linear (not cosine) spacing here: VLM/AVL apply their own internal
    # (typically cosine) spanwise spacing between xsecs, so cosine-spaced
    # xsecs on top of that packs many near-duplicate, near-zero-width panels
    # close to the root/tip -- which was driving the AIC matrix singular and
    # the VLM solution to blow up. AeroBuildup (strip theory) doesn't care
    # either way, so linear spacing is the safe default for both.
    y_stations = make_span_stations(n_analysis_stations, cosine_spacing=False)
    span_station_m = y_stations * (params.planform.span_m / 2.0)
    station = _evaluate_stations(params, y_stations)

    xsecs = []
    for i in range(n_analysis_stations):
        airfoil = generate_airfoil(
            station["thickness_scale"][i], station["camber_scale"][i], station["reflex_scale"][i],
            name=f"mh64_derived_y{y_stations[i]:.3f}",
        )
        xsecs.append(
            asb.WingXSec(
                xyz_le=[station["x_le_m"][i], span_station_m[i], station["z_le_m"][i]],
                chord=station["chord_m"][i],
                twist=station["twist_deg"][i],
                airfoil=airfoil,
            )
        )

    wing = asb.Wing(name="main_wing", xsecs=xsecs, symmetric=True)

    # Rough xyz_ref at 25% of the mean aerodynamic chord, on the symmetry
    # plane -- refined later once mass estimation is available.
    mac_num = np.trapezoid(station["chord_m"] ** 2, span_station_m)
    mac_den = np.trapezoid(station["chord_m"], span_station_m)
    mac = mac_num / mac_den
    x_le_root = station["x_le_m"][0]
    xyz_ref = [x_le_root + 0.25 * mac, 0.0, 0.0]

    return asb.Airplane(name="flying_wing", xyz_ref=xyz_ref, wings=[wing])


def build_aircraft(params: DesignParameters) -> Aircraft:
    """Generate the complete aircraft (mesh + AeroSandbox model + derived
    properties) from a `DesignParameters` vector. This is the one entry
    point the optimizer's proposed vectors flow through."""
    n = params.n_span_stations
    y_stations = make_span_stations(n, cosine_spacing=True)
    span_station_m = y_stations * (params.planform.span_m / 2.0)

    station = _evaluate_stations(params, y_stations)
    thickness_ratio = np.array([max_thickness_ratio(t) for t in station["thickness_scale"]])

    upper_surface_m, lower_surface_m = _loft_surface_mesh(y_stations, span_station_m, station)

    airplane = build_airplane(params)

    fuselage_fit = check_fuselage_fit(
        y_stations=y_stations,
        span_m=params.planform.span_m,
        chord_m=station["chord_m"],
        thickness_ratio=thickness_ratio,
    )

    return Aircraft(
        params=params,
        y_stations=y_stations,
        span_station_m=span_station_m,
        chord_m=station["chord_m"],
        twist_deg=station["twist_deg"],
        x_le_m=station["x_le_m"],
        z_le_m=station["z_le_m"],
        thickness_scale=station["thickness_scale"],
        camber_scale=station["camber_scale"],
        reflex_scale=station["reflex_scale"],
        thickness_ratio=thickness_ratio,
        upper_surface_m=upper_surface_m,
        lower_surface_m=lower_surface_m,
        airplane=airplane,
        fuselage_fit=fuselage_fit,
    )
