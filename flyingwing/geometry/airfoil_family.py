"""MH64-derived airfoil family.

Only one airfoil family is used in this project. Every section airfoil is
generated from the same base MH64 profile by applying three smooth,
spanwise-varying modifications:

    thickness_scale  -- multiplies the baseline thickness envelope
    camber_scale     -- multiplies the baseline camber line
    reflex_scale     -- adds an aft-loaded "reflex" deflection to the camber
                         line (on top of whatever reflex MH64 already has),
                         so the amount of self-stabilizing reflex can be
                         tuned per span station independently of camber.

The modifications are done in physical (x, y)/chord space (camber/thickness
decomposition), not in an abstract parameterization, so `thickness(y)`,
`camber(y)`, `reflex(y)` stay physically meaningful the way the project spec
requires. Linear interpolation between spanwise control stations (see
`spanwise.py`) then produces a continuously varying airfoil family along the
span, `MH64(thickness(y), camber(y), reflex(y))`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import aerosandbox as asb

from ..config import BASE_AIRFOIL_NAME

N_SURFACE_POINTS = 161  # points per surface, cosine-spaced in x/c
REFLEX_X0 = 0.55  # chord fraction where the reflex bump begins
REFLEX_NOMINAL_AMPLITUDE = 0.02  # y/c deflection at the TE for reflex_scale = 1.0

# Sanity bounds used for validation elsewhere in the framework.
MIN_THICKNESS_RATIO = 0.035  # 3.5% chord: thinnest realistic tip section
MAX_THICKNESS_RATIO = 0.18  # 18% chord: thickest realistic root section


def _cosine_x_grid(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return 0.5 * (1.0 - np.cos(np.pi * t))


def _split_surfaces(coordinates: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split Selig-format coordinates (TE -> upper surface -> LE -> lower
    surface -> TE) into upper(x) and lower(x) resampled on `x_grid`."""
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    le_idx = int(np.argmin(x))

    upper_x = x[: le_idx + 1][::-1]
    upper_y = y[: le_idx + 1][::-1]
    lower_x = x[le_idx:]
    lower_y = y[le_idx:]

    upper_on_grid = np.interp(x_grid, upper_x, upper_y)
    lower_on_grid = np.interp(x_grid, lower_x, lower_y)
    return upper_on_grid, lower_on_grid


@dataclass(frozen=True)
class _BaseShape:
    x_grid: np.ndarray
    camber: np.ndarray
    thickness: np.ndarray
    max_thickness_ratio: float


@lru_cache(maxsize=1)
def _base_shape() -> _BaseShape:
    base = asb.Airfoil(BASE_AIRFOIL_NAME)
    x_grid = _cosine_x_grid(N_SURFACE_POINTS)
    upper, lower = _split_surfaces(base.coordinates, x_grid)

    camber = 0.5 * (upper + lower)
    thickness = upper - lower
    if np.any(thickness < -1e-9):
        raise RuntimeError("base MH64 upper/lower surfaces cross -- unexpected airfoil data")
    thickness = np.clip(thickness, 0.0, None)

    return _BaseShape(
        x_grid=x_grid,
        camber=camber,
        thickness=thickness,
        max_thickness_ratio=float(np.max(thickness)),
    )


def _reflex_bump(x_grid: np.ndarray) -> np.ndarray:
    """Smoothstep bump, 0 for x < REFLEX_X0, ramping (C1) up to 1 at the TE."""
    s = np.clip((x_grid - REFLEX_X0) / (1.0 - REFLEX_X0), 0.0, 1.0)
    return s ** 2 * (3.0 - 2.0 * s)


def max_thickness_ratio(thickness_scale: float) -> float:
    """Max thickness / chord that results from a given thickness_scale.

    Exact (not approximate): scaling the whole thickness envelope by a
    constant factor scales its max by the same factor.
    """
    return thickness_scale * _base_shape().max_thickness_ratio


@lru_cache(maxsize=1)
def max_thickness_x_over_c() -> float:
    """x/c where the base MH64 profile reaches its maximum thickness -- the
    standard, physically sensible spar chordwise location (best bending
    stiffness per unit spar mass), computed directly from the profile
    instead of assumed as a tunable fraction. Exact, not approximate:
    thickness_scale multiplies the whole envelope by a constant, so it never
    shifts WHERE the peak occurs."""
    shape = _base_shape()
    return float(shape.x_grid[np.argmax(shape.thickness)])


@lru_cache(maxsize=1)
def shell_centroid_x_over_c() -> float:
    """Arc-length-weighted x/c centroid of the base MH64 profile's upper +
    lower surface. The shell/skin's mass follows wetted surface length, not
    enclosed area, so this -- not an assumed chord fraction -- is the
    physically correct chordwise centroid for shell mass. Computed once from
    the unscaled base shape as a quick, cheap approximation (ignores the
    small shift a given section's own thickness/camber scaling would cause)."""
    shape = _base_shape()
    x = shape.x_grid
    upper = shape.camber + shape.thickness / 2.0
    lower = shape.camber - shape.thickness / 2.0

    def _surface_moment(y: np.ndarray) -> tuple[float, float]:
        dx, dy = np.diff(x), np.diff(y)
        ds = np.sqrt(dx ** 2 + dy ** 2)
        x_mid = 0.5 * (x[:-1] + x[1:])
        return float(np.sum(x_mid * ds)), float(np.sum(ds))

    mx_u, len_u = _surface_moment(upper)
    mx_l, len_l = _surface_moment(lower)
    return (mx_u + mx_l) / (len_u + len_l)


def generate_airfoil_surfaces(
    thickness_scale: float,
    camber_scale: float,
    reflex_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_grid, upper_y, lower_y) for one MH64-derived section, all in
    chord-normalized coordinates (x, y in [0, 1] roughly).

    Args:
        thickness_scale: multiplies the baseline MH64 thickness envelope.
        camber_scale: multiplies the baseline MH64 camber line.
        reflex_scale: added aft-loading deflection, in units of
            `REFLEX_NOMINAL_AMPLITUDE` at the trailing edge.
    """
    shape = _base_shape()
    x = shape.x_grid

    thickness = shape.thickness * thickness_scale
    camber = shape.camber * camber_scale + reflex_scale * REFLEX_NOMINAL_AMPLITUDE * _reflex_bump(x)

    upper = camber + thickness / 2.0
    lower = camber - thickness / 2.0
    return x, upper, lower


def generate_airfoil(
    thickness_scale: float,
    camber_scale: float,
    reflex_scale: float,
    name: str = "mh64_derived",
) -> asb.Airfoil:
    """Build one MH64-derived section airfoil (as an AeroSandbox Airfoil)."""
    x, upper, lower = generate_airfoil_surfaces(thickness_scale, camber_scale, reflex_scale)

    # Reassemble Selig format: TE -> LE along the upper surface, then LE -> TE
    # along the lower surface, sharing a single LE point.
    upper_x_te_to_le = x[::-1]
    upper_y_te_to_le = upper[::-1]
    lower_x_le_to_te = x[1:]
    lower_y_le_to_te = lower[1:]

    coords_x = np.concatenate([upper_x_te_to_le, lower_x_le_to_te])
    coords_y = np.concatenate([upper_y_te_to_le, lower_y_le_to_te])
    coordinates = np.column_stack([coords_x, coords_y])

    return asb.Airfoil(name=name, coordinates=coordinates)


def validate_airfoil_shape(thickness_scale: float, camber_scale: float, reflex_scale: float) -> list[str]:
    """Cheap geometric sanity checks that don't require building the airfoil.

    Returns a list of human-readable violation strings (empty if valid).
    """
    violations = []
    t_ratio = max_thickness_ratio(thickness_scale)
    if t_ratio < MIN_THICKNESS_RATIO:
        violations.append(f"thickness ratio {t_ratio:.3f} below minimum {MIN_THICKNESS_RATIO:.3f}")
    if t_ratio > MAX_THICKNESS_RATIO:
        violations.append(f"thickness ratio {t_ratio:.3f} above maximum {MAX_THICKNESS_RATIO:.3f}")
    if thickness_scale <= 0:
        violations.append("thickness_scale must be positive (non-self-intersecting surfaces)")
    return violations
