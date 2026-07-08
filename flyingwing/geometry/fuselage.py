"""Internal fuselage box fit check.

The fuselage is modelled as a required internal rectangular box (width x
height x length) that must fit entirely inside the generated centre body.
No separate fuselage geometry is generated -- this just checks whether the
wing's own upper/lower surface envelope near the root leaves enough room,
and *where* the box would have to sit to fit.

Box placement:
  - Width: centered on the symmetry plane, spanning y in [-half_width, half_width]
    (fixed by symmetry -- not searched).
  - Length (x) and height (z): searched. The box's chordwise window [x0, x0+L]
    is swept across the footprint to find the position that maximizes vertical
    clearance -- simultaneously requiring, at every span station within the
    footprint, that the window falls entirely within that station's chord
    AND that the local upper/lower surface gap within the window is at least
    the required height. This replaces an earlier version that only checked
    "is there enough thickness somewhere along the chord" and "is the chord
    long enough" independently, without requiring both at the same x-position
    -- which could report a fit even when no single box placement actually
    worked, and gave a visualization no principled way to place the box.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import (
    FUSELAGE_MIN_INTERNAL_WIDTH_M,
    FUSELAGE_MIN_INTERNAL_HEIGHT_M,
    FUSELAGE_MIN_INTERNAL_LENGTH_M,
)

N_X_CANDIDATES = 80


@dataclass
class FuselageFitResult:
    fits: bool
    min_height_margin_m: float  # available - required, at the best box placement found (negative = violation)
    min_length_margin_m: float  # shortest footprint station's chord - required length (independent of placement)
    footprint_y_stations: int

    # The box placement that achieves min_height_margin_m (NaN if no
    # footprint stations / no valid length window at all).
    box_x_min_m: float = float("nan")
    box_x_max_m: float = float("nan")
    box_z_min_m: float = float("nan")
    box_z_max_m: float = float("nan")

    @property
    def violations(self) -> list[str]:
        out = []
        if self.min_height_margin_m < 0:
            out.append(f"fuselage height margin {self.min_height_margin_m * 1000:.1f} mm short")
        if self.min_length_margin_m < 0:
            out.append(f"fuselage length margin {self.min_length_margin_m * 1000:.1f} mm short")
        if self.footprint_y_stations == 0:
            out.append("no span stations fall within the fuselage width footprint")
        return out


def check_fuselage_fit(
    y_stations: np.ndarray,
    span_m: float,
    upper_surface_m: np.ndarray,
    lower_surface_m: np.ndarray,
    required_width_m: float = FUSELAGE_MIN_INTERNAL_WIDTH_M,
    required_height_m: float = FUSELAGE_MIN_INTERNAL_HEIGHT_M,
    required_length_m: float = FUSELAGE_MIN_INTERNAL_LENGTH_M,
    n_x_candidates: int = N_X_CANDIDATES,
) -> FuselageFitResult:
    half_width = required_width_m / 2.0
    y_fuselage_edge = half_width / (span_m / 2.0)  # normalized y where the box footprint ends

    footprint = y_stations <= y_fuselage_edge
    n_footprint = int(np.count_nonzero(footprint))
    if n_footprint == 0:
        return FuselageFitResult(
            fits=False,
            min_height_margin_m=-required_height_m,
            min_length_margin_m=-required_length_m,
            footprint_y_stations=0,
        )

    xs = upper_surface_m[footprint, :, 0]  # (n_footprint, n_chord) -- x_le/twist differ per station
    zu = upper_surface_m[footprint, :, 2]
    zl = lower_surface_m[footprint, :, 2]

    x_le_per_station = xs.min(axis=1)
    x_te_per_station = xs.max(axis=1)
    min_length_margin = float((x_te_per_station - x_le_per_station).min() - required_length_m)

    # The box's chordwise window must fall entirely within EVERY footprint
    # station's own chord -- the feasible range for the window's start x.
    x_start_lo = float(x_le_per_station.max())
    x_start_hi = float((x_te_per_station - required_length_m).min())

    if x_start_hi < x_start_lo:
        # No single window fits within every footprint station's chord --
        # i.e. min_length_margin < 0 already covers this; no valid placement.
        return FuselageFitResult(
            fits=False, min_height_margin_m=-required_height_m, min_length_margin_m=min_length_margin,
            footprint_y_stations=n_footprint,
        )

    best_margin = -np.inf
    best_placement = None
    for x_start in np.linspace(x_start_lo, x_start_hi, n_x_candidates):
        x_end = x_start + required_length_m
        mask = (xs >= x_start) & (xs <= x_end)
        if not mask.any(axis=1).all():
            continue  # window falls between grid points for some station -- skip, dense grid makes this rare

        min_upper = np.where(mask, zu, np.inf).min(axis=1)
        max_lower = np.where(mask, zl, -np.inf).max(axis=1)
        margin = float((min_upper - max_lower - required_height_m).min())
        if margin > best_margin:
            best_margin = margin
            best_placement = (x_start, x_end, float(min_upper.min()), float(max_lower.max()))

    if best_placement is None:
        return FuselageFitResult(
            fits=False, min_height_margin_m=-required_height_m, min_length_margin_m=min_length_margin,
            footprint_y_stations=n_footprint,
        )

    x_start, x_end, min_upper_overall, max_lower_overall = best_placement
    z_center = (min_upper_overall + max_lower_overall) / 2.0

    return FuselageFitResult(
        fits=(best_margin >= 0) and (min_length_margin >= 0),
        min_height_margin_m=best_margin, min_length_margin_m=min_length_margin,
        footprint_y_stations=n_footprint,
        box_x_min_m=x_start, box_x_max_m=x_end,
        box_z_min_m=z_center - required_height_m / 2.0, box_z_max_m=z_center + required_height_m / 2.0,
    )
