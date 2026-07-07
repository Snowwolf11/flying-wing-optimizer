"""Internal fuselage box fit check.

The fuselage is modelled as a required internal rectangular box (width x
height x length) that must fit entirely inside the generated centre body.
No separate fuselage geometry is generated -- this just checks whether the
wing's own thickness/chord envelope near the root leaves enough room.

Box placement assumptions (kept simple and explicit so they're easy to
revisit):
  - Width: centered on the symmetry plane, spanning y in [0, half_width].
  - Length: chordwise, must fit within the local chord at every station the
    box spans, roughly centered on the local chord.
  - Height: must fit within the local thickness envelope (thickness_ratio *
    chord) at every station the box spans.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import (
    FUSELAGE_MIN_INTERNAL_WIDTH_M,
    FUSELAGE_MIN_INTERNAL_HEIGHT_M,
    FUSELAGE_MIN_INTERNAL_LENGTH_M,
)


@dataclass
class FuselageFitResult:
    fits: bool
    min_height_margin_m: float  # available - required, minimum over the footprint (negative = violation)
    min_length_margin_m: float
    footprint_y_stations: int

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
    chord_m: np.ndarray,
    thickness_ratio: np.ndarray,
    required_width_m: float = FUSELAGE_MIN_INTERNAL_WIDTH_M,
    required_height_m: float = FUSELAGE_MIN_INTERNAL_HEIGHT_M,
    required_length_m: float = FUSELAGE_MIN_INTERNAL_LENGTH_M,
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

    available_height = thickness_ratio[footprint] * chord_m[footprint]
    available_length = chord_m[footprint]

    min_height_margin = float(np.min(available_height) - required_height_m)
    min_length_margin = float(np.min(available_length) - required_length_m)

    return FuselageFitResult(
        fits=(min_height_margin >= 0) and (min_length_margin >= 0),
        min_height_margin_m=min_height_margin,
        min_length_margin_m=min_length_margin,
        footprint_y_stations=n_footprint,
    )
