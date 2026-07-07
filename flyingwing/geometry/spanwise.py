"""Spanwise distributions: control points -> continuous function of y in [0, 1].

Every geometric quantity that varies along the span (chord, twist, thickness
scale, ...) is represented the same way: a small number of (y, value) control
points, interpolated (and flat-extrapolated outside [0, 1]) to whatever span
stations the geometry generator needs. Two interpolation kinds are supported:

- "linear": used for the airfoil schedule (thickness/camber/reflex scale).
  Deliberately linear -- see project notes: robust, predictable, cheap, and
  sufficient given ~200 span stations.
- "pchip": used for the planform (chord/twist/LE offset/z offset), so the
  loft is one smooth curve instead of a faceted, kinked line. PCHIP (shape-
  preserving cubic Hermite) is used rather than a natural cubic spline
  because it cannot overshoot between control points -- if the control
  points are monotonic (e.g. monotonically decreasing chord), the
  interpolated curve is guaranteed to stay monotonic too.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class SpanwiseDistribution:
    """A quantity defined at control stations along the span, y in [0, 1]."""

    y_control: np.ndarray
    values: np.ndarray
    kind: str = "linear"

    def __post_init__(self) -> None:
        self.y_control = np.asarray(self.y_control, dtype=float)
        self.values = np.asarray(self.values, dtype=float)

        if self.y_control.ndim != 1 or self.values.ndim != 1:
            raise ValueError("y_control and values must be 1-D")
        if self.y_control.shape != self.values.shape:
            raise ValueError(
                f"y_control (len {len(self.y_control)}) and values "
                f"(len {len(self.values)}) must have the same length"
            )
        if len(self.y_control) < 2:
            raise ValueError("need at least 2 control stations")
        if np.any(np.diff(self.y_control) <= 0):
            raise ValueError("y_control must be strictly increasing")
        if not np.isclose(self.y_control[0], 0.0):
            raise ValueError("y_control must start at 0.0 (symmetry plane)")
        if not np.isclose(self.y_control[-1], 1.0):
            raise ValueError("y_control must end at 1.0 (wing tip)")
        if self.kind not in ("linear", "pchip"):
            raise ValueError(f"unknown interpolation kind {self.kind!r}")

        self._pchip = PchipInterpolator(self.y_control, self.values) if self.kind == "pchip" else None

    def __call__(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.kind == "pchip":
            return self._pchip(y)
        return np.interp(y, self.y_control, self.values)

    def is_monotonically_decreasing(self, strict: bool = False) -> bool:
        d = np.diff(self.values)
        return bool(np.all(d < 0)) if strict else bool(np.all(d <= 0))

    def is_monotonically_increasing(self, strict: bool = False) -> bool:
        d = np.diff(self.values)
        return bool(np.all(d > 0)) if strict else bool(np.all(d >= 0))


def make_span_stations(n: int, cosine_spacing: bool = True) -> np.ndarray:
    """Generate `n` normalized span stations y in [0, 1].

    Cosine spacing concentrates stations near the root (centre-body / airfoil
    transition region) and near the tip (winglet region), where geometry
    changes fastest -- standard practice for lifting-line / VLM discretization.
    """
    if n < 2:
        raise ValueError("need at least 2 span stations")
    if cosine_spacing:
        t = np.linspace(0.0, 1.0, n)
        return 0.5 * (1.0 - np.cos(np.pi * t))
    return np.linspace(0.0, 1.0, n)
