"""Flat vector <-> DesignParameters translation layer.

This is what keeps the geometry module independent of the optimizer: an
`Optimizer` only ever sees a flat numpy vector and per-dimension bounds; it
knows nothing about `DesignParameters`, `AirfoilSchedule`, or `Planform`. A
`ParameterSet` is the small, explicit mapping between the two, built by
whatever stage (Stage 1: airfoil schedule, Stage 2: planform) needs it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..geometry.params import DesignParameters


@dataclass
class Var:
    name: str
    lower: float
    upper: float
    default: float


BuildFn = Callable[[np.ndarray, DesignParameters], DesignParameters]


@dataclass
class ParameterSet:
    """A named, bounded list of scalar variables, plus a function that
    builds a complete `DesignParameters` from their values (layered onto a
    fixed baseline for whatever part of the design isn't being optimized)."""

    variables: list[Var]
    build_fn: BuildFn
    baseline: DesignParameters

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(v.lower, v.upper) for v in self.variables]

    @property
    def default_vector(self) -> np.ndarray:
        return np.array([v.default for v in self.variables])

    def build(self, x: np.ndarray) -> DesignParameters:
        return self.build_fn(np.asarray(x, dtype=float), self.baseline)

    def clip(self, x: np.ndarray) -> np.ndarray:
        lo = np.array([v.lower for v in self.variables])
        hi = np.array([v.upper for v in self.variables])
        return np.clip(x, lo, hi)


def resolve_per_station_bounds(value, n: int) -> list[tuple[float, float]]:
    """A Stage 1/2 bound config value (e.g. CHORD_STATION_M_BOUNDS) may be
    either a single (lo, hi) pair -- broadcast to all `n` stations/segments,
    today's behavior -- or a list of exactly `n` (lo, hi) pairs, one per
    station/segment, for finer-grained control (e.g. tightening only the
    outboard segments). Returns a list of `n` (lo, hi) pairs either way.
    """
    seq = list(value)
    if len(seq) == 2 and all(isinstance(v, (int, float)) for v in seq):
        return [(float(seq[0]), float(seq[1]))] * n
    if len(seq) == n and all(len(v) == 2 for v in seq):
        return [(float(v[0]), float(v[1])) for v in seq]
    raise ValueError(
        f"bounds must be a single (lo, hi) pair or a list of {n} (lo, hi) pairs, got {value!r}"
    )
