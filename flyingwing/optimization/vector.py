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
