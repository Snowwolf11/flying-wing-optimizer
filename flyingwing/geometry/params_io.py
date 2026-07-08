"""Save/load a `DesignParameters` as YAML.

Used to hand a design off between processes -- e.g. the GUI's Run tab
launches optimizer subprocesses that need to receive a baseline design, and
the Design tab's "current values" need to survive that handoff. Generally
useful beyond the GUI too: saving/reloading a specific design by hand.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from .params import DesignParameters, Planform, AirfoilSchedule, default_design_parameters

DEFAULT_DESIGN_YAML = "configs/default_design.yaml"


def _tuples_to_lists(obj):
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_tuples_to_lists(v) for v in obj]
    return obj


def _lists_to_tuples(obj):
    if isinstance(obj, dict):
        return {k: _lists_to_tuples(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return tuple(_lists_to_tuples(v) for v in obj)
    return obj


def save_design_parameters(params: DesignParameters, path: str | Path) -> None:
    data = _tuples_to_lists(asdict(params))
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_design_parameters(path: str | Path) -> DesignParameters:
    with open(path) as f:
        data = yaml.safe_load(f)
    data = _lists_to_tuples(data)
    planform = Planform(**data["planform"])
    airfoil_schedule = AirfoilSchedule(**data["airfoil_schedule"])
    return DesignParameters(
        planform=planform, airfoil_schedule=airfoil_schedule, n_span_stations=data["n_span_stations"],
    )


def load_default_design_parameters(path: str | Path = DEFAULT_DESIGN_YAML) -> DesignParameters:
    """The design used as a baseline wherever none is otherwise specified --
    scripts/CLI runs with no --baseline-yaml, the GUI Run tab's "use
    defaults" option, and the Design tab's initial values. Reads `path` if
    it exists (see the Design tab's "Save as Default Design" button, which
    writes here via save_design_parameters), else falls back to the
    framework's hardcoded geometry.params.default_design_parameters().
    Kept here rather than in params.py to avoid a circular import (this
    module already imports from params.py for the YAML (de)serialization)."""
    p = Path(path)
    if p.exists():
        return load_design_parameters(p)
    return default_design_parameters()
