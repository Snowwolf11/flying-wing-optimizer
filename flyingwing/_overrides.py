"""Internal helper: apply a YAML file of module-level constant overrides.

Used by `config.py` and `objective/mass.py` so their tunable constants
(search bounds, structural/mass assumptions) can be edited -- e.g. from the
GUI's Bounds & Weights tab -- without touching source code. Only names
already listed in `allowed_names` are ever overridden; this is not a
general "trust arbitrary YAML" mechanism, and a missing file is a no-op
(so a fresh checkout with no override file behaves exactly like the
hardcoded defaults).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def apply_overrides(module_globals: dict[str, Any], yaml_path: Path, allowed_names: set[str]) -> None:
    if not yaml_path.exists():
        return
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    for key, value in data.items():
        if key not in allowed_names:
            continue
        current = module_globals.get(key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        module_globals[key] = value
