"""Scan `outputs/` for optimization run results, and load them.

Any subdirectory of `outputs/` containing a `result.pkl` (written by
`scripts/run_stage1.py`, `run_stage2.py`, or `run_multi_cycle.py`) is a "run"
this module can find and load -- whether it came from the CLI's fixed
directory names (`stage1_run`, `stage2_run`, `multi_cycle_run`) or the GUI's
timestamped ones. `run_type` is stored explicitly in newer pickles; older
ones (predating that field) are inferred from which keys are present.
"""
from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

from ..config import OUTPUT_DIR
from ..geometry.aircraft import Aircraft, build_aircraft
from ..geometry.params import DesignParameters
from ..objective.metrics import DesignMetrics


@dataclass
class RunSummary:
    output_dir_name: str
    run_type: str
    mtime: float
    best_score: float
    valid: bool


def _run_type(data: dict) -> str:
    if "run_type" in data:
        return data["run_type"]
    if "multi_cycle_result" in data:
        return "multi_cycle"
    return "stage_unknown"  # an old stage1/stage2 pickle predating the run_type field


def load_run(output_dir_name: str) -> dict:
    pkl_path = OUTPUT_DIR / output_dir_name / "result.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def run_best_params(data: dict) -> DesignParameters:
    if _run_type(data) == "multi_cycle":
        return data["multi_cycle_result"].best_params
    return data["best_params"]


def run_best_metrics(data: dict) -> DesignMetrics:
    if _run_type(data) == "multi_cycle":
        return data["multi_cycle_result"].best_record.result.best_candidate.extra["metrics"]
    return data["result"].best_candidate.extra["metrics"]


def run_best_score(data: dict) -> float:
    if _run_type(data) == "multi_cycle":
        return data["multi_cycle_result"].best_record.result.best_score
    return data["result"].best_score


def list_runs() -> list[RunSummary]:
    summaries: list[RunSummary] = []
    if not OUTPUT_DIR.exists():
        return summaries

    for d in sorted(OUTPUT_DIR.iterdir()):
        pkl_path = d / "result.pkl"
        if not pkl_path.exists():
            continue
        try:
            data = load_run(d.name)
            metrics = run_best_metrics(data)
            summaries.append(RunSummary(
                output_dir_name=d.name,
                run_type=_run_type(data),
                mtime=pkl_path.stat().st_mtime,
                best_score=run_best_score(data),
                valid=metrics.valid,
            ))
        except Exception:
            continue  # a malformed/partial pickle shouldn't break the whole list

    summaries.sort(key=lambda s: s.mtime, reverse=True)
    return summaries


def load_run_aircraft(output_dir_name: str) -> tuple[dict, DesignParameters, DesignMetrics, Aircraft]:
    data = load_run(output_dir_name)
    params = run_best_params(data)
    metrics = run_best_metrics(data)
    aircraft = build_aircraft(params)
    return data, params, metrics, aircraft
