"""Subprocess-based optimizer run launching, tracking, and log polling.

Runs are launched as separate OS processes invoking the existing
`scripts/run_*.py` CLI scripts (extended with argparse) -- never by
importing and calling the optimizer in-process. This keeps a crashed/hung
run from taking down the GUI, and means "running from the GUI" and
"running from the terminal" are exactly the same code path. Only one run
is tracked at a time (a simple, safe default given each run already uses
multiprocessing internally) -- starting a new one while one is active is
rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
import time

from ..config import OUTPUT_DIR, PROJECT_ROOT

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SCRIPT_BY_RUN_TYPE = {
    "stage1": "run_stage1.py",
    "stage2": "run_stage2.py",
    "multi_cycle": "run_multi_cycle.py",
}


@dataclass
class RunHandle:
    run_type: str
    output_dir_name: str
    log_path: Path
    process: subprocess.Popen
    started_at: float = field(default_factory=time.time)


_current_run: RunHandle | None = None


def is_running() -> bool:
    return _current_run is not None and _current_run.process.poll() is None


def launch_run(run_type: str, args: list[str], output_dir_name: str) -> RunHandle:
    """Start `scripts/run_<run_type>.py --output-dir-name <output_dir_name> <args...>`
    as a background subprocess, logging to `outputs/<output_dir_name>/run.log`."""
    global _current_run
    if is_running():
        raise RuntimeError("A run is already in progress -- wait for it to finish first.")
    if run_type not in SCRIPT_BY_RUN_TYPE:
        raise ValueError(f"unknown run_type {run_type!r}")

    script = SCRIPTS_DIR / SCRIPT_BY_RUN_TYPE[run_type]
    out_dir = OUTPUT_DIR / output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    full_args = [sys.executable, str(script), "--output-dir-name", output_dir_name, *args]
    log_file = open(log_path, "w")
    try:
        process = subprocess.Popen(full_args, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
    finally:
        log_file.close()  # the child has its own duplicated handle; safe to close ours

    _current_run = RunHandle(run_type=run_type, output_dir_name=output_dir_name, log_path=log_path, process=process)
    return _current_run


def get_current_run() -> RunHandle | None:
    return _current_run


def read_log_tail(max_chars: int = 6000) -> str:
    if _current_run is None or not _current_run.log_path.exists():
        return ""
    return _current_run.log_path.read_text(errors="replace")[-max_chars:]


def status() -> str:
    """One of: 'idle', 'running', 'completed', 'failed'."""
    if _current_run is None:
        return "idle"
    if _current_run.process.poll() is None:
        return "running"
    if _current_run.process.returncode == 0 and "RUN_COMPLETE" in read_log_tail():
        return "completed"
    return "failed"
