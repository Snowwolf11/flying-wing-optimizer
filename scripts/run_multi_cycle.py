"""Run a multi-cycle Stage1<->Stage2 optimization, and write out plots for
the final design plus the cross-cycle convergence history.

All arguments are optional and default to the values used throughout this
project's demos -- running with no arguments reproduces exactly what earlier
plain-CLI usage did. The GUI's Run tab invokes this same script as a
subprocess with different arguments rather than duplicating this logic.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flyingwing.geometry.params_io import load_default_design_parameters, load_design_parameters
from flyingwing.geometry.aircraft import build_aircraft
from flyingwing.objective.metrics import evaluate_design
from flyingwing.objective.objective import score, ObjectiveWeights, NormalizationConstants
from flyingwing.objective.performance import estimate_performance
from flyingwing.optimization.hierarchical import HierarchicalGridSearch
from flyingwing.optimization.cycle import run_multi_cycle
from flyingwing.viz.geometry_plots import save_all as save_geometry_plots
from flyingwing.viz.optimization_plots import plot_multi_cycle_convergence
from flyingwing.config import OUTPUT_DIR, CRUISE_SPEED_MS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-cycles", type=int, default=2)
    p.add_argument("--convergence-tol", type=float, default=None)
    p.add_argument("--start-with", choices=["stage1", "stage2"], default="stage1")
    p.add_argument("--stage1-n-stages", type=int, default=3)
    p.add_argument("--stage1-n-samples-per-stage", type=int, default=24)
    p.add_argument("--stage1-retain-best-n", type=int, default=5)
    p.add_argument("--stage2-n-stages", type=int, default=3)
    p.add_argument("--stage2-n-samples-per-stage", type=int, default=40)
    p.add_argument("--stage2-retain-best-n", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--output-dir-name", type=str, default="multi_cycle_run")
    p.add_argument("--weights-yaml", type=str, default=None)
    p.add_argument("--normalization-yaml", type=str, default=None)
    p.add_argument("--baseline-yaml", type=str, default=None)
    return p.parse_args()


def _load_weights(weights_yaml: str | None) -> ObjectiveWeights:
    path = Path(weights_yaml) if weights_yaml else Path("configs/objective_weights.yaml")
    if path.exists():
        return ObjectiveWeights.from_yaml(path)
    return ObjectiveWeights()


def _load_normalization(normalization_yaml: str | None) -> NormalizationConstants:
    path = Path(normalization_yaml) if normalization_yaml else Path("configs/normalization.yaml")
    if path.exists():
        return NormalizationConstants.from_yaml(path)
    return NormalizationConstants()


def _print_progress(info: dict) -> None:
    """One machine-parseable line per optimizer stage, for the GUI's Run tab
    to poll and render as a progress readout (see gui/run_manager.py)."""
    print(f"PROGRESS {json.dumps(info)}", flush=True)


def main():
    args = parse_args()

    baseline = load_design_parameters(args.baseline_yaml) if args.baseline_yaml else load_default_design_parameters()
    weights = _load_weights(args.weights_yaml)
    normalization = _load_normalization(args.normalization_yaml)

    baseline_metrics = evaluate_design(baseline)
    baseline_score = score(baseline_metrics, weights, normalization)

    stage1_optimizer = HierarchicalGridSearch(
        n_stages=args.stage1_n_stages, n_samples_per_stage=args.stage1_n_samples_per_stage,
        retain_best_n=args.stage1_retain_best_n, seed=args.seed, n_jobs=args.n_jobs,
    )
    stage2_optimizer = HierarchicalGridSearch(
        n_stages=args.stage2_n_stages, n_samples_per_stage=args.stage2_n_samples_per_stage,
        retain_best_n=args.stage2_retain_best_n, seed=args.seed, n_jobs=args.n_jobs,
    )

    print("Running multi-cycle Stage1<->Stage2 optimization...", flush=True)
    mc = run_multi_cycle(
        baseline, n_cycles=args.n_cycles, weights=weights, normalization=normalization,
        stage1_optimizer=stage1_optimizer, stage2_optimizer=stage2_optimizer,
        start_with=args.start_with, convergence_tol=args.convergence_tol,
        progress_cb=_print_progress,
    )

    best_metrics = mc.best_record.result.best_candidate.extra["metrics"]
    best_params = mc.best_params

    print(f"\nScore history: {[f'{s:.2f}' for s in mc.score_history]}")
    print(f"Baseline score: {baseline_score.score:.2f}")
    print(f"Best overall: cycle {mc.best_record.cycle}, {mc.best_record.stage}, score {mc.best_record.result.best_score:.2f}")
    print()
    print(f"{'metric':<28}{'baseline':>12}{'optimized':>12}")
    for field in [
        "span_m", "aspect_ratio", "wing_area_m2",
        "cruise_L_over_D", "fast_L_over_D", "root_cl_max", "min_safety_factor",
        "total_structural_mass_kg", "payload_volume_margin_m3", "static_margin",
        "soaring_power_w", "cruise_glide_angle_deg", "cruise_Clb_per_rad",
    ]:
        b = getattr(baseline_metrics, field)
        o = getattr(best_metrics, field)
        print(f"{field:<28}{b:>12.4f}{o:>12.4f}")

    out_dir = OUTPUT_DIR / args.output_dir_name
    aircraft = build_aircraft(best_params)

    perf = estimate_performance(aircraft, best_metrics.total_structural_mass_kg, CRUISE_SPEED_MS)
    battery_range = (
        f"{best_metrics.battery_x_min_m * 1000:.0f}-{best_metrics.battery_x_max_m * 1000:.0f} mm from root LE"
        if best_metrics.battery_range_feasible else "none feasible within the airframe"
    )
    print(f"\nBattery x-range for target static margin: {battery_range}")
    print(f"Best glide ratio: {perf.glide_ratio_max:.1f}  at alpha {perf.glide_alpha_deg:.1f} deg  (glide angle {perf.glide_angle_deg:.1f} deg, sink {perf.sink_rate_ms:.2f} m/s)")
    print(f"Cruise power: {perf.cruise_power_w:.1f} W   Est. endurance: {perf.estimated_endurance_min:.0f} min   Est. range: {perf.estimated_range_km:.0f} km")

    save_geometry_plots(aircraft, out_dir)
    plot_multi_cycle_convergence(mc).write_html(str(out_dir / "multi_cycle_convergence.html"), include_plotlyjs="cdn")

    with open(out_dir / "result.pkl", "wb") as f:
        pickle.dump({"run_type": "multi_cycle", "multi_cycle_result": mc, "baseline": baseline}, f)

    print(f"\nPlots written to {out_dir}")
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
