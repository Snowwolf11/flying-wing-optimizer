"""Run Stage 1 (airfoil-schedule) optimization, and write out the optimized
design's metrics, geometry plots, and optimization convergence/parameter-
evolution plots.

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
from flyingwing.optimization.cmaes import CMAESOptimizer
from flyingwing.optimization.stage1 import run_stage1, make_stage1_parameter_set
from flyingwing.viz.geometry_plots import save_all as save_geometry_plots
from flyingwing.viz.optimization_plots import save_all as save_optimization_plots
from flyingwing.config import OUTPUT_DIR, CRUISE_SPEED_MS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer", choices=["cma", "lhs"], default="cma", help="cma = CMA-ES (default). lhs = the original hierarchical Latin Hypercube search, kept as a selectable alternative.")
    # CMA-ES options (optimization/cmaes.py::CMAESOptimizer)
    p.add_argument("--cma-sigma0", type=float, default=0.25)
    p.add_argument("--cma-population-size", type=int, default=None, help="None = pycma's own default heuristic")
    p.add_argument("--cma-max-generations", type=int, default=100)
    p.add_argument("--cma-n-restarts", type=int, default=2)
    # LHS options (optimization/hierarchical.py::HierarchicalGridSearch)
    p.add_argument("--n-stages", type=int, default=4)
    p.add_argument("--n-samples-per-stage", type=int, default=32)
    p.add_argument("--retain-best-n", type=int, default=5)
    p.add_argument("--shrink-factor", type=float, default=0.4)
    # Shared
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--output-dir-name", type=str, default="stage1_run")
    p.add_argument("--weights-yaml", type=str, default=None, help="Path to an ObjectiveWeights YAML file; defaults to configs/objective_weights.yaml if present, else built-in defaults.")
    p.add_argument("--normalization-yaml", type=str, default=None, help="Path to a NormalizationConstants YAML file; defaults to configs/normalization.yaml if present, else built-in (default-baseline-derived) defaults.")
    p.add_argument("--baseline-yaml", type=str, default=None, help="Path to a DesignParameters YAML file (see geometry/params_io.py); defaults to the built-in default design.")
    return p.parse_args()


def _build_optimizer(args):
    if args.optimizer == "lhs":
        return HierarchicalGridSearch(
            n_stages=args.n_stages, n_samples_per_stage=args.n_samples_per_stage,
            retain_best_n=args.retain_best_n, shrink_factor=args.shrink_factor,
            seed=args.seed, n_jobs=args.n_jobs,
        )
    return CMAESOptimizer(
        sigma0=args.cma_sigma0, population_size=args.cma_population_size,
        max_generations=args.cma_max_generations, n_restarts=args.cma_n_restarts,
        seed=args.seed, n_jobs=args.n_jobs,
    )


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

    optimizer = _build_optimizer(args)

    print(f"Running Stage 1 optimization ({args.optimizer})...", flush=True)
    result, best_params = run_stage1(baseline, weights=weights, normalization=normalization, optimizer=optimizer, progress_cb=_print_progress)
    best_metrics = result.best_candidate.extra["metrics"]
    best_score = result.best_candidate.score

    print(f"\nTotal evaluations: {sum(len(s) for s in result.history)}")
    print(f"Baseline score: {baseline_score.score:.2f}  (valid={baseline_metrics.valid})")
    print(f"Optimized score: {best_score:.2f}  (valid={best_metrics.valid})")
    print()
    print(f"{'metric':<28}{'baseline':>12}{'optimized':>12}")
    for field in [
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

    parameter_set = make_stage1_parameter_set(baseline)
    save_optimization_plots(result, parameter_set.names, out_dir)

    with open(out_dir / "result.pkl", "wb") as f:
        pickle.dump({
            "run_type": "stage1",
            "result": result, "best_params": best_params, "baseline": baseline,
            "variable_names": parameter_set.names,
        }, f)

    print(f"\nPlots written to {out_dir}")
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
