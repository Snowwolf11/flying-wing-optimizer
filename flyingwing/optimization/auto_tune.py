"""Automatic optimizer-settings selection for a target wall-clock duration.

For users who just want to try the framework without learning what sigma0,
population_size, max_generations, n_restarts, n_jobs, and n_cycles each do
(see cmaes.py's module docstring for the real explanation): given only a
target duration and a problem's dimensionality, pick CMA-ES settings --
CMA-ES is always used here, being the default/best-evidenced method, see
cmaes.py -- that approximately fill that time budget while staying sensible
(not, say, population_size=2 with a million generations).

This module only contains the pure "eval budget -> settings" allocation
math. The wall-clock timing measurement it needs as an input (real per-eval
cost, including actual ProcessPoolExecutor overhead) is the caller's job --
see gui/run_tab.py's `_time_one_batch`, which already does this for the
"Estimate Time" button and is reused for Simple mode too.

Design note (v2, after real Simple-mode runs kept "converging quickly and
then barely improving"): CMA-ES's own convergence criteria (tolx/tolfun,
see cmaes.py's `es.stop()`) frequently end a restart well before
max_generations is reached, especially once it has settled near one basin --
so past a certain point, handing a restart *more* generations is often
wasted, while a *restart* (fresh random start, see cmaes.py's IPOP
docstring) is the mechanism that actually reaches a genuinely different
region. v1 capped restarts at a small fixed number and let excess budget
inflate a single restart's generation count instead -- exactly backwards
for a long run. v2 instead grants restarts fairly liberally (see
TARGET_GENERATIONS_PER_RESTART/MAX_RESTARTS) and also scales sigma0 (the
initial step size) up with the budget, since a longer run can afford a
wider initial search and still have time to converge afterwards, while a
short run should stay close to the seed (which is already guaranteed at
least as good as what came before it -- see cmaes.py's x0-injection) rather
than risk a wasted wide jump with little time to recover from it. These
anchor values (generations/restart, sigma0 range, eval-budget log-range,
cycle cap) are reasoned defaults, not empirically tuned against real runs
across the full budget range -- treat them as a starting point.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# A restart needs at least this many generations to have had a fair chance
# to adapt its covariance matrix and actually converge -- below this it's
# essentially wasted. Below MAX_RESTARTS of these can be afforded before
# falling back to fewer, deeper restarts.
TARGET_GENERATIONS_PER_RESTART = 40
MAX_RESTARTS = 10  # a soft sanity cap -- IPOP's exponential 2**r population growth already self-limits how many restarts any realistic budget affords well before this many

# A whole extra multi-cycle pass is worth having even if comparatively
# shallow -- alternating which variables are being optimized is itself a
# source of improvement (see the "improvements often come from switching
# stages" observation this was built to accommodate) independent of how
# deep any single stage's own search goes -- so this floor is deliberately
# lower than TARGET_GENERATIONS_PER_RESTART above.
MIN_GENERATIONS_FOR_EXTRA_CYCLE = 15
MAX_CYCLES = 6  # a *weak* cap, per explicit request -- more cycles helps, but with rapidly diminishing returns past a handful, and each one adds re-optimization overhead

SIGMA0_MIN = 0.20  # short runs: stay close to the (already-good) seed rather than risk a wide, wasted jump with little time to recover
SIGMA0_MAX = 0.45  # long runs: room to still converge after a genuinely broad initial search; kept under 0.5 (half the normalized [0,1] domain) to avoid most samples landing outside bounds and getting clipped
_SIGMA0_LOG_EVALS_LO = np.log10(200)     # roughly a short (single-digit-minutes) run
_SIGMA0_LOG_EVALS_HI = np.log10(50_000)  # roughly a very long (many-hours) run


def default_n_jobs() -> int:
    """Use most of the machine's cores, leaving one free for the GUI/OS."""
    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count - 1)


def auto_sigma0(n_evals_budget: float) -> float:
    """Bigger evaluation budgets get a bigger initial step size -- see this
    module's docstring for why. Interpolated log-linearly in eval-budget
    (budgets span multiple orders of magnitude -- 10 min to 1000 min is
    already ~100x -- so a linear scale would barely move for realistic short
    runs and blow up for long ones), clamped to [SIGMA0_MIN, SIGMA0_MAX].
    """
    frac = (np.log10(max(n_evals_budget, 1.0)) - _SIGMA0_LOG_EVALS_LO) / (_SIGMA0_LOG_EVALS_HI - _SIGMA0_LOG_EVALS_LO)
    frac = float(np.clip(frac, 0.0, 1.0))
    return SIGMA0_MIN + frac * (SIGMA0_MAX - SIGMA0_MIN)


@dataclass
class AutoCMASettings:
    sigma0: float
    population_size: int  # informational only -- what pycma's own auto heuristic (4 + 3*ln(n)) will pick; never passed explicitly, so each stage of a multi-cycle run still gets its own dimension-appropriate size
    max_generations: int
    n_restarts: int
    planned_evals: int


def auto_cma_settings(n_dims: int, n_evals_budget: float) -> AutoCMASettings:
    """Choose (sigma0, max_generations, n_restarts) for a problem of n_dims
    dimensions given roughly n_evals_budget evaluations to spend.

    Prefers more IPOP restarts -- each one starts from a fresh random point
    (cmaes.py), the actual mechanism for reaching a different region of the
    search space -- as long as every restart still gets at least
    TARGET_GENERATIONS_PER_RESTART generations to work with; only once
    that's no longer affordable does it fall back to fewer, deeper restarts.
    """
    population_size = max(4, int(4 + 3 * np.log(max(n_dims, 1))))
    sigma0 = auto_sigma0(n_evals_budget)

    best_restarts = 1
    for n_restarts in range(1, MAX_RESTARTS + 1):
        cost_factor = (2 ** n_restarts) - 1  # sum(2**r for r in range(n_restarts)) -- IPOP doubles population each restart
        max_gen = n_evals_budget / (population_size * cost_factor)
        if max_gen < TARGET_GENERATIONS_PER_RESTART and n_restarts > 1:
            break
        best_restarts = n_restarts

    cost_factor = (2 ** best_restarts) - 1
    max_generations = max(1, int(n_evals_budget / (population_size * cost_factor)))
    planned_evals = sum(population_size * (2 ** r) * max_generations for r in range(best_restarts))
    return AutoCMASettings(
        sigma0=sigma0, population_size=population_size, max_generations=max_generations,
        n_restarts=best_restarts, planned_evals=planned_evals,
    )


def auto_n_cycles(n_evals_budget_total: float, n_dims_larger_stage: int) -> int:
    """How many Stage1<->Stage2 cycles a multi-cycle run's total eval budget
    can afford -- weakly (see MAX_CYCLES): checked against a single restart's
    worth of depth (MIN_GENERATIONS_FOR_EXTRA_CYCLE), not
    TARGET_GENERATIONS_PER_RESTART, since a cycle is worth having even if its
    own stages don't get multiple restarts -- restarts remain the primary
    exploration lever within each stage, cycles a secondary one across them.
    """
    population_size = max(4, int(4 + 3 * np.log(max(n_dims_larger_stage, 1))))
    best_n_cycles = 1
    for n_cycles in range(1, MAX_CYCLES + 1):
        per_substage_budget = n_evals_budget_total / (n_cycles * 2)
        gens_at_one_restart = per_substage_budget / population_size
        if gens_at_one_restart < MIN_GENERATIONS_FOR_EXTRA_CYCLE and n_cycles > 1:
            break
        best_n_cycles = n_cycles
    return best_n_cycles
