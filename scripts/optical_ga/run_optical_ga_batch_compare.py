#!/usr/bin/env python
"""
Batch-comparison runner: MCX vs PyXOpto shared-parameter-span GA comparison.

Loads one shared-bounds config (configs/optical_ga_shared_bounds.json) and
runs *N* seeds for both branches with identical GA hyper-parameters and
target colour.  Per-seed metrics and aggregate statistics (best / median /
IQR / std / win-rate) are saved as JSON + CSV summaries + leaderboard.

Two forward modes are supported:

- **surrogate** (default for smoke tests): lightweight analytical
  approximation.  Both branches produce identical results (same seed, same
  evaluator) — useful for validating infrastructure.
- **realistic** (physical MC): each branch attempts its own Monte Carlo
  backend.  Falls back to surrogate if the backend is unavailable unless
  ``--require-physical`` is passed.

Important geometry note:
- This batch compare is a **backend comparison under layered assumptions**.
  PyXOpto uses `skin.Skin3()` (layered MCML), and the MCX branch uses an
  internally generated voxelized 3-layer slab from genome thickness values.
- It does **not** consume arbitrary voxel label volumes in this script.
  For complex geometry MCX runs, use `scripts/simulation/mcx_build_volume.py`
  and `scripts/simulation/mcx_batch_runner.py`.

Usage
-----
Smoke test (2 seeds, tiny budget, surrogate)::

    python scripts/optical_ga/run_optical_ga_batch_compare.py \\
        --num-seeds 2 --generations 3 --population-size 8 \\
        --forward-mode surrogate --output-dir /tmp/batch_smoke

Real run with physical backends (when MCX / xopto are installed)::

    python scripts/optical_ga/run_optical_ga_batch_compare.py \\
        --config configs/optical_ga_shared_bounds.json \\
        --forward-mode realistic --require-physical \\
        --num-seeds 10 --generations 20 \\
        --output-dir outputs/optical_ga/batch_compare
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repo root is on sys.path for internal imports
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.optical_ga.ga_optimiser_optical import (
    GAOptimiserOptical,
)
from scripts.optical_ga.genome_encoding_optical import (
    CORE_PARAM_DEFS,
    param_names,
    num_params,
    genome_from_normalised_vector,
    normalised_vector_from_genome,
)
from scripts.optical_ga.forward_model import run_forward
from scripts.optical_ga.optical_fitness import compute_fitness, FITNESS_MODES
from scripts.optical_ga.mc_wrapper import check_xopto, check_mcx

try:
    from tqdm import tqdm
except ImportError:
    # Provide a no-op fallback so the script works without tqdm
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BranchSeedResult:
    """Results from one GA run for one branch at one seed."""
    seed: int
    branch: str
    best_fitness: float
    best_lab: Tuple[float, float, float]
    best_genome: Dict[str, float]
    generation_count: int
    elapsed_s: float
    forward_mode_used: str
    succeeded: bool
    error_msg: str = ""


@dataclass
class AggregateStats:
    """Aggregate statistics for one branch across seeds."""
    branch: str
    num_seeds: int
    num_succeeded: int
    best_fitness: float
    worst_fitness: float
    median_fitness: float
    mean_fitness: float
    std_fitness: float
    q25_fitness: float
    q75_fitness: float
    iqr_fitness: float
    best_fitness_seed: int
    best_fitness_seeds: List[int]


# ---------------------------------------------------------------------------
# Shared bounds enforcement
# ---------------------------------------------------------------------------


def enforce_shared_bounds(
    config_bounds: Dict[str, List[float]],
) -> int:
    """Validate that all shared-bounds entries match core param definitions.

    For each parameter in *config_bounds* we verify the name exists in
    ``CORE_PARAM_DEFS``.  Parameters not listed keep their default bounds.

    Returns the number of parameter checks that passed.
    """
    core_names = {p[0] for p in CORE_PARAM_DEFS}
    checked = 0
    for name, (low, high) in config_bounds.items():
        if name not in core_names:
            print(
                f"  [WARN] Unknown parameter in shared_bounds: {name!r}. "
                f"Skipping.",
                file=sys.stderr,
            )
            continue
        if low >= high:
            raise ValueError(
                f"Invalid bounds for {name!r}: low={low} >= high={high}"
            )
        checked += 1
    return checked


def apply_shared_bounds(
    config_bounds: Dict[str, List[float]],
) -> int:
    """Apply shared bounds in-place to CORE_PARAM_DEFS.

    Returns the number of parameters updated.
    """
    if not config_bounds:
        return 0

    updated = 0
    for i, entry in enumerate(CORE_PARAM_DEFS):
        name, low, high, default, desc = entry
        if name not in config_bounds:
            continue
        new_low, new_high = config_bounds[name]
        if new_low >= new_high:
            raise ValueError(
                f"Invalid bounds for {name!r}: low={new_low} >= high={new_high}"
            )
        # Keep default in-range for sanity and reproducibility.
        new_default = float(np.clip(default, new_low, new_high))
        CORE_PARAM_DEFS[i] = (name, float(new_low), float(new_high), new_default, desc)
        updated += 1
    return updated


# ---------------------------------------------------------------------------
# Branch-aware evaluator factory
# ---------------------------------------------------------------------------


def _make_branch_evaluator(
    lab_target: Tuple[float, float, float],
    branch_name: str,
    forward_mode: str,
    fitness_mode: str,
    num_photons: int,
    with_specular: bool,
    require_physical: bool,
    allow_surrogate_fallback: bool,
    verbose: bool,
):
    """Return a callable ``evaluator(genome) -> (fitness, comps, lab_sim)``.

    The evaluator attempts the requested *branch_name* backend when
    *forward_mode* is ``"realistic"``.  When the backend is unavailable,
    it falls back to surrogate only if **both**:
    - *allow_surrogate_fallback* is ``True``
    - *require_physical* is not set (kept for backwards compatibility)

    Otherwise, a ``RuntimeError`` with an actionable message is raised.
    """
    if forward_mode == "surrogate":
        # Both branches use identical surrogate evaluator
        if verbose:
            print(f"    [{branch_name}] Using surrogate evaluator")

        def _eval_surrogate(genome):
            lab_sim, _, _, _ = run_forward(
                genome, mode="surrogate",
            )
            fitness, comps = compute_fitness(
                lab_sim, lab_target, mode=fitness_mode,
            )
            return fitness, comps, lab_sim

        return _eval_surrogate

    elif forward_mode == "realistic":
        # --- PyXOpto branch ---
        if branch_name == "pyxopto":
            if check_xopto():
                if verbose:
                    print(f"    [{branch_name}] Using xopto (MCML) backend")

                def _eval_xopto(genome):
                    lab_sim, _, _, _ = run_forward(
                        genome, mode="realistic",
                        num_photons=num_photons,
                        with_specular=with_specular,
                    )
                    fitness, comps = compute_fitness(
                        lab_sim, lab_target, mode=fitness_mode,
                    )
                    return fitness, comps, lab_sim

                return _eval_xopto
            else:
                if (not allow_surrogate_fallback) or require_physical:
                    from scripts.optical_ga.mc_wrapper import xopto_source
                    raise RuntimeError(
                        "PyXOpto (xopto) not available.  "
                        "Install it with: pip install xopto\n"
                        "Auto-discovery source: "
                        f"{xopto_source()}\n"
                        "To bypass this check and use surrogate mode, "
                        "re-run with --allow-surrogate-fallback."
                    )
                if verbose:
                    print(
                        f"    [{branch_name}] xopto not available — "
                        f"falling back to surrogate "
                        f"(allow_surrogate_fallback=True)",
                    )

                def _eval_pyxopto_fallback(genome):
                    lab_sim, _, _, _ = run_forward(
                        genome, mode="surrogate",
                    )
                    fitness, comps = compute_fitness(
                        lab_sim, lab_target, mode=fitness_mode,
                    )
                    return fitness, comps, lab_sim

                return _eval_pyxopto_fallback

        # --- MCX branch ---
        elif branch_name == "mcx":
            if check_mcx():
                if verbose:
                    print(
                        f"    [{branch_name}] MCX binary found — "
                        f"using MCX backend (call forward_model with "
                        f"mode='realistic', backend='mcx')",
                    )

                def _eval_mcx(genome):
                    from scripts.optical_ga.forward_model import run_realistic as _rr

                    lab_sim, _, _, _ = _rr(
                        genome,
                        num_photons=num_photons,
                        with_specular=with_specular,
                        backend="mcx",  # force MCX, no xopto routing
                    )
                    fitness, comps = compute_fitness(
                        lab_sim, lab_target, mode=fitness_mode,
                    )
                    return fitness, comps, lab_sim

                return _eval_mcx

            # MCX binary not available
            if (not allow_surrogate_fallback) or require_physical:
                raise RuntimeError(
                    "MCX backend not available.  "
                    "The MCX binary (http://mcx.space/) was not found.\n"
                    "To bypass this check and use surrogate mode, "
                    "re-run with --allow-surrogate-fallback."
                )
            if verbose:
                print(
                    f"    [{branch_name}] MCX not available — "
                    f"falling back to surrogate "
                    f"(allow_surrogate_fallback=True)",
                )

            def _eval_mcx_fallback(genome):
                lab_sim, _, _, _ = run_forward(
                    genome, mode="surrogate",
                )
                fitness, comps = compute_fitness(
                    lab_sim, lab_target, mode=fitness_mode,
                )
                return fitness, comps, lab_sim

            return _eval_mcx_fallback

        else:
            raise ValueError(f"Unknown branch name: {branch_name!r}")

    else:
        raise ValueError(
            f"Unknown forward_mode: {forward_mode!r}. "
            f"Expected 'surrogate' or 'realistic'."
        )


# ---------------------------------------------------------------------------
# Single run helper
# ---------------------------------------------------------------------------


def _run_single_ga(
    seed: int,
    branch_name: str,
    lab_target: Tuple[float, float, float],
    ga_hyperparams: Dict[str, Any],
    batch_settings: Dict[str, Any],
    require_physical: bool,
    allow_surrogate_fallback: bool,
    verbose: bool,
    output_dir: Optional[Path] = None,
) -> BranchSeedResult:
    """Run GA for one branch/seed and return the result."""
    forward_mode = batch_settings.get("forward_mode", "surrogate")
    fitness_mode = batch_settings.get("fitness_mode", "lab")
    num_photons = batch_settings.get("num_photons", 1_000_000)
    with_specular = batch_settings.get("with_specular", True)
    generations = batch_settings.get("generations", 10)
    use_dermal = batch_settings.get("use_dermal_chromophores", False)

    evaluator = _make_branch_evaluator(
        lab_target=lab_target,
        branch_name=branch_name,
        forward_mode=forward_mode,
        fitness_mode=fitness_mode,
        num_photons=num_photons,
        with_specular=with_specular,
        require_physical=require_physical,
        allow_surrogate_fallback=allow_surrogate_fallback,
        verbose=verbose,
    )

    ga = GAOptimiserOptical(
        evaluator=evaluator,
        population_size=ga_hyperparams.get("population_size", 50),
        mutation_rate=ga_hyperparams.get("mutation_rate", 0.2),
        mutation_strength=ga_hyperparams.get("mutation_strength", 0.1),
        elite_fraction=ga_hyperparams.get("elite_fraction", 0.1),
        tournament_size=ga_hyperparams.get("tournament_size", 3),
        seed=seed,
        use_dermal_chromophores=use_dermal,
    )

    t_start = time.perf_counter()
    forward_mode_used = forward_mode

    try:
        ga.run(generations=generations, verbose=False)
        elapsed = time.perf_counter() - t_start

        if ga.best_individual is None:
            return BranchSeedResult(
                seed=seed, branch=branch_name,
                best_fitness=-float("inf"),
                best_lab=(0.0, 0.0, 0.0),
                best_genome={},
                generation_count=generations,
                elapsed_s=elapsed,
                forward_mode_used=forward_mode_used,
                succeeded=False,
                error_msg="GA finished but no best_individual",
            )

        # Save per-run outputs if output_dir is provided
        if output_dir is not None:
            run_dir = output_dir / branch_name / f"seed_{seed:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            ga.save_best_genome(run_dir / "best_genome.json")
            ga.save_history(run_dir / "ga_history.csv")
            ga.save_population(run_dir / "population_final.json")

        return BranchSeedResult(
            seed=seed,
            branch=branch_name,
            best_fitness=ga.best_individual.fitness,
            best_lab=ga.best_individual.lab_sim,
            best_genome=ga.best_individual.genome,
            generation_count=generations,
            elapsed_s=elapsed,
            forward_mode_used=forward_mode_used,
            succeeded=True,
        )

    except Exception as e:
        elapsed = time.perf_counter() - t_start
        err_msg = f"{type(e).__name__}: {e}"
        return BranchSeedResult(
            seed=seed, branch=branch_name,
            best_fitness=-float("inf"),
            best_lab=(0.0, 0.0, 0.0),
            best_genome={},
            generation_count=0,
            elapsed_s=elapsed,
            forward_mode_used=forward_mode_used,
            succeeded=False,
            error_msg=err_msg,
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def compute_aggregate_stats(
    results: List[BranchSeedResult],
) -> Dict[str, AggregateStats]:
    """Compute per-branch aggregate statistics from per-seed results."""
    by_branch: Dict[str, List[BranchSeedResult]] = {}
    for r in results:
        by_branch.setdefault(r.branch, []).append(r)

    agg: Dict[str, AggregateStats] = {}
    for branch, branch_results in by_branch.items():
        succeeded = [r for r in branch_results if r.succeeded]
        fitnesses = [r.best_fitness for r in succeeded] if succeeded else [0.0]

        arr = np.array(fitnesses, dtype=np.float64)
        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        median = float(np.median(arr))
        best_val = float(np.max(arr))
        worst_val = float(np.min(arr))
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))

        # Find seed(s) with best fitness
        best_mask = arr == best_val
        best_seeds = [succeeded[i].seed for i in range(len(succeeded)) if best_mask[i]]

        agg[branch] = AggregateStats(
            branch=branch,
            num_seeds=len(branch_results),
            num_succeeded=len(succeeded),
            best_fitness=best_val,
            worst_fitness=worst_val,
            median_fitness=median,
            mean_fitness=mean_val,
            std_fitness=std_val,
            q25_fitness=q25,
            q75_fitness=q75,
            iqr_fitness=q75 - q25,
            best_fitness_seed=best_seeds[0] if best_seeds else -1,
            best_fitness_seeds=best_seeds,
        )

    return agg


def compute_win_rates(
    results: List[BranchSeedResult],
) -> Dict[str, Any]:
    """Compute head-to-head win/loss/tie rates across seeds.

    Only considers seeds where both branches succeeded.
    """
    by_seed: Dict[int, Dict[str, BranchSeedResult]] = {}
    for r in results:
        by_seed.setdefault(r.seed, {})[r.branch] = r

    branch_names = sorted(
        set(r.branch for r in results)
    )

    if len(branch_names) < 2:
        return {"error": "Need at least two branches for win-rate computation"}

    wins: Dict[str, int] = {b: 0 for b in branch_names}
    ties = 0
    total_comparisons = 0

    for seed, seed_results in by_seed.items():
        # Only compare seeds where all branches succeeded
        if any(
            not seed_results.get(b) or not seed_results[b].succeeded
            for b in branch_names
        ):
            continue

        fitnesses = {b: seed_results[b].best_fitness for b in branch_names}
        total_comparisons += 1

        # Determine winner(s)
        max_fit = max(fitnesses.values())
        winners = [b for b, f in fitnesses.items() if abs(f - max_fit) < 1e-12]

        if len(winners) == len(branch_names):
            ties += 1
        else:
            for w in winners:
                wins[w] += 1

    return {
        "branch_names": branch_names,
        "total_comparisons": total_comparisons,
        "wins": wins,
        "ties": ties,
        "win_rates": {
            b: wins[b] / max(total_comparisons, 1)
            for b in branch_names
        },
        "tie_rate": ties / max(total_comparisons, 1),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_per_seed_metrics_json(
    results: List[BranchSeedResult],
    path: Path,
) -> None:
    """Write per-seed metrics as a JSON array."""
    data = []
    for r in sorted(results, key=lambda x: (x.branch, x.seed)):
        d = {
            "seed": r.seed,
            "branch": r.branch,
            "best_fitness": r.best_fitness,
            "best_lab": list(r.best_lab),
            "generation_count": r.generation_count,
            "elapsed_s": round(r.elapsed_s, 4),
            "forward_mode_used": r.forward_mode_used,
            "succeeded": r.succeeded,
        }
        if r.error_msg:
            d["error_msg"] = r.error_msg
        data.append(d)

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_aggregate_stats_json(
    aggs: Dict[str, AggregateStats],
    win_rates: Dict[str, Any],
    path: Path,
) -> None:
    """Write aggregate stats as JSON."""
    data: Dict[str, Any] = {
        "per_branch": {},
        "win_rates": win_rates,
    }
    for branch, a in aggs.items():
        data["per_branch"][branch] = {
            "num_seeds": a.num_seeds,
            "num_succeeded": a.num_succeeded,
            "best_fitness": a.best_fitness,
            "worst_fitness": a.worst_fitness,
            "median_fitness": a.median_fitness,
            "mean_fitness": a.mean_fitness,
            "std_fitness": a.std_fitness,
            "q25_fitness": a.q25_fitness,
            "q75_fitness": a.q75_fitness,
            "iqr_fitness": a.iqr_fitness,
            "best_fitness_seed": a.best_fitness_seed,
            "best_fitness_seeds": a.best_fitness_seeds,
        }

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_summary_csv(
    results: List[BranchSeedResult],
    path: Path,
) -> None:
    """Write per-seed summary CSV."""
    rows = sorted(results, key=lambda x: (x.branch, x.seed))
    fieldnames = [
        "seed", "branch", "best_fitness", "best_L", "best_a", "best_b",
        "generation_count", "elapsed_s", "forward_mode_used", "succeeded",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "seed": r.seed,
                "branch": r.branch,
                "best_fitness": f"{r.best_fitness:.6f}",
                "best_L": f"{r.best_lab[0]:.4f}",
                "best_a": f"{r.best_lab[1]:.4f}",
                "best_b": f"{r.best_lab[2]:.4f}",
                "generation_count": r.generation_count,
                "elapsed_s": f"{r.elapsed_s:.4f}",
                "forward_mode_used": r.forward_mode_used,
                "succeeded": r.succeeded,
            })


def write_leaderboard_csv(
    results: List[BranchSeedResult],
    path: Path,
) -> None:
    """Write leaderboard CSV — best per-seed results across branches."""
    # Group by seed, take the better of the two branches
    by_seed: Dict[int, List[BranchSeedResult]] = {}
    for r in results:
        by_seed.setdefault(r.seed, []).append(r)

    leaderboard: List[dict] = []
    for seed, seed_results in sorted(by_seed.items()):
        succeeded = [r for r in seed_results if r.succeeded]
        if not succeeded:
            continue
        best = max(succeeded, key=lambda r: r.best_fitness)
        leaderboard.append({
            "seed": best.seed,
            "branch": best.branch,
            "best_fitness": f"{best.best_fitness:.6f}",
            "best_L": f"{best.best_lab[0]:.4f}",
            "best_a": f"{best.best_lab[1]:.4f}",
            "best_b": f"{best.best_lab[2]:.4f}",
            "elapsed_s": f"{best.elapsed_s:.4f}",
        })

    # Sort by best_fitness descending
    leaderboard.sort(
        key=lambda x: float(x["best_fitness"]), reverse=True
    )

    fieldnames = [
        "seed", "branch", "best_fitness", "best_L", "best_a", "best_b",
        "elapsed_s",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(leaderboard)


def write_batch_manifest(
    config: Dict[str, Any],
    aggs: Dict[str, AggregateStats],
    win_rates: Dict[str, Any],
    results: List[BranchSeedResult],
    args: argparse.Namespace,
    path: Path,
) -> None:
    """Write the top-level manifest summarising the batch run."""
    total_elapsed = sum(r.elapsed_s for r in results)
    succeeded = sum(1 for r in results if r.succeeded)

    data: Dict[str, Any] = {
        "batch_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_used": {
            "target_lab": config.get("target_lab"),
            "ga_hyperparams": config.get("ga_hyperparams"),
            "batch": config.get("batch"),
            "branches": config.get("branches"),
        },
        "cli_overrides": {
            "num_seeds": args.num_seeds,
            "generations": args.generations,
            "forward_mode": args.forward_mode,
            "require_physical": args.require_physical,
            "allow_surrogate_fallback": args.allow_surrogate_fallback,
            "seed_offset": args.seed_offset,
        },
        "summary": {
            "total_runs": len(results),
            "succeeded": succeeded,
            "failed": len(results) - succeeded,
            "total_elapsed_s": round(total_elapsed, 2),
            "num_seeds": args.num_seeds,
        },
        "aggregate_stats": {
            branch: {
                "num_seeds": a.num_seeds,
                "num_succeeded": a.num_succeeded,
                "best_fitness": a.best_fitness,
                "median_fitness": a.median_fitness,
                "mean_fitness": a.mean_fitness,
                "std_fitness": a.std_fitness,
                "iqr_fitness": a.iqr_fitness,
                "best_fitness_seed": a.best_fitness_seed,
            }
            for branch, a in aggs.items()
        },
        "win_rates": win_rates,
    }

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Batch-compare MCX vs PyXOpto GA for skin optical parameters.\n\n"
            "Loads a shared-bounds JSON config and runs N seeds for both "
            "branches (MCX and PyXOpto) with identical GA hyper-parameters "
            "and target colour.  Outputs JSON/CSV summaries + leaderboard."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--config", type=str, default="configs/optical_ga_shared_bounds.json",
        help="Path to shared-bounds JSON config (default: "
             "configs/optical_ga_shared_bounds.json)",
    )
    ap.add_argument(
        "--output-dir", type=str, default="outputs/optical_ga/batch_compare",
        help="Output directory (default: outputs/optical_ga/batch_compare)",
    )

    # Batch overrides
    batch_group = ap.add_argument_group("Batch overrides")
    batch_group.add_argument(
        "--num-seeds", type=int, default=None,
        help="Override number of seeds (default: from config)",
    )
    batch_group.add_argument(
        "--generations", type=int, default=None,
        help="Override generations per GA run (default: from config)",
    )
    batch_group.add_argument(
        "--population-size", type=int, default=None,
        help="Override population size (default: from config)",
    )
    batch_group.add_argument(
        "--forward-mode", type=str, default=None,
        choices=["surrogate", "realistic"],
        help="Override forward mode (default: from config)",
    )
    batch_group.add_argument(
        "--fitness-mode", type=str, default=None,
        choices=FITNESS_MODES,
        help="Override fitness mode (default: from config)",
    )
    batch_group.add_argument(
        "--seed-offset", type=int, default=0,
        help="Starting seed offset (default: 0). "
             "Seeds used: offset, offset+1, ..., offset+num_seeds-1",
    )
    batch_group.add_argument(
        "--require-physical", action="store_true",
        help="(deprecated, use --allow-surrogate-fallback instead) "
             "Fail if a physical MC backend is unavailable "
             "(no surrogate fallback)",
    )
    batch_group.add_argument(
        "--allow-surrogate-fallback", action="store_true",
        default=False,
        help="Allow silent fallback to surrogate when the physical MC "
             "backend is unavailable (default: strict — fail with "
             "actionable error when backend is missing). "
             "Only meaningful when --forward-mode=realistic.",
    )

    # Branch toggles
    branch_group = ap.add_argument_group("Branch toggles")
    branch_group.add_argument(
        "--disable-mcx", action="store_true",
        help="Disable MCX branch",
    )
    branch_group.add_argument(
        "--disable-pyxopto", action="store_true",
        help="Disable PyXOpto branch",
    )

    # Other
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print per-run details",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print configuration and exit without running",
    )

    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        config: Dict[str, Any] = json.load(f)

    shared_bounds = config.get("shared_bounds", {})
    target_lab_tuple = tuple(config.get("target_lab", [60.0, 10.0, 15.0]))
    ga_hyperparams = dict(config.get("ga_hyperparams", {}))
    batch_settings = dict(config.get("batch", {}))
    branches_config = config.get("branches", {})

    # Validate shared bounds
    print("=== Optical GA Batch Compare ===", file=sys.stderr)
    print(f"  Config       : {config_path}", file=sys.stderr)
    n_checked = enforce_shared_bounds(shared_bounds)
    print(f"  Bounds validated: {n_checked} parameters", file=sys.stderr)
    n_updated = apply_shared_bounds(shared_bounds)
    print(f"  Bounds applied  : {n_updated} parameters", file=sys.stderr)

    # Apply CLI overrides
    if args.num_seeds is not None:
        batch_settings["num_seeds"] = args.num_seeds
    if args.generations is not None:
        batch_settings["generations"] = args.generations
    if args.population_size is not None:
        ga_hyperparams["population_size"] = args.population_size
    if args.forward_mode is not None:
        batch_settings["forward_mode"] = args.forward_mode
    if args.fitness_mode is not None:
        batch_settings["fitness_mode"] = args.fitness_mode

    num_seeds = batch_settings.get("num_seeds", 10)
    generations = batch_settings.get("generations", 20)
    forward_mode = batch_settings.get("forward_mode", "realistic")
    fitness_mode = batch_settings.get("fitness_mode", "lab")

    # Determine enabled branches
    branch_names: List[str] = []
    for bname, bconf in branches_config.items():
        if bconf.get("enabled", True):
            branch_names.append(bname)

    if args.disable_mcx:
        branch_names = [b for b in branch_names if b != "mcx"]
    if args.disable_pyxopto:
        branch_names = [b for b in branch_names if b != "pyxopto"]

    if not branch_names:
        print("Error: No branches enabled.", file=sys.stderr)
        sys.exit(1)

    # Print run summary (diagnostic info → stderr; keeps stdout machine-parseable)
    print(f"  Target Lab   : {list(target_lab_tuple)}", file=sys.stderr)
    print(f"  Forward mode : {forward_mode}", file=sys.stderr)
    print(f"  Fitness mode : {fitness_mode}", file=sys.stderr)
    print(f"  Generations  : {generations}", file=sys.stderr)
    print(f"  Population   : {ga_hyperparams.get('population_size', 50)}", file=sys.stderr)
    print(f"  Num seeds    : {num_seeds}", file=sys.stderr)
    print(f"  Seed offset  : {args.seed_offset}", file=sys.stderr)
    print(f"  Branches     : {', '.join(branch_names)}", file=sys.stderr)
    print(f"  Require phys : {args.require_physical}", file=sys.stderr)
    print(f"  Allow surrogate fallback: {args.allow_surrogate_fallback}", file=sys.stderr)
    print(file=sys.stderr)

    # Output dir
    output_dir = Path(args.output_dir)
    if args.dry_run:
        print("=== DRY RUN (no execution) ===")
        print(f"  Would write outputs to: {output_dir}/")
        print(f"  Would run {num_seeds} seed(s) × {len(branch_names)} branch(es)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-branch output subdirectories (for individual GA run artifacts)
    for bname in branch_names:
        (output_dir / bname).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Batch loop
    # ------------------------------------------------------------------
    seeds = [args.seed_offset + i for i in range(num_seeds)]
    all_results: List[BranchSeedResult] = []
    seed_iterator = tqdm(
        seeds, desc="Batch seeds",
        unit="seed", ncols=80,
    )

    for seed in seed_iterator:
        if args.verbose:
            print(f"\n  --- Seed {seed} ---")

        for branch_name in branch_names:
            if args.verbose:
                print(f"    [{branch_name}] Starting GA run...")

            result = _run_single_ga(
                seed=seed,
                branch_name=branch_name,
                lab_target=target_lab_tuple,
                ga_hyperparams=ga_hyperparams,
                batch_settings=batch_settings,
                require_physical=args.require_physical,
                allow_surrogate_fallback=args.allow_surrogate_fallback,
                verbose=args.verbose,
                output_dir=output_dir,
            )
            all_results.append(result)

            if args.verbose:
                status = "OK" if result.succeeded else "FAIL"
                fit_str = (
                    f"{result.best_fitness:.4f}"
                    if result.succeeded
                    else result.error_msg[:60]
                )
                print(
                    f"    [{branch_name}] Seed {seed}: {status}  "
                    f"fitness={fit_str}  "
                    f"({result.elapsed_s:.1f}s)"
                )

        # Update tqdm description with latest results
        if not args.verbose:
            # Show live best-fitness summary in progress bar
            last_results = [
                r for r in all_results
                if r.seed == seed and r.succeeded
            ]
            if last_results:
                fits = [r.best_fitness for r in last_results]
                seed_iterator.set_postfix(
                    best=max(fits) if fits else -1,
                )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    print(f"\n  Aggregating results ({len(all_results)} runs)...")

    aggs = compute_aggregate_stats(all_results)
    win_rates = compute_win_rates(all_results)

    # Print per-branch stats
    for branch, a in sorted(aggs.items()):
        print()
        print(f"  [{branch}]")
        print(f"    Succeeded   : {a.num_succeeded} / {a.num_seeds}")
        print(f"    Best        : {a.best_fitness:.6f}  (seed {a.best_fitness_seed})")
        print(f"    Median      : {a.median_fitness:.6f}")
        print(f"    Mean        : {a.mean_fitness:.6f}")
        print(f"    Std         : {a.std_fitness:.6f}")
        print(f"    Q25         : {a.q25_fitness:.6f}")
        print(f"    Q75         : {a.q75_fitness:.6f}")
        print(f"    IQR         : {a.iqr_fitness:.6f}")

    # Print win rates
    print()
    print(f"  Win rates (head-to-head):")
    wr = win_rates
    if "error" not in wr:
        for b in wr.get("branch_names", []):
            wr_pct = wr.get("win_rates", {}).get(b, 0.0) * 100
            print(f"    {b:12s}  wins {wr_pct:5.1f}%")
        tie_pct = wr.get("tie_rate", 0.0) * 100
        print(f"    {'Ties':12s}  {tie_pct:5.1f}%")
        print(f"    (based on {wr.get('total_comparisons', 0)} comparable seeds)")
    else:
        print(f"    {wr['error']}")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    write_per_seed_metrics_json(all_results, output_dir / "per_seed_metrics.json")
    write_aggregate_stats_json(aggs, win_rates, output_dir / "aggregate_stats.json")
    write_summary_csv(all_results, output_dir / "summary.csv")
    write_leaderboard_csv(all_results, output_dir / "leaderboard.csv")
    write_batch_manifest(config, aggs, win_rates, all_results, args,
                         output_dir / "batch_manifest.json")

    print()
    print(f"  Outputs written to: {output_dir}/")
    print(f"    batch_manifest.json")
    print(f"    per_seed_metrics.json")
    print(f"    aggregate_stats.json")
    print(f"    summary.csv")
    print(f"    leaderboard.csv")
    for bname in branch_names:
        n_runs = len([r for r in all_results if r.branch == bname])
        n_ok = len([r for r in all_results if r.branch == bname and r.succeeded])
        print(f"    {bname}/  ({n_ok}/{n_runs} runs succeeded)")


if __name__ == "__main__":
    main()
