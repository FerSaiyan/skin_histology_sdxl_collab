#!/usr/bin/env python
"""
Adapter: run optical GA from a label volume + priors JSON in one step.

Pipeline:
  1. Load label-volume .npy and priors JSON (from ``label_to_optical_priors.py``).
  2. Identify present tissue classes and their voxel counts.
  3. Derive aggregated bounded search ranges from present classes
     (documented strategy — see ``_derive_bounds_from_priors``).
  4. Construct a ``GAOptimiserOptical`` run targeting a user-supplied
     L\\*a\\*b\\* (or defaults).
  5. Seed the initial population within the derived bounds.
  6. Save GA outputs + ``adapter_manifest.json``.

Usage::

    python scripts/optical_ga/run_optical_ga_from_labels.py \\
        --label-npy /tmp/labels.npy \\
        --priors-json /tmp/optical_priors.json \\
        --output-dir /tmp/optical_ga_adapter_run

Smoke test (synthetic fallback)::

    python scripts/optical_ga/run_optical_ga_from_labels.py \\
        --label-npy /tmp/synth_labels.npy \\
        --priors-json /tmp/synth_priors.json \\
        --output-dir /tmp/optical_ga_smoke \\
        --generations 3 --population-size 8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
    OpticalEvaluator,
    Individual,
)
from scripts.optical_ga.genome_encoding_optical import (
    get_param_defs,
    param_names,
    check_genome,
    random_genome,
)
from scripts.optical_ga.optical_fitness import FITNESS_MODES

# ---------------------------------------------------------------------------
# Mapping: priors JSON key stem → genome parameter name
# ---------------------------------------------------------------------------
# The priors JSON has keys like "melanin_min", "melanin_max", etc.
# The genome has keys like "melanin", "blood_layer1", etc.
# For the 19 core parameters the stem matches the genome parameter name.
# These are the parameters that have priors (all 19 core parameters).
_PARAM_STEMS: List[str] = [
    "melanin",
    "blood_layer1",
    "blood_layer2",
    "spo2",
    "g_layer0",
    "g_layer1",
    "g_layer2",
    "d_layer0",
    "d_layer1",
    "amp_layer0",
    "amp_layer1",
    "amp_layer2",
    "water_layer0",
    "water_layer1",
    "water_layer2",
    "fat_layer2",
    "n_mult_layer0",
    "n_mult_layer1",
    "n_mult_layer2",
]


# ---------------------------------------------------------------------------
# Bounds derivation
# ---------------------------------------------------------------------------
# Strategy (documented in every adapter_manifest.json):
#
# Given N present tissue classes, where class i defines per-parameter
# [min_i, max_i] prior ranges:
#
#   For each genome parameter p:
#     derived_min[p] = min(min_1[p], min_2[p], ..., min_N[p])
#     derived_max[p] = max(max_1[p], max_2[p], ..., max_N[p])
#
# Background class (ID 0) is excluded because it represents empty/glass
# rather than real tissue.
#
# If no non-background classes are present, the global genome bounds from
# genome_encoding_optical.py are used as fallback.
# Dermal chromophore parameters (bilirubin, etc.) are NOT covered by priors;
# they keep their genome defaults and are seeded at default values.
# ---------------------------------------------------------------------------


def _load_label_volume(path: str) -> np.ndarray:
    """Load a .npy label volume, converting float to int if needed."""
    p = Path(path)
    if not p.exists():
        print(f"Error: label volume not found: {path}", file=sys.stderr)
        sys.exit(1)
    arr = np.load(str(p))
    if arr.dtype.kind in ("f",):
        arr = arr.astype(np.int32)
    return arr


def _load_priors_json(path: str) -> Dict[str, Any]:
    """Load the priors JSON produced by label_to_optical_priors.py."""
    p = Path(path)
    if not p.exists():
        print(f"Error: priors JSON not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(str(p), "r") as f:
        data = json.load(f)
    # Support both top-level "per_class_priors" and flat structure
    if "per_class_priors" in data:
        return data["per_class_priors"]
    return data


def _exclude_background(
    present_classes: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Remove class ID 0 (background) from the dict."""
    filtered: Dict[str, Dict[str, Any]] = {}
    for k, v in present_classes.items():
        try:
            if int(k) != 0:
                filtered[k] = v
        except (TypeError, ValueError):
            # Ignore non-class keys (e.g. malformed metadata entries).
            continue
    return filtered


def _derive_bounds_from_priors(
    present_classes: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """Derive aggregated min/max bounds from present-class priors.

    Returns
    -------
    derived_min : dict
        Lower bound per genome parameter name.
    derived_max : dict
        Upper bound per genome parameter name.
    derivation_log : dict
        Human-readable log of the derivation strategy and per-parameter ranges.
    """
    non_bg = _exclude_background(present_classes)

    class_ids_sorted = sorted(int(k) for k in non_bg.keys())

    derivation_log: Dict[str, Any] = {
        "strategy": (
            "For each parameter, derived_min = min(class_min_i) across present "
            "non-background classes; derived_max = max(class_max_i) across present "
            "non-background classes."
        ),
        "num_non_background_classes": len(non_bg),
        "non_background_class_ids": class_ids_sorted,
        "excluded_background": True,
        "fallback_to_global_if_empty": True,
    }

    derived_min: Dict[str, float] = {}
    derived_max: Dict[str, float] = {}

    # Fallback: use global genome encoding bounds
    global_bounds: Dict[str, Tuple[float, float]] = {
        name: (low, high)
        for name, low, high, _, _ in get_param_defs(use_dermal_chromophores=False)
    }

    if not non_bg:
        for stem in _PARAM_STEMS:
            if stem in global_bounds:
                low, high = global_bounds[stem]
                derived_min[stem] = low
                derived_max[stem] = high
        derivation_log["fallback_triggered"] = (
            "No non-background classes present; used global genome bounds "
            "from genome_encoding_optical.py."
        )
        # Add per-parameter ranges to log
        derivation_log["per_parameter"] = {
            p: {"derived_min": derived_min[p], "derived_max": derived_max[p],
                "global_low": global_bounds.get(p, (0, 1))[0],
                "global_high": global_bounds.get(p, (0, 1))[1]}
            for p in _PARAM_STEMS
        }
        return derived_min, derived_max, derivation_log

    # Collect per-class mins and maxes for each parameter stem
    param_mins: Dict[str, List[float]] = {s: [] for s in _PARAM_STEMS}
    param_maxes: Dict[str, List[float]] = {s: [] for s in _PARAM_STEMS}

    for ckey, entry in non_bg.items():
        priors = entry.get("priors", {})
        for stem in _PARAM_STEMS:
            min_key = f"{stem}_min"
            max_key = f"{stem}_max"
            if min_key in priors:
                param_mins[stem].append(float(priors[min_key]))
            if max_key in priors:
                param_maxes[stem].append(float(priors[max_key]))

    # Derive bounds, falling back to global bounds if a parameter has no data
    per_param_log: Dict[str, Dict[str, Any]] = {}
    for stem in _PARAM_STEMS:
        low, high = global_bounds.get(stem, (0.0, 1.0))

        if param_mins[stem]:
            d_min = min(param_mins[stem])
        else:
            d_min = low

        if param_maxes[stem]:
            d_max = max(param_maxes[stem])
        else:
            d_max = high

        # Clamp to global bounds
        d_min = max(low, min(high, d_min))
        d_max = max(low, min(high, d_max))

        # Ensure d_min <= d_max
        if d_min > d_max:
            d_min, d_max = low, high

        derived_min[stem] = d_min
        derived_max[stem] = d_max

        per_param_log[stem] = {
            "derived_min": d_min,
            "derived_max": d_max,
            "global_low": low,
            "global_high": high,
            "num_class_contributors": len(param_mins[stem]),
        }

    derivation_log["per_parameter"] = per_param_log
    return derived_min, derived_max, derivation_log


# ---------------------------------------------------------------------------
# Seed generation within derived bounds
# ---------------------------------------------------------------------------


def _generate_seed_genomes(
    derived_min: Dict[str, float],
    derived_max: Dict[str, float],
    num_seeds: int,
    rng: random.Random,
    use_dermal_chromophores: bool = False,
) -> List[Dict[str, float]]:
    """Generate random seed genomes within derived bounds.

    Parameters not covered by derived bounds (dermal chromophores) are
    set to their default values from the genome encoding.
    """
    defs = get_param_defs(use_dermal_chromophores)
    seeds: List[Dict[str, float]] = []

    for _ in range(num_seeds):
        genome: Dict[str, float] = {}
        for name, low, high, default, _ in defs:
            if name in derived_min and name in derived_max:
                dl = derived_min[name]
                dh = derived_max[name]
                # Clamp to global bounds (already done in derivation, but be safe)
                dl = max(low, min(high, dl))
                dh = max(low, min(high, dh))
                if dl >= dh:
                    genome[name] = dl
                else:
                    genome[name] = rng.uniform(dl, dh)
            else:
                # Dermal chromophore not in priors → start at default
                genome[name] = default

        seeds.append(genome)

    return seeds


# ---------------------------------------------------------------------------
# Present-class helpers
# ---------------------------------------------------------------------------


def _identify_present_classes(
    label_vol: np.ndarray,
    priors_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Filter priors data to only classes that actually appear in the volume.

    Returns a dict keyed by string class ID, with the per-class priors entry
    plus a ``count_voxels`` field.
    """
    unique = np.unique(label_vol)
    present_ids = set(int(u) for u in unique)

    present: Dict[str, Dict[str, Any]] = {}
    for ckey, entry in priors_data.items():
        try:
            cid = int(ckey)
        except (TypeError, ValueError):
            # Ignore non-class keys (e.g. metadata blocks).
            continue
        if cid in present_ids:
            # Count voxels
            count = int(np.sum(label_vol == cid))
            entry_copy = dict(entry)
            entry_copy["count_voxels"] = count
            entry_copy["class_id"] = cid
            present[ckey] = entry_copy

    # Also log class IDs in the volume that are NOT in priors_data
    for cid in sorted(present_ids):
        ckey = str(cid)
        if ckey not in present:
            present[ckey] = {
                "class_id": cid,
                "class_name": f"unknown_class_{cid}",
                "count_voxels": int(np.sum(label_vol == cid)),
                "warning": "Class ID not found in priors data",
            }

    return present


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def _build_manifest(
    label_npy: str,
    priors_json: str,
    output_dir: str,
    present_classes: Dict[str, Dict[str, Any]],
    derived_min: Dict[str, float],
    derived_max: Dict[str, float],
    derivation_log: Dict[str, Any],
    ga_args: Dict[str, Any],
    ga_outputs: Dict[str, str],
    elapsed_s: float,
) -> Dict[str, Any]:
    """Build the adapter manifest dict."""
    return {
        "manifest_type": "optical_ga_adapter_manifest",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "label_npy": str(Path(label_npy).resolve()),
            "priors_json": str(Path(priors_json).resolve()),
        },
        "output_dir": str(Path(output_dir).resolve()),
        "present_classes": {
            ckey: {
                "class_id": entry.get("class_id", int(ckey)),
                "class_name": entry.get("class_name", f"class_{ckey}"),
                "count_voxels": entry.get("count_voxels", 0),
            }
            for ckey, entry in sorted(present_classes.items(), key=lambda x: int(x[0]))
        },
        "bounds_derivation": derivation_log,
        "ga_configuration": {
            k: v for k, v in ga_args.items()
        },
        "ga_outputs": ga_outputs,
        "elapsed_seconds": round(elapsed_s, 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run optical GA from a label volume + priors JSON.\n\n"
            "Reads a class-ID label volume (.npy) and its per-class optical "
            "priors JSON (from label_to_optical_priors.py), derives aggregated "
            "search bounds, seeds the GA, and saves outputs plus an "
            "adapter_manifest.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--label-npy", required=True,
        help="Path to class-ID label volume .npy (from build_histoseg_label_volume.py)",
    )
    ap.add_argument(
        "--priors-json", required=True,
        help="Path to per-class priors JSON (from label_to_optical_priors.py)",
    )
    ap.add_argument(
        "--output-dir", required=True,
        help="Output directory for GA outputs and adapter manifest",
    )

    # Target colour
    target_group = ap.add_argument_group("Target colour (optional)")
    target_group.add_argument(
        "--target-L", type=float, default=60.0,
        help="Target CIE L* (default: 60.0)",
    )
    target_group.add_argument(
        "--target-a", type=float, default=10.0,
        help="Target CIE a* (default: 10.0)",
    )
    target_group.add_argument(
        "--target-b", type=float, default=15.0,
        help="Target CIE b* (default: 15.0)",
    )

    # GA parameters
    ga_group = ap.add_argument_group("GA settings (smoke defaults)")
    ga_group.add_argument(
        "--generations", type=int, default=3,
        help="Number of GA generations (default: 3)",
    )
    ga_group.add_argument(
        "--population-size", type=int, default=8,
        help="Population size per generation (default: 8)",
    )
    ga_group.add_argument(
        "--mutation-rate", type=float, default=0.2,
        help="Per-parameter mutation probability (default: 0.2)",
    )
    ga_group.add_argument(
        "--mutation-strength", type=float, default=0.1,
        help="Mutation stddev in normalised space (default: 0.1)",
    )
    ga_group.add_argument(
        "--elite-fraction", type=float, default=0.1,
        help="Fraction of top individuals preserved (default: 0.1)",
    )
    ga_group.add_argument(
        "--tournament-size", type=int, default=3,
        help="Tournament selection size (default: 3)",
    )
    ga_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    # Mode selection
    mode_group = ap.add_argument_group("Mode selection")
    mode_group.add_argument(
        "--forward-mode", type=str, default="surrogate",
        choices=["surrogate", "realistic"],
        help="Forward model mode (default: surrogate)",
    )
    mode_group.add_argument(
        "--fitness-mode", type=str, default="lab",
        choices=FITNESS_MODES,
        help="Fitness mode (default: lab)",
    )
    mode_group.add_argument(
        "--use-dermal-chromophores", action="store_true",
        help="Include 4 optional dermal chromophores (bilirubin, etc.)",
    )

    # Output control
    output_group = ap.add_argument_group("Output control")
    output_group.add_argument(
        "--verbose", action="store_true",
        help="Print per-generation details",
    )
    output_group.add_argument(
        "--dry-run", action="store_true",
        help="Print configuration and exit without running GA",
    )

    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_npy = args.label_npy
    priors_json = args.priors_json
    lab_target = (args.target_L, args.target_a, args.target_b)

    # ---- 1. Load inputs ---------------------------------------------------
    print("=== Optical GA Adapter ===")
    print(f"  Label volume  : {label_npy}")
    print(f"  Priors JSON   : {priors_json}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Target Lab    : {lab_target}")
    print(f"  Forward mode  : {args.forward_mode}")
    print(f"  Fitness mode  : {args.fitness_mode}")
    print()

    label_vol = _load_label_volume(label_npy)
    print(f"  Label shape   : {label_vol.shape}  dtype={label_vol.dtype}")

    priors_data = _load_priors_json(priors_json)
    print(f"  Priors classes: {len(priors_data)}")

    # ---- 2. Identify present classes --------------------------------------
    present_classes = _identify_present_classes(label_vol, priors_data)
    print(f"  Present in vol: {len(present_classes)} class(es)")
    for ckey in sorted(present_classes.keys(), key=int):
        entry = present_classes[ckey]
        print(f"    Class {ckey} ({entry.get('class_name', '?')}): "
              f"{entry.get('count_voxels', 0)} voxels")

    # ---- 3. Derive bounds -------------------------------------------------
    derived_min, derived_max, derivation_log = _derive_bounds_from_priors(
        present_classes,
    )
    print(f"\n  Derived bounds for {len(derived_min)} parameters "
          f"(non-background classes only).")
    if args.verbose:
        for p in sorted(derived_min.keys()):
            print(f"    {p}: [{derived_min[p]:.6e}, {derived_max[p]:.6e}]")

    # ---- 4. Construct GA config -------------------------------------------
    rng = random.Random(args.seed)

    # Generate seed genomes within derived bounds
    num_seeds = max(1, args.population_size // 2)
    seed_genomes = _generate_seed_genomes(
        derived_min,
        derived_max,
        num_seeds=num_seeds,
        rng=rng,
        use_dermal_chromophores=args.use_dermal_chromophores,
    )

    ga_config = {
        "label_npy": label_npy,
        "priors_json": priors_json,
        "lab_target": list(lab_target),
        "generations": args.generations,
        "population_size": args.population_size,
        "mutation_rate": args.mutation_rate,
        "mutation_strength": args.mutation_strength,
        "elite_fraction": args.elite_fraction,
        "tournament_size": args.tournament_size,
        "seed": args.seed,
        "forward_mode": args.forward_mode,
        "fitness_mode": args.fitness_mode,
        "use_dermal_chromophores": args.use_dermal_chromophores,
        "num_seed_genomes": num_seeds,
    }

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print("GA configuration:")
        print(json.dumps(ga_config, indent=2))
        print(f"\nSeed genomes: {num_seeds}")
        if args.verbose:
            for i, g in enumerate(seed_genomes):
                print(f"  Seed {i}: {g}")
        print("\nDerivation log:")
        print(json.dumps(derivation_log, indent=2))
        return

    # ---- 5. Run GA --------------------------------------------------------
    evaluator = OpticalEvaluator(
        lab_target=lab_target,
        forward_mode=args.forward_mode,
        fitness_mode=args.fitness_mode,
        verbose=args.verbose,
    )

    ga = GAOptimiserOptical(
        evaluator=evaluator,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        elite_fraction=args.elite_fraction,
        tournament_size=args.tournament_size,
        seed=args.seed,
        use_dermal_chromophores=args.use_dermal_chromophores,
    )

    t_start = time.perf_counter()
    ga.run(
        generations=args.generations,
        verbose=args.verbose,
        seed_genomes=seed_genomes,
    )
    elapsed = time.perf_counter() - t_start

    # ---- 6. Save outputs --------------------------------------------------
    ga.save_best_genome(output_dir / "best_genome.json")
    ga.save_history(output_dir / "ga_history.csv")
    ga.save_population(output_dir / "population_final.json")

    ga_outputs = {
        "best_genome": str((output_dir / "best_genome.json").resolve()),
        "ga_history": str((output_dir / "ga_history.csv").resolve()),
        "population_final": str((output_dir / "population_final.json").resolve()),
    }

    print(f"\n  GA outputs saved to: {output_dir}/")
    for label, path in ga_outputs.items():
        print(f"    {label}: {path}")

    # ---- 7. Write adapter manifest ----------------------------------------
    manifest = _build_manifest(
        label_npy=label_npy,
        priors_json=priors_json,
        output_dir=str(output_dir),
        present_classes=present_classes,
        derived_min=derived_min,
        derived_max=derived_max,
        derivation_log=derivation_log,
        ga_args=ga_config,
        ga_outputs=ga_outputs,
        elapsed_s=elapsed,
    )

    manifest_path = output_dir / "adapter_manifest.json"
    with open(str(manifest_path), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  adapter_manifest.json: {manifest_path}")

    # ---- 8. Summary -------------------------------------------------------
    if ga.best_individual is not None:
        print(f"\n=== Summary ===")
        print(f"  Best fitness : {ga.best_individual.fitness:.4f}")
        print(f"  Best Lab     : {ga.best_individual.lab_sim}")
        print(f"  Elapsed      : {elapsed:.2f}s")

    # Validate manifest was written correctly by re-reading
    try:
        with open(str(manifest_path), "r") as f:
            check = json.load(f)
        assert "present_classes" in check
        assert "bounds_derivation" in check
        assert "ga_configuration" in check
        assert "ga_outputs" in check
        print(f"  Manifest OK  : {manifest_path}")
    except Exception as e:
        print(f"  WARNING: manifest validation failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
