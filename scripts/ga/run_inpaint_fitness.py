#!/usr/bin/env python
"""
Classifier-free fitness evaluation for inpainting hyperparameters.

This module provides a weighted objective over available numeric metrics
from the QC / coherence pipeline.  It does NOT require a classifier;
it only needs a JSON file with pre-computed metric values (or a mock).

Metrics used (all are geometry/QC-based and classifier-free):
    - seam_quality          (lower is better)
    - non_roi_drift         (lower is better)
    - histogram_similarity  (higher is better)
    - adjacent_ssim         (higher is better, from Z-coherence)
    - z_gradient_smoothness (lower is better)

The fitness is computed as a weighted sum of normalised scores.
Missing fields are handled gracefully (weight redistributed or dropped).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Metric specification
# ---------------------------------------------------------------------------
# Each entry: (name, direction, weight, default)
#   direction:  "lower_better" or "higher_better"
#   weight:     relative importance in the weighted sum
#   default:    fallback if field is missing (None = skip)

METRIC_SPECS: List[Tuple[str, str, float, Optional[float]]] = [
    ("seam_quality", "lower_better", 0.20, None),
    ("non_roi_drift", "lower_better", 0.15, None),
    ("histogram_similarity", "higher_better", 0.20, None),
    ("adjacent_ssim", "higher_better", 0.25, None),
    ("z_gradient_smoothness", "lower_better", 0.20, None),
]

# Reference ranges for min-max normalisation.
# These are domain heuristics; they can be overridden with observed ranges.
# Format: (expected_min, expected_max)
_REF_RANGES: Dict[str, Tuple[float, float]] = {
    "seam_quality": (0.0, 20.0),          # 0 = perfect seam, >20 = bad
    "non_roi_drift": (0.0, 10.0),         # 0 = no drift, >10 = large drift
    "histogram_similarity": (0.0, 1.0),   # 0 = no correlation, 1 = identical
    "adjacent_ssim": (0.0, 1.0),          # SSIM range
    "z_gradient_smoothness": (0.0, 1.0),  # 0 = very smooth, 1 = very rough
}


# ---------------------------------------------------------------------------
# Fitness computation
# ---------------------------------------------------------------------------

def compute_fitness(
    metrics: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[float, Dict[str, float], List[str]]:
    """Compute a weighted fitness score from metric dict.

    Args:
        metrics: Dict of metric_name -> numeric value.
        verbose: If True, prints per-metric contributions.

    Returns:
        (total_fitness, component_scores, warnings)
    """
    component_scores: Dict[str, float] = {}
    warnings: List[str] = []
    weighted_sum = 0.0
    total_weight = 0.0

    for name, direction, weight, default_val in METRIC_SPECS:
        raw = metrics.get(name, default_val)
        if raw is None:
            warnings.append(f"Metric '{name}' not found; skipping.")
            continue
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            warnings.append(f"Metric '{name}' has non-numeric value {raw!r}; skipping.")
            continue

        # Check for NaN / inf
        if not np.isfinite(raw):
            warnings.append(f"Metric '{name}' is {raw}; skipping.")
            continue

        low, high = _REF_RANGES.get(name, (0.0, 1.0))
        if high - low < 1e-12:
            normalised = 1.0
        else:
            normalised = (raw - low) / (high - low)
            normalised = max(0.0, min(1.0, normalised))

        # Invert if lower_is_better
        if direction == "lower_better":
            score = 1.0 - normalised
        else:
            score = normalised

        component_scores[f"raw_{name}"] = raw
        component_scores[f"score_{name}"] = score
        weighted_sum += weight * score
        total_weight += weight

        if verbose:
            print(f"  {name}: raw={raw:.4f}, norm={normalised:.4f}, score={score:.4f}, weight={weight:.2f}")

    if total_weight > 0.0:
        total_fitness = weighted_sum / total_weight
    else:
        total_fitness = 0.0
        warnings.append("No valid metrics found; fitness = 0.")

    if verbose:
        print(f"  -> total_fitness = {total_fitness:.4f}")

    return total_fitness, component_scores, warnings


# ---------------------------------------------------------------------------
# Fitness from JSON file
# ---------------------------------------------------------------------------

def fitness_from_json(
    json_path: Path,
    verbose: bool = False,
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    """Load metrics from a JSON file and compute fitness.

    The JSON can be either a flat dict of metric->value, or a list of
    such dicts (in which case the mean fitness across all entries is returned).

    Returns:
        (fitness, component_scores, raw_metrics)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        # Multiple evaluation entries — average fitness
        fitnesses: List[float] = []
        comp_list: List[Dict[str, float]] = []
        for entry in data:
            f, comps, warns = compute_fitness(entry, verbose=verbose)
            fitnesses.append(f)
            comp_list.append(comps)
        avg_fitness = float(np.mean(fitnesses)) if fitnesses else 0.0
        # Use last component scores as representative
        return avg_fitness, comp_list[-1] if comp_list else {}, data
    elif isinstance(data, dict):
        fitness, comps, warns = compute_fitness(data, verbose=verbose)
        return fitness, comps, data
    else:
        raise ValueError(f"Unexpected JSON structure in {json_path}")


# ---------------------------------------------------------------------------
# Mock metrics generator (for offline / smoke testing)
# ---------------------------------------------------------------------------

def mock_metrics(
    genome: Dict[str, float],
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Generate synthetic metrics for testing.

    The mock creates plausible metric values that correlate loosely
    with the genome parameters so the GA has a signal to optimise.

    Specifically:
      - strength near 0.5 → lower seam_quality (good)
      - guidance_scale near 7 → higher adjacent_SSIM
      - mask_radius moderate (0.15-0.2) → better histogram_similarity
    """
    if rng is None:
        rng = np.random.default_rng(42)

    strength = genome.get("strength", 0.55)
    guidance = genome.get("guidance_scale", 7.0)
    radius = genome.get("mask_radius", 0.15)

    # Optimal regions
    seam_base = 5.0 * abs(strength - 0.5) + 2.0
    ssim_base = 0.85 - 0.02 * abs(guidance - 7.0)
    hist_base = 0.90 - 0.3 * abs(radius - 0.18)

    metrics = {
        "seam_quality": float(seam_base + rng.normal(0, 0.5)),
        "non_roi_drift": float(2.0 + 3.0 * abs(strength - 0.45) + rng.normal(0, 0.3)),
        "histogram_similarity": float(max(0.0, min(1.0, hist_base + rng.normal(0, 0.02)))),
        "adjacent_ssim": float(max(0.0, min(1.0, ssim_base + rng.normal(0, 0.01)))),
        "z_gradient_smoothness": float(0.2 + 0.3 * abs(radius - 0.2) + rng.normal(0, 0.05)),
    }
    return metrics
