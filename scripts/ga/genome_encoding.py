#!/usr/bin/env python
"""
Genome encoding for SDXL inpainting hyperparameters.

Provides parameter bounds and encode/decode helpers so that the GA
can operate on a normalised real-valued vector and project back to
the original parameter space.

Current parameters (Phase A, geometry-only):
    - mask_cx          (0.0 .. 1.0)       – normalised mask centre x
    - mask_cy          (0.0 .. 1.0)       – normalised mask centre y
    - mask_radius      (0.05 .. 0.35)     – mask radius (fraction of image)
    - strength         (0.2 .. 0.8)       – denoising strength
    - guidance_scale   (3.0 .. 12.0)      – classifier-free guidance scale
    - seed             (0 .. 2**32 - 1)   – random seed (integer)
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parameter registry
# ---------------------------------------------------------------------------
# Each entry: (name, low, high, is_int, default)
# For continuous params is_int=False and the GA may sample any float in [low, high].
# For seed (is_int=True) the GA still works in normalised space but decode
# maps to an integer via round + clamp.

PARAM_DEFS: List[Tuple[str, float, float, bool, float]] = [
    ("mask_cx", 0.0, 1.0, False, 0.5),
    ("mask_cy", 0.0, 1.0, False, 0.5),
    ("mask_radius", 0.05, 0.35, False, 0.15),
    ("strength", 0.2, 0.8, False, 0.55),
    ("guidance_scale", 3.0, 12.0, False, 7.0),
    ("seed", 0.0, float(2**32 - 1), True, 42.0),
]

NUM_PARAMS = len(PARAM_DEFS)
PARAM_NAMES = [p[0] for p in PARAM_DEFS]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def param_bounds(name: str) -> Tuple[float, float]:
    """Return (low, high) for a given parameter name."""
    for pname, low, high, _, _ in PARAM_DEFS:
        if pname == name:
            return low, high
    raise KeyError(f"Unknown parameter: {name}")


def is_int_param(name: str) -> bool:
    """Return True if the parameter should be treated as integer."""
    for pname, _, _, is_int, _ in PARAM_DEFS:
        if pname == name:
            return bool(is_int)
    raise KeyError(f"Unknown parameter: {name}")


# ---------------------------------------------------------------------------
# Normalised vector ↔ genome dict conversion
# ---------------------------------------------------------------------------

def normalised_vector_from_genome(genome: Dict[str, float]) -> np.ndarray:
    """Convert a genome dict to a [0,1]^D normalised vector."""
    vec = np.empty(NUM_PARAMS, dtype=np.float64)
    for i, (name, low, high, _, _) in enumerate(PARAM_DEFS):
        raw = genome[name]
        # clip to bounds
        raw = max(low, min(high, raw))
        # normalise
        if high - low > 1e-12:
            vec[i] = (raw - low) / (high - low)
        else:
            vec[i] = 0.5
    return vec


def genome_from_normalised_vector(vec: np.ndarray) -> Dict[str, float]:
    """Convert a normalised vector back to a genome dict with denormalised values."""
    genome: Dict[str, float] = {}
    for i, (name, low, high, is_int, _) in enumerate(PARAM_DEFS):
        v = float(vec[i])
        v = max(0.0, min(1.0, v))
        raw = low + v * (high - low)
        if is_int:
            raw = round(raw)
        genome[name] = raw
    return genome


# ---------------------------------------------------------------------------
# Default genome
# ---------------------------------------------------------------------------

def default_genome() -> Dict[str, float]:
    """Return a genome dict with default values."""
    return {p[0]: p[4] for p in PARAM_DEFS}


# ---------------------------------------------------------------------------
# Random genome generation
# ---------------------------------------------------------------------------

def random_genome(rng: random.Random) -> Dict[str, float]:
    """Generate a random genome using a supplied Random instance (deterministic)."""
    vec = np.array([rng.uniform(0.0, 1.0) for _ in range(NUM_PARAMS)], dtype=np.float64)
    return genome_from_normalised_vector(vec)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def genome_to_str(genome: Dict[str, float], precision: int = 4) -> str:
    """Return a compact string representation."""
    parts = []
    for name, low, high, is_int, _ in PARAM_DEFS:
        val = genome[name]
        if is_int:
            parts.append(f"{name}={int(val)}")
        else:
            parts.append(f"{name}={val:.{precision}f}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Bounds check
# ---------------------------------------------------------------------------

def check_genome(genome: Dict[str, float]) -> List[str]:
    """Return a list of violation messages (empty if valid)."""
    issues: List[str] = []
    for name, low, high, _, _ in PARAM_DEFS:
        val = genome.get(name)
        if val is None:
            issues.append(f"Missing parameter: {name}")
            continue
        if val < low or val > high:
            issues.append(f"{name}={val} out of bounds [{low}, {high}]")
    return issues


# ---------------------------------------------------------------------------
# Mutation helpers (used by GA)
# ---------------------------------------------------------------------------

def mutate_genome(
    genome: Dict[str, float],
    mutation_rate: float,
    mutation_strength: float,
    rng: random.Random,
) -> Dict[str, float]:
    """Return a mutated copy of the genome.

    Each normalised parameter is perturbed with probability *mutation_rate*
    by adding Gaussian noise scaled by *mutation_strength*.
    """
    vec = normalised_vector_from_genome(genome)
    for i in range(NUM_PARAMS):
        if rng.random() < mutation_rate:
            vec[i] += rng.gauss(0.0, mutation_strength)
            vec[i] = max(0.0, min(1.0, vec[i]))
    return genome_from_normalised_vector(vec)


def crossover_genomes(
    parent_a: Dict[str, float],
    parent_b: Dict[str, float],
    rng: random.Random,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Uniform crossover: each parameter is swapped with probability 0.5.

    Returns two children.
    """
    va = normalised_vector_from_genome(parent_a)
    vb = normalised_vector_from_genome(parent_b)
    for i in range(NUM_PARAMS):
        if rng.random() < 0.5:
            va[i], vb[i] = vb[i], va[i]
    return genome_from_normalised_vector(va), genome_from_normalised_vector(vb)
