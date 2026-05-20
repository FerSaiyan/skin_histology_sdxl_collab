#!/usr/bin/env python
"""
Genome encoding for skin biophysical optical parameters.

Defines 19 core parameters (and 4 optional dermal chromophores) with
bounds and defaults aligned to the friend's reference script
(``external_refs/friend_optical_ga/find_ita_graph_all.py``).

The genome operates on a normalised real-valued vector in [0, 1]^D for use
by the GA; decode projects back to the original parameter space.

Wavelength handling: 380–780 nm with 5 nm step (81 wavelengths).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Wavelengths
# ---------------------------------------------------------------------------

WAVELENGTHS: List[int] = list(range(380, 781, 5))  # 380–780 nm, 5 nm step
NUM_WAVELENGTHS: int = len(WAVELENGTHS)  # 81

# ---------------------------------------------------------------------------
# Parameter definitions
# ---------------------------------------------------------------------------
# Each entry: (name, low, high, default, description)
# The 19 core parameters are always used.
# 4 optional dermal chromophores (bilirubin, biliverdin, COHb, metHb) can be
# appended when ``use_dermal_chromophores=True``.

CORE_PARAM_DEFS: List[Tuple[str, float, float, float, str]] = [
    ("melanin",            0.0,       5e-1,     0.1,       "Melanin concentration in epidermis"),
    ("blood_layer1",       0.0,       1e-1,     0.02,      "Blood volume fraction in dermis"),
    ("blood_layer2",       0.0,       1e-1,     0.02,      "Blood volume fraction in subcutis"),
    ("spo2",               0.50,      1.00,     0.75,      "Oxygen saturation"),
    ("g_layer0",           0.85,      0.95,     0.90,      "Anisotropy g for epidermis"),
    ("g_layer1",           0.85,      0.95,     0.90,      "Anisotropy g for dermis"),
    ("g_layer2",           0.85,      0.95,     0.90,      "Anisotropy g for subcutis"),
    ("d_layer0",           3e-5,      1.5e-4,   8e-5,      "Epidermis thickness (m)"),
    ("d_layer1",           0.1e-3,    0.5e-3,   0.3e-3,    "Dermis thickness (m)"),
    ("amp_layer0",         0.5,       1.5,      1.0,       "Scattering amplitude multiplier (epidermis)"),
    ("amp_layer1",         0.5,       1.5,      1.0,       "Scattering amplitude multiplier (dermis)"),
    ("amp_layer2",         0.5,       1.5,      1.0,       "Scattering amplitude multiplier (subcutis)"),
    ("water_layer0",       0.01,      0.8,      0.65,      "Water content (epidermis)"),
    ("water_layer1",       0.01,      0.8,      0.7,       "Water content (dermis)"),
    ("water_layer2",       0.01,      0.8,      0.7,       "Water content (subcutis)"),
    ("fat_layer2",         0.05,      0.8,      0.2,       "Fat content (subcutis)"),
    ("n_mult_layer0",      0.8,       1.2,      1.0,       "Refractive index multiplier (epidermis)"),
    ("n_mult_layer1",      0.8,       1.2,      1.0,       "Refractive index multiplier (dermis)"),
    ("n_mult_layer2",      0.8,       1.2,      1.0,       "Refractive index multiplier (subcutis)"),
]

DERMAL_CHROMOPHORE_DEFS: List[Tuple[str, float, float, float, str]] = [
    ("bilirubin_layer1",   0.0,       0.01,     0.0,       "Bilirubin concentration in dermis (optional)"),
    ("biliverdin_layer1",  0.0,       0.01,     0.0,       "Biliverdin concentration in dermis (optional)"),
    ("cohb_layer1",        0.0,       0.01,     0.0,       "Carboxyhemoglobin in dermis (optional)"),
    ("methb_layer1",       0.0,       0.01,     0.0,       "Methemoglobin in dermis (optional)"),
]

_PARAM_NAMES_CORE: List[str] = [p[0] for p in CORE_PARAM_DEFS]
_PARAM_NAMES_DERMAL: List[str] = [p[0] for p in DERMAL_CHROMOPHORE_DEFS]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_param_defs(
    use_dermal_chromophores: bool = False,
) -> List[Tuple[str, float, float, float, str]]:
    """Return the full list of parameter definitions."""
    defs = list(CORE_PARAM_DEFS)
    if use_dermal_chromophores:
        defs.extend(DERMAL_CHROMOPHORE_DEFS)
    return defs


def param_names(
    use_dermal_chromophores: bool = False,
) -> List[str]:
    """Return list of parameter names."""
    names = list(_PARAM_NAMES_CORE)
    if use_dermal_chromophores:
        names.extend(_PARAM_NAMES_DERMAL)
    return names


def param_bounds(
    name: str,
    use_dermal_chromophores: bool = False,
) -> Tuple[float, float]:
    """Return (low, high) for a given parameter name."""
    for pname, low, high, _, _ in get_param_defs(use_dermal_chromophores):
        if pname == name:
            return low, high
    raise KeyError(f"Unknown parameter: {name}")


def num_params(use_dermal_chromophores: bool = False) -> int:
    """Return the dimension of the genome vector."""
    return len(get_param_defs(use_dermal_chromophores))


# ---------------------------------------------------------------------------
# Normalised vector ↔ genome dict conversion
# ---------------------------------------------------------------------------

def normalised_vector_from_genome(
    genome: Dict[str, float],
    use_dermal_chromophores: bool = False,
) -> np.ndarray:
    """Convert a genome dict to a [0, 1]^D normalised vector."""
    defs = get_param_defs(use_dermal_chromophores)
    vec = np.empty(len(defs), dtype=np.float64)
    for i, (name, low, high, _, _) in enumerate(defs):
        raw = genome.get(name, (low + high) / 2)
        raw = max(low, min(high, raw))
        if high - low > 1e-12:
            vec[i] = (raw - low) / (high - low)
        else:
            vec[i] = 0.5
    return vec


def genome_from_normalised_vector(
    vec: np.ndarray,
    use_dermal_chromophores: bool = False,
) -> Dict[str, float]:
    """Convert a normalised vector back to a genome dict."""
    defs = get_param_defs(use_dermal_chromophores)
    genome: Dict[str, float] = {}
    for i, (name, low, high, _, _) in enumerate(defs):
        v = float(vec[i]) if i < len(vec) else 0.5
        v = max(0.0, min(1.0, v))
        genome[name] = low + v * (high - low)
    return genome


# ---------------------------------------------------------------------------
# Default genome
# ---------------------------------------------------------------------------

def default_genome(
    use_dermal_chromophores: bool = False,
) -> Dict[str, float]:
    """Return a genome dict with default values."""
    defs = get_param_defs(use_dermal_chromophores)
    return {p[0]: p[3] for p in defs}


# ---------------------------------------------------------------------------
# Random genome generation
# ---------------------------------------------------------------------------

def random_genome(
    rng: random.Random,
    use_dermal_chromophores: bool = False,
) -> Dict[str, float]:
    """Generate a random genome using a supplied Random instance."""
    n = num_params(use_dermal_chromophores)
    vec = np.array([rng.uniform(0.0, 1.0) for _ in range(n)], dtype=np.float64)
    return genome_from_normalised_vector(vec, use_dermal_chromophores)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def genome_to_str(
    genome: Dict[str, float],
    precision: int = 4,
    use_dermal_chromophores: bool = False,
) -> str:
    """Return a compact string representation."""
    parts = []
    for name, _, _, _, _ in get_param_defs(use_dermal_chromophores):
        val = genome.get(name, 0.0)
        parts.append(f"{name}={val:.{precision}e}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Bounds check
# ---------------------------------------------------------------------------

def check_genome(
    genome: Dict[str, float],
    use_dermal_chromophores: bool = False,
) -> List[str]:
    """Return a list of violation messages (empty if valid)."""
    issues: List[str] = []
    for name, low, high, _, _ in get_param_defs(use_dermal_chromophores):
        val = genome.get(name)
        if val is None:
            issues.append(f"Missing parameter: {name}")
            continue
        if val < low or val > high:
            issues.append(f"{name}={val} out of bounds [{low}, {high}]")
    return issues


# ---------------------------------------------------------------------------
# Mutation & crossover helpers
# ---------------------------------------------------------------------------

def mutate_genome(
    genome: Dict[str, float],
    mutation_rate: float,
    mutation_strength: float,
    rng: random.Random,
    use_dermal_chromophores: bool = False,
) -> Dict[str, float]:
    """Return a mutated copy of the genome.

    Each normalised parameter is perturbed with probability *mutation_rate*
    by adding Gaussian noise scaled by *mutation_strength*.
    """
    vec = normalised_vector_from_genome(genome, use_dermal_chromophores)
    for i in range(len(vec)):
        if rng.random() < mutation_rate:
            vec[i] += rng.gauss(0.0, mutation_strength)
            vec[i] = max(0.0, min(1.0, vec[i]))
    return genome_from_normalised_vector(vec, use_dermal_chromophores)


def crossover_genomes(
    parent_a: Dict[str, float],
    parent_b: Dict[str, float],
    rng: random.Random,
    use_dermal_chromophores: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Uniform crossover: each parameter is swapped with probability 0.5.

    Returns two children.
    """
    va = normalised_vector_from_genome(parent_a, use_dermal_chromophores)
    vb = normalised_vector_from_genome(parent_b, use_dermal_chromophores)
    for i in range(len(va)):
        if rng.random() < 0.5:
            va[i], vb[i] = vb[i], va[i]
    return (
        genome_from_normalised_vector(va, use_dermal_chromophores),
        genome_from_normalised_vector(vb, use_dermal_chromophores),
    )
