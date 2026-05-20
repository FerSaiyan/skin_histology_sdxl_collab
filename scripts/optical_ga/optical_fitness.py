#!/usr/bin/env python
"""
Fitness evaluation for the optical-parameter GA.

Provides fitness functions based on L\\*a\\*b\\* target matching with
normalised denominators (following the friend's reference script approach)
and an optional ITA-mode.

Fitness modes
-------------
1. **Lab mode** (default)
   Targets specific L\\*, a\\*, b\\* values.  Error is computed as:

   .. math::

       err_L = \\frac{L_{sim} - L_{target}}{60}
       err_a = \\frac{a_{sim} - a_{target}}{20}
       err_b = \\frac{b_{sim} - b_{target}}{25}

   Fitness = ``1 / (sqrt(err_L^2 + err_a^2 + err_b^2) + eps)``

   The ``a`` target can have an optional constraint range (e.g. [2, 20]),
   where violations are penalised.

2. **ITA mode**
   Targets a specific ITA value (and optionally L\\*).  The ITA constraint
   is combined with L\\* matching.

3. **ITANoA mode**
   Targets only ITA (no a* constraint), matching the friend's ITA-graph
   approach where only L\\* and b\\* are matched and a\\* is softly
   constrained to a range.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.optical_ga.colorimetry import calculate_ita


# ---------------------------------------------------------------------------
# Fitness modes
# ---------------------------------------------------------------------------

FITNESS_MODES = ["lab", "ita", "ita_no_a"]


# ---------------------------------------------------------------------------
# Core fitness computation
# ---------------------------------------------------------------------------


def compute_lab_fitness(
    lab_sim: Tuple[float, float, float],
    lab_target: Tuple[float, float, float],
    a_range: Optional[Tuple[float, float]] = None,
    eps: float = 1e-6,
) -> Tuple[float, Dict[str, float]]:
    """Compute fitness from L\\*a\\*b\\* simulation vs target.

    Uses normalised denominators as in the reference script:
    - L\\* error normalised by 60
    - a\\* error normalised by 20
    - b\\* error normalised by 25

    Parameters
    ----------
    lab_sim : (float, float, float)
        Simulated L\\*, a\\*, b\\*.
    lab_target : (float, float, float)
        Target L\\*, a\\*, b\\*.
    a_range : (float, float) or None
        Optional soft constraint for a\\* (e.g. (2, 20)).
        If violated, additional penalty is applied.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    fitness : float
        Higher is better (maximisation).
    components : dict
        Per-component errors and scores.
    """
    L_sim, a_sim, b_sim = lab_sim
    L_tgt, a_tgt, b_tgt = lab_target

    err_L = (L_sim - L_tgt) / 60.0
    err_b = (b_sim - b_tgt) / 25.0

    # a* error with optional range constraint
    if a_range is not None:
        a_low, a_high = a_range
        if a_low <= a_sim <= a_high:
            err_a = (a_sim - a_tgt) / 20.0
        else:
            # Penalty: clamped violated distance normalised by 20
            clamped = max(a_low, min(a_high, a_sim))
            err_a = (clamped - a_tgt) / 20.0 + abs(a_sim - clamped) / 20.0
    else:
        err_a = (a_sim - a_tgt) / 20.0

    total_err = np.sqrt(err_L ** 2 + err_a ** 2 + err_b ** 2)
    fitness = 1.0 / (total_err + eps)

    components = {
        "err_L": float(err_L),
        "err_a": float(err_a),
        "err_b": float(err_b),
        "total_err": float(total_err),
        "L_sim": float(L_sim),
        "a_sim": float(a_sim),
        "b_sim": float(b_sim),
        "L_target": float(L_tgt),
        "a_target": float(a_tgt),
        "b_target": float(b_tgt),
    }

    return float(fitness), components


def compute_ita_fitness(
    lab_sim: Tuple[float, float, float],
    lab_target: Tuple[float, float, float],
    eps: float = 1e-6,
) -> Tuple[float, Dict[str, float]]:
    """Compute fitness using ITA matching (L\\* + b\\* target, ITA soft).

    The ITA angle is computed from both simulated and target L\\* and b\\*,
    and the error is a combination of:
    - L\\* error (normalised by 60)
    - ITA error (normalised by 55)

    a\\* is softly constrained to [2, 20].

    Parameters
    ----------
    lab_sim : (float, float, float)
        Simulated L\\*, a\\*, b\\*.
    lab_target : (float, float, float)
        Target L\\*, a\\*, b\\*.

    Returns
    -------
    fitness : float
    components : dict
    """
    L_sim, a_sim, b_sim = lab_sim
    L_tgt, _, b_tgt = lab_target

    ita_sim = calculate_ita(L_sim, b_sim)
    ita_tgt = calculate_ita(L_tgt, b_tgt)

    err_L = (L_sim - L_tgt) / 60.0
    err_ita = (ita_sim - ita_tgt) / 55.0

    # a* soft constraint
    err_a = 0.0
    if a_sim < 2:
        err_a = (a_sim - 2.0) / 20.0
    elif a_sim > 20:
        err_a = (a_sim - 20.0) / 20.0

    total_err = np.sqrt(err_L ** 2 + err_ita ** 2 + err_a ** 2)
    fitness = 1.0 / (total_err + eps)

    components = {
        "err_L": float(err_L),
        "err_ita": float(err_ita),
        "err_a": float(err_a),
        "total_err": float(total_err),
        "L_sim": float(L_sim),
        "a_sim": float(a_sim),
        "b_sim": float(b_sim),
        "ita_sim": float(ita_sim),
        "ita_target": float(ita_tgt),
        "L_target": float(L_tgt),
        "b_target": float(b_tgt),
    }

    return float(fitness), components


def compute_ita_no_a_fitness(
    lab_sim: Tuple[float, float, float],
    lab_target: Tuple[float, float, float],
    eps: float = 1e-6,
) -> Tuple[float, Dict[str, float]]:
    """Compute fitness using only L\\* and b\\* (ITA) matching.

    This matches the friend's ITA-graph method: only L\\* and b\\* are
    targeted; a\\* is softly constrained to [2, 20].

    Parameters
    ----------
    lab_sim : (float, float, float)
        Simulated L\\*, a\\*, b\\*.
    lab_target : (float, float, float)
        Target L\\*, b\\* (a_target is ignored).

    Returns
    -------
    fitness : float
    components : dict
    """
    L_sim, a_sim, b_sim = lab_sim
    L_tgt, _, b_tgt = lab_target

    err_L = (L_sim - L_tgt) / 60.0
    err_b = (b_sim - b_tgt) / 25.0

    # a* soft constraint only (no target)
    err_a = 0.0
    if a_sim < 2:
        err_a = (a_sim - 2.0) / 20.0
    elif a_sim > 20:
        err_a = (a_sim - 20.0) / 20.0

    total_err = np.sqrt(err_L ** 2 + err_b ** 2 + err_a ** 2)
    fitness = 1.0 / (total_err + eps)

    components = {
        "err_L": float(err_L),
        "err_b": float(err_b),
        "err_a": float(err_a),
        "total_err": float(total_err),
        "L_sim": float(L_sim),
        "a_sim": float(a_sim),
        "b_sim": float(b_sim),
        "L_target": float(L_tgt),
        "b_target": float(b_tgt),
    }

    return float(fitness), components


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_FITNESS_FUNCS = {
    "lab": compute_lab_fitness,
    "ita": compute_ita_fitness,
    "ita_no_a": compute_ita_no_a_fitness,
}


def compute_fitness(
    lab_sim: Tuple[float, float, float],
    lab_target: Tuple[float, float, float],
    mode: str = "lab",
    a_range: Optional[Tuple[float, float]] = None,
    eps: float = 1e-6,
) -> Tuple[float, Dict[str, float]]:
    """Compute fitness in the specified mode.

    Parameters
    ----------
    lab_sim : (float, float, float)
        Simulated L\\*, a\\*, b\\*.
    lab_target : (float, float, float)
        Target L\\*, a\\*, b\\*.
    mode : str
        ``"lab"``, ``"ita"``, or ``"ita_no_a"``.
    a_range : (float, float) or None
        Optional a\\* soft constraint (lab mode only).
    eps : float
        Small epsilon for numerical stability.

    Returns
    -------
    fitness : float
    components : dict
    """
    func = _FITNESS_FUNCS.get(mode)
    if func is None:
        raise ValueError(
            f"Unknown fitness mode: {mode!r}. "
            f"Available: {list(_FITNESS_FUNCS.keys())}"
        )

    if mode == "lab":
        return func(lab_sim, lab_target, a_range=a_range, eps=eps)
    else:
        return func(lab_sim, lab_target, eps=eps)
