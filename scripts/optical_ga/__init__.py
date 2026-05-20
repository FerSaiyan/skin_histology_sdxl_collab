"""
Optical-parameter Genetic Algorithm (GA) for skin colour estimation.

This package implements a GA that searches for skin biophysical optical
parameters that produce a target colour (L\\*a\\*b\\* or ITA).

Methodology follows the approach described in ``external_refs/friend_optical_ga/``
(papers + reference script by Murilo).

Modules
-------
genome_encoding_optical
    Biophysical parameter definitions, bounds, encode/decode helpers.
forward_model
    Surrogate (lightweight) forward model for smoke tests / CI;
    realistic mode expects external MC backend (xopto/mcx).
colorimetry
    L\\*a\\*b\\*, ITA angle, and colour-space utilities (wraps colour-science).
optical_fitness
    Fitness function based on L\\*a\\*b\\* target matching with normalised
    denominators and optional ITA-mode.
ga_optimiser_optical
    GA loop CLI with configurable population, generations, and mode.
"""

from __future__ import annotations
