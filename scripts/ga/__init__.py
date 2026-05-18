# GA (Genetic Algorithm) skeleton for SDXL inpainting hyperparameter optimisation.
#
# Phase A: classifier-free GA skeleton usable today with geometry-only fitness.
#
# Modules:
#   genome_encoding.py   — parameter bounds, encode/decode helpers
#   run_inpaint_fitness.py — classifier-free geometry/QC-based fitness
#   ga_optimiser.py       — GA loop CLI
#   ga_report.py          — lightweight reporting from history CSV

from __future__ import annotations
