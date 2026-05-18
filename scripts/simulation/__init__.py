# Simulation scaffolding: MCX light transport, colorimetry, thermal.
#
# Phase C (implemented):
#   thermal_build_model.py — convert label volume → thermal property arrays
#   thermal_solve.py       — explicit FD Pennes bioheat solver
#   thermal_visualise.py   — temperature slice PNG + summary stats
#
# Phase B (current):
#   mcx_build_volume.py    — convert NIfTI/npy volume → MCX artifacts
#   mcx_batch_runner.py    — batch submit / dry-run MCX simulations
#   mcx_extract_fluence.py — extract stats + projection PNG from MCX output
#
# Phase A (implemented):
#   colorimetry_metrics.py — CIELAB ΔE*, RGB MAE, channel drift (classifier-free)

from __future__ import annotations
