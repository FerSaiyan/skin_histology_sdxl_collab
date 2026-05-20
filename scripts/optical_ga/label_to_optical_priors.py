#!/usr/bin/env python
"""
Convert a class-ID label map (.npy) into optical-prior descriptor artifacts.

Reads a class-ID label map (2D HxW or 3D DxHxW) and assigns per-class optical
priors (chromophore/scattering bounds) from a priors config JSON.  The output
is a JSON file documenting priors actually used plus pixel/voxel counts per
class.

Optionally saves a per-voxel prior index map (.npz) for downstream use.

Usage:
  python scripts/optical_ga/label_to_optical_priors.py \
      --label-npy /tmp/labels.npy \
      --output-priors-json /tmp/priors.json \
      --output-voxel-prior-npz /tmp/voxel_prior_idx.npz

Defaults for major Histo-Seg classes are built-in when no --priors-config is
provided.  Unknown class IDs trigger a warning and fallback to generic tissue
priors.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Built-in default priors table.
# ---------------------------------------------------------------------------
# Each entry maps a class ID (as str key) to a dict:
#   class_name  – human-readable name
#   priors      – min/max ranges for optical-GA biophysical parameters
#
# These defaults cover the 12 Histo-Seg classes and are also used as the
# fallback when no --priors-config is provided.

DEFAULT_PRIORS_TABLE: Dict[str, Dict[str, Any]] = {
    "0": {
        "class_name": "background",
        "priors": {
            "melanin_min": 0.0, "melanin_max": 0.001,
            "blood_layer1_min": 0.0, "blood_layer1_max": 0.001,
            "blood_layer2_min": 0.0, "blood_layer2_max": 0.001,
            "spo2_min": 0.5, "spo2_max": 0.5,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
            "d_layer1_min": 0.1e-3, "d_layer1_max": 0.5e-3,
            "amp_layer0_min": 0.5, "amp_layer0_max": 1.5,
            "amp_layer1_min": 0.5, "amp_layer1_max": 1.5,
            "amp_layer2_min": 0.5, "amp_layer2_max": 1.5,
            "water_layer0_min": 0.01, "water_layer0_max": 0.8,
            "water_layer1_min": 0.01, "water_layer1_max": 0.8,
            "water_layer2_min": 0.01, "water_layer2_max": 0.8,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.8,
            "n_mult_layer0_min": 0.8, "n_mult_layer0_max": 1.2,
            "n_mult_layer1_min": 0.8, "n_mult_layer1_max": 1.2,
            "n_mult_layer2_min": 0.8, "n_mult_layer2_max": 1.2,
        },
    },
    "1": {
        "class_name": "epidermis",
        "priors": {
            "melanin_min": 0.05, "melanin_max": 0.5,
            "blood_layer1_min": 0.0, "blood_layer1_max": 0.02,
            "blood_layer2_min": 0.0, "blood_layer2_max": 0.02,
            "spo2_min": 0.6, "spo2_max": 0.95,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 3e-5, "d_layer0_max": 1.2e-4,
            "d_layer1_min": 0.1e-3, "d_layer1_max": 0.4e-3,
            "amp_layer0_min": 0.6, "amp_layer0_max": 1.4,
            "amp_layer1_min": 0.6, "amp_layer1_max": 1.4,
            "amp_layer2_min": 0.6, "amp_layer2_max": 1.4,
            "water_layer0_min": 0.1, "water_layer0_max": 0.7,
            "water_layer1_min": 0.2, "water_layer1_max": 0.7,
            "water_layer2_min": 0.2, "water_layer2_max": 0.7,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.4,
            "n_mult_layer0_min": 0.9, "n_mult_layer0_max": 1.15,
            "n_mult_layer1_min": 0.9, "n_mult_layer1_max": 1.15,
            "n_mult_layer2_min": 0.9, "n_mult_layer2_max": 1.15,
        },
    },
    "2": {
        "class_name": "reticular_dermis",
        "priors": {
            "melanin_min": 0.0, "melanin_max": 0.05,
            "blood_layer1_min": 0.005, "blood_layer1_max": 0.08,
            "blood_layer2_min": 0.005, "blood_layer2_max": 0.08,
            "spo2_min": 0.6, "spo2_max": 0.95,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
            "d_layer1_min": 0.2e-3, "d_layer1_max": 0.5e-3,
            "amp_layer0_min": 0.6, "amp_layer0_max": 1.4,
            "amp_layer1_min": 0.7, "amp_layer1_max": 1.5,
            "amp_layer2_min": 0.6, "amp_layer2_max": 1.4,
            "water_layer0_min": 0.2, "water_layer0_max": 0.7,
            "water_layer1_min": 0.3, "water_layer1_max": 0.8,
            "water_layer2_min": 0.2, "water_layer2_max": 0.7,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.5,
            "n_mult_layer0_min": 0.85, "n_mult_layer0_max": 1.1,
            "n_mult_layer1_min": 0.9, "n_mult_layer1_max": 1.15,
            "n_mult_layer2_min": 0.85, "n_mult_layer2_max": 1.1,
        },
    },
    "3": {
        "class_name": "papillary_dermis",
        "priors": {
            "melanin_min": 0.0, "melanin_max": 0.05,
            "blood_layer1_min": 0.01, "blood_layer1_max": 0.1,
            "blood_layer2_min": 0.01, "blood_layer2_max": 0.08,
            "spo2_min": 0.65, "spo2_max": 0.98,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
            "d_layer1_min": 0.15e-3, "d_layer1_max": 0.5e-3,
            "amp_layer0_min": 0.6, "amp_layer0_max": 1.4,
            "amp_layer1_min": 0.7, "amp_layer1_max": 1.5,
            "amp_layer2_min": 0.6, "amp_layer2_max": 1.4,
            "water_layer0_min": 0.2, "water_layer0_max": 0.7,
            "water_layer1_min": 0.3, "water_layer1_max": 0.8,
            "water_layer2_min": 0.2, "water_layer2_max": 0.7,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.4,
            "n_mult_layer0_min": 0.85, "n_mult_layer0_max": 1.1,
            "n_mult_layer1_min": 0.9, "n_mult_layer1_max": 1.15,
            "n_mult_layer2_min": 0.85, "n_mult_layer2_max": 1.1,
        },
    },
    "4": {
        "class_name": "dermis",
        "priors": {
            "melanin_min": 0.0, "melanin_max": 0.03,
            "blood_layer1_min": 0.005, "blood_layer1_max": 0.08,
            "blood_layer2_min": 0.005, "blood_layer2_max": 0.08,
            "spo2_min": 0.6, "spo2_max": 0.95,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
            "d_layer1_min": 0.2e-3, "d_layer1_max": 0.5e-3,
            "amp_layer0_min": 0.6, "amp_layer0_max": 1.4,
            "amp_layer1_min": 0.7, "amp_layer1_max": 1.5,
            "amp_layer2_min": 0.6, "amp_layer2_max": 1.4,
            "water_layer0_min": 0.2, "water_layer0_max": 0.7,
            "water_layer1_min": 0.3, "water_layer1_max": 0.8,
            "water_layer2_min": 0.2, "water_layer2_max": 0.7,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.5,
            "n_mult_layer0_min": 0.85, "n_mult_layer0_max": 1.1,
            "n_mult_layer1_min": 0.9, "n_mult_layer1_max": 1.15,
            "n_mult_layer2_min": 0.85, "n_mult_layer2_max": 1.1,
        },
    },
    "5": {
        "class_name": "keratin",
        "priors": {
            "melanin_min": 0.0, "melanin_max": 0.02,
            "blood_layer1_min": 0.0, "blood_layer1_max": 0.01,
            "blood_layer2_min": 0.0, "blood_layer2_max": 0.01,
            "spo2_min": 0.5, "spo2_max": 0.8,
            "g_layer0_min": 0.88, "g_layer0_max": 0.98,
            "g_layer1_min": 0.88, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
            "d_layer1_min": 0.1e-3, "d_layer1_max": 0.5e-3,
            "amp_layer0_min": 0.8, "amp_layer0_max": 1.5,
            "amp_layer1_min": 0.6, "amp_layer1_max": 1.4,
            "amp_layer2_min": 0.5, "amp_layer2_max": 1.3,
            "water_layer0_min": 0.01, "water_layer0_max": 0.5,
            "water_layer1_min": 0.01, "water_layer1_max": 0.5,
            "water_layer2_min": 0.01, "water_layer2_max": 0.5,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.4,
            "n_mult_layer0_min": 0.9, "n_mult_layer0_max": 1.2,
            "n_mult_layer1_min": 0.85, "n_mult_layer1_max": 1.1,
            "n_mult_layer2_min": 0.85, "n_mult_layer2_max": 1.1,
        },
    },
    "6": {
        "class_name": "inflammation",
        "priors": {
            "melanin_min": 0.0, "melanin_max": 0.05,
            "blood_layer1_min": 0.02, "blood_layer1_max": 0.1,
            "blood_layer2_min": 0.02, "blood_layer2_max": 0.1,
            "spo2_min": 0.5, "spo2_max": 0.9,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
            "d_layer1_min": 0.2e-3, "d_layer1_max": 0.5e-3,
            "amp_layer0_min": 0.5, "amp_layer0_max": 1.4,
            "amp_layer1_min": 0.6, "amp_layer1_max": 1.5,
            "amp_layer2_min": 0.5, "amp_layer2_max": 1.4,
            "water_layer0_min": 0.3, "water_layer0_max": 0.8,
            "water_layer1_min": 0.4, "water_layer1_max": 0.8,
            "water_layer2_min": 0.3, "water_layer2_max": 0.8,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.4,
            "n_mult_layer0_min": 0.85, "n_mult_layer0_max": 1.1,
            "n_mult_layer1_min": 0.9, "n_mult_layer1_max": 1.15,
            "n_mult_layer2_min": 0.85, "n_mult_layer2_max": 1.1,
        },
    },
    "7": {
        "class_name": "hair_follicles",
        "priors": {
            "melanin_min": 0.2, "melanin_max": 0.5,
            "blood_layer1_min": 0.005, "blood_layer1_max": 0.05,
            "blood_layer2_min": 0.005, "blood_layer2_max": 0.05,
            "spo2_min": 0.6, "spo2_max": 0.95,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
            "d_layer1_min": 0.15e-3, "d_layer1_max": 0.5e-3,
            "amp_layer0_min": 0.6, "amp_layer0_max": 1.4,
            "amp_layer1_min": 0.6, "amp_layer1_max": 1.4,
            "amp_layer2_min": 0.6, "amp_layer2_max": 1.4,
            "water_layer0_min": 0.1, "water_layer0_max": 0.6,
            "water_layer1_min": 0.1, "water_layer1_max": 0.6,
            "water_layer2_min": 0.1, "water_layer2_max": 0.6,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.4,
            "n_mult_layer0_min": 0.85, "n_mult_layer0_max": 1.1,
            "n_mult_layer1_min": 0.85, "n_mult_layer1_max": 1.1,
            "n_mult_layer2_min": 0.85, "n_mult_layer2_max": 1.1,
        },
    },
    "8": {
        "class_name": "glands",
        "priors": {
            "melanin_min": 0.0, "melanin_max": 0.03,
            "blood_layer1_min": 0.005, "blood_layer1_max": 0.06,
            "blood_layer2_min": 0.005, "blood_layer2_max": 0.06,
            "spo2_min": 0.6, "spo2_max": 0.95,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
            "d_layer1_min": 0.15e-3, "d_layer1_max": 0.5e-3,
            "amp_layer0_min": 0.5, "amp_layer0_max": 1.3,
            "amp_layer1_min": 0.6, "amp_layer1_max": 1.4,
            "amp_layer2_min": 0.5, "amp_layer2_max": 1.3,
            "water_layer0_min": 0.3, "water_layer0_max": 0.8,
            "water_layer1_min": 0.3, "water_layer1_max": 0.8,
            "water_layer2_min": 0.2, "water_layer2_max": 0.7,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.5,
            "n_mult_layer0_min": 0.85, "n_mult_layer0_max": 1.1,
            "n_mult_layer1_min": 0.85, "n_mult_layer1_max": 1.1,
            "n_mult_layer2_min": 0.85, "n_mult_layer2_max": 1.1,
        },
    },
    "9": {
        "class_name": "basal_cell_carcinoma",
        "priors": {
            "melanin_min": 0.02, "melanin_max": 0.3,
            "blood_layer1_min": 0.01, "blood_layer1_max": 0.1,
            "blood_layer2_min": 0.01, "blood_layer2_max": 0.1,
            "spo2_min": 0.5, "spo2_max": 0.9,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 4e-5, "d_layer0_max": 1.5e-4,
            "d_layer1_min": 0.2e-3, "d_layer1_max": 0.5e-3,
            "amp_layer0_min": 0.5, "amp_layer0_max": 1.5,
            "amp_layer1_min": 0.6, "amp_layer1_max": 1.5,
            "amp_layer2_min": 0.5, "amp_layer2_max": 1.4,
            "water_layer0_min": 0.2, "water_layer0_max": 0.7,
            "water_layer1_min": 0.2, "water_layer1_max": 0.7,
            "water_layer2_min": 0.2, "water_layer2_max": 0.7,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.4,
            "n_mult_layer0_min": 0.85, "n_mult_layer0_max": 1.15,
            "n_mult_layer1_min": 0.85, "n_mult_layer1_max": 1.15,
            "n_mult_layer2_min": 0.85, "n_mult_layer2_max": 1.15,
        },
    },
    "10": {
        "class_name": "squamous_cell_carcinoma",
        "priors": {
            "melanin_min": 0.02, "melanin_max": 0.3,
            "blood_layer1_min": 0.01, "blood_layer1_max": 0.1,
            "blood_layer2_min": 0.01, "blood_layer2_max": 0.1,
            "spo2_min": 0.5, "spo2_max": 0.9,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 4e-5, "d_layer0_max": 1.5e-4,
            "d_layer1_min": 0.2e-3, "d_layer1_max": 0.5e-3,
            "amp_layer0_min": 0.5, "amp_layer0_max": 1.5,
            "amp_layer1_min": 0.6, "amp_layer1_max": 1.5,
            "amp_layer2_min": 0.5, "amp_layer2_max": 1.4,
            "water_layer0_min": 0.2, "water_layer0_max": 0.7,
            "water_layer1_min": 0.2, "water_layer1_max": 0.7,
            "water_layer2_min": 0.2, "water_layer2_max": 0.7,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.4,
            "n_mult_layer0_min": 0.85, "n_mult_layer0_max": 1.15,
            "n_mult_layer1_min": 0.85, "n_mult_layer1_max": 1.15,
            "n_mult_layer2_min": 0.85, "n_mult_layer2_max": 1.15,
        },
    },
    "11": {
        "class_name": "intraepidermal_carcinoma",
        "priors": {
            "melanin_min": 0.02, "melanin_max": 0.35,
            "blood_layer1_min": 0.005, "blood_layer1_max": 0.08,
            "blood_layer2_min": 0.005, "blood_layer2_max": 0.08,
            "spo2_min": 0.5, "spo2_max": 0.9,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
            "d_layer1_min": 0.15e-3, "d_layer1_max": 0.5e-3,
            "amp_layer0_min": 0.5, "amp_layer0_max": 1.5,
            "amp_layer1_min": 0.6, "amp_layer1_max": 1.5,
            "amp_layer2_min": 0.5, "amp_layer2_max": 1.4,
            "water_layer0_min": 0.2, "water_layer0_max": 0.7,
            "water_layer1_min": 0.2, "water_layer1_max": 0.7,
            "water_layer2_min": 0.2, "water_layer2_max": 0.7,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.4,
            "n_mult_layer0_min": 0.85, "n_mult_layer0_max": 1.15,
            "n_mult_layer1_min": 0.85, "n_mult_layer1_max": 1.15,
            "n_mult_layer2_min": 0.85, "n_mult_layer2_max": 1.15,
        },
    },
}

# Generic fallback for unknown class IDs
_GENERIC_FALLBACK_PRIORS: Dict[str, Any] = {
    "class_name": "generic_tissue",
    "priors": {
        "melanin_min": 0.0, "melanin_max": 0.1,
        "blood_layer1_min": 0.0, "blood_layer1_max": 0.05,
        "blood_layer2_min": 0.0, "blood_layer2_max": 0.05,
        "spo2_min": 0.5, "spo2_max": 0.95,
        "g_layer0_min": 0.85, "g_layer0_max": 0.95,
        "g_layer1_min": 0.85, "g_layer1_max": 0.95,
        "g_layer2_min": 0.85, "g_layer2_max": 0.95,
        "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
        "d_layer1_min": 0.1e-3, "d_layer1_max": 0.5e-3,
        "amp_layer0_min": 0.5, "amp_layer0_max": 1.5,
        "amp_layer1_min": 0.5, "amp_layer1_max": 1.5,
        "amp_layer2_min": 0.5, "amp_layer2_max": 1.5,
        "water_layer0_min": 0.1, "water_layer0_max": 0.7,
        "water_layer1_min": 0.1, "water_layer1_max": 0.7,
        "water_layer2_min": 0.1, "water_layer2_max": 0.7,
        "fat_layer2_min": 0.05, "fat_layer2_max": 0.5,
        "n_mult_layer0_min": 0.85, "n_mult_layer0_max": 1.15,
        "n_mult_layer1_min": 0.85, "n_mult_layer1_max": 1.15,
        "n_mult_layer2_min": 0.85, "n_mult_layer2_max": 1.15,
    },
}

# Names of all expected prior bound keys.
# There are 19 core parameters, represented as min/max bounds => 38 keys.
_EXPECTED_PRIOR_KEYS: List[str] = [
    "melanin_min", "melanin_max",
    "blood_layer1_min", "blood_layer1_max",
    "blood_layer2_min", "blood_layer2_max",
    "spo2_min", "spo2_max",
    "g_layer0_min", "g_layer0_max",
    "g_layer1_min", "g_layer1_max",
    "g_layer2_min", "g_layer2_max",
    "d_layer0_min", "d_layer0_max",
    "d_layer1_min", "d_layer1_max",
    "amp_layer0_min", "amp_layer0_max",
    "amp_layer1_min", "amp_layer1_max",
    "amp_layer2_min", "amp_layer2_max",
    "water_layer0_min", "water_layer0_max",
    "water_layer1_min", "water_layer1_max",
    "water_layer2_min", "water_layer2_max",
    "fat_layer2_min", "fat_layer2_max",
    "n_mult_layer0_min", "n_mult_layer0_max",
    "n_mult_layer1_min", "n_mult_layer1_max",
    "n_mult_layer2_min", "n_mult_layer2_max",
]


def _load_priors_config(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load a priors config JSON, or return built-in defaults.

    The file is expected to have the same structure as DEFAULT_PRIORS_TABLE:
    a dict keyed by string class IDs, each with ``class_name`` and ``priors``.
    """
    if path is None:
        return DEFAULT_PRIORS_TABLE

    p = Path(path)
    if not p.exists():
        print(
            f"Warning: priors config not found: {path}. Using built-in defaults.",
            file=sys.stderr,
        )
        return DEFAULT_PRIORS_TABLE

    with open(str(p), "r") as f:
        data = json.load(f)

    # Remove metadata key if present
    if "metadata" in data:
        data.pop("metadata")

    return data


def _load_label_volume(path: str) -> np.ndarray:
    """Load a .npy label volume (2D or 3D)."""
    p = Path(path)
    if not p.exists():
        print(f"Error: label volume file not found: {path}", file=sys.stderr)
        sys.exit(1)
    arr = np.load(str(p))
    if arr.dtype.kind in ("f",):
        arr = arr.astype(np.int32)
    return arr


def _compute_label_counts(label_vol: np.ndarray) -> Dict[str, int]:
    """Count voxels/pixels per class ID, keyed by string class ID."""
    unique, counts = np.unique(label_vol, return_counts=True)
    result: Dict[str, int] = {}
    for uid, cnt in zip(unique, counts):
        result[str(int(uid))] = int(cnt)
    return result


def _build_prior_index_map(
    label_vol: np.ndarray,
    priors_table: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    """Build a per-voxel prior index map.

    Each voxel stores the index into the sorted list of class IDs present in
    priors_table.  Unknown class IDs are mapped to an explicit generic-tissue
    fallback index.

    Fallback index determination:
    - If class ``"0"`` is present in *priors_table* (e.g. a background class),
      its sorted index is used as the fallback.  This preserves the invariant
      that class-0 voxels (background) and any unknown-label voxels share the
      same generic-tissue prior index when ``"0"`` is configured.
    - If class ``"0"`` is **not** present in *priors_table*, a dedicated
      synthetic slot is appended **after** all configured classes.  The slot
      index is ``len(sorted_class_keys)``, i.e. one past the last configured
      entry.  No entry is added to *priors_table* itself.
    - Raises ``ValueError`` if the required fallback index exceeds the int8
      capacity (127).

    The output dtype is int8 (supports up to 127 classes).
    """
    sorted_class_keys = sorted(priors_table.keys(), key=int)
    # Build lookup: class ID → index
    lookup: Dict[int, int] = {}
    for idx, ckey in enumerate(sorted_class_keys):
        lookup[int(ckey)] = idx

    # ---- explicit fallback index -------------------------------------------
    if "0" in priors_table:
        fallback_index = lookup[0]
    else:
        # Dedicated synthetic slot after all configured classes.
        fallback_index = len(sorted_class_keys)

    if fallback_index > 127:
        raise ValueError(
            f"Fallback prior index {fallback_index} exceeds int8 capacity "
            f"(127).  Too many configured classes "
            f"({len(sorted_class_keys)}) + fallback."
        )
    # ------------------------------------------------------------------------

    flat = label_vol.ravel()
    index_flat = np.full_like(flat, fill_value=127, dtype=np.int8)
    for cid, idx in lookup.items():
        mask = flat == cid
        index_flat[mask] = np.int8(idx)

    # Unknown class IDs → the explicit generic-tissue fallback index
    index_flat[index_flat == 127] = np.int8(fallback_index)

    return index_flat.reshape(label_vol.shape)


def _validate_priors_table(priors_table: Dict[str, Dict[str, Any]]) -> List[str]:
    """Validate structure of priors table. Return list of issues (empty if OK)."""
    issues: List[str] = []
    for ckey, entry in priors_table.items():
        if "class_name" not in entry:
            issues.append(f"Class {ckey}: missing 'class_name'")
        if "priors" not in entry:
            issues.append(f"Class {ckey}: missing 'priors' dict")
            continue
        priors = entry["priors"]
        for expected_key in _EXPECTED_PRIOR_KEYS:
            if expected_key not in priors:
                issues.append(f"Class {ckey}: missing prior '{expected_key}'")
    return issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert a class-ID label map (.npy) into optical-prior "
        "descriptor artifacts.",
    )
    ap.add_argument(
        "--label-npy",
        required=True,
        help="Path to input class-ID label .npy array (2D HxW or 3D DxHxW).",
    )
    ap.add_argument(
        "--priors-config",
        default=None,
        help="Optional custom priors JSON path. Falls back to built-in defaults.",
    )
    ap.add_argument(
        "--output-priors-json",
        required=True,
        help="Path for output per-class priors + counts JSON.",
    )
    ap.add_argument(
        "--output-voxel-prior-npz",
        default=None,
        help="Optional path for per-voxel prior index map .npz. "
        "Contains array 'prior_index_map' with same shape as input.",
    )
    return ap.parse_args(argv[1:])


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    # Load priors config
    priors_table = _load_priors_config(args.priors_config)

    # Validate priors table
    issues = _validate_priors_table(priors_table)
    for issue in issues:
        print(f"Warning: {issue}", file=sys.stderr)

    # Load label volume
    label_vol = _load_label_volume(args.label_npy)
    if label_vol.ndim not in (2, 3):
        print(
            f"Error: expected 2D or 3D label volume, got shape {label_vol.shape}",
            file=sys.stderr,
        )
        return 1
    print(f"Loaded label volume: {args.label_npy}  shape={label_vol.shape}  "
          f"dtype={label_vol.dtype}")

    # Count labels
    label_counts = _compute_label_counts(label_vol)
    print(f"Label counts: {label_counts}")

    # Build output structure: per-class priors + counts
    seen_classes: set = set()
    output_priors: Dict[str, Dict[str, Any]] = {}
    unaccounted_ids: List[int] = []

    for ckey in sorted(priors_table.keys(), key=int):
        cid = int(ckey)
        count = label_counts.get(ckey, 0)
        entry = dict(priors_table[ckey])  # shallow copy
        entry["count_voxels"] = count
        entry["class_id"] = cid
        output_priors[ckey] = entry
        seen_classes.add(cid)

    # Warn about class IDs in label volume but not in priors table
    for ckey, count in sorted(label_counts.items(), key=lambda x: int(x[0])):
        cid = int(ckey)
        if cid not in seen_classes:
            print(
                f"Warning: class ID {cid} (count={count}) not found in priors "
                f"config. Using generic tissue priors.",
                file=sys.stderr,
            )
            fallback = dict(_GENERIC_FALLBACK_PRIORS)
            fallback["count_voxels"] = count
            fallback["class_id"] = cid
            fallback["fallback_reason"] = f"class_{cid}_not_in_config"
            output_priors[ckey] = fallback
            unaccounted_ids.append(cid)

    # Build top-level metadata
    result: Dict[str, Any] = {
        "metadata": {
            "source_label_npy": str(Path(args.label_npy).resolve()),
            "label_volume_shape": list(label_vol.shape),
            "priors_config_source": (
                args.priors_config if args.priors_config else "built-in"
            ),
            "num_classes_in_volume": len(label_counts),
            "num_classes_with_priors": len(output_priors),
            "unaccounted_class_ids": unaccounted_ids,
            "num_prior_parameters": len(_EXPECTED_PRIOR_KEYS),
        },
        "per_class_priors": output_priors,
    }

    # Write output JSON
    out_path = Path(args.output_priors_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved priors + counts: {out_path}")

    # Optional voxel prior index map
    if args.output_voxel_prior_npz:
        prior_index_map = _build_prior_index_map(label_vol, priors_table)
        npz_path = Path(args.output_voxel_prior_npz)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(npz_path), prior_index_map=prior_index_map)
        print(f"Saved voxel prior index map: {npz_path}  "
              f"shape={prior_index_map.shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
