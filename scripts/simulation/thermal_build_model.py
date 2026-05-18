#!/usr/bin/env python
"""
Build a thermal model (property arrays) from a label volume.

Reads a label volume (.npy or NIfTI) and assigns per-label thermal properties
(rho, c, k, wb, qmet) from a built-in default table.  Optional ``--properties-json``
overrides individual label entries.

Outputs (in ``--output-dir``):
  - ``thermal_model.npz``     — arrays: rho, c, k, wb, qmet, label_volume
  - ``thermal_model_manifest.json`` — provenance + per-label property table

Built-in defaults (labels 0 / 1 / 2):

  | Label | Name             | rho (kg/m³) | c (J/kg·K) | k (W/m·K) | wb (1/s)  | qmet (W/m³) |
  |-------|------------------|-------------|------------|-----------|-----------|-------------|
  | 0     | air/background   |    1.2      |   1005     |   0.026   |    0      |     0       |
  | 1     | healthy tissue   | 1100        |   3391     |   0.37    |    0.0018 |   368       |
  | 2     | lesion/tumour    | 1050        |   3850     |   0.51    |    0.008  |  5000       |

All units are SI (kg, m, s, K/W).  The property arrays share the same shape
as the input label volume.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional nibabel (for NIfTI input)
# ---------------------------------------------------------------------------
_HAS_NIBABEL = False
try:
    import nibabel as nib  # noqa: F401

    _HAS_NIBABEL = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Default thermal properties per label
# ---------------------------------------------------------------------------
# Literature sources: Jacques (2013), Hasgall et al. (IT'IS Database v4.0).
# Key:
#   rho   — density                  [kg/m³]
#   c     — specific heat capacity   [J/(kg·K)]
#   k     — thermal conductivity     [W/(m·K)]
#   wb    — blood perfusion rate     [1/s]
#   qmet  — metabolic heat source    [W/m³]

DEFAULT_THERMAL_TABLE: Dict[str, Dict[str, float]] = {
    "0": {
        "rho": 1.2,
        "c": 1005.0,
        "k": 0.026,
        "wb": 0.0,
        "qmet": 0.0,
        "label_name": "air/background",
    },
    "1": {
        "rho": 1100.0,
        "c": 3391.0,
        "k": 0.37,
        "wb": 0.0018,
        "qmet": 368.0,
        "label_name": "healthy_tissue",
    },
    "2": {
        "rho": 1050.0,
        "c": 3850.0,
        "k": 0.51,
        "wb": 0.008,
        "qmet": 5000.0,
        "label_name": "lesion_tumour",
    },
}

# Blood properties (used at solve time, baked into property arrays as reference)
BLOOD_RHO = 1060.0  # kg/m³
BLOOD_C = 3617.0  # J/(kg·K)

# Default label for voxels not present in table
_FALLBACK_LABEL_KEY = "1"  # healthy tissue


# ---------------------------------------------------------------------------
# Volume loading helpers
# ---------------------------------------------------------------------------


def _load_label_volume(volume_path: str) -> np.ndarray:
    """Load a 3D label volume from .nii/.nii.gz or .npy.

    Returns:
        A 3D numpy array (Z, Y, X) as int.

    Raises:
        SystemExit on missing file, unsupported extension, or shape < 3D.
    """
    p = Path(volume_path)
    if not p.exists():
        raise SystemExit(f"Label volume file not found: {volume_path}")

    ext = p.suffix.lower()
    is_nii_gz = p.suffixes[-2:] == [".nii", ".gz"]
    if ext == ".npy":
        arr = np.load(str(p))
    elif ext == ".nii" or is_nii_gz:
        if not _HAS_NIBABEL:
            raise SystemExit(
                "nibabel is required to load NIfTI files. "
                "Install it with: pip install nibabel"
            )
        nii = nib.load(str(p))
        arr = np.asanyarray(nii.dataobj)
    else:
        raise SystemExit(
            f"Unsupported volume extension '{ext}'. "
            "Use .nii, .nii.gz, or .npy."
        )

    if arr.ndim < 3:
        raise SystemExit(f"Label volume must be at least 3D, got shape {arr.shape}")

    # Squeeze trailing singleton dims beyond 3
    while arr.ndim > 3:
        if arr.shape[-1] > 1:
            warnings.warn(
                f"Volume has {arr.ndim} dims, using first 3 dimensions. "
                f"Shape: {arr.shape}"
            )
            arr = arr[(slice(None), slice(None), slice(None)) + (0,) * (arr.ndim - 3)]
            break
        arr = np.squeeze(arr, axis=-1)

    # Convert to integer labels
    arr = arr.astype(np.int32)
    return arr


# ---------------------------------------------------------------------------
# Property table helpers
# ---------------------------------------------------------------------------


def _load_properties_override(json_path: str) -> Dict[str, Dict[str, Any]]:
    """Load a per-label property override JSON.

    Expected format:
        {
            "1": {"k": 0.45, "wb": 0.002},
            "2": {"rho": 1080.0}
        }

    Only supplied fields are overridden; missing fields keep the table default.
    The override keys must match the table structure (str label -> dict of property floats).
    """
    p = Path(json_path)
    if not p.exists():
        raise SystemExit(f"Properties JSON not found: {json_path}")

    overrides: Dict[str, Dict[str, Any]] = {}
    raw: Dict[str, Dict[str, Any]] = json.loads(p.read_text(encoding="utf-8"))

    # Validate known numeric properties are positive where expected
    _POSITIVE_PROPS = {"rho", "c", "k"}
    _NONNEGATIVE_PROPS = {"wb", "qmet"}

    for label_key, props in raw.items():
        if not isinstance(props, dict):
            warnings.warn(
                f"  [WARN] Override for label '{label_key}' is not a dict; skipping."
            )
            continue
        sanitised: Dict[str, Any] = {}
        for prop_name, val in props.items():
            if prop_name in _POSITIVE_PROPS:
                try:
                    fval = float(val)
                    if fval <= 0:
                        warnings.warn(
                            f"  [WARN] Override for label '{label_key}' "
                            f"{prop_name}={val} is not > 0; skipping."
                        )
                        continue
                except (TypeError, ValueError):
                    warnings.warn(
                        f"  [WARN] Override for label '{label_key}' "
                        f"{prop_name}={val} is not a valid number; skipping."
                    )
                    continue
                sanitised[prop_name] = val
            elif prop_name in _NONNEGATIVE_PROPS:
                try:
                    fval = float(val)
                    if fval < 0:
                        warnings.warn(
                            f"  [WARN] Override for label '{label_key}' "
                            f"{prop_name}={val} is not >= 0; skipping."
                        )
                        continue
                except (TypeError, ValueError):
                    warnings.warn(
                        f"  [WARN] Override for label '{label_key}' "
                        f"{prop_name}={val} is not a valid number; skipping."
                    )
                    continue
                sanitised[prop_name] = val
            elif prop_name == "label_name":
                sanitised[prop_name] = str(val)
        if sanitised:
            overrides[label_key] = sanitised

    return overrides


def _build_thermal_table(
    labels: np.ndarray,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build the per-label thermal property table from defaults + overrides."""
    observed = sorted(int(v) for v in np.unique(labels))
    table: Dict[str, Dict[str, Any]] = {}

    for label_id in observed:
        key = str(label_id)
        # Start with default or fallback
        if key in DEFAULT_THERMAL_TABLE:
            entry = dict(DEFAULT_THERMAL_TABLE[key])
        else:
            entry = dict(DEFAULT_THERMAL_TABLE[_FALLBACK_LABEL_KEY])
            entry["label_name"] = f"label_{label_id}"
            entry["_from_fallback"] = True

        # Apply overrides
        if overrides and key in overrides:
            entry.update(overrides[key])
            entry.setdefault("_overridden", True)

        table[key] = entry

    return table


# ---------------------------------------------------------------------------
# Core build function
# ---------------------------------------------------------------------------


def build_thermal_model(
    label_volume_path: str,
    output_dir: str,
    properties_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Build thermal property arrays from a label volume.

    Returns a manifest dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load label volume ---
    print(f"Loading label volume: {label_volume_path}")
    labels = _load_label_volume(label_volume_path)
    print(f"  Shape: {labels.shape}, dtype={labels.dtype}")
    print(f"  Unique labels: {sorted(int(v) for v in np.unique(labels))}")

    # --- Load overrides ---
    overrides = None
    if properties_json:
        print(f"Loading property overrides: {properties_json}")
        overrides = _load_properties_override(properties_json)
        if overrides:
            print(f"  Override labels: {sorted(overrides.keys())}")

    # --- Build thermal table ---
    table = _build_thermal_table(labels, overrides)

    # --- Allocate property arrays ---
    shape = labels.shape
    rho = np.zeros(shape, dtype=np.float64)
    c = np.zeros(shape, dtype=np.float64)
    k = np.zeros(shape, dtype=np.float64)
    wb = np.zeros(shape, dtype=np.float64)
    qmet = np.zeros(shape, dtype=np.float64)

    for label_key, props in table.items():
        mask = labels == int(label_key)
        rho[mask] = props["rho"]
        c[mask] = props["c"]
        k[mask] = props["k"]
        wb[mask] = props["wb"]
        qmet[mask] = props["qmet"]
        count = int(mask.sum())
        print(f"  Label {label_key} ({props['label_name']}): {count} voxels")

    # --- Write thermal_model.npz ---
    npz_out = out / "thermal_model.npz"
    np.savez_compressed(
        str(npz_out),
        rho=rho,
        c=c,
        k=k,
        wb=wb,
        qmet=qmet,
        label_volume=labels,
    )
    print(f"  Wrote: {npz_out}")

    # --- Write thermal_model_manifest.json ---
    manifest: Dict[str, Any] = {
        "source_label_volume": str(Path(label_volume_path).resolve()),
        "source_properties_json": (
            str(Path(properties_json).resolve()) if properties_json else None
        ),
        "volume_shape": list(shape),
        "num_voxels": int(np.prod(shape)),
        "labels_unique": [int(v) for v in np.unique(labels).tolist()],
        "output_dir": str(out.resolve()),
        "npz_arrays": ["rho", "c", "k", "wb", "qmet", "label_volume"],
        "blood_properties": {
            "rho_blood": BLOOD_RHO,
            "c_blood": BLOOD_C,
            "units": "SI (kg, m, s, K, W)",
        },
        "per_label_properties": table,
        "artifacts": {
            "thermal_model.npz": str(npz_out.resolve()),
        },
    }
    manifest_out = out / "thermal_model_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Wrote: {manifest_out}")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Build a thermal model (property arrays) from a label volume. "
            "Outputs a .npz with rho, c, k, wb, qmet, label_volume, plus "
            "a provenance manifest with the per-label property table."
        ),
    )
    ap.add_argument(
        "--label-volume",
        required=True,
        help="Path to label volume (.nii, .nii.gz, or .npy). "
        "Labels should be integer-coded (0=air/background, 1=healthy, 2=lesion).",
    )
    ap.add_argument(
        "--properties-json",
        default=None,
        help="Optional JSON with per-label property overrides. "
        "Only supplied fields override the built-in defaults.",
    )
    ap.add_argument(
        "--output-dir",
        default="./thermal_model_output",
        help="Output directory for thermal model artifacts "
        "(default: ./thermal_model_output).",
    )
    return ap


def main() -> None:
    ap = _build_cli()
    args = ap.parse_args()

    manifest = build_thermal_model(
        label_volume_path=args.label_volume,
        output_dir=args.output_dir,
        properties_json=args.properties_json,
    )

    print("\n=== Thermal Model Build Complete ===")
    print(f"  Output dir: {manifest['output_dir']}")
    print(f"  Volume shape: {manifest['volume_shape']}")
    print(f"  Labels: {manifest['labels_unique']}")
    print(f"  Artifacts:")
    for name, path in manifest["artifacts"].items():
        print(f"    {name}: {path}")


if __name__ == "__main__":
    main()
