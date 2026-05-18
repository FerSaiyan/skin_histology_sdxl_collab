#!/usr/bin/env python
"""
Build MCX-ready artifacts from a volume file (NIfTI or .npy) + optional label map.

Compatible with MCX v2025.10 (``mcx -f mcx_config.json ...``).

Outputs in the specified output directory:
  - mcx_volume.npy         : uint8 or int label array (3D, Python/numpy use)
  - mcx_volume.raw         : raw binary uint8 volume for MCX input (via Domain.Vol)
  - mcx_media_table.json   : optical properties per label with literature defaults
  - mcx_config.json        : MCX v2025.10-compatible input config
  - mcx_build_manifest.json: provenance info about the build

Volume binary layout:
  Numpy C-order bytes from shape (Z,Y,X) produce the same byte layout as
  MCX's column-major interpretation with Dim=[X,Y,Z].  No transpose is needed.
  Set ``OriginType=0`` for 0-based indexing (Python/C convention).

Optical defaults (Jacques 2013 / literature skin values):
  - label 0 (background/air):  μa=0.0001, μs=0.0001, g=1.0, n=1.0
  - label 1 (generic tissue): μa=0.02,   μs=10.0,  g=0.9, n=1.37
  - label 2 (lesion/tumour):  μa=0.05,   μs=8.0,   g=0.85, n=1.4
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
# Optional nibabel (for NIfTI input)
# ---------------------------------------------------------------------------
_HAS_NIBABEL = False
try:
    import nibabel as nib  # noqa: F401

    _HAS_NIBABEL = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Default optical properties per label
#   Keys match the MCX v2025.10 ``Domain.Media`` array format.
#   Index 0 = label 0, index 1 = label 1, etc.
# ---------------------------------------------------------------------------
DEFAULT_OPTICAL_TABLE: Dict[str, Dict[str, float]] = {
    "0": {
        "mua": 0.0001,
        "mus": 0.0001,
        "g": 1.0,
        "n": 1.0,
        "label_name": "background/air",
    },
    "1": {
        "mua": 0.02,
        "mus": 10.0,
        "g": 0.9,
        "n": 1.37,
        "label_name": "generic_tissue",
    },
    "2": {
        "mua": 0.05,
        "mus": 8.0,
        "g": 0.85,
        "n": 1.4,
        "label_name": "lesion",
    },
}

# ---------------------------------------------------------------------------
# Volume loading helpers
# ---------------------------------------------------------------------------


def _load_volume(volume_path: str) -> np.ndarray:
    """Load a 3D volume from .nii/.nii.gz or .npy.

    Returns:
        A 3D numpy array (H, W, D).

    Raises:
        SystemExit on missing file, unsupported extension, or shape < 3D.
    """
    p = Path(volume_path)
    if not p.exists():
        raise SystemExit(f"Volume file not found: {volume_path}")

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
        raise SystemExit(
            f"Volume must be at least 3D, got shape {arr.shape}"
        )

    # Squeeze trailing singleton dims beyond 3
    while arr.ndim > 3:
        # If a trailing dim is >1, we keep it as-is but warn
        if arr.shape[-1] > 1:
            warnings.warn(
                f"Volume has {arr.ndim} dims, using first 3 dimensions. "
                f"Shape: {arr.shape}"
            )
            # Keep first 3 axes (H,W,D)
            arr = arr[(slice(None), slice(None), slice(None)) + (0,) * (arr.ndim - 3)]
            break
        arr = np.squeeze(arr, axis=-1)
    return arr


def _load_label_map(label_path: str, ref_shape) -> np.ndarray:
    """Load and validate a label map (NIfTI or .npy) against reference shape."""
    p = Path(label_path)
    if not p.exists():
        raise SystemExit(f"Label map file not found: {label_path}")

    ext = p.suffix.lower()
    is_nii_gz = p.suffixes[-2:] == [".nii", ".gz"]
    if ext == ".npy":
        labels = np.load(str(p))
    elif ext == ".nii" or is_nii_gz:
        if not _HAS_NIBABEL:
            raise SystemExit(
                "nibabel is required to load NIfTI label maps. "
                "Install with: pip install nibabel"
            )
        nii = nib.load(str(p))
        labels = np.asanyarray(nii.dataobj)
    else:
        raise SystemExit(
            f"Unsupported label extension '{ext}'. "
            "Use .nii, .nii.gz, or .npy."
        )

    if labels.shape != ref_shape:
        # Common case: NIfTI label map with trailing singleton dimension
        while labels.ndim > 3 and labels.shape[-1] == 1:
            labels = np.squeeze(labels, axis=-1)

    if labels.ndim > 3:
        warnings.warn(
            f"Label map has {labels.ndim} dims, using first 3 dimensions. "
            f"Shape: {labels.shape}"
        )
        labels = labels[(slice(None), slice(None), slice(None)) + (0,) * (labels.ndim - 3)]

    if labels.shape != ref_shape:
        raise SystemExit(
            f"Label map shape {labels.shape} does not match "
            f"volume shape {ref_shape}"
        )
    return labels


# ---------------------------------------------------------------------------
# MCX artifact builders
# ---------------------------------------------------------------------------


def _infer_labels_from_volume(volume: np.ndarray) -> np.ndarray:
    """If volume has few unique values, treat as label map; otherwise binarize."""
    unique = np.unique(volume)
    if unique.size <= 20:
        # Already a label map
        return volume.astype(np.int32)
    else:
        # Binarize: > mean(volume) is tissue (1), rest is background (0)
        thresh = float(volume.mean())
        labels = (volume > thresh).astype(np.int32)
        print(
            f"  [INFO] Volume has {unique.size} unique values; "
            f"binarized at mean={thresh:.2f} -> {int(labels.max())} labels"
        )
        return labels


def _build_media_table(labels: np.ndarray) -> Dict[str, Any]:
    """Build media table JSON from observed labels + defaults.

    Returns a dict with two representations:
      - "by_label": per-label dict (for the standalone media table file)
      - "mcx_array": ordered list for MCX v2025.10 inline ``Domain.Media``
    """
    observed = sorted(int(v) for v in np.unique(labels))
    if any(v < 0 for v in observed):
        raise SystemExit(
            f"Label map contains negative label IDs: {observed}. "
            "MCX media labels must be non-negative integers."
        )

    # MCX uses label values as direct indices into Domain.Media.
    # Therefore Domain.Media must be dense from 0..max_label, even if some
    # labels are absent in the current volume.
    max_label = max(observed) if observed else 0
    by_label: Dict[str, Dict[str, Any]] = {}
    mcx_array: list[Dict[str, float]] = []

    for label_id in range(max_label + 1):
        key = str(label_id)
        if key in DEFAULT_OPTICAL_TABLE:
            entry = dict(DEFAULT_OPTICAL_TABLE[key])
        else:
            entry = {
                "mua": 0.02,
                "mus": 10.0,
                "g": 0.9,
                "n": 1.37,
                "label_name": f"label_{label_id}",
            }
        label_name = entry.pop("label_name", f"label_{label_id}")
        entry["label_name"] = label_name
        by_label[key] = dict(entry)
        # MCX inline media: only mua, mus, g, n — no label_name
        mcx_array.append({k: entry[k] for k in ("mua", "mus", "g", "n")})

    return {
        "media": by_label,
        "mcx_array": mcx_array,
        "description": (
            "Optical properties per label. "
            "Defaults based on Jacques 2013 skin/tissue literature values. "
            "mua [1/mm], mus [1/mm], g [unitless], n [unitless]."
        ),
    }


def build_mcx_volume(
    volume_path: str,
    label_path: Optional[str],
    output_dir: str,
    media_table: Optional[Dict[str, Any]] = None,
    nphoton: int = 10000000,
) -> Dict[str, Any]:
    """Core build function.

    Args:
        volume_path: Path to the input volume file.
        label_path: Optional path to label map.
        output_dir: Output directory for artifacts.
        media_table: Optional pre-built media table (otherwise auto-built).
        nphoton: Number of photons for MCX simulation (default: 10,000,000).

    Returns a manifest dict.

    Raises:
        ValueError if *nphoton* is not positive.
    """
    if nphoton <= 0:
        raise ValueError(
            f"nphoton must be > 0, got {nphoton}"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load volume ---
    print(f"Loading volume: {volume_path}")
    volume = _load_volume(volume_path)
    print(f"  Shape: {volume.shape}, dtype={volume.dtype}")

    # --- Load or infer labels ---
    if label_path:
        print(f"Loading label map: {label_path}")
        labels = _load_label_map(label_path, volume.shape)
    else:
        print("  No label map provided; inferring labels from volume...")
        labels = _infer_labels_from_volume(volume)
    print(f"  Labels: unique={sorted(int(v) for v in np.unique(labels))}")

    if labels.min() < 0:
        raise SystemExit("Label IDs must be >= 0 for MCX Domain.Media indexing.")
    if labels.max() > 255:
        raise SystemExit(
            f"Label IDs must be <= 255 for MediaFormat='byte', got max label {int(labels.max())}."
        )

    # --- Write mcx_volume.npy (for Python/analysis use) ---
    vol_out = out / "mcx_volume.npy"
    vol_array = labels.astype(np.int32)
    np.save(str(vol_out), vol_array)
    print(f"  Wrote: {vol_out}  shape={vol_array.shape} dtype={vol_array.dtype}")

    # --- Write mcx_volume.raw (raw binary for MCX v2025.10) ---
    # MCX expects raw binary voxel data.
    # Numpy C-order bytes from shape (Z,Y,X) produce the same byte layout as
    # MCX Fortran-order interpretation with Dim=[X,Y,Z]; see docstring notes.
    raw_out = out / "mcx_volume.raw"
    vol_array.astype(np.uint8).tofile(str(raw_out))
    print(f"  Wrote: {raw_out}  ({raw_out.stat().st_size} bytes, uint8)")

    # --- Write mcx_media_table.json ---
    media = media_table or _build_media_table(labels)
    media_out = out / "mcx_media_table.json"
    media_out.write_text(json.dumps(media, indent=2), encoding="utf-8")
    print(f"  Wrote: {media_out}")

    # --- Build MCX v2025.10-compatible config JSON ---
    dim_np = list(vol_array.shape)          # [Z, Y, X]
    dim_mcx = [dim_np[2], dim_np[1], dim_np[0]]  # [X, Y, Z] for MCX
    # Place source at centre of XY plane, Z=0 (top surface)
    cx, cy = dim_np[2] // 2, dim_np[1] // 2
    # Positions use 0-based indexing (OriginType=0)
    source_pos = [float(cx), float(cy), 0.0]
    # Forward time gate — 5 ns total (typical for mm-scale tissue)
    t1 = 5e-9
    dt = 5e-9
    config: Dict[str, Any] = {
        "Session": {
            "ID": "skin_mcx_sim",
            "Photons": nphoton,
            "RNGSeed": 0,
            "DoNormalize": True,
            "DoAutoThread": True,
            "DoSaveVolume": True,
            "DoPartialPath": False,
            "OutputFormat": "mc2",
            "OutputType": "F",
        },
        "Forward": {
            "T0": 0.0,
            "T1": t1,
            "Dt": dt,
        },
        "Domain": {
            "MediaFormat": "byte",
            "LengthUnit": 1.0,
            "Media": media["mcx_array"],
            "Dim": dim_mcx,
            "VolumeFile": str(raw_out.resolve()),
            "OriginType": 0,
        },
        "Optode": {
            "Source": {
                "Type": "pencil",
                "Pos": source_pos,
                "Dir": [0.0, 0.0, 1.0, 0.0],
                "Param1": [0.0, 0.0, 0.0, 0.0],
                "Param2": [0.0, 0.0, 0.0, 0.0],
            },
            "Detector": [
                {
                    "Pos": [float(cx), float(cy), 0.0],
                    "R": 1.0,
                },
            ],
        },
        "Notes": (
            "MCX v2025.10-compatible config auto-generated. "
            "Adjust Photons, source Pos/Dir, or Forward time gates as needed."
        ),
    }
    config_out = out / "mcx_config.json"
    config_out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"  Wrote: {config_out}")

    # --- Write mcx_build_manifest.json ---
    manifest: Dict[str, Any] = {
        "source_volume": str(Path(volume_path).resolve()),
        "source_label_map": str(Path(label_path).resolve()) if label_path else None,
        "volume_shape": dim_np,
        "volume_dtype": str(volume.dtype),
        "labels_unique": [int(v) for v in np.unique(labels).tolist()],
        "output_dir": str(out.resolve()),
        "mcx_schema": "v2025.10",
        "artifacts": {
            "mcx_volume.npy": str(vol_out.resolve()),
            "mcx_volume.raw": str(raw_out.resolve()),
            "mcx_media_table.json": str(media_out.resolve()),
            "mcx_config.json": str(config_out.resolve()),
        },
    }
    manifest_out = out / "mcx_build_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Wrote: {manifest_out}")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Build MCX-ready artifacts from a volume file (NIfTI or .npy) "
            "and optional label map."
        )
    )
    ap.add_argument(
        "--volume",
        required=True,
        help="Path to input volume (.nii, .nii.gz, or .npy).",
    )
    ap.add_argument(
        "--label-map",
        default=None,
        help="Optional path to label map (.nii, .nii.gz, or .npy). "
        "If omitted, labels are inferred via binarization at mean intensity "
        "(multi-label if volume already has few unique values).",
    )
    ap.add_argument(
        "--output-dir",
        default="./mcx_build_output",
        help="Output directory for MCX artifacts (default: ./mcx_build_output).",
    )
    ap.add_argument(
        "--nphoton",
        type=int,
        default=10000000,
        help="Number of photons for MCX simulation (default: 10000000). Must be > 0.",
    )
    return ap


def main() -> None:
    ap = _build_cli()
    args = ap.parse_args()

    if args.nphoton <= 0:
        print(f"ERROR: --nphoton must be > 0, got {args.nphoton}", file=sys.stderr)
        sys.exit(1)

    manifest = build_mcx_volume(
        volume_path=args.volume,
        label_path=args.label_map,
        output_dir=args.output_dir,
        nphoton=args.nphoton,
    )

    print("\n=== MCX Build Complete ===")
    print(f"  Output dir: {manifest['output_dir']}")
    print(f"  Volume shape: {manifest['volume_shape']}")
    print(f"  Labels: {manifest['labels_unique']}")
    print(f"  Artifacts:")
    for name, path in manifest["artifacts"].items():
        print(f"    {name}: {path}")


if __name__ == "__main__":
    main()
