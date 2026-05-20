#!/usr/bin/env python
"""
Stack ordered 2D slice images into a NIfTI volume (.nii.gz).

Supports glob patterns and CSV input. All images must have the same dimensions.
Images are loaded as grayscale (single-channel).  If a source is RGB, the
mean across channels is used.

Output: a standard NIfTI-1 file with configurable voxel spacing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sequence_utils import (
    SequenceValidationError,
    collect_ordered_paths,
    validate_contiguous_selection,
)


def _gather_paths(
    input_glob: str | None,
    input_csv: str | None,
    allow_noncontiguous: bool = False,
) -> list[str]:
    """Return sorted list of image file paths."""
    if input_csv:
        import pandas as pd

        df = pd.read_csv(input_csv)
        col = [c for c in df.columns if "path" in c.lower() or "slice" in c.lower()]
        if not col:
            col = [df.columns[0]]
        paths = df[col[0]].tolist()
    elif input_glob:
        paths = [str(p) for p in collect_ordered_paths(input_glob)]
    else:
        raise SystemExit("Provide one of --input-glob or --input-csv.")

    if not paths:
        raise SystemExit("No slice images matched the given pattern / CSV.")

    if not allow_noncontiguous:
        try:
            validate_contiguous_selection(paths, context="NIfTI input slices")
        except SequenceValidationError as exc:
            raise SystemExit(f"Invalid slice sequence: {exc}")
    return paths


def _load_slice_grayscale(path: str) -> np.ndarray:
    """Load an image as a 2D grayscale array (H, W), uint8."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stack ordered 2D slices into a NIfTI volume."
    )
    ap.add_argument(
        "--input-glob",
        default=None,
        help='Glob pattern, e.g. "slices/slice_*.png".',
    )
    ap.add_argument(
        "--input-csv",
        default=None,
        help="CSV with a column containing slice file paths.",
    )
    ap.add_argument("--output-nifti", required=True, help="Output .nii.gz path.")
    ap.add_argument(
        "--pixdim",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("DX", "DY", "DZ"),
        help="Voxel dimensions in mm (default: 1.0 1.0 1.0).",
    )
    ap.add_argument(
        "--stats-json", default=None, help="Optional stats JSON output path."
    )
    ap.add_argument(
        "--allow-noncontiguous-slices",
        action="store_true",
        help="Debug escape hatch. By default, input slices must be contiguous.",
    )
    args = ap.parse_args()

    # --- Gather slice paths ---
    paths = _gather_paths(args.input_glob, args.input_csv, args.allow_noncontiguous_slices)
    print(f"Found {len(paths)} slices.")

    # --- Load and stack ---
    slices: list[np.ndarray] = []
    for p in paths:
        arr = _load_slice_grayscale(p)
        slices.append(arr)

    # Validate shape consistency
    ref_shape = slices[0].shape
    for i, arr in enumerate(slices[1:], start=1):
        if arr.shape != ref_shape:
            raise SystemExit(
                f"Shape mismatch: slice 0 {ref_shape} vs slice {i} {arr.shape} ({paths[i]})"
            )

    # Stack along last axis: (H, W, Z)
    volume = np.stack(slices, axis=-1)
    print(f"Volume shape: {volume.shape}  dtype={volume.dtype}")

    # --- Build NIfTI ---
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = args.pixdim[0]
    affine[1, 1] = args.pixdim[1]
    affine[2, 2] = args.pixdim[2]

    nifti_img = nib.Nifti1Image(volume, affine)
    out_path = Path(args.output_nifti)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti_img, str(out_path))
    print(f"Saved: {out_path}")

    # --- Optional stats ---
    if args.stats_json:
        stats = {
            "num_slices": len(slices),
            "height": int(volume.shape[0]),
            "width": int(volume.shape[1]),
            "depth": int(volume.shape[2]),
            "pixdim": args.pixdim,
            "dtype": str(volume.dtype),
            "min": float(volume.min()),
            "max": float(volume.max()),
            "mean": float(volume.mean()),
            "std": float(volume.std()),
            "output_path": str(out_path.resolve()),
        }
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
