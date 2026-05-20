#!/usr/bin/env python
"""
Build a metadata CSV compatible with scripts/patches/inpaint_roi_patches.py
from an ordered stack of volume slice images and a mask directory.

Inputs:
  --slice-glob  : glob pattern matching ordered slice images (e.g. "slices/*.png")
  --slice-csv   : CSV with a column containing slice image paths (alternative to glob)
  --mask-dir    : directory containing mask images (one per slice, same basename stem)
  --volume-id   : optional volume identifier for provenance
  --coarse-label: optional label for all slices (passed through to inpaint prompt)
  --output-csv  : output metadata CSV path

Output: CSV with columns:
  slice_id, patch_image, patch_mask, coarse_label, volume_id
(Compatible with inpaint_roi_patches.py which expects slice_id, patch_image, patch_mask.)

Slice order is determined by natural sort of the glob expansion or CSV row order.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sequence_utils import (
    SequenceValidationError,
    collect_ordered_paths,
    validate_contiguous_selection,
)


def _gather_slice_paths(
    slice_glob: str | None, slice_csv: str | None, allow_noncontiguous: bool = False
) -> list[str]:
    """Return sorted list of slice image file paths."""
    if slice_csv:
        import pandas as pd

        df = pd.read_csv(slice_csv)
        col_candidates = [c for c in df.columns if "path" in c.lower() or "slice" in c.lower()]
        col = col_candidates[0] if col_candidates else df.columns[0]
        paths = df[col].dropna().tolist()
        # Preserve CSV row order (user-defined ordering)
    elif slice_glob:
        paths = [str(p) for p in collect_ordered_paths(slice_glob)]
    else:
        raise SystemExit("Provide one of --slice-glob or --slice-csv.")

    if not paths:
        raise SystemExit("No slice images matched the given pattern / CSV.")
    if not allow_noncontiguous:
        try:
            validate_contiguous_selection(paths, context="metadata slice input")
        except SequenceValidationError as exc:
            raise SystemExit(f"Invalid slice sequence: {exc}")
    return paths


def _find_mask_for_slice(
    slice_path: str, mask_dir: str, mask_ext: str | None
) -> str | None:
    """Find a mask image matching the slice filename stem in mask_dir."""
    stem = Path(slice_path).stem
    mask_dir_p = Path(mask_dir)

    if mask_ext:
        candidate = mask_dir_p / f"{stem}{mask_ext}"
        if candidate.exists():
            return str(candidate.resolve())

    # Try common extensions
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"]:
        candidate = mask_dir_p / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate.resolve())

    # Try with "mask_" prefix
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"]:
        candidate = mask_dir_p / f"mask_{stem}{ext}"
        if candidate.exists():
            return str(candidate.resolve())

    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build inpaint metadata CSV from volume slice images and masks."
    )
    ap.add_argument(
        "--slice-glob", default=None,
        help='Glob pattern for ordered slice images, e.g. "data/raw/my_volume/slice_*.png".',
    )
    ap.add_argument(
        "--slice-csv", default=None,
        help="CSV with a column containing slice image paths (alternative to --slice-glob).",
    )
    ap.add_argument(
        "--mask-dir", required=True,
        help="Directory containing mask images (matched by filename stem to slices).",
    )
    ap.add_argument(
        "--mask-ext", default=None,
        help="Explicit mask file extension (e.g. '.png'). Auto-detected if not given.",
    )
    ap.add_argument(
        "--volume-id", default="volume_001",
        help="Optional volume identifier for provenance metadata.",
    )
    ap.add_argument(
        "--coarse-label", default="non_cancer",
        help="Label for all slices (e.g. 'cancer' or 'non_cancer'). Passed through to inpaint prompt.",
    )
    ap.add_argument(
        "--output-csv", required=True,
        help="Output metadata CSV path.",
    )
    ap.add_argument(
        "--allow-noncontiguous-slices",
        action="store_true",
        help="Debug escape hatch. By default, input slices must be contiguous.",
    )
    args = ap.parse_args()

    if not args.slice_glob and not args.slice_csv:
        ap.error("Provide one of --slice-glob or --slice-csv.")
    if args.slice_glob and args.slice_csv:
        ap.error("Provide only one of --slice-glob or --slice-csv (not both).")

    # --- Gather slice paths ---
    slice_paths = _gather_slice_paths(args.slice_glob, args.slice_csv, args.allow_noncontiguous_slices)
    print(f"Found {len(slice_paths)} slice images.")

    # --- Resolve mask paths ---
    mask_dir = Path(args.mask_dir)
    if not mask_dir.is_dir():
        raise SystemExit(f"Mask directory does not exist: {mask_dir}")

    rows = []
    missing_masks = 0
    for idx, sp in enumerate(slice_paths):
        sp_abs = str(Path(sp).resolve())
        mask_path = _find_mask_for_slice(sp_abs, str(mask_dir), args.mask_ext)
        if mask_path is None:
            print(f"  [WARN] No mask found for slice {idx}: {sp}")
            missing_masks += 1
            continue

        slice_id = f"{args.volume_id}_slice_{idx:04d}"
        rows.append({
            "slice_id": slice_id,
            "patch_image": sp_abs,
            "patch_mask": mask_path,
            "coarse_label": args.coarse_label,
            "volume_id": args.volume_id,
        })

    if not rows:
        raise SystemExit("No slice/mask pairs found. Cannot build metadata CSV.")

    # --- Write CSV ---
    import pandas as pd

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote metadata CSV: {out_path}")
    print(f"  Total slice/mask pairs: {len(rows)}")
    print(f"  Volume ID:              {args.volume_id}")
    print(f"  Coarse label:           {args.coarse_label}")
    if missing_masks:
        print(f"  WARNING: {missing_masks} slices had no matching mask.")

    # Print column names for compatibility check
    print(f"  Columns: {list(out_df.columns)}")


if __name__ == "__main__":
    main()
