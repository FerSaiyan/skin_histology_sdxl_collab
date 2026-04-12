#!/usr/bin/env python
"""
Build ROI masks directly from Histo-Seg ground-truth segmentation masks.

By default, ROI is defined as all non-background classes (class id != 0).
Outputs are saved as single-channel PNG/JPG-like mask images with 0/255 values.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image


Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


HISTOSEG_COLOR_TO_CLASS_ID: Dict[Tuple[int, int, int], int] = {
    (0, 0, 0): 0,
    (224, 224, 224): 1,
    (96, 96, 96): 2,
    (150, 150, 0): 3,
    (127, 255, 255): 4,
    (255, 156, 0): 5,
    (255, 0, 255): 6,
    (0, 255, 0): 7,
    (0, 156, 255): 8,
    (127, 96, 255): 9,
    (112, 48, 160): 10,
    (0, 0, 128): 11,
}


def _parse_id_list(values: Optional[str]) -> Set[int]:
    if not values:
        return set()
    out: Set[int] = set()
    for token in values.split(","):
        token = token.strip()
        if not token:
            continue
        out.add(int(token))
    return out


def _decode_mask_to_class_ids(mask_rgb: np.ndarray) -> np.ndarray:
    h, w = mask_rgb.shape[:2]
    class_map = np.full((h, w), -1, dtype=np.int16)

    for rgb, class_id in HISTOSEG_COLOR_TO_CLASS_ID.items():
        match = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=2)
        class_map[match] = class_id

    # Unknown colors are treated as background for safety.
    class_map[class_map < 0] = 0
    return class_map


def _build_roi(
    class_map: np.ndarray,
    foreground_mode: str,
    include_class_ids: Set[int],
    exclude_class_ids: Set[int],
) -> np.ndarray:
    if foreground_mode == "all_non_background":
        roi = class_map != 0
    else:
        roi = np.isin(class_map, list(include_class_ids))

    if exclude_class_ids:
        roi = roi & (~np.isin(class_map, list(exclude_class_ids)))

    return (roi.astype(np.uint8) * 255)


def _output_names(row: pd.Series, default_name: str, extra_columns: Iterable[str]) -> List[str]:
    names = [default_name]
    for col in extra_columns:
        if col in row and isinstance(row[col], str) and row[col].strip():
            names.append(row[col].strip())

    dedup: List[str] = []
    seen = set()
    for n in names:
        if n not in seen:
            seen.add(n)
            dedup.append(n)
    return dedup


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ROI masks from GT segmentation masks.")
    ap.add_argument("--csv", required=True, help="CSV path with mask paths and filenames.")
    ap.add_argument("--mask-column", default="mask_path", help="CSV column containing GT mask paths.")
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory where 0/255 ROI masks will be written.",
    )
    ap.add_argument(
        "--primary-name-column",
        default="filename",
        help="CSV column used as the primary output filename.",
    )
    ap.add_argument(
        "--extra-name-columns",
        default="mask_filename",
        help="Comma-separated filename columns to also emit as aliases.",
    )
    ap.add_argument(
        "--foreground-mode",
        choices=["all_non_background", "include_class_ids"],
        default="all_non_background",
        help="ROI mode. Default uses all non-background classes.",
    )
    ap.add_argument(
        "--include-class-ids",
        default="",
        help="Comma-separated class IDs used when foreground-mode=include_class_ids.",
    )
    ap.add_argument(
        "--exclude-class-ids",
        default="",
        help="Comma-separated class IDs to force to background.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.mask_column not in df.columns:
        raise SystemExit(f"CSV missing required column: {args.mask_column}")
    if args.primary_name_column not in df.columns:
        raise SystemExit(f"CSV missing required column: {args.primary_name_column}")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        for p in output_dir.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_ids = _parse_id_list(args.include_class_ids)
    exclude_ids = _parse_id_list(args.exclude_class_ids)
    if args.foreground_mode == "include_class_ids" and not include_ids:
        raise SystemExit("foreground-mode=include_class_ids requires --include-class-ids")

    extra_cols = [x.strip() for x in args.extra_name_columns.split(",") if x.strip()]

    rows_total = len(df)
    rows_written = 0
    coverage: List[float] = []

    for i, row in df.iterrows():
        mask_path = Path(str(row[args.mask_column]))
        if not mask_path.is_file():
            print(f"[GT ROI] Missing mask: {mask_path}")
            continue

        with Image.open(mask_path) as im:
            rgb = np.array(im.convert("RGB"))
        class_map = _decode_mask_to_class_ids(rgb)
        roi = _build_roi(
            class_map=class_map,
            foreground_mode=args.foreground_mode,
            include_class_ids=include_ids,
            exclude_class_ids=exclude_ids,
        )
        coverage.append(float((roi > 0).mean()))

        primary_name = str(row[args.primary_name_column])
        out_names = _output_names(row=row, default_name=primary_name, extra_columns=extra_cols)
        for out_name in out_names:
            out_path = output_dir / out_name
            Image.fromarray(roi).save(out_path)

        rows_written += 1
        if (i + 1) % 25 == 0 or (i + 1) == rows_total:
            print(f"[GT ROI] Processed {i + 1}/{rows_total}")

    meta = {
        "csv": str(Path(args.csv).resolve()),
        "mask_column": args.mask_column,
        "output_dir": str(output_dir.resolve()),
        "foreground_mode": args.foreground_mode,
        "include_class_ids": sorted(include_ids),
        "exclude_class_ids": sorted(exclude_ids),
        "rows_total": int(rows_total),
        "rows_written": int(rows_written),
        "coverage_mean": float(np.mean(coverage)) if coverage else 0.0,
        "coverage_median": float(np.median(coverage)) if coverage else 0.0,
    }
    (output_dir / "build_roi_masks_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[GT ROI] Done. Wrote {rows_written} masks to {output_dir}")


if __name__ == "__main__":
    main()
