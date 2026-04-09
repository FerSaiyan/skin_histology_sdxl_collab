#!/usr/bin/env python
"""
Build paired image/mask CSV for Histo-Seg dataset.

Outputs a CSV with columns suitable for Grad-CAM and SDXL inpainting prep:
  - image_path
  - mask_path
  - filename
  - mask_filename
  - slice_id
  - volume_id
  - slice_index
  - classes_present
  - dominant_class_id
  - histology_label
  - coarse_label

The default coarse label is binary:
  - "cancer" if any class id in {9,10,11} is present in mask
  - "non_cancer" otherwise
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


CLASS_NAME_BY_ID: Dict[int, str] = {
    0: "background",
    1: "epidermis",
    2: "reticular_dermis",
    3: "papillary_dermis",
    4: "dermis",
    5: "keratin",
    6: "inflammation",
    7: "hair_follicles",
    8: "glands",
    9: "basal_cell_carcinoma",
    10: "squamous_cell_carcinoma",
    11: "intraepidermal_carcinoma",
}

CANCER_CLASS_IDS = {9, 10, 11}


def _parse_volume_slice(stem: str) -> Tuple[str, int]:
    """
    Best-effort parsing for names like: MD22-04144(B1-3)
    Returns (volume_id, slice_index). If parsing fails -> (stem, -1).
    """
    normalized = stem[:-1] if stem.endswith(")") else stem
    m = re.match(r"^(.*?)-(\d+)$", normalized)
    if not m:
        return stem, -1

    volume_base = m.group(1)
    if stem.endswith(")") and not volume_base.endswith(")"):
        volume_base = volume_base + ")"
    return volume_base, int(m.group(2))


def _mask_stats(mask_path: Path) -> Tuple[List[int], int]:
    arr = np.array(Image.open(mask_path))

    channel = None
    if arr.ndim == 2:
        channel = arr

    elif arr.ndim == 3:
        if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
            channel = arr[..., 0]
        else:
            # Fallback: non-grayscale RGB mask. Use first channel to keep pipeline moving.
            channel = arr[..., 0]

    else:
        raise ValueError(f"Unexpected mask dimensions for {mask_path}: {arr.shape}")

    vals, counts = np.unique(channel, return_counts=True)
    class_ids = sorted(int(x) for x in vals.tolist())

    dominant_class_id = 0
    non_bg = [(int(v), int(c)) for v, c in zip(vals.tolist(), counts.tolist()) if int(v) != 0]
    if non_bg:
        non_bg.sort(key=lambda x: x[1], reverse=True)
        dominant_class_id = int(non_bg[0][0])

    return class_ids, dominant_class_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Create Histo-Seg paired CSV for Grad-CAM and LoRA training.")
    ap.add_argument("--dataset-dir", required=True, help="Directory containing .jpg images and .png masks.")
    ap.add_argument("--output-csv", required=True, help="Output CSV path.")
    ap.add_argument("--stats-json", default=None, help="Optional summary JSON path.")
    args = ap.parse_args()

    Image.MAX_IMAGE_PIXELS = None
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")

    jpgs = sorted(dataset_dir.glob("*.jpg"))
    pngs = sorted(dataset_dir.glob("*.png"))

    mask_by_stem = {p.stem: p for p in pngs}
    rows: List[Dict] = []
    missing_masks = []

    for img_path in jpgs:
        stem = img_path.stem
        mask_path = mask_by_stem.get(stem)
        if mask_path is None:
            missing_masks.append(img_path.name)
            continue

        class_ids, dominant_class_id = _mask_stats(mask_path)
        non_bg_ids = [x for x in class_ids if x != 0]

        if dominant_class_id != 0:
            histology_label = CLASS_NAME_BY_ID.get(dominant_class_id, f"class_{dominant_class_id}")
        else:
            histology_label = CLASS_NAME_BY_ID[0]

        coarse_label = "cancer" if any(x in CANCER_CLASS_IDS for x in non_bg_ids) else "non_cancer"
        volume_id, slice_index = _parse_volume_slice(stem)

        rows.append(
            {
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "filename": img_path.name,
                "mask_filename": mask_path.name,
                "slice_id": stem,
                "volume_id": volume_id,
                "slice_index": slice_index,
                "classes_present": ",".join(str(x) for x in class_ids),
                "dominant_class_id": dominant_class_id,
                "histology_label": histology_label,
                "coarse_label": coarse_label,
            }
        )

    if not rows:
        raise SystemExit("No image/mask pairs found. Check dataset-dir and file names.")

    df = pd.DataFrame(rows).sort_values(["volume_id", "slice_index", "filename"]).reset_index(drop=True)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Wrote paired CSV: {out_csv}")
    print(f"Pairs: {len(df)} | Missing masks: {len(missing_masks)}")
    print("coarse_label counts:")
    print(df["coarse_label"].value_counts())

    if args.stats_json:
        stats = {
            "dataset_dir": str(dataset_dir.resolve()),
            "total_jpg": len(jpgs),
            "total_png": len(pngs),
            "paired_rows": len(df),
            "missing_masks": missing_masks,
            "coarse_label_counts": df["coarse_label"].value_counts().to_dict(),
            "histology_label_counts": df["histology_label"].value_counts().to_dict(),
        }
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Wrote stats JSON: {stats_path}")


if __name__ == "__main__":
    main()
