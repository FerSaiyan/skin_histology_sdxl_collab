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
FILENAME_GROUP_TO_COARSE = {
    "A": "non_cancer",
    "B": "cancer",
    "C": "cancer",
    "D": "cancer",
}

# Histo-Seg masks are RGB-coded with a fixed 12-color palette.
# This mapping follows the dataset class order in the publication description.
HISTOSEG_COLOR_TO_CLASS_ID: Dict[Tuple[int, int, int], int] = {
    (0, 0, 0): 0,          # background
    (224, 224, 224): 1,    # epidermis
    (96, 96, 96): 2,       # reticular dermis
    (150, 150, 0): 3,      # papillary dermis
    (127, 255, 255): 4,    # dermis
    (255, 156, 0): 5,      # keratin
    (255, 0, 255): 6,      # inflammation
    (0, 255, 0): 7,        # hair follicles
    (0, 156, 255): 8,      # glands
    (127, 96, 255): 9,     # basal cell carcinoma
    (112, 48, 160): 10,    # squamous cell carcinoma
    (0, 0, 128): 11,       # intraepidermal carcinoma
}


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


def _parse_group_code(stem: str) -> str:
    m = re.search(r"\(([A-Z])\d-\d\)", stem)
    return m.group(1) if m else "UNK"


def _mask_stats(mask_path: Path) -> Tuple[List[int], int]:
    im = Image.open(mask_path)
    colors = im.getcolors(maxcolors=16_777_216)
    if colors is None:
        raise ValueError(f"Too many unique colors in mask (unexpected): {mask_path}")

    class_counts: Dict[int, int] = {}
    for count, color in colors:
        if isinstance(color, int):
            class_id = int(color)
        else:
            rgb = tuple(int(x) for x in color[:3])
            class_id = int(HISTOSEG_COLOR_TO_CLASS_ID.get(rgb, -1))
        class_counts[class_id] = class_counts.get(class_id, 0) + int(count)

    class_ids = sorted(x for x in class_counts.keys() if x >= 0)

    non_bg = [(cid, cnt) for cid, cnt in class_counts.items() if cid > 0]
    dominant_class_id = 0
    if non_bg:
        non_bg.sort(key=lambda x: x[1], reverse=True)
        dominant_class_id = int(non_bg[0][0])

    return class_ids, dominant_class_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Create Histo-Seg paired CSV for Grad-CAM and LoRA training.")
    ap.add_argument("--dataset-dir", required=True, help="Directory containing .jpg images and .png masks.")
    ap.add_argument("--output-csv", required=True, help="Output CSV path.")
    ap.add_argument("--stats-json", default=None, help="Optional summary JSON path.")
    ap.add_argument(
        "--coarse-label-mode",
        choices=["filename_group", "mask_classes"],
        default="filename_group",
        help=(
            "How to derive coarse_label. "
            "filename_group: A->non_cancer, B/C/D->cancer (default). "
            "mask_classes: cancer if class id 9/10/11 present in mask."
        ),
    )
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

        group_code = _parse_group_code(stem)
        if args.coarse_label_mode == "filename_group":
            coarse_label = FILENAME_GROUP_TO_COARSE.get(group_code, "non_cancer")
        else:
            coarse_label = "cancer" if any(x in CANCER_CLASS_IDS for x in non_bg_ids) else "non_cancer"

        volume_id, slice_index = _parse_volume_slice(stem)

        rows.append(
            {
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "filename": img_path.name,
                "mask_filename": mask_path.name,
                "slice_id": stem,
                "group_code": group_code,
                "volume_id": volume_id,
                "slice_index": slice_index,
                "classes_present": ",".join(str(x) for x in class_ids),
                "dominant_class_id": dominant_class_id,
                "histology_label": histology_label,
                "coarse_label": coarse_label,
                "coarse_label_source": args.coarse_label_mode,
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
