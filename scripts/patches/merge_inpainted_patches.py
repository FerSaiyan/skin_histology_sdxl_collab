#!/usr/bin/env python
"""
Paste inpainted patches back into full-resolution source slices.

This script:
- Loads original full-resolution slices
- Loads inpainted patches and their metadata
- Upscales patches back to original extraction size
- Blends inpainted region into original image with feathering
- Saves edited full-resolution slices with audit metadata
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _create_feather_mask(
    shape: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    feather_radius: int = 16,
) -> np.ndarray:
    """Create a feathered mask for blending at ROI boundaries."""
    y_min, x_min, y_max, x_max = bbox
    mask = np.zeros(shape, dtype=np.float32)

    mask[y_min:y_max, x_min:x_max] = 1.0

    if feather_radius > 0:
        from scipy.ndimage import gaussian_filter

        mask = gaussian_filter(mask, sigma=feather_radius / 2.0)

        y_min_f = max(0, y_min - feather_radius)
        y_max_f = min(shape[0], y_max + feather_radius)
        x_min_f = max(0, x_min - feather_radius)
        x_max_f = min(shape[1], x_max + feather_radius)

        full_mask = np.zeros(shape, dtype=np.float32)
        full_mask[y_min_f:y_max_f, x_min_f:x_max_f] = 1.0
        mask = mask * full_mask

    return mask


def _blend_images(
    original: np.ndarray,
    edited_patch: np.ndarray,
    bbox: Tuple[int, int, int, int],
    feather_radius: int = 16,
) -> np.ndarray:
    """Blend edited patch into original image with feathering."""
    result = original.copy()
    y_min, x_min, y_max, x_max = bbox

    patch_h = y_max - y_min
    patch_w = x_max - x_min

    if edited_patch.shape[0] != patch_h or edited_patch.shape[1] != patch_w:
        edited_pil = Image.fromarray(edited_patch)
        edited_pil = edited_pil.resize((patch_w, patch_h), Image.Resampling.BILINEAR)
        edited_patch = np.array(edited_pil)

    feather_mask = _create_feather_mask(original.shape[:2], bbox, feather_radius)

    if len(original.shape) == 3:
        feather_mask = feather_mask[:, :, np.newaxis]

    result_region = result[y_min:y_max, x_min:x_max]

    if edited_patch.shape[:2] != result_region.shape[:2]:
        edited_patch = np.array(
            Image.fromarray(edited_patch).resize(
                (result_region.shape[1], result_region.shape[0]),
                Image.Resampling.BILINEAR,
            )
        )

    blended_region = (
        result_region * (1 - feather_mask[y_min:y_max, x_min:x_max])
        + edited_patch * feather_mask[y_min:y_max, x_min:x_max]
    ).astype(np.uint8)

    result[y_min:y_max, x_min:x_max] = blended_region

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Paste inpainted patches back into source slices.")
    ap.add_argument("--inpaint-metadata", required=True, help="Path to inpaint_metadata.csv")
    ap.add_argument("--extract-metadata", required=True, help="Path to patches_metadata.csv")
    ap.add_argument("--output-dir", required=True, help="Output directory for merged slices")
    ap.add_argument("--feather-radius", type=int, default=16, help="Feather radius for blending")
    ap.add_argument("--max-slices", type=int, default=0, help="Limit slices (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = ap.parse_args()

    inpaint_meta = pd.read_csv(args.inpaint_metadata)
    extract_meta = pd.read_csv(args.extract_metadata)

    merged = inpaint_meta.merge(
        extract_meta,
        on="slice_id",
        suffixes=("_inpaint", "_extract"),
    )

    if args.max_slices > 0:
        merged = merged.head(args.max_slices)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "slices").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Slices to process: {len(merged)}")
        for idx, row in merged.iterrows():
            print(f"  [{idx}] {row['slice_id']}")
        return

    merge_metadata: List[Dict[str, Any]] = []

    for idx, row in merged.iterrows():
        slice_id = row["slice_id"]

        source_image_path = Path(row["source_image"])
        inpainted_path = Path(row["inpainted_image"])

        if not source_image_path.exists():
            print(f"[WARN] Source image not found: {source_image_path}")
            continue

        if not inpainted_path.exists():
            print(f"[WARN] Inpainted image not found: {inpainted_path}")
            continue

        print(f"[{idx+1}/{len(merged)}] Merging {slice_id}")

        original = np.array(Image.open(source_image_path).convert("RGB"))
        inpainted = np.array(Image.open(inpainted_path).convert("RGB"))

        bbox = (
            int(row["bbox_y_min"]),
            int(row["bbox_x_min"]),
            int(row["bbox_y_max"]),
            int(row["bbox_x_max"]),
        )

        merged_img = _blend_images(original, inpainted, bbox, args.feather_radius)

        out_path = output_dir / "slices" / f"{slice_id}_edited.png"
        Image.fromarray(merged_img).save(out_path)

        merge_meta = {
            "slice_id": slice_id,
            "source_image": str(source_image_path),
            "inpainted_patch": str(inpainted_path),
            "merged_image": str(out_path),
            "bbox_y_min": int(bbox[0]),
            "bbox_x_min": int(bbox[1]),
            "bbox_y_max": int(bbox[2]),
            "bbox_x_max": int(bbox[3]),
            "feather_radius": args.feather_radius,
            "original_shape": list(original.shape),
            "patch_size": [int(row["bbox_height"]), int(row["bbox_width"])],
            "target_size": int(row["target_size"]),
            "seed": int(row["seed"]),
            "prompt": row["prompt"],
            "coarse_label": row.get("coarse_label", ""),
            "merged_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        merge_metadata.append(merge_meta)

    if merge_metadata:
        merge_df = pd.DataFrame(merge_metadata)
        merge_meta_path = output_dir / "metadata" / "merge_metadata.csv"
        merge_df.to_csv(merge_meta_path, index=False)
        print(f"Saved merge metadata: {merge_meta_path}")

        stats = {
            "total_slices_merged": len(merge_metadata),
            "feather_radius": args.feather_radius,
        }
        stats_path = output_dir / "metadata" / "merge_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
