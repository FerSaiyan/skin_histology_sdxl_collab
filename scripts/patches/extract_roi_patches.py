#!/usr/bin/env python
"""
Extract ROI-centered patches from large histology slices.

For each slice in the pairs CSV:
- Load the corresponding ROI mask (from Grad-CAM stage)
- Compute bounding box of the mask
- Extract a square patch around the ROI with context padding
- Resize patch to model working resolution
- Save patch image, patch mask, and metadata

Inputs:
- --csv: path to histoseg_pairs.csv
- --image-dir: directory containing source .jpg images
- --mask-dir: directory containing ROI masks (output from Grad-CAM stage)
- --output-dir: directory to save extracted patches

Outputs:
- {output_dir}/patches/{slice_id}_patch.png
- {output_dir}/masks/{slice_id}_mask.png
- {output_dir}/metadata/patches_metadata.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _compute_bbox(
    mask: np.ndarray, padding_ratio: float = 0.15
) -> Tuple[int, int, int, int]:
    """
    Compute bounding box of non-zero mask region with padding.

    Returns (y_min, x_min, y_max, x_max).
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not rows.any() or not cols.any():
        return 0, 0, mask.shape[0], mask.shape[1]

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    y_min, y_max = y_indices[0], y_indices[-1] + 1
    x_min, x_max = x_indices[0], x_indices[-1] + 1

    h = y_max - y_min
    w = x_max - x_min

    pad_y = int(h * padding_ratio)
    pad_x = int(w * padding_ratio)

    y_min = max(0, y_min - pad_y)
    y_max = min(mask.shape[0], y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(mask.shape[1], x_max + pad_x)

    return y_min, x_min, y_max, x_max


def _make_square(
    y_min: int, x_min: int, y_max: int, x_max: int, max_h: int, max_w: int
) -> Tuple[int, int, int, int]:
    """Make bbox square by extending to larger dimension."""
    h = y_max - y_min
    w = x_max - x_min

    if h > w:
        diff = h - w
        half = diff // 2
        x_min = max(0, x_min - half)
        x_max = min(max_w, x_min + h)
        if x_max - x_min < h:
            x_min = max(0, x_max - h)
    elif w > h:
        diff = w - h
        half = diff // 2
        y_min = max(0, y_min - half)
        y_max = min(max_h, y_min + w)
        if y_max - y_min < w:
            y_min = max(0, y_max - w)

    return y_min, x_min, y_max, x_max


def _to_fixed_size_bbox(
    y_min: int, x_min: int, y_max: int, x_max: int,
    patch_size: int, max_h: int, max_w: int,
) -> Tuple[int, int, int, int]:
    """Produce a patch_size x patch_size bbox centered on the input bbox center.

    Tries to shift the crop to stay within image bounds.  If the image is
    smaller than patch_size on a given axis, the result is clamped to image
    bounds (and the caller must zero-pad via _extract_patch_padded).
    """
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2
    half = patch_size // 2

    y_min_f = cy - half
    if patch_size % 2 == 0:
        y_max_f = cy + half
    else:
        y_max_f = cy + half + 1
    x_min_f = cx - half
    if patch_size % 2 == 0:
        x_max_f = cx + half
    else:
        x_max_f = cx + half + 1

    # Shift to stay within vertical bounds if possible
    if y_min_f < 0:
        y_max_f -= y_min_f
        y_min_f = 0
    if y_max_f > max_h:
        y_min_f -= (y_max_f - max_h)
        y_max_f = max_h

    # Shift to stay within horizontal bounds if possible
    if x_min_f < 0:
        x_max_f -= x_min_f
        x_min_f = 0
    if x_max_f > max_w:
        x_min_f -= (x_max_f - max_w)
        x_max_f = max_w

    # Final clamp
    y_min_f = max(0, y_min_f)
    y_max_f = min(max_h, y_max_f)
    x_min_f = max(0, x_min_f)
    x_max_f = min(max_w, x_max_f)

    return y_min_f, x_min_f, y_max_f, x_max_f


def _extract_patch(
    image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract patch from image and mask using bbox."""
    y_min, x_min, y_max, x_max = bbox
    patch_img = image[y_min:y_max, x_min:x_max]
    patch_mask = mask[y_min:y_max, x_min:x_max]
    return patch_img, patch_mask


def _extract_patch_padded(
    image: np.ndarray, mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract patch, zero-padding if bbox extends outside image.

    The bbox may be partially or fully outside the image (after
    _to_fixed_size_bbox shifting + clamping).  This function extracts
    the intersection and zero-pads to match the requested bbox size.
    """
    y_min, x_min, y_max, x_max = bbox
    img_h, img_w = image.shape[:2]

    # Intersect with image bounds
    y_min_c = max(0, y_min)
    y_max_c = min(img_h, y_max)
    x_min_c = max(0, x_min)
    x_max_c = min(img_w, x_max)

    # Extract valid region
    patch_img = image[y_min_c:y_max_c, x_min_c:x_max_c]
    patch_mask = mask[y_min_c:y_max_c, x_min_c:x_max_c]

    # Compute needed padding
    pad_top = max(0, -y_min)
    pad_bottom = max(0, y_max - img_h)
    pad_left = max(0, -x_min)
    pad_right = max(0, x_max - img_w)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        if patch_img.ndim == 3:
            patch_img = np.pad(
                patch_img,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant', constant_values=0,
            )
        else:
            patch_img = np.pad(
                patch_img,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant', constant_values=0,
            )
        patch_mask = np.pad(
            patch_mask,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant', constant_values=0,
        )

    return patch_img, patch_mask


def _resize_patch(
    patch_img: np.ndarray,
    patch_mask: np.ndarray,
    target_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize patch to target size."""
    pil_img = Image.fromarray(patch_img)
    pil_mask = Image.fromarray(patch_mask)

    pil_img = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    pil_mask = pil_mask.resize((target_size, target_size), Image.Resampling.NEAREST)

    return np.array(pil_img), np.array(pil_mask)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract ROI patches from histology slices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Crop modes:\n"
            "  auto (default):       Variable-sized crop determined by mask bbox + padding + square.\n"
            "  fixed:                Fixed-size crop of exactly --patch-size x --patch-size centered on\n"
            "                        the mask ROI bbox center (with zero-padding at boundaries), then\n"
            "                        resized to --target-size.\n"
            "  strict_fixed_raw:     Strict raw-bbox mode (no padding, no square transform).\n"
            "                        Computes raw mask bounding box, centers --patch-size x --patch-size\n"
            "                        tile on bbox center, clamps to image boundaries (no zero-padding).\n"
            "                        Use when raw bbox is guaranteed <= --patch-size.\n"
        ),
    )
    ap.add_argument("--csv", required=True, help="Path to histoseg_pairs.csv")
    ap.add_argument("--image-dir", required=True, help="Directory containing source .jpg images")
    ap.add_argument("--mask-dir", required=True, help="Directory containing ROI masks")
    ap.add_argument("--output-dir", required=True, help="Output directory for patches")
    ap.add_argument("--patch-size", type=int, default=1024,
                    help=("Extraction crop size in pixels (default: 1024).  "
                          "In crop-mode=fixed, the extracted patch is exactly "
                          "patch_size x patch_size (zero-padded at boundaries).  "
                          "In crop-mode=auto, this value is recorded in metadata "
                          "but does not control the crop."))
    ap.add_argument("--target-size", type=int, default=512,
                    help="Target model resolution after resize (default: 512)")
    ap.add_argument("--padding-ratio", type=float, default=0.15,
                    help="Context padding ratio around ROI bbox (crop-mode=auto only, default: 0.15)")
    ap.add_argument("--crop-mode", choices=["auto", "fixed", "strict_fixed_raw"], default="auto",
                    help=("Crop sizing mode. 'auto' = variable-size from bbox + padding + square "
                          "(legacy).  'fixed' = exactly --patch-size x --patch-size centered on "
                          "ROI center with zero-padding at boundaries.  'strict_fixed_raw' = raw "
                          "bbox (no padding/square), tile centered and clamped to bounds (default: auto)"))
    ap.add_argument("--max-slices", type=int, default=0, help="Limit number of slices (0 = all)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    np.random.seed(args.seed)

    csv_path = Path(args.csv)
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "patches").mkdir(exist_ok=True)
    (output_dir / "masks").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)

    if args.max_slices > 0:
        df = df.head(args.max_slices)

    metadata: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        slice_id = row["slice_id"]
        img_path = image_dir / row["filename"]

        mask_filename = str(row.get("mask_filename", ""))
        mask_path = mask_dir / mask_filename if mask_filename else None

        # Grad-CAM masks are usually saved with source image basename (e.g., .jpg).
        if mask_path is None or not mask_path.exists():
            fallback_name = str(row.get("filename", ""))
            fallback_path = mask_dir / fallback_name if fallback_name else None
            if fallback_path and fallback_path.exists():
                mask_path = fallback_path
            elif fallback_name:
                stem = Path(fallback_name).stem
                for ext in [".jpg", ".png", ".jpeg"]:
                    cand = mask_dir / f"{stem}{ext}"
                    if cand.exists():
                        mask_path = cand
                        break

        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            continue

        if mask_path is None or not mask_path.exists():
            print(f"[WARN] Mask not found for {slice_id}")
            continue

        print(f"[{idx+1}/{len(df)}] Processing {slice_id}")

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if mask.shape[:2] != img.shape[:2]:
            print(f"[WARN] Mask shape mismatch for {slice_id}: {mask.shape} vs {img.shape}")
            continue

        if args.crop_mode == "strict_fixed_raw":
            # Strict mode: compute raw bbox (no padding, no square)
            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)
            if not rows.any() or not cols.any():
                y_min, x_min, y_max, x_max = 0, 0, img.shape[0], img.shape[1]
            else:
                y_indices = np.where(rows)[0]
                x_indices = np.where(cols)[0]
                y_min = int(y_indices[0])
                y_max = int(y_indices[-1]) + 1
                x_min = int(x_indices[0])
                x_max = int(x_indices[-1]) + 1

            bbox = _to_fixed_size_bbox(
                y_min, x_min, y_max, x_max,
                args.patch_size, img.shape[0], img.shape[1],
            )
            patch_img, patch_mask = _extract_patch(img, mask, bbox)
        else:
            bbox = _compute_bbox(mask, padding_ratio=args.padding_ratio)
            bbox = _make_square(*bbox, img.shape[0], img.shape[1])

            if args.crop_mode == "fixed":
                bbox = _to_fixed_size_bbox(*bbox, args.patch_size, img.shape[0], img.shape[1])
                patch_img, patch_mask = _extract_patch_padded(img, mask, bbox)
            else:
                patch_img, patch_mask = _extract_patch(img, mask, bbox)

        orig_h = bbox[2] - bbox[0]
        orig_w = bbox[3] - bbox[1]

        scale_x = args.target_size / max(orig_w, 1)
        scale_y = args.target_size / max(orig_h, 1)

        patch_img_resized, patch_mask_resized = _resize_patch(
            patch_img, patch_mask, args.target_size
        )

        patch_img_path = output_dir / "patches" / f"{slice_id}_patch.png"
        patch_mask_path = output_dir / "masks" / f"{slice_id}_mask.png"
        patch_mask_alias_path = output_dir / "masks" / f"{slice_id}_patch.png"

        Image.fromarray(patch_img_resized).save(patch_img_path)
        Image.fromarray(patch_mask_resized).save(patch_mask_path)
        Image.fromarray(patch_mask_resized).save(patch_mask_alias_path)

        meta = {
            "slice_id": slice_id,
            "source_image": str(img_path),
            "source_mask": str(mask_path),
            "patch_image": str(patch_img_path),
            "patch_mask": str(patch_mask_path),
            "bbox_y_min": int(bbox[0]),
            "bbox_x_min": int(bbox[1]),
            "bbox_y_max": int(bbox[2]),
            "bbox_x_max": int(bbox[3]),
            "bbox_height": int(orig_h),
            "bbox_width": int(orig_w),
            "target_size": args.target_size,
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "crop_mode": args.crop_mode,
            "patch_size": args.patch_size,
            "coarse_label": row.get("coarse_label", ""),
            "group_code": row.get("group_code", ""),
            "seed_placeholder": 0,
        }
        metadata.append(meta)

    if metadata:
        meta_df = pd.DataFrame(metadata)
        meta_csv_path = output_dir / "metadata" / "patches_metadata.csv"
        meta_df.to_csv(meta_csv_path, index=False)
        print(f"Saved metadata: {meta_csv_path}")

        strict_mode = args.crop_mode == "strict_fixed_raw"
        stats = {
            "total_slices_processed": len(metadata),
            "crop_mode": args.crop_mode,
            "strict_bbox_mode": strict_mode,
            "patch_size": args.patch_size,
            "target_size": args.target_size,
            "padding_ratio": args.padding_ratio if not strict_mode else 0.0,
        }
        stats_path = output_dir / "metadata" / "extraction_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Saved stats: {stats_path}")


if __name__ == "__main__":
    main()
