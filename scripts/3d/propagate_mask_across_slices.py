#!/usr/bin/env python
"""
Replicate a single 2D binary mask across a slice stack (cylindrical strategy).

If --mask is provided, that mask is replicated identically for all slices.
If --mask is omitted, a centered circular (or elliptical) mask is generated
using --height, --width, --radius, and --shape-mode.

Output: mask_slice_<index>.png files + optional stats JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def _make_circular_mask(
    height: int, width: int, radius_frac: float, shape_mode: str,
    center_x: int | None = None, center_y: int | None = None,
) -> np.ndarray:
    """Generate a centered (or custom-center) binary mask.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        radius_frac: Radius as fraction of the smaller dimension (0.0–1.0).
        shape_mode: 'circle' or 'ellipse'.
        center_x: X-coordinate of mask center. Defaults to image center.
        center_y: Y-coordinate of mask center. Defaults to image center.

    Returns:
        uint8 array (H, W) with values 0 or 255.
    """
    Y, X = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0 if center_y is None else float(center_y)
    cx = (width - 1) / 2.0 if center_x is None else float(center_x)

    if shape_mode == "circle":
        r = min(height, width) * radius_frac / 2.0
        mask_vals = (X - cx) ** 2 + (Y - cy) ** 2 <= r**2
    elif shape_mode == "ellipse":
        rx = width * radius_frac / 2.0
        ry = height * radius_frac / 2.0
        mask_vals = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0
    else:
        raise ValueError(f"Unknown shape_mode: {shape_mode}")

    return mask_vals.astype(np.uint8) * 255


def _load_mask(path: str) -> np.ndarray:
    """Load a grayscale mask and binarize at 127; return (H, W) uint8 {0, 255}."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    return (arr > 127).astype(np.uint8) * 255


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replicate a 2D mask across a slice stack (cylindrical strategy)."
    )
    ap.add_argument(
        "--mask",
        default=None,
        help="Reference mask image to replicate (grayscale, thresholded at 127).",
    )
    ap.add_argument("--num-slices", type=int, default=10, help="Number of slices.")
    ap.add_argument(
        "--height", type=int, default=512, help="Mask height (ignored if --mask given)."
    )
    ap.add_argument(
        "--width", type=int, default=512, help="Mask width (ignored if --mask given)."
    )
    ap.add_argument(
        "--radius",
        type=float,
        default=0.3,
        help="Radius as fraction of smaller dim (0–1). Default 0.3.",
    )
    ap.add_argument(
        "--shape-mode",
        choices=["circle", "ellipse"],
        default="circle",
        help="Shape of generated mask (ignored if --mask given).",
    )
    ap.add_argument(
        "--center-x", type=int, default=None,
        help="X-coordinate of mask center. Defaults to image center.",
    )
    ap.add_argument(
        "--center-y", type=int, default=None,
        help="Y-coordinate of mask center. Defaults to image center.",
    )
    ap.add_argument("--output-dir", required=True, help="Output directory for masks.")
    ap.add_argument(
        "--stats-json", default=None, help="Optional path for stats JSON."
    )
    ap.add_argument(
        "--dtype",
        default="png",
        choices=["png", "npy"],
        help="Output format. PNG (binary) or NPY (raw array).",
    )
    args = ap.parse_args()

    # --- Input validation ---
    if args.num_slices <= 0:
        ap.error("--num-slices must be > 0")
    if not args.mask:
        if args.height <= 0:
            ap.error("--height must be > 0 when --mask is not provided")
        if args.width <= 0:
            ap.error("--width must be > 0 when --mask is not provided")
        if args.radius <= 0 or args.radius > 1.0:
            ap.error("--radius must be in (0, 1] when --mask is not provided")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Determine the mask ---
    if args.mask:
        mask = _load_mask(args.mask)
        print(f"Loaded reference mask: {args.mask}  shape={mask.shape}")
    else:
        mask = _make_circular_mask(
            args.height, args.width, args.radius, args.shape_mode,
            center_x=args.center_x, center_y=args.center_y,
        )
        center_str = ""
        if args.center_x is not None and args.center_y is not None:
            center_str = f"  center=({args.center_x},{args.center_y})"
        print(
            f"Generated {args.shape_mode} mask  shape={mask.shape}  "
            f"radius_frac={args.radius}{center_str}"
        )

    # --- Write each slice ---
    fg_pixels = int(mask.sum() / 255)
    total_pixels = mask.shape[0] * mask.shape[1]
    fg_frac = fg_pixels / total_pixels if total_pixels > 0 else 0.0

    for i in range(args.num_slices):
        stem = f"mask_slice_{i:04d}"
        if args.dtype == "png":
            out_path = out_dir / f"{stem}.png"
            Image.fromarray(mask).save(out_path)
        else:
            out_path = out_dir / f"{stem}.npy"
            np.save(str(out_path), mask)

    print(
        f"Wrote {args.num_slices} masks to {out_dir}  "
        f"(fg_frac={fg_frac:.4f}, shape={mask.shape})"
    )

    # --- Optional stats ---
    if args.stats_json:
        stats = {
            "num_slices": args.num_slices,
            "height": int(mask.shape[0]),
            "width": int(mask.shape[1]),
            "mask_source": "reference" if args.mask else args.shape_mode,
            "reference_mask": args.mask,
            "foreground_pixels": fg_pixels,
            "foreground_fraction": round(fg_frac, 6),
            "output_format": args.dtype,
            "output_dir": str(out_dir.resolve()),
        }
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
