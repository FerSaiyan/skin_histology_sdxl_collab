#!/usr/bin/env python
"""
Generate synthetic test slice images for volume inpainting smoke tests.

Creates simple gradient-based images (varying across Z) to simulate
a slice stack with known properties for coherence testing.

Usage:
  python scripts/3d/generate_synthetic_slices.py \\
    --num-slices 10 --height 256 --width 256 \\
    --output-dir /tmp/test_volume_slices
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate synthetic slice images for volume inpainting testing."
    )
    ap.add_argument("--num-slices", type=int, default=10, help="Number of slices.")
    ap.add_argument("--height", type=int, default=256, help="Image height.")
    ap.add_argument("--width", type=int, default=256, help="Image width.")
    ap.add_argument("--output-dir", required=True, help="Output directory.")
    ap.add_argument("--pattern", choices=["gradient", "noise", "checkerboard"],
                    default="gradient", help="Image pattern type.")
    ap.add_argument("--stats-json", default=None, help="Optional stats JSON path.")
    args = ap.parse_args()

    if args.num_slices <= 0:
        ap.error("--num-slices must be > 0")
    if args.height <= 0:
        ap.error("--height must be > 0")
    if args.width <= 0:
        ap.error("--width must be > 0")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Y, X = np.ogrid[:args.height, :args.width]
    cy, cx = (args.height - 1) / 2.0, (args.width - 1) / 2.0

    for i in range(args.num_slices):
        if args.pattern == "gradient":
            # Radial gradient that shifts center slightly per slice
            shift = i * 3  # pixels per slice
            dist = np.sqrt((X - cx - shift) ** 2 + (Y - cy) ** 2)
            arr = (dist / dist.max() * 255).astype(np.uint8)
            # Add a bright spot that moves across slices
            spot = np.exp(-((X - cx) ** 2 + (Y - cy - i * 5) ** 2) / 500)
            arr = np.clip(arr + (spot * 80).astype(np.uint8), 0, 255).astype(np.uint8)
        elif args.pattern == "noise":
            # Correlated noise (each slice similar to previous)
            if i == 0:
                base = np.random.RandomState(42).randint(0, 256, (args.height, args.width), dtype=np.uint8)
            else:
                noise = np.random.RandomState(42 + i).randint(0, 30, (args.height, args.width), dtype=np.uint8)
                base = np.clip(base.astype(np.int16) + (noise.astype(np.int16) - 15), 0, 255).astype(np.uint8)
            arr = base
        elif args.pattern == "checkerboard":
            size = 32
            arr = (((X // size) + (Y // size) + i) % 2 * 255).astype(np.uint8)

        out_path = out_dir / f"slice_{i:04d}.png"
        Image.fromarray(arr, mode="L").save(out_path)

    print(f"Generated {args.num_slices} synthetic slice images in {out_dir}")
    print(f"  Pattern:  {args.pattern}")
    print(f"  Shape:    {args.height} x {args.width}")

    if args.stats_json:
        stats = {
            "num_slices": args.num_slices,
            "height": args.height,
            "width": args.width,
            "pattern": args.pattern,
            "output_dir": str(out_dir.resolve()),
        }
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
