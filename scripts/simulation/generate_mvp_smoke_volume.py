#!/usr/bin/env python
"""
Generate a tiny label volume for MVP multi-physics smoke testing.

Creates a small 3D label volume (default 8x8x8) with:
  - label 0 (background/air) — most voxels
  - label 1 (generic tissue) — central block
  - label 2 (lesion)         — single voxel at center

Output is a .npy file suitable for ``run_mvp_multiphysics_pipeline.py``
and ``run_mvp_multiphysics_from_config.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def build_smoke_volume(shape: tuple[int, int, int]) -> np.ndarray:
    """Create a simple 3-label volume for smoke testing.

    Args:
        shape: (Z, Y, X) dimensions — all three must be >= 2.

    Returns:
        int32 array of shape ``shape`` with labels 0, 1, and optionally 2.

    Raises:
        ValueError if any dimension < 2.
    """
    if any(s < 2 for s in shape):
        raise ValueError(
            f"All dimensions must be >= 2, got {shape}"
        )

    vol = np.zeros(shape, dtype=np.int32)

    # Central block as label 1 (generic tissue)
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    z_start, z_end = max(0, cz - 1), min(shape[0], cz + 2)
    y_start, y_end = max(0, cy - 1), min(shape[1], cy + 2)
    x_start, x_end = max(0, cx - 1), min(shape[2], cx + 2)
    vol[z_start:z_end, y_start:y_end, x_start:x_end] = 1

    # Single voxel as label 2 (lesion), if the central block is large enough
    if shape[0] >= 3 and shape[1] >= 3 and shape[2] >= 3:
        if (cz < shape[0] and cy < shape[1] and cx < shape[2]):
            vol[cz, cy, cx] = 2

    return vol


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a tiny label volume for MVP multi-physics smoke testing."
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Path to write the .npy label volume.",
    )
    ap.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=[8, 8, 8],
        metavar=("Z", "Y", "X"),
        help="Volume shape as three integers Z Y X (default: 8 8 8).",
    )
    args = ap.parse_args()

    shape = tuple(args.shape)

    if any(s < 2 for s in shape):
        print(
            f"ERROR: All dimensions must be >= 2, got {shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    vol = build_smoke_volume(shape)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), vol)

    unique_labels = sorted(int(v) for v in np.unique(vol))
    print(f"Wrote label volume: {out}")
    print(f"  Shape : {vol.shape}")
    print(f"  Labels: {unique_labels}")
    print(f"  Counts: {[int((vol == l).sum()) for l in unique_labels]}")


if __name__ == "__main__":
    main()
