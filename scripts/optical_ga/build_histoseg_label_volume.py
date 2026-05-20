#!/usr/bin/env python
"""
Build a class-ID label volume from a Histo-Seg RGB mask image.

Reads a Histo-Seg RGB segmentation mask (PNG or JPG) and converts each pixel
to a class-ID (0..11) using the canonical 12-color palette defined in this repo
(see ``scripts/build_histoseg_pairs_csv.py``).  The resulting 2D label map can
optionally be extruded to a 3D volume via ``--depth``.

Outputs:
  - required: class-ID array as .npy  (always 3D DxHxW; depth=1 yields D=1)
  - optional: metadata JSON with class histogram and unknown-color report

Usage:
  python scripts/optical_ga/build_histoseg_label_volume.py \
      --mask data/raw/histo_seg_v2/masks/some_mask.png \
      --output-label-npy /tmp/labels.npy \
      --output-meta-json /tmp/label_meta.json \
      --depth 5
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Canonical Histo-Seg color → class-ID mapping (from build_histoseg_pairs_csv.py)
# ---------------------------------------------------------------------------

HISTOSEG_COLOR_TO_CLASS_ID: Dict[Tuple[int, int, int], int] = {
    (0, 0, 0): 0,           # background
    (224, 224, 224): 1,     # epidermis
    (96, 96, 96): 2,        # reticular dermis
    (150, 150, 0): 3,       # papillary dermis
    (127, 255, 255): 4,     # dermis
    (255, 156, 0): 5,       # keratin
    (255, 0, 255): 6,       # inflammation
    (0, 255, 0): 7,         # hair follicles
    (0, 156, 255): 8,       # glands
    (127, 96, 255): 9,      # basal cell carcinoma
    (112, 48, 160): 10,     # squamous cell carcinoma
    (0, 0, 128): 11,        # intraepidermal carcinoma
}

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

# Build reverse lookup: we'll use direct pixel matching instead
_COLOR_ARRAY = np.array(list(HISTOSEG_COLOR_TO_CLASS_ID.keys()), dtype=np.uint8)
_CLASS_IDS = np.array(list(HISTOSEG_COLOR_TO_CLASS_ID.values()), dtype=np.uint8)


def _rgb_to_class_map(mask_rgb: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Convert an RGB mask array to a class-ID label map.

    Returns:
        class_map: 2D array of class IDs (uint8), HxW.
        unknown_colors: list of (R, G, B) tuples encountered but not in the palette.
            Unknown pixels are mapped to class ID 0 (background).
    """
    h, w = mask_rgb.shape[:2]
    class_map = np.zeros((h, w), dtype=np.uint8)
    unknown_colors: List[Tuple[int, int, int]] = []

    for i, rgb in enumerate(_COLOR_ARRAY):
        match = np.all(mask_rgb == rgb, axis=2)
        class_map[match] = _CLASS_IDS[i]

    # Detect unknown colors by checking which pixels are still 0 but not black
    # (black is explicitly mapped to class 0, so we need to differentiate).
    # Strategy: any pixel that didn't match any of the 12 known colors → unknown.
    # We check by re-mapping and flagging non-matching non-zero-originals.
    known_mask = np.zeros((h, w), dtype=bool)
    for rgb in _COLOR_ARRAY:
        known_mask |= np.all(mask_rgb == rgb, axis=2)

    unknown_mask = ~known_mask
    if np.any(unknown_mask):
        unique_unknown: List[Tuple[int, int, int]] = []
        seen: set = set()
        # sample unknown pixels (limit to avoid huge lists)
        uy, ux = np.where(unknown_mask)
        step = max(1, len(uy) // 1000)  # sample at most 1000 pixels
        for idx in range(0, len(uy), step):
            y, x = int(uy[idx]), int(ux[idx])
            rgb_tup = (int(mask_rgb[y, x, 0]),
                       int(mask_rgb[y, x, 1]),
                       int(mask_rgb[y, x, 2]))
            if rgb_tup not in seen:
                seen.add(rgb_tup)
                unique_unknown.append(rgb_tup)

        unknown_colors = unique_unknown
        # Map all unknown pixels to class 0 (already done by np.zeros init)

    return class_map, unknown_colors


def _compute_histogram(class_map: np.ndarray) -> Dict[str, int]:
    """Count pixel occurrences per class ID."""
    counts: Dict[str, int] = {}
    for cid in range(12):
        count = int(np.sum(class_map == cid))
        if count > 0:
            name = CLASS_NAME_BY_ID.get(cid, f"class_{cid}")
            counts[name] = count
    return counts


def _extrude_to_3d(class_map: np.ndarray, depth: int) -> np.ndarray:
    """Repeat the 2D label map along a new leading axis to produce a 3D volume.

    Result shape: (depth, H, W) as uint8.
    """
    if depth < 1:
        raise ValueError(f"Depth must be >= 1, got {depth}")
    if depth == 1:
        return class_map[np.newaxis, ...]  # (1, H, W)
    return np.repeat(class_map[np.newaxis, ...], depth, axis=0)


def _load_mask(path: str) -> np.ndarray:
    """Load an RGB mask image from path.  Handles PNG and JPG.

    Returns:
        uint8 array of shape (H, W, 3).
    """
    p = Path(path)
    if not p.exists():
        print(f"Error: mask file not found: {path}", file=sys.stderr)
        sys.exit(1)

    im = Image.open(str(p))
    if im.mode == "P":
        im = im.convert("RGBA")
    if im.mode == "RGBA":
        # Drop alpha channel
        im = im.convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")

    arr = np.asarray(im, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        print(f"Error: expected RGB image, got shape {arr.shape}", file=sys.stderr)
        sys.exit(1)

    return arr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a class-ID label volume from a Histo-Seg RGB mask image.",
    )
    ap.add_argument(
        "--mask",
        required=True,
        help="Path to input RGB mask image (PNG or JPG).",
    )
    ap.add_argument(
        "--output-label-npy",
        required=True,
        help="Path for output class-ID .npy array (always 3D DxHxW).",
    )
    ap.add_argument(
        "--output-meta-json",
        default=None,
        help="Optional path for metadata JSON (class histogram, unknown colors).",
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Number of Z-slices to extrude to (default: 1 -> shape (1,H,W)).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, exit with non-zero status when unknown colors are encountered.",
    )
    return ap.parse_args(argv[1:])


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    # Load mask
    mask_rgb = _load_mask(args.mask)
    h, w = mask_rgb.shape[:2]
    print(f"Loaded mask: {args.mask}  shape={mask_rgb.shape}")

    # Convert to label map
    class_map, unknown_colors = _rgb_to_class_map(mask_rgb)

    # Report unknowns
    if unknown_colors:
        print(
            f"Warning: {len(unknown_colors)} unknown color(s) found. "
            f"Mapped to class 0 (background).  Unknowns: {unknown_colors}",
            file=sys.stderr,
        )
        if args.strict:
            print(
                "Error: --strict mode, exiting due to unknown colors.",
                file=sys.stderr,
            )
            return 1
    else:
        print("All colors matched the canonical palette.")

    # Histogram
    hist = _compute_histogram(class_map)
    print(f"Class histogram (pixels): {hist}")

    # Extrude to 3D if requested
    vol = _extrude_to_3d(class_map, args.depth)
    print(f"Output label volume shape: {vol.shape}  dtype={vol.dtype}")

    # Save label array
    out_path = Path(args.output_label_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), vol)
    print(f"Saved label volume: {out_path}")

    # Optional metadata
    if args.output_meta_json:
        meta: dict = {
            "source_mask": str(Path(args.mask).resolve()),
            "depth": args.depth,
            "shape_2d": [h, w],
            "shape_3d": list(vol.shape),
            "dtype": str(vol.dtype),
            "class_histogram_pixels": hist,
            "unknown_colors_found": len(unknown_colors),
            "unknown_colors_sampled": unknown_colors,
            "strict_mode": args.strict,
        }
        meta_path = Path(args.output_meta_json)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(meta_path), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata: {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
