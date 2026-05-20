#!/usr/bin/env python
"""
Select a rotated tile from a high-res histology image, aligned so that the
epidermis-air interface normal points toward the top edge of the output tile.

Pipeline
--------
1. Load high-res RGB image and segmentation mask.
2. (Optional) Run ``estimate_epidermis_normal`` if no pre-computed normal
   is supplied; otherwise accept normal angle / confidence from caller or
   a previous JSON result.
3. Resolve tile centre (manual center args, or centroid from normal JSON,
   or mask-derived centroid depending on ``--center-mode``).
4. Compute a rotation that aligns the outward normal (air → epidermis)
   to the upward direction (image −y, angle −90°) in the output tile.
5. Extract a 512×512 rotated tile from the *original* image coordinates
   via affine warp — no pre-padded canvas is used.
6. Apply the same affine transform to the mask to produce a rotated
   mask tile.
7. Save the tile image, rotated mask tile, and a metadata JSON.

Convention
----------
- The outward normal points from air (class 0) toward epidermis
  (class 1).
- Rotation is applied so the normal points "up" (top edge of the output
  tile, i.e. image −y direction).
- The warp samples directly from the source image; no virtual/padded
  canvas introduces synthetic pixels.

Usage
-----
::

    python scripts/optical_ga/select_orient_tile_for_incidence.py \\
        --image /path/to/slice.png \\
        --mask /path/to/mask.png \\
        --output-dir /tmp/tiles

    # provide pre-computed normal JSON from estimate_epidermis_normal.py
    python scripts/optical_ga/select_orient_tile_for_incidence.py \\
        --image /path/to/slice.png \\
        --mask /path/to/mask.png \\
        --normal-json /tmp/normal.json \\
        --output-dir /tmp/tiles
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.optical_ga.estimate_epidermis_normal import (
    estimate_epidermis_normal,
)

# ---------------------------------------------------------------------------
# Affine warp — try cv2 first, fall back to scipy.ndimage
# ---------------------------------------------------------------------------

_HAS_CV2 = False
try:
    import cv2  # noqa: F401

    _HAS_CV2 = True
except Exception:
    pass

_HAS_SCIPY = False
try:
    from scipy.ndimage import affine_transform as _scipy_affine  # noqa: F811

    _HAS_SCIPY = True
except Exception:
    pass


def _warp_affine(
    src: np.ndarray,
    rot_deg: float,
    center_xy: Tuple[float, float],
    dsize: Tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    """Rotate *src* by ``rot_deg`` CCW around ``center_xy`` and crop.

    The source pixel at ``center_xy`` maps to the centre of the output
    tile.  Sampling is from the original image coordinates — no
    pre-padded canvas is introduced.

    Uses ``scipy.ndimage.affine_transform`` as the primary backend.
    Falls back to ``cv2.warpAffine`` only if scipy is unavailable
    (scipy's (row, col) convention is the canonical choice for NumPy
    arrays).

    Parameters
    ----------
    src : np.ndarray
        Source image ``(H, W [, C])``.
    rot_deg : float
        CCW rotation angle in degrees.
    center_xy : tuple of float
        ``(cx, cy)`` rotation centre in pixel coordinates (x = col,
        y = row).
    dsize : tuple of int
        ``(width, height)`` of the output tile.
    order : int
        Interpolation order (1 = bilinear for images, 0 = nearest for
        masks).

    Returns
    -------
    dst : np.ndarray
        Warped tile of shape ``(dsize[1], dsize[0], C)`` or
        ``(dsize[1], dsize[0])`` for 2-D input.
    """
    w, h = dsize
    theta = np.radians(rot_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = center_xy
    # Tile centre in output (row, col) coordinates — pixel-centre convention
    ct_r = (h - 1) / 2.0  # row centre in output
    ct_c = (w - 1) / 2.0  # col centre in output

    # ---- scipy backend ----
    # scipy.ndimage.affine_transform maps output (row, col) → input (row, col).
    #
    # Forward rotation by θ CCW around (cx, cy) in image (col, row) space:
    #   col_out = cx + (col_src - cx)*cos θ - (row_src - cy)*sin θ
    #   row_out = cy + (col_src - cx)*sin θ + (row_src - cy)*cos θ
    #
    # Inverse (warp) — output → source:
    #   col_src = cx + (col_out - ct_c)*cos θ + (row_out - ct_r)*sin θ
    #   row_src = cy - (col_out - ct_c)*sin θ + (row_out - ct_r)*cos θ
    #
    # Rearranged as output (row_out, col_out) → input (row_src, col_src):
    #   row_src =  sin θ · col_out  +  cos θ · row_out  +  off_r
    #   col_src =  cos θ · col_out  + -sin θ · row_out  +  off_c
    #
    #   off_r = cy - sin_t * ct_c - cos_t * ct_r
    #   off_c = cx - cos_t * ct_c + sin_t * ct_r

    off_r = cy - sin_t * ct_c - cos_t * ct_r
    off_c = cx - cos_t * ct_c + sin_t * ct_r

    matrix = np.array([[cos_t, sin_t],
                       [-sin_t, cos_t]], dtype=np.float64)
    offset = np.array([off_r, off_c], dtype=np.float64)

    output_shape = (h, w)

    if _HAS_SCIPY:
        if src.ndim == 2:
            dst = _scipy_affine(
                src, matrix, offset=offset, output_shape=output_shape,
                order=order, mode="constant", cval=0.0, prefilter=False,
            )
        else:
            channels = []
            for c in range(src.shape[2]):
                ch = _scipy_affine(
                    src[..., c], matrix, offset=offset,
                    output_shape=output_shape, order=order,
                    mode="constant", cval=0.0, prefilter=False,
                )
                channels.append(ch)
            dst = np.stack(channels, axis=-1)
        return dst

    # ---- cv2 fallback ----
    if not _HAS_CV2:
        raise RuntimeError(
            "Neither scipy.ndimage nor cv2 is available.  "
            "Install one of: scipy, opencv-python."
        )

    # cv2 uses (x, y) = (col, row) convention.  Build matrix that maps
    # output (col, row) → source (col, row) with centre mapped to tile centre.
    tx = cx - cos_t * ct_c - sin_t * ct_r
    ty = cy + sin_t * ct_c - cos_t * ct_r
    M = np.array([[cos_t, sin_t, tx],
                  [-sin_t, cos_t, ty]], dtype=np.float64)
    interp = cv2.INTER_NEAREST if order == 0 else cv2.INTER_LINEAR
    dst = cv2.warpAffine(
        src, M, (w, h), flags=interp,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return dst


# ---------------------------------------------------------------------------
# Core tile extraction
# ---------------------------------------------------------------------------


def select_orient_tile(
    image: np.ndarray,
    mask: np.ndarray,
    normal_deg: float,
    confidence: float,
    *,
    center_xy: Optional[Tuple[float, float]] = None,
    tile_size: int = 512,
    seed: int = 42,
) -> Dict[str, Any]:
    """Extract a rotated tile aligned to the epidermis-air normal.

    Parameters
    ----------
    image : np.ndarray
        High-resolution RGB image ``(H, W, 3)`` uint8.
    mask : np.ndarray
        Segmentation mask ``(H, W)`` integer.
    normal_deg : float
        Outward normal angle in degrees (image coords).
    confidence : float
        Interface confidence in [0, 1] (passed through to metadata).
    center_xy : tuple of float or None
        ``(x, y)`` = ``(col, row)`` centre for the tile.  If ``None``,
        defaults to the image centre.
    tile_size : int
        Width/height of the square output tile (default 512).
    seed : int
        Deterministic seed (default 42).

    Returns
    -------
    result : dict
        Keys:
        - ``tile``: ``np.ndarray (tile_size, tile_size, 3)`` uint8
        - ``tile_mask``: ``np.ndarray (tile_size, tile_size)`` uint8
        - ``metadata``: dict with rotation info, source coords, etc.
    """
    # Default centre
    h, w = image.shape[:2]
    if center_xy is None:
        center_xy = (w / 2.0, h / 2.0)

    cx, cy = center_xy

    # Compute rotation angle so that the normal points "up" (image −y,
    # angle −90°).
    #
    # With cv2 convention: rotation by θ CCW moves a source feature at
    # angle α to output angle α − θ.
    # We want output_angle = −90°, source_angle = normal_deg.
    #   −90° = normal_deg − θ  →  θ = normal_deg + 90°
    rotation_deg = normal_deg + 90.0

    np.random.seed(seed)

    # Warp image
    tile = _warp_affine(
        image, rotation_deg, center_xy=(cx, cy),
        dsize=(tile_size, tile_size), order=1,
    )

    # Warp mask (nearest-neighbour)
    tile_mask = _warp_affine(
        mask.astype(np.float64), rotation_deg, center_xy=(cx, cy),
        dsize=(tile_size, tile_size), order=0,
    )
    tile_mask = np.round(tile_mask).astype(np.uint8)

    # Build metadata
    metadata: Dict[str, Any] = {
        "tile_size": tile_size,
        "rotation_deg": float(rotation_deg),
        "normal_deg_input": float(normal_deg),
        "confidence": float(confidence),
        "source_center_x": float(cx),
        "source_center_y": float(cy),
        "source_image_shape": list(image.shape[:2]),
        "normal_convention": (
            "outward normal points from air (0) toward epidermis (1); "
            "image coords: x right, y down; angle CCW from +x"
        ),
        "rotation_convention": (
            "rotation_deg applied as CCW rotation so that the outward "
            "normal aligns with the top edge of the output tile "
            "(image −y direction, angle −90°)"
        ),
        "backend": "cv2" if _HAS_CV2 else "scipy",
        "seed_used": seed,
    }

    return {
        "tile": tile,
        "tile_mask": tile_mask,
        "metadata": metadata,
    }


def select_orient_tile_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    center_xy: Optional[Tuple[float, float]] = None,
    tile_size: int = 512,
    air_value: int = 0,
    epidermis_value: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Convenience: estimate normal from mask, then extract oriented tile.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W, 3)`` uint8.
    mask : np.ndarray
        ``(H, W)`` integer.
    center_xy : tuple or None
        Tile centre.  If None, uses the boundary centroid.
    tile_size : int
        Output tile size.
    air_value, epidermis_value : int
        Class labels in *mask*.
    seed : int
        RNG seed.

    Returns
    -------
    result : dict
        Same layout as :func:`select_orient_tile`.
    """
    normal_deg, normal_vec, confidence, boundary_pts, _ = estimate_epidermis_normal(
        mask,
        air_value=air_value,
        epidermis_value=epidermis_value,
    )

    if center_xy is None and len(boundary_pts) > 0:
        # Use boundary centroid as tile centre
        centroid_col = float(boundary_pts[:, 1].mean())
        centroid_row = float(boundary_pts[:, 0].mean())
        center_xy = (centroid_col, centroid_row)

    return select_orient_tile(
        image, mask,
        normal_deg=normal_deg,
        confidence=confidence,
        center_xy=center_xy,
        tile_size=tile_size,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a rotated 512×512 tile from a high-res histology "
            "image with the epidermis-air normal aligned to the top "
            "edge.  The tile is sampled directly from the original image "
            "via affine warp (no padded canvas)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Convention: outward normal points from air (0) toward "
            "epidermis (1).  The rotation aligns this normal to the "
            "top edge of the output tile.\n\n"
            "Examples:\n"
            "  # full pipeline (estimate normal + extract tile)\n"
            "  python scripts/optical_ga/select_orient_tile_for_incidence.py \\\n"
            "      --image slice.png --mask mask.png --output-dir /tmp/tiles\n\n"
            "  # use pre-computed normal\n"
            "  python scripts/optical_ga/select_orient_tile_for_incidence.py \\\n"
            "      --image slice.png --mask mask.png \\\n"
            "      --normal-json /tmp/normal.json --output-dir /tmp/tiles"
        ),
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to high-resolution RGB histology slice.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help=(
            "Path to segmentation mask (grayscale, 0=air/background, "
            "1=epidermis)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for output tile PNG + rotated mask + metadata JSON.",
    )
    parser.add_argument(
        "--normal-json",
        type=str,
        default=None,
        help=(
            "Optional pre-computed normal JSON (from "
            "estimate_epidermis_normal.py).  If omitted, the normal "
            "is estimated from --mask."
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Width/height of the square output tile (default 512).",
    )
    parser.add_argument(
        "--center-x",
        type=float,
        default=None,
        help="Optional tile centre x (column) in source-image coordinates.",
    )
    parser.add_argument(
        "--center-y",
        type=float,
        default=None,
        help="Optional tile centre y (row) in source-image coordinates.",
    )
    parser.add_argument(
        "--center-mode",
        type=str,
        choices=["auto", "centroid", "image"],
        default="auto",
        help=(
            "How to choose tile center when --center-x/--center-y are not set: "
            "auto=use centroid if available else image center; "
            "centroid=require centroid; image=force image center."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed (default 42).",
    )
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    from PIL import Image as PILImage

    # Load image
    img_pil = PILImage.open(args.image).convert("RGB")
    image_np = np.asarray(img_pil, dtype=np.uint8)

    # Load mask
    mask_pil = PILImage.open(args.mask).convert("L")
    mask_np = np.asarray(mask_pil, dtype=np.int32)
    # Normalise uint8 masks (values 0-255) to 0-1
    if mask_np.max() > 1:
        mask_np = (mask_np > 127).astype(np.int32)

    # Load or compute normal
    boundary_pts = None
    centroid_from_json = None
    if args.normal_json:
        with open(args.normal_json) as f:
            norm_data = json.load(f)
        normal_deg = float(norm_data["normal_deg"])
        confidence = float(norm_data["confidence"])
        if "centroid_col" in norm_data and "centroid_row" in norm_data:
            centroid_from_json = (
                float(norm_data["centroid_col"]),
                float(norm_data["centroid_row"]),
            )
    else:
        normal_deg, _, confidence, boundary_pts, _ = estimate_epidermis_normal(mask_np)

    # Resolve center
    has_manual_x = args.center_x is not None
    has_manual_y = args.center_y is not None
    if has_manual_x != has_manual_y:
        raise ValueError("Both --center-x and --center-y must be provided together.")

    center_source = "image_center"
    center_xy = None
    if has_manual_x and has_manual_y:
        center_xy = (float(args.center_x), float(args.center_y))
        center_source = "manual"
    elif args.center_mode == "image":
        center_xy = (image_np.shape[1] / 2.0, image_np.shape[0] / 2.0)
        center_source = "image_center"
    elif centroid_from_json is not None:
        center_xy = centroid_from_json
        center_source = "normal_json_centroid"
    elif boundary_pts is not None and len(boundary_pts) > 0:
        center_xy = (
            float(boundary_pts[:, 1].mean()),
            float(boundary_pts[:, 0].mean()),
        )
        center_source = "mask_boundary_centroid"
    elif args.center_mode == "centroid":
        raise ValueError(
            "--center-mode centroid requested, but no centroid is available "
            "from --normal-json or mask boundary estimation."
        )
    else:
        center_xy = (image_np.shape[1] / 2.0, image_np.shape[0] / 2.0)
        center_source = "image_center_fallback"

    # Extract tile
    result = select_orient_tile(
        image_np,
        mask_np,
        normal_deg=normal_deg,
        confidence=confidence,
        center_xy=center_xy,
        tile_size=args.tile_size,
        seed=args.seed,
    )

    tile = result["tile"]
    tile_mask = result["tile_mask"]
    metadata = result["metadata"]
    metadata["center_source"] = center_source

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tile_path = out_dir / "orient_tile.png"
        PILImage.fromarray(tile).save(tile_path)

        mask_path = out_dir / "orient_tile_mask.png"
        PILImage.fromarray(tile_mask).save(mask_path)

        json_path = out_dir / "orient_tile_metadata.json"
        json_path.write_text(json.dumps(metadata, indent=2))

        print(f"Tile saved:        {tile_path}")
        print(f"Rotated mask:      {mask_path}")
        print(f"Metadata:          {json_path}")
    else:
        # Print metadata to stdout
        print(json.dumps(metadata, indent=2))
        print(f"Tile shape: {tile.shape}, dtype={tile.dtype}")
        print(f"Mask shape: {tile_mask.shape}, dtype={tile_mask.dtype}")


if __name__ == "__main__":
    _main()
