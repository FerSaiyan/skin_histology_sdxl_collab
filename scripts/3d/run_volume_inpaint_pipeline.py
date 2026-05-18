#!/usr/bin/env python
"""
Tile-first volume inpainting pipeline for high-res histology slice stacks.

THE TILE-FIRST FLOW (no full-slice resizing to model resolution):
  Step 1: Generate/validate full-slice cylindrical masks
  Step 2: Build pairs CSV for patch extraction
  Step 3: Extract ROI patches from full slices + masks (reuse extract_roi_patches.py)
  Step 4: Inpaint ROI patches at model resolution (reuse inpaint_roi_patches.py)
  Step 5: Merge inpainted patches back into full-res slices (reuse merge_inpainted_patches.py)
  Step 6: Stack merged full-res slices to NIfTI
   Step 7: Compute Z-coherence metrics and evaluate optional thresholds
   Step 8: Save manifest JSON

WHY TILE-FIRST?
  Full high-res slices (e.g., 5000x5000) cannot be directly resized to 512x512
  for inpainting — the tissue architecture is lost.  Instead, we extract a
  local patch around the ROI/mask, inpaint at model resolution, and merge
  the edited region back into the original full-res slice.  This preserves
  all tissue context outside the mask.

Supports --dry-run to exercise the metadata + extraction metadata + stacking +
metrics pipeline without requiring a real SDXL model or LoRA weights.
"""

from __future__ import annotations

import argparse
import csv
import glob as _glob
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import re

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _check_python() -> str:
    """Return the python executable path."""
    return sys.executable


def _resolve_common_parent(glob_pattern: str) -> Path:
    """Get the common parent directory from the first glob match."""
    files = sorted(_glob.glob(glob_pattern))
    if not files:
        raise SystemExit(f"No files match glob: {glob_pattern}")
    return Path(files[0]).resolve().parent


def _infer_slice_shape(glob_pattern: str) -> tuple[int, int]:
    """Infer (height, width) from first slice image matched by glob."""
    files = sorted(_glob.glob(glob_pattern))
    if not files:
        raise SystemExit(f"No files match glob: {glob_pattern}")
    img = Image.open(files[0])
    w, h = img.size
    return h, w


def _collect_slice_files(glob_pattern: str) -> list[str]:
    """Return sorted list of slice file paths matching glob."""
    files = sorted(_glob.glob(glob_pattern))
    if not files:
        raise SystemExit(f"No files match glob pattern: {glob_pattern}")
    return files


def _make_circular_mask(
    height: int, width: int, radius_frac: float, shape_mode: str,
    center_x: int | None = None, center_y: int | None = None,
) -> np.ndarray:
    """Generate a centered (or custom-center) binary mask (uint8 0/255).

    Mirrors propagate_mask_across_slices._make_circular_mask.
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


def _compute_mask_bbox(
    mask: np.ndarray,
) -> tuple[int, int, int, int]:
    """Compute bounding box of non-zero mask region.

    Returns (y_min, x_min, y_max, x_max).
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return 0, 0, mask.shape[0], mask.shape[1]
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y_min, y_max = int(y_indices[0]), int(y_indices[-1]) + 1
    x_min, x_max = int(x_indices[0]), int(x_indices[-1]) + 1
    return y_min, x_min, y_max, x_max


def _extract_numeric_index(filename: str) -> int | None:
    """Extract the trailing numeric index from a filename stem.

    E.g. 'slice_0003.png' -> 3, 'mask_slice_0042.png' -> 42,
    'volume001_slice09.png' -> 9.  Returns None if no digits found.
    """
    stem = Path(filename).stem
    # Prefer _<digits> at end (slice_0003, mask_slice_0012)
    m = re.search(r'_(\d+)$', stem)
    if m:
        return int(m.group(1))
    # Fall back to any trailing digits (slice09, vol003)
    m = re.search(r'(\d+)$', stem)
    if m:
        return int(m.group(1))
    return None


def _match_slices_to_masks(
    slice_files: list[str],
    mask_files: list[Path],
) -> list[tuple[str, Path]]:
    """Match slice files to masks by numeric index token.

    Parses trailing numeric indices from both lists (e.g. slice_0003 <-> mask_slice_0003)
    and creates pairs by matching indices.  Falls back to lexicographic zip if
    indices cannot be parsed, with a clear warning.
    """
    # Extract numeric indices
    sidx_list = [_extract_numeric_index(sf) for sf in slice_files]
    midx_list = [_extract_numeric_index(mf.name) for mf in mask_files]

    sidx_valid = all(i is not None for i in sidx_list)
    midx_valid = all(i is not None for i in midx_list)

    if sidx_valid and midx_valid:
        # Index-token matching
        mask_by_idx: dict[int, Path] = dict(zip(midx_list, mask_files))  # type: ignore[arg-type]
        matched: list[tuple[str, Path]] = []
        unmatched_slices = 0
        for sf, sidx in zip(slice_files, sidx_list):
            if sidx in mask_by_idx:  # type: ignore[operator]
                matched.append((sf, mask_by_idx[sidx]))  # type: ignore[index]
            else:
                print(f"  [WARN] No mask found for slice index {sidx}: {Path(sf).name}")
                unmatched_slices += 1

        # Warn about unmatched masks
        matched_slice_indices = {i for i in sidx_list if i is not None}
        for midx, mf in zip(midx_list, mask_files):
            if midx not in matched_slice_indices:
                print(f"  [WARN] No slice found for mask index {midx}: {mf.name}")

        if not matched:
            raise SystemExit("No slice/mask pairs could be matched by index token.")
        if unmatched_slices > 0:
            print(f"  [WARN] {unmatched_slices} slice(s) have no matching mask (skipped).")
        return matched

    # Fallback: lexicographic zip
    print(
        "  [WARN] Could not parse numeric indices from filenames. "
        "Falling back to lexicographic zip for slice/mask pairing."
    )
    num_pairs = min(len(slice_files), len(mask_files))
    return [(slice_files[i], mask_files[i]) for i in range(num_pairs)]


def _make_square_bbox(
    y_min: int, x_min: int, y_max: int, x_max: int,
    max_h: int, max_w: int,
) -> tuple[int, int, int, int]:
    """Make bbox square by extending to larger dimension.

    Mirrors extract_roi_patches._make_square for validation consistency.
    """
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


def _compute_tissue_validity(
    arr: np.ndarray, white_threshold: int = 235, black_threshold: int = 10,
) -> np.ndarray:
    """Compute tissue-validity boolean map for an image array.

    Handles RGB (H,W,3) and grayscale (H,W) modes:
      - white-void:   all channels >= white_threshold (RGB) or pixel >= white_threshold (gray)
      - black-void:   all channels <= black_threshold (RGB) or pixel <= black_threshold (gray)
      - tissue-valid: not white and not black

    Returns (H, W) bool array where True = tissue-valid.
    """
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # RGB or RGBA
        white_void = np.logical_and.reduce(
            [arr[:, :, c] >= white_threshold for c in range(3)]
        )
        black_void = np.logical_and.reduce(
            [arr[:, :, c] <= black_threshold for c in range(3)]
        )
    else:
        # Grayscale
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        white_void = arr >= white_threshold
        black_void = arr <= black_threshold

    return np.logical_and(~white_void, ~black_void)


def _compute_mask_overlap_stats_for_slice(
    slice_arr: np.ndarray,
    mask: np.ndarray,
    white_threshold: int = 235,
    black_threshold: int = 10,
) -> dict[str, float]:
    """Compute tissue overlap, white void, and black void fractions inside mask.

    Args:
        slice_arr: Full-res slice array (H, W) or (H, W, 3/4).
        mask: Binary mask uint8 (H, W) with values 0 or 255.
        white_threshold: Pixels >= this are white-void.
        black_threshold: Pixels <= this are black-void.

    Returns dict with keys:
        tissue_overlap_frac, white_void_frac, black_void_frac
    """
    mask_bool = mask > 0
    n_mask = int(mask_bool.sum())
    if n_mask == 0:
        return {
            "tissue_overlap_frac": 0.0,
            "white_void_frac": 0.0,
            "black_void_frac": 0.0,
        }

    tissue = _compute_tissue_validity(slice_arr, white_threshold, black_threshold)
    masked_tissue = tissue & mask_bool

    # Compute void stats inside mask
    if slice_arr.ndim == 3 and slice_arr.shape[2] >= 3:
        white_inside = np.logical_and.reduce(
            [slice_arr[:, :, c] >= white_threshold for c in range(3)]
        ) & mask_bool
        black_inside = np.logical_and.reduce(
            [slice_arr[:, :, c] <= black_threshold for c in range(3)]
        ) & mask_bool
    else:
        arr_2d = slice_arr[:, :, 0] if slice_arr.ndim == 3 else slice_arr
        white_inside = (arr_2d >= white_threshold) & mask_bool
        black_inside = (arr_2d <= black_threshold) & mask_bool

    return {
        "tissue_overlap_frac": float(masked_tissue.sum()) / float(n_mask),
        "white_void_frac": float(white_inside.sum()) / float(n_mask),
        "black_void_frac": float(black_inside.sum()) / float(n_mask),
    }


def _find_tissue_aware_mask_center(
    slice_files: list[str],
    num_slices: int,
    slice_height: int,
    slice_width: int,
    mask_radius_frac: float,
    patch_size: int,
    *,
    white_threshold: int = 235,
    black_threshold: int = 10,
    min_tissue_overlap: float = 0.85,
    max_black_frac: float = 0.10,
    max_attempts: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[int, int, float]:
    """Find a tissue-aware center for a cylindrical mask across sequential slices.

    Steps:
      1. Load each slice, compute tissue-validity map.
      2. Intersect all validity maps → common_valid (pixels valid in ALL slices).
      3. Restrict to coordinates where a patch_size tile centered at (y,x) stays
         fully within slice bounds (no zero-padding needed).
      4. From remaining valid candidates, randomly sample a center.
      5. Generate a cylindrical mask around that center.
      6. Validate tissue overlap & black void fraction across all slices.
      7. Retry up to max_attempts if validation fails.

    Returns:
        (center_y, center_x, combined_tissue_overlap)
        where combined_tissue_overlap is the minimum overlap across slices.

    Raises:
        SystemExit if no valid center found after max_attempts.
    """
    # Load all slices and compute validity
    validity_maps: list[np.ndarray] = []
    slice_arrays: list[np.ndarray] = []
    for sf in slice_files[:num_slices]:
        img = Image.open(sf)
        arr = np.array(img)
        slice_arrays.append(arr)
        valid = _compute_tissue_validity(arr, white_threshold, black_threshold)
        validity_maps.append(valid)

    # Intersect validity across all slices
    common_valid = np.logical_and.reduce(validity_maps)
    valid_coords = np.argwhere(common_valid)  # (N, 2) array of [y, x]

    if valid_coords.shape[0] == 0:
        raise SystemExit(
            f"No tissue-valid pixel found common to all {num_slices} slices. "
            f"Thresholds: white>={white_threshold}, black<={black_threshold}. "
            f"Try relaxing thresholds."
        )

    # NOTE: No "full tile in bounds" restriction needed. The extraction step
    # (in _create_synthetic_extraction_metadata) clamps the centered tile to
    # slice bounds, so partial-overflow tiles are handled gracefully. The
    # tissue validity of the (possibly cropped) tile is validated below per
    # candidate center.

    rng_center = np.random.default_rng(seed)

    if verbose:
        print(
            f"  [TISSUE-AWARE] Common-valid pixels: {valid_coords.shape[0]} "
            f"(out of {slice_height}x{slice_width} = {slice_height*slice_width} total)"
        )

    best_center = None
    best_overlap = 0.0
    best_stats: list[dict] = []

    for attempt in range(max_attempts):
        # Pick a random center from valid coords
        idx = rng_center.integers(0, valid_coords.shape[0])
        cy, cx = int(valid_coords[idx, 0]), int(valid_coords[idx, 1])

        # Generate cylindrical mask at this center
        mask = _make_circular_mask(
            slice_height, slice_width, mask_radius_frac, "circle",
            center_x=cx, center_y=cy,
        )

        # Validate bbox fits within patch_size
        y_min, x_min, y_max, x_max = _compute_mask_bbox(mask)
        bbox_h = y_max - y_min
        bbox_w = x_max - x_min
        if bbox_h > patch_size or bbox_w > patch_size:
            if verbose:
                print(f"    Attempt {attempt+1}: bbox {bbox_h}x{bbox_w} > {patch_size}, skip")
            continue

        # Compute overlap stats for all slices
        per_slice_stats: list[dict] = []
        all_ok = True
        for sarr in slice_arrays:
            stats = _compute_mask_overlap_stats_for_slice(
                sarr, mask, white_threshold, black_threshold,
            )
            per_slice_stats.append(stats)
            if stats["tissue_overlap_frac"] < min_tissue_overlap:
                all_ok = False
            if stats["black_void_frac"] > max_black_frac:
                all_ok = False

        if not all_ok:
            continue

        # Basic sanity: check that the encompassing tile (centered and clamped
        # to bounds, matching _create_synthetic_extraction_metadata behavior)
        # is not entirely void. The mask overlap validation above already
        # ensures the cylindrical mask itself lands in tissue; the tile is the
        # extraction boundary and will include context beyond the mask.
        half = patch_size // 2
        tile_y_min = max(0, cy - half)
        tile_y_max = min(slice_height, cy + half + (patch_size % 2))
        tile_x_min = max(0, cx - half)
        tile_x_max = min(slice_width, cx + half + (patch_size % 2))
        tile_valid = True
        for sarr in slice_arrays:
            tile_arr = sarr[tile_y_min:tile_y_max, tile_x_min:tile_x_max]
            if tile_arr.size == 0:
                tile_valid = False
                break
            tile_tissue = _compute_tissue_validity(tile_arr, white_threshold, black_threshold)
            if float(tile_tissue.sum()) < 1:  # at least 1 tissue pixel in tile
                tile_valid = False
                break

        if not tile_valid:
            continue

        best_center = (cy, cx)
        best_overlap = min(s["tissue_overlap_frac"] for s in per_slice_stats)
        best_stats = per_slice_stats
        if verbose:
            print(
                f"  [TISSUE-AWARE] Found valid center ({cx},{cy}) "
                f"on attempt {attempt+1}/{max_attempts} "
                f"(min tissue overlap: {best_overlap:.4f})"
            )
        break

    if best_center is None:
        raise SystemExit(
            f"Could not find a valid tissue-aware mask center after "
            f"{max_attempts} attempts. "
            f"Thresholds: white>={white_threshold}, black<={black_threshold}, "
            f"min_tissue_overlap={min_tissue_overlap}, "
            f"max_black_frac={max_black_frac}. "
            f"Try relaxing constraints or increasing --mask-center-max-attempts."
        )

    return best_center[0], best_center[1], best_overlap


def _save_mask_diagnostics(
    slice_files: list[str],
    mask_dir: str,
    center_y: int,
    center_x: int,
    num_slices: int,
    white_threshold: int,
    black_threshold: int,
    run_dir: Path,
) -> str:
    """Compute per-slice mask overlap diagnostics and save to JSON/CSV.

    Returns path to the diagnostics JSON file.
    """
    mask_dir_p = Path(mask_dir)
    diags: list[dict] = []
    n_eval = min(num_slices, len(slice_files))
    for i in range(n_eval):
        slice_arr = np.array(Image.open(slice_files[i]))
        mask_path = mask_dir_p / f"mask_slice_{i:04d}.png"
        if not mask_path.exists():
            continue
        mask_arr = np.array(Image.open(mask_path).convert("L"))
        stats = _compute_mask_overlap_stats_for_slice(
            slice_arr, mask_arr, white_threshold, black_threshold,
        )
        stats["slice_index"] = i
        stats["filename"] = Path(slice_files[i]).name
        diags.append(stats)

    # Compute summary
    tissue_vals = [d["tissue_overlap_frac"] for d in diags]
    white_vals = [d["white_void_frac"] for d in diags]
    black_vals = [d["black_void_frac"] for d in diags]

    summary = {
        "center_y": center_y,
        "center_x": center_x,
        "white_threshold": white_threshold,
        "black_threshold": black_threshold,
        "num_slices": len(diags),
        "mask_radius_frac": None,  # filled by caller
        "min_tissue_overlap": min(tissue_vals) if tissue_vals else None,
        "mean_tissue_overlap": float(np.mean(tissue_vals)) if tissue_vals else None,
        "max_tissue_overlap": max(tissue_vals) if tissue_vals else None,
        "min_white_void": min(white_vals) if white_vals else None,
        "mean_white_void": float(np.mean(white_vals)) if white_vals else None,
        "max_white_void": max(white_vals) if white_vals else None,
        "min_black_void": min(black_vals) if black_vals else None,
        "mean_black_void": float(np.mean(black_vals)) if black_vals else None,
        "max_black_void": max(black_vals) if black_vals else None,
    }
    all_data = {"summary": summary, "per_slice": diags}

    diag_path = run_dir / "tissue_aware_mask_diagnostics.json"
    diag_path.write_text(json.dumps(all_data, indent=2, default=str), encoding="utf-8")

    # Also save CSV
    import csv as _csv
    csv_path = run_dir / "tissue_aware_mask_diagnostics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=[
            "slice_index", "filename",
            "tissue_overlap_frac", "white_void_frac", "black_void_frac",
        ])
        writer.writeheader()
        for d in diags:
            writer.writerow({
                "slice_index": d["slice_index"],
                "filename": d["filename"],
                "tissue_overlap_frac": f"{d['tissue_overlap_frac']:.6f}",
                "white_void_frac": f"{d['white_void_frac']:.6f}",
                "black_void_frac": f"{d['black_void_frac']:.6f}",
            })

    print(f"  [DIAGNOSTICS] Tissue-aware mask summary:")
    print(f"    Center: ({center_x}, {center_y})")
    if diags:
        print(f"    Tissue overlap: min={summary['min_tissue_overlap']:.4f}, "
              f"mean={summary['mean_tissue_overlap']:.4f}, "
              f"max={summary['max_tissue_overlap']:.4f}")
        print(f"    White void: min={summary['min_white_void']:.4f}, "
              f"mean={summary['mean_white_void']:.4f}, "
              f"max={summary['max_white_void']:.4f}")
        print(f"    Black void: min={summary['min_black_void']:.4f}, "
              f"mean={summary['mean_black_void']:.4f}, "
              f"max={summary['max_black_void']:.4f}")
    else:
        print("    No diagnostics rows were produced.")
    print(f"    Saved: {diag_path}")
    print(f"    Saved CSV: {csv_path}")

    return str(diag_path)


def _validate_mask_containment(
    mask_dir: str,
    patch_size: int,
    padding_ratio: float,
    strict: bool = False,
) -> None:
    """Validate that ALL mask bboxes fit within patch_size.

    In strict mode: validates raw bbox dimensions (no padding, no square) —
    both width and height must be <= patch_size.

    In non-strict mode: validates padded + squared bbox <= patch_size.
    """
    mask_dir_p = Path(mask_dir)
    mask_files = sorted(mask_dir_p.glob("mask_slice_*.png"))
    if not mask_files:
        mask_files = sorted(mask_dir_p.glob("*.png"))
    if not mask_files:
        print("  [VALIDATION] No masks found to validate containment.")
        return

    failed: list[str] = []
    for mf in mask_files:
        mask = np.array(Image.open(mf).convert("L"))
        h, w = mask.shape

        y_min, x_min, y_max, x_max = _compute_mask_bbox(mask)
        raw_h = y_max - y_min
        raw_w = x_max - x_min

        if strict:
            # Strict raw-bbox validation: no padding, no square
            if raw_h > patch_size or raw_w > patch_size:
                failed.append(
                    f"  Mask '{mf.name}': raw bbox {raw_h}x{raw_w} > configured "
                    f"--patch-size={patch_size} (strict mode: both dimensions must fit)."
                )
        else:
            # Legacy: apply padding + square
            pad_y = int(raw_h * padding_ratio)
            pad_x = int(raw_w * padding_ratio)
            y_min_p = max(0, y_min - pad_y)
            y_max_p = min(h, y_max + pad_y)
            x_min_p = max(0, x_min - pad_x)
            x_max_p = min(w, x_max + pad_x)
            y_min_sq, x_min_sq, y_max_sq, x_max_sq = _make_square_bbox(
                y_min_p, x_min_p, y_max_p, x_max_p, h, w,
            )
            bbox_dim = max(y_max_sq - y_min_sq, x_max_sq - x_min_sq)
            if bbox_dim > patch_size:
                failed.append(
                    f"  Mask '{mf.name}': required {bbox_dim}px > configured {patch_size}px "
                    f"(raw bbox {raw_h}x{raw_w}, padded + squared)"
                )

    if failed:
        rule = "strict raw-bbox" if strict else "padded + squared"
        error_msg = (
            f"Mask containment validation FAILED for {len(failed)} of {len(mask_files)} masks.\n"
            f"  Rule: {rule}\n"
            f"  Configured --patch-size={patch_size}"
        )
        if not strict:
            error_msg += f" (with --padding-ratio={padding_ratio})"
        error_msg += "\n  Failures:\n"
        error_msg += "\n".join(failed)

        if strict:
            error_msg += (
                f"\n  Suggestions:\n"
                f"    - Increase --patch-size (at least one raw bbox needs >={patch_size+1} in a dimension)\n"
                f"    - Decrease --mask-radius so the cylindrical mask is smaller\n"
                f"    - Disable --strict-bbox to allow padding + square expansion"
            )
        else:
            error_msg += (
                f"\n  Suggestions:\n"
                f"    - Increase --patch-size (at least one mask needs >={patch_size+1})\n"
                f"    - Decrease --mask-radius so the cylindrical mask is smaller\n"
                f"    - Decrease --padding-ratio (currently {padding_ratio})"
            )
        raise SystemExit(error_msg)

    mode_note = " (strict raw-bbox)" if strict else ""
    print(f"  [VALIDATION] Mask containment OK{mode_note}: all {len(mask_files)} mask bboxes fit within --patch-size={patch_size}")


def _step_generate_masks(
    args: argparse.Namespace, run_dir: Path, step_log: dict,
) -> None:
    """Step 1: Generate propagated masks at full-slice resolution.

    Mask generation is fast (writes small PNGs), so it always runs even in
    dry-run mode.  Masks are generated at the real slice resolution (inferred
    from the first slice image if --slice-height/--slice-width are 0).
    """
    if args.skip_mask_gen:
        step_log["status"] = "skipped"
        step_log["message"] = "--skip-mask-gen set, assuming masks exist"
        print("[Step 1] Skipping mask generation (--skip-mask-gen).")
        return

    # Auto-detect slice resolution if not explicitly provided
    slice_h = args.slice_height
    slice_w = args.slice_width
    if slice_h == 0 or slice_w == 0:
        slice_h, slice_w = _infer_slice_shape(args.slice_glob)
        print(f"  Auto-detected slice dimensions: {slice_h} x {slice_w}")

    # Tissue-aware mask center selection
    center_x = None
    center_y = None
    tissue_overlap_diag = None
    if args.tissue_aware_mask:
        slice_files = _collect_slice_files(args.slice_glob)
        center_window = args.tissue_center_window if args.tissue_center_window > 0 else args.num_slices
        cy, cx, min_overlap = _find_tissue_aware_mask_center(
            slice_files,
            center_window,
            slice_h,
            slice_w,
            args.mask_radius,
            args.patch_size,
            white_threshold=args.white_threshold,
            black_threshold=args.black_threshold,
            min_tissue_overlap=args.min_mask_tissue_overlap,
            max_black_frac=args.max_mask_black_frac,
            max_attempts=args.mask_center_max_attempts,
            seed=args.mask_center_seed,
        )
        center_y, center_x = cy, cx
        print(
            f"  [TISSUE-AWARE] Selected mask center: ({center_x}, {center_y}) "
            f"(min tissue overlap across slices: {min_overlap:.4f})"
        )

    mask_script = str(
        Path(__file__).resolve().parent / "propagate_mask_across_slices.py"
    )

    cmd = [
        _check_python(),
        mask_script,
        "--num-slices", str(args.num_slices),
        "--height", str(slice_h),
        "--width", str(slice_w),
        "--radius", str(args.mask_radius),
        "--output-dir", str(run_dir / "masks"),
    ]
    if args.mask_reference:
        cmd.extend(["--mask", args.mask_reference])
    if args.mask_shape:
        cmd.extend(["--shape-mode", args.mask_shape])
    if center_x is not None:
        cmd.extend(["--center-x", str(center_x)])
    if center_y is not None:
        cmd.extend(["--center-y", str(center_y)])

    print(f"[Step 1] Generating masks at {slice_h}x{slice_w}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(
            f"Mask generation failed (exit code {result.returncode})"
        )

    step_log["status"] = "completed"
    step_log["mask_dir"] = str(run_dir / "masks")
    step_log["num_slices"] = args.num_slices
    step_log["slice_height"] = slice_h
    step_log["slice_width"] = slice_w

    # Record tissue-aware center if used
    if center_x is not None:
        step_log["tissue_aware_center"] = {"x": center_x, "y": center_y}
        step_log["tissue_aware_thresholds"] = {
            "white_threshold": args.white_threshold,
            "black_threshold": args.black_threshold,
            "min_mask_tissue_overlap": args.min_mask_tissue_overlap,
            "max_mask_black_frac": args.max_mask_black_frac,
        }
        # Compute and save diagnostics
        slice_files = _collect_slice_files(args.slice_glob)
        _save_mask_diagnostics(
            slice_files,
            str(run_dir / "masks"),
            center_y,
            center_x,
            args.num_slices,
            args.white_threshold,
            args.black_threshold,
            run_dir,
        )

    # Validate mask containment within patch_size
    _validate_mask_containment(
        str(run_dir / "masks"),
        args.patch_size,
        args.padding_ratio,
        strict=args.strict_bbox,
    )


def _step_build_pairs_csv(
    args: argparse.Namespace, run_dir: Path, step_log: dict,
) -> str:
    """Step 2: Build pairs CSV for extraction.

    Creates a CSV mapping each slice image to its mask file, with columns
    compatible with extract_roi_patches.py.

    Uses numeric index token matching (slice_0003 <-> mask_slice_0003) for
    robust pairing, with fallback to lexicographic zip.

    Returns path to the pairs CSV.
    """
    slice_files = _collect_slice_files(args.slice_glob)
    mask_dir = run_dir / "masks"
    mask_files = sorted(mask_dir.glob("mask_slice_*.png"))

    if not mask_files:
        raise SystemExit(
            f"No mask files found in {mask_dir}. "
            "Run mask generation first or check --skip-mask-gen."
        )

    # Robust matching by index token
    pairs = _match_slices_to_masks(slice_files, mask_files)
    num_pairs = len(pairs)

    pairs_dir = run_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = str(pairs_dir / "extraction_pairs.csv")

    # Determine common image directory for extract script
    image_dir = _resolve_common_parent(args.slice_glob)

    with open(pairs_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slice_id", "filename", "mask_filename",
                "coarse_label", "group_code",
            ],
        )
        writer.writeheader()
        for i, (slice_path_str, mask_path) in enumerate(pairs):
            slice_path = Path(slice_path_str)
            slice_id = f"{args.volume_id}_slice_{i:04d}"
            writer.writerow({
                "slice_id": slice_id,
                "filename": slice_path.name,
                "mask_filename": mask_path.name,
                "coarse_label": args.coarse_label,
                "group_code": "",
            })

    step_log["status"] = "completed"
    step_log["pairs_csv"] = pairs_csv
    step_log["image_dir"] = str(image_dir)
    step_log["mask_dir"] = str(mask_dir)
    step_log["num_pairs"] = num_pairs
    step_log["pairing_method"] = "index_token"
    print(f"[Step 2] Wrote pairs CSV: {pairs_csv} ({num_pairs} slice/mask pairs via index-token matching)")
    print(f"  image_dir: {image_dir}")
    print(f"  mask_dir:  {mask_dir}")
    return pairs_csv


def _step_extract_patches(
    args: argparse.Namespace, run_dir: Path, pairs_csv: str, step_log: dict,
) -> str:
    """Step 3: Extract ROI patches from full slices + masks.

    Calls extract_roi_patches.py.  In dry-run mode, creates synthetic
    extraction metadata instead (no actual image I/O).

    Returns path to patches_metadata.csv.
    """
    extract_script = str(
        Path(__file__).resolve().parent.parent
        / "patches" / "extract_roi_patches.py"
    )

    # Determine image and mask directories from pairs CSV context
    image_dir = _resolve_common_parent(args.slice_glob)
    mask_dir = run_dir / "masks"

    patches_out = str(run_dir / "patches")
    patches_meta = str(Path(patches_out) / "metadata" / "patches_metadata.csv")

    # Determine slice height/width for the extract script's internal bbox logic
    slice_h = args.slice_height
    slice_w = args.slice_width
    if slice_h == 0 or slice_w == 0:
        slice_h, slice_w = _infer_slice_shape(args.slice_glob)

    if args.dry_run:
        # In dry-run: compute bbox from masks and write synthetic metadata
        print("[Step 3] DRY RUN — creating synthetic extraction metadata")
        synthetic_metadata = _create_synthetic_extraction_metadata(
            pairs_csv, args, image_dir, mask_dir, slice_h, slice_w,
        )
        meta_out = Path(patches_out) / "metadata"
        meta_out.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        pd.DataFrame(synthetic_metadata).to_csv(patches_meta, index=False)

        crop_mode = "strict_fixed_raw" if args.strict_bbox else "fixed"
        stats = {
            "total_slices_processed": len(synthetic_metadata),
            "dry_run": True,
            "crop_mode": crop_mode,
            "strict_bbox_mode": args.strict_bbox,
            "patch_size": args.patch_size,
            "target_size": args.target_size,
        }
        stats_path = Path(patches_out) / "metadata" / "extraction_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        step_log["status"] = "dry_run"
        step_log["patches_metadata"] = patches_meta
        step_log["message"] = (
            "Synthetic metadata only (dry-run). "
            "Step 6 (stacking) will fall back to masks as a proxy volume."
        )
        print(f"  Wrote synthetic metadata: {patches_meta}")
        return patches_meta

    # Real extraction
    crop_mode = "strict_fixed_raw" if args.strict_bbox else "fixed"
    cmd = [
        _check_python(),
        extract_script,
        "--csv", pairs_csv,
        "--image-dir", str(image_dir),
        "--mask-dir", str(mask_dir),
        "--output-dir", patches_out,
        "--patch-size", str(args.patch_size),
        "--target-size", str(args.target_size),
        "--crop-mode", crop_mode,
    ]
    if not args.strict_bbox:
        cmd.extend(["--padding-ratio", str(args.padding_ratio)])

    print(f"[Step 3] Extracting patches: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(
            f"Patch extraction failed (exit code {result.returncode})"
        )

    step_log["status"] = "completed"
    step_log["patches_metadata"] = patches_meta
    step_log["patch_size"] = args.patch_size
    step_log["target_size"] = args.target_size
    return patches_meta


def _create_synthetic_extraction_metadata(
    pairs_csv: str,
    args: argparse.Namespace,
    image_dir: str,
    mask_dir: str,
    slice_h: int,
    slice_w: int,
) -> list[dict]:
    """Build synthetic extraction metadata from masks without running the
    heavy extraction script.  Used during dry-run.

    Computes bounding boxes from masks and creates metadata that downstream
    steps (inpaint/merge dry-run) can consume.
    """
    import pandas as pd
    df = pd.read_csv(pairs_csv)
    rows: list[dict] = []

    for _, row in df.iterrows():
        slice_id = row["slice_id"]
        source_image = str(Path(image_dir) / row["filename"])
        mask_path = str(Path(mask_dir) / row["mask_filename"])

        if not Path(mask_path).exists():
            print(f"  [WARN] Mask not found for dry-run bbox: {mask_path}")
            continue

        mask = np.array(Image.open(mask_path).convert("L"))
        y_min, x_min, y_max, x_max = _compute_mask_bbox(mask)
        raw_h = y_max - y_min
        raw_w = x_max - x_min

        if args.strict_bbox:
            # Strict mode: raw bbox → center tile → clamp (no padding, no square)
            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2
            half = args.patch_size // 2
            y_min = cy - half
            y_max = cy + half if args.patch_size % 2 == 0 else cy + half + 1
            x_min = cx - half
            x_max = cx + half if args.patch_size % 2 == 0 else cx + half + 1
            # Clamp to slice bounds (no zero-padding)
            y_min = max(0, y_min)
            y_max = min(slice_h, y_max)
            x_min = max(0, x_min)
            x_max = min(slice_w, x_max)
        else:
            # Legacy: padding + square + fixed crop
            pad_y = int(raw_h * args.padding_ratio)
            pad_x = int(raw_w * args.padding_ratio)
            y_min = max(0, y_min - pad_y)
            y_max = min(slice_h, y_max + pad_y)
            x_min = max(0, x_min - pad_x)
            x_max = min(slice_w, x_max + pad_x)

            # Make square
            y_min, x_min, y_max, x_max = _make_square_bbox(
                y_min, x_min, y_max, x_max, slice_h, slice_w,
            )

            # Fixed-size crop centered on bbox center (matching --crop-mode fixed)
            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2
            half = args.patch_size // 2
            y_min = cy - half
            y_max = cy + half if args.patch_size % 2 == 0 else cy + half + 1
            x_min = cx - half
            x_max = cx + half if args.patch_size % 2 == 0 else cx + half + 1
            # Clamp to slice bounds
            y_min = max(0, y_min)
            y_max = min(slice_h, y_max)
            x_min = max(0, x_min)
            x_max = min(slice_w, x_max)

        orig_h = y_max - y_min
        orig_w = x_max - x_min
        scale_x = args.target_size / max(orig_w, 1)
        scale_y = args.target_size / max(orig_h, 1)

        stylized_patch_image = str(
            Path(args.output_dir) / "patches" / "patches" / f"{slice_id}_patch.png"
        )
        stylized_patch_mask = str(
            Path(args.output_dir) / "patches" / "masks" / f"{slice_id}_mask.png"
        )

        rows.append({
            "slice_id": slice_id,
            "source_image": source_image,
            "source_mask": mask_path,
            "patch_image": stylized_patch_image,
            "patch_mask": stylized_patch_mask,
            "bbox_y_min": y_min,
            "bbox_x_min": x_min,
            "bbox_y_max": y_max,
            "bbox_x_max": x_max,
            "bbox_height": orig_h,
            "bbox_width": orig_w,
            "target_size": args.target_size,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "coarse_label": str(row.get("coarse_label", "")),
            "group_code": "",
            "seed_placeholder": 0,
        })

    return rows


def _step_run_inpainting(
    args: argparse.Namespace, run_dir: Path, patches_meta: str, step_log: dict,
) -> str:
    """Step 4: Inpaint ROI patches via inpaint_roi_patches.py.

    Returns the inpaint output directory.
    """
    inpaint_script = str(
        Path(__file__).resolve().parent.parent
        / "patches" / "inpaint_roi_patches.py"
    )

    inpaint_out = str(run_dir / "inpainted")
    cmd = [
        _check_python(),
        inpaint_script,
        "--metadata-csv", patches_meta,
        "--output-dir", inpaint_out,
    ]

    run_dry_inpaint = args.dry_run or not args.base_model
    if run_dry_inpaint:
        cmd.append("--dry-run")
    else:
        cmd.extend(["--base-model", args.base_model])
        if args.lora_weights:
            cmd.extend(["--lora-weights", args.lora_weights])
        if args.inpaint_device:
            cmd.extend(["--device", args.inpaint_device])
        if args.inpaint_steps:
            cmd.extend(["--num-steps", str(args.inpaint_steps)])
        if args.guidance_scale:
            cmd.extend(["--guidance-scale", str(args.guidance_scale)])
        if args.inpaint_strength:
            cmd.extend(["--strength", str(args.inpaint_strength)])
        if args.seed_base:
            cmd.extend(["--seed-base", str(args.seed_base)])

    print(f"[Step 4] Running inpainting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(
            f"Inpainting failed (exit code {result.returncode})"
        )

    step_log["status"] = "dry_run" if run_dry_inpaint else "completed"
    step_log["inpaint_dir"] = inpaint_out
    step_log["base_model"] = args.base_model
    step_log["lora_weights"] = args.lora_weights
    return inpaint_out


def _step_merge_patches(
    args: argparse.Namespace, run_dir: Path, inpaint_dir: str,
    patches_meta: str, step_log: dict,
) -> str:
    """Step 5: Merge inpainted patches back into full-res slices.

    Calls merge_inpainted_patches.py.  In dry-run or when inpaint produced
    no output, reports status and skips the subprocess call (since the
    merge script reads input CSVs before checking --dry-run).

    Returns path to merge output slices directory.
    """
    merge_out = str(run_dir / "merged")

    inpaint_meta_csv = str(Path(inpaint_dir) / "metadata" / "inpaint_metadata.csv")
    inpaint_meta_exists = Path(inpaint_meta_csv).is_file()

    force_dry_merge = args.dry_run or not inpaint_meta_exists
    if force_dry_merge:
        if not inpaint_meta_exists and not args.dry_run and args.base_model:
            print(
                "  [WARN] inpaint_metadata.csv not found. "
                "Has inpainting been run?"
            )
        print("[Step 5] DRY RUN — no inpaint output to merge (dry-run or no model).")
        step_log["status"] = "dry_run"
        step_log["message"] = "No inpaint metadata to merge (dry-run or no model)."
        step_log["merge_dir"] = merge_out
        return merge_out

    # Real merge call
    merge_script = str(
        Path(__file__).resolve().parent.parent
        / "patches" / "merge_inpainted_patches.py"
    )

    cmd = [
        _check_python(),
        merge_script,
        "--inpaint-metadata", inpaint_meta_csv,
        "--extract-metadata", patches_meta,
        "--output-dir", merge_out,
        "--feather-radius", str(args.feather_radius),
    ]

    print(f"[Step 5] Merging patches: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(
            f"Merge failed (exit code {result.returncode})"
        )

    step_log["status"] = "completed"
    step_log["merge_dir"] = merge_out
    return merge_out


def _step_stack_nifti(
    args: argparse.Namespace, run_dir: Path, merge_dir: str, step_log: dict,
) -> str:
    """Step 6: Stack merged full-res slices into a NIfTI volume.

    Reads from the merge output directory if populated; otherwise falls back
    to masks (for testing) or skips if nothing is available.
    """
    stack_script = str(
        Path(__file__).resolve().parent / "stack_slices_to_nifti.py"
    )

    out_nifti = str(run_dir / "volume.nii.gz")
    stats_json = str(run_dir / "volume_stats.json")

    # Determine which images to stack — prefer merged, then masks as fallback
    merged_images_glob = str(Path(merge_dir) / "slices" / "*_edited.png")
    merged_files = sorted(_glob.glob(merged_images_glob))

    masks_glob = str(run_dir / "masks" / "mask_slice_*.png")

    if merged_files:
        input_glob = merged_images_glob
        source_desc = "merged slices (inpainted patches pasted in)"
        mask_fallback = False
    else:
        input_glob = masks_glob
        source_desc = (
            "masks (fallback — no merged/inpainted output available)"
        )
        mask_fallback = True
        step_log["mask_fallback"] = True
        step_log["message"] = (
            "No merged slices available; stacked masks as proxy volume. "
            "Coherence metrics will reflect mask consistency, not inpaint quality."
        )
        print(
            "[Step 6] No merged slices found; "
            "using masks as proxy volume for testing."
        )

    cmd = [
        _check_python(),
        stack_script,
        "--input-glob", input_glob,
        "--output-nifti", out_nifti,
        "--pixdim",
        str(args.pixdim[0]),
        str(args.pixdim[1]),
        str(args.pixdim[2]),
        "--stats-json", stats_json,
    ]

    print(f"[Step 6] Stacking to NIfTI ({source_desc}): {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        step_log["status"] = "skipped_no_images"
        step_log["message"] = (
            f"No images found for stacking ({source_desc})."
        )
        return out_nifti

    step_log["status"] = "completed"
    step_log["source"] = source_desc
    step_log["nifti_path"] = out_nifti
    step_log["stats_json"] = stats_json
    return out_nifti


def _step_compute_coherence(
    args: argparse.Namespace, run_dir: Path, nifti_path: str, step_log: dict,
) -> None:
    """Step 7: Compute Z-coherence metrics from the NIfTI volume and evaluate
    optional threshold parameters (--min-adjacent-ssim,
    --min-z-gradient-smoothness)."""
    metrics_script = str(
        Path(__file__).resolve().parent / "compute_z_coherence_metrics.py"
    )

    out_json = str(run_dir / "coherence_metrics.json")

    cmd = [
        _check_python(),
        metrics_script,
        "--volume-nifti", nifti_path,
        "--output-json", out_json,
    ]

    print(f"[Step 7] Computing coherence: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        step_log["status"] = "skipped_no_volume"
        step_log["message"] = "Volume NIfTI not found. Skipping."
        return

    step_log["status"] = "completed"
    step_log["metrics_json"] = out_json

    # --- Z-coherence threshold evaluation (optional) ---
    thresholds: dict[str, float] = {}
    if args.min_adjacent_ssim is not None:
        thresholds["min_adjacent_ssim"] = args.min_adjacent_ssim
    if args.min_z_gradient_smoothness is not None:
        thresholds["min_z_gradient_smoothness"] = args.min_z_gradient_smoothness

    if not thresholds:
        return  # no threshold gating configured; preserve existing behavior

    metrics_path = Path(out_json)
    if not metrics_path.is_file():
        step_log["threshold_evaluation"] = {
            "thresholds": thresholds,
            "error": f"Metrics file not found: {out_json}",
            "passed": False,
        }
        print(f"  [WARN] Threshold evaluation skipped — {out_json} not found.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    evaluation: dict[str, Any] = {
        "thresholds": dict(thresholds),
        "metrics_used": {
            "adjacent_ssim_mean": metrics.get("adjacent_ssim", {}).get("mean"),
            "z_gradient_smoothness_mean": (
                metrics.get("z_gradient_smoothness", {}).get("mean_abs_gradient")
            ),
        },
        "checks": {},
        "passed": True,
    }

    # Check adjacent SSIM
    if "min_adjacent_ssim" in thresholds:
        actual_ssim = metrics.get("adjacent_ssim", {}).get("mean")
        if actual_ssim is not None:
            ssim_ok = bool(actual_ssim >= thresholds["min_adjacent_ssim"])
            evaluation["checks"]["adjacent_ssim"] = {
                "threshold": thresholds["min_adjacent_ssim"],
                "actual": actual_ssim,
                "passed": ssim_ok,
            }
            if not ssim_ok:
                evaluation["passed"] = False

    # Check Z-gradient smoothness (lower is smoother, so actual must be <= threshold)
    if "min_z_gradient_smoothness" in thresholds:
        actual_grad = metrics.get("z_gradient_smoothness", {}).get("mean_abs_gradient")
        if actual_grad is not None:
            grad_ok = bool(actual_grad <= thresholds["min_z_gradient_smoothness"])
            evaluation["checks"]["z_gradient_smoothness"] = {
                "threshold": thresholds["min_z_gradient_smoothness"],
                "actual": actual_grad,
                "passed": grad_ok,
            }
            if not grad_ok:
                evaluation["passed"] = False

    if evaluation["passed"]:
        print("  [THRESHOLDS] All coherence thresholds met.")
    else:
        failed_checks = [
            f"{name}: actual={chk['actual']:.6f}, threshold={chk['threshold']}"
            for name, chk in evaluation["checks"].items()
            if not chk["passed"]
        ]
        warn_msg = "Coherence thresholds NOT met: " + "; ".join(failed_checks)
        print(f"  [WARN] {warn_msg}")

    step_log["threshold_evaluation"] = evaluation


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Tile-first volume inpainting pipeline for high-res histology "
            "slice stacks.  Extracts ROI patches, inpaints at model "
            "resolution, and merges back — no full-slice resizing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples (dry-run smoke test):\n\n"
            "  # Strict raw-bbox mode (default for B.1 tile-first):\n"
            "  python scripts/3d/run_volume_inpaint_pipeline.py \\\n"
            "    --slice-glob '/tmp/slices/slice_*.png' \\\n"
            "    --volume-id test_vol --coarse-label non_cancer \\\n"
            "    --num-slices 10 --mask-radius 0.15 \\\n"
            "    --strict-bbox --patch-size 512 --target-size 512 \\\n"
            "    --output-dir /tmp/pipeline_run --dry-run\n\n"
            "  # Non-strict mode with padding + square:\n"
            "  python scripts/3d/run_volume_inpaint_pipeline.py \\\n"
            "    --slice-glob '/tmp/slices/slice_*.png' \\\n"
            "    --volume-id test_vol --coarse-label non_cancer \\\n"
            "    --num-slices 10 --mask-radius 0.3 \\\n"
            "    --patch-size 1024 --target-size 512 --padding-ratio 0.15 \\\n"
            "    --output-dir /tmp/pipeline_run --dry-run\n"
        ),
    )

    # Slice input
    ap.add_argument(
        "--slice-glob", required=True,
        help=(
            'Glob pattern for ordered slice images, e.g. '
            '"slices/volume_*/slice_*.png".'
        ),
    )
    ap.add_argument(
        "--volume-id", default="volume_001",
        help="Volume identifier for provenance.",
    )
    ap.add_argument(
        "--coarse-label", default="non_cancer",
        help="Label for all slices (passed through to inpaint prompt).",
    )
    ap.add_argument(
        "--num-slices", type=int, default=10,
        help="Number of slices in the volume (for mask generation).",
    )

    # Mask geometry
    ap.add_argument(
        "--slice-height", type=int, default=0,
        help="Slice height in pixels. If 0 (default), auto-detected from first slice image.",
    )
    ap.add_argument(
        "--slice-width", type=int, default=0,
        help="Slice width in pixels. If 0 (default), auto-detected from first slice image.",
    )
    ap.add_argument(
        "--mask-radius", type=float, default=0.3,
        help="Mask radius as fraction of smaller dimension (0-1).",
    )
    ap.add_argument(
        "--mask-shape", choices=["circle", "ellipse"], default="circle",
        help="Shape of generated mask.",
    )
    ap.add_argument(
        "--mask-reference", default=None,
        help="Reference mask image to replicate (overrides --mask-radius/shape).",
    )
    ap.add_argument(
        "--skip-mask-gen", action="store_true",
        help="Skip mask generation (masks must already exist in run dir).",
    )

    # Strict vs non-strict mode
    ap.add_argument(
        "--strict-bbox", action="store_true",
        help=(
            "Strict raw-bbox containment mode (no padding, no square transform). "
            "Validates raw mask bbox width AND height <= --patch-size. "
            "Uses crop-mode=strict_fixed_raw in extraction. "
            "Ignores --padding-ratio. "
            "When set without explicit --patch-size, defaults to 512."
        ),
    )

    # Tile/patch extraction
    ap.add_argument(
        "--patch-size", type=int, default=1024,
        help=(
            "Full-resolution patch crop size in pixels. "
            "In strict mode (--strict-bbox): both raw bbox dimensions must be <= this. "
            "In non-strict mode: the cylindrical mask bbox (after padding + square) "
            "must fit within this. Default 1024."
        ),
    )
    ap.add_argument(
        "--target-size", type=int, default=512,
        help=(
            "Model working resolution for inpainting. Patches are resized "
            "to this size before being passed to SDXL. Default 512."
        ),
    )
    ap.add_argument(
        "--padding-ratio", type=float, default=0.15,
        help=(
            "Context padding as fraction of mask bbox dimension, added "
            "around the ROI before extraction. Ignored in strict mode. Default 0.15."
        ),
    )
    ap.add_argument(
        "--feather-radius", type=int, default=16,
        help="Feather radius (px) for blending the inpainted patch back "
        "into the full-res slice. Default 16.",
    )

    # Output
    ap.add_argument(
        "--output-dir",
        default="data/artifacts/3d/volume_inpaint_runs",
        help="Root output directory for this run.",
    )

    # Inpainting model
    ap.add_argument(
        "--base-model", default=None,
        help=(
            "Path to SDXL base model. If omitted, runs in dry-inpaint mode."
        ),
    )
    ap.add_argument(
        "--lora-weights", default=None,
        help="Path to LoRA weights file.",
    )
    ap.add_argument("--inpaint-device", default="cuda")
    ap.add_argument("--inpaint-steps", type=int, default=40)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--inpaint-strength", type=float, default=0.55)
    ap.add_argument("--seed-base", type=int, default=42)

    # NIfTI stacking
    ap.add_argument(
        "--pixdim", type=float, nargs=3, default=[1.0, 1.0, 2.0],
        metavar=("DX", "DY", "DZ"),
        help="Voxel dimensions in mm.",
    )

    # Tissue-aware mask placement
    ap.add_argument(
        "--tissue-aware-mask", action="store_true",
        help=(
            "Enable tissue-aware mask placement. The mask center is "
            "chosen inside tissue-valid areas common to ALL sequential "
            "slices, avoiding white/black voids. See also --white-threshold, "
            "--black-threshold."
        ),
    )
    ap.add_argument(
        "--white-threshold", type=int, default=235,
        help=(
            "White-void threshold for tissue-aware mask. "
            "Pixels with all RGB channels >= this are considered white "
            "void and excluded. Default 235."
        ),
    )
    ap.add_argument(
        "--black-threshold", type=int, default=10,
        help=(
            "Black-void threshold for tissue-aware mask. "
            "Pixels with all RGB channels <= this are considered black "
            "void and excluded. Default 10."
        ),
    )
    ap.add_argument(
        "--min-mask-tissue-overlap", type=float, default=0.85,
        help=(
            "Minimum fraction of mask pixels that must be tissue-valid "
            "in each slice. Default 0.85."
        ),
    )
    ap.add_argument(
        "--max-mask-black-frac", type=float, default=0.10,
        help=(
            "Maximum allowed black-void fraction inside mask per slice. "
            "Default 0.10."
        ),
    )
    ap.add_argument(
        "--mask-center-seed", type=int, default=42,
        help=(
            "Random seed for tissue-aware mask center sampling. "
            "Default 42."
        ),
    )
    ap.add_argument(
        "--mask-center-max-attempts", type=int, default=50,
        help=(
            "Maximum attempts to find a valid tissue-aware mask center. "
            "Default 50."
        ),
    )
    ap.add_argument(
        "--tissue-center-window", type=int, default=0,
        help=(
            "Number of leading slices to use for tissue-aware center finding "
            "(default 0 = use all --num-slices slices). "
            "Useful when later slices have poor tissue overlap with earlier ones."
        ),
    )

    # Z-coherence thresholding (optional)
    ap.add_argument(
        "--min-adjacent-ssim", type=float, default=None,
        help=(
            "Minimum acceptable mean adjacent-slice SSIM (0-1). "
            "If the computed mean SSIM is below this threshold, "
            "a warning is issued and the manifest marks "
            "coherence_thresholds.passed as false. "
            "If not set, no threshold gating is applied."
        ),
    )
    ap.add_argument(
        "--min-z-gradient-smoothness", type=float, default=None,
        help=(
            "Minimum acceptable Z-gradient smoothness. "
            "The metric is mean absolute gradient between consecutive "
            "slices (lower = smoother). If the computed gradient exceeds "
            "this threshold, a warning is issued. "
            "If not set, no threshold gating is applied."
        ),
    )

    # Mode
    ap.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Print plan without running heavy steps. Still runs mask "
            "generation, pairs CSV, extraction metadata, and stacking/"
            "metrics if possible."
        ),
    )
    ap.add_argument(
        "--no-merge", action="store_true",
        help="Stop after inpainting (skip merge, stack, and coherence).",
    )

    args = ap.parse_args()

    # --- Strict-bbox overrides ---
    if args.strict_bbox:
        args.padding_ratio = 0.0  # padding ignored in strict mode
        if args.patch_size == 1024:  # user didn't override default → 512
            args.patch_size = 512

    # --- Validations ---
    if args.patch_size < args.target_size:
        ap.error(
            f"--patch-size ({args.patch_size}) must be >= "
            f"--target-size ({args.target_size}). "
            f"The patch must be at least as large as the model target."
        )
    if args.num_slices <= 0:
        ap.error("--num-slices must be > 0")
    if not (0 <= args.white_threshold <= 255):
        ap.error("--white-threshold must be in [0, 255]")
    if not (0 <= args.black_threshold <= 255):
        ap.error("--black-threshold must be in [0, 255]")
    if args.white_threshold <= args.black_threshold:
        ap.error("--white-threshold must be > --black-threshold")
    if not (0.0 <= args.min_mask_tissue_overlap <= 1.0):
        ap.error("--min-mask-tissue-overlap must be in [0, 1]")
    if not (0.0 <= args.max_mask_black_frac <= 1.0):
        ap.error("--max-mask-black-frac must be in [0, 1]")
    if args.mask_center_max_attempts <= 0:
        ap.error("--mask-center-max-attempts must be > 0")

    if args.min_adjacent_ssim is not None and not (0.0 <= args.min_adjacent_ssim <= 1.0):
        ap.error("--min-adjacent-ssim must be in [0, 1]")
    if args.min_z_gradient_smoothness is not None and args.min_z_gradient_smoothness < 0:
        ap.error("--min-z-gradient-smoothness must be >= 0")

    # --- Setup run directory ---
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.volume_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Detect slice dimensions for display
    display_h = args.slice_height
    display_w = args.slice_width
    if display_h == 0 or display_w == 0:
        try:
            display_h, display_w = _infer_slice_shape(args.slice_glob)
        except Exception:
            pass

    print("=" * 64)
    print("Tile-First Volume Inpainting Pipeline")
    print(f"  Volume ID:     {args.volume_id}")
    print(f"  Output dir:    {run_dir}")
    print(f"  Slice glob:    {args.slice_glob}")
    print(f"  Slice dims:    {display_h} x {display_w}" if display_h else "")
    print(f"  Mask radius:   {args.mask_radius}")
    print(f"  Patch size:    {args.patch_size}")
    print(f"  Target size:   {args.target_size}")
    print(f"  Padding ratio: {args.padding_ratio}")
    print(f"  Strict bbox:   {args.strict_bbox}")
    print(f"  Tissue-aware:  {args.tissue_aware_mask}")
    if args.tissue_aware_mask:
        print(f"  White thresh:  {args.white_threshold}")
        print(f"  Black thresh:  {args.black_threshold}")
        print(f"  Min overlap:   {args.min_mask_tissue_overlap}")
        print(f"  Max black:     {args.max_mask_black_frac}")
    print(f"  Base model:    {args.base_model or '(dry-run mode)'}")
    print(f"  Dry run:       {args.dry_run}")
    print("=" * 64)

    manifest: dict[str, Any] = {
        "pipeline": "run_volume_inpaint_pipeline",
        "version": "2.3.0",
        "description": "Tile-first volume inpainting (patch-based, no full-slice resize)",
        "volume_id": args.volume_id,
        "timestamp_utc": timestamp,
        "run_dir": str(run_dir.resolve()),
        "args": vars(args),
        "strict_bbox_mode": args.strict_bbox,
        "tissue_aware_mask": args.tissue_aware_mask,
        "containment_rule": "raw_bbox_no_padding_square" if args.strict_bbox else "padded_squared",
        "steps": {},
    }

    # --- Step 1: Generate masks ---
    print("\n--- Step 1 / 8: Mask Generation ---")
    manifest["steps"]["mask_generation"] = {}
    _step_generate_masks(args, run_dir, manifest["steps"]["mask_generation"])

    # --- Step 2: Build pairs CSV ---
    print("\n--- Step 2 / 8: Build Pairs CSV for Extraction ---")
    manifest["steps"]["pairs_csv"] = {}
    pairs_csv = _step_build_pairs_csv(
        args, run_dir, manifest["steps"]["pairs_csv"],
    )

    # --- Step 3: Extract patches ---
    print("\n--- Step 3 / 8: Extract ROI Patches ---")
    manifest["steps"]["extraction"] = {}
    patches_meta = _step_extract_patches(
        args, run_dir, pairs_csv, manifest["steps"]["extraction"],
    )

    # --- Step 4: Inpaint patches ---
    print("\n--- Step 4 / 8: Inpaint ROI Patches ---")
    manifest["steps"]["inpainting"] = {}
    inpaint_dir = _step_run_inpainting(
        args, run_dir, patches_meta, manifest["steps"]["inpainting"],
    )

    # --- Step 5: Merge patches (optional) ---
    merge_dir = inpaint_dir  # fallback for stacking
    if not args.no_merge:
        print("\n--- Step 5 / 8: Merge Inpainted Patches ---")
        manifest["steps"]["merge"] = {}
        merge_dir = _step_merge_patches(
            args, run_dir, inpaint_dir, patches_meta,
            manifest["steps"]["merge"],
        )
    else:
        print("\n--- Step 5 / 8: Merge — SKIPPED (--no-merge) ---")
        manifest["steps"]["merge"] = {
            "status": "skipped",
            "message": "--no-merge flag set",
        }

    # --- Step 6: Stack to NIfTI ---
    print("\n--- Step 6 / 8: Stack to NIfTI ---")
    manifest["steps"]["stacking"] = {}
    nifti_path = _step_stack_nifti(
        args, run_dir, merge_dir, manifest["steps"]["stacking"],
    )

    # --- Step 7: Coherence metrics + threshold evaluation ---
    print("\n--- Step 7 / 8: Coherence Metrics + Threshold Evaluation ---")
    manifest["steps"]["coherence"] = {}
    _step_compute_coherence(
        args, run_dir, nifti_path, manifest["steps"]["coherence"],
    )

    # --- Step 8: Save manifest ---
    print("\n--- Step 8 / 8: Save Manifest ---")
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8",
    )
    print(f"Saved manifest: {manifest_path}")

    # --- Summary ---
    print("\n" + "=" * 64)
    print("Run Summary:")
    print(f"  Run dir:       {run_dir}")
    print(f"  Steps ran:     {len(manifest['steps'])}")
    for step_name, step_data in manifest["steps"].items():
        print(f"    {step_name}: {step_data.get('status', 'unknown')}")
    print("=" * 64)


if __name__ == "__main__":
    main()
