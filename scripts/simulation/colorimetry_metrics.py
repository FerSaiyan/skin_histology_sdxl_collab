#!/usr/bin/env python
"""
Colorimetry metrics for comparing reference and edited images.

Provides reusable functions and a CLI for computing color difference
metrics between a reference (original) image and an edited image, with
optional mask-based ROI decomposition.

Metrics (classifier-free):
  - RGB mean absolute error (global, inside mask, outside mask)
  - CIELAB delta proxy (mean delta in LAB space, inside/outside mask)
    (requires cv2; gracefully degrades if unavailable)
  - Channel distribution drift (KS statistic per channel)

Mask handling:
  - If a mask is provided, metrics are split into inside/outside regions.
  - If mask is None / all-zero, global metrics are returned with
    inside/outside fields set to NaN and a status flag.

Usage:
  python scripts/simulation/colorimetry_metrics.py \\
      --reference ref.png --edited edit.png [--mask mask.png] \\
      [--output-json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Optional imports (graceful degradation)
# ---------------------------------------------------------------------------

_HAS_CV2 = False
try:
    # Suppress stderr noise from cv2/numpy compat issues during import
    import contextlib

    with contextlib.redirect_stderr(open(os.dev_null, "w")):
        import cv2  # noqa: F401
    _HAS_CV2 = True
except (ImportError, AttributeError, Exception):
    pass

_HAS_SCIPY_STATS = False
try:
    from scipy.stats import ks_2samp as _ks_2samp

    _HAS_SCIPY_STATS = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_array(path: str) -> np.ndarray:
    """Load an image as an RGB uint8 numpy array.

    Raises:
        FileNotFoundError if the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    return np.array(Image.open(str(p)).convert("RGB"))


def _ensure_binary_mask(
    mask: Optional[np.ndarray],
    ref_shape: Tuple[int, ...],
) -> Tuple[Optional[np.ndarray], bool]:
    """Validate and convert mask to boolean.

    Returns (mask_bool, is_empty).  If mask is None or all-zero,
    returns (None, True).
    """
    if mask is None:
        return None, True
    if mask.ndim == 3:
        mask = mask[..., 0]  # take first channel
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return None, True
    # Broadcast / warn if shape mismatch
    if mask_bool.shape[:2] != ref_shape[:2]:
        print(
            f"[WARN] Mask shape {mask_bool.shape} differs from image shape "
            f"{ref_shape[:2]}; will not apply mask.",
            file=sys.stderr,
        )
        return None, True
    return mask_bool, False


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_rgb_mae(
    original: np.ndarray,
    edited: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute RGB mean absolute error (global / inside / outside mask).

    Args:
        original: Reference RGB uint8 image (H, W, 3).
        edited: Edited RGB uint8 image (H, W, 3).
        mask: Optional binary mask (H, W).  Region where mask > 0 is
              treated as ROI.

    Returns:
        dict with keys:
          rgb_mae_global  : float  — MAE over all pixels
          rgb_mae_inside  : float or NaN — MAE inside mask ROI
          rgb_mae_outside : float or NaN — MAE outside mask ROI
          rgb_mae_mask_status : str — "applied" | "no_mask" | "empty_mask"
    """
    orig_f = original.astype(np.float64)
    edit_f = edited.astype(np.float64)
    diff = np.abs(orig_f - edit_f)                     # (H, W, 3)
    diff_channel_mean = diff.mean(axis=2)              # (H, W)

    result: Dict[str, Any] = {}
    result["rgb_mae_global"] = float(diff_channel_mean.mean())

    mask_bool, is_empty = _ensure_binary_mask(mask, original.shape)
    if mask_bool is None:
        status = "no_mask" if is_empty else "empty_mask"
        result["rgb_mae_inside"] = float("nan")
        result["rgb_mae_outside"] = float("nan")
        result["rgb_mae_mask_status"] = status
    else:
        result["rgb_mae_mask_status"] = "applied"
        inside = diff_channel_mean[mask_bool]
        outside = diff_channel_mean[~mask_bool]
        result["rgb_mae_inside"] = float(inside.mean()) if inside.size > 0 else float("nan")
        result["rgb_mae_outside"] = float(outside.mean()) if outside.size > 0 else float("nan")

    return result


def compute_lab_delta_proxy(
    original: np.ndarray,
    edited: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute mean CIELAB delta proxy inside and outside the mask.

    Uses ``cv2.cvtColor(..., cv2.COLOR_RGB2LAB)`` for the conversion.
    If cv2 is unavailable, returns a dict with ``lab_available=False``.

    Returns:
        dict with keys:
          lab_available       : bool
          lab_delta_inside    : float or NaN — mean LAB Euclidean distance in ROI
          lab_delta_outside   : float or NaN — mean LAB Euclidean distance outside ROI
          lab_delta_global    : float or NaN — mean LAB distance over all pixels
          lab_delta_status    : str — "ok" | "cv2_unavailable" | "no_mask" | "empty_mask"
    """
    result: Dict[str, Any] = {"lab_available": False}

    if not _HAS_CV2:
        result["lab_delta_global"] = float("nan")
        result["lab_delta_inside"] = float("nan")
        result["lab_delta_outside"] = float("nan")
        result["lab_delta_status"] = "cv2_unavailable"
        return result

    # Convert to LAB
    lab_orig = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab_edit = cv2.cvtColor(edited, cv2.COLOR_RGB2LAB).astype(np.float64)

    # Euclidean distance per pixel in LAB space
    delta = np.sqrt(((lab_orig - lab_edit) ** 2).sum(axis=2))  # (H, W)

    result["lab_available"] = True
    result["lab_delta_global"] = float(delta.mean())

    mask_bool, is_empty = _ensure_binary_mask(mask, original.shape)
    if mask_bool is None:
        status = "no_mask" if is_empty else "empty_mask"
        result["lab_delta_inside"] = float("nan")
        result["lab_delta_outside"] = float("nan")
        result["lab_delta_status"] = status
    else:
        result["lab_delta_status"] = "ok"
        inside = delta[mask_bool]
        outside = delta[~mask_bool]
        result["lab_delta_inside"] = float(inside.mean()) if inside.size > 0 else float("nan")
        result["lab_delta_outside"] = float(outside.mean()) if outside.size > 0 else float("nan")

    return result


def compute_channel_drift(
    original: np.ndarray,
    edited: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """KS-statistic-based per-channel distribution drift.

    For each RGB channel, computes the two-sample KS statistic between
    original and edited pixel values.  If a mask is provided, also
    computes inside/outside drift.

    If scipy.stats is unavailable, returns ``channel_drift_available=False``.

    Returns:
        dict with keys:
          channel_drift_available  : bool
          channel_drift_global     : list[float] — KS stat per channel (R, G, B)
          channel_drift_inside     : list[float] or None
          channel_drift_outside    : list[float] or None
          channel_drift_status     : str
    """
    result: Dict[str, Any] = {"channel_drift_available": False}

    if not _HAS_SCIPY_STATS:
        result["channel_drift_global"] = None
        result["channel_drift_inside"] = None
        result["channel_drift_outside"] = None
        result["channel_drift_status"] = "scipy_unavailable"
        return result

    def _ks_per_channel(arr_a: np.ndarray, arr_b: np.ndarray) -> list:
        stats = []
        for c in range(3):
            stat, _ = _ks_2samp(arr_a[..., c].ravel(), arr_b[..., c].ravel())
            stats.append(float(stat))
        return stats

    result["channel_drift_available"] = True
    result["channel_drift_global"] = _ks_per_channel(original, edited)

    mask_bool, is_empty = _ensure_binary_mask(mask, original.shape)
    if mask_bool is None:
        status = "no_mask" if is_empty else "empty_mask"
        result["channel_drift_inside"] = None
        result["channel_drift_outside"] = None
        result["channel_drift_status"] = status
    else:
        result["channel_drift_status"] = "applied"
        orig_inside = original[mask_bool]
        orig_outside = original[~mask_bool]
        edit_inside = edited[mask_bool]
        edit_outside = edited[~mask_bool]

        if orig_inside.size > 0 and edit_inside.size > 0:
            result["channel_drift_inside"] = _ks_per_channel(orig_inside, edit_inside)
        else:
            result["channel_drift_inside"] = None

        if orig_outside.size > 0 and edit_outside.size > 0:
            result["channel_drift_outside"] = _ks_per_channel(orig_outside, edit_outside)
        else:
            result["channel_drift_outside"] = None

    return result


def compute_all_colorimetry(
    original: np.ndarray,
    edited: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Convenience wrapper computing all colorimetry metrics.

    Returns a single flat dict combining results from
    ``compute_rgb_mae``, ``compute_lab_delta_proxy``, and
    ``compute_channel_drift``.
    """
    out = {}
    out.update(compute_rgb_mae(original, edited, mask))
    out.update(compute_lab_delta_proxy(original, edited, mask))
    out.update(compute_channel_drift(original, edited, mask))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Compute colorimetry metrics between reference and edited images."
    )
    ap.add_argument(
        "--reference",
        required=True,
        help="Path to reference (original) image.",
    )
    ap.add_argument(
        "--edited",
        required=True,
        help="Path to edited image.",
    )
    ap.add_argument(
        "--mask",
        default=None,
        help="Optional path to mask image (pixel values > 0 define ROI).",
    )
    ap.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write metrics as JSON.",
    )
    return ap


def main() -> None:
    ap = _build_cli()
    args = ap.parse_args()

    # Load images
    original = _load_array(args.reference)
    edited = _load_array(args.edited)

    # Validate matching shapes for metric comparability
    if original.shape[:2] != edited.shape[:2]:
        raise SystemExit(
            f"ERROR: Reference shape {original.shape[:2]} does not match "
            f"edited shape {edited.shape[:2]}. Images must have the same "
            f"height and width for pixel-wise metric computation."
        )

    mask: Optional[np.ndarray] = None
    if args.mask:
        mask = _load_array(args.mask)
        if mask.shape[:2] != original.shape[:2]:
            raise SystemExit(
                f"ERROR: Mask shape {mask.shape[:2]} does not match "
                f"reference image shape {original.shape[:2]}."
            )

    # Compute all metrics
    metrics = compute_all_colorimetry(original, edited, mask)

    # Compact summary
    print("=== Colorimetry Metrics ===")
    print(f"  RGB MAE global      : {metrics.get('rgb_mae_global', 'N/A'):>8.4f}")
    print(f"  RGB MAE inside      : {metrics.get('rgb_mae_inside', 'N/A'):>8.4f}")
    print(f"  RGB MAE outside     : {metrics.get('rgb_mae_outside', 'N/A'):>8.4f}")
    print(f"  RGB MAE mask status : {metrics.get('rgb_mae_mask_status', 'N/A')}")

    if metrics.get("lab_available"):
        print(f"  LAB delta global    : {metrics['lab_delta_global']:>8.4f}")
        print(f"  LAB delta inside    : {metrics['lab_delta_inside']:>8.4f}")
        print(f"  LAB delta outside   : {metrics['lab_delta_outside']:>8.4f}")
        print(f"  LAB delta status    : {metrics['lab_delta_status']}")
    else:
        print(f"  LAB delta           : unavailable ({metrics.get('lab_delta_status', 'N/A')})")

    if metrics.get("channel_drift_available"):
        g = metrics["channel_drift_global"]
        print(f"  Channel drift global: R={g[0]:.4f} G={g[1]:.4f} B={g[2]:.4f}")
        print(f"  Channel drift status: {metrics['channel_drift_status']}")
    else:
        print(f"  Channel drift       : unavailable ({metrics.get('channel_drift_status', 'N/A')})")

    # Write JSON if requested
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nWrote metrics to {out_path}")


if __name__ == "__main__":
    main()
