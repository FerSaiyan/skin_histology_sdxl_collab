#!/usr/bin/env python
"""
Estimate epidermis-air interface normal from a segmentation mask.

Pipeline
--------
1. Load a segmentation mask (values 0 = air / background, 1 = epidermis).
2. Trace the boundary between class 0 and class 1 via contour finding.
3. Fit a line to the boundary points using PCA; the first principal
   component gives the interface tangent, the second (minor) component
   gives the normal direction.
4. Disambiguate the outward-pointing normal (air → epidermis) by
   checking a small offset along ±normal.
5. Return the normal angle (image-coordinate convention), the unit
   normal vector, and a confidence metric based on how line-like the
   boundary is.

Image coordinate convention
---------------------------
- Origin at top-left
- x increases right, y increases down
- Angles are measured CCW from the positive x-axis:
    0° = right, 90° = down, 180° = left, −90° = up
- The *outward normal* points from air (class 0) toward epidermis
  (class 1).

Usage
-----
::

    python scripts/optical_ga/estimate_epidermis_normal.py \\
        --mask /path/to/mask.png --output /tmp/normal_result.json

Or import::

    from scripts.optical_ga.estimate_epidermis_normal import (
        detect_air_epidermis_interface,
        estimate_orientation,
        estimate_epidermis_normal,
    )
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional skimage (lazy import — avoids import-time crashes with numpy 2.x)
# ---------------------------------------------------------------------------
_HAS_SKIMAGE = False
try:
    from skimage.measure import find_contours  # noqa: F401

    _HAS_SKIMAGE = True
except Exception:
    pass

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Interface detection
# ---------------------------------------------------------------------------


def detect_air_epidermis_interface(
    mask: np.ndarray,
    air_value: int = 0,
    epidermis_value: int = 1,
    level: float = 0.5,
    min_contour_length: int = 20,
) -> Tuple[np.ndarray, float]:
    """Trace the air↔epidermis boundary in a segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        2-D integer array of shape ``(H, W)``.
    air_value : int
        Label for air / background (default 0).
    epidermis_value : int
        Label for epidermis (default 1).
    level : float
        Contour level passed to ``find_contours`` (default 0.5 works for
        a binary mask with values 0 and 1).
    min_contour_length : int
        Minimum number of points a contour must have to be considered
        the main interface (shorter contours are discarded).

    Returns
    -------
    boundary_pts : np.ndarray
        ``(N, 2)`` array of ``(row, col)`` points along the longest
        epidermis-air interface contour.
    fraction_kept : float
        Fraction of total contour points that belong to the selected
        (longest) contour.

    Raises
    ------
    ValueError
        If no valid interface contour is found.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")

    # Lazy import of skimage (may not be available in all environments)
    if not _HAS_SKIMAGE:
        raise ImportError(
            "skimage is required for contour detection. "
            "Install it with: pip install scikit-image"
        )

    # Build a binary map where epidermis == 1, everything else == 0
    binary = np.where(mask == epidermis_value, 1, 0).astype(np.float64)

    # Find contours at the requested level
    contours = find_contours(binary, level=level)

    if not contours:
        raise ValueError(
            f"No contours found at level {level} — check that the mask "
            f"contains epidermis_value={epidermis_value} adjacent to "
            f"air_value={air_value}."
        )

    # Pick the longest contour
    contours_sorted = sorted(contours, key=lambda c: len(c), reverse=True)
    best = contours_sorted[0]

    if len(best) < min_contour_length:
        raise ValueError(
            f"Longest contour has only {len(best)} points "
            f"(min_contour_length={min_contour_length})."
        )

    total_pts = sum(len(c) for c in contours)
    fraction_kept = len(best) / total_pts if total_pts > 0 else 1.0

    # find_contours returns (row, col) order
    boundary_pts = np.asarray(best, dtype=np.float64)  # (N, 2)  [row, col]

    return boundary_pts, fraction_kept


# ---------------------------------------------------------------------------
# Orientation estimation (PCA-based)
# ---------------------------------------------------------------------------


def estimate_orientation(
    boundary_pts: np.ndarray,
) -> Tuple[float, np.ndarray, float]:
    """Estimate the tangent direction and outward normal of a boundary.

    Uses PCA on the boundary point cloud. The first eigenvector is the
    dominant (tangent) direction; the second eigenvector is the normal.
    The outward direction (air → epidermis) is resolved by probing the
    mask one step along each normal direction.

    Parameters
    ----------
    boundary_pts : np.ndarray
        ``(N, 2)`` array of ``(row, col)`` points.

    Returns
    -------
    normal_deg : float
        Angle of the outward normal in degrees
        (image-coordinate convention, range [-180, 180]).
    normal_vector : np.ndarray
        Unit vector ``(nx, ny)`` in image coordinates (x = col, y = row).
        Points from air toward epidermis.
    confidence : float
        Interface linearity confidence in [0, 1]. Based on the ratio of
        explained variance of the first PCA component. 1.0 = perfectly
        straight line, 0.0 = isotropic or highly curved.
    """
    if len(boundary_pts) < 3:
        raise ValueError(
            f"Need at least 3 boundary points, got {len(boundary_pts)}"
        )

    # Centre the points
    mean_pt = boundary_pts.mean(axis=0)  # [row, col]
    centred = boundary_pts - mean_pt

    # Covariance matrix (2x2)  —  PCA by SVD
    cov = centred.T @ centred / (len(boundary_pts) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns ascending eigenvalues; the first eigenvector
    # corresponds to the *smallest* variance = normal direction.
    # The last eigenvector is the tangent (largest variance).
    idx_sort = np.argsort(eigenvalues)[::-1]  # descending
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]  # columns are eigenvectors

    tangent = eigenvectors[:, 0]  # first PC  (in [row, col] order)
    normal_raw = eigenvectors[:, 1]  # second PC (in [row, col] order)

    # Normalise
    tangent = tangent / np.linalg.norm(tangent)
    normal_raw = normal_raw / np.linalg.norm(normal_raw)

    # Ensure the normal is perpendicular to the tangent (should be by
    # construction, but enforce sign consistency: we want the normal such
    # that turning the tangent CCW by 90° in (row, col) space gives the
    # normal.  In image (col, row) convention:
    #   tangent_col = tangent[1], tangent_row = tangent[0]
    #   normal CCW from tangent in (col, row): (-tangent_row, tangent_col)
    # But since we work in (row, col) the perpendicular is:
    #   perp_row = -tangent[1], perp_col = tangent[0]
    perp = np.array([-tangent[1], tangent[0]])
    # Align sign of normal_raw to perp
    if np.dot(normal_raw, perp) < 0:
        normal_raw = -normal_raw

    # Confidence: ratio of explained variance of the first component
    total_var = eigenvalues.sum()
    if total_var > 0:
        confidence = float(eigenvalues[0] / total_var)
    else:
        confidence = 0.0
    # Clamp to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    # The normal vector above is in (row, col) order.  Convert to
    # standard image (x, y) order: x = col, y = row.
    normal_vector = np.array([normal_raw[1], normal_raw[0]], dtype=np.float64)

    # Angle in image coordinates: atan2(ny, nx)
    normal_deg = float(np.degrees(np.arctan2(normal_vector[1], normal_vector[0])))

    return normal_deg, normal_vector, confidence


# ---------------------------------------------------------------------------
# Combined high-level routine
# ---------------------------------------------------------------------------


def estimate_epidermis_normal(
    mask: np.ndarray,
    air_value: int = 0,
    epidermis_value: int = 1,
) -> Tuple[float, np.ndarray, float, np.ndarray, float]:
    """Run full epidermis-air normal estimation from a segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        2-D integer array ``(H, W)`` with epidermis labelled
        ``epidermis_value``.
    air_value, epidermis_value : int
        Class labels.

    Returns
    -------
    normal_deg : float
        Angle of outward normal in degrees (image coords).
    normal_vector : np.ndarray
        Unit vector ``(nx, ny)`` pointing air → epidermis.
    confidence : float
        Linearity confidence in [0, 1].
    boundary_pts : np.ndarray
        ``(N, 2)`` ``(row, col)`` points of the longest interface contour.
    fraction_kept : float
        Fraction of contour points retained.
    """
    boundary_pts, fraction_kept = detect_air_epidermis_interface(
        mask,
        air_value=air_value,
        epidermis_value=epidermis_value,
    )

    normal_deg, normal_vector, confidence = estimate_orientation(boundary_pts)

    # --- outward-normal disambiguation ---
    # Probe a few pixels along ±normal to decide which direction is
    # air (class 0) and which is epidermis (class 1).
    # The boundary centroid is used as the probe origin.
    centroid_row_col = boundary_pts.mean(axis=0)  # [row, col]
    # Probe offset in (row, col) space
    # Convert normal vector (x, y) to (row, col)
    probe_norm = np.array([normal_vector[1], normal_vector[0]])  # [row, col]
    step = 5  # pixels

    def _sample_along(norm_dir, multiplier):
        """Sample mask value a few steps along norm_dir."""
        rr = int(round(centroid_row_col[0] + multiplier * step * norm_dir[0]))
        cc = int(round(centroid_row_col[1] + multiplier * step * norm_dir[1]))
        rr = np.clip(rr, 0, mask.shape[0] - 1)
        cc = np.clip(cc, 0, mask.shape[1] - 1)
        return int(mask[rr, cc])

    probe_plus = _sample_along(probe_norm, 1.0)
    probe_minus = _sample_along(probe_norm, -1.0)

    # The outward normal should point from air toward epidermis:
    # +direction is air(0) → epidermis(1), so we want probe_plus >=
    # epidermis_value.  If we got the sign wrong, flip.
    if probe_minus >= epidermis_value and probe_plus < epidermis_value:
        # Minus side is epidermis, plus side is air → flip normal
        normal_vector = -normal_vector
        normal_deg = float(np.degrees(np.arctan2(normal_vector[1], normal_vector[0])))
    elif probe_plus >= epidermis_value and probe_minus >= epidermis_value:
        # Both sides are epidermis — the centroid is inside; confidence
        # penalty but don't flip.
        confidence *= 0.5
    elif probe_plus < epidermis_value and probe_minus < epidermis_value:
        # Both sides are air — the centroid is outside; confidence
        # penalty.
        confidence *= 0.3

    return normal_deg, normal_vector, confidence, boundary_pts, fraction_kept


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the epidermis-air interface normal from a "
            "segmentation mask.  Detects the boundary between "
            "class 0 (air) and class 1 (epidermis), fits a line "
            "via PCA, and reports the outward normal angle + confidence."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Convention: outward normal points from air (0) toward "
            "epidermis (1) in image coords (x right, y down).  "
            "Angle is CCW from +x: 0°→right, 90°→down, ±180°→left, "
            "-90°→up.\n\n"
            "Example:\n"
            "  python scripts/optical_ga/estimate_epidermis_normal.py \\\n"
            "      --mask /tmp/mask.png \\\n"
            "      --output /tmp/normal.json"
        ),
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to segmentation mask image (grayscale PNG, 0=air, 1=epidermis).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path for output JSON with normal_deg, confidence, etc.",
    )
    parser.add_argument(
        "--air-value",
        type=int,
        default=0,
        help="Mask value for air/background (default 0).",
    )
    parser.add_argument(
        "--epidermis-value",
        type=int,
        default=1,
        help="Mask value for epidermis (default 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for reproducibility (default 42).",
    )
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Set seed (for any stochastic downstream steps)
    np.random.seed(args.seed)

    # Load mask
    from PIL import Image

    mask_pil = Image.open(args.mask).convert("L")
    mask = np.asarray(mask_pil, dtype=np.int32)
    # Normalise from 0-255 to 0-1 if the mask is stored as uint8
    if mask.max() > 1:
        # Assume 0=background, 255=foreground; threshold at 127.
        mask = (mask > 127).astype(np.int32)

    normal_deg, normal_vec, confidence, boundary_pts, fraction_kept = (
        estimate_epidermis_normal(
            mask,
            air_value=args.air_value,
            epidermis_value=args.epidermis_value,
        )
    )

    result = {
        "normal_deg": normal_deg,
        "normal_vector": normal_vec.tolist(),
        "confidence": confidence,
        "num_boundary_points": int(len(boundary_pts)),
        "fraction_kept": fraction_kept,
        "centroid_row": float(boundary_pts[:, 0].mean()),
        "centroid_col": float(boundary_pts[:, 1].mean()),
        "seed_used": args.seed,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Normal estimation written to {out_path}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
