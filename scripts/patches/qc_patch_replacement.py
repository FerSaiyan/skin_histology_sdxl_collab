#!/usr/bin/env python
"""
QC and rejection for merged slice edits.

This script:
- Computes quality metrics for edited slices
- Compares original vs edited regions
- Checks non-ROI drift
- Emits pass/fail flag and summary report

Metrics computed:
- mask_coverage: fraction of mask that was actually edited
- non_roi_drift: mean absolute difference in non-ROI regions
- seam_quality: edge difference at ROI boundary
- intensity_preservation: histogram similarity in non-ROI regions
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


def _compute_mask_coverage(
    original: np.ndarray,
    edited: np.ndarray,
    bbox: Tuple[int, int, int, int],
    threshold: float = 10.0,
) -> float:
    """Compute fraction of ROI region that was meaningfully edited."""
    y_min, x_min, y_max, x_max = bbox
    orig_roi = original[y_min:y_max, x_min:x_max]
    edit_roi = edited[y_min:y_max, x_min:x_max]

    diff = np.abs(orig_roi.astype(float) - edit_roi.astype(float))
    if len(diff.shape) == 3:
        diff = diff.mean(axis=2)

    edited_pixels = (diff > threshold).sum()
    total_pixels = diff.size

    return float(edited_pixels / total_pixels) if total_pixels > 0 else 0.0


def _compute_non_roi_drift(
    original: np.ndarray,
    edited: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> float:
    """Compute mean absolute difference in non-ROI regions."""
    mask = np.ones(original.shape[:2], dtype=bool)
    y_min, x_min, y_max, x_max = bbox
    mask[y_min:y_max, x_min:x_max] = False

    if len(original.shape) == 3:
        orig_non_roi = original[mask].reshape(-1, 3)
        edit_non_roi = edited[mask].reshape(-1, 3)
    else:
        orig_non_roi = original[mask]
        edit_non_roi = edited[mask]

    drift = np.abs(orig_non_roi.astype(float) - edit_non_roi.astype(float)).mean()

    return float(drift)


def _compute_seam_quality(
    original: np.ndarray,
    edited: np.ndarray,
    bbox: Tuple[int, int, int, int],
    boundary_width: int = 8,
) -> float:
    """Compute edge quality at ROI boundary (lower is better)."""
    y_min, x_min, y_max, x_max = bbox

    edges = []

    for y in range(max(0, y_min - boundary_width), min(original.shape[0], y_min + boundary_width)):
        if 0 <= x_min < original.shape[1]:
            diff = np.abs(original[y, x_min].astype(float) - edited[y, x_min].astype(float))
            edges.append(diff.mean() if len(diff.shape) > 0 else diff)

    for y in range(max(0, y_max - boundary_width), min(original.shape[0], y_max + boundary_width)):
        if 0 <= x_max - 1 < original.shape[1]:
            diff = np.abs(original[y, x_max - 1].astype(float) - edited[y, x_max - 1].astype(float))
            edges.append(diff.mean() if len(diff.shape) > 0 else diff)

    return float(np.mean(edges)) if edges else 0.0


def _compute_histogram_similarity(
    original: np.ndarray,
    edited: np.ndarray,
    bbox: Tuple[int, int, int, int],
    bins: int = 64,
) -> float:
    """Compute histogram correlation in non-ROI regions."""
    mask = np.ones(original.shape[:2], dtype=bool)
    y_min, x_min, y_max, x_max = bbox
    mask[y_min:y_max, x_min:x_max] = False

    if len(original.shape) == 3:
        orig_non_roi = original[mask].reshape(-1, 3)
        edit_non_roi = edited[mask].reshape(-1, 3)

        correlations = []
        for c in range(3):
            orig_hist, _ = np.histogram(orig_non_roi[:, c], bins=bins, range=(0, 255))
            edit_hist, _ = np.histogram(edit_non_roi[:, c], bins=bins, range=(0, 255))

            orig_hist = orig_hist.astype(float)
            edit_hist = edit_hist.astype(float)

            if orig_hist.sum() > 0 and edit_hist.sum() > 0:
                orig_hist /= orig_hist.sum()
                edit_hist /= edit_hist.sum()
                corr = np.corrcoef(orig_hist, edit_hist)[0, 1]
                correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 1.0
    else:
        orig_hist, _ = np.histogram(original[mask], bins=bins, range=(0, 255))
        edit_hist, _ = np.histogram(edited[mask], bins=bins, range=(0, 255))

        orig_hist = orig_hist.astype(float)
        edit_hist = edit_hist.astype(float)

        if orig_hist.sum() > 0 and edit_hist.sum() > 0:
            orig_hist /= orig_hist.sum()
            edit_hist /= edit_hist.sum()
            return float(np.corrcoef(orig_hist, edit_hist)[0, 1])
        return 1.0


def _compute_all_metrics(
    original: np.ndarray,
    edited: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Dict[str, float]:
    """Compute all QC metrics."""
    return {
        "mask_coverage": _compute_mask_coverage(original, edited, bbox),
        "non_roi_drift": _compute_non_roi_drift(original, edited, bbox),
        "seam_quality": _compute_seam_quality(original, edited, bbox),
        "histogram_similarity": _compute_histogram_similarity(original, edited, bbox),
    }


def _apply_rejection_rules(
    metrics: Dict[str, float],
    thresholds: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Apply rejection rules based on thresholds."""
    passed = True
    reasons: List[str] = []

    if metrics["mask_coverage"] < thresholds.get("min_mask_coverage", 0.01):
        passed = False
        reasons.append(f"mask_coverage {metrics['mask_coverage']:.4f} < {thresholds['min_mask_coverage']}")

    if metrics["non_roi_drift"] > thresholds.get("max_non_roi_drift", 5.0):
        passed = False
        reasons.append(f"non_roi_drift {metrics['non_roi_drift']:.2f} > {thresholds['max_non_roi_drift']}")

    if metrics["seam_quality"] > thresholds.get("max_seam_quality", 20.0):
        passed = False
        reasons.append(f"seam_quality {metrics['seam_quality']:.2f} > {thresholds['max_seam_quality']}")

    if metrics["histogram_similarity"] < thresholds.get("min_histogram_similarity", 0.9):
        passed = False
        reasons.append(f"histogram_similarity {metrics['histogram_similarity']:.4f} < {thresholds['min_histogram_similarity']}")

    return passed, reasons


def main() -> None:
    ap = argparse.ArgumentParser(description="QC and rejection for merged slice edits.")
    ap.add_argument("--merge-metadata", required=True, help="Path to merge_metadata.csv")
    ap.add_argument("--output-dir", required=True, help="Output directory for QC reports")
    ap.add_argument("--min-mask-coverage", type=float, default=0.01)
    ap.add_argument("--max-non-roi-drift", type=float, default=5.0)
    ap.add_argument("--max-seam-quality", type=float, default=20.0)
    ap.add_argument("--min-histogram-similarity", type=float, default=0.9)
    ap.add_argument("--max-slices", type=int, default=0, help="Limit slices (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = ap.parse_args()

    merge_meta = pd.read_csv(args.merge_metadata)

    if args.max_slices > 0:
        merge_meta = merge_meta.head(args.max_slices)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    thresholds = {
        "min_mask_coverage": args.min_mask_coverage,
        "max_non_roi_drift": args.max_non_roi_drift,
        "max_seam_quality": args.max_seam_quality,
        "min_histogram_similarity": args.min_histogram_similarity,
    }

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Slices to QC: {len(merge_meta)}")
        print(f"Thresholds: {thresholds}")
        for idx, row in merge_meta.iterrows():
            print(f"  [{idx}] {row['slice_id']}")
        return

    qc_metadata: List[Dict[str, Any]] = []
    passed_count = 0
    failed_count = 0

    for idx, row in merge_meta.iterrows():
        slice_id = row["slice_id"]

        source_path = Path(row["source_image"])
        merged_path = Path(row["merged_image"])

        if not source_path.exists() or not merged_path.exists():
            print(f"[WARN] Missing files for {slice_id}")
            continue

        print(f"[{idx+1}/{len(merge_meta)}] QC for {slice_id}")

        original = np.array(Image.open(source_path).convert("RGB"))
        edited = np.array(Image.open(merged_path).convert("RGB"))

        bbox = (
            int(row["bbox_y_min"]),
            int(row["bbox_x_min"]),
            int(row["bbox_y_max"]),
            int(row["bbox_x_max"]),
        )

        metrics = _compute_all_metrics(original, edited, bbox)
        passed, reasons = _apply_rejection_rules(metrics, thresholds)

        if passed:
            passed_count += 1
        else:
            failed_count += 1

        qc_meta = {
            "slice_id": slice_id,
            "source_image": str(source_path),
            "merged_image": str(merged_path),
            "passed": passed,
            "rejection_reasons": ";".join(reasons) if reasons else "",
            "mask_coverage": metrics["mask_coverage"],
            "non_roi_drift": metrics["non_roi_drift"],
            "seam_quality": metrics["seam_quality"],
            "histogram_similarity": metrics["histogram_similarity"],
            "qc_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        qc_metadata.append(qc_meta)

    if qc_metadata:
        qc_df = pd.DataFrame(qc_metadata)
        qc_meta_path = output_dir / "metadata" / "qc_metadata.csv"
        qc_df.to_csv(qc_meta_path, index=False)
        print(f"Saved QC metadata: {qc_meta_path}")

        summary = {
            "total_slices_checked": len(qc_metadata),
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": passed_count / len(qc_metadata) if qc_metadata else 0.0,
            "thresholds": thresholds,
        }
        summary_path = output_dir / "reports" / "qc_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved QC summary: {summary_path}")

        print(f"\nQC Summary: {passed_count} passed, {failed_count} failed")


if __name__ == "__main__":
    main()
