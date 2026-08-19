#!/usr/bin/env python3
"""Build leakage-safe train/val/test splits for tiles_for_simulation.

The simulation tiles overlap spatially (512 px tiles with a 256 px default stride),
so splitting individual tiles would leak near-duplicate tissue into validation/test.
This script groups by ``slide_id`` from ``tiles_manifest.csv`` and assigns whole
slides to exactly one split.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tiles-root",
        default="data/artifacts/tiles_for_simulation",
        help="Root containing tiles_manifest.csv, rgb/, and label_id/.",
    )
    ap.add_argument("--manifest-csv", default="", help="Override input manifest CSV.")
    ap.add_argument(
        "--output-csv",
        default="",
        help="Output split manifest (default: <tiles-root>/segmentation_splits.csv).",
    )
    ap.add_argument(
        "--stats-json",
        default="",
        help="Output stats JSON (default: <tiles-root>/segmentation_splits_stats.json).",
    )
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-class-stats",
        action="store_true",
        help="Do not scan label_id .npy files for per-class pixel/coverage stats.",
    )
    return ap.parse_args()


def _load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Manifest is empty: {path}")
    required = {"tile_id", "slide_id", "rgb_path", "label_id_npy_path"}
    missing = required.difference(rows[0])
    if missing:
        raise SystemExit(f"Manifest missing columns: {sorted(missing)}")
    return rows


def _split_counts(n: int, train_frac: float, val_frac: float, test_frac: float) -> Dict[str, int]:
    if n < 1:
        raise ValueError("Need at least one slide")
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}")
    if min(train_frac, val_frac, test_frac) < 0:
        raise ValueError("Split fractions must be non-negative")

    raw = np.array([train_frac, val_frac, test_frac], dtype=np.float64) * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    order = np.argsort(-(raw - counts))
    for i in order[:remainder]:
        counts[i] += 1

    # With at least three slides, keep every non-zero requested split represented.
    if n >= 3:
        requested = np.array([train_frac, val_frac, test_frac]) > 0
        for idx in np.where(requested & (counts == 0))[0]:
            donors = np.where(counts > 1)[0]
            if donors.size:
                donor = int(donors[np.argmax(counts[donors])])
                counts[donor] -= 1
                counts[idx] += 1

    return {"train": int(counts[0]), "val": int(counts[1]), "test": int(counts[2])}


def assign_slide_splits(
    slide_ids: Iterable[str],
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Dict[str, str]:
    slides = sorted({str(x) for x in slide_ids if str(x)})
    counts = _split_counts(len(slides), train_frac, val_frac, test_frac)
    rng = np.random.default_rng(seed)
    perm = [slides[int(i)] for i in rng.permutation(len(slides))]

    assignment: Dict[str, str] = {}
    cursor = 0
    for split in ("train", "val", "test"):
        take = counts[split]
        for slide_id in perm[cursor : cursor + take]:
            assignment[slide_id] = split
        cursor += take
    return assignment


def _resolve_under_root(path_text: str, root: Path) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else root / p


def _collect_class_stats(rows: List[Dict[str, str]], root: Path) -> Dict[str, object]:
    pixel_counts: Dict[str, Counter] = defaultdict(Counter)
    tiles_with_class: Dict[str, Counter] = defaultdict(Counter)
    missing_labels: List[str] = []

    for row in rows:
        split = row["split"]
        label_path = _resolve_under_root(row["label_id_npy_path"], root)
        if not label_path.is_file():
            missing_labels.append(str(label_path))
            continue
        labels = np.load(label_path, mmap_mode="r")
        uniq, counts = np.unique(labels, return_counts=True)
        for cid, count in zip(uniq.tolist(), counts.tolist()):
            key = str(int(cid))
            pixel_counts[split][key] += int(count)
            tiles_with_class[split][key] += 1

    return {
        "pixel_counts_by_split": {k: dict(v) for k, v in pixel_counts.items()},
        "tiles_with_class_by_split": {k: dict(v) for k, v in tiles_with_class.items()},
        "missing_label_files": missing_labels[:100],
        "missing_label_file_count": len(missing_labels),
    }


def main() -> int:
    args = _parse_args()
    tiles_root = Path(args.tiles_root).resolve()
    manifest = Path(args.manifest_csv).resolve() if args.manifest_csv else tiles_root / "tiles_manifest.csv"
    output_csv = Path(args.output_csv).resolve() if args.output_csv else tiles_root / "segmentation_splits.csv"
    stats_json = Path(args.stats_json).resolve() if args.stats_json else tiles_root / "segmentation_splits_stats.json"

    rows = _load_rows(manifest)
    assignment = assign_slide_splits(
        (row["slide_id"] for row in rows),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )

    for row in rows:
        row["split"] = assignment[row["slide_id"]]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    if "split" not in fieldnames:
        fieldnames.append("split")
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    slides_by_split: Dict[str, set] = defaultdict(set)
    tiles_by_split: Counter = Counter()
    for row in rows:
        slides_by_split[row["split"]].add(row["slide_id"])
        tiles_by_split[row["split"]] += 1

    stats: Dict[str, object] = {
        "source_manifest": str(manifest),
        "output_csv": str(output_csv),
        "seed": int(args.seed),
        "fractions": {
            "train": float(args.train_frac),
            "val": float(args.val_frac),
            "test": float(args.test_frac),
        },
        "slide_counts": {k: len(v) for k, v in slides_by_split.items()},
        "tile_counts": dict(tiles_by_split),
        "slides": {k: sorted(v) for k, v in slides_by_split.items()},
    }
    if not args.skip_class_stats:
        stats.update(_collect_class_stats(rows, tiles_root))

    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Wrote split manifest: {output_csv}")
    print(f"Wrote split stats   : {stats_json}")
    for split in ("train", "val", "test"):
        print(
            f"{split:>5}: {len(slides_by_split.get(split, set()))} slides, "
            f"{tiles_by_split.get(split, 0)} tiles"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
