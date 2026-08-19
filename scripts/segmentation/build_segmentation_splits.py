#!/usr/bin/env python3
"""Build leakage-safe, class-aware train/val/test splits for simulation tiles.

Tiles from one source slide overlap spatially, so every slide is kept entirely in
one split. Instead of assigning slides once at random, this script evaluates many
whole-slide assignments and keeps the one that best balances:

1) rare-class coverage across splits when the available slide support permits it,
2) per-class pixel/tile distribution, and
3) total tile distribution.

No tile from a slide can ever cross a split boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles-root", default="data/artifacts/tiles_for_simulation")
    ap.add_argument("--manifest-csv", default="")
    ap.add_argument("--output-csv", default="")
    ap.add_argument("--stats-json", default="")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--search-attempts", type=int, default=5000)
    ap.add_argument("--num-classes", type=int, default=12)
    ap.add_argument("--min-class-pixels-per-slide", type=int, default=1)
    ap.add_argument("--skip-class-stats", action="store_true")
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
    if n >= 3:
        requested = np.array([train_frac, val_frac, test_frac]) > 0
        for idx in np.where(requested & (counts == 0))[0]:
            donors = np.where(counts > 1)[0]
            if donors.size:
                donor = int(donors[np.argmax(counts[donors])])
                counts[donor] -= 1
                counts[idx] += 1
    return {"train": int(counts[0]), "val": int(counts[1]), "test": int(counts[2])}


def _resolve_under_root(path_text: str, root: Path) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else root / p


def _build_slide_stats(rows, root: Path, num_classes: int) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    slides = sorted({row["slide_id"] for row in rows})
    slide_to_idx = {slide: i for i, slide in enumerate(slides)}
    pixel_counts = np.zeros((len(slides), num_classes), dtype=np.int64)
    tiles_with_class = np.zeros((len(slides), num_classes), dtype=np.int64)
    tile_counts = np.zeros(len(slides), dtype=np.int64)
    for row in rows:
        si = slide_to_idx[row["slide_id"]]
        label_path = _resolve_under_root(row["label_id_npy_path"], root)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        labels = np.load(label_path, mmap_mode="r")
        uniq, counts = np.unique(labels, return_counts=True)
        tile_counts[si] += 1
        for cid, count in zip(uniq.tolist(), counts.tolist()):
            cid = int(cid)
            if 0 <= cid < num_classes:
                pixel_counts[si, cid] += int(count)
                tiles_with_class[si, cid] += 1
    return slides, pixel_counts, tiles_with_class, tile_counts


def _assignment_from_permutation(perm, slides, counts):
    assignment = {}
    cursor = 0
    for split in ("train", "val", "test"):
        take = counts[split]
        for idx in perm[cursor:cursor + take]:
            assignment[slides[int(idx)]] = split
        cursor += take
    return assignment


def _score_assignment(assignment, slides, pixel_counts, tiles_with_class, tile_counts, desired_fracs, min_class_pixels_per_slide):
    split_names = ("train", "val", "test")
    split_idx = {name: i for i, name in enumerate(split_names)}
    slide_split = np.array([split_idx[assignment[s]] for s in slides], dtype=np.int64)
    num_classes = pixel_counts.shape[1]
    split_pixels = np.zeros((3, num_classes), dtype=np.float64)
    split_class_tiles = np.zeros((3, num_classes), dtype=np.float64)
    split_tiles = np.zeros(3, dtype=np.float64)
    for si in range(len(slides)):
        sp = slide_split[si]
        split_pixels[sp] += pixel_counts[si]
        split_class_tiles[sp] += tiles_with_class[si]
        split_tiles[sp] += tile_counts[si]
    score = 0.0
    tile_total = max(float(split_tiles.sum()), 1.0)
    score += 60.0 * float(np.abs(split_tiles / tile_total - desired_fracs).sum())
    for cid in range(1, num_classes):
        support = pixel_counts[:, cid] >= int(min_class_pixels_per_slide)
        support_n = int(support.sum())
        if support_n == 0:
            continue
        present = np.zeros(3, dtype=bool)
        for sp in range(3):
            present[sp] = bool(np.any(support & (slide_split == sp)))
        if support_n >= 3:
            score += 1500.0 * float((~present).sum())
        elif support_n == 2:
            if not present[0]:
                score += 1500.0
            if not (present[1] or present[2]):
                score += 1000.0
        else:
            if not present[0]:
                score += 1500.0
        total_px = float(split_pixels[:, cid].sum())
        if total_px > 0:
            px_frac = split_pixels[:, cid] / total_px
            rarity = 1.0 / np.sqrt(max(support_n, 1))
            score += 80.0 * rarity * float(np.abs(px_frac - desired_fracs).sum())
        total_ct = float(split_class_tiles[:, cid].sum())
        if total_ct > 0:
            ct_frac = split_class_tiles[:, cid] / total_ct
            score += 40.0 * float(np.abs(ct_frac - desired_fracs).sum())
    return float(score)


def assign_slide_splits_class_aware(slides, pixel_counts, tiles_with_class, tile_counts, *, train_frac, val_frac, test_frac, seed, search_attempts, min_class_pixels_per_slide):
    counts = _split_counts(len(slides), train_frac, val_frac, test_frac)
    desired = np.array([train_frac, val_frac, test_frac], dtype=np.float64)
    rng = np.random.default_rng(seed)
    attempts = max(1, int(search_attempts))
    best_assignment = None
    best_score = float("inf")
    for _ in range(attempts):
        perm = rng.permutation(len(slides))
        assignment = _assignment_from_permutation(perm, slides, counts)
        score = _score_assignment(assignment, slides, pixel_counts, tiles_with_class, tile_counts, desired, min_class_pixels_per_slide)
        if score < best_score:
            best_score = score
            best_assignment = assignment
    assert best_assignment is not None
    return best_assignment, best_score


def _collect_class_stats(rows, root: Path):
    pixel_counts = defaultdict(Counter)
    tiles_with_class = defaultdict(Counter)
    missing_labels = []
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
    slides, slide_pixels, slide_class_tiles, slide_tile_counts = _build_slide_stats(rows, tiles_root, int(args.num_classes))
    assignment, assignment_score = assign_slide_splits_class_aware(
        slides, slide_pixels, slide_class_tiles, slide_tile_counts,
        train_frac=float(args.train_frac), val_frac=float(args.val_frac), test_frac=float(args.test_frac),
        seed=int(args.seed), search_attempts=int(args.search_attempts),
        min_class_pixels_per_slide=int(args.min_class_pixels_per_slide),
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
    slides_by_split = defaultdict(set)
    tiles_by_split = Counter()
    for row in rows:
        slides_by_split[row["split"]].add(row["slide_id"])
        tiles_by_split[row["split"]] += 1
    support_report = {}
    for cid in range(1, int(args.num_classes)):
        supporting = slide_pixels[:, cid] >= int(args.min_class_pixels_per_slide)
        support_report[str(cid)] = {
            "supporting_slides_total": int(supporting.sum()),
            "supporting_slides_by_split": {
                split: int(sum(bool(supporting[i]) and assignment[slide] == split for i, slide in enumerate(slides)))
                for split in ("train", "val", "test")
            },
        }
    stats = {
        "source_manifest": str(manifest), "output_csv": str(output_csv), "seed": int(args.seed),
        "search_attempts": int(args.search_attempts), "assignment_score": float(assignment_score),
        "fractions": {"train": float(args.train_frac), "val": float(args.val_frac), "test": float(args.test_frac)},
        "slide_counts": {k: len(v) for k, v in slides_by_split.items()}, "tile_counts": dict(tiles_by_split),
        "slides": {k: sorted(v) for k, v in slides_by_split.items()}, "class_slide_support": support_report,
    }
    if not args.skip_class_stats:
        stats.update(_collect_class_stats(rows, tiles_root))
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote split manifest: {output_csv}")
    print(f"Wrote split stats   : {stats_json}")
    print(f"Class-aware assignment score: {assignment_score:.3f}")
    for split in ("train", "val", "test"):
        print(f"{split:>5}: {len(slides_by_split.get(split, set()))} slides, {tiles_by_split.get(split, 0)} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
