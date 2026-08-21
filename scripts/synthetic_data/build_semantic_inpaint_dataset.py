#!/usr/bin/env python3
"""Build a class-aligned SDXL inpainting dataset from simulation tiles.

Each training example is (RGB tile, semantic target class, geometry-aware mask,
class-specific caption). Masks are sampled from the aligned Histo-Seg label map
instead of being placed randomly on arbitrary tissue.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from semantic_mask_utils import (  # noqa: E402
    CLASS_NAME_BY_ID,
    PROMPT_NAME_BY_ID,
    TARGET_CLASS_IDS,
    SemanticMaskConfig,
    adjacency_counts,
    class_prompt,
    sample_semantic_mask,
    valid_target_classes,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles-root", default="data/artifacts/tiles_for_simulation")
    ap.add_argument("--manifest-csv", default="", help="Default: <tiles-root>/tiles_manifest.csv")
    ap.add_argument(
        "--splits-csv",
        default="auto",
        help=(
            "CSV containing tile_id,split. 'auto' tries segmentation_splits_v2.csv then "
            "segmentation_splits.csv; if absent, a deterministic slide split is created."
        ),
    )
    ap.add_argument("--output-dir", default="data/artifacts/semantic_inpaint_v1")
    ap.add_argument("--seed", type=int, default=222)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)

    ap.add_argument("--boundary-fraction", type=float, default=0.25)
    ap.add_argument("--base-masks-per-pair", type=int, default=1)
    ap.add_argument("--max-masks-per-pair", type=int, default=4)
    ap.add_argument("--eval-masks-per-pair", type=int, default=1)
    ap.add_argument(
        "--rare-balance-power",
        type=float,
        default=0.5,
        help="Training mask multiplier scales as (median_support/class_support)^power.",
    )
    ap.add_argument("--max-mask-iou", type=float, default=0.80)

    ap.add_argument("--min-component-pixels", type=int, default=256)
    ap.add_argument("--min-mask-pixels", type=int, default=256)
    ap.add_argument("--max-mask-area-frac", type=float, default=0.16)
    ap.add_argument("--interior-min-purity", type=float, default=0.90)
    ap.add_argument("--boundary-min-purity", type=float, default=0.60)
    ap.add_argument("--boundary-max-purity", type=float, default=0.92)
    ap.add_argument("--min-minor-radius", type=float, default=3.0)
    ap.add_argument("--max-major-radius", type=float, default=72.0)
    ap.add_argument("--mask-attempts", type=int, default=80)
    ap.add_argument("--context-ring-px", type=int, default=10)

    ap.add_argument(
        "--link-mode",
        choices=["auto", "symlink", "hardlink", "copy"],
        default="auto",
        help="How to materialize repeated RGB sources under the class-conditioned dataset.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-tiles", type=int, default=0, help="Development limit; 0 means all tiles")
    return ap.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_under(root: Path, raw: str) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else root / p


def _auto_splits_path(tiles_root: Path) -> Path | None:
    for name in ("segmentation_splits_v2.csv", "segmentation_splits.csv"):
        p = tiles_root / name
        if p.is_file():
            return p
    return None


def _derive_slide_splits(rows: Sequence[Dict[str, str]], seed: int, train_frac: float, val_frac: float) -> Dict[str, str]:
    slides = sorted({str(r["slide_id"]) for r in rows})
    rng = random.Random(int(seed))
    rng.shuffle(slides)
    n = len(slides)
    if n < 3:
        raise SystemExit("Need at least 3 independent slides to derive train/val/test splits.")
    n_train = max(1, int(round(n * float(train_frac))))
    n_val = max(1, int(round(n * float(val_frac))))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    split_by_slide: Dict[str, str] = {}
    for i, slide in enumerate(slides):
        if i < n_train:
            split_by_slide[slide] = "train"
        elif i < n_train + n_val:
            split_by_slide[slide] = "val"
        else:
            split_by_slide[slide] = "test"
    return {str(r["tile_id"]): split_by_slide[str(r["slide_id"])] for r in rows}


def _load_split_map(
    tiles_root: Path,
    rows: Sequence[Dict[str, str]],
    splits_arg: str,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> tuple[Dict[str, str], str]:
    split_path: Path | None
    if str(splits_arg).strip().lower() == "auto":
        split_path = _auto_splits_path(tiles_root)
    elif str(splits_arg).strip():
        split_path = Path(splits_arg).resolve()
    else:
        split_path = None

    if split_path is not None and split_path.is_file():
        data = _read_csv(split_path)
        if not data or "tile_id" not in data[0] or "split" not in data[0]:
            raise SystemExit(f"Split CSV must contain tile_id and split columns: {split_path}")
        out = {str(r["tile_id"]): str(r["split"]).strip().lower() for r in data}
        missing = [str(r["tile_id"]) for r in rows if str(r["tile_id"]) not in out]
        if missing:
            raise SystemExit(f"Split CSV is missing {len(missing)} tiles; first: {missing[:3]}")
        return out, str(split_path)

    return _derive_slide_splits(rows, seed, train_frac, val_frac), "derived_slide_split"


def _assert_slide_isolation(rows: Sequence[Dict[str, str]], split_map: Dict[str, str]) -> None:
    seen: Dict[str, set[str]] = defaultdict(set)
    for r in rows:
        seen[str(r["slide_id"])].add(split_map[str(r["tile_id"])])
    leaking = {slide: sorted(splits) for slide, splits in seen.items() if len(splits) > 1}
    if leaking:
        preview = list(leaking.items())[:5]
        raise SystemExit(f"Slide leakage detected in split assignment: {preview}")


def _link_image(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    modes = [mode] if mode != "auto" else ["symlink", "hardlink", "copy"]
    last_error: Exception | None = None
    for candidate in modes:
        try:
            if candidate == "symlink":
                rel = os.path.relpath(src.resolve(), dst.parent.resolve())
                dst.symlink_to(rel)
            elif candidate == "hardlink":
                os.link(src, dst)
            elif candidate == "copy":
                shutil.copy2(src, dst)
            else:
                raise ValueError(candidate)
            return candidate
        except Exception as exc:  # pragma: no cover - filesystem dependent
            last_error = exc
            if dst.exists() or dst.is_symlink():
                dst.unlink()
    raise RuntimeError(f"Could not materialize {src} -> {dst}: {last_error}")


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    return float(inter / max(union, 1))


def _desired_masks_for_class(
    class_id: int,
    split: str,
    train_support: Dict[int, int],
    reference_support: float,
    base_masks: int,
    max_masks: int,
    eval_masks: int,
    balance_power: float,
) -> int:
    if split != "train":
        return max(1, int(eval_masks))
    support = max(1, int(train_support.get(class_id, 0)))
    ratio = max(1.0, float(reference_support) / float(support))
    boost = int(math.ceil(ratio ** max(0.0, float(balance_power))))
    return max(1, min(int(max_masks), int(base_masks) * boost))


def _json_hist(hist: Dict[int, int]) -> str:
    return json.dumps({str(k): int(v) for k, v in sorted(hist.items())}, separators=(",", ":"))


def main() -> int:
    args = _parse_args()
    if not (0.0 <= args.boundary_fraction <= 1.0):
        raise SystemExit("--boundary-fraction must be in [0,1]")
    if not (0.0 < args.train_frac < 1.0 and 0.0 < args.val_frac < 1.0):
        raise SystemExit("train/val fractions must be in (0,1)")

    tiles_root = Path(args.tiles_root).resolve()
    manifest_csv = Path(args.manifest_csv).resolve() if args.manifest_csv else tiles_root / "tiles_manifest.csv"
    if not manifest_csv.is_file():
        raise SystemExit(f"Tile manifest not found: {manifest_csv}")
    rows = _read_csv(manifest_csv)
    required = {"tile_id", "slide_id", "rgb_path", "label_id_npy_path"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"Tile manifest missing required columns {sorted(required)}")
    if args.max_tiles > 0:
        rows = rows[: int(args.max_tiles)]

    split_map, split_source = _load_split_map(
        tiles_root, rows, args.splits_csv, args.seed, args.train_frac, args.val_frac
    )
    _assert_slide_isolation(rows, split_map)

    cfg = SemanticMaskConfig(
        min_component_pixels=int(args.min_component_pixels),
        min_mask_pixels=int(args.min_mask_pixels),
        max_mask_area_frac=float(args.max_mask_area_frac),
        interior_min_purity=float(args.interior_min_purity),
        boundary_min_purity=float(args.boundary_min_purity),
        boundary_max_purity=float(args.boundary_max_purity),
        min_minor_radius=float(args.min_minor_radius),
        max_major_radius=float(args.max_major_radius),
        max_attempts=int(args.mask_attempts),
        ring_px=int(args.context_ring_px),
    )

    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    eligible: Dict[str, tuple[int, ...]] = {}
    support_by_split: Dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}
    slides_by_split_class: Dict[str, Dict[int, set[str]]] = {
        s: defaultdict(set) for s in ("train", "val", "test")
    }
    tiles_containing_by_split: Dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}
    train_adjacency = np.zeros((12, 12), dtype=np.int64)

    for r in tqdm(rows, desc="Inspect semantic support", unit="tile"):
        tile_id = str(r["tile_id"])
        split = split_map[tile_id]
        label_path = _resolve_under(tiles_root, str(r["label_id_npy_path"]))
        labels = np.load(label_path)
        if labels.shape != (512, 512):
            raise SystemExit(f"Expected 512x512 label tile: {label_path}, got {labels.shape}")
        present = [int(x) for x in np.unique(labels) if int(x) in TARGET_CLASS_IDS]
        for cid in present:
            tiles_containing_by_split[split][cid] += 1
        valid = valid_target_classes(labels, cfg)
        eligible[tile_id] = valid
        for cid in valid:
            support_by_split[split][cid] += 1
            slides_by_split_class[split][cid].add(str(r["slide_id"]))
        if split == "train":
            train_adjacency += adjacency_counts(labels, num_classes=12)

    nonzero_train_support = [support_by_split["train"][cid] for cid in TARGET_CLASS_IDS if support_by_split["train"][cid] > 0]
    reference_support = float(np.median(nonzero_train_support)) if nonzero_train_support else 1.0

    manifest_records: List[Dict[str, object]] = []
    generated_counts: Dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}
    link_modes_used = Counter()
    skipped_sampling = Counter()

    for row_index, r in enumerate(tqdm(rows, desc="Build semantic masks", unit="tile")):
        tile_id = str(r["tile_id"])
        slide_id = str(r["slide_id"])
        split = split_map[tile_id]
        rgb_path = _resolve_under(tiles_root, str(r["rgb_path"]))
        label_path = _resolve_under(tiles_root, str(r["label_id_npy_path"]))
        if not rgb_path.is_file() or not label_path.is_file():
            raise SystemExit(f"Missing tile artifacts for {tile_id}: {rgb_path}, {label_path}")
        labels = np.load(label_path)

        for cid in eligible[tile_id]:
            desired = _desired_masks_for_class(
                cid,
                split,
                support_by_split["train"],
                reference_support,
                args.base_masks_per_pair,
                args.max_masks_per_pair,
                args.eval_masks_per_pair,
                args.rare_balance_power,
            )
            accepted_masks: List[np.ndarray] = []
            sample_attempt_index = 0
            while len(accepted_masks) < desired and sample_attempt_index < desired * 8:
                seed = int(args.seed) + row_index * 1_000_003 + int(cid) * 10_007 + sample_attempt_index * 101
                rng = np.random.default_rng(seed)
                wanted_mode = "boundary" if rng.random() < float(args.boundary_fraction) else "interior"
                spec = sample_semantic_mask(labels, cid, rng, wanted_mode, cfg)
                if spec is None and wanted_mode == "boundary":
                    spec = sample_semantic_mask(labels, cid, rng, "interior", cfg)
                sample_attempt_index += 1
                if spec is None:
                    continue
                mask_bool = np.asarray(spec.pop("mask"), dtype=bool)
                if any(_mask_iou(mask_bool, prev) > float(args.max_mask_iou) for prev in accepted_masks):
                    continue
                accepted_masks.append(mask_bool)

                sample_no = len(accepted_masks) - 1
                class_slug = CLASS_NAME_BY_ID[cid]
                sample_id = f"{tile_id}__c{cid:02d}_{class_slug}__m{sample_no:02d}"
                split_root = output_root / split
                image_out = split_root / "images" / f"{sample_id}.png"
                mask_out = split_root / "masks" / f"{sample_id}.png"
                caption_out = image_out.with_suffix(".caption")

                used_mode = _link_image(rgb_path, image_out, args.link_mode)
                link_modes_used[used_mode] += 1
                Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L").save(mask_out)
                prompt = class_prompt(cid)
                caption_out.write_text(prompt, encoding="utf-8")

                generated_counts[split][cid] += 1
                manifest_records.append(
                    {
                        "sample_id": sample_id,
                        "tile_id": tile_id,
                        "slide_id": slide_id,
                        "split": split,
                        "source_rgb_path": str(rgb_path),
                        "source_label_path": str(label_path),
                        "image_path": str(image_out),
                        "mask_path": str(mask_out),
                        "caption_path": str(caption_out),
                        "target_class_id": int(cid),
                        "target_class_name": CLASS_NAME_BY_ID[cid],
                        "target_prompt_name": PROMPT_NAME_BY_ID[cid],
                        "prompt": prompt,
                        "mask_mode": str(spec["mode"]),
                        "component_id": int(spec["component_id"]),
                        "component_pixels": int(spec["component_pixels"]),
                        "mask_pixels": int(spec["mask_pixels"]),
                        "mask_area_fraction": float(spec["mask_area_fraction"]),
                        "target_pixels_in_tile": int(spec["target_pixels_in_tile"]),
                        "target_pixels_in_mask": int(spec["target_pixels_in_mask"]),
                        "target_purity": float(spec["target_purity"]),
                        "target_class_coverage": float(spec["target_class_coverage"]),
                        "center_x": float(spec["center_x"]),
                        "center_y": float(spec["center_y"]),
                        "major_radius": float(spec["major_radius"]),
                        "minor_radius": float(spec["minor_radius"]),
                        "angle_deg": float(spec["angle_deg"]),
                        "neighbor_class_histogram": _json_hist(spec["neighbor_histogram"]),
                        "sampling_seed": int(seed),
                        "link_mode": used_mode,
                    }
                )

            if len(accepted_masks) < desired:
                skipped_sampling[(split, cid)] += desired - len(accepted_masks)

    manifest_out = output_root / "semantic_inpaint_manifest.csv"
    if manifest_records:
        with manifest_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_records[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_records)
    else:
        raise SystemExit("No valid semantic inpainting samples were generated.")

    adjacency_out = output_root / "train_class_adjacency.json"
    adjacency_payload = {
        "class_names": {str(k): v for k, v in CLASS_NAME_BY_ID.items()},
        "counts": train_adjacency.tolist(),
        "row_probabilities": [],
    }
    for cid in range(12):
        row_counts = train_adjacency[cid].astype(np.float64)
        total = float(row_counts.sum())
        adjacency_payload["row_probabilities"].append((row_counts / total).tolist() if total > 0 else [0.0] * 12)
    adjacency_out.write_text(json.dumps(adjacency_payload, indent=2), encoding="utf-8")

    stats = {
        "tiles_root": str(tiles_root),
        "tiles_manifest": str(manifest_csv),
        "split_source": split_source,
        "output_dir": str(output_root),
        "num_source_tiles": len(rows),
        "num_samples": len(manifest_records),
        "reference_train_eligible_pairs_median": reference_support,
        "config": vars(args),
        "mask_config": cfg.__dict__,
        "link_modes_used": dict(link_modes_used),
        "per_split": {},
        "sampling_shortfall": {
            f"{split}:class_{cid}": int(count)
            for (split, cid), count in sorted(skipped_sampling.items())
            if count > 0
        },
    }
    for split in ("train", "val", "test"):
        stats["per_split"][split] = {
            "tiles": int(sum(1 for r in rows if split_map[str(r["tile_id"])] == split)),
            "classes": {
                str(cid): {
                    "class_name": CLASS_NAME_BY_ID[cid],
                    "tiles_containing_class": int(tiles_containing_by_split[split][cid]),
                    "eligible_tile_class_pairs": int(support_by_split[split][cid]),
                    "unique_slides_eligible": int(len(slides_by_split_class[split][cid])),
                    "masks_generated": int(generated_counts[split][cid]),
                    "train_masks_per_pair": (
                        _desired_masks_for_class(
                            cid,
                            "train",
                            support_by_split["train"],
                            reference_support,
                            args.base_masks_per_pair,
                            args.max_masks_per_pair,
                            args.eval_masks_per_pair,
                            args.rare_balance_power,
                        )
                        if split == "train" and support_by_split["train"][cid] > 0
                        else None
                    ),
                }
                for cid in TARGET_CLASS_IDS
            },
        }
    stats_out = output_root / "semantic_inpaint_stats.json"
    stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Semantic inpaint samples: {len(manifest_records)}")
    print(f"Manifest               : {manifest_out}")
    print(f"Statistics             : {stats_out}")
    print(f"Train adjacency profile: {adjacency_out}")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: {sum(generated_counts[split].values())} masks")
        for cid in TARGET_CLASS_IDS:
            if generated_counts[split][cid]:
                print(
                    f"    {cid:2d} {CLASS_NAME_BY_ID[cid]:28s} "
                    f"pairs={support_by_split[split][cid]:4d} masks={generated_counts[split][cid]:4d} "
                    f"slides={len(slides_by_split_class[split][cid]):3d}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
