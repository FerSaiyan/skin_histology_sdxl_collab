#!/usr/bin/env python
"""
Build a 512x512 tile index from large images using mask coverage filtering.

The output is a CSV with tile coordinates only (no tile image files), so storage
stays low. Tiles can be materialized on demand by the training script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


Image.MAX_IMAGE_PIXELS = None


def _positions(limit: int, size: int, stride: int) -> List[int]:
    if limit < size:
        return []
    pos = list(range(0, limit - size + 1, stride))
    last = limit - size
    if not pos:
        pos = [0]
    elif pos[-1] != last:
        pos.append(last)
    return pos


def _select_tiles(
    candidates: List[Tuple[int, int, float]],
    max_tiles: int,
    mode: str,
    rng: np.random.Generator,
) -> List[Tuple[int, int, float]]:
    if max_tiles <= 0 or len(candidates) <= max_tiles:
        return candidates
    if mode == "top_coverage":
        ranked = sorted(candidates, key=lambda t: t[2], reverse=True)
        return ranked[:max_tiles]
    idx = rng.choice(len(candidates), size=max_tiles, replace=False)
    return [candidates[int(i)] for i in idx]


def _process_image_row(
    row: Dict,
    row_idx: int,
    *,
    tile_size: int,
    stride: int,
    min_mask_frac: float,
    max_mask_frac: float,
    max_tiles_per_image: int,
    selection_mode: str,
    seed: int,
    max_white_frac: float,
    white_threshold: int,
) -> Dict:
    img_path = Path(str(row["image_path"]))
    mask_path = Path(str(row["mask_path"]))
    if not img_path.is_file() or not mask_path.is_file():
        return {"rows": [], "skipped_small": 0}

    with Image.open(img_path) as im:
        w, h = im.size
    if w < tile_size or h < tile_size:
        return {"rows": [], "skipped_small": 1}

    xs = _positions(w, tile_size, stride)
    ys = _positions(h, tile_size, stride)

    candidates: List[Tuple[int, int, float]] = []
    with Image.open(mask_path) as m:
        mask_l = m.convert("L")
        mw, mh = mask_l.size
        if mw != w or mh != h:
            return {"rows": [], "skipped_small": 0}
        for y in ys:
            for x in xs:
                tile_mask = np.asarray(mask_l.crop((x, y, x + tile_size, y + tile_size)), dtype=np.uint8)
                frac = float((tile_mask > 0).mean())
                if min_mask_frac <= frac <= max_mask_frac:
                    candidates.append((x, y, frac))

    local_rng = np.random.default_rng(int(seed) + int(row_idx) * 10007)
    selected = _select_tiles(
        candidates,
        max_tiles=max_tiles_per_image,
        mode=selection_mode,
        rng=local_rng,
    )

    selected_before_white = len(selected)
    dropped_by_white = 0

    if max_white_frac < 1.0 and selected:
        filtered: List[Tuple[int, int, float]] = []
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            for x, y, frac in selected:
                tile = np.asarray(rgb.crop((x, y, x + tile_size, y + tile_size)), dtype=np.uint8)
                white_frac = float(
                    np.logical_and.reduce(
                        [tile[:, :, 0] >= white_threshold, tile[:, :, 1] >= white_threshold, tile[:, :, 2] >= white_threshold]
                    ).mean()
                )
                if white_frac <= max_white_frac:
                    filtered.append((x, y, frac, white_frac))
                else:
                    dropped_by_white += 1
        selected_rows = filtered
    else:
        selected_rows = [(x, y, frac, None) for x, y, frac in selected]

    stem = str(Path(row["filename"]).stem)
    out_rows: List[Dict] = []
    for x, y, frac, white_frac in selected_rows:
        fn = f"{stem}__x{x:05d}_y{y:05d}_s{tile_size}.png"
        out_rows.append(
            {
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "filename": fn,
                "slice_id": str(row.get("slice_id", stem)),
                "volume_id": str(row.get("volume_id", "")),
                "group_code": str(row.get("group_code", "")),
                "coarse_label": str(row.get("coarse_label", "")),
                "tile_x": int(x),
                "tile_y": int(y),
                "tile_size": int(tile_size),
                "tile_w": int(tile_size),
                "tile_h": int(tile_size),
                "tile_mask_coverage": float(frac),
                "tile_white_frac": float(white_frac) if white_frac is not None else None,
            }
        )

    return {
        "rows": out_rows,
        "skipped_small": 0,
        "selected_before_white": int(selected_before_white),
        "dropped_by_white": int(dropped_by_white),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tile index CSV from image/mask pairs.")
    ap.add_argument("--pairs-csv", required=True, help="Input CSV with image_path, mask_path, labels.")
    ap.add_argument("--output-csv", required=True, help="Output tile index CSV.")
    ap.add_argument("--stats-json", default=None, help="Optional stats JSON path.")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--min-mask-frac", type=float, default=0.15)
    ap.add_argument("--max-mask-frac", type=float, default=0.95)
    ap.add_argument("--max-tiles-per-image", type=int, default=250)
    ap.add_argument("--max-total-tiles", type=int, default=0)
    ap.add_argument(
        "--selection-mode",
        choices=["top_coverage", "random"],
        default="random",
        help="How to subsample when too many candidates are found.",
    )
    ap.add_argument("--seed", type=int, default=222)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Thread workers for per-image tiling (0 = auto).",
    )
    ap.add_argument(
        "--max-white-frac",
        type=float,
        default=1.0,
        help="Drop selected tiles with white fraction above this threshold. Applied after per-image cap.",
    )
    ap.add_argument(
        "--white-threshold",
        type=int,
        default=235,
        help="RGB threshold for counting near-white pixels (all channels >= threshold).",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.pairs_csv)
    for col in ["image_path", "mask_path", "filename", "slice_id", "coarse_label"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column in pairs CSV: {col}")

    tile_size = int(args.tile_size)
    stride = int(args.stride)
    workers = int(args.workers)
    if workers <= 0:
        workers = min(16, max(1, (os.cpu_count() or 1)))

    out_rows: List[Dict] = []
    skipped_small = 0
    selected_before_white_total = 0
    dropped_by_white_total = 0

    records = df.to_dict(orient="records")
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, row in enumerate(records):
            futures.append(
                ex.submit(
                    _process_image_row,
                    row,
                    idx,
                    tile_size=tile_size,
                    stride=stride,
                    min_mask_frac=float(args.min_mask_frac),
                    max_mask_frac=float(args.max_mask_frac),
                    max_tiles_per_image=int(args.max_tiles_per_image),
                    selection_mode=str(args.selection_mode),
                    seed=int(args.seed),
                    max_white_frac=float(args.max_white_frac),
                    white_threshold=int(args.white_threshold),
                )
            )

        processed = 0
        total = len(futures)
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            out_rows.extend(res.get("rows", []))
            skipped_small += int(res.get("skipped_small", 0))
            selected_before_white_total += int(res.get("selected_before_white", 0))
            dropped_by_white_total += int(res.get("dropped_by_white", 0))
            processed += 1
            if processed % 5 == 0 or processed == total:
                print(f"[tiles] processed {processed}/{total} images (workers={workers})")

    if not out_rows:
        raise SystemExit("No tiles selected. Relax min/max-mask-frac or stride.")

    out_df = pd.DataFrame(out_rows).sort_values(["image_path", "tile_y", "tile_x", "filename"]).reset_index(drop=True)
    if args.max_total_tiles > 0 and len(out_df) > int(args.max_total_tiles):
        out_df = out_df.sample(n=int(args.max_total_tiles), random_state=int(args.seed)).reset_index(drop=True)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"Wrote tile index: {out_csv}")
    print(f"Tiles: {len(out_df)}")
    if selected_before_white_total > 0:
        print(
            "White filter: "
            f"selected_before_white={selected_before_white_total}, "
            f"dropped={dropped_by_white_total}, "
            f"kept={selected_before_white_total - dropped_by_white_total}"
        )

    if args.stats_json:
        by_label = out_df["coarse_label"].value_counts().to_dict()
        stats = {
            "pairs_csv": str(Path(args.pairs_csv).resolve()),
            "output_csv": str(out_csv.resolve()),
            "rows": int(len(out_df)),
            "unique_source_images": int(out_df["image_path"].nunique()),
            "tile_size": int(tile_size),
            "stride": int(stride),
            "min_mask_frac": float(args.min_mask_frac),
            "max_mask_frac": float(args.max_mask_frac),
            "max_tiles_per_image": int(args.max_tiles_per_image),
            "max_total_tiles": int(args.max_total_tiles),
            "selection_mode": str(args.selection_mode),
            "max_white_frac": float(args.max_white_frac),
            "white_threshold": int(args.white_threshold),
            "coverage_mean": float(out_df["tile_mask_coverage"].mean()),
            "coverage_median": float(out_df["tile_mask_coverage"].median()),
            "coverage_p25": float(out_df["tile_mask_coverage"].quantile(0.25)),
            "coverage_p75": float(out_df["tile_mask_coverage"].quantile(0.75)),
            "white_frac_mean": float(out_df["tile_white_frac"].dropna().mean()) if "tile_white_frac" in out_df.columns else None,
            "white_frac_median": float(out_df["tile_white_frac"].dropna().median()) if "tile_white_frac" in out_df.columns else None,
            "selected_before_white_total": int(selected_before_white_total),
            "dropped_by_white_total": int(dropped_by_white_total),
            "count_by_label": by_label,
            "skipped_small_images": int(skipped_small),
        }
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
