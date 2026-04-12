#!/usr/bin/env python
"""
Materialize a tile dataset once from a tile-index CSV.

This avoids recropping on every training run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image


Image.MAX_IMAGE_PIXELS = None


def _to_int(v) -> int | None:
    if v in (None, "", "null"):
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def _crop_from_row(rgb: Image.Image, row: Dict) -> Image.Image:
    w, h = rgb.size
    x = _to_int(row.get("tile_x")) or 0
    y = _to_int(row.get("tile_y")) or 0
    size = _to_int(row.get("tile_size")) or 512
    tw = _to_int(row.get("tile_w")) or size
    th = _to_int(row.get("tile_h")) or size

    x0 = max(0, min(x, w - 1))
    y0 = max(0, min(y, h - 1))
    x1 = max(x0 + 1, min(x0 + tw, w))
    y1 = max(y0 + 1, min(y0 + th, h))
    return rgb.crop((x0, y0, x1, y1))


def _process_source_group(
    source_path: Path,
    rows: List[Dict],
    output_dir: Path,
    overwrite: bool,
) -> Dict[str, int]:
    if not source_path.is_file():
        return {"written": 0, "skipped": 0, "missing": len(rows)}

    written = 0
    skipped = 0
    with Image.open(source_path) as im:
        rgb = im.convert("RGB")
        for row in rows:
            fn = str(row["filename"])
            out_path = output_dir / fn
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            tile = _crop_from_row(rgb, row)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tile.save(out_path)
            written += 1
    return {"written": written, "skipped": skipped, "missing": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize tile images from tile-index CSV.")
    ap.add_argument("--labels-csv", required=True, help="Tile CSV with image_path/filename/tile coords")
    ap.add_argument("--output-dir", required=True, help="Output directory for tile images")
    ap.add_argument("--workers", type=int, default=0, help="Thread workers (0=auto)")
    ap.add_argument("--clean", action="store_true", help="Delete output dir before materialization")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--stats-json", default=None, help="Optional stats output JSON")
    args = ap.parse_args()

    labels_csv = Path(args.labels_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not labels_csv.is_file():
        raise SystemExit(f"labels CSV not found: {labels_csv}")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_csv)
    for col in ["image_path", "filename"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column in labels CSV: {col}")

    grouped: Dict[str, List[Dict]] = {}
    for row in df.to_dict(orient="records"):
        grouped.setdefault(str(row["image_path"]), []).append(row)

    workers = int(args.workers)
    if workers <= 0:
        workers = min(16, max(1, (os.cpu_count() or 1)))

    totals = {"written": 0, "skipped": 0, "missing": 0}
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for src, rows in grouped.items():
            futures.append(
                ex.submit(
                    _process_source_group,
                    Path(src),
                    rows,
                    output_dir,
                    bool(args.overwrite),
                )
            )

        done = 0
        total = len(futures)
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            totals["written"] += int(res["written"])
            totals["skipped"] += int(res["skipped"])
            totals["missing"] += int(res["missing"])
            done += 1
            if done % 5 == 0 or done == total:
                print(f"[materialize] processed {done}/{total} source images (workers={workers})")

    summary = {
        "labels_csv": str(labels_csv),
        "output_dir": str(output_dir),
        "rows_in_csv": int(len(df)),
        "source_images": int(len(grouped)),
        "workers": int(workers),
        "written": int(totals["written"]),
        "skipped_existing": int(totals["skipped"]),
        "missing_source_rows": int(totals["missing"]),
    }
    print(json.dumps(summary, indent=2))

    if args.stats_json:
        stats_path = Path(args.stats_json).resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
