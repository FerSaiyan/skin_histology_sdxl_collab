#!/usr/bin/env python
"""
Generate random inpaint masks for tile images with multithreading.

Masks are aligned one-to-one with tile filenames.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _sample_random_brush_mask(
    *,
    width: int,
    height: int,
    rng: np.random.Generator,
    min_area_frac: float,
    max_area_frac: float,
    min_strokes: int,
    max_strokes: int,
    min_vertices: int,
    max_vertices: int,
    min_brush_px: int,
    max_brush_px: int,
) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    n_strokes = int(rng.integers(min_strokes, max_strokes + 1))
    for _ in range(n_strokes):
        n_points = int(rng.integers(min_vertices, max_vertices + 1))
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        points = [(x, y)]
        for _ in range(n_points - 1):
            dx = int(rng.integers(-width // 3, width // 3 + 1))
            dy = int(rng.integers(-height // 3, height // 3 + 1))
            x = max(0, min(width - 1, x + dx))
            y = max(0, min(height - 1, y + dy))
            points.append((x, y))

        brush_w = int(rng.integers(min_brush_px, max_brush_px + 1))
        draw.line(points, fill=255, width=brush_w, joint="curve")
        r = max(1, brush_w // 2)
        for px, py in points:
            draw.ellipse((px - r, py - r, px + r, py + r), fill=255)

    arr = np.asarray(canvas, dtype=np.uint8)
    frac = float((arr > 0).mean())
    if frac < min_area_frac or frac > max_area_frac:
        return np.zeros_like(arr)
    return arr


def _generate_one(
    img_path: Path,
    out_dir: Path,
    idx: int,
    *,
    seed: int,
    white_threshold: int,
    tissue_min_overlap: float,
    max_attempts: int,
    min_area_frac: float,
    max_area_frac: float,
    min_strokes: int,
    max_strokes: int,
    min_vertices: int,
    max_vertices: int,
    min_brush_px: int,
    max_brush_px: int,
    feather_radius: float,
    mask_strength: float,
    overwrite: bool,
) -> Dict[str, int]:
    out_path = out_dir / img_path.name
    if out_path.exists() and not overwrite:
        return {"written": 0, "skipped": 1, "fallback": 0}

    with Image.open(img_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    tissue = np.logical_or.reduce(
        [rgb[:, :, 0] < white_threshold, rgb[:, :, 1] < white_threshold, rgb[:, :, 2] < white_threshold]
    )

    rng = np.random.default_rng(int(seed) + int(idx) * 10007)
    mask_arr = None
    for _ in range(max(1, int(max_attempts))):
        cand = _sample_random_brush_mask(
            width=w,
            height=h,
            rng=rng,
            min_area_frac=float(min_area_frac),
            max_area_frac=float(max_area_frac),
            min_strokes=int(min_strokes),
            max_strokes=int(max_strokes),
            min_vertices=int(min_vertices),
            max_vertices=int(max_vertices),
            min_brush_px=int(min_brush_px),
            max_brush_px=int(max_brush_px),
        )
        if cand.max() == 0:
            continue
        if tissue_min_overlap > 0:
            m = cand > 0
            overlap = float(np.logical_and(m, tissue).sum()) / float(max(m.sum(), 1))
            if overlap < tissue_min_overlap:
                continue
        mask_arr = cand
        break

    used_fallback = 0
    if mask_arr is None:
        fallback = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(fallback)
        rx = max(8, w // 6)
        ry = max(8, h // 6)
        cx, cy = w // 2, h // 2
        d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
        mask_arr = np.asarray(fallback, dtype=np.uint8)
        used_fallback = 1

    mask = Image.fromarray(mask_arr)
    if feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))
    if abs(mask_strength - 1.0) > 1e-6:
        arr = np.asarray(mask, dtype=np.float32) * float(mask_strength)
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        mask = Image.fromarray(arr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(out_path)
    return {"written": 1, "skipped": 0, "fallback": used_fallback}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate random masks for materialized tile dataset.")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=222)
    ap.add_argument("--white-threshold", type=int, default=235)
    ap.add_argument("--tissue-min-overlap", type=float, default=0.6)
    ap.add_argument("--max-attempts", type=int, default=20)
    ap.add_argument("--min-area-frac", type=float, default=0.12)
    ap.add_argument("--max-area-frac", type=float, default=0.40)
    ap.add_argument("--min-strokes", type=int, default=1)
    ap.add_argument("--max-strokes", type=int, default=4)
    ap.add_argument("--min-vertices", type=int, default=3)
    ap.add_argument("--max-vertices", type=int, default=8)
    ap.add_argument("--min-brush-px", type=int, default=24)
    ap.add_argument("--max-brush-px", type=int, default=128)
    ap.add_argument("--feather-radius", type=float, default=0.0)
    ap.add_argument("--mask-strength", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--stats-json", default=None)
    args = ap.parse_args()

    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"image-dir not found: {image_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}])
    if not image_paths:
        raise SystemExit(f"No images found in {image_dir}")

    workers = int(args.workers)
    if workers <= 0:
        workers = min(20, max(1, (os.cpu_count() or 1)))

    totals = {"written": 0, "skipped": 0, "fallback": 0}
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, img_path in enumerate(image_paths):
            futures.append(
                ex.submit(
                    _generate_one,
                    img_path,
                    output_dir,
                    idx,
                    seed=int(args.seed),
                    white_threshold=int(args.white_threshold),
                    tissue_min_overlap=float(args.tissue_min_overlap),
                    max_attempts=int(args.max_attempts),
                    min_area_frac=float(args.min_area_frac),
                    max_area_frac=float(args.max_area_frac),
                    min_strokes=int(args.min_strokes),
                    max_strokes=int(args.max_strokes),
                    min_vertices=int(args.min_vertices),
                    max_vertices=int(args.max_vertices),
                    min_brush_px=int(args.min_brush_px),
                    max_brush_px=int(args.max_brush_px),
                    feather_radius=float(args.feather_radius),
                    mask_strength=float(args.mask_strength),
                    overwrite=bool(args.overwrite),
                )
            )

        done = 0
        total = len(futures)
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            totals["written"] += int(res["written"])
            totals["skipped"] += int(res["skipped"])
            totals["fallback"] += int(res["fallback"])
            done += 1
            if done % 500 == 0 or done == total:
                print(f"[tile-masks] processed {done}/{total} (workers={workers})")

    summary = {
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "images_total": int(len(image_paths)),
        "workers": int(workers),
        "written": int(totals["written"]),
        "skipped": int(totals["skipped"]),
        "fallback_used": int(totals["fallback"]),
        "white_threshold": int(args.white_threshold),
        "tissue_min_overlap": float(args.tissue_min_overlap),
        "min_area_frac": float(args.min_area_frac),
        "max_area_frac": float(args.max_area_frac),
    }
    print(json.dumps(summary, indent=2))

    if args.stats_json:
        stats_path = Path(args.stats_json).resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
