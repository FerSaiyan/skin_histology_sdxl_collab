#!/usr/bin/env python
"""Build a fixed-canvas stack from a contiguous sequence of source slices.

This is the safe path for Zenodo melanoma HR subsets: pass the full raw glob
(`data/raw/zenodo_melanoma/cropped_slices/*.png`) and select a contiguous
`--start-index`/`--num-slices` window. Filtered globs like `*_a.png` are rejected
because they skip intervening source-order files.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sequence_utils import (
    SequenceValidationError,
    collect_ordered_paths,
    sequence_summary,
    validate_contiguous_selection,
)

Image.MAX_IMAGE_PIXELS = None


def _paste_offset(canvas_size: tuple[int, int], image_size: tuple[int, int], anchor: str) -> tuple[int, int]:
    canvas_w, canvas_h = canvas_size
    image_w, image_h = image_size
    if anchor == "top-left":
        return 0, 0
    if anchor == "center":
        return max(0, (canvas_w - image_w) // 2), max(0, (canvas_h - image_h) // 2)
    raise ValueError(f"Unsupported anchor: {anchor}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a fixed-canvas sequential slice stack from contiguous source images."
    )
    parser.add_argument("--source-glob", required=True, help="Source image glob. Use the full raw glob, not a filtered subset.")
    parser.add_argument("--output-dir", required=True, help="Directory for slice_XXXX.png outputs.")
    parser.add_argument("--start-index", type=int, default=0, help="0-based index into the full natural source order.")
    parser.add_argument("--num-slices", type=int, default=0, help="Number of slices to output (0 = all from start-index).")
    parser.add_argument("--canvas-width", type=int, default=0, help="Fixed canvas width. Default: max selected width.")
    parser.add_argument("--canvas-height", type=int, default=0, help="Fixed canvas height. Default: max selected height.")
    parser.add_argument("--anchor", choices=["top-left", "center"], default="top-left", help="Where to paste smaller source images on the canvas.")
    parser.add_argument("--fill", choices=["black", "white"], default="black", help="Canvas fill color.")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON path. Default: output_dir/sequence_manifest.json")
    parser.add_argument(
        "--allow-noncontiguous-source-glob",
        action="store_true",
        help="Debug escape hatch. Do not use for sequential-volume claims.",
    )
    args = parser.parse_args()

    if args.start_index < 0:
        parser.error("--start-index must be >= 0")
    if args.num_slices < 0:
        parser.error("--num-slices must be >= 0")

    all_paths = collect_ordered_paths(args.source_glob)
    if not args.allow_noncontiguous_source_glob:
        try:
            validate_contiguous_selection(all_paths, context="source glob")
        except SequenceValidationError as exc:
            raise SystemExit(
                f"ERROR: {exc}\n"
                "Use the full raw source glob and choose a contiguous --start-index/--num-slices window."
            )

    end_index = len(all_paths) if args.num_slices == 0 else args.start_index + args.num_slices
    selected = all_paths[args.start_index:end_index]
    if not selected:
        raise SystemExit("No selected slices. Check --start-index/--num-slices.")
    if end_index > len(all_paths):
        raise SystemExit(
            f"Requested slice window [{args.start_index}, {end_index}) exceeds source count {len(all_paths)}."
        )

    try:
        validate_contiguous_selection(selected, reference_paths=all_paths, context="selected slice window")
    except SequenceValidationError as exc:
        raise SystemExit(f"ERROR: {exc}")

    sizes: list[tuple[int, int]] = []
    for path in selected:
        with Image.open(path) as img:
            sizes.append(img.size)

    canvas_w = args.canvas_width or max(width for width, _ in sizes)
    canvas_h = args.canvas_height or max(height for _, height in sizes)
    if canvas_w <= 0 or canvas_h <= 0:
        parser.error("Canvas dimensions must be positive")

    too_large = [(path.name, size) for path, size in zip(selected, sizes) if size[0] > canvas_w or size[1] > canvas_h]
    if too_large:
        details = ", ".join(f"{name}={w}x{h}" for name, (w, h) in too_large[:5])
        raise SystemExit(f"Selected image(s) exceed canvas {canvas_w}x{canvas_h}: {details}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fill = (0, 0, 0) if args.fill == "black" else (255, 255, 255)

    rows = []
    for out_idx, (path, size) in enumerate(zip(selected, sizes)):
        with Image.open(path).convert("RGB") as img:
            canvas = Image.new("RGB", (canvas_w, canvas_h), fill)
            x_off, y_off = _paste_offset((canvas_w, canvas_h), img.size, args.anchor)
            canvas.paste(img, (x_off, y_off))
            out_name = f"slice_{out_idx:04d}.png"
            out_path = output_dir / out_name
            canvas.save(out_path)

        rows.append({
            "output_index": out_idx,
            "output_file": out_name,
            "source_order_index": args.start_index + out_idx,
            "source_file": path.name,
            "source_path": str(path.resolve()),
            "source_width": size[0],
            "source_height": size[1],
            "paste_x": x_off,
            "paste_y": y_off,
            "canvas_width": canvas_w,
            "canvas_height": canvas_h,
        })

    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_glob": args.source_glob,
        "source_summary": sequence_summary(all_paths),
        "selected_summary": sequence_summary(selected),
        "start_index": args.start_index,
        "num_slices": len(selected),
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "anchor": args.anchor,
        "fill": args.fill,
        "rows": rows,
    }
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "sequence_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} sequential slices to {output_dir}")
    print(f"Source order: {selected[0].name} -> {selected[-1].name}")
    print(f"Canvas: {canvas_w}x{canvas_h}, anchor={args.anchor}, fill={args.fill}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
