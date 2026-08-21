#!/usr/bin/env python3
"""Summarize and visually audit a semantic inpainting dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from semantic_mask_utils import CLASS_NAME_BY_ID, TARGET_CLASS_IDS  # noqa: E402


LABEL_COLORS = {
    0: (0, 0, 0), 1: (0, 114, 178), 2: (0, 158, 115), 3: (86, 180, 233),
    4: (230, 159, 0), 5: (240, 228, 66), 6: (0, 200, 90), 7: (0, 80, 170),
    8: (0, 150, 150), 9: (213, 94, 0), 10: (255, 190, 0), 11: (0, 180, 130),
}


def _read_manifest(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve(raw: str, manifest_parent: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (manifest_parent / p).resolve()


def _label_vis(labels: np.ndarray) -> Image.Image:
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for cid, color in LABEL_COLORS.items():
        out[labels == cid] = color
    return Image.fromarray(out)


def _mask_overlay(rgb: Image.Image, mask: Image.Image, color=(0, 255, 80), alpha=0.42) -> Image.Image:
    base = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask.convert("L")) > 127
    overlay = base.copy()
    c = np.asarray(color, dtype=np.float32)
    overlay[m] = (1.0 - alpha) * overlay[m] + alpha * c
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def _draw_text(img: Image.Image, text: str, height: int = 26) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + height), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, img.height + 4), text, fill="black", font=ImageFont.load_default())
    return canvas


def _contact_sheet(rows, out_path: Path, manifest_parent: Path, max_samples: int) -> None:
    selected = rows[:max_samples]
    if not selected:
        return
    thumb = 220
    cards = []
    for row in selected:
        rgb = Image.open(_resolve(row["image_path"], manifest_parent)).convert("RGB").resize((thumb, thumb))
        mask = Image.open(_resolve(row["mask_path"], manifest_parent)).convert("L").resize((thumb, thumb), Image.Resampling.NEAREST)
        labels = np.load(_resolve(row["source_label_path"], manifest_parent))
        label_img = _label_vis(labels).resize((thumb, thumb), Image.Resampling.NEAREST)
        masked = _mask_overlay(rgb, mask)
        trio = Image.new("RGB", (thumb * 3, thumb), "white")
        trio.paste(rgb, (0, 0))
        trio.paste(masked, (thumb, 0))
        trio.paste(label_img, (thumb * 2, 0))
        text = (
            f"{row['split']} | {row['mask_mode']} | purity={float(row['target_purity']):.2f} "
            f"| area={float(row['mask_area_fraction']):.3f}"
        )
        cards.append(_draw_text(trio, text))

    width = thumb * 3
    height = sum(card.height for card in cards)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for card in cards:
        sheet.paste(card, (0, y))
        y += card.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/artifacts/semantic_inpaint_v1/semantic_inpaint_manifest.csv")
    ap.add_argument("--output-dir", default="data/artifacts/semantic_inpaint_v1/audit")
    ap.add_argument("--samples-per-class", type=int, default=12)
    args = ap.parse_args()

    manifest = Path(args.manifest).resolve()
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")
    rows = _read_manifest(manifest)
    if not rows:
        raise SystemExit("Manifest is empty")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    summary_rows = []
    for split in ("train", "val", "test"):
        for cid in TARGET_CLASS_IDS:
            subset = [r for r in rows if r["split"] == split and int(r["target_class_id"]) == cid]
            grouped[(split, cid)] = subset
            purities = np.array([float(r["target_purity"]) for r in subset], dtype=float)
            areas = np.array([float(r["mask_area_fraction"]) for r in subset], dtype=float)
            slides = {r["slide_id"] for r in subset}
            modes = Counter(r["mask_mode"] for r in subset)
            summary_rows.append({
                "split": split,
                "target_class_id": cid,
                "target_class_name": CLASS_NAME_BY_ID[cid],
                "samples": len(subset),
                "unique_slides": len(slides),
                "interior_samples": int(modes.get("interior", 0)),
                "boundary_samples": int(modes.get("boundary", 0)),
                "purity_mean": float(purities.mean()) if len(purities) else None,
                "purity_min": float(purities.min()) if len(purities) else None,
                "mask_area_mean": float(areas.mean()) if len(areas) else None,
            })

    csv_out = output / "semantic_inpaint_audit.csv"
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_out = output / "semantic_inpaint_audit.json"
    json_out.write_text(json.dumps({"manifest": str(manifest), "rows": summary_rows}, indent=2), encoding="utf-8")

    for cid in TARGET_CLASS_IDS:
        subset = grouped[("train", cid)]
        if not subset:
            continue
        idx = np.unique(np.linspace(0, len(subset) - 1, num=min(args.samples_per_class, len(subset)), dtype=int))
        chosen = [subset[int(i)] for i in idx]
        _contact_sheet(
            chosen,
            output / "contact_sheets" / f"class_{cid:02d}_{CLASS_NAME_BY_ID[cid]}.png",
            manifest.parent,
            len(chosen),
        )

    print(f"Audit CSV     : {csv_out}")
    print(f"Audit JSON    : {json_out}")
    print(f"Contact sheets: {output / 'contact_sheets'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
