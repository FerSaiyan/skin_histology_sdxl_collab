#!/usr/bin/env python
"""Generate QC visuals for a tiles_for_simulation dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _to_float(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=np.float64)


def _to_int(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    return np.array([int(float(r[key])) for r in rows], dtype=np.int64)


def _plot_hist(values: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 30) -> None:
    fig = plt.figure(figsize=(8, 4.5), dpi=140)
    ax = fig.add_subplot(111)
    ax.hist(values, bins=bins, color="#3a86ff", edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_scatter_tissue_air(rows: List[Dict[str, str]], out_path: Path) -> None:
    x = _to_float(rows, "rot_tissue_frac")
    y = _to_float(rows, "rot_air_frac")
    c = _to_float(rows, "normal_confidence")

    fig = plt.figure(figsize=(6.4, 5.2), dpi=140)
    ax = fig.add_subplot(111)
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=18, alpha=0.8)
    ax.set_title("Rotated Tile Tissue vs Air Fractions")
    ax.set_xlabel("rot_tissue_frac")
    ax.set_ylabel("rot_air_frac")
    ax.grid(alpha=0.2)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("normal_confidence")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_class_presence(rows: List[Dict[str, str]], dataset_root: Path, out_path: Path, top_k: int = 12) -> None:
    counts: Dict[int, int] = {}
    for r in rows:
        label_path = dataset_root / r["label_id_npy_path"]
        arr = np.load(label_path)
        unique = np.unique(arr)
        for cid in unique:
            k = int(cid)
            counts[k] = counts.get(k, 0) + 1

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    if not items:
        return

    xs = [str(k) for k, _ in items]
    ys = [v for _, v in items]

    fig = plt.figure(figsize=(8, 4.5), dpi=140)
    ax = fig.add_subplot(111)
    ax.bar(xs, ys, color="#ff7f11")
    ax.set_title("Class Presence Across Kept Tiles")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Tiles containing class")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _make_contact_sheet(rows: List[Dict[str, str]], dataset_root: Path, out_path: Path, num_samples: int) -> List[str]:
    if not rows:
        return []

    n = min(num_samples, len(rows))
    sampled = rows[:]
    random.shuffle(sampled)
    sampled = sampled[:n]

    fig, axes = plt.subplots(n, 3, figsize=(13, 4.2 * n), dpi=120)
    if n == 1:
        axes = np.array([axes])

    chosen_ids: List[str] = []
    for i, r in enumerate(sampled):
        chosen_ids.append(r["tile_id"])
        rgb = np.asarray(Image.open(dataset_root / r["rgb_path"]).convert("RGB"))
        vis = np.asarray(Image.open(dataset_root / r["label_vis_path"]).convert("RGB"))
        epi = np.asarray(Image.open(dataset_root / r["epidermis_mask_path"]).convert("L"))

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(
            f"RGB {r['tile_id']}\nrot={float(r['rotation_deg']):.1f} conf={float(r['normal_confidence']):.2f}",
            fontsize=9,
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(vis)
        axes[i, 1].set_title("Semantic Label Visualization", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(epi, cmap="gray")
        axes[i, 2].set_title(
            f"Epidermis Mask\ntissue={float(r['rot_tissue_frac']):.2f}, air={float(r['rot_air_frac']):.2f}",
            fontsize=9,
        )
        axes[i, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return chosen_ids


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="QC plots + contact sheets for tiles_for_simulation.")
    ap.add_argument("--dataset-dir", required=True, help="tiles_for_simulation dataset root.")
    ap.add_argument("--manifest-csv", default="", help="Manifest CSV path (defaults to <dataset>/tiles_manifest.csv).")
    ap.add_argument("--output-dir", required=True, help="Directory for QC figures/json.")
    ap.add_argument("--num-samples", type=int, default=12, help="Number of random sample rows in contact sheet.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    random.seed(args.seed)

    dataset_root = Path(args.dataset_dir).resolve()
    manifest = Path(args.manifest_csv).resolve() if args.manifest_csv else (dataset_root / "tiles_manifest.csv")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest(manifest)
    if not rows:
        raise RuntimeError(f"Manifest has zero rows: {manifest}")

    rot = _to_float(rows, "rotation_deg")
    conf = _to_float(rows, "normal_confidence")
    tissue = _to_float(rows, "rot_tissue_frac")
    air = _to_float(rows, "rot_air_frac")
    interface_px = _to_int(rows, "rot_air_epidermis_interface_px")

    _plot_hist(rot, "Rotation Degrees Distribution", "rotation_deg", out_dir / "rotation_hist.png")
    _plot_hist(conf, "Normal Confidence Distribution", "normal_confidence", out_dir / "confidence_hist.png")
    _plot_hist(interface_px, "Air-Epidermis Interface Pixel Distribution", "rot_air_epidermis_interface_px", out_dir / "interface_pixels_hist.png")
    _plot_scatter_tissue_air(rows, out_dir / "tissue_air_scatter.png")
    _plot_class_presence(rows, dataset_root, out_dir / "class_presence_bar.png")

    chosen_ids = _make_contact_sheet(rows, dataset_root, out_dir / "rotated_tiles_contact_sheet.png", args.num_samples)

    summary = {
        "dataset_dir": str(dataset_root),
        "manifest_csv": str(manifest),
        "num_tiles": len(rows),
        "rotation_deg": {
            "min": float(rot.min()),
            "max": float(rot.max()),
            "mean": float(rot.mean()),
            "std": float(rot.std()),
        },
        "normal_confidence": {
            "min": float(conf.min()),
            "max": float(conf.max()),
            "mean": float(conf.mean()),
            "std": float(conf.std()),
        },
        "rot_tissue_frac": {
            "min": float(tissue.min()),
            "max": float(tissue.max()),
            "mean": float(tissue.mean()),
        },
        "rot_air_frac": {
            "min": float(air.min()),
            "max": float(air.max()),
            "mean": float(air.mean()),
        },
        "sample_tile_ids": chosen_ids,
        "artifacts": {
            "rotation_hist": str((out_dir / "rotation_hist.png")),
            "confidence_hist": str((out_dir / "confidence_hist.png")),
            "interface_pixels_hist": str((out_dir / "interface_pixels_hist.png")),
            "tissue_air_scatter": str((out_dir / "tissue_air_scatter.png")),
            "class_presence_bar": str((out_dir / "class_presence_bar.png")),
            "rotated_tiles_contact_sheet": str((out_dir / "rotated_tiles_contact_sheet.png")),
        },
    }
    (out_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
