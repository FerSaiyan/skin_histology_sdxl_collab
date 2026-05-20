#!/usr/bin/env python
"""
Build a physics-ready `tiles_for_simulation` dataset from full-resolution Histo-Seg
slides.

For each image/mask pair, this script:
1) scans a 512x512 candidate grid over the full-resolution source image,
2) filters candidates to keep tiles with a visible air-epidermis interface and
   healthy tissue-vs-air balance,
3) estimates the local outward normal (air -> epidermis),
4) rotates each accepted tile so the outward normal points toward the tile top,
5) applies the exact same transform to the semantic mask to recover per-tile
   class-ID labels for simulation.

Outputs are written under one dataset root with a CSV manifest so downstream
MCX/PyXOpto/thermal workflows can consume the same aligned tile set.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.optical_ga.build_histoseg_label_volume import (  # noqa: E402
    CLASS_NAME_BY_ID,
    HISTOSEG_COLOR_TO_CLASS_ID,
)
from scripts.optical_ga.estimate_epidermis_normal import (  # noqa: E402
    _HAS_SKIMAGE,
    estimate_epidermis_normal,
)

_HAS_SCIPY = False
try:
    from scipy.ndimage import affine_transform as _scipy_affine

    _HAS_SCIPY = True
except Exception:
    pass


CLASS_ID_TO_COLOR: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (224, 224, 224),
    2: (96, 96, 96),
    3: (150, 150, 0),
    4: (127, 255, 255),
    5: (255, 156, 0),
    6: (255, 0, 255),
    7: (0, 255, 0),
    8: (0, 156, 255),
    9: (127, 96, 255),
    10: (112, 48, 160),
    11: (0, 0, 128),
}


@dataclass
class TileFilters:
    min_tissue_frac: float
    max_tissue_frac: float
    min_epidermis_frac: float
    min_air_epidermis_interface_px: int
    min_normal_confidence: float


def _warp_affine(
    src: np.ndarray,
    rot_deg: float,
    center_xy: Tuple[float, float],
    dsize: Tuple[int, int],
    order: int,
) -> np.ndarray:
    """Rotate src by rot_deg CCW around center_xy and crop to dsize."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for affine tile extraction")

    w, h = dsize
    theta = np.radians(rot_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = center_xy
    ct_r = (h - 1) / 2.0
    ct_c = (w - 1) / 2.0

    off_r = cy - sin_t * ct_c - cos_t * ct_r
    off_c = cx - cos_t * ct_c + sin_t * ct_r
    matrix = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)
    offset = np.array([off_r, off_c], dtype=np.float64)
    output_shape = (h, w)

    if src.ndim == 2:
        return _scipy_affine(
            src,
            matrix,
            offset=offset,
            output_shape=output_shape,
            order=order,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

    channels = []
    for c in range(src.shape[2]):
        channels.append(
            _scipy_affine(
                src[..., c],
                matrix,
                offset=offset,
                output_shape=output_shape,
                order=order,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
        )
    return np.stack(channels, axis=-1)


def _load_pairs_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"pairs csv not found: {path}")

    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "image_path" not in row or "mask_path" not in row:
                raise ValueError(
                    f"CSV missing required columns image_path/mask_path: {path}"
                )
            rows.append(row)
    return rows


def _rgb_mask_to_class_map(mask_rgb: np.ndarray) -> np.ndarray:
    h, w = mask_rgb.shape[:2]
    class_map = np.zeros((h, w), dtype=np.uint8)

    known_mask = np.zeros((h, w), dtype=bool)
    for rgb, cid in HISTOSEG_COLOR_TO_CLASS_ID.items():
        match = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=2)
        class_map[match] = np.uint8(cid)
        known_mask |= match

    unknown = int((~known_mask).sum())
    if unknown > 0:
        warnings.warn(
            f"Found {unknown} unknown mask pixels; mapped to class 0.",
            RuntimeWarning,
        )
    return class_map


def _air_epidermis_interface_px(epi_mask01: np.ndarray) -> int:
    """Count 4-neighbor boundary edges where air(0) touches epidermis(1)."""
    if epi_mask01.ndim != 2:
        raise ValueError("epi_mask01 must be 2D")
    a = epi_mask01
    lr = np.logical_xor(a[:, :-1] == 1, a[:, 1:] == 1).sum()
    ud = np.logical_xor(a[:-1, :] == 1, a[1:, :] == 1).sum()
    return int(lr + ud)


def _top_bottom_air_fraction(label_id_tile: np.ndarray, band_px: int) -> Tuple[float, float]:
    top = float((label_id_tile[:band_px, :] == 0).mean())
    bottom = float((label_id_tile[-band_px:, :] == 0).mean())
    return top, bottom


def _iter_grid_positions(h: int, w: int, tile_size: int, stride: int) -> Iterable[Tuple[int, int]]:
    if tile_size > h or tile_size > w:
        return
    y_last = h - tile_size
    x_last = w - tile_size
    ys = list(range(0, y_last + 1, stride))
    xs = list(range(0, x_last + 1, stride))
    if ys[-1] != y_last:
        ys.append(y_last)
    if xs[-1] != x_last:
        xs.append(x_last)
    for y0 in ys:
        for x0 in xs:
            yield y0, x0


def _class_hist(tile_label: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for cid in np.unique(tile_label):
        c = int(cid)
        out[str(c)] = int((tile_label == c).sum())
    return out


def _label_vis(tile_label: np.ndarray) -> np.ndarray:
    h, w = tile_label.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, rgb in CLASS_ID_TO_COLOR.items():
        vis[tile_label == cid] = rgb
    return vis


def _sanitize_rot_tag(rotation_deg: float) -> str:
    tag = f"{rotation_deg:+07.2f}"
    return tag.replace("+", "p").replace("-", "m").replace(".", "d")


def _resolve_existing(path_str: str, csv_parent: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    p2 = (csv_parent / p).resolve()
    if p2.exists():
        return p2
    return p


def _process_pair(
    row: Dict[str, str],
    csv_parent: Path,
    out_root: Path,
    tile_size: int,
    stride: int,
    filters: TileFilters,
    max_tiles_per_image: int,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    image_path = _resolve_existing(row["image_path"], csv_parent)
    mask_path = _resolve_existing(row["mask_path"], csv_parent)

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"mask not found: {mask_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask_rgb = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)

    if image.shape[:2] != mask_rgb.shape[:2]:
        raise ValueError(
            f"shape mismatch image/mask for {image_path.name}: "
            f"{image.shape[:2]} vs {mask_rgb.shape[:2]}"
        )

    class_map = _rgb_mask_to_class_map(mask_rgb)
    full_epi_map = (class_map == 1).astype(np.float64)
    h, w = class_map.shape
    slide_id = image_path.stem

    out_rgb = out_root / "rgb"
    out_label = out_root / "label_id"
    out_label_vis = out_root / "label_vis"
    out_mask = out_root / "epidermis_mask"
    out_meta = out_root / "meta"
    for d in (out_rgb, out_label, out_label_vis, out_mask, out_meta):
        d.mkdir(parents=True, exist_ok=True)

    stats = {
        "num_candidates": 0,
        "num_keep": 0,
        "reject_tissue_frac": 0,
        "reject_epidermis_frac": 0,
        "reject_interface_px": 0,
        "reject_normal_error": 0,
        "reject_normal_conf": 0,
    }

    manifest_rows: List[Dict[str, object]] = []

    for y0, x0 in _iter_grid_positions(h, w, tile_size, stride):
        stats["num_candidates"] += 1
        raw_label = class_map[y0:y0 + tile_size, x0:x0 + tile_size]

        tissue_frac = float((raw_label != 0).mean())
        air_frac = 1.0 - tissue_frac
        epi_mask01 = (raw_label == 1).astype(np.int32)
        epi_frac = float(epi_mask01.mean())

        if tissue_frac < filters.min_tissue_frac or tissue_frac > filters.max_tissue_frac:
            stats["reject_tissue_frac"] += 1
            continue
        if epi_frac < filters.min_epidermis_frac:
            stats["reject_epidermis_frac"] += 1
            continue

        interface_px = _air_epidermis_interface_px(epi_mask01)
        if interface_px < filters.min_air_epidermis_interface_px:
            stats["reject_interface_px"] += 1
            continue

        try:
            normal_deg, normal_vec, conf, _, _ = estimate_epidermis_normal(epi_mask01)
        except Exception:
            stats["reject_normal_error"] += 1
            continue

        if conf < filters.min_normal_confidence:
            stats["reject_normal_conf"] += 1
            continue

        cx = float(x0 + (tile_size - 1) / 2.0)
        cy = float(y0 + (tile_size - 1) / 2.0)
        # First-pass orientation: outward normal (air -> epidermis) points down,
        # which makes air tend to sit near the top edge.
        rotation_deg = float(normal_deg - 90.0)

        def _render_at_rotation(rot_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            rot_rgb_local = _warp_affine(
                image,
                rot_deg,
                center_xy=(cx, cy),
                dsize=(tile_size, tile_size),
                order=1,
            )
            rot_epi_local = _warp_affine(
                full_epi_map,
                rot_deg,
                center_xy=(cx, cy),
                dsize=(tile_size, tile_size),
                order=0,
            )
            rot_epi_local = np.round(rot_epi_local).astype(np.uint8)
            rot_label_local = _warp_affine(
                class_map.astype(np.float64),
                rot_deg,
                center_xy=(cx, cy),
                dsize=(tile_size, tile_size),
                order=0,
            )
            rot_label_local = np.round(rot_label_local).astype(np.uint8)
            return rot_rgb_local, rot_epi_local, rot_label_local

        rot_rgb, rot_epi, rot_label = _render_at_rotation(rotation_deg)

        air_band = max(8, tile_size // 16)
        top_air_frac, bottom_air_frac = _top_bottom_air_fraction(rot_label, air_band)
        air_top_flip_applied = False

        # Deterministic correction: if air is still heavier near the bottom,
        # flip orientation by 180 deg.
        if bottom_air_frac > top_air_frac:
            rotation_deg += 180.0
            rot_rgb, rot_epi, rot_label = _render_at_rotation(rotation_deg)
            top_air_frac, bottom_air_frac = _top_bottom_air_fraction(rot_label, air_band)
            air_top_flip_applied = True

        rot_interface_px = _air_epidermis_interface_px(rot_epi.astype(np.int32))
        if rot_interface_px < filters.min_air_epidermis_interface_px:
            stats["reject_interface_px"] += 1
            continue

        rot_tissue_frac = float((rot_label != 0).mean())
        if rot_tissue_frac < filters.min_tissue_frac or rot_tissue_frac > filters.max_tissue_frac:
            stats["reject_tissue_frac"] += 1
            continue

        tile_id = (
            f"{slide_id}__x{x0:05d}_y{y0:05d}"
            f"__rot{_sanitize_rot_tag(rotation_deg)}"
        )

        rgb_path = out_rgb / f"{tile_id}.png"
        label_npy_path = out_label / f"{tile_id}.npy"
        label_vis_path = out_label_vis / f"{tile_id}.png"
        epi_mask_path = out_mask / f"{tile_id}.png"
        meta_path = out_meta / f"{tile_id}.json"

        Image.fromarray(rot_rgb.astype(np.uint8)).save(rgb_path)
        np.save(label_npy_path, rot_label)
        Image.fromarray(_label_vis(rot_label)).save(label_vis_path)
        Image.fromarray((rot_epi * 255).astype(np.uint8)).save(epi_mask_path)

        class_hist = _class_hist(rot_label)
        meta = {
            "tile_id": tile_id,
            "slide_id": slide_id,
            "source_image": str(image_path),
            "source_mask": str(mask_path),
            "x0": int(x0),
            "y0": int(y0),
            "tile_size": int(tile_size),
            "center_x": cx,
            "center_y": cy,
            "normal_deg": float(normal_deg),
            "normal_vector": [float(normal_vec[0]), float(normal_vec[1])],
            "rotation_deg": rotation_deg,
            "rotation_deg_initial": float(normal_deg - 90.0),
            "normal_confidence": float(conf),
            "air_top_flip_applied": bool(air_top_flip_applied),
            "air_top_frac": top_air_frac,
            "air_bottom_frac": bottom_air_frac,
            "raw_tissue_frac": tissue_frac,
            "raw_air_frac": air_frac,
            "raw_epidermis_frac": epi_frac,
            "raw_air_epidermis_interface_px": int(interface_px),
            "rot_tissue_frac": rot_tissue_frac,
            "rot_air_frac": 1.0 - rot_tissue_frac,
            "rot_epidermis_frac": float((rot_label == 1).mean()),
            "rot_air_epidermis_interface_px": int(rot_interface_px),
            "class_histogram": class_hist,
            "class_names_present": [
                CLASS_NAME_BY_ID.get(int(cid), f"class_{int(cid)}")
                for cid in sorted(int(x) for x in class_hist.keys())
            ],
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        manifest_rows.append(
            {
                "tile_id": tile_id,
                "slide_id": slide_id,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "x0": int(x0),
                "y0": int(y0),
                "tile_size": int(tile_size),
                "center_x": cx,
                "center_y": cy,
                "normal_deg": float(normal_deg),
                "rotation_deg": rotation_deg,
                "rotation_deg_initial": float(normal_deg - 90.0),
                "normal_confidence": float(conf),
                "air_top_flip_applied": bool(air_top_flip_applied),
                "air_top_frac": top_air_frac,
                "air_bottom_frac": bottom_air_frac,
                "raw_tissue_frac": tissue_frac,
                "raw_air_frac": air_frac,
                "raw_epidermis_frac": epi_frac,
                "raw_air_epidermis_interface_px": int(interface_px),
                "rot_tissue_frac": rot_tissue_frac,
                "rot_air_frac": 1.0 - rot_tissue_frac,
                "rot_epidermis_frac": float((rot_label == 1).mean()),
                "rot_air_epidermis_interface_px": int(rot_interface_px),
                "rgb_path": str(rgb_path.relative_to(out_root)),
                "label_id_npy_path": str(label_npy_path.relative_to(out_root)),
                "label_vis_path": str(label_vis_path.relative_to(out_root)),
                "epidermis_mask_path": str(epi_mask_path.relative_to(out_root)),
                "tile_meta_path": str(meta_path.relative_to(out_root)),
            }
        )

        stats["num_keep"] += 1
        if max_tiles_per_image > 0 and stats["num_keep"] >= max_tiles_per_image:
            break

    return manifest_rows, stats


def _write_manifest(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        cols = [
            "tile_id",
            "slide_id",
            "source_image",
            "source_mask",
            "x0",
            "y0",
            "tile_size",
            "center_x",
            "center_y",
            "normal_deg",
            "rotation_deg",
            "rotation_deg_initial",
            "normal_confidence",
            "air_top_flip_applied",
            "air_top_frac",
            "air_bottom_frac",
            "raw_tissue_frac",
            "raw_air_frac",
            "raw_epidermis_frac",
            "raw_air_epidermis_interface_px",
            "rot_tissue_frac",
            "rot_air_frac",
            "rot_epidermis_frac",
            "rot_air_epidermis_interface_px",
            "rgb_path",
            "label_id_npy_path",
            "label_vis_path",
            "epidermis_mask_path",
            "tile_meta_path",
        ]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build tiles_for_simulation by scanning high-res Histo-Seg slides, "
            "filtering for air-epidermis interface tiles, orienting normals upward, "
            "and materializing RGB + semantic label tiles."
        )
    )
    ap.add_argument("--pairs-csv", required=True, help="CSV with image_path/mask_path columns.")
    ap.add_argument("--output-dir", required=True, help="Output dataset root directory.")
    ap.add_argument("--manifest-csv", default=None, help="Output manifest CSV path.")
    ap.add_argument("--stats-json", default=None, help="Output stats JSON path.")

    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--min-tissue-frac", type=float, default=0.25)
    ap.add_argument("--max-tissue-frac", type=float, default=0.98)
    ap.add_argument("--min-epidermis-frac", type=float, default=0.02)
    ap.add_argument("--min-air-epidermis-interface-px", type=int, default=64)
    ap.add_argument("--min-normal-confidence", type=float, default=0.55)
    ap.add_argument("--max-tiles-per-image", type=int, default=0)
    ap.add_argument("--max-total-tiles", type=int, default=0)
    return ap.parse_args(argv[1:])


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    if not _HAS_SKIMAGE:
        raise RuntimeError(
            "build_tiles_for_simulation requires scikit-image for interface "
            "contour detection. Install with: pip install scikit-image"
        )

    pairs_csv = Path(args.pairs_csv).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_csv = Path(args.manifest_csv).resolve() if args.manifest_csv else (out_root / "tiles_manifest.csv")
    stats_json = Path(args.stats_json).resolve() if args.stats_json else (out_root / "tiles_stats.json")

    rows = _load_pairs_csv(pairs_csv)
    filters = TileFilters(
        min_tissue_frac=float(args.min_tissue_frac),
        max_tissue_frac=float(args.max_tissue_frac),
        min_epidermis_frac=float(args.min_epidermis_frac),
        min_air_epidermis_interface_px=int(args.min_air_epidermis_interface_px),
        min_normal_confidence=float(args.min_normal_confidence),
    )

    all_manifest_rows: List[Dict[str, object]] = []
    per_image_stats: Dict[str, Dict[str, int]] = {}

    for i, row in enumerate(rows):
        img_path = Path(row["image_path"])
        slide_key = img_path.stem
        mrows, st = _process_pair(
            row=row,
            csv_parent=pairs_csv.parent,
            out_root=out_root,
            tile_size=int(args.tile_size),
            stride=int(args.stride),
            filters=filters,
            max_tiles_per_image=int(args.max_tiles_per_image),
        )

        per_image_stats[slide_key] = st
        all_manifest_rows.extend(mrows)

        print(
            f"[{i + 1}/{len(rows)}] {slide_key}: "
            f"candidates={st['num_candidates']} keep={st['num_keep']}"
        )

        if args.max_total_tiles > 0 and len(all_manifest_rows) >= int(args.max_total_tiles):
            all_manifest_rows = all_manifest_rows[: int(args.max_total_tiles)]
            break

    _write_manifest(all_manifest_rows, manifest_csv)

    total_candidates = int(sum(v["num_candidates"] for v in per_image_stats.values()))
    total_kept = int(sum(v["num_keep"] for v in per_image_stats.values()))
    stats = {
        "pairs_csv": str(pairs_csv),
        "output_dir": str(out_root),
        "manifest_csv": str(manifest_csv),
        "num_source_pairs": len(rows),
        "num_tiles_kept": total_kept,
        "num_candidates_seen": total_candidates,
        "tile_size": int(args.tile_size),
        "stride": int(args.stride),
        "filters": {
            "min_tissue_frac": filters.min_tissue_frac,
            "max_tissue_frac": filters.max_tissue_frac,
            "min_epidermis_frac": filters.min_epidermis_frac,
            "min_air_epidermis_interface_px": filters.min_air_epidermis_interface_px,
            "min_normal_confidence": filters.min_normal_confidence,
            "max_tiles_per_image": int(args.max_tiles_per_image),
            "max_total_tiles": int(args.max_total_tiles),
        },
        "per_image_stats": per_image_stats,
    }
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(stats, indent=2))

    print(f"Saved manifest: {manifest_csv}")
    print(f"Saved stats:    {stats_json}")
    print(f"Kept {total_kept} / {total_candidates} candidate tiles")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
