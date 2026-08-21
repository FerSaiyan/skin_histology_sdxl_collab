#!/usr/bin/env python
"""Build fast, aligned Histo-Seg tile datasets.

Two modes share the same RGB and class-ID outputs:

``simulation``
    Keep tissue-air surface tiles with an air-epidermis interface, estimate the
    local surface normal, and orient the surface toward the tile top.

``segmentation``
    Keep any tile with enough labeled tissue. Epidermis and surface orientation
    are not required, so this mode can densely sample all annotated classes.

Candidate fractions are computed from block-summed integral images. Per-slide
result files make interrupted builds resumable. Only RGB, class-ID labels, the
manifest, and aggregate stats are required; derived QC files are opt-in.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import json
import os
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import h5py
except Exception:  # pragma: no cover - only required for segmentation shards
    h5py = None

Image.MAX_IMAGE_PIXELS = None

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

MANIFEST_COLUMNS = [
    "tile_id",
    "tile_mode",
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
    "class_histogram",
    "storage",
    "shard_path",
    "shard_index",
    "rgb_path",
    "label_id_npy_path",
    "label_vis_path",
    "epidermis_mask_path",
    "tile_meta_path",
]


@dataclass(frozen=True)
class TileFilters:
    min_tissue_frac: float
    max_tissue_frac: float
    min_epidermis_frac: float
    min_air_epidermis_interface_px: int
    min_normal_confidence: float


@dataclass(frozen=True)
class BuildOptions:
    mode: str
    tile_size: int
    stride: int
    screen_downsample: int
    filters: TileFilters
    max_tiles_per_image: int
    strict_air_epidermis: bool
    write_label_vis: bool
    write_epidermis_mask: bool
    write_tile_meta: bool
    png_compress_level: int
    rgb_format: str
    jpeg_quality: int
    storage: str
    staging_dir: str


def _warp_affine(
    src: np.ndarray,
    rot_deg: float,
    center_xy: Tuple[float, float],
    dsize: Tuple[int, int],
    order: int,
) -> np.ndarray:
    """Rotate ``src`` around a source coordinate and crop one output tile."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for affine tile extraction")

    w, h = dsize
    theta = np.radians(rot_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = center_xy
    ct_r = (h - 1) / 2.0
    ct_c = (w - 1) / 2.0
    matrix = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)
    offset = np.array(
        [cy - sin_t * ct_c - cos_t * ct_r, cx - cos_t * ct_c + sin_t * ct_r],
        dtype=np.float64,
    )
    output_shape = (h, w)

    if src.ndim == 2:
        return _scipy_affine(
            src,
            matrix,
            offset=offset,
            output_shape=output_shape,
            order=order,
            mode="constant",
            cval=0,
            prefilter=False,
        )

    channels = [
        _scipy_affine(
            src[..., channel],
            matrix,
            offset=offset,
            output_shape=output_shape,
            order=order,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        for channel in range(src.shape[2])
    ]
    return np.stack(channels, axis=-1)


def _load_pairs_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"pairs csv not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if any("image_path" not in row or "mask_path" not in row for row in rows):
        raise ValueError(f"CSV missing required columns image_path/mask_path: {path}")
    return rows


@lru_cache(maxsize=1)
def _color_lookup_tables() -> Tuple[np.ndarray, np.ndarray]:
    class_lut = np.zeros((256, 256, 256), dtype=np.uint8)
    known_lut = np.zeros((256, 256, 256), dtype=bool)
    for rgb, class_id in HISTOSEG_COLOR_TO_CLASS_ID.items():
        class_lut[rgb] = np.uint8(class_id)
        known_lut[rgb] = True
    return class_lut, known_lut


def _rgb_mask_to_class_map(mask_rgb: np.ndarray) -> np.ndarray:
    """Decode exact Histo-Seg colors with one vectorized lookup."""
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB mask, got {mask_rgb.shape}")
    class_lut, known_lut = _color_lookup_tables()
    red, green, blue = mask_rgb[..., 0], mask_rgb[..., 1], mask_rgb[..., 2]
    class_map = class_lut[red, green, blue]
    unknown = int((~known_lut[red, green, blue]).sum())
    if unknown:
        warnings.warn(f"Found {unknown} unknown mask pixels; mapped to class 0.", RuntimeWarning)
    return class_map


def _air_epidermis_interface_px(label_id: np.ndarray) -> int:
    """Count 4-neighbor edges that specifically join air(0) and epidermis(1)."""
    if label_id.ndim != 2:
        raise ValueError("label_id must be 2D")
    left, right = label_id[:, :-1], label_id[:, 1:]
    upper, lower = label_id[:-1, :], label_id[1:, :]
    horizontal = ((left == 0) & (right == 1)) | ((left == 1) & (right == 0))
    vertical = ((upper == 0) & (lower == 1)) | ((upper == 1) & (lower == 0))
    return int(horizontal.sum() + vertical.sum())


def _epidermis_boundary_px(label_id: np.ndarray) -> int:
    """Legacy metric: count epidermis edges against every other class."""
    epidermis = label_id == 1
    return int(
        np.logical_xor(epidermis[:, :-1], epidermis[:, 1:]).sum()
        + np.logical_xor(epidermis[:-1, :], epidermis[1:, :]).sum()
    )


def _top_bottom_air_fraction(label_id_tile: np.ndarray, band_px: int) -> Tuple[float, float]:
    return (
        float((label_id_tile[:band_px, :] == 0).mean()),
        float((label_id_tile[-band_px:, :] == 0).mean()),
    )


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


def _block_integral(binary: np.ndarray, downsample: int) -> Tuple[np.ndarray, int]:
    """Build an exact integral table of square block sums."""
    scale = max(1, int(downsample))
    h, w = binary.shape
    block_h = h // scale
    block_w = w // scale
    trimmed = binary[: block_h * scale, : block_w * scale]
    blocks = trimmed.reshape(block_h, scale, block_w, scale).sum(axis=(1, 3), dtype=np.uint16)
    integral = np.zeros((block_h + 1, block_w + 1), dtype=np.uint32)
    integral[1:, 1:] = blocks.cumsum(axis=0, dtype=np.uint32).cumsum(axis=1, dtype=np.uint32)
    return integral, scale


def _window_count(
    binary: np.ndarray,
    integral: np.ndarray,
    scale: int,
    y0: int,
    x0: int,
    tile_size: int,
) -> int:
    if y0 % scale or x0 % scale or tile_size % scale:
        return int(binary[y0:y0 + tile_size, x0:x0 + tile_size].sum())
    y0s, x0s = y0 // scale, x0 // scale
    y1s, x1s = (y0 + tile_size) // scale, (x0 + tile_size) // scale
    if y1s >= integral.shape[0] or x1s >= integral.shape[1]:
        return int(binary[y0:y0 + tile_size, x0:x0 + tile_size].sum())
    return (
        int(integral[y1s, x1s])
        - int(integral[y0s, x1s])
        - int(integral[y1s, x0s])
        + int(integral[y0s, x0s])
    )


def _class_hist(tile_label: np.ndarray) -> Dict[str, int]:
    counts = np.bincount(tile_label.ravel(), minlength=max(CLASS_NAME_BY_ID) + 1)
    return {str(class_id): int(count) for class_id, count in enumerate(counts) if count}


def _label_vis(tile_label: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for class_id, rgb in CLASS_ID_TO_COLOR.items():
        lut[class_id] = rgb
    return lut[tile_label]


def _sanitize_rot_tag(rotation_deg: float) -> str:
    return f"{rotation_deg:+07.2f}".replace("+", "p").replace("-", "m").replace(".", "d")


def _resolve_existing(path_str: str, csv_parent: Path) -> Path:
    path = Path(path_str)
    candidates = [path, csv_parent / path, csv_parent.parent / "raw" / "histo_seg_v2" / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def _empty_stats() -> Dict[str, int]:
    return {
        "num_candidates": 0,
        "num_keep": 0,
        "reject_tissue_low": 0,
        "reject_tissue_high": 0,
        "reject_epidermis_frac": 0,
        "reject_interface_px": 0,
        "reject_normal_error": 0,
        "reject_normal_conf": 0,
        "reject_post_rotation": 0,
    }


def _screen_candidates(class_map: np.ndarray, options: BuildOptions) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    tissue = class_map != 0
    epidermis = class_map == 1
    tissue_integral, scale = _block_integral(tissue, options.screen_downsample)
    epidermis_integral, _ = _block_integral(epidermis, options.screen_downsample)
    tile_pixels = options.tile_size * options.tile_size
    stats = _empty_stats()
    candidates: List[Dict[str, object]] = []

    for y0, x0 in _iter_grid_positions(*class_map.shape, options.tile_size, options.stride):
        stats["num_candidates"] += 1
        tissue_count = _window_count(tissue, tissue_integral, scale, y0, x0, options.tile_size)
        tissue_frac = tissue_count / tile_pixels
        if tissue_frac < options.filters.min_tissue_frac:
            stats["reject_tissue_low"] += 1
            continue
        if tissue_frac > options.filters.max_tissue_frac:
            stats["reject_tissue_high"] += 1
            continue

        candidate: Dict[str, object] = {
            "x0": x0,
            "y0": y0,
            "tissue_frac": tissue_frac,
            "epidermis_frac": 0.0,
            "interface_px": 0,
            "normal_deg": None,
            "normal_vector": None,
            "normal_confidence": None,
        }
        if options.mode == "segmentation":
            candidates.append(candidate)
            continue

        epidermis_count = _window_count(
            epidermis, epidermis_integral, scale, y0, x0, options.tile_size
        )
        epidermis_frac = epidermis_count / tile_pixels
        if epidermis_frac < options.filters.min_epidermis_frac:
            stats["reject_epidermis_frac"] += 1
            continue
        raw_label = class_map[y0:y0 + options.tile_size, x0:x0 + options.tile_size]
        interface_px = (
            _air_epidermis_interface_px(raw_label)
            if options.strict_air_epidermis else _epidermis_boundary_px(raw_label)
        )
        if interface_px < options.filters.min_air_epidermis_interface_px:
            stats["reject_interface_px"] += 1
            continue
        try:
            surface_mask = (
                (raw_label != 0).astype(np.uint8)
                if options.strict_air_epidermis else (raw_label == 1).astype(np.uint8)
            )
            normal_deg, normal_vector, confidence, _, _ = estimate_epidermis_normal(surface_mask)
        except Exception:
            stats["reject_normal_error"] += 1
            continue
        if confidence < options.filters.min_normal_confidence:
            stats["reject_normal_conf"] += 1
            continue
        candidate.update(
            epidermis_frac=epidermis_frac,
            interface_px=interface_px,
            normal_deg=float(normal_deg),
            normal_vector=[float(normal_vector[0]), float(normal_vector[1])],
            normal_confidence=float(confidence),
        )
        candidates.append(candidate)

    if options.max_tiles_per_image > 0:
        candidates = candidates[: options.max_tiles_per_image]
    return candidates, stats


def _save_png(array: np.ndarray, path: Path, compress_level: int) -> None:
    Image.fromarray(array).save(path, compress_level=compress_level)


def _save_rgb(array: np.ndarray, path: Path, options: BuildOptions) -> None:
    image = Image.fromarray(array)
    if options.rgb_format == "jpg":
        image.save(path, quality=options.jpeg_quality, subsampling=0, optimize=False)
    else:
        image.save(path, compress_level=options.png_compress_level)


def _materialize_candidate(
    *,
    image: np.ndarray,
    class_map: np.ndarray,
    image_path: Path,
    mask_path: Path,
    slide_id: str,
    candidate: Dict[str, object],
    out_root: Path,
    options: BuildOptions,
) -> Dict[str, object] | None:
    x0, y0 = int(candidate["x0"]), int(candidate["y0"])
    tile_size = options.tile_size
    cx = float(x0 + (tile_size - 1) / 2.0)
    cy = float(y0 + (tile_size - 1) / 2.0)
    normal_deg = candidate["normal_deg"]
    rotation_initial = float(normal_deg) - 90.0 if normal_deg is not None else 0.0
    rotation_deg = rotation_initial
    air_top_flip_applied = False

    if options.mode == "simulation":
        tile_rgb = _warp_affine(
            image, rotation_deg, center_xy=(cx, cy), dsize=(tile_size, tile_size), order=1
        ).astype(np.uint8)
        tile_label = _warp_affine(
            class_map, rotation_deg, center_xy=(cx, cy), dsize=(tile_size, tile_size), order=0
        ).astype(np.uint8)
        top_air_frac, bottom_air_frac = _top_bottom_air_fraction(tile_label, max(8, tile_size // 16))
        if bottom_air_frac > top_air_frac:
            rotation_deg += 180.0
            tile_rgb = np.ascontiguousarray(tile_rgb[::-1, ::-1])
            tile_label = np.ascontiguousarray(tile_label[::-1, ::-1])
            top_air_frac, bottom_air_frac = bottom_air_frac, top_air_frac
            air_top_flip_applied = True
        tile_id = f"{slide_id}__x{x0:05d}_y{y0:05d}__rot{_sanitize_rot_tag(rotation_deg)}"
    else:
        tile_rgb = np.ascontiguousarray(image[y0:y0 + tile_size, x0:x0 + tile_size])
        tile_label = np.ascontiguousarray(class_map[y0:y0 + tile_size, x0:x0 + tile_size])
        top_air_frac, bottom_air_frac = _top_bottom_air_fraction(tile_label, max(8, tile_size // 16))
        tile_id = f"{slide_id}__x{x0:05d}_y{y0:05d}__raw"

    rot_tissue_frac = float((tile_label != 0).mean())
    if not (options.filters.min_tissue_frac <= rot_tissue_frac <= options.filters.max_tissue_frac):
        return None
    rot_epi_frac = float((tile_label == 1).mean())
    rot_interface_px = (
        _air_epidermis_interface_px(tile_label)
        if options.strict_air_epidermis else _epidermis_boundary_px(tile_label)
    )
    if options.mode == "simulation" and rot_interface_px < options.filters.min_air_epidermis_interface_px:
        return None

    rgb_path = out_root / "rgb" / f"{tile_id}.{options.rgb_format}"
    label_path = out_root / "label_id" / f"{tile_id}.npy"
    label_vis_path = out_root / "label_vis" / f"{tile_id}.png"
    epidermis_path = out_root / "epidermis_mask" / f"{tile_id}.png"
    meta_path = out_root / "meta" / f"{tile_id}.json"
    _save_rgb(tile_rgb, rgb_path, options)
    np.save(label_path, tile_label)

    class_hist = _class_hist(tile_label)
    meta = {
        "tile_id": tile_id,
        "tile_mode": options.mode,
        "slide_id": slide_id,
        "source_image": str(image_path),
        "source_mask": str(mask_path),
        "x0": x0,
        "y0": y0,
        "tile_size": tile_size,
        "center_x": cx,
        "center_y": cy,
        "normal_deg": normal_deg,
        "normal_vector": candidate["normal_vector"],
        "rotation_deg": rotation_deg,
        "rotation_deg_initial": rotation_initial,
        "normal_confidence": candidate["normal_confidence"],
        "air_top_flip_applied": air_top_flip_applied,
        "air_top_frac": top_air_frac,
        "air_bottom_frac": bottom_air_frac,
        "raw_tissue_frac": candidate["tissue_frac"],
        "raw_air_frac": 1.0 - float(candidate["tissue_frac"]),
        "raw_epidermis_frac": candidate["epidermis_frac"],
        "raw_air_epidermis_interface_px": candidate["interface_px"],
        "rot_tissue_frac": rot_tissue_frac,
        "rot_air_frac": 1.0 - rot_tissue_frac,
        "rot_epidermis_frac": rot_epi_frac,
        "rot_air_epidermis_interface_px": rot_interface_px,
        "class_histogram": class_hist,
        "class_names_present": [
            CLASS_NAME_BY_ID.get(int(class_id), f"class_{class_id}") for class_id in class_hist
        ],
    }
    if options.write_label_vis:
        _save_png(_label_vis(tile_label), label_vis_path, options.png_compress_level)
    if options.write_epidermis_mask:
        _save_png(((tile_label == 1) * 255).astype(np.uint8), epidermis_path, options.png_compress_level)
    if options.write_tile_meta:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        key: value
        for key, value in {
            **meta,
            "class_histogram": json.dumps(class_hist, separators=(",", ":")),
            "storage": "files",
            "shard_path": "",
            "shard_index": "",
            "rgb_path": str(rgb_path.relative_to(out_root)),
            "label_id_npy_path": str(label_path.relative_to(out_root)),
            "label_vis_path": str(label_vis_path.relative_to(out_root)) if options.write_label_vis else "",
            "epidermis_mask_path": str(epidermis_path.relative_to(out_root))
            if options.write_epidermis_mask else "",
            "tile_meta_path": str(meta_path.relative_to(out_root)) if options.write_tile_meta else "",
        }.items()
        if key in MANIFEST_COLUMNS
    }


def _segmentation_manifest_row(
    *,
    tile_id: str,
    slide_id: str,
    image_path: Path,
    mask_path: Path,
    candidate: Dict[str, object],
    tile_label: np.ndarray,
    shard_path: Path,
    shard_index: int,
    out_root: Path,
    tile_size: int,
) -> Dict[str, object]:
    x0, y0 = int(candidate["x0"]), int(candidate["y0"])
    tissue_frac = float(candidate["tissue_frac"])
    epidermis_frac = float((tile_label == 1).mean())
    interface_px = _air_epidermis_interface_px(tile_label)
    top_air, bottom_air = _top_bottom_air_fraction(tile_label, max(8, tile_size // 16))
    class_hist = _class_hist(tile_label)
    return {
        "tile_id": tile_id,
        "tile_mode": "segmentation",
        "slide_id": slide_id,
        "source_image": str(image_path),
        "source_mask": str(mask_path),
        "x0": x0,
        "y0": y0,
        "tile_size": tile_size,
        "center_x": float(x0 + (tile_size - 1) / 2.0),
        "center_y": float(y0 + (tile_size - 1) / 2.0),
        "normal_deg": "",
        "rotation_deg": 0.0,
        "rotation_deg_initial": 0.0,
        "normal_confidence": "",
        "air_top_flip_applied": False,
        "air_top_frac": top_air,
        "air_bottom_frac": bottom_air,
        "raw_tissue_frac": tissue_frac,
        "raw_air_frac": 1.0 - tissue_frac,
        "raw_epidermis_frac": epidermis_frac,
        "raw_air_epidermis_interface_px": interface_px,
        "rot_tissue_frac": tissue_frac,
        "rot_air_frac": 1.0 - tissue_frac,
        "rot_epidermis_frac": epidermis_frac,
        "rot_air_epidermis_interface_px": interface_px,
        "class_histogram": json.dumps(class_hist, separators=(",", ":")),
        "storage": "hdf5",
        "shard_path": str(shard_path.relative_to(out_root)),
        "shard_index": int(shard_index),
        "rgb_path": "",
        "label_id_npy_path": "",
        "label_vis_path": "",
        "epidermis_mask_path": "",
        "tile_meta_path": "",
    }


def _materialize_segmentation_shard(
    *,
    image: np.ndarray,
    class_map: np.ndarray,
    image_path: Path,
    mask_path: Path,
    slide_id: str,
    candidates: List[Dict[str, object]],
    out_root: Path,
    options: BuildOptions,
) -> List[Dict[str, object]]:
    if h5py is None:
        raise RuntimeError("HDF5 storage requires h5py; install h5py")
    shard_path = out_root / "shards" / f"{slide_id}.h5"
    if options.staging_dir:
        temporary = Path(options.staging_dir) / f"{slide_id}.h5.tmp"
    else:
        temporary = shard_path.with_suffix(".h5.tmp")
    tile_size = options.tile_size
    manifest_rows: List[Dict[str, object]] = []
    with h5py.File(temporary, "w") as shard:
        label_dataset = shard.create_dataset(
            "labels",
            shape=(len(candidates), tile_size, tile_size),
            dtype=np.uint8,
            chunks=(1, tile_size, tile_size),
            compression="lzf",
            shuffle=True,
        )
        jpeg_parts: List[bytes] = []
        jpeg_offsets = [0]
        label_batch: List[np.ndarray] = []
        batch_start = 0
        for index, candidate in enumerate(candidates):
            x0, y0 = int(candidate["x0"]), int(candidate["y0"])
            tile_rgb = np.ascontiguousarray(image[y0:y0 + tile_size, x0:x0 + tile_size])
            tile_label = np.ascontiguousarray(class_map[y0:y0 + tile_size, x0:x0 + tile_size])
            encoded = io.BytesIO()
            Image.fromarray(tile_rgb).save(
                encoded,
                format="JPEG",
                quality=options.jpeg_quality,
                subsampling=0,
                optimize=False,
            )
            encoded_bytes = encoded.getvalue()
            jpeg_parts.append(encoded_bytes)
            jpeg_offsets.append(jpeg_offsets[-1] + len(encoded_bytes))
            label_batch.append(tile_label)
            if len(label_batch) >= 64 or index + 1 == len(candidates):
                label_dataset[batch_start:index + 1] = np.stack(label_batch)
                label_batch.clear()
                batch_start = index + 1
            tile_id = f"{slide_id}__x{x0:05d}_y{y0:05d}__raw"
            manifest_rows.append(
                _segmentation_manifest_row(
                    tile_id=tile_id,
                    slide_id=slide_id,
                    image_path=image_path,
                    mask_path=mask_path,
                    candidate=candidate,
                    tile_label=tile_label,
                    shard_path=shard_path,
                    shard_index=index,
                    out_root=out_root,
                    tile_size=tile_size,
                )
            )
        jpeg_blob = b"".join(jpeg_parts)
        shard.create_dataset("rgb_bytes", data=np.frombuffer(jpeg_blob, dtype=np.uint8))
        shard.create_dataset("rgb_offsets", data=np.asarray(jpeg_offsets, dtype=np.int64))
        shard.attrs["slide_id"] = slide_id
        shard.attrs["tile_size"] = tile_size
        shard.attrs["num_tiles"] = len(candidates)
    if options.staging_dir:
        destination_tmp = shard_path.with_suffix(".h5.tmp")
        shutil.copyfile(temporary, destination_tmp)
        destination_tmp.replace(shard_path)
        temporary.unlink()
    else:
        temporary.replace(shard_path)
    return manifest_rows


def _process_pair(
    row: Dict[str, str],
    csv_parent: Path,
    out_root: Path,
    options: BuildOptions,
) -> Tuple[List[Dict[str, object]], Dict[str, int], float]:
    started = time.perf_counter()
    image_path = _resolve_existing(row["image_path"], csv_parent)
    mask_path = _resolve_existing(row["mask_path"], csv_parent)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"mask not found: {mask_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        mask_rgb = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
    class_map = _rgb_mask_to_class_map(mask_rgb)
    del mask_rgb
    candidates, stats = _screen_candidates(class_map, options)
    if not candidates:
        return [], stats, time.perf_counter() - started

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    if image.shape[:2] != class_map.shape:
        raise ValueError(
            f"shape mismatch image/mask for {image_path.name}: {image.shape[:2]} vs {class_map.shape}"
        )

    slide_id = str(row.get("slice_id") or image_path.stem)
    if options.storage == "hdf5":
        rows = _materialize_segmentation_shard(
            image=image,
            class_map=class_map,
            image_path=image_path,
            mask_path=mask_path,
            slide_id=slide_id,
            candidates=candidates,
            out_root=out_root,
            options=options,
        )
        stats["num_keep"] = len(rows)
        return rows, stats, time.perf_counter() - started

    rows: List[Dict[str, object]] = []
    for candidate in candidates:
        manifest_row = _materialize_candidate(
            image=image,
            class_map=class_map,
            image_path=image_path,
            mask_path=mask_path,
            slide_id=slide_id,
            candidate=candidate,
            out_root=out_root,
            options=options,
        )
        if manifest_row is None:
            stats["reject_post_rotation"] += 1
        else:
            rows.append(manifest_row)
    stats["num_keep"] = len(rows)
    return rows, stats, time.perf_counter() - started


def _write_manifest(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _state_key(row: Dict[str, str]) -> str:
    return str(row.get("slice_id") or Path(row["image_path"]).stem).replace(os.sep, "_")


def _build_fingerprint(args: argparse.Namespace, options: BuildOptions, pairs_csv: Path) -> Dict[str, object]:
    return {
        "pairs_csv": str(pairs_csv),
        "options": {**asdict(options), "filters": asdict(options.filters)},
        "max_total_tiles": int(args.max_total_tiles),
        "max_source_pairs": int(args.max_source_pairs),
    }


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", required=True, help="CSV with image_path/mask_path columns.")
    parser.add_argument("--output-dir", required=True, help="Output dataset root directory.")
    parser.add_argument("--manifest-csv", default=None, help="Output manifest CSV path.")
    parser.add_argument("--stats-json", default=None, help="Output stats JSON path.")
    parser.add_argument("--mode", choices=["simulation", "segmentation"], default="simulation")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Grid stride. Defaults to 256 for simulation and 128 for segmentation.",
    )
    parser.add_argument("--screen-downsample", type=int, default=4)
    parser.add_argument("--min-tissue-frac", type=float, default=0.25)
    parser.add_argument("--max-tissue-frac", type=float, default=None)
    parser.add_argument("--min-epidermis-frac", type=float, default=0.02)
    parser.add_argument("--min-air-epidermis-interface-px", type=int, default=64)
    parser.add_argument("--min-normal-confidence", type=float, default=0.55)
    parser.add_argument(
        "--strict-air-epidermis",
        action="store_true",
        help="Require a literal air(0)-epidermis(1) edge instead of the legacy full epidermis contour.",
    )
    parser.add_argument("--max-tiles-per-image", type=int, default=0)
    parser.add_argument("--max-total-tiles", type=int, default=0)
    parser.add_argument("--max-source-pairs", type=int, default=0, help="Limit source pairs for smoke tests.")
    parser.add_argument("--workers", type=int, default=0, help="Worker processes (0=auto).")
    parser.add_argument("--state-dir", default=None, help="Resumable state directory.")
    parser.add_argument("--reset-state", action="store_true", help="Discard matching resumable state.")
    parser.add_argument("--write-label-vis", action="store_true")
    parser.add_argument("--write-epidermis-mask", action="store_true")
    parser.add_argument("--write-tile-meta", action="store_true")
    parser.add_argument("--png-compress-level", type=int, choices=range(0, 10), default=1)
    parser.add_argument(
        "--rgb-format",
        choices=["png", "jpg"],
        default=None,
        help="Defaults to PNG for simulation and JPEG for segmentation.",
    )
    parser.add_argument("--jpeg-quality", type=int, choices=range(1, 101), default=95)
    parser.add_argument(
        "--storage",
        choices=["files", "hdf5"],
        default=None,
        help="Defaults to individual files for simulation and per-slide HDF5 shards for segmentation.",
    )
    parser.add_argument(
        "--staging-dir",
        default="",
        help="Fast local scratch directory for HDF5 construction before copying completed shards.",
    )
    return parser.parse_args(argv[1:])


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    if args.mode == "simulation" and not _HAS_SKIMAGE:
        raise RuntimeError("simulation mode requires scikit-image; install scikit-image")
    if not _HAS_SCIPY and args.mode == "simulation":
        raise RuntimeError("simulation mode requires scipy; install scipy")
    if args.tile_size <= 0 or (args.stride is not None and args.stride <= 0):
        raise SystemExit("--tile-size and --stride must be positive")

    pairs_csv = Path(args.pairs_csv).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    storage = str(args.storage or ("files" if args.mode == "simulation" else "hdf5"))
    if storage == "hdf5" and args.mode != "segmentation":
        raise SystemExit("HDF5 storage is currently supported only in segmentation mode")
    if storage == "hdf5" and (args.write_label_vis or args.write_epidermis_mask or args.write_tile_meta):
        raise SystemExit("Optional per-tile files require --storage files")
    if storage == "hdf5":
        (out_root / "shards").mkdir(parents=True, exist_ok=True)
    else:
        for required_dir in (out_root / "rgb", out_root / "label_id"):
            required_dir.mkdir(parents=True, exist_ok=True)
    if args.write_label_vis:
        (out_root / "label_vis").mkdir(parents=True, exist_ok=True)
    if args.write_epidermis_mask:
        (out_root / "epidermis_mask").mkdir(parents=True, exist_ok=True)
    if args.write_tile_meta:
        (out_root / "meta").mkdir(parents=True, exist_ok=True)
    staging_dir = str(Path(args.staging_dir).resolve()) if args.staging_dir else ""
    if staging_dir:
        Path(staging_dir).mkdir(parents=True, exist_ok=True)

    manifest_csv = Path(args.manifest_csv).resolve() if args.manifest_csv else out_root / "tiles_manifest.csv"
    stats_json = Path(args.stats_json).resolve() if args.stats_json else out_root / "tiles_stats.json"
    state_dir = Path(args.state_dir).resolve() if args.state_dir else out_root / ".build_state"
    if args.reset_state and state_dir.exists():
        shutil.rmtree(state_dir)
    results_dir = state_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    stride = int(args.stride) if args.stride is not None else (256 if args.mode == "simulation" else 128)
    max_tissue_frac = (
        float(args.max_tissue_frac)
        if args.max_tissue_frac is not None
        else (0.98 if args.mode == "simulation" else 1.0)
    )
    filters = TileFilters(
        min_tissue_frac=float(args.min_tissue_frac),
        max_tissue_frac=max_tissue_frac,
        min_epidermis_frac=float(args.min_epidermis_frac),
        min_air_epidermis_interface_px=int(args.min_air_epidermis_interface_px),
        min_normal_confidence=float(args.min_normal_confidence),
    )
    options = BuildOptions(
        mode=str(args.mode),
        tile_size=int(args.tile_size),
        stride=stride,
        screen_downsample=int(args.screen_downsample),
        filters=filters,
        max_tiles_per_image=int(args.max_tiles_per_image),
        strict_air_epidermis=bool(args.strict_air_epidermis),
        write_label_vis=bool(args.write_label_vis),
        write_epidermis_mask=bool(args.write_epidermis_mask),
        write_tile_meta=bool(args.write_tile_meta),
        png_compress_level=int(args.png_compress_level),
        rgb_format=str(args.rgb_format or ("png" if args.mode == "simulation" else "jpg")),
        jpeg_quality=int(args.jpeg_quality),
        storage=storage,
        staging_dir=staging_dir,
    )
    fingerprint = _build_fingerprint(args, options, pairs_csv)
    progress_path = state_dir / "progress.json"
    if progress_path.is_file():
        existing = json.loads(progress_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise SystemExit(
                "Existing state has different settings. Use --reset-state or a different output/state directory."
            )
    else:
        _write_json_atomic(progress_path, {"fingerprint": fingerprint})

    rows = _load_pairs_csv(pairs_csv)
    if args.max_source_pairs > 0:
        rows = rows[: int(args.max_source_pairs)]
    all_manifest_rows: List[Dict[str, object]] = []
    per_image_stats: Dict[str, Dict[str, object]] = {}
    pending: List[Dict[str, str]] = []
    for row in rows:
        result_path = results_dir / f"{_state_key(row)}.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            all_manifest_rows.extend(result["manifest_rows"])
            per_image_stats[_state_key(row)] = result["stats"]
        else:
            pending.append(row)

    workers = int(args.workers)
    if workers <= 0:
        workers = max(1, min(8, os.cpu_count() or 1))
    started = time.perf_counter()

    def consume(row: Dict[str, str], manifest_rows: List[Dict[str, object]], stats: Dict[str, int], seconds: float) -> None:
        key = _state_key(row)
        stats_with_time: Dict[str, object] = {**stats, "elapsed_seconds": seconds}
        result = {"manifest_rows": manifest_rows, "stats": stats_with_time}
        _write_json_atomic(results_dir / f"{key}.json", result)
        all_manifest_rows.extend(manifest_rows)
        per_image_stats[key] = stats_with_time

    if workers == 1:
        for row in tqdm(pending, desc=f"{args.mode} slides", unit="slide"):
            consume(row, *_process_pair(row, pairs_csv.parent, out_root, options))
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_row = {
                executor.submit(_process_pair, row, pairs_csv.parent, out_root, options): row for row in pending
            }
            progress = tqdm(total=len(pending), desc=f"{args.mode} slides ({workers} workers)", unit="slide")
            for future in cf.as_completed(future_to_row):
                row = future_to_row[future]
                consume(row, *future.result())
                progress.update(1)
                progress.set_postfix(tiles=len(all_manifest_rows))
            progress.close()

    all_manifest_rows.sort(key=lambda row: (str(row["slide_id"]), int(row["y0"]), int(row["x0"])))
    if args.max_total_tiles > 0:
        all_manifest_rows = all_manifest_rows[: int(args.max_total_tiles)]
    _write_manifest(all_manifest_rows, manifest_csv)

    total_candidates = int(sum(int(stats["num_candidates"]) for stats in per_image_stats.values()))
    total_kept = len(all_manifest_rows)
    elapsed = time.perf_counter() - started
    stats = {
        "mode": options.mode,
        "pairs_csv": str(pairs_csv),
        "output_dir": str(out_root),
        "manifest_csv": str(manifest_csv),
        "num_source_pairs": len(rows),
        "num_tiles_kept": total_kept,
        "num_candidates_seen": total_candidates,
        "keep_fraction": total_kept / total_candidates if total_candidates else 0.0,
        "tile_size": options.tile_size,
        "stride": options.stride,
        "screen_downsample": options.screen_downsample,
        "workers": workers,
        "elapsed_seconds_this_invocation": elapsed,
        "storage": options.storage,
        "required_outputs": (
            ["shards", "tiles_manifest.csv", "tiles_stats.json"]
            if options.storage == "hdf5"
            else ["rgb", "label_id", "tiles_manifest.csv", "tiles_stats.json"]
        ),
        "optional_outputs": {
            "label_vis": options.write_label_vis,
            "epidermis_mask": options.write_epidermis_mask,
            "tile_meta": options.write_tile_meta,
        },
        "filters": asdict(filters),
        "per_image_stats": per_image_stats,
    }
    _write_json_atomic(stats_json, stats)
    print(f"Saved manifest: {manifest_csv}")
    print(f"Saved stats:    {stats_json}")
    print(f"Mode {options.mode}: kept {total_kept} / {total_candidates} candidates ({stats['keep_fraction']:.2%})")
    print(f"Invocation time: {elapsed:.2f} seconds; completed slides reused from state: {len(rows) - len(pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
