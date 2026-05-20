#!/usr/bin/env python3
"""
plot_run_comparison.py

Generate matplotlib visualizations comparing 3D volume inpainting runs.
Outputs PNGs to --out-dir for Telegram review.

Usage:
    python scripts/analysis/plot_run_comparison.py \
        --run-dirs /path/to/run1 /path/to/run2 \
        --labels "Base Strict" "Tissue-Aware" \
        --out-dir /tmp/melanoma_compare_plots_<ts>

If --run-dirs is omitted, the script auto-selects the latest high-resolution
melanoma tissue-aware run from:
    - data/artifacts/3d/melanoma_strict_runs_base_tissueaware_hr/

Legacy low-resolution runs are blocked by default. Use --allow-lowres-source
only for explicit debugging of old runs.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

# ── matplotlib ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, Rectangle

Image.MAX_IMAGE_PIXELS = None

# ── optional skimage for SSIM ──────────────────────────────────────────────
try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SKIMAGE = True
except (ImportError, ValueError):
    HAVE_SKIMAGE = False

# ── constants ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SEARCH_PATHS = {
    "tissue_aware_hr": REPO_ROOT / "data/artifacts/3d/melanoma_strict_runs_base_tissueaware_hr",
    "base_strict": REPO_ROOT / "data/artifacts/3d/melanoma_strict_runs_base",
    "tissue_aware": REPO_ROOT / "data/artifacts/3d/melanoma_strict_runs_base_tissueaware",
    "melanoma_strict": REPO_ROOT / "data/artifacts/3d/melanoma_strict_runs",
}

DEFAULT_SOURCE_DIR = REPO_ROOT / "data/benchmarks/melanoma_3d/slices"
MIN_HR_SIDE_PIXELS = 1024


# ── helpers ────────────────────────────────────────────────────────────────


def _latest_subdir(parent: Path, exclude_tissue_aware: bool = False) -> Path | None:
    """Return the most recently modified subdirectory under *parent*.

    If *exclude_tissue_aware* is True, skip subdirectories whose
    ``run_manifest.json`` has ``tissue_aware_mask: true``.
    """
    if not parent.is_dir():
        return None
    entries = [p for p in parent.iterdir() if p.is_dir()]
    if not entries:
        return None
    # sort by mtime descending
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if exclude_tissue_aware:
        for e in entries:
            mf = e / "run_manifest.json"
            if mf.exists():
                try:
                    with open(mf) as f:
                        manifest = json.load(f)
                    if manifest.get("tissue_aware_mask") or \
                       manifest.get("args", {}).get("tissue_aware_mask"):
                        continue
                except Exception:
                    pass
            return e
        return None
    return entries[0]


def _load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _run_label_from_dir(run_dir: Path) -> str:
    """Derive a human-readable label from a run directory path."""
    name = run_dir.name
    # heuristic: strip trailing timestamp
    cleaned = re.sub(r"_\d{8}_\d{6}$", "", name)
    cleaned = cleaned.replace("_", " ").strip().title()
    if not cleaned:
        cleaned = name
    return cleaned


def _load_image(path: Path) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception:
        return None


def _load_image_thumbnail(path: Path | None, max_dim: int = 1200) -> tuple[np.ndarray | None, tuple[float, float]]:
    """Load an image as a thumbnail and return x/y scale factors from original pixels."""
    if not path:
        return None, (1.0, 1.0)
    try:
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        arr = np.array(img)
        scale_x = arr.shape[1] / orig_w if orig_w else 1.0
        scale_y = arr.shape[0] / orig_h if orig_h else 1.0
        return arr, (scale_x, scale_y)
    except Exception:
        return None, (1.0, 1.0)


def _image_shape(path: Path | None) -> tuple[int, int] | None:
    if not path:
        return None
    try:
        img = Image.open(path)
        w, h = img.size
        return h, w
    except Exception:
        return None


def _resolve_repo_path(path_value: str | os.PathLike | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _source_dir_from_manifest(manifest: dict | None) -> Path:
    """Resolve the original slice directory recorded by the run manifest."""
    if not manifest:
        return DEFAULT_SOURCE_DIR

    pairs = manifest.get("steps", {}).get("pairs_csv", {})
    image_dir = _resolve_repo_path(pairs.get("image_dir"))
    if image_dir:
        return image_dir

    slice_glob = _resolve_repo_path(manifest.get("args", {}).get("slice_glob"))
    if slice_glob:
        return slice_glob.parent

    return DEFAULT_SOURCE_DIR


def _extract_slice_num(value: str) -> int | None:
    match = re.search(r"slice[_-](\d+)", value)
    if not match:
        match = re.search(r"_(\d+)(?:_|$)", value)
    if not match:
        return None
    return int(match.group(1))


def _find_slice_image(directory: Path | None, slice_num: int, role: str) -> Path | None:
    if not directory or not directory.is_dir():
        return None

    token = f"slice_{slice_num:04d}"
    unpadded = f"slice_{slice_num}"
    if role == "source":
        patterns = [f"{token}.png", f"{unpadded}.png", f"{token}.*", f"{unpadded}.*"]
    elif role == "merged":
        patterns = [f"*{token}_edited.png", f"*{token}*.png", f"*{unpadded}_edited.png", f"*{unpadded}*.png"]
    elif role == "mask":
        patterns = [f"mask_{token}.png", f"*{token}*.png", f"*{unpadded}*.png"]
    else:
        patterns = [f"*{token}*.png", f"*{unpadded}*.png"]

    for pattern in patterns:
        candidates = sorted(directory.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def _load_source_image(run_data: dict, slice_num: int) -> np.ndarray | None:
    return _load_image(_find_slice_image(run_data.get("source_dir"), slice_num, "source"))


def _load_merged_image(run_data: dict, slice_num: int) -> np.ndarray | None:
    return _load_image(_find_slice_image(run_data.get("merged_dir"), slice_num, "merged"))


def _load_mask_array(run_data: dict, slice_num: int) -> np.ndarray | None:
    mask_path = _find_slice_image(run_data.get("masks_dir"), slice_num, "mask")
    if not mask_path:
        return None
    try:
        return np.array(Image.open(mask_path).convert("L"))
    except Exception:
        return None


def _source_shape_for_dir(source_dir: Path | None) -> tuple[int, int] | None:
    if not source_dir or not source_dir.is_dir():
        return None
    first = None
    for pattern in ("slice_*.png", "*.png", "*.jpg", "*.jpeg"):
        candidates = sorted(source_dir.glob(pattern))
        if candidates:
            first = candidates[0]
            break
    return _image_shape(first)


def _is_hr_source(run_data: dict) -> bool:
    shape = run_data.get("source_shape")
    if not shape:
        return False
    return min(shape) >= MIN_HR_SIDE_PIXELS


def _resize_to_match(arr: np.ndarray, reference: np.ndarray, resample=Image.LANCZOS) -> np.ndarray | None:
    if arr is None or reference is None:
        return None
    if arr.shape[:2] == reference.shape[:2]:
        return arr
    try:
        return np.array(Image.fromarray(arr).resize((reference.shape[1], reference.shape[0]), resample))
    except Exception:
        return None


def _select_common_slice_index(runs_data: list[dict]) -> int | None:
    slice_sets = [set(rd.get("slice_indices", [])) for rd in runs_data if rd.get("slice_indices")]
    if not slice_sets:
        return None
    common = sorted(set.intersection(*slice_sets))
    if common:
        return common[len(common) // 2]
    fallback = sorted(slice_sets[0])
    return fallback[len(fallback) // 2] if fallback else None


def _select_case_indices(run_data: dict, max_cases: int) -> list[int]:
    indices = sorted(run_data.get("slice_indices", []))
    if not indices:
        return []
    if max_cases <= 0 or max_cases >= len(indices):
        return indices
    positions = np.linspace(0, len(indices) - 1, max_cases)
    selected = []
    for pos in positions:
        idx = indices[int(round(pos))]
        if idx not in selected:
            selected.append(idx)
    return selected


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "run"


def _is_contiguous(indices: list[int]) -> bool:
    if not indices:
        return False
    return indices == list(range(indices[0], indices[-1] + 1))


def _overlay_mask_on_rgb(rgb: np.ndarray | None, mask: np.ndarray | None, alpha: float = 0.35) -> np.ndarray | None:
    if rgb is None:
        return None
    if mask is None:
        return rgb
    if mask.shape[:2] != rgb.shape[:2]:
        mask = _resize_to_match(mask, rgb, Image.NEAREST)
    if mask is None:
        return rgb
    out = rgb.astype(np.float32).copy()
    mask_bool = mask > 0
    color = np.array([255.0, 220.0, 0.0], dtype=np.float32)
    out[mask_bool] = out[mask_bool] * (1.0 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _patch_row_for_slice(run_data: dict, slice_num: int) -> dict | None:
    return run_data.get("patch_by_slice", {}).get(slice_num)


def _metadata_path(row: dict | None, key: str) -> Path | None:
    if not row:
        return None
    return _resolve_repo_path(row.get(key))


def _inpainted_patch_path(run_data: dict, row: dict | None) -> Path | None:
    if not row:
        return None
    slice_id = row.get("slice_id")
    if not slice_id:
        return None
    path = run_data.get("inpainted_dir") / f"{slice_id}_inpainted.png"
    return path if path.exists() else None


def _bbox_from_patch_row(row: dict | None) -> tuple[int, int, int, int] | None:
    if not row:
        return None
    try:
        return (
            int(float(row["bbox_y_min"])),
            int(float(row["bbox_x_min"])),
            int(float(row["bbox_y_max"])),
            int(float(row["bbox_x_max"])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _draw_scaled_bbox(ax, bbox: tuple[int, int, int, int] | None, scale: tuple[float, float]) -> None:
    if not bbox:
        return
    y_min, x_min, y_max, x_max = bbox
    scale_x, scale_y = scale
    rect = Rectangle(
        (x_min * scale_x, y_min * scale_y),
        (x_max - x_min) * scale_x,
        (y_max - y_min) * scale_y,
        linewidth=1.4,
        edgecolor="#00e5ff",
        facecolor="none",
    )
    ax.add_patch(rect)


def _compute_ssim_batch(orig_dir: Path, merged_dir: Path, slice_ids: list[str]) -> dict:
    """Compute SSIM(original, merged) for as many slices as possible."""
    results = {}
    if not HAVE_SKIMAGE:
        return results
    for sid in slice_ids:
        slice_num = _extract_slice_num(sid)
        if slice_num is None:
            continue

        merged_path = _find_slice_image(merged_dir, slice_num, "merged")
        orig_path = _find_slice_image(orig_dir, slice_num, "source")
        if not merged_path or not orig_path:
            continue

        merged_arr = _load_image(merged_path)
        orig_arr = _load_image(orig_path)
        if merged_arr is None or orig_arr is None:
            continue
        if merged_arr.shape != orig_arr.shape:
            merged_arr = _resize_to_match(merged_arr, orig_arr)
            if merged_arr is None:
                continue
        # compute SSIM per channel and average
        try:
            s = ssim(orig_arr, merged_arr, channel_axis=-1, data_range=255)
        except Exception:
            try:
                s = ssim(orig_arr, merged_arr, multichannel=True, data_range=255)
            except Exception:
                s = None
        if s is not None:
            results[sid] = s
    return results


# ── run data gatherer ──────────────────────────────────────────────────────


def gather_run_data(run_dir: Path, label: str = None) -> dict:
    """Collect all metrics and image paths for a single run directory."""
    run_dir = Path(run_dir)
    if label is None:
        label = _run_label_from_dir(run_dir)

    data = {
        "label": label,
        "run_dir": run_dir,
        "short_name": run_dir.name,
    }

    # manifest
    manifest = _load_json(run_dir / "run_manifest.json")
    data["manifest"] = manifest
    if manifest:
        data["volume_id"] = manifest.get("volume_id", run_dir.name)
        args = manifest.get("args", {})
        data["num_slices"] = args.get("num_slices", 0)
        data["tissue_aware"] = manifest.get("tissue_aware_mask", False) or \
            args.get("tissue_aware_mask", False)
        data["mask_radius"] = args.get("mask_radius", "N/A")
        data["mask_shape"] = args.get("mask_shape", "N/A")
        data["strict_bbox"] = args.get("strict_bbox", False)
        data["lora_weights"] = str(args.get("lora_weights", "N/A"))
    else:
        data["volume_id"] = run_dir.name
        data["num_slices"] = 0
        data["tissue_aware"] = False

    # coherence metrics
    cm = _load_json(run_dir / "coherence_metrics.json")
    data["coherence"] = cm

    # mask diagnostics (tissue-aware format)
    tad = _load_json(run_dir / "tissue_aware_mask_diagnostics.json")
    if tad:
        data["mask_diag"] = tad
        data["mask_diag_type"] = "tissue_aware"
        data["per_slice_mask"] = tad.get("per_slice", [])
    else:
        # fallback: mask_tissue_overlap_report
        tor = _load_json(run_dir / "mask_tissue_overlap_report.json")
        if tor:
            data["mask_diag"] = tor
            data["mask_diag_type"] = "overlap_report"
            # convert to uniform per_slice format
            per_slice = []
            for r in tor.get("results", []):
                sid = r.get("slice_id", "")
                num_match = re.search(r"_(\d+)$", sid)
                si = int(num_match.group(1)) if num_match else 0
                per_slice.append({
                    "slice_index": si,
                    "filename": f"slice_{si:04d}.png",
                    "tissue_overlap_frac": r.get("tissue_overlap_frac", 0.0),
                    "white_void_frac": r.get("white_void_frac", 0.0),
                    "black_void_frac": r.get("black_void_frac", 0.0),
                })
            data["per_slice_mask"] = per_slice
        else:
            data["mask_diag"] = None
            data["per_slice_mask"] = []

    # image directories
    data["merged_dir"] = run_dir / "merged/slices"
    data["inpainted_dir"] = run_dir / "inpainted/images"
    data["masks_dir"] = run_dir / "masks"
    data["source_dir"] = _source_dir_from_manifest(manifest)
    data["source_shape"] = _source_shape_for_dir(data["source_dir"])
    try:
        data["source_key"] = str(data["source_dir"].resolve())
    except Exception:
        data["source_key"] = str(data["source_dir"])

    patch_metadata_csv = run_dir / "patches/metadata/patches_metadata.csv"
    patch_rows = _load_csv(patch_metadata_csv) or []
    patch_by_slice = {}
    for row in patch_rows:
        slice_num = _extract_slice_num(row.get("slice_id", ""))
        if slice_num is not None:
            patch_by_slice[slice_num] = row
    data["patch_metadata_csv"] = patch_metadata_csv
    data["patch_rows"] = patch_rows
    data["patch_by_slice"] = patch_by_slice

    # Collect slice IDs (from merged files)
    merged_dir = data["merged_dir"]
    slice_ids = []
    slice_indices = []
    if merged_dir.is_dir():
        for f in sorted(merged_dir.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                sid = f.stem
                sid = sid.replace("_edited", "")
                slice_ids.append(sid)
                slice_num = _extract_slice_num(sid)
                if slice_num is not None:
                    slice_indices.append(slice_num)
    data["slice_ids"] = slice_ids
    data["slice_indices"] = sorted(set(slice_indices))
    data["slice_indices_contiguous"] = _is_contiguous(data["slice_indices"])

    # compute SSIM against original source slices
    source_dir = data["source_dir"]
    if HAVE_SKIMAGE and source_dir.is_dir():
        data["ssim_original_vs_merged"] = _compute_ssim_batch(
            source_dir, merged_dir, slice_ids
        )
    else:
        data["ssim_original_vs_merged"] = {}

    return data


def auto_select_runs(include_lowres: bool = False) -> list[Path]:
    """Auto-select latest HR run by default; low-res runs require opt-in."""
    selected = []
    paths = []

    # Default: latest high-resolution melanoma tile-first run only.
    latest = _latest_subdir(DEFAULT_SEARCH_PATHS["tissue_aware_hr"])
    if latest:
        paths.append(latest)
    if not include_lowres:
        return paths

    # 1. base_strict – want the latest NON-tissue-aware run
    latest = _latest_subdir(DEFAULT_SEARCH_PATHS["base_strict"],
                            exclude_tissue_aware=True)
    if latest:
        paths.append(latest)

    # 2. tissue_aware – want the latest tissue-aware run
    latest = _latest_subdir(DEFAULT_SEARCH_PATHS["tissue_aware"])
    if latest:
        paths.append(latest)

    # 3. melanoma_strict (optional)
    latest = _latest_subdir(DEFAULT_SEARCH_PATHS["melanoma_strict"])
    if latest:
        paths.append(latest)

    # deduplicate by realpath
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            selected.append(p)
    return selected


# ── visualization routines ────────────────────────────────────────────────


def plot_01_tissue_overlap_profile(runs_data: list[dict], out_dir: Path):
    """Bar chart of tissue_overlap_frac per slice for each run."""
    n = len(runs_data)
    if n == 0:
        return
    fig, ax = plt.subplots(figsize=(max(10, 3 * n), 5))

    for i, rd in enumerate(runs_data):
        per_slice = rd.get("per_slice_mask", [])
        if not per_slice:
            continue
        slices = [p.get("slice_index", 0) for p in per_slice]
        overlaps = [p.get("tissue_overlap_frac", 0) for p in per_slice]
        offset = i * 0.25
        width = 0.2
        ax.bar([s + offset for s in slices], overlaps, width * 0.9,
               label=rd["label"], alpha=0.85)

    ax.set_xlabel("Slice Index")
    ax.set_ylabel("Tissue Overlap Fraction")
    ax.set_title("Tissue Overlap per Slice by Run")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "01_tissue_overlap_profile.png", dpi=150)
    plt.close(fig)


def plot_02_void_composition_per_slice(runs_data: list[dict], out_dir: Path):
    """Stacked bar: tissue vs white void vs black void per slice, per run."""
    n = len(runs_data)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    axes = axes[0]

    for i, rd in enumerate(runs_data):
        ax = axes[i]
        per_slice = rd.get("per_slice_mask", [])
        if not per_slice:
            ax.text(0.5, 0.5, "No mask diagnostic data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(rd["label"])
            continue

        slices = [p.get("slice_index", 0) for p in per_slice]
        tissue = [p.get("tissue_overlap_frac", 0) for p in per_slice]
        white = [p.get("white_void_frac", 0) for p in per_slice]
        black = [p.get("black_void_frac", 0) for p in per_slice]

        ax.bar(slices, tissue, label="Tissue", color="#2ca02c", alpha=0.8)
        ax.bar(slices, white, bottom=tissue, label="White Void", color="#ff7f0e", alpha=0.8)
        bottom2 = [t + w for t, w in zip(tissue, white)]
        ax.bar(slices, black, bottom=bottom2, label="Black Void", color="#d62728", alpha=0.8)

        ax.set_xlabel("Slice Index")
        ax.set_ylabel("Fraction of Mask Area")
        ax.set_title(f"Mask Composition — {rd['label']}")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "02_void_composition_per_slice.png", dpi=150)
    plt.close(fig)


def plot_03_z_coherence_comparison(runs_data: list[dict], out_dir: Path):
    """Line plot of per-slice z-gradient smoothness (mean_abs_gradient per slice)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for rd in runs_data:
        cm = rd.get("coherence")
        if cm is None:
            continue
        per_slice = cm.get("z_gradient_smoothness", {}).get("per_slice_mean")
        if not per_slice:
            continue
        x = list(range(len(per_slice)))
        ax.plot(x, per_slice, marker="o", markersize=3, label=rd["label"], alpha=0.85,
                linewidth=1.5)

    ax.set_xlabel("Slice Index (z)")
    ax.set_ylabel("Mean |Gradient| (intensity units)")
    ax.set_title("Z-Gradient Smoothness per Slice")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "03_z_coherence_comparison.png", dpi=150)
    plt.close(fig)


def plot_04_overlap_vs_coherence_scatter(runs_data: list[dict], out_dir: Path):
    """Scatter: tissue_overlap_frac vs z_gradient per slice, colored by run."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(runs_data)))

    for i, rd in enumerate(runs_data):
        per_slice_mask = rd.get("per_slice_mask", [])
        cm = rd.get("coherence")
        if not per_slice_mask or cm is None:
            continue
        z_grad = cm.get("z_gradient_smoothness", {}).get("per_slice_mean")
        if not z_grad:
            continue
        # align by slice_index
        overlap_by_idx = {}
        for p in per_slice_mask:
            idx = p.get("slice_index")
            overlap_by_idx[idx] = p.get("tissue_overlap_frac", 0)

        xs, ys = [], []
        for idx, gval in enumerate(z_grad):
            if idx in overlap_by_idx:
                xs.append(overlap_by_idx[idx])
                ys.append(gval)
        if xs:
            ax.scatter(xs, ys, c=[colors[i]], label=rd["label"], alpha=0.7, s=30,
                       edgecolors="k", linewidths=0.3)

    ax.set_xlabel("Tissue Overlap Fraction")
    ax.set_ylabel("Mean |Gradient| (z-smoothness)")
    ax.set_title("Overlap vs Z-Coherence per Slice")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "04_overlap_vs_coherence_scatter.png", dpi=150)
    plt.close(fig)


def plot_05_ssim_boxplot(runs_data: list[dict], out_dir: Path):
    """Box plot of adjacent SSIM values + per-run original-vs-merged SSIM."""
    fig, ax = plt.subplots(figsize=(max(6, len(runs_data) * 2.5), 5))

    positions_adj = []
    labels_adj = []
    data_adj = []
    positions_om = []
    data_om = []

    for i, rd in enumerate(runs_data):
        cm = rd.get("coherence")
        if cm and "adjacent_ssim" in cm:
            vals = cm["adjacent_ssim"].get("values", [])
            if vals:
                pos = i * 2 + 1
                positions_adj.append(pos)
                data_adj.append(vals)
                labels_adj.append(f"{rd['label']}\n(adjacent)")

        # original-vs-merged SSIM
        om_vals = list(rd.get("ssim_original_vs_merged", {}).values())
        if om_vals:
            pos = i * 2 + 2
            positions_om.append(pos)
            data_om.append(om_vals)

    all_positions = []
    all_data = []
    all_labels = []

    if data_adj:
        all_positions.extend(positions_adj)
        all_data.extend(data_adj)
        # Build labels
        for i, rd in enumerate(runs_data):
            all_labels.append(f"{rd['label']}\n(adjacent SSIM)")

    if data_om:
        offset = max(all_positions) + 1 if all_positions else 1
        for j, (pos, vals) in enumerate(zip(positions_om, data_om)):
            new_pos = offset + j
            all_positions.append(new_pos)
            all_data.append(vals)
            all_labels.append(f"{runs_data[j]['label']}\n(orig vs merged)")

    if not all_data:
        ax.text(0.5, 0.5, "No SSIM data available", ha="center", va="center",
                transform=ax.transAxes)
    else:
        bp = ax.boxplot(all_data, positions=all_positions, patch_artist=True, widths=0.5)
        # Color
        colors = plt.cm.Set2(np.linspace(0, 1, len(all_data)))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels, fontsize=8)
        ax.set_ylabel("SSIM")
        ax.set_title("SSIM Comparison Across Runs")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "05_ssim_boxplot.png", dpi=150)
    plt.close(fig)


def plot_06_before_mask_inpaint_merged_grid(runs_data: list[dict], out_dir: Path):
    """Grid: original / mask / merged full slice / diff for one common slice index."""
    n = len(runs_data)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(4 * 4, 3 * n + 1))
    if n == 1:
        axes = axes[np.newaxis, :]

    common_slice = _select_common_slice_index(runs_data)

    for i, rd in enumerate(runs_data):
        slice_indices = rd.get("slice_indices", [])
        if common_slice is None and not slice_indices:
            for j in range(4):
                axes[i, j].text(0.5, 0.5, "No images", ha="center", va="center",
                                transform=axes[i, j].transAxes)
                axes[i, j].set_title(rd["label"] if j == 0 else "")
            continue

        slice_num = common_slice
        if slice_num not in slice_indices and slice_indices:
            slice_num = slice_indices[len(slice_indices) // 2]

        orig_arr = _load_source_image(rd, slice_num)
        _show_or_text(axes[i, 0], orig_arr, "Original")

        mask_arr = _load_mask_array(rd, slice_num)
        _show_or_text(axes[i, 1], mask_arr, "Mask", cmap="gray")

        merged_arr = _load_merged_image(rd, slice_num)
        _show_or_text(axes[i, 2], merged_arr, "Merged Full Slice")

        diff_arr = None
        if orig_arr is not None and merged_arr is not None:
            merged_for_diff = _resize_to_match(merged_arr, orig_arr)
            if merged_for_diff is not None:
                diff_arr = np.abs(orig_arr.astype(np.float32) - merged_for_diff.astype(np.float32)).mean(axis=2)
        _show_or_text(axes[i, 3], diff_arr, "|Diff| Original vs Merged", cmap="hot")

        # row label
        axes[i, 0].set_ylabel(f"{rd['label']}\nslice_{slice_num:04d}", fontsize=9,
                              fontweight="bold")

    title_slice = f"slice_{common_slice:04d}" if common_slice is not None else "representative slice"
    fig.suptitle(f"Qualitative Full-Slice Comparison ({title_slice})", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "06_before_mask_inpaint_merged_grid.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _show_or_text(ax, arr, title, cmap=None):
    """Show image array or placeholder text."""
    if arr is not None:
        if len(arr.shape) == 2:
            ax.imshow(arr, cmap=cmap or "gray")
        else:
            ax.imshow(arr)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_07_diff_heatmaps(runs_data: list[dict], out_dir: Path):
    """
    Difference heatmaps (|original - merged|) for a representative slice per run.

    Shows three panels per run:
      left  — full diff heatmap (raw |orig-merged|, vmax auto-scaled at 99th %ile)
      right — thresholded overlay (diff > 3 shown in colour on dimmed original,
              with mask boundary in white)
    """
    n = len(runs_data)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)
    common_slice = _select_common_slice_index(runs_data)

    for i, rd in enumerate(runs_data):
        ax_raw = axes[0, i]
        ax_ov  = axes[1, i]
        slice_indices = rd.get("slice_indices", [])
        if common_slice is None and not slice_indices:
            for ax in (ax_raw, ax_ov):
                ax.text(0.5, 0.5, "No images", ha="center", va="center",
                        transform=ax.transAxes)
            ax_raw.set_title(rd["label"])
            continue

        slice_num = common_slice
        if slice_num not in slice_indices and slice_indices:
            slice_num = slice_indices[len(slice_indices) // 2]

        orig_arr = _load_source_image(rd, slice_num)
        merged_arr = _load_merged_image(rd, slice_num)
        mask_arr = _load_mask_array(rd, slice_num)

        if orig_arr is None or merged_arr is None:
            for ax in (ax_raw, ax_ov):
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(rd["label"])
                ax.axis("off")
            continue

        if merged_arr.shape != orig_arr.shape:
            merged_arr = _resize_to_match(merged_arr, orig_arr)

        if merged_arr is None or merged_arr.shape != orig_arr.shape:
            ax_raw.text(0.5, 0.5, "Shape mismatch", ha="center", va="center",
                        transform=ax_raw.transAxes)
            ax_raw.set_title(rd["label"])
            ax_raw.axis("off")
            ax_ov.axis("off")
            continue

        diff = np.abs(orig_arr.astype(np.float32) - merged_arr.astype(np.float32))
        diff_gray = diff.mean(axis=2)

        # Ensure mask_arr dimensions match orig_arr if present
        mask_resized = None
        if mask_arr is not None:
            mask_resized = _resize_to_match(mask_arr, orig_arr, Image.NEAREST)

        # ── Panel 1: raw diff with auto-scaled vmax (99th percentile) ──
        vmax_raw = max(5.0, float(np.percentile(diff_gray, 99)))
        im = ax_raw.imshow(diff_gray, cmap="hot", vmin=0, vmax=vmax_raw)
        fig.colorbar(im, ax=ax_raw, fraction=0.046, pad=0.04, label="|diff|")
        # annotate stats
        stats_text = (
            f"mean={diff_gray.mean():.2f}  "
            f"p99={vmax_raw:.0f}  "
            f"max={diff_gray.max():.0f}"
        )
        ax_raw.text(0.02, 0.98, stats_text, transform=ax_raw.transAxes,
                    fontsize=7, color="white", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
        ax_raw.set_title(f"|Diff| raw — {rd['label']} slice_{slice_num:04d}", fontsize=9)
        ax_raw.axis("off")

        # ── Panel 2: thresholded overlay on dimmed original ──
        # Dim original for background
        bg = (orig_arr * 0.35).astype(np.uint8)
        ax_ov.imshow(bg)

        # Overlay diff > 3 threshold with "hot" colormap
        thresh = 3.0
        mask_overlay = diff_gray > thresh
        if mask_overlay.any():
            # Create RGBA overlay: only show where diff > thresh
            normed = np.clip(diff_gray / max(30.0, diff_gray.max()), 0, 1)
            hot_cmap = plt.colormaps["hot"]
            overlay_rgba = hot_cmap(normed)
            overlay_rgba[~mask_overlay, :] = [0, 0, 0, 0]  # transparent
            ax_ov.imshow(overlay_rgba)

        # Draw mask boundary in white
        if mask_resized is not None:
            from scipy.ndimage import binary_dilation
            mask_binary = mask_resized > 0
            if mask_binary.any():
                # dilate slightly so the outline is visible
                outline = binary_dilation(mask_binary, iterations=2) & ~mask_binary
                ax_ov.contour(outline, levels=[0.5], colors="white", linewidths=1.0)

        # Stats annotation
        pct_over = (diff_gray > thresh).sum() / diff_gray.size * 100
        pct_over_mask = "?"
        if mask_resized is not None and mask_resized.any():
            in_mask = diff_gray[mask_resized > 0]
            pct_over_mask = f"{((in_mask > thresh).sum() / max(1, in_mask.size) * 100):.0f}%"
        ov_text = (
            f"|diff| > {thresh:.0f}: {pct_over:.1f}% of image  "
            f"(mask: {pct_over_mask})"
        )
        ax_ov.text(0.02, 0.98, ov_text, transform=ax_ov.transAxes,
                   fontsize=7, color="white", va="top",
                   bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
        ax_ov.set_title(f"|Diff| thresholded (>{thresh:.0f}) + mask — {rd['label']}", fontsize=9)
        ax_ov.axis("off")

    title_slice = f"slice_{common_slice:04d}" if common_slice is not None else "representative slice"
    fig.suptitle(f"Diff Heatmaps ({title_slice}): raw (top) vs thresholded overlay with mask boundary (bottom)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "07_diff_heatmaps.png", dpi=150)
    plt.close(fig)


def plot_08_summary_table(runs_data: list[dict], out_dir: Path):
    """Render a summary statistics table as a matplotlib figure."""
    if not runs_data:
        return

    rows = []
    for rd in runs_data:
        cm = rd.get("coherence") or {}
        adj = cm.get("adjacent_ssim", {})
        grad = cm.get("z_gradient_smoothness", {})
        md = rd.get("mask_diag") or {}

        row = {
            "Run": rd["label"],
            "Source": rd.get("source_dir", Path("?")).name,
            "Slices": str(rd.get("num_slices", "?")),
            "Tissue-Aware": "Yes" if rd.get("tissue_aware") else "No",
            "Mask Shape": str(rd.get("mask_shape", "?")),
            "Strict BBox": "Yes" if rd.get("strict_bbox") else "No",
            "SSIM Mean": f"{adj.get('mean', 0):.4f}",
            "SSIM Std": f"{adj.get('std', 0):.4f}",
            "SSIM Min": f"{adj.get('min', 0):.4f}",
            "Z-Grad Mean": f"{grad.get('mean_abs_gradient', 0):.2f}",
            "Z-Grad Std": f"{grad.get('std_abs_gradient', 0):.2f}",
        }

        # mask summary if available
        if md:
            if "summary" in md:
                s = md["summary"]
                row["Mean Tissue Overlap"] = f"{s.get('mean_tissue_overlap', 0):.3f}"
                row["Mean Black Void"] = f"{s.get('mean_black_void', 0):.3f}"
            elif "aggregate" in md:
                agg = md["aggregate"]
                to = agg.get("tissue_overlap_frac", {}).get("mean", "N/A")
                bv = agg.get("black_void_frac", {}).get("mean", "N/A")
                row["Mean Tissue Overlap"] = f"{to}"
                row["Mean Black Void"] = f"{bv}"
        else:
            row["Mean Tissue Overlap"] = "N/A"
            row["Mean Black Void"] = "N/A"

        rows.append(row)

    if not rows:
        return

    # Convert to table
    col_labels = list(rows[0].keys())
    cell_text = [[r.get(h, "") for h in col_labels] for r in rows]

    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 1.5), 2 + 0.4 * len(rows)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#ecf0f1")
            else:
                cell.set_facecolor("white")

    ax.set_title("Summary Comparison Table", fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(out_dir / "08_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_09_tile_inpaint_cases(runs_data: list[dict], out_dir: Path, num_cases: int = 8):
    """Show the exact extracted tile + mask sent to inpainting and the inpainted tile output."""
    rows = []
    for rd in runs_data:
        for slice_num in _select_case_indices(rd, num_cases):
            rows.append((rd, slice_num))

    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 4, figsize=(16, max(3.0, 2.8 * len(rows))))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (rd, slice_num) in enumerate(rows):
        patch_row = _patch_row_for_slice(rd, slice_num)
        source_path = _find_slice_image(rd.get("source_dir"), slice_num, "source")
        merged_path = _find_slice_image(rd.get("merged_dir"), slice_num, "merged")
        patch_path = _metadata_path(patch_row, "patch_image")
        patch_mask_path = _metadata_path(patch_row, "patch_mask")
        inpainted_path = _inpainted_patch_path(rd, patch_row)

        source_thumb, source_scale = _load_image_thumbnail(source_path, max_dim=900)
        patch_img = _load_image(patch_path)
        patch_mask = None
        if patch_mask_path:
            try:
                patch_mask = np.array(Image.open(patch_mask_path).convert("L"))
            except Exception:
                patch_mask = None
        patch_overlay = _overlay_mask_on_rgb(patch_img, patch_mask)
        inpainted_img = _load_image(inpainted_path)
        merged_thumb, _ = _load_image_thumbnail(merged_path, max_dim=900)
        bbox = _bbox_from_patch_row(patch_row)

        ax = axes[row_idx, 0]
        _show_or_text(ax, source_thumb, "Full HR Source + Tile BBox")
        _draw_scaled_bbox(ax, bbox, source_scale)

        _show_or_text(axes[row_idx, 1], patch_overlay, "Tile Sent to Inpaint + Mask")
        _show_or_text(axes[row_idx, 2], inpainted_img, "Inpainted Tile Output")
        _show_or_text(axes[row_idx, 3], merged_thumb, "Merged Full Slice")

        axes[row_idx, 0].set_ylabel(
            f"{rd['label']}\nslice_{slice_num:04d}", fontsize=8, fontweight="bold",
        )

    fig.suptitle("Tile-Level Inpainting QC: Input Tile With Mask vs Inpainted Tile", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "09_tile_mask_inpaint_cases.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sequence_contact_sheet(
    runs_data: list[dict],
    out_dir: Path,
    *,
    kind: str,
    basename: str,
    title_prefix: str,
    max_cols: int = 5,
) -> list[Path]:
    """Render all ordered full slices used by each run as thumbnail contact sheets."""
    outputs = []
    for rd in runs_data:
        indices = sorted(rd.get("slice_indices", []))
        if not indices:
            continue

        cols = min(max_cols, len(indices))
        rows = int(np.ceil(len(indices) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.6 * rows))
        axes_arr = np.atleast_1d(axes).reshape(rows, cols)

        for ax in axes_arr.flat:
            ax.axis("off")

        for pos, slice_num in enumerate(indices):
            ax = axes_arr.flat[pos]
            directory = rd.get("source_dir") if kind == "source" else rd.get("merged_dir")
            role = "source" if kind == "source" else "merged"
            img_path = _find_slice_image(directory, slice_num, role)
            thumb, _ = _load_image_thumbnail(img_path, max_dim=900)
            _show_or_text(ax, thumb, f"slice_{slice_num:04d}")

        source_label = rd.get("source_dir", Path("?")).name
        contiguous = "contiguous" if rd.get("slice_indices_contiguous") else "NOT CONTIGUOUS"
        fig.suptitle(
            f"{title_prefix}: {rd['label']} ({contiguous}, source={source_label})",
            fontsize=12,
            y=1.01,
        )
        fig.tight_layout()

        suffix = "" if len(runs_data) == 1 else f"_{_safe_slug(rd['label'])}"
        out_path = out_dir / f"{basename}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)

    return outputs


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison visualizations for 3D volume inpainting runs."
    )
    parser.add_argument(
        "--run-dirs", nargs="*", default=None,
        help="Paths to run directories. If omitted, auto-select latest."
    )
    parser.add_argument(
        "--labels", nargs="*", default=None,
        help="Labels for runs (one per --run-dirs)."
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for PNGs. Default: /tmp/melanoma_compare_plots_<ts>"
    )
    parser.add_argument(
        "--num-cases", type=int, default=8,
        help="Number of evenly spaced tile-level cases to show in plot 09 (0 = all)."
    )
    parser.add_argument(
        "--allow-lowres-source", action="store_true",
        help="Allow explicit low-resolution source runs. Default is HR-only."
    )
    parser.add_argument(
        "--include-lowres-auto", action="store_true",
        help="When --run-dirs is omitted, include legacy low-res runs in auto-selection."
    )
    args = parser.parse_args()

    # ── resolve run directories ──────────────────────────────────────────
    if args.run_dirs:
        run_paths = [Path(p) for p in args.run_dirs]
    else:
        run_paths = auto_select_runs(include_lowres=args.include_lowres_auto)
        if not run_paths:
            print("ERROR: No run directories found. Use --run-dirs to specify.")
            sys.exit(1)
        print(f"Auto-selected {len(run_paths)} run director{'y' if len(run_paths)==1 else 'ies'}:")

    # resolve labels
    if args.labels:
        if len(args.labels) != len(run_paths):
            print(f"WARNING: {len(args.labels)} labels provided but {len(run_paths)} runs. "
                  "Ignoring labels, using auto-derived names.")
            labels = None
        else:
            labels = args.labels
    else:
        labels = None

    # ── gather data ───────────────────────────────────────────────────────
    missing_files = {p.name: [] for p in run_paths}
    runs_data = []
    for i, rp in enumerate(run_paths):
        label = labels[i] if labels else None
        print(f"  [{i}] {rp.name} ...")
        rd = gather_run_data(rp, label=label)
        runs_data.append(rd)

        # track missing
        if not rd.get("coherence"):
            missing_files[rp.name].append("coherence_metrics.json")
        if not rd.get("mask_diag"):
            missing_files[rp.name].append("tissue_aware_mask_diagnostics.json/mask_tissue_overlap_report.json")
        if not rd.get("manifest"):
            missing_files[rp.name].append("run_manifest.json")
        if not rd.get("merged_dir", "").is_dir():
            missing_files[rp.name].append("merged/slices/")
        if not rd.get("inpainted_dir", "").is_dir():
            missing_files[rp.name].append("inpainted/images/")
        if not rd.get("masks_dir", "").is_dir():
            missing_files[rp.name].append("masks/")
        om = rd.get("ssim_original_vs_merged", {})
        if not om and HAVE_SKIMAGE:
            missing_files[rp.name].append("source slices (SSIM not computed)")

    print()

    if not args.allow_lowres_source:
        lowres = []
        for rd in runs_data:
            shape = rd.get("source_shape")
            if not _is_hr_source(rd):
                lowres.append((rd["label"], rd.get("source_dir"), shape))
        if lowres:
            print("ERROR: Low-resolution source runs are blocked by default for melanoma QC plots.")
            for label, source_dir, shape in lowres:
                shape_text = f"{shape[0]}x{shape[1]}" if shape else "unknown"
                print(f"  {label}: source={source_dir} shape={shape_text}")
            print("Use --allow-lowres-source only for an explicit legacy/debug comparison.")
            sys.exit(2)

    for rd in runs_data:
        if rd.get("slice_indices") and not rd.get("slice_indices_contiguous"):
            print(f"  [!] {rd['label']}: slice indices are NOT contiguous: {rd.get('slice_indices')}")

    for rd in runs_data:
        if rd.get("source_shape"):
            h, w = rd["source_shape"]
            print(f"  [source] {rd['label']}: {rd.get('source_dir')} ({h}x{w})")

    source_groups = sorted({rd.get("source_key", "?") for rd in runs_data})
    if len(source_groups) > 1:
        print("  [!] Runs use multiple source slice directories.")
        print("      Qualitative/diff plots now use each run's own source images, but direct")
        print("      visual comparisons are cleanest when generated one source group at a time.")
        for src in source_groups:
            print(f"      source: {src}")
        print()

    # report missing
    for rname, missing in missing_files.items():
        if missing:
            print(f"  [!] {rname}: missing {', '.join(missing)}")
    print()

    # ── output directory ──────────────────────────────────────────────────
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"/tmp/melanoma_compare_plots_{ts}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ── generate plots ────────────────────────────────────────────────────
    plot_01_tissue_overlap_profile(runs_data, out_dir)
    print("  ✓ 01_tissue_overlap_profile.png")

    plot_02_void_composition_per_slice(runs_data, out_dir)
    print("  ✓ 02_void_composition_per_slice.png")

    plot_03_z_coherence_comparison(runs_data, out_dir)
    print("  ✓ 03_z_coherence_comparison.png")

    plot_04_overlap_vs_coherence_scatter(runs_data, out_dir)
    print("  ✓ 04_overlap_vs_coherence_scatter.png")

    plot_05_ssim_boxplot(runs_data, out_dir)
    print("  ✓ 05_ssim_boxplot.png")

    plot_06_before_mask_inpaint_merged_grid(runs_data, out_dir)
    print("  ✓ 06_before_mask_inpaint_merged_grid.png")

    plot_07_diff_heatmaps(runs_data, out_dir)
    print("  ✓ 07_diff_heatmaps.png")

    plot_08_summary_table(runs_data, out_dir)
    print("  ✓ 08_summary_table.png")

    plot_09_tile_inpaint_cases(runs_data, out_dir, num_cases=args.num_cases)
    print("  ✓ 09_tile_mask_inpaint_cases.png")

    source_sequence = _plot_sequence_contact_sheet(
        runs_data,
        out_dir,
        kind="source",
        basename="10_sequential_source_slices_used",
        title_prefix="Sequential HR Source Full Slices Used",
    )
    for path in source_sequence:
        print(f"  ✓ {path.name}")

    merged_sequence = _plot_sequence_contact_sheet(
        runs_data,
        out_dir,
        kind="merged",
        basename="11_sequential_merged_slices_used_for_metrics",
        title_prefix="Sequential Merged Full Slices Used for Metrics",
    )
    for path in merged_sequence:
        print(f"  ✓ {path.name}")

    # ── report ────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("GENERATED FILES:")
    for f in sorted(out_dir.iterdir()):
        if f.suffix == ".png":
            print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
    print()
    print(f"Total: {len(list(out_dir.glob('*.png')))} PNGs in {out_dir}")
    print("=" * 60)

    # Print comparison info
    print()
    print("Runs compared:")
    for rd in runs_data:
        cm = rd.get("coherence") or {}
        adj = cm.get("adjacent_ssim", {})
        grad = cm.get("z_gradient_smoothness", {})
        md = rd.get("mask_diag") or {}
        n_slices = rd.get("num_slices", 0)
        ta = "Y" if rd.get("tissue_aware") else "N"

        # mask overlap stats
        if md:
            if "summary" in md:
                mt = md["summary"].get("mean_tissue_overlap", "?")
                mb = md["summary"].get("mean_black_void", "?")
            elif "aggregate" in md:
                mt = md["aggregate"].get("tissue_overlap_frac", {}).get("mean", "?")
                mb = md["aggregate"].get("black_void_frac", {}).get("mean", "?")
            else:
                mt, mb = "?", "?"
        else:
            mt, mb = "?", "?"

        print(f"  {rd['label']:30s}  slices={n_slices}  "
              f"tissue_aware={ta}  "
              f"ssim_mean={adj.get('mean', 0):.4f}  "
              f"z_grad={grad.get('mean_abs_gradient', 0):.2f}  "
              f"tissue_overlap={mt}  "
              f"black_void={mb}  "
              f"source={rd.get('source_dir', Path('?')).name}")


if __name__ == "__main__":
    main()
