#!/usr/bin/env python
"""
Visualise a 3D temperature array from :py:mod:`thermal_solve`.

Reads a ``temperature_final.npy`` file (or any compatible 3D numpy array),
computes summary statistics, and optionally saves a 2D slice as a PNG image.

If matplotlib is not installed, the script still writes the summary JSON
and prints stats to stdout.

Input:
  --temperature-npy  — 3D array [°C] (Z, Y, X) from thermal_solve output.

Outputs (in ``--output-dir``):
  - If ``--slice-axis`` is given:
      ``slice_Z<idx>_<axis>.png``  (requires matplotlib)
  - ``thermal_visualise_summary.json`` — stats + slice metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional matplotlib for PNG rendering
# ---------------------------------------------------------------------------
_HAS_MPL = False
try:
    import matplotlib.pyplot as plt  # noqa: F401

    _HAS_MPL = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AXIS_NAMES = {0: "Z (axial / slice)", 1: "Y (coronal / row)", 2: "X (sagittal / column)"}


def _load_temperature(path: str) -> np.ndarray:
    """Load a 3D temperature array from a .npy file."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Temperature file not found: {path}")

    arr = np.load(str(p))
    if arr.ndim < 2:
        raise SystemExit(
            f"Temperature array must be at least 2D, got shape {arr.shape}"
        )
    return arr


def _extract_slice(
    arr: np.ndarray, axis: int, index: int
) -> np.ndarray:
    """Extract a 2D slice along a given axis.

    If the index is out of range (e.g., -1), clamps to the nearest valid index.
    """
    if index < 0 or index >= arr.shape[axis]:
        clamped = max(0, min(index, arr.shape[axis] - 1))
        print(f"  [WARN] Slice index {index} out of range [0, {arr.shape[axis] - 1}] "
              f"for axis {axis}; using {clamped}")
        index = clamped

    return np.take(arr, index, axis=axis)


def _save_slice_png(
    slice_2d: np.ndarray,
    output_path: str,
    axis: int,
    index: int,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Save a 2D temperature slice as a PNG using matplotlib.

    Args:
        slice_2d: 2D array of temperature values.
        output_path: Path for the output PNG.
        axis: Axis along which the slice was taken (0/1/2).
        index: Slice index along that axis.
        vmin, vmax: Color scale bounds.  If None, use data min/max.
    """
    if not _HAS_MPL:
        print("  [SKIP] matplotlib not available; skipping PNG export.")
        return

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    if vmin is None:
        vmin = float(slice_2d.min())
    if vmax is None:
        vmax = float(slice_2d.max())

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(
        slice_2d,
        cmap="inferno",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.set_title(
        f"Temperature slice — {_AXIS_NAMES.get(axis, f'axis {axis}')} "
        f"index={index}"
    )
    ax.set_xlabel("X (columns)" if axis != 2 else "Y (rows)")
    ax.set_ylabel("Y (rows)" if axis != 0 else "Z (slices)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (°C)")

    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved slice PNG: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Visualise a 3D temperature array — print summary statistics and "
            "optionally export a 2D slice as a PNG overlay."
        ),
    )
    ap.add_argument(
        "--temperature-npy",
        required=True,
        help="Path to temperature_final.npy (3D float array in °C).",
    )
    ap.add_argument(
        "--output-dir",
        default="./thermal_visualise_output",
        help="Output directory for visualisation artifacts "
        "(default: ./thermal_visualise_output).",
    )
    ap.add_argument(
        "--slice-axis",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Axis to slice along for PNG export: 0=Z (axial), 1=Y (coronal), "
        "2=X (sagittal).  If omitted, only JSON summary is written.",
    )
    ap.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Slice index along the chosen axis.  If omitted, defaults to "
        "the mid-point of that axis.",
    )
    return ap


def main() -> None:
    ap = _build_cli()
    args = ap.parse_args()

    # --- Load ---
    print(f"Loading temperature array: {args.temperature_npy}")
    arr = _load_temperature(args.temperature_npy)
    print(f"  Shape: {arr.shape}, dtype={arr.dtype}")

    # --- Compute statistics ---
    flat = arr.ravel()
    stats: Dict[str, Any] = {
        "shape": list(arr.shape),
        "ndim": arr.ndim,
        "dtype": str(arr.dtype),
        "min_degC": float(round(arr.min(), 4)),
        "max_degC": float(round(arr.max(), 4)),
        "mean_degC": float(round(arr.mean(), 4)),
        "std_degC": float(round(arr.std(), 4)),
        "median_degC": float(round(np.median(flat), 4)),
        "p1_degC": float(round(np.percentile(flat, 1), 4)),
        "p99_degC": float(round(np.percentile(flat, 99), 4)),
    }

    # --- Optional slice ---
    slice_metadata: Dict[str, Any] = {}
    if args.slice_axis is not None:
        axis = args.slice_axis
        if axis < 0 or axis >= arr.ndim:
            raise SystemExit(
                f"slice-axis {axis} is out of range for {arr.ndim}D array. "
                f"Valid: 0..{arr.ndim - 1}"
            )

        index = args.slice_index
        if index is None:
            index = arr.shape[axis] // 2

        slice_2d = _extract_slice(arr, axis, index)

        slice_metadata = {
            "axis": axis,
            "axis_name": _AXIS_NAMES.get(axis, f"axis_{axis}"),
            "index": index,
            "shape": list(slice_2d.shape),
            "min_degC": float(round(slice_2d.min(), 4)),
            "max_degC": float(round(slice_2d.max(), 4)),
            "mean_degC": float(round(slice_2d.mean(), 4)),
        }

        # Save PNG
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        png_name = f"slice_{axis}_{index:04d}.png"
        png_path = out / png_name

        if _HAS_MPL:
            _save_slice_png(
                slice_2d,
                str(png_path),
                axis=axis,
                index=index,
                vmin=stats["min_degC"],
                vmax=stats["max_degC"],
            )
        else:
            print("  [SKIP] matplotlib not available; no PNG generated.")

    # --- Write summary JSON ---
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "source_temperature_npy": str(Path(args.temperature_npy).resolve()),
        "output_dir": str(out_path.resolve()),
        "volume_stats": stats,
    }
    if slice_metadata:
        summary["slice"] = slice_metadata

    json_out = out_path / "thermal_visualise_summary.json"
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Wrote summary JSON: {json_out}")

    # --- Print summary ---
    print(f"\n=== Temperature Visualisation Summary ===")
    print(f"  Shape            : {stats['shape']}")
    print(f"  Min              : {stats['min_degC']} °C")
    print(f"  Max              : {stats['max_degC']} °C")
    print(f"  Mean             : {stats['mean_degC']} °C")
    print(f"  Std              : {stats['std_degC']} °C")
    print(f"  Median           : {stats['median_degC']} °C")
    print(f"  p1 / p99         : {stats['p1_degC']} / {stats['p99_degC']} °C")

    if slice_metadata:
        print(f"  Slice axis {slice_metadata['axis']} "
              f"({slice_metadata['axis_name']}) "
              f"index {slice_metadata['index']}: "
              f"[{slice_metadata['min_degC']}, {slice_metadata['max_degC']}] °C")


if __name__ == "__main__":
    main()
