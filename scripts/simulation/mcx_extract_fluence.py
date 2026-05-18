#!/usr/bin/env python
"""
Lightweight extractor for MCX fluence / output arrays.

Reads MCX v2025.10 output (.npy, .mc2, .npz, or .mch).
Computes summary statistics (min, max, mean, std, percentiles) and writes
a JSON report.  Optionally saves an axial projection (maximum intensity
projection along the last axis) as PNG if matplotlib is available.

Typical MCX output shape: (D, H, W) with fluence values in W/mm² or
absorbed energy density in J/mm³.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional matplotlib for PNG projection
# ---------------------------------------------------------------------------
_HAS_MPL = False
try:
    import matplotlib.pyplot as plt  # noqa: F401

    _HAS_MPL = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_fluence(path: str) -> np.ndarray:
    """Load an MCX output array from disk.

    Supported formats:
      - .npy   : numpy binary
      - .mc2   : MCX native mc2 format (raw float32 binary)
      - .npz   : numpy archive (takes first array)
      - .mch   : MCX history file (attempt numpy load; may fail for non-trivial)

    Returns:
        2D or 3D numpy array (flat for .mc2/.mch; caller should reshape
        based on known volume dimensions).

    Raises:
        SystemExit on unsupported format or load failure.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Fluence file not found: {path}")

    ext = p.suffix.lower()

    if ext == ".npy":
        return np.load(str(p))

    elif ext == ".mc2":
        # MCX v2025.10 mc2 format: raw 32-bit float binary, no header.
        # The output shape matches the Domain.Dim dimensions but may be
        # transposed.  Caller should inspect the JSON config for shape.
        arr = np.fromfile(str(p), dtype=np.float32)
        print(f"  Loaded .mc2: {len(arr)} floats ({len(arr)*4} bytes)")
        warnings.warn(
            ".mc2 loaded as flat 1-D array. Reshape using Domain.Dim "
            "from the accompanying mcx_config.json (Dim=[X,Y,Z])."
        )
        return arr

    elif ext == ".npz":
        with np.load(str(p)) as data:
            # Take the first array in the archive
            for key in data.files:
                arr = data[key]
                print(f"  Loaded array '{key}' from .npz, shape={arr.shape}")
                return arr
            raise SystemExit(f".npz file {path} contains no arrays.")

    elif ext == ".mch":
        # .mch is a custom MCX binary format.  Attempt to load as raw numpy
        # (works for trivial files).  For real MCX .mch output, this likely
        # fails and requires the MCX Matlab/Python toolbox.
        try:
            arr = np.fromfile(str(p), dtype=np.float32)
            warnings.warn(
                ".mch loaded as flat binary — shape may be incorrect. "
                "Prefer .mc2 or use the MCX Python toolbox."
            )
            return arr
        except Exception as e:
            raise SystemExit(
                f"Failed to load .mch file {path}: {e}. "
                "Try generating MCX output with OutputFormat=npy or mc2."
            )

    else:
        raise SystemExit(
            f"Unsupported extension '{ext}'. "
            "Use .npy, .mc2 (preferred), .npz, or .mch (fallback)."
        )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _compute_summary_stats(arr: np.ndarray) -> Dict[str, Any]:
    """Compute standard summary statistics for a fluence array."""
    if arr.size == 0:
        raise SystemExit("Input array is empty; cannot compute summary statistics.")

    # Flatten for percentile computation
    flat = arr.ravel()

    stats: Dict[str, Any] = {
        "shape": list(arr.shape),
        "ndim": arr.ndim,
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(flat)),
        "p1": float(np.percentile(flat, 1)),
        "p5": float(np.percentile(flat, 5)),
        "p25": float(np.percentile(flat, 25)),
        "p75": float(np.percentile(flat, 75)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "num_zeros": int((arr == 0).sum()),
        "num_nonzero": int((arr != 0).sum()),
        "fraction_nonzero": float((arr != 0).mean()),
    }
    return stats


def _sanitize_stats(stats: Dict[str, Any]) -> bool:
    """Replace non-finite float values with ``None`` for JSON compliance.

    Mutates *stats* in place.  Returns ``True`` if any value was replaced.
    """
    found = False
    for key, val in stats.items():
        if isinstance(val, float) and not math.isfinite(val):
            stats[key] = None
            found = True
    return found


# ---------------------------------------------------------------------------
# Projection PNG
# ---------------------------------------------------------------------------


def _save_axial_projection(arr: np.ndarray, output_path: str) -> None:
    """Maximum intensity projection along the last (depth) axis as PNG.

    If arr is 2D, saves it directly.  If 3D, projects along axis=-1.
    Skips gracefully if matplotlib is unavailable.
    """
    if not _HAS_MPL:
        print("  [SKIP] matplotlib not available; skipping PNG projection.")
        return

    if arr.ndim == 2:
        proj = arr
    elif arr.ndim >= 3:
        proj = arr.max(axis=-1)
    else:
        print(f"  [SKIP] unexpected array ndim={arr.ndim}; skipping projection.")
        return

    # Normalize to 0-255 (uint8), handling flat / constant arrays
    pmin, pmax = float(proj.min()), float(proj.max())
    if pmax > pmin:
        proj_uint8 = ((proj - pmin) / (pmax - pmin) * 255).astype(np.uint8)
    else:
        proj_uint8 = np.zeros_like(proj, dtype=np.uint8)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(proj_uint8, cmap="hot", aspect="auto")
    ax.set_title("MCX Fluence (axial MIP)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved projection PNG: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Extract summary statistics and optionally a PNG projection "
        "from an MCX output fluence array.",
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Path to MCX output file (.npy, .mc2, .npz, or fallback .mch).",
    )
    ap.add_argument(
        "--output-json",
        required=True,
        help="Path to write summary statistics JSON.",
    )
    ap.add_argument(
        "--output-png",
        default=None,
        help="Optional path for axial projection PNG (requires matplotlib).",
    )
    return ap


def main() -> None:
    ap = _build_cli()
    args = ap.parse_args()

    # --- Load ---
    print(f"Loading fluence from: {args.input}")
    arr = _load_fluence(args.input)
    print(f"  Shape: {arr.shape}, dtype={arr.dtype}")

    # --- Compute stats ---
    stats = _compute_summary_stats(arr)

    # --- Sanitize for JSON compliance (NaN/Inf → null) ---
    # Operate on a copy so the original stats dict (used below for
    # terminal output) still contains raw values.
    json_safe_stats = dict(stats)
    if _sanitize_stats(json_safe_stats):
        warnings.warn(
            "Non-finite values (NaN/Inf) detected in fluence array; "
            "replaced with null in output JSON."
        )

    # --- Write JSON ---
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_safe_stats, indent=2), encoding="utf-8")
    print(f"Wrote stats JSON: {json_path}")

    # --- Print concise summary ---
    print(f"\n=== Fluence Summary ===")
    print(f"  Shape   : {stats['shape']}")
    print(f"  Min     : {stats['min']:.6e}")
    print(f"  Max     : {stats['max']:.6e}")
    print(f"  Mean    : {stats['mean']:.6e}")
    print(f"  Median  : {stats['median']:.6e}")
    print(f"  Std     : {stats['std']:.6e}")
    print(f"  p1/p99  : {stats['p1']:.6e} / {stats['p99']:.6e}")
    print(f"  Non-zero: {stats['fraction_nonzero']*100:.1f}%")

    # --- Optional PNG projection ---
    if args.output_png:
        _save_axial_projection(arr, args.output_png)


if __name__ == "__main__":
    main()
