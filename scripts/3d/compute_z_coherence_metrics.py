#!/usr/bin/env python
"""
Compute Z-coherence metrics for a NIfTI volume.

Metrics produced:
  adjacent_ssim        — Structural Similarity Index between consecutive slices
                         (per-pair values + aggregate statistics).
  z_gradient_smoothness — Mean absolute intensity difference between
                         consecutive slices (lower = smoother).

An optional --mask-nifti restricts computation to foreground voxels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import convolve


# ---------------------------------------------------------------------------
# Lightweight SSIM (no skimage dependency)
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Return a 2D Gaussian kernel normalized to sum 1."""
    ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    gauss = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d


def _ssim_single(a: np.ndarray, b: np.ndarray,
                 data_range: float = 255.0,
                 kernel: np.ndarray | None = None) -> float:
    """SSIM between two 2D arrays (single channel)."""
    if kernel is None:
        kernel = _gaussian_kernel_2d()

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu_a = convolve(a.astype(np.float64), kernel, mode="reflect")
    mu_b = convolve(b.astype(np.float64), kernel, mode="reflect")

    sigma_a2 = convolve(a.astype(np.float64) ** 2, kernel, mode="reflect") - mu_a ** 2
    sigma_b2 = convolve(b.astype(np.float64) ** 2, kernel, mode="reflect") - mu_b ** 2
    sigma_ab = convolve(a.astype(np.float64) * b.astype(np.float64), kernel, mode="reflect") - mu_a * mu_b

    ssim_map = ((2.0 * mu_a * mu_b + C1) * (2.0 * sigma_ab + C2)) / \
               ((mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2))
    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute Z-coherence metrics for a NIfTI volume."
    )
    ap.add_argument("--volume-nifti", required=True, help="Input .nii.gz volume.")
    ap.add_argument(
        "--mask-nifti",
        default=None,
        help="Optional binary mask .nii.gz (foreground voxels only).",
    )
    ap.add_argument("--output-json", required=True, help="Output metrics JSON.")
    ap.add_argument(
        "--ssim-data-range",
        type=float,
        default=255.0,
        help="Data range for SSIM (default 255 for uint8).",
    )
    args = ap.parse_args()

    # --- Load volume ---
    vol_img = nib.load(args.volume_nifti)
    data = np.asanyarray(vol_img.dataobj).astype(np.float64)

    # Handle 4D with single channel
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise SystemExit(f"Expected 3D volume (H,W,Z), got shape {data.shape}")

    H, W, Z = data.shape
    print(f"Volume: {data.shape}  pixdim={vol_img.header.get_zooms()[:3]}")

    if Z < 2:
        raise SystemExit(
            f"Volume must have at least 2 slices along Z to compute adjacent metrics; got Z={Z}."
        )

    # --- Load optional mask ---
    mask_data: np.ndarray | None = None
    if args.mask_nifti:
        m_img = nib.load(args.mask_nifti)
        mask_data = (np.asanyarray(m_img.dataobj) > 0)
        if mask_data.ndim == 4:
            mask_data = mask_data[..., 0]
        if mask_data.ndim not in (2, 3):
            raise SystemExit(
                f"Expected 2D/3D mask after load, got shape {mask_data.shape}"
            )
        if mask_data.shape[:2] != (H, W):
            raise SystemExit(
                f"Mask spatial shape {mask_data.shape[:2]} does not match volume {(H, W)}"
            )
        print(f"Mask: {mask_data.shape}")

    # --- Gaussian kernel (cached) ---
    kernel = _gaussian_kernel_2d()

    # --- 1. Adjacent SSIM ---
    ssim_values: list[float] = []
    for z_idx in range(Z - 1):
        a = data[..., z_idx]
        b = data[..., z_idx + 1]

        if mask_data is not None:
            # Determine active region for this pair
            if args.mask_nifti and mask_data.ndim == 3:
                m = mask_data[..., z_idx if z_idx < mask_data.shape[-1] else -1]
            else:
                m = mask_data
            if m.sum() == 0:
                ssim_values.append(0.0)
                continue
            # Restrict to mask region
            a_masked = np.where(m, a, 0.0)
            b_masked = np.where(m, b, 0.0)
            val = _ssim_single(a_masked, b_masked, data_range=args.ssim_data_range, kernel=kernel)
        else:
            val = _ssim_single(a, b, data_range=args.ssim_data_range, kernel=kernel)
        ssim_values.append(val)

    ssim_arr = np.array(ssim_values, dtype=np.float64)

    # --- 2. Z-gradient smoothness ---
    # Mean absolute difference between consecutive slices
    diffs = np.abs(np.diff(data, axis=-1))  # (H, W, Z-1)
    if mask_data is not None:
        # Mask the gradient computation
        per_slice_grad_vals: list[float] = []
        for z_idx in range(Z - 1):
            if mask_data.ndim == 3:
                m = mask_data[..., z_idx] if z_idx < mask_data.shape[-1] else mask_data[..., -1]
            else:
                m = mask_data
            if m.sum() == 0:
                per_slice_grad_vals.append(0.0)
            else:
                per_slice_grad_vals.append(float(diffs[..., z_idx][m].mean()))
        per_slice_grad = np.array(per_slice_grad_vals, dtype=np.float64)
    else:
        per_slice_grad = diffs.reshape(H * W, Z - 1).mean(axis=0)

    metrics = {
        "adjacent_ssim": {
            "values": [round(v, 6) for v in ssim_values],
            "mean": round(float(np.mean(ssim_arr)), 6),
            "std": round(float(np.std(ssim_arr)), 6),
            "min": round(float(np.min(ssim_arr)), 6),
            "max": round(float(np.max(ssim_arr)), 6),
        },
        "z_gradient_smoothness": {
            "per_slice_mean": [round(float(v), 6) for v in per_slice_grad.tolist()],
            "mean_abs_gradient": round(float(np.mean(per_slice_grad)), 6),
            "std_abs_gradient": round(float(np.std(per_slice_grad)), 6),
        },
        "volume_shape": [H, W, Z],
        "mask_used": args.mask_nifti is not None,
    }

    # --- Save ---
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics: {out_path}")
    print(f"  Mean adjacent SSIM:          {metrics['adjacent_ssim']['mean']:.4f}")
    print(f"  Mean Z-gradient (abs):       {metrics['z_gradient_smoothness']['mean_abs_gradient']:.4f}")


if __name__ == "__main__":
    main()
