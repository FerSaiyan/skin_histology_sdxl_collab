#!/usr/bin/env python
"""
Interactive 3D volume viewer for prepared MCX input volumes.

Supports `.npy`, `.nii`, and `.nii.gz`.
Shows axial/coronal/sagittal slices with sliders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_volume(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"Volume not found: {path}")
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
    elif path.suffix.lower() == ".nii" or path.suffixes[-2:] == [".nii", ".gz"]:
        try:
            import nibabel as nib
        except ImportError as e:
            raise SystemExit("nibabel is required for NIfTI volumes: pip install nibabel") from e
        nii = nib.load(str(path))
        arr = np.asanyarray(nii.dataobj)
    else:
        raise SystemExit(f"Unsupported extension for {path}")

    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim != 3:
        raise SystemExit(f"Expected 3D volume, got shape {arr.shape}")
    return arr


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Interactive 3D volume viewer.")
    ap.add_argument("--volume", required=True, help="Path to volume (.npy/.nii/.nii.gz).")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap.")
    return ap


def main() -> int:
    args = _build_cli().parse_args()

    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except ImportError as e:
        raise SystemExit("matplotlib is required: pip install matplotlib") from e

    vol = _load_volume(Path(args.volume).resolve())
    zmax, ymax, xmax = vol.shape
    zi, yi, xi = zmax // 2, ymax // 2, xmax // 2

    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    plt.subplots_adjust(bottom=0.25)

    im0 = axs[0].imshow(vol[zi, :, :], cmap=args.cmap)
    axs[0].set_title("Axial (Z)")
    im1 = axs[1].imshow(vol[:, yi, :], cmap=args.cmap)
    axs[1].set_title("Coronal (Y)")
    im2 = axs[2].imshow(vol[:, :, xi], cmap=args.cmap)
    axs[2].set_title("Sagittal (X)")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    ax_z = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_y = plt.axes([0.15, 0.1, 0.7, 0.03])
    ax_x = plt.axes([0.15, 0.05, 0.7, 0.03])

    s_z = Slider(ax_z, "Z", 0, zmax - 1, valinit=zi, valstep=1)
    s_y = Slider(ax_y, "Y", 0, ymax - 1, valinit=yi, valstep=1)
    s_x = Slider(ax_x, "X", 0, xmax - 1, valinit=xi, valstep=1)

    def _update(_: float) -> None:
        z = int(s_z.val)
        y = int(s_y.val)
        x = int(s_x.val)
        im0.set_data(vol[z, :, :])
        im1.set_data(vol[:, y, :])
        im2.set_data(vol[:, :, x])
        fig.canvas.draw_idle()

    s_z.on_changed(_update)
    s_y.on_changed(_update)
    s_x.on_changed(_update)

    fig.suptitle(f"Volume: {Path(args.volume).name} shape={vol.shape}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
