#!/usr/bin/env python
"""
build_roi_masks_gradcam.py
=================================

Utility to build coarse lesion ROI masks for masked-loss / inpainting
training, using a classifier + Grad-CAM. Masks are saved as single-channel
PNG files (0=background, 255=lesion ROI) with the same filenames as the
source images.

This script does *not* depend on SAM2 directly so it can run in a plain
PyTorch environment. If you want to refine the Grad-CAM masks with SAM2,
you can:
  - take the saved heatmaps / masks from this script, and
  - feed their peak points / bounding boxes into a separate SAM2 pipeline.

Typical usage (from repo root):

  python scripts/synthetic_data/build_roi_masks_gradcam.py \\
      --csv data/processed/multisource_train.csv \\
      --image-column image_path \\
      --output-dir data/artifacts/roi_masks/multisource_train \\
      --checkpoint /path/to/best_effnet.ckpt \\
      --model-config configs/studies/effnet_imagenet_b3_ar_multisource.yaml \\
      --device cuda:0

The resulting `output-dir` can be pointed to by `lora_mask_dir` in
configs/diffusion/finetune/sdxl_lora.yaml, and the unified finetune script
will use it when `lora_use_masked_loss: true`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

# Ensure repo root (with `src/`) is on sys.path so `src.oral_lesions...`
# imports work even when this script is invoked from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.oral_lesions.models.factory import create_model


@dataclass
class GradCAMConfig:
    layer_name: str
    img_size: int = 384
    resize_mode: str = "preserve"
    target_class_from_csv: bool = True
    threshold_quantile: float = 0.85
    min_area_frac: float = 0.001
    max_area_frac: float = 0.6
    dilate_radius: int = 9
    feather_radius: float = 0.0


def _load_study_model_config(path: Path) -> Tuple[dict, dict]:
    """
    Load a study YAML with top-level keys {study, models}, returning
    (study_cfg, model_cfg).
    """
    import yaml

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    study_cfg = cfg.get("study", {}) or {}
    models = cfg.get("models", []) or []
    if not models:
        raise ValueError(f"No 'models' section found in {path}")
    model_cfg = models[0]  # first model definition
    return study_cfg, model_cfg


def _build_classifier(
    model_cfg: dict,
    num_classes: int,
    checkpoint_path: Optional[Path],
    device: torch.device,
) -> torch.nn.Module:
    m = create_model(model_cfg, num_classes=num_classes, device=device)
    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[GradCAM] Loading classifier weights from: {checkpoint_path}")
        sd = torch.load(checkpoint_path, map_location=device)
        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        # allow non-strict to be robust to minor differences
        m.load_state_dict(sd, strict=False)
    m.to(device)
    m.eval()
    return m


class SimpleGradCAM:
    """
    Minimal Grad-CAM for CNN backbones (e.g., EfficientNet).
    Hooks a named convolutional layer and uses its activations + gradients.
    """

    def __init__(self, model: torch.nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.activations = None
        self.gradients = None

        layer = dict(model.named_modules()).get(layer_name)
        if layer is None:
            names = ", ".join(dict(model.named_modules()).keys())
            raise ValueError(f"Layer '{layer_name}' not found in model. Available: {names}")

        def fwd_hook(_module, _inp, out):
            self.activations = out.detach()

        def bwd_hook(_module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        layer.register_forward_hook(fwd_hook)
        layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        """
        Returns a [H, W] heatmap tensor normalized to [0, 1] in the
        resized input space.
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if logits.ndim != 2:
            raise RuntimeError(f"Expected classifier output [B, C], got {logits.shape}")
        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        # activations: [B, C, H, W], gradients: [B, C, H, W]
        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * acts).sum(dim=1)  # [B, H, W]
        cam = F.relu(cam)

        # Normalize per-image
        b, h, w = cam.shape
        cam = cam.view(b, -1)
        cam -= cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        cam_max[cam_max == 0] = 1.0
        cam = cam / cam_max
        cam = cam.view(b, h, w)
        return cam


def _preprocess_image(
    img_bgr: np.ndarray,
    img_size: int,
    resize_mode: str = "preserve",
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int, int, int]]:
    """
    Basic preprocessing aligned with effnet transforms:
      - resize / pad to square (preserve aspect if requested)
      - normalize to ImageNet stats
      - return (tensor[1,3,H,W], original_hw, padding_info)
        padding_info is (y0, x0, new_h, new_w) relative to img_size square
    """
    if img_bgr is None:
        raise ValueError("img_bgr is None")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    if resize_mode == "preserve":
        # longest side -> img_size, pad to square
        scale = float(img_size) / max(orig_h, orig_w)
        new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((img_size, img_size, 3), dtype=resized.dtype)
        y0 = (img_size - new_h) // 2
        x0 = (img_size - new_w) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        img_proc = canvas
        padding_info = (y0, x0, new_h, new_w)
    else:
        img_proc = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
        padding_info = (0, 0, img_size, img_size)

    img = img_proc.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # C,H,W
    tensor = torch.from_numpy(img).unsqueeze(0)  # [1,3,H,W]
    return tensor, (orig_h, orig_w), padding_info


def _heatmap_to_mask(
    cam: np.ndarray,
    orig_hw: Tuple[int, int],
    threshold_quantile: float,
    min_area_frac: float,
    max_area_frac: float,
    dilate_radius: int,
    feather_radius: float,
    padding_info: Tuple[int, int, int, int],
    img_size: int = 384,
) -> np.ndarray:
    """
    Convert a [Hc, Wc] heatmap in [0,1] to a binary mask in original
    image space, with simple area sanity checks and dilation.
    """
    cam = np.clip(cam, 0.0, 1.0)
    
    # Crop heatmap to actual image area (removing padding)
    y0, x0, h_p, w_p = padding_info
    hc, wc = cam.shape
    # Map padding coords from img_size space to heatmap space
    sy, sx = hc / float(img_size), wc / float(img_size)
    
    y0_c, x0_c = int(round(y0 * sy)), int(round(x0 * sx))
    hp_c, wp_c = int(round(h_p * sy)), int(round(w_p * sx))
    
    cam_cropped = cam[y0_c : y0_c + hp_c, x0_c : x0_c + wp_c]
    
    # Resize back to original image dimensions
    cam_resized = cv2.resize(cam_cropped, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)

    flat = cam_resized.flatten()
    thr = np.quantile(flat, threshold_quantile)
    mask = (cam_resized >= thr).astype(np.uint8)

    # remove tiny noise and very large blobs
    area = mask.sum()
    total = mask.size
    frac = area / float(max(total, 1))
    if frac < min_area_frac:
        # fallback to slightly lower threshold
        thr = np.quantile(flat, 0.75)
        mask = (cam_resized >= thr).astype(np.uint8)
        area = mask.sum()
        frac = area / float(max(total, 1))
    if frac > max_area_frac:
        # clamp by raising threshold
        thr = np.quantile(flat, 0.95)
        mask = (cam_resized >= thr).astype(np.uint8)

    if dilate_radius > 0:
        k = max(1, int(dilate_radius))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    mask = mask.astype(np.float32)
    if feather_radius > 0:
        mask = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=float(feather_radius), sigmaY=float(feather_radius))
    mask = np.clip(mask, 0.0, 1.0)
    return (mask * 255).astype(np.uint8)


def build_roi_masks(
    csv_path: Path,
    image_column: str,
    label_column: str,
    output_dir: Path,
    model_cfg_path: Path,
    checkpoint_path: Optional[Path],
    device_str: str,
    gradcam_layer: str,
    cam_cfg: Optional[GradCAMConfig] = None,
) -> None:
    import pandas as pd

    csv_path = csv_path.resolve()
    output_dir = output_dir.resolve()
    
    # Clear directory to avoid 'extra conditioning data' errors in kohya scripts
    if output_dir.exists():
        print(f"[GradCAM] Clearing existing masks in {output_dir}...")
        for p in output_dir.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GradCAM] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if image_column not in df.columns:
        raise ValueError(f"CSV missing '{image_column}' column.")
    if label_column not in df.columns:
        raise ValueError(f"CSV missing '{label_column}' column.")

    # Build label mapping
    labels = sorted(df[label_column].astype(str).unique())
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    num_classes = len(labels)
    print(f"[GradCAM] Found {num_classes} labels: {labels}")

    has_filename_col = "filename" in df.columns

    study_cfg, model_cfg = _load_study_model_config(model_cfg_path)
    # Ensure the factory knows what model type to build
    model_cfg.setdefault("model_type", "effnet")

    device = torch.device(device_str)
    model = _build_classifier(model_cfg, num_classes=num_classes, checkpoint_path=checkpoint_path, device=device)

    cam_cfg = cam_cfg or GradCAMConfig(layer_name=gradcam_layer)
    grad_cam = SimpleGradCAM(model, layer_name=cam_cfg.layer_name)

    n_rows = len(df)
    for idx, row in df.iterrows():
        img_path = Path(row[image_column])
        lbl = str(row[label_column])
        class_idx = label_to_idx.get(lbl, None)
        if class_idx is None:
            print(f"[GradCAM] Skipping row {idx}: unknown label {lbl!r}")
            continue
        if not img_path.is_file():
            print(f"[GradCAM] Missing image: {img_path}")
            continue

        try:
            # Use PIL to load image in RAW orientation (matching Kohya behavior)
            with Image.open(img_path) as pil_img:
                img_rgb = np.array(pil_img.convert("RGB"))
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            x, orig_hw, pad_info = _preprocess_image(
                img_bgr,
                img_size=cam_cfg.img_size,
                resize_mode=cam_cfg.resize_mode,
            )
            x = x.to(device)

            with torch.enable_grad():
                heat = grad_cam(x, class_idx=class_idx)[0].detach().cpu().numpy()

            mask = _heatmap_to_mask(
                heat,
                orig_hw=orig_hw,
                threshold_quantile=cam_cfg.threshold_quantile,
                min_area_frac=cam_cfg.min_area_frac,
                max_area_frac=cam_cfg.max_area_frac,
                dilate_radius=cam_cfg.dilate_radius,
                feather_radius=cam_cfg.feather_radius,
                padding_info=pad_info,
                img_size=cam_cfg.img_size,
            )

            # Use 'filename' column if it exists to match materialized dataset names
            out_name = row["filename"] if has_filename_col else img_path.name
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), mask)
        except Exception as e:
            print(f"[GradCAM] ERROR processing {img_path}: {type(e).__name__}: {e}")
            continue

        if (idx + 1) % 50 == 0 or (idx + 1) == n_rows:
            print(f"[GradCAM] Processed {idx+1}/{n_rows} images")

    # Store a small manifest for reproducibility
    meta = {
        "csv_path": str(csv_path),
        "image_column": image_column,
        "label_column": label_column,
        "output_dir": str(output_dir),
        "model_config": str(model_cfg_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "labels": labels,
        "cam_layer": gradcam_layer,
    }
    (output_dir / "build_roi_masks_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[GradCAM] Done. Masks written to {output_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build Grad-CAM based ROI masks for lesion inpainting.")
    ap.add_argument("--csv", required=True, help="CSV with image paths and labels (e.g., multisource_train.csv)")
    ap.add_argument(
        "--image-column",
        default="image_path",
        help="Column in CSV with full image paths (default: image_path)",
    )
    ap.add_argument(
        "--label-column",
        default="coarse_label",
        help="Column in CSV with labels (default: coarse_label)",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write ROI masks (same filenames as source images).",
    )
    ap.add_argument(
        "--model-config",
        required=True,
        help="Path to classifier study/model YAML (e.g., configs/studies/effnet_imagenet_b3_ar_multisource.yaml).",
    )
    ap.add_argument(
        "--checkpoint",
        required=False,
        help="Path to classifier checkpoint (.pth/.ckpt). If omitted, random weights are used (not recommended).",
    )
    ap.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for classifier inference (default: cuda:0 if available).",
    )
    ap.add_argument(
        "--gradcam-layer",
        default="model.layers[-1]",
        help=(
            "Layer name inside the classifier to hook for Grad-CAM. "
            "Use something like 'blocks.6.conv_pwl' for EfficientNet; "
            "see printed module names if unsure."
        ),
    )
    ap.add_argument(
        "--img-size",
        type=int,
        default=384,
        help="Input size for the classifier (default: 384, matching effnet_b3_multisource).",
    )
    ap.add_argument(
        "--resize-mode",
        choices=["stretch", "preserve"],
        default="preserve",
        help="Aspect-ratio policy when resizing input images (default: preserve).",
    )
    ap.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.85,
        help="Quantile of Grad-CAM heatmap to threshold for ROI (default: 0.85).",
    )
    ap.add_argument(
        "--min-area-frac",
        type=float,
        default=0.001,
        help="Minimum area fraction for ROI mask; below this triggers a fallback threshold (default: 0.001).",
    )
    ap.add_argument(
        "--max-area-frac",
        type=float,
        default=0.6,
        help="Maximum area fraction for ROI mask; above this clamps threshold (default: 0.6).",
    )
    ap.add_argument(
        "--dilate-radius",
        type=int,
        default=9,
        help="Radius (in pixels) for morphological dilation of the ROI mask (default: 9).",
    )
    ap.add_argument(
        "--feather-radius",
        type=float,
        default=0.0,
        help="Gaussian blur radius (sigma, in pixels) for soft mask edges (default: 0.0 = disabled).",
    )

    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    out_dir = Path(args.output_dir)
    model_cfg_path = Path(args.model_config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    cam_cfg = GradCAMConfig(
        layer_name=args.gradcam_layer,
        img_size=int(args.img_size),
        resize_mode=args.resize_mode,
        threshold_quantile=float(args.threshold_quantile),
        min_area_frac=float(args.min_area_frac),
        max_area_frac=float(args.max_area_frac),
        dilate_radius=int(args.dilate_radius),
        feather_radius=float(args.feather_radius),
    )

    build_roi_masks(
        csv_path=csv_path,
        image_column=args.image_column,
        label_column=args.label_column,
        output_dir=out_dir,
        model_cfg_path=model_cfg_path,
        checkpoint_path=checkpoint_path,
        device_str=args.device,
        gradcam_layer=args.gradcam_layer,
        cam_cfg=cam_cfg,
    )


if __name__ == "__main__":
    main()
