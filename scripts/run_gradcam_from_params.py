#!/usr/bin/env python
"""
Run Grad-CAM ROI mask generation from params.yaml values.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj or {}


def _need(name: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise SystemExit(f"Missing required params value: {name}")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Run build_roi_masks_gradcam.py using params.yaml")
    ap.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    params = _load_yaml(Path(args.params))
    gradcam = params.get("gradcam", {}) or {}
    models = params.get("models", {}) or {}

    cmd = [
        sys.executable,
        str(root / "scripts" / "synthetic_data" / "build_roi_masks_gradcam.py"),
        "--csv",
        _need("gradcam.csv", gradcam.get("csv")),
        "--image-column",
        str(gradcam.get("image_column", "image_path")),
        "--label-column",
        str(gradcam.get("label_column", "coarse_label")),
        "--output-dir",
        _need("gradcam.output_dir", gradcam.get("output_dir")),
        "--model-config",
        _need("gradcam.model_config", gradcam.get("model_config")),
        "--checkpoint",
        _need("models.classifier_checkpoint", models.get("classifier_checkpoint")),
        "--gradcam-layer",
        str(gradcam.get("gradcam_layer", "enet.conv_head")),
        "--img-size",
        str(int(gradcam.get("img_size", 384))),
        "--threshold-quantile",
        str(float(gradcam.get("threshold_quantile", 0.85))),
        "--dilate-radius",
        str(int(gradcam.get("dilate_radius", 9))),
        "--device",
        str(gradcam.get("device", "cuda:0")),
    ]

    if gradcam.get("resize_mode"):
        cmd.extend(["--resize-mode", str(gradcam["resize_mode"])])
    if gradcam.get("min_area_frac") is not None:
        cmd.extend(["--min-area-frac", str(gradcam["min_area_frac"])])
    if gradcam.get("max_area_frac") is not None:
        cmd.extend(["--max-area-frac", str(gradcam["max_area_frac"])])
    if gradcam.get("feather_radius") is not None:
        cmd.extend(["--feather-radius", str(gradcam["feather_radius"])])

    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
