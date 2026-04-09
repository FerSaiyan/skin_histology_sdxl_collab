#!/usr/bin/env python3
"""
Score generated LoRA sample images with a trained classifier on CPU.

This script is intended for training-time sample folders produced by kohya
(`.../sample/*.png`), where filenames typically include a prompt index:

  000500_02_20260129163014_512.png
          ^^
          prompt index (line number in prompts_to_check.txt)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _repo_root(start: Optional[Path] = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
        if (parent / "dvc.yaml").exists():
            return parent
    return Path.cwd().resolve()


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.exp.config import load_config
from src.oral_lesions.models.factory import create_model


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
PROMPT_RE = re.compile(r"diagnosis:\s*([A-Za-z0-9_]+)")


def _build_classes(csv_paths: List[Path], label_col: str = "coarse_label") -> List[str]:
    labels = set()
    for p in csv_paths:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if label_col not in df.columns:
            continue
        labels |= set(df[label_col].dropna().astype(str).tolist())
    return sorted(labels)


def _load_prompt_labels(prompts_file: Path) -> List[str]:
    if not prompts_file.exists():
        return []
    labels: List[str] = []
    with prompts_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = PROMPT_RE.search(line)
            labels.append(m.group(1) if m else "")
    return labels


def _prompt_index_from_name(name: str) -> Optional[int]:
    parts = Path(name).stem.split("_")
    if len(parts) < 2:
        return None
    if parts[1].isdigit():
        return int(parts[1])
    return None


def _infer_num_classes(state_dict: Dict) -> Optional[int]:
    candidate_keys = [
        "myfc.weight",
        "model.myfc.weight",
        "vit_model.head.weight",
        "model.vit_model.head.weight",
        "head.weight",
        "model.head.weight",
    ]
    for k in candidate_keys:
        v = state_dict.get(k)
        if torch.is_tensor(v) and v.ndim == 2:
            return int(v.shape[0])
    return None


def _normalize_state_dict(sd: Dict) -> Dict:
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    if isinstance(sd, dict) and any(k.startswith("model.") for k in sd.keys()):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    return sd


def _resolve_class_names(
    class_csvs: List[Path],
    prompt_labels: List[str],
    inferred_num_classes: Optional[int],
    label_col: str,
) -> List[str]:
    classes = _build_classes(class_csvs, label_col=label_col)
    if inferred_num_classes is None:
        return classes
    if classes and len(classes) == inferred_num_classes:
        return classes

    ordered_prompt_unique = []
    seen = set()
    for x in prompt_labels:
        if not x:
            continue
        if x not in seen:
            ordered_prompt_unique.append(x)
            seen.add(x)
    if len(ordered_prompt_unique) == inferred_num_classes:
        return sorted(ordered_prompt_unique)

    if inferred_num_classes == 4:
        return sorted(["healthy", "benign_lesion", "opmd", "cancer"])

    # Last resort: generic class names.
    return [f"class_{i}" for i in range(inferred_num_classes)]


class FolderDataset(Dataset):
    def __init__(self, image_paths: List[Path], img_size: int, resize_mode: str, prompt_labels: List[str]):
        self.image_paths = image_paths
        self.img_size = int(img_size)
        self.resize_mode = str(resize_mode).lower()
        self.prompt_labels = prompt_labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        x = _preprocess_pil(img, img_size=self.img_size, resize_mode=self.resize_mode)

        prompt_idx = _prompt_index_from_name(p.name)
        expected = None
        if prompt_idx is not None and 0 <= prompt_idx < len(self.prompt_labels):
            expected = self.prompt_labels[prompt_idx] or None

        return x, str(p), prompt_idx, expected


def _preprocess_pil(img: Image.Image, img_size: int, resize_mode: str) -> torch.Tensor:
    if resize_mode == "preserve":
        w, h = img.size
        scale = float(img_size) / max(h, w)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (img_size, img_size), (0, 0, 0))
        x0 = (img_size - new_w) // 2
        y0 = (img_size - new_h) // 2
        canvas.paste(resized, (x0, y0))
        img = canvas
    else:
        img = img.resize((img_size, img_size), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    return torch.from_numpy(arr)


def _resolve_device(device_arg: str) -> torch.device:
    d = (device_arg or "auto").strip().lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device=cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if d == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"Unsupported --device value: {device_arg} (expected: auto|cuda|cpu)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score LoRA training samples with a classifier checkpoint")
    ap.add_argument("--images-dir", required=True, help="Directory containing generated sample images")
    ap.add_argument("--checkpoint", required=True, help="Classifier checkpoint (.pth)")
    ap.add_argument(
        "--study-config",
        default=str(REPO_ROOT / "configs/studies/uni_effnet_imagenet_ar.yaml"),
        help="Study YAML used to build classifier architecture/transforms",
    )
    ap.add_argument(
        "--prompts-file",
        default=str(REPO_ROOT / "kohya_ss/sd-scripts/prompts_to_check.txt"),
        help="Kohya prompts file used during sampling (to recover expected label per prompt index)",
    )
    ap.add_argument(
        "--class-csv",
        action="append",
        default=[],
        help="CSV(s) with label column used to infer class names/order; can be passed multiple times",
    )
    ap.add_argument("--label-col", default="coarse_label", help="Label column name in class CSV(s)")
    ap.add_argument("--device", default="auto", help="Inference device: auto|cuda|cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--output-json",
        default=None,
        help="Where to write summary JSON (default: <images-dir>/classifier_eval_cpu.json)",
    )
    ap.add_argument(
        "--output-csv",
        default=None,
        help="Where to write per-image predictions CSV (default: <images-dir>/classifier_eval_cpu.csv)",
    )
    args = ap.parse_args()

    images_dir = Path(args.images_dir).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    study_cfg_path = Path(args.study_config).resolve()
    prompts_file = Path(args.prompts_file).resolve()

    if not images_dir.is_dir():
        raise SystemExit(f"--images-dir does not exist or is not a directory: {images_dir}")
    if not ckpt_path.is_file():
        raise SystemExit(f"--checkpoint not found: {ckpt_path}")
    if not study_cfg_path.is_file():
        raise SystemExit(f"--study-config not found: {study_cfg_path}")

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    cfg = load_config(str(study_cfg_path))
    study_cfg = cfg.get("study", {}) or {}
    model_cfg = (cfg.get("models", []) or [None])[0]
    if model_cfg is None:
        raise SystemExit(f"No models section found in study config: {study_cfg_path}")

    model_type = str(model_cfg.get("model_type", "effnet"))
    img_size = int(model_cfg.get("img_size", 384 if model_type == "effnet" else 224))
    resize_mode = str(study_cfg.get("resize_mode", "preserve"))

    prompt_labels = _load_prompt_labels(prompts_file)

    class_csvs: List[Path] = [Path(p).resolve() for p in args.class_csv]
    if not class_csvs:
        for k in ("train_csv_path", "val_csv_path", "test_csv_path"):
            raw = study_cfg.get(k)
            if not raw:
                continue
            p = Path(str(raw))
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            class_csvs.append(p)

    device = _resolve_device(args.device)

    raw_state = torch.load(ckpt_path, map_location=device)
    state = _normalize_state_dict(raw_state)
    inferred_num_classes = _infer_num_classes(state) if isinstance(state, dict) else None
    classes = _resolve_class_names(class_csvs, prompt_labels, inferred_num_classes, label_col=args.label_col)
    num_classes = len(classes)
    if num_classes < 2:
        raise SystemExit(
            f"Could not resolve class names/size from class CSVs + checkpoint. "
            f"class_csvs={class_csvs}, inferred_num_classes={inferred_num_classes}"
        )

    # Prevent env overrides from loading external custom weights while building architecture.
    for env_key in ("CUSTOM_EFFNET_PATH", "CUSTOM_VIT_PATH", "CUSTOM_VIT_BACKBONE_PATH", "PRETRAINED_MODELS_DIR"):
        os.environ.pop(env_key, None)

    model_build_cfg: Dict = {"model_type": model_type, "dropout_rate": model_cfg.get("dropout_rate", 0.5)}
    if model_type == "effnet":
        model_build_cfg["enet_backbone"] = model_cfg.get("enet_backbone", "tf_efficientnet-b7_ns")
        model_build_cfg["pretrained_source"] = model_cfg.get("pretrained_source", "imagenet")
    elif model_type == "vit":
        model_build_cfg["vit_model_name"] = model_cfg.get("vit_model_name", "vit_base_patch16_224")
        model_build_cfg["pretrained_source"] = model_cfg.get("pretrained_source", "imagenet_timm")
    else:
        raise SystemExit(f"Unsupported model_type='{model_type}'")

    model = create_model(model_build_cfg, num_classes=num_classes, device=device).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = FolderDataset(
        image_paths=image_paths,
        img_size=img_size,
        resize_mode=resize_mode,
        prompt_labels=prompt_labels,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    idx_to_class = {i: c for i, c in enumerate(classes)}

    rows = []
    pred_counts = Counter()
    expected_counts = Counter()
    per_expected_correct = Counter()
    per_expected_total = Counter()
    confusion = defaultdict(Counter)

    with torch.no_grad():
        for xb, paths, prompt_idxs, expected_labels in dl:
            logits = model(xb.to(device, non_blocking=(device.type == "cuda")))
            probs = torch.softmax(logits, dim=1)
            confs, pred_idx = torch.max(probs, dim=1)

            for i in range(len(paths)):
                pred_label = idx_to_class[int(pred_idx[i].item())]
                confidence = float(confs[i].item())
                expected = expected_labels[i] if expected_labels[i] not in ("", None) else None
                prompt_idx = int(prompt_idxs[i].item()) if torch.is_tensor(prompt_idxs[i]) else prompt_idxs[i]

                pred_counts[pred_label] += 1
                if expected is not None:
                    expected_counts[expected] += 1
                    per_expected_total[expected] += 1
                    if pred_label == expected:
                        per_expected_correct[expected] += 1
                    confusion[expected][pred_label] += 1

                rows.append(
                    {
                        "image_path": paths[i],
                        "prompt_index": prompt_idx,
                        "expected_label": expected,
                        "pred_label": pred_label,
                        "confidence": confidence,
                    }
                )

    n = len(rows)
    if n == 0:
        raise SystemExit("No predictions generated.")

    expected_available = sum(1 for r in rows if r["expected_label"] is not None)
    correct = sum(1 for r in rows if r["expected_label"] is not None and r["expected_label"] == r["pred_label"])
    consistency = (correct / expected_available) if expected_available else None

    per_expected_acc = {}
    for k, total in per_expected_total.items():
        if total > 0:
            per_expected_acc[k] = per_expected_correct[k] / total

    summary = {
        "images_dir": str(images_dir),
        "checkpoint": str(ckpt_path),
        "study_config": str(study_cfg_path),
        "prompts_file": str(prompts_file),
        "device": str(device),
        "n_images": n,
        "n_with_expected_label": expected_available,
        "consistency_top1": consistency,
        "pred_distribution": dict(pred_counts),
        "expected_distribution": dict(expected_counts),
        "per_expected_top1": per_expected_acc,
        "confusion_expected_to_pred": {k: dict(v) for k, v in confusion.items()},
        "avg_confidence": float(np.mean([r["confidence"] for r in rows])),
    }

    default_stem = "classifier_eval_gpu" if device.type == "cuda" else "classifier_eval_cpu"
    out_json = Path(args.output_json).resolve() if args.output_json else (images_dir / f"{default_stem}.json")
    out_csv = Path(args.output_csv).resolve() if args.output_csv else (images_dir / f"{default_stem}.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "prompt_index", "expected_label", "pred_label", "confidence"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Classifier scoring finished on {device}.")
    print(f"Images: {n}")
    if consistency is None:
        print("Top-1 consistency: n/a (could not infer expected labels from prompts/file names)")
    else:
        print(f"Top-1 consistency: {consistency:.4f} ({correct}/{expected_available})")
    print("Predicted class distribution:", dict(pred_counts))
    print(f"Summary JSON: {out_json}")
    print(f"Predictions CSV: {out_csv}")


if __name__ == "__main__":
    main()
