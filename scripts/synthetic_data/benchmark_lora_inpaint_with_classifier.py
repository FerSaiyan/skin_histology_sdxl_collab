#!/usr/bin/env python3
"""
Benchmark Grad-CAM-guided SDXL inpainting for one or more LoRA checkpoints
using a classifier-as-judge setup.

For each selected source image, the script runs inpainting with:
- the source class prompt ("same_class"), and
- every other class prompt ("cross_class")

for every provided LoRA checkpoint using the same source images, masks,
prompts, and seeds. It then scores each output with the classifier and saves
side-by-side comparison panels:

  Original | Grad-CAM overlay | <model 1> | <model 2> | ...

The main metrics are:
- same_class_target_top1: classifier predicts the prompt/target class when the
  target matches the original source class
- cross_class_target_top1: classifier predicts the prompt/target class when the
  target differs from the original source class
- same_class_target_prob_mean: average classifier probability assigned to the
  target class for same-class prompts
- cross_class_target_source_prob_margin_mean: average classifier probability
  margin p(target) - p(source) for cross-class prompts
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from tqdm.auto import tqdm

from diffusers.image_processor import VaeImageProcessor

try:
    from diffusers import StableDiffusionXLInpaintPipeline
except Exception:
    StableDiffusionXLInpaintPipeline = None


DEFAULT_TOKENS = ["healthy", "benign_lesion", "opmd", "cancer"]
DEFAULT_DESCRIPTORS = {
    "healthy": "normal oral mucosa, no suspicious lesion",
    "benign_lesion": "small benign-appearing oral lesion with smooth borders",
    "opmd": "oral potentially malignant disorder, leukoplakia-like irregular plaque",
    "cancer": "ulcerated malignant-appearing oral lesion with irregular infiltrative margins",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


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


def _coerce_descriptors(raw: Any) -> Dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if raw is None:
        return dict(DEFAULT_DESCRIPTORS)

    text = str(raw).strip()
    if not text:
        return dict(DEFAULT_DESCRIPTORS)

    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None

    if isinstance(parsed, dict):
        return {str(k): str(v) for k, v in parsed.items()}

    raise ValueError(
        "--descriptors-json must be a JSON/Python dict mapping class names to descriptor strings."
    )


def _parse_descriptors(raw: str | None) -> Dict[str, str]:
    if not raw:
        return dict(DEFAULT_DESCRIPTORS)
    text = str(raw).strip()
    if not text:
        return dict(DEFAULT_DESCRIPTORS)
    try:
        p = Path(text)
        if p.is_file():
            return _coerce_descriptors(p.read_text(encoding="utf-8"))
    except OSError:
        pass
    return _coerce_descriptors(text)


def _parse_float_map(raw: str | None) -> Dict[str, float]:
    if not raw:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        p = Path(text)
        if p.is_file():
            payload = json.loads(p.read_text(encoding="utf-8"))
        else:
            payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Could not parse float mapping: {raw}") from exc
    except OSError:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON/Python dict for float mapping.")
    out: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            out[str(key)] = float(value)
        except Exception as exc:
            raise ValueError(f"Invalid float mapping value for key '{key}': {value}") from exc
    return out


def _prompt_text(template: str, token: str, descriptors: Dict[str, str]) -> str:
    descriptor = str(descriptors.get(token, "")).strip()
    descriptor_suffix = f", {descriptor}" if descriptor else ""
    return template.format(
        token=token,
        descriptor=descriptor,
        descriptor_suffix=descriptor_suffix,
    ).strip()


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
    inferred_num_classes: Optional[int],
    label_col: str,
    fallback_tokens: List[str],
) -> List[str]:
    classes = _build_classes(class_csvs, label_col=label_col)
    if inferred_num_classes is None:
        return classes or sorted(fallback_tokens)
    if classes and len(classes) == inferred_num_classes:
        return classes
    fallback_unique = sorted(set(fallback_tokens))
    if len(fallback_unique) == inferred_num_classes:
        return fallback_unique
    if inferred_num_classes == 4:
        return sorted(DEFAULT_TOKENS)
    return [f"class_{i}" for i in range(inferred_num_classes)]


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
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


@dataclass
class ClassifierBundle:
    model: torch.nn.Module
    device: torch.device
    idx_to_class: Dict[int, str]
    class_to_idx: Dict[str, int]
    img_size: int
    resize_mode: str
    checkpoint_path: Path
    study_config_path: Path


def _load_classifier(
    checkpoint_path: Path,
    study_config_path: Path,
    fallback_tokens: List[str],
    label_col: str,
    device: torch.device,
) -> ClassifierBundle:
    cfg = load_config(str(study_config_path))
    study_cfg = cfg.get("study", {}) or {}
    model_cfg = (cfg.get("models", []) or [None])[0]
    if model_cfg is None:
        raise SystemExit(f"No models section found in study config: {study_config_path}")

    class_csvs: List[Path] = []
    for k in ("train_csv_path", "val_csv_path", "test_csv_path"):
        raw = study_cfg.get(k)
        if not raw:
            continue
        p = Path(str(raw))
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        class_csvs.append(p)

    raw_state = torch.load(checkpoint_path, map_location=device)
    state = _normalize_state_dict(raw_state)
    inferred_num_classes = _infer_num_classes(state) if isinstance(state, dict) else None
    classes = _resolve_class_names(class_csvs, inferred_num_classes, label_col=label_col, fallback_tokens=fallback_tokens)
    if len(classes) < 2:
        raise SystemExit(f"Could not resolve classifier classes for checkpoint: {checkpoint_path}")

    for env_key in ("CUSTOM_EFFNET_PATH", "CUSTOM_VIT_PATH", "CUSTOM_VIT_BACKBONE_PATH", "PRETRAINED_MODELS_DIR"):
        os.environ.pop(env_key, None)

    model_type = str(model_cfg.get("model_type", "effnet"))
    model_build_cfg: Dict = {"model_type": model_type, "dropout_rate": model_cfg.get("dropout_rate", 0.5)}
    if model_type == "effnet":
        model_build_cfg["enet_backbone"] = model_cfg.get("enet_backbone", "tf_efficientnet-b7_ns")
        model_build_cfg["pretrained_source"] = model_cfg.get("pretrained_source", "imagenet")
    elif model_type == "vit":
        model_build_cfg["vit_model_name"] = model_cfg.get("vit_model_name", "vit_base_patch16_224")
        model_build_cfg["pretrained_source"] = model_cfg.get("pretrained_source", "imagenet_timm")
    else:
        raise SystemExit(f"Unsupported model_type='{model_type}'")

    model = create_model(model_build_cfg, num_classes=len(classes), device=device).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    return ClassifierBundle(
        model=model,
        device=device,
        idx_to_class={i: c for i, c in enumerate(classes)},
        class_to_idx={c: i for i, c in enumerate(classes)},
        img_size=int(model_cfg.get("img_size", 384 if model_type == "effnet" else 224)),
        resize_mode=str(study_cfg.get("resize_mode", "preserve")),
        checkpoint_path=checkpoint_path,
        study_config_path=study_config_path,
    )


def _predict_with_scores(
    bundle: ClassifierBundle,
    image: Image.Image,
    tracked_labels: Sequence[str],
) -> tuple[str, float, Dict[str, float]]:
    x = _preprocess_pil(image.convert("RGB"), img_size=bundle.img_size, resize_mode=bundle.resize_mode)
    xb = x.unsqueeze(0).to(bundle.device)
    with torch.no_grad():
        logits = bundle.model(xb)
        probs = torch.softmax(logits, dim=1)
        confs, pred_idx = torch.max(probs, dim=1)
    probs_vec = probs[0].detach().cpu()
    pred_label = bundle.idx_to_class[int(pred_idx[0].item())]
    tracked = {}
    for label in tracked_labels:
        idx = bundle.class_to_idx.get(str(label))
        if idx is None:
            continue
        tracked[str(label)] = float(probs_vec[idx].item())
    return pred_label, float(confs[0].item()), tracked


def _safe_stem(path: Path) -> str:
    keep = []
    for ch in path.stem:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "image"


def _mask_name_for_row(row: pd.Series) -> str:
    if "filename" in row.index and str(row["filename"]).strip():
        return str(row["filename"]).strip()
    return Path(str(row["image_path"])).name


def _make_overlay(image: Image.Image, mask: Image.Image, alpha: float = 0.40) -> Image.Image:
    img_arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    overlay = img_arr.copy()
    overlay[..., 0] = overlay[..., 0] * (1.0 - alpha * mask_arr) + 255.0 * alpha * mask_arr
    overlay[..., 1] = overlay[..., 1] * (1.0 - alpha * mask_arr)
    overlay[..., 2] = overlay[..., 2] * (1.0 - alpha * mask_arr)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def _dilate_mask(mask: Image.Image, dilate_px: float) -> Image.Image:
    radius = max(0, int(round(float(dilate_px))))
    if radius <= 0:
        return mask
    return mask.filter(ImageFilter.MaxFilter(size=radius * 2 + 1))


def _prepare_mask_variant(
    *,
    image: Image.Image,
    raw_mask: Image.Image,
    feather_radius: float,
    dilate_px: float,
) -> dict:
    mask = _dilate_mask(raw_mask, dilate_px=dilate_px)
    mask_soft = mask.filter(ImageFilter.GaussianBlur(radius=float(feather_radius))) if feather_radius > 0 else mask
    return {
        "mask_soft": mask_soft,
        "overlay": _make_overlay(image, mask_soft),
    }


def _load_source_inputs(
    image_path: Path,
    mask_path: Path,
    width: int,
    height: int,
) -> dict:
    image = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    mask = Image.open(mask_path).convert("L").resize((width, height), Image.LANCZOS)
    return {
        "image": image,
        "raw_mask": mask,
    }


def _panel_title(row: Optional[dict], model_alias: str) -> str:
    if row is None:
        return f"{model_alias}\nmissing"
    return (
        f"{model_alias}\n"
        f"target={row['target_label']} pred={row['pred_label']}\n"
        f"conf={row['confidence']:.3f}"
    )


def _pretty_model_title(alias: str) -> str:
    parts = str(alias).replace("-", "_").split("_")
    return " ".join(p.capitalize() for p in parts if p) or str(alias)


def _save_panel(
    out_path: Path,
    source_image: Image.Image,
    overlay_image: Image.Image,
    model_rows: Sequence[Tuple[str, Optional[dict]]],
) -> None:
    fig_cols = 2 + len(model_rows)
    fig, axes = plt.subplots(1, fig_cols, figsize=(4.5 * fig_cols, 5))
    axes[0].imshow(source_image)
    axes[0].set_title("Original")
    axes[1].imshow(overlay_image)
    axes[1].set_title("Grad-CAM overlay")

    for idx, (model_alias, row) in enumerate(model_rows, start=2):
        model_img = Image.open(row["generated_path"]).convert("RGB") if row else Image.new("RGB", source_image.size)
        axes[idx].imshow(model_img)
        axes[idx].set_title(_panel_title(row, _pretty_model_title(model_alias)))
    for ax in axes:
        ax.axis("off")

    first_row = next((row for _, row in model_rows if row is not None), None)
    if first_row is None:
        raise RuntimeError("Panel rendering received no populated model rows.")
    target = first_row["target_label"]
    source_label = first_row["source_label"]
    mode = first_row["prompt_mode"]
    fig.suptitle(f"source={source_label} -> target={target} ({mode})", fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _seed_for_task(seed_base: int, source_idx: int, target_idx: int) -> int:
    return int(seed_base + source_idx * 1000 + target_idx)


def _sample_sources(
    df: pd.DataFrame,
    tokens: List[str],
    label_col: str,
    mask_dir: Path,
    sources_per_class: int,
    max_sources: int,
    random_state: int,
) -> pd.DataFrame:
    def _has_mask(row: pd.Series) -> bool:
        return (mask_dir / _mask_name_for_row(row)).is_file()

    if "image_path" not in df.columns:
        raise ValueError("Expected 'image_path' column in labels CSV")
    if label_col not in df.columns:
        raise ValueError(f"Expected '{label_col}' column in labels CSV")

    df = df[df[label_col].astype(str).isin(tokens)].copy()
    df = df[df.apply(_has_mask, axis=1)].copy()
    if df.empty:
        raise RuntimeError(f"No source images with masks found in {mask_dir}")

    rng = np.random.RandomState(int(random_state))
    chosen_parts = []
    if int(sources_per_class) > 0:
        for token in tokens:
            subset = df[df[label_col].astype(str) == token]
            if subset.empty:
                continue
            take_n = min(int(sources_per_class), len(subset))
            take = subset.sample(n=take_n, random_state=int(rng.randint(0, 2**31 - 1)))
            chosen_parts.append(take)
        if not chosen_parts:
            raise RuntimeError("Balanced sampling found no eligible source images.")
        chosen = pd.concat(chosen_parts, ignore_index=True)
    else:
        chosen = df.copy()

    chosen = chosen.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    chosen["mask_name"] = chosen.apply(_mask_name_for_row, axis=1)
    if int(max_sources) > 0 and len(chosen) > int(max_sources):
        chosen = chosen.sample(n=int(max_sources), random_state=random_state).reset_index(drop=True)
    return chosen


def _metric_summary(rows: List[dict], prompt_mode: str) -> dict:
    subset = [r for r in rows if r["prompt_mode"] == prompt_mode]
    n = len(subset)
    correct = sum(1 for r in subset if r["pred_label"] == r["target_label"])
    target_probs = [float(r["target_prob"]) for r in subset if r.get("target_prob") is not None]
    source_probs = [float(r["source_prob"]) for r in subset if r.get("source_prob") is not None]
    margins = [float(r["target_source_prob_margin"]) for r in subset if r.get("target_source_prob_margin") is not None]
    per_target_total = Counter(r["target_label"] for r in subset)
    per_target_correct = Counter(r["target_label"] for r in subset if r["pred_label"] == r["target_label"])
    per_source_total = Counter(r["source_label"] for r in subset)
    per_source_correct = Counter(r["source_label"] for r in subset if r["pred_label"] == r["target_label"])
    confusion = defaultdict(Counter)
    for row in subset:
        confusion[row["target_label"]][row["pred_label"]] += 1
    margin_mean = float(np.mean(margins)) if margins else None
    return {
        "n": n,
        "top1": (correct / n) if n else None,
        "target_prob_mean": float(np.mean(target_probs)) if target_probs else None,
        "source_prob_mean": float(np.mean(source_probs)) if source_probs else None,
        "target_source_prob_margin_mean": margin_mean,
        "target_source_prob_margin_score": (0.5 * (margin_mean + 1.0)) if margin_mean is not None else None,
        "positive_margin_rate": (sum(1 for margin in margins if margin > 0.0) / len(margins)) if margins else None,
        "per_target_top1": {
            k: (per_target_correct[k] / per_target_total[k]) for k in sorted(per_target_total)
        },
        "per_source_class_top1": {
            k: (per_source_correct[k] / per_source_total[k]) for k in sorted(per_source_total)
        },
        "confusion_target_to_pred": {k: dict(v) for k, v in sorted(confusion.items())},
    }


def _parse_model_specs(model_args: List[str], legacy_pairs: List[Tuple[str, Optional[str]]]) -> List[Tuple[str, Path]]:
    specs: List[Tuple[str, Path]] = []
    seen = set()
    for raw in model_args or []:
        if "=" not in raw:
            raise SystemExit(f"Invalid --model value '{raw}'. Expected alias=/path/to/checkpoint.safetensors")
        alias, raw_path = raw.split("=", 1)
        alias = alias.strip()
        raw_path = raw_path.strip()
        if not alias or not raw_path:
            raise SystemExit(f"Invalid --model value '{raw}'. Expected alias=/path/to/checkpoint.safetensors")
        if alias in seen:
            raise SystemExit(f"Duplicate model alias in --model arguments: {alias}")
        specs.append((alias, Path(raw_path).resolve()))
        seen.add(alias)

    if specs:
        return specs

    for alias, raw_path in legacy_pairs:
        if raw_path in (None, "", "null"):
            continue
        if alias in seen:
            continue
        specs.append((alias, Path(str(raw_path)).resolve()))
        seen.add(alias)
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark one or more LoRA checkpoints with Grad-CAM inpainting")
    ap.add_argument("--labels-csv", required=True, help="CSV with image_path and source label column")
    ap.add_argument("--label-col", default="coarse_label", help="Source label column in labels CSV")
    ap.add_argument("--mask-dir", required=True, help="Directory with Grad-CAM masks aligned by source filename")
    ap.add_argument("--base-model", required=True, help="SDXL base or inpaint checkpoint (.safetensors)")
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model spec in the form alias=/path/to/checkpoint.safetensors. Can be passed multiple times.",
    )
    ap.add_argument("--phase1-lora", default=None, help="Legacy alias for Phase 1 LoRA checkpoint (.safetensors)")
    ap.add_argument("--phase2-lora", default=None, help="Legacy alias for Phase 2 LoRA checkpoint (.safetensors)")
    ap.add_argument("--phase3-lora", default=None, help="Legacy alias for Phase 3 LoRA checkpoint (.safetensors)")
    ap.add_argument("--classifier-ckpt", required=True, help="Classifier checkpoint (.pth)")
    ap.add_argument("--classifier-study-config", required=True, help="Classifier study YAML")
    ap.add_argument("--out-dir", required=True, help="Output directory for benchmark artifacts")
    ap.add_argument("--device", default="auto", help="Inpainting device: auto|cuda|cpu")
    ap.add_argument("--classifier-device", default="cpu", help="Classifier device: auto|cuda|cpu")
    ap.add_argument("--tokens", default=",".join(DEFAULT_TOKENS), help="Comma-separated class order")
    ap.add_argument(
        "--prompt-template",
        default="clinical intraoral photo, diagnosis: {token}{descriptor_suffix}",
        help="Prompt template with {token} and optional {descriptor_suffix}",
    )
    ap.add_argument("--descriptors-json", default=None, help="JSON string or file with token->descriptor map")
    ap.add_argument("--negative-prompt", default="lowres,bad anatomy")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument(
        "--same-class-strength",
        type=float,
        default=None,
        help="Optional same-class inpaint strength override (defaults to --strength).",
    )
    ap.add_argument(
        "--cross-class-strength",
        type=float,
        default=None,
        help="Optional cross-class inpaint strength override (defaults to --strength).",
    )
    ap.add_argument("--lora-scale", type=float, default=0.75)
    ap.add_argument("--mask-feather-radius", type=float, default=6.0)
    ap.add_argument(
        "--same-class-mask-dilate-px",
        type=float,
        default=0.0,
        help="Optional mask dilation radius in pixels for same-class prompts.",
    )
    ap.add_argument(
        "--cross-class-mask-dilate-px",
        type=float,
        default=0.0,
        help="Optional mask dilation radius in pixels for cross-class prompts.",
    )
    ap.add_argument(
        "--cross-class-target-strengths-json",
        default=None,
        help="Optional JSON/file mapping target_label->strength override for cross-class prompts.",
    )
    ap.add_argument(
        "--cross-class-target-mask-dilate-json",
        default=None,
        help="Optional JSON/file mapping target_label->mask dilation override for cross-class prompts.",
    )
    ap.add_argument("--sources-per-class", type=int, default=2)
    ap.add_argument("--max-sources", type=int, default=0)
    ap.add_argument("--seed-base", type=int, default=222)
    ap.add_argument(
        "--selected-sources-csv",
        default=None,
        help="Optional precomputed selected_sources.csv to reuse across repeated benchmark runs.",
    )
    ap.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Ignore existing generated images and rerun inpainting.",
    )
    args = ap.parse_args()

    if StableDiffusionXLInpaintPipeline is None:
        raise SystemExit("StableDiffusionXLInpaintPipeline is unavailable in the current diffusers install.")

    labels_csv = Path(args.labels_csv).resolve()
    mask_dir = Path(args.mask_dir).resolve()
    base_model = Path(args.base_model).resolve()
    classifier_ckpt = Path(args.classifier_ckpt).resolve()
    classifier_cfg = Path(args.classifier_study_config).resolve()
    out_dir = Path(args.out_dir).resolve()
    selected_sources_override = (
        Path(args.selected_sources_csv).resolve() if args.selected_sources_csv not in (None, "", "null") else None
    )
    model_specs = _parse_model_specs(
        model_args=args.model,
        legacy_pairs=[
            ("phase1", args.phase1_lora),
            ("phase2", args.phase2_lora),
            ("phase3", args.phase3_lora),
        ],
    )
    if not model_specs:
        raise SystemExit("No LoRA checkpoints provided. Pass --model alias=/path or a legacy --phase*-lora argument.")

    for p, label in (
        (labels_csv, "--labels-csv"),
        (mask_dir, "--mask-dir"),
        (base_model, "--base-model"),
        (classifier_ckpt, "--classifier-ckpt"),
        (classifier_cfg, "--classifier-study-config"),
    ):
        if not p.exists():
            raise SystemExit(f"{label} not found: {p}")
    for alias, ckpt_path in model_specs:
        if not ckpt_path.exists():
            raise SystemExit(f"--model {alias} not found: {ckpt_path}")

    tokens = [t.strip() for t in str(args.tokens).split(",") if t.strip()]
    if not tokens:
        raise SystemExit("No tokens provided.")
    descriptors = _parse_descriptors(args.descriptors_json)
    cross_class_target_strengths = _parse_float_map(args.cross_class_target_strengths_json)
    cross_class_target_mask_dilate = _parse_float_map(args.cross_class_target_mask_dilate_json)
    same_class_strength = float(args.same_class_strength) if args.same_class_strength is not None else float(args.strength)
    cross_class_strength = (
        float(args.cross_class_strength) if args.cross_class_strength is not None else float(args.strength)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generated").mkdir(parents=True, exist_ok=True)
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)

    selected_sources_csv = out_dir / "selected_sources.csv"
    if selected_sources_override is not None and selected_sources_override.is_file():
        selected = pd.read_csv(selected_sources_override)
        if "mask_name" not in selected.columns:
            selected["mask_name"] = selected.apply(_mask_name_for_row, axis=1)
        if "source_index" not in selected.columns:
            selected = selected.reset_index(drop=True)
            selected["source_index"] = np.arange(len(selected))
        if selected_sources_csv != selected_sources_override:
            selected.to_csv(selected_sources_csv, index=False)
        print(f"Reusing selected sources from override: {selected_sources_override}")
    elif selected_sources_csv.is_file():
        selected = pd.read_csv(selected_sources_csv)
        if "mask_name" not in selected.columns:
            selected["mask_name"] = selected.apply(_mask_name_for_row, axis=1)
        if "source_index" not in selected.columns:
            selected = selected.reset_index(drop=True)
            selected["source_index"] = np.arange(len(selected))
        print(f"Reusing selected sources from: {selected_sources_csv}")
    else:
        source_df = pd.read_csv(labels_csv)
        selected = _sample_sources(
            source_df,
            tokens=tokens,
            label_col=args.label_col,
            mask_dir=mask_dir,
            sources_per_class=args.sources_per_class,
            max_sources=args.max_sources,
            random_state=args.seed_base,
        )
        selected = selected.reset_index(drop=True)
        selected["source_index"] = np.arange(len(selected))
        selected.to_csv(selected_sources_csv, index=False)

    tasks: List[dict] = []
    for _, row in selected.iterrows():
        source_label = str(row[args.label_col])
        source_idx = int(row["source_index"])
        source_key = f"{source_idx:04d}_{_safe_stem(Path(str(row['image_path'])))}"
        for target_idx, target_label in enumerate(tokens):
            is_same = target_label == source_label
            task_strength = same_class_strength if is_same else cross_class_target_strengths.get(target_label, cross_class_strength)
            task_mask_dilate = (
                float(args.same_class_mask_dilate_px)
                if is_same
                else cross_class_target_mask_dilate.get(target_label, float(args.cross_class_mask_dilate_px))
            )
            tasks.append(
                {
                    "source_index": source_idx,
                    "source_key": source_key,
                    "source_path": str(Path(str(row["image_path"])).resolve()),
                    "mask_path": str((mask_dir / str(row["mask_name"])).resolve()),
                    "mask_name": str(row["mask_name"]),
                    "source_label": source_label,
                    "target_label": target_label,
                    "target_index": target_idx,
                    "prompt_mode": "same_class" if is_same else "cross_class",
                    "prompt": _prompt_text(args.prompt_template, target_label, descriptors),
                    "seed": _seed_for_task(args.seed_base, source_idx, target_idx),
                    "strength": task_strength,
                    "mask_dilate_px": task_mask_dilate,
                }
            )

    if not tasks:
        raise SystemExit("No benchmark tasks were created.")

    print(f"Selected source images: {len(selected)}")
    print(f"Per-model tasks        : {len(tasks)}")
    print(f"Models                 : {', '.join(alias for alias, _ in model_specs)}")
    print(f"Total generations      : {len(tasks) * len(model_specs)}")
    print(f"Output dir             : {out_dir}")

    device = _resolve_device(args.device)
    classifier_device = _resolve_device(args.classifier_device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    classifier = _load_classifier(
        checkpoint_path=classifier_ckpt,
        study_config_path=classifier_cfg,
        fallback_tokens=tokens,
        label_col=args.label_col,
        device=classifier_device,
    )

    pipe = StableDiffusionXLInpaintPipeline.from_single_file(
        str(base_model),
        torch_dtype=dtype,
        safety_checker=None,
        use_safetensors=True,
    )
    pipe = pipe.to(device)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    pipe.set_progress_bar_config(disable=True)
    pipe.mask_processor = VaeImageProcessor(
        vae_scale_factor=pipe.vae_scale_factor,
        do_normalize=False,
        do_binarize=False,
        do_convert_grayscale=True,
    )

    cached_inputs: Dict[str, dict] = {}
    rows: List[dict] = []
    checkpoints = list(model_specs)

    try:
        for model_alias, ckpt_path in checkpoints:
            model_out_dir = out_dir / "generated" / model_alias
            model_out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[{model_alias}] loading LoRA: {ckpt_path}")
            pipe.load_lora_weights(str(ckpt_path.parent), weight_name=ckpt_path.name)
            try:
                pbar = tqdm(tasks, desc=f"{model_alias} inpaint", unit="image")
                for task in pbar:
                    source_key = str(task["source_key"])
                    file_name = f"{source_key}__to__{task['target_label']}.png"
                    gen_path = model_out_dir / file_name
                    if source_key not in cached_inputs:
                        cached_inputs[source_key] = _load_source_inputs(
                            image_path=Path(task["source_path"]),
                            mask_path=Path(task["mask_path"]),
                            width=int(args.width),
                            height=int(args.height),
                        )
                    source_inputs = cached_inputs[source_key]
                    mask_variant = _prepare_mask_variant(
                        image=source_inputs["image"],
                        raw_mask=source_inputs["raw_mask"],
                        feather_radius=float(args.mask_feather_radius),
                        dilate_px=float(task["mask_dilate_px"]),
                    )
                    reused_existing = False
                    if gen_path.is_file() and not args.force_regenerate:
                        gen = Image.open(gen_path).convert("RGB")
                        reused_existing = True
                    else:
                        generator = torch.Generator(device=device.type).manual_seed(int(task["seed"]))
                        result = pipe(
                            prompt=str(task["prompt"]),
                            negative_prompt=args.negative_prompt,
                            image=source_inputs["image"],
                            mask_image=mask_variant["mask_soft"],
                            strength=float(task["strength"]),
                            guidance_scale=float(args.guidance_scale),
                            num_inference_steps=int(args.steps),
                            generator=generator,
                            cross_attention_kwargs={"scale": float(args.lora_scale)},
                        )
                        gen = result.images[0].convert("RGB")
                        gen.save(gen_path)

                    pred_label, confidence, tracked_probs = _predict_with_scores(
                        classifier,
                        gen,
                        tracked_labels=[str(task["target_label"]), str(task["source_label"])],
                    )
                    target_prob = tracked_probs.get(str(task["target_label"]))
                    source_prob = tracked_probs.get(str(task["source_label"]))
                    target_source_prob_margin = None
                    if (
                        task["target_label"] != task["source_label"]
                        and target_prob is not None
                        and source_prob is not None
                    ):
                        target_source_prob_margin = float(target_prob - source_prob)
                    row = dict(task)
                    row.update(
                        {
                            "model_alias": model_alias,
                            "lora_checkpoint": str(ckpt_path),
                            "generated_path": str(gen_path),
                            "pred_label": pred_label,
                            "confidence": confidence,
                            "target_prob": target_prob,
                            "source_prob": source_prob,
                            "target_source_prob_margin": target_source_prob_margin,
                            "target_correct": bool(pred_label == task["target_label"]),
                            "reused_existing": reused_existing,
                        }
                    )
                    rows.append(row)
                    pbar.set_postfix(
                        target=task["target_label"],
                        correct=int(row["target_correct"]),
                        reused=int(reused_existing),
                    )
            finally:
                pipe.unload_lora_weights()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows_csv = out_dir / "benchmark_rows.csv"
    with rows_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_alias",
                "source_index",
                "source_key",
                "source_path",
                "mask_path",
                "mask_name",
                "source_label",
                "target_label",
                "target_index",
                "prompt_mode",
                "prompt",
                "seed",
                "strength",
                "mask_dilate_px",
                "lora_checkpoint",
                "generated_path",
                "pred_label",
                "confidence",
                "target_prob",
                "source_prob",
                "target_source_prob_margin",
                "target_correct",
                "reused_existing",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    rows_by_key = defaultdict(dict)
    for row in rows:
        group_key = (row["source_key"], row["target_label"])
        rows_by_key[group_key][row["model_alias"]] = row

    panel_rows = []
    panel_pbar = tqdm(sorted(rows_by_key.items()), desc="render panels", unit="panel")
    for (source_key, target_label), group in panel_pbar:
        sample_row = next((group.get(alias) for alias, _ in checkpoints if group.get(alias) is not None), None)
        source_inputs = cached_inputs[source_key]
        overlay_image = _prepare_mask_variant(
            image=source_inputs["image"],
            raw_mask=source_inputs["raw_mask"],
            feather_radius=float(args.mask_feather_radius),
            dilate_px=float(sample_row.get("mask_dilate_px", 0.0)),
        )["overlay"]
        panel_mode = sample_row["prompt_mode"]
        panel_path = out_dir / "panels" / panel_mode / f"{source_key}__to__{target_label}.png"
        _save_panel(
            out_path=panel_path,
            source_image=source_inputs["image"],
            overlay_image=overlay_image,
            model_rows=[(alias, group.get(alias)) for alias, _ in checkpoints],
        )
        panel_rows.append(
            {
                "source_key": source_key,
                "target_label": target_label,
                "prompt_mode": panel_mode,
                "panel_path": str(panel_path),
            }
        )

    panel_csv = out_dir / "panel_index.csv"
    with panel_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_key", "target_label", "prompt_mode", "panel_path"])
        writer.writeheader()
        writer.writerows(panel_rows)

    summary = {
        "benchmark_type": "gradcam_inpaint_lora_multi_model",
        "labels_csv": str(labels_csv),
        "label_col": args.label_col,
        "mask_dir": str(mask_dir),
        "base_model": str(base_model),
        "models": [{"alias": alias, "checkpoint": str(path)} for alias, path in checkpoints],
        "classifier_ckpt": str(classifier_ckpt),
        "classifier_study_config": str(classifier_cfg),
        "classifier_device": str(classifier_device),
        "generation_device": str(device),
        "tokens": tokens,
        "selected_sources_csv": str(out_dir / "selected_sources.csv"),
        "rows_csv": str(rows_csv),
        "panel_index_csv": str(panel_csv),
        "n_sources": int(len(selected)),
        "n_tasks_per_model": int(len(tasks)),
        "n_generated_total": int(len(rows)),
        "settings": {
            "width": int(args.width),
            "height": int(args.height),
            "steps": int(args.steps),
            "guidance_scale": float(args.guidance_scale),
            "strength": float(args.strength),
            "same_class_strength": same_class_strength,
            "cross_class_strength": cross_class_strength,
            "cross_class_target_strengths": cross_class_target_strengths,
            "lora_scale": float(args.lora_scale),
            "mask_feather_radius": float(args.mask_feather_radius),
            "same_class_mask_dilate_px": float(args.same_class_mask_dilate_px),
            "cross_class_mask_dilate_px": float(args.cross_class_mask_dilate_px),
            "cross_class_target_mask_dilate": cross_class_target_mask_dilate,
            "negative_prompt": args.negative_prompt,
            "prompt_template": args.prompt_template,
            "seed_base": int(args.seed_base),
        },
        "metrics_by_model": {},
    }

    for model_alias, _ in checkpoints:
        model_rows = [r for r in rows if r["model_alias"] == model_alias]
        same_metrics = _metric_summary(model_rows, "same_class")
        cross_metrics = _metric_summary(model_rows, "cross_class")
        summary["metrics_by_model"][model_alias] = {
            "n_rows": len(model_rows),
            "same_class_target_top1": same_metrics,
            "cross_class_target_top1": cross_metrics,
            "avg_confidence": float(np.mean([r["confidence"] for r in model_rows])) if model_rows else None,
            "pred_distribution": dict(Counter(r["pred_label"] for r in model_rows)),
        }

    summary_json = out_dir / "benchmark_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSummary:")
    for model_alias, metrics in summary["metrics_by_model"].items():
        same_top1 = metrics["same_class_target_top1"]["top1"]
        cross_top1 = metrics["cross_class_target_top1"]["top1"]
        same_prob = metrics["same_class_target_top1"]["target_prob_mean"]
        cross_margin = metrics["cross_class_target_top1"]["target_source_prob_margin_mean"]
        same_str = f"{same_top1:.4f}" if same_top1 is not None else "n/a"
        cross_str = f"{cross_top1:.4f}" if cross_top1 is not None else "n/a"
        same_prob_str = f"{same_prob:.4f}" if same_prob is not None else "n/a"
        cross_margin_str = f"{cross_margin:.4f}" if cross_margin is not None else "n/a"
        print(
            f"  {model_alias}: same_top1={same_str} same_target_prob={same_prob_str} | "
            f"cross_top1={cross_str} cross_margin={cross_margin_str}"
        )
    print(f"Saved rows   : {rows_csv}")
    print(f"Saved panels : {out_dir / 'panels'}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
