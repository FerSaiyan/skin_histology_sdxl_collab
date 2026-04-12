#!/usr/bin/env python3
"""
Phase 3 reward-guided LoRA continuation focused on Grad-CAM inpainting.

This loop keeps training in masked-loss / inpaint mode, then after each short
train chunk it scores candidate checkpoints with the Grad-CAM inpainting
benchmark. Checkpoints are ranked by a weighted reward that prioritizes:

1. cross_class_target_top1
2. same_class_target_top1

The same-class term is kept in the reward so the model does not drift away from
the "add variety without changing diagnosis" behavior while being optimized for
cross-class lesion morphing.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


def _repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
        if (parent / "dvc.yaml").exists():
            return parent
    return Path.cwd().resolve()


REPO = _repo_root(Path(__file__))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.exp.config import load_config


DEFAULT_TOKENS = ["healthy", "benign_lesion", "opmd", "cancer"]
DEFAULT_DESCRIPTORS = {
    "healthy": "normal oral mucosa, no suspicious lesion",
    "benign_lesion": "small benign-appearing oral lesion with smooth borders",
    "opmd": "oral potentially malignant disorder, leukoplakia-like irregular plaque",
    "cancer": "ulcerated malignant-appearing oral lesion with irregular infiltrative margins",
}


def _run(cmd: List[str], cwd: Optional[Path] = None, dry_run: bool = False) -> None:
    line = " ".join(cmd)
    print(f">> {line}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _resolve_path(raw: Any, cfg_dir: Path, project_root: Path) -> Path:
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p.resolve()
    from_cfg = (cfg_dir / p).resolve()
    if from_cfg.exists():
        return from_cfg
    return (project_root / p).resolve()


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_optional_float(x: Any) -> Optional[float]:
    if x in (None, "", "null"):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _stable_sort_ckpts(paths: List[Path]) -> List[Path]:
    def _key(p: Path) -> Tuple[int, float, str]:
        return (0, p.stat().st_mtime, p.name)

    return sorted(paths, key=_key)


def _selector_checkpoint_names(produced_ckpts: List[Path], *, include_last: bool) -> str:
    ordered = _stable_sort_ckpts(produced_ckpts)
    if len(ordered) > 1 and not include_last:
        ordered = [p for p in ordered if p.name != "last.safetensors"] or ordered
    return ",".join(p.name for p in ordered)


def _maybe_write_descriptors(
    selector_cfg: Dict[str, Any], work_dir: Path, cfg_dir: Path, project_root: Path
) -> Optional[Path]:
    raw = selector_cfg.get("descriptors_json", None)
    if raw is None:
        return None
    if isinstance(raw, dict):
        out = work_dir / "selector_descriptors.json"
        out.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        return out
    text = str(raw).strip()
    if not text:
        return None
    try:
        p = _resolve_path(text, cfg_dir=cfg_dir, project_root=project_root)
        if p.is_file():
            return p
    except OSError:
        p = None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
    if isinstance(parsed, dict):
        out = work_dir / "selector_descriptors.json"
        out.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        return out

    out = work_dir / "selector_descriptors.json"
    out.write_text(text, encoding="utf-8")
    return out


def _maybe_write_json_arg(raw: Any, out_path: Path) -> Optional[str]:
    if raw in (None, "", "null"):
        return None
    if isinstance(raw, (dict, list)):
        out_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        return str(out_path)
    text = str(raw).strip()
    if not text:
        return None
    return text


def _load_descriptors_map(raw: Any, descriptors_path: Optional[Path]) -> Dict[str, str]:
    if descriptors_path is not None and descriptors_path.is_file():
        try:
            payload = json.loads(descriptors_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {str(k): str(v) for k, v in payload.items()}
        except Exception:
            pass
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    return dict(DEFAULT_DESCRIPTORS)


def _load_repeat_map(raw: Any) -> Dict[str, int]:
    if raw in (None, "", "null"):
        return {}
    payload: Any = raw
    if isinstance(raw, Path):
        payload = json.loads(raw.read_text(encoding="utf-8"))
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            p = Path(text)
            if p.is_file():
                payload = json.loads(p.read_text(encoding="utf-8"))
            else:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = ast.literal_eval(text)
        except OSError:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = ast.literal_eval(text)
    if not isinstance(payload, dict):
        raise SystemExit("Repeat map must resolve to a dict.")
    out: Dict[str, int] = {}
    for k, v in payload.items():
        out[str(k)] = max(0, int(v))
    return out


def _normalize_token(raw: Any) -> str:
    token = str(raw or "").strip().casefold().replace(" ", "_")
    aliases = {
        "healthy": "healthy",
        "benign_lesion": "benign_lesion",
        "benignlesion": "benign_lesion",
        "benign-lesion": "benign_lesion",
        "opmd": "opmd",
        "cancer": "cancer",
    }
    token = aliases.get(token, token)
    return token if token in DEFAULT_TOKENS else "benign_lesion"


def _safe_name(raw: str) -> str:
    keep = []
    for ch in str(raw):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "item"


def _sample_balanced_df(
    df: pd.DataFrame,
    *,
    label_col: str,
    max_per_class: int,
    random_state: int,
) -> pd.DataFrame:
    if max_per_class <= 0:
        return df.reset_index(drop=True)
    rng = np.random.RandomState(int(random_state))
    parts = []
    for label in sorted(df[label_col].astype(str).unique()):
        subset = df[df[label_col].astype(str) == label]
        if subset.empty:
            continue
        take_n = min(int(max_per_class), len(subset))
        parts.append(subset.sample(n=take_n, random_state=int(rng.randint(0, 2**31 - 1))))
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True).reset_index(drop=True)


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _find_running_training_processes(path_hint: Path) -> List[Tuple[int, str]]:
    target = str(path_hint.resolve())
    current_pid = os.getpid()
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    except Exception:
        return []
    hits: List[Tuple[int, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        args = parts[1]
        if pid == current_pid:
            continue
        if "sdxl_train_network.py" not in args and "finetune_stable_diffusion_unified.py" not in args:
            continue
        if target not in args:
            continue
        hits.append((pid, args))
    return hits


def _raise_if_training_running(path_hint: Path, label: str) -> None:
    hits = _find_running_training_processes(path_hint)
    if not hits:
        return
    details = "\n".join(f"  pid={pid} {args}" for pid, args in hits[:6])
    raise SystemExit(
        f"{label} already has an active training process using this path:\n"
        f"{path_hint}\n"
        f"Stop the existing run or wait for it to finish before launching another.\n"
        f"{details}"
    )


def _write_mask_aligned_to_image(src_mask: Path, reference_img: Path, dst_mask: Path) -> None:
    dst_mask.parent.mkdir(parents=True, exist_ok=True)
    if dst_mask.exists() or dst_mask.is_symlink():
        dst_mask.unlink()

    with Image.open(reference_img) as ref_im:
        ref_size = ref_im.size
    with Image.open(src_mask) as mask_im:
        mask_l = mask_im.convert("L")
        if mask_l.size == ref_size:
            try:
                dst_mask.symlink_to(src_mask)
                return
            except Exception:
                shutil.copy2(src_mask, dst_mask)
                return
        resized = mask_l.resize(ref_size, Image.Resampling.NEAREST)
        resized.save(dst_mask)


def _caption_text(template: str, token: str, descriptors: Dict[str, str], use_descriptors: bool) -> str:
    descriptor = str(descriptors.get(token, "")).strip()
    descriptor_suffix = f", {descriptor}" if (use_descriptors and descriptor) else ""
    return template.format(
        token=token,
        descriptor=descriptor,
        descriptor_suffix=descriptor_suffix,
    ).strip()


def _materialize_real_anchor_dataset(
    *,
    labels_csv: Path,
    label_col: str,
    mask_dir: Path,
    dataset_dir: Path,
    masks_dir: Path,
    caption_template: str,
    caption_descriptors: Dict[str, str],
    caption_use_descriptors: bool,
    max_images_per_class: int,
    random_state: int,
) -> Dict[str, int]:
    _clear_dir(dataset_dir)
    _clear_dir(masks_dir)
    df = pd.read_csv(labels_csv)
    if "image_path" not in df.columns:
        raise SystemExit(f"Phase 3 curriculum requires image_path in {labels_csv}")
    if label_col not in df.columns:
        raise SystemExit(f"Phase 3 curriculum requires '{label_col}' in {labels_csv}")
    df = _sample_balanced_df(
        df,
        label_col=label_col,
        max_per_class=max_images_per_class,
        random_state=random_state,
    )

    kept = 0
    missing_masks = 0
    missing_images = 0
    for idx, row in df.iterrows():
        src_img = Path(str(row["image_path"])).expanduser()
        if not src_img.is_absolute():
            src_img = (labels_csv.parent / src_img).resolve()
        if not src_img.is_file():
            missing_images += 1
            continue
        mask_name = str(row["filename"]).strip() if "filename" in row.index and str(row["filename"]).strip() else src_img.name
        src_mask = (mask_dir / mask_name).resolve()
        if not src_mask.is_file():
            missing_masks += 1
            continue

        token = _normalize_token(row[label_col])
        stem = f"real_{idx:06d}_{_safe_name(Path(mask_name).stem)}"
        dst_img = dataset_dir / f"{stem}{src_img.suffix.lower() or '.png'}"
        dst_mask = masks_dir / f"{stem}{src_mask.suffix.lower() or '.png'}"
        _symlink_or_copy(src_img.resolve(), dst_img)
        _write_mask_aligned_to_image(src_mask, src_img.resolve(), dst_mask)
        dst_img.with_suffix(".caption").write_text(
            _caption_text(caption_template, token, caption_descriptors, caption_use_descriptors),
            encoding="utf-8",
        )
        kept += 1

    return {
        "real_anchor_images": kept,
        "real_anchor_missing_images": missing_images,
        "real_anchor_missing_masks": missing_masks,
        "real_anchor_images_per_class_cap": max_images_per_class if max_images_per_class > 0 else "",
    }


def _append_benchmark_curriculum(
    *,
    benchmark_dir: Path,
    dataset_dir: Path,
    masks_dir: Path,
    cross_class_repeat: int,
    same_class_repeat: int,
    cross_class_min_margin: Optional[float],
    cross_class_min_target_prob: Optional[float],
    same_class_min_target_prob: Optional[float],
    cross_class_target_repeat_map: Dict[str, int],
) -> Dict[str, int]:
    rows_csv = benchmark_dir / "benchmark_rows.csv"
    if not rows_csv.is_file():
        raise SystemExit(f"Benchmark rows CSV not found for curriculum build: {rows_csv}")

    kept = 0
    cross_rows = 0
    same_rows = 0
    dropped_cross_margin = 0
    dropped_cross_prob = 0
    dropped_same_prob = 0
    with rows_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_mode = str(row.get("prompt_mode", "")).strip()
            target_label = str(row.get("target_label", "")).strip()
            repeat = int(cross_class_repeat) if prompt_mode == "cross_class" else int(same_class_repeat)
            if prompt_mode == "cross_class":
                repeat = max(repeat, int(cross_class_target_repeat_map.get(target_label, repeat)))
                margin = _as_optional_float(row.get("target_source_prob_margin"))
                target_prob = _as_optional_float(row.get("target_prob"))
                if cross_class_min_margin is not None and (margin is None or margin < cross_class_min_margin):
                    dropped_cross_margin += 1
                    continue
                if cross_class_min_target_prob is not None and (target_prob is None or target_prob < cross_class_min_target_prob):
                    dropped_cross_prob += 1
                    continue
            else:
                target_prob = _as_optional_float(row.get("target_prob"))
                if same_class_min_target_prob is not None and (target_prob is None or target_prob < same_class_min_target_prob):
                    dropped_same_prob += 1
                    continue
            if repeat <= 0:
                continue
            src_img = Path(str(row.get("generated_path", ""))).resolve()
            src_mask = Path(str(row.get("mask_path", ""))).resolve()
            if not src_img.is_file() or not src_mask.is_file():
                continue
            prompt = str(row.get("prompt", "")).strip()
            base = (
                f"{prompt_mode}_{_safe_name(str(row.get('source_key', 'src')))}"
                f"__to__{_safe_name(str(row.get('target_label', 'target')))}"
            )
            for rep_idx in range(repeat):
                stem = f"{base}__r{rep_idx + 1:02d}"
                dst_img = dataset_dir / f"{stem}{src_img.suffix.lower() or '.png'}"
                dst_mask = masks_dir / f"{stem}{src_mask.suffix.lower() or '.png'}"
                _symlink_or_copy(src_img, dst_img)
                _write_mask_aligned_to_image(src_mask, src_img, dst_mask)
                dst_img.with_suffix(".caption").write_text(prompt, encoding="utf-8")
                kept += 1
            if prompt_mode == "cross_class":
                cross_rows += 1
            else:
                same_rows += 1

    return {
        "pseudo_rows_kept": kept,
        "pseudo_cross_rows": cross_rows,
        "pseudo_same_rows": same_rows,
        "pseudo_dropped_cross_margin": dropped_cross_margin,
        "pseudo_dropped_cross_prob": dropped_cross_prob,
        "pseudo_dropped_same_prob": dropped_same_prob,
    }


def _link_anchor_dataset(anchor_dataset_dir: Path, anchor_masks_dir: Path, dataset_dir: Path, masks_dir: Path) -> int:
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif", ".jxl"}
    linked = 0
    for img in anchor_dataset_dir.iterdir():
        if img.suffix.lower() not in image_exts:
            continue
        cap = img.with_suffix(".caption")
        if not cap.is_file():
            continue
        dst_img = dataset_dir / img.name
        dst_cap = dataset_dir / cap.name
        mask_candidates = list(anchor_masks_dir.glob(f"{img.stem}.*"))
        if not mask_candidates:
            continue
        _symlink_or_copy(img.resolve(), dst_img)
        _symlink_or_copy(cap.resolve(), dst_cap)
        _write_mask_aligned_to_image(mask_candidates[0].resolve(), img.resolve(), masks_dir / mask_candidates[0].name)
        linked += 1
    return linked


def _dump_cycle_tables(rows: List[Dict[str, Any]], work_dir: Path) -> None:
    rows_json = work_dir / "phase3_history.json"
    rows_csv = work_dir / "phase3_history.csv"
    rows_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = [
        "cycle",
        "start_checkpoint",
        "train_dataset_path",
        "train_mask_dir",
        "cycle_best_checkpoint",
        "cycle_best_combined_reward",
        "cycle_best_cross_class_target_top1",
        "cycle_best_same_class_target_top1",
        "cycle_best_cross_class_margin_score",
        "cycle_best_cross_class_target_prob_margin_mean",
        "cycle_best_same_class_target_prob_mean",
        "global_best_checkpoint",
        "global_best_combined_reward",
        "improved_global_best",
        "reverted_to_global_best",
        "next_start_checkpoint",
        "linked_real_anchor_images",
        "real_anchor_images_per_class_cap",
        "pseudo_rows_kept",
        "pseudo_cross_rows",
        "pseudo_same_rows",
        "pseudo_dropped_cross_margin",
        "pseudo_dropped_cross_prob",
        "pseudo_dropped_same_prob",
        "next_curriculum_dataset_path",
        "next_curriculum_mask_dir",
        "holdout_combined_reward",
        "holdout_cross_class_margin_score",
        "holdout_cross_class_target_prob_margin_mean",
        "holdout_cross_class_target_top1",
        "holdout_same_class_target_prob_mean",
        "holdout_same_class_target_top1",
        "holdout_summary_json",
        "ranking_json",
        "ranking_csv",
    ]
    with rows_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 morph-reward-guided LoRA training loop")
    ap.add_argument("--config", required=True, help="YAML/JSON config for the phase-3 loop")
    ap.add_argument("--max-cycles", type=int, default=0, help="Optional override for cycles (>0)")
    ap.add_argument(
        "--restart-from-scratch",
        action="store_true",
        help="Delete the existing Phase 3 work_dir before starting the loop.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print planned commands without running")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"--config not found: {cfg_path}")
    cfg_dir = cfg_path.parent
    cfg = load_config(str(cfg_path)) or {}

    project_root = _resolve_path(cfg.get("project_path", str(REPO)), cfg_dir=cfg_dir, project_root=REPO)
    python_exec = str(cfg.get("python_exec", sys.executable))

    base_train_cfg_path = _resolve_path(cfg.get("base_train_config"), cfg_dir=cfg_dir, project_root=project_root)
    if not base_train_cfg_path.is_file():
        raise SystemExit(f"base_train_config not found: {base_train_cfg_path}")
    base_train_cfg = load_config(str(base_train_cfg_path)) or {}

    work_dir = _resolve_path(cfg.get("work_dir"), cfg_dir=cfg_dir, project_root=project_root)
    if args.restart_from_scratch:
        _raise_if_training_running(work_dir, "Phase 3 work_dir")
        if args.dry_run:
            print(f"[dry-run] would remove existing Phase 3 work_dir: {work_dir}")
        elif work_dir.exists():
            print(f"Restarting from scratch: removing existing Phase 3 work_dir {work_dir}")
            shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    initial_lora = _resolve_path(cfg.get("initial_lora_weights"), cfg_dir=cfg_dir, project_root=project_root)
    if not initial_lora.is_file():
        raise SystemExit(f"initial_lora_weights not found: {initial_lora}")

    cycles = _as_int(cfg.get("cycles", 4), 4)
    if args.max_cycles > 0:
        cycles = int(args.max_cycles)
    if cycles < 1:
        raise SystemExit("cycles must be >= 1")

    steps_per_cycle = _as_int(cfg.get("steps_per_cycle", 500), 500)
    if steps_per_cycle < 1:
        raise SystemExit("steps_per_cycle must be >= 1")
    save_every_n_steps = _as_int(cfg.get("save_every_n_steps", steps_per_cycle), steps_per_cycle)
    if save_every_n_steps < 1:
        raise SystemExit("save_every_n_steps must be >= 1")

    sample_every_override = cfg.get("sample_every_n_steps_override", None)
    lora_dim_from_weights = bool(cfg.get("lora_dim_from_weights", True))
    force_materialize = cfg.get("force_materialize_from_labels_csv", None)
    train_overrides = cfg.get("train_overrides", {}) or {}
    inpaint_overrides = cfg.get("inpaint_train_overrides", {}) or {}

    seed_mode = str(cfg.get("cycle_seed_mode", "incremental")).strip().lower()
    seed_base = _as_int(cfg.get("cycle_seed_base", base_train_cfg.get("seed", 222)), 222)
    seed_step = _as_int(cfg.get("cycle_seed_step", 1), 1)
    if seed_mode not in {"fixed", "incremental"}:
        raise SystemExit("cycle_seed_mode must be 'fixed' or 'incremental'")

    epsilon = _as_float(cfg.get("improvement_epsilon", 1e-4), 1e-4)
    patience = _as_int(cfg.get("patience", 0), 0)
    revert_on_drop = bool(cfg.get("revert_on_drop", True))

    finetune_script = _resolve_path(
        cfg.get("finetune_script", "scripts/synthetic_data/finetune_stable_diffusion_unified.py"),
        cfg_dir=cfg_dir,
        project_root=project_root,
    )
    selector_script = _resolve_path(
        cfg.get("selector_script", "scripts/synthetic_data/select_best_lora_inpaint_checkpoint.py"),
        cfg_dir=cfg_dir,
        project_root=project_root,
    )
    if not finetune_script.is_file():
        raise SystemExit(f"finetune_script not found: {finetune_script}")
    if not selector_script.is_file():
        raise SystemExit(f"selector_script not found: {selector_script}")

    selector_cfg = cfg.get("selector", {}) or {}
    base_model = _resolve_path(selector_cfg.get("base_model"), cfg_dir=cfg_dir, project_root=project_root)
    labels_csv = _resolve_path(selector_cfg.get("labels_csv"), cfg_dir=cfg_dir, project_root=project_root)
    mask_dir = _resolve_path(selector_cfg.get("mask_dir"), cfg_dir=cfg_dir, project_root=project_root)
    classifier_ckpt = _resolve_path(
        selector_cfg.get("classifier_ckpt"),
        cfg_dir=cfg_dir,
        project_root=project_root,
    )
    classifier_study_config = _resolve_path(
        selector_cfg.get("classifier_study_config"),
        cfg_dir=cfg_dir,
        project_root=project_root,
    )
    for label, path, must_be_dir in [
        ("selector.base_model", base_model, False),
        ("selector.labels_csv", labels_csv, False),
        ("selector.mask_dir", mask_dir, True),
        ("selector.classifier_ckpt", classifier_ckpt, False),
        ("selector.classifier_study_config", classifier_study_config, False),
    ]:
        ok = path.is_dir() if must_be_dir else path.is_file()
        if not ok:
            raise SystemExit(f"{label} not found: {path}")

    descriptors_path = _maybe_write_descriptors(
        selector_cfg=selector_cfg,
        work_dir=work_dir,
        cfg_dir=cfg_dir,
        project_root=project_root,
    )

    generation_device = str(selector_cfg.get("generation_device", "auto"))
    classifier_device = str(selector_cfg.get("classifier_device", "cpu"))
    selector_batch_size = _as_int(selector_cfg.get("batch_size", 64), 64)
    selector_images_per_class = _as_int(selector_cfg.get("images_per_class", 2), 2)
    selector_max_sources = _as_int(selector_cfg.get("max_sources", 0), 0)
    selector_steps = _as_int(selector_cfg.get("steps", 40), 40)
    selector_guidance = _as_float(selector_cfg.get("guidance_scale", 5.0), 5.0)
    selector_strength = _as_float(selector_cfg.get("strength", 0.55), 0.55)
    selector_same_class_strength = _as_float(selector_cfg.get("same_class_strength", selector_strength), selector_strength)
    selector_cross_class_strength = _as_float(
        selector_cfg.get("cross_class_strength", selector_strength),
        selector_strength,
    )
    selector_lora_scale = _as_float(selector_cfg.get("lora_scale", 0.75), 0.75)
    selector_negative = str(selector_cfg.get("negative_prompt", "lowres,bad anatomy"))
    selector_width = _as_int(selector_cfg.get("width", 1024), 1024)
    selector_height = _as_int(selector_cfg.get("height", 1024), 1024)
    selector_mask_feather = _as_float(selector_cfg.get("mask_feather_radius", 6.0), 6.0)
    selector_same_class_mask_dilate_px = _as_float(selector_cfg.get("same_class_mask_dilate_px", 0.0), 0.0)
    selector_cross_class_mask_dilate_px = _as_float(selector_cfg.get("cross_class_mask_dilate_px", 0.0), 0.0)
    selector_cross_class_target_strengths_arg = _maybe_write_json_arg(
        selector_cfg.get("cross_class_target_strengths_json", None),
        work_dir / "selector_cross_class_target_strengths.json",
    )
    selector_cross_class_target_mask_dilate_arg = _maybe_write_json_arg(
        selector_cfg.get("cross_class_target_mask_dilate_json", None),
        work_dir / "selector_cross_class_target_mask_dilate.json",
    )
    selector_seed_base = _as_int(selector_cfg.get("seed_base", 222), 222)
    selector_prompt_template = str(
        selector_cfg.get(
            "prompt_template",
            "clinical intraoral photo, diagnosis: {token}{descriptor_suffix}",
        )
    )
    selector_tokens_raw = selector_cfg.get("tokens", ["healthy", "benign_lesion", "opmd", "cancer"])
    if isinstance(selector_tokens_raw, str):
        selector_tokens = [x.strip() for x in selector_tokens_raw.split(",") if x.strip()]
    else:
        selector_tokens = [str(x).strip() for x in selector_tokens_raw if str(x).strip()]
    if not selector_tokens:
        raise SystemExit("selector.tokens resolved to empty list.")

    selector_descriptors = selector_cfg.get("descriptors_json", None)
    selector_cross_weight = _as_float(selector_cfg.get("cross_class_weight", 0.75), 0.75)
    selector_same_weight = _as_float(selector_cfg.get("same_class_weight", 0.25), 0.25)
    exclude_last = bool(selector_cfg.get("exclude_last", False))
    max_checkpoints = _as_int(selector_cfg.get("max_checkpoints", 0), 0)
    selector_include_last_checkpoint = bool(cfg.get("selector_include_last_checkpoint", False))
    label_col = str(selector_cfg.get("label_col", "coarse_label"))
    reuse_selected_sources = bool(selector_cfg.get("reuse_selected_sources_across_cycles", True))
    reuse_existing_cycle_outputs = bool(cfg.get("reuse_existing_cycle_outputs", True)) and not args.restart_from_scratch
    curriculum_cfg = cfg.get("curriculum", {}) or {}
    curriculum_enabled = bool(curriculum_cfg.get("enabled", True))
    curriculum_include_real_anchor = bool(curriculum_cfg.get("include_real_anchor", True))
    curriculum_real_anchor_images_per_class = _as_int(curriculum_cfg.get("real_anchor_images_per_class", 0), 0)
    curriculum_cross_class_repeat = _as_int(curriculum_cfg.get("cross_class_repeat", 2), 2)
    curriculum_same_class_repeat = _as_int(curriculum_cfg.get("same_class_repeat", 1), 1)
    curriculum_cross_class_min_margin = _as_optional_float(curriculum_cfg.get("cross_class_min_margin", None))
    curriculum_cross_class_min_target_prob = _as_optional_float(curriculum_cfg.get("cross_class_min_target_prob", None))
    curriculum_same_class_min_target_prob = _as_optional_float(curriculum_cfg.get("same_class_min_target_prob", None))
    curriculum_cross_class_target_repeat_map = _load_repeat_map(curriculum_cfg.get("cross_class_target_repeat_map", {}))
    holdout_cfg = cfg.get("holdout_eval", {}) or {}
    holdout_enabled = bool(holdout_cfg.get("enabled", False))
    holdout_images_per_class = _as_int(holdout_cfg.get("images_per_class", 0), 0)
    holdout_max_sources = _as_int(holdout_cfg.get("max_sources", 0), 0)
    holdout_seed_base = _as_int(holdout_cfg.get("seed_base", selector_seed_base + 1000), selector_seed_base + 1000)
    holdout_out_dir_name = str(holdout_cfg.get("out_dir_name", "holdout_eval")).strip() or "holdout_eval"
    holdout_cross_weight = _as_float(holdout_cfg.get("cross_class_weight", selector_cross_weight), selector_cross_weight)
    holdout_same_weight = _as_float(holdout_cfg.get("same_class_weight", selector_same_weight), selector_same_weight)
    holdout_selector_like = bool(holdout_cfg.get("inherit_selector_settings", True))
    holdout_width = _as_int(holdout_cfg.get("width", selector_width if holdout_selector_like else 512), selector_width)
    holdout_height = _as_int(holdout_cfg.get("height", selector_height if holdout_selector_like else 512), selector_height)
    holdout_steps = _as_int(holdout_cfg.get("steps", selector_steps), selector_steps)
    holdout_guidance = _as_float(holdout_cfg.get("guidance_scale", selector_guidance), selector_guidance)
    holdout_same_strength = _as_float(holdout_cfg.get("same_class_strength", selector_same_class_strength), selector_same_class_strength)
    holdout_cross_strength = _as_float(holdout_cfg.get("cross_class_strength", selector_cross_class_strength), selector_cross_class_strength)
    holdout_same_mask_dilate = _as_float(holdout_cfg.get("same_class_mask_dilate_px", selector_same_class_mask_dilate_px), selector_same_class_mask_dilate_px)
    holdout_cross_mask_dilate = _as_float(holdout_cfg.get("cross_class_mask_dilate_px", selector_cross_class_mask_dilate_px), selector_cross_class_mask_dilate_px)
    holdout_cross_class_target_strengths_arg = _maybe_write_json_arg(
        holdout_cfg.get("cross_class_target_strengths_json", selector_cfg.get("cross_class_target_strengths_json", None)),
        work_dir / "holdout_cross_class_target_strengths.json",
    )
    holdout_cross_class_target_mask_dilate_arg = _maybe_write_json_arg(
        holdout_cfg.get("cross_class_target_mask_dilate_json", selector_cfg.get("cross_class_target_mask_dilate_json", None)),
        work_dir / "holdout_cross_class_target_mask_dilate.json",
    )
    caption_template = str(
        train_overrides.get(
            "caption_template",
            base_train_cfg.get("caption_template", "clinical photo of an oral lesion, diagnosis: {token}{descriptor_suffix}"),
        )
    )
    caption_use_descriptors = bool(
        train_overrides.get(
            "caption_use_class_descriptors",
            base_train_cfg.get("caption_use_class_descriptors", False),
        )
    )
    caption_descriptors = _load_descriptors_map(selector_cfg.get("descriptors_json", None), descriptors_path)

    current_start_ckpt = initial_lora
    global_best_ckpt: Optional[Path] = None
    global_best_benchmark_dir: Optional[Path] = None
    global_best_score = -1.0
    no_improve = 0
    history: List[Dict[str, Any]] = []
    global_selected_sources = work_dir / "selected_sources_phase3.csv"
    global_holdout_selected_sources = work_dir / "selected_sources_phase3_holdout.csv"
    current_curriculum_dataset: Optional[Path] = None
    current_curriculum_masks: Optional[Path] = None
    real_anchor_dataset = work_dir / "curriculum" / "real_anchor" / "dataset"
    real_anchor_masks = work_dir / "curriculum" / "real_anchor" / "masks"
    real_anchor_stats: Dict[str, int] = {}
    if curriculum_enabled and curriculum_include_real_anchor:
        real_anchor_stats = _materialize_real_anchor_dataset(
            labels_csv=labels_csv,
            label_col=label_col,
            mask_dir=mask_dir,
            dataset_dir=real_anchor_dataset,
            masks_dir=real_anchor_masks,
            caption_template=caption_template,
            caption_descriptors=caption_descriptors,
            caption_use_descriptors=caption_use_descriptors,
            max_images_per_class=curriculum_real_anchor_images_per_class,
            random_state=selector_seed_base,
        )

    print("Phase-3 morph reward loop")
    print("  phase3_mode        : inpaint")
    print(f"  work_dir           : {work_dir}")
    print(f"  base_train_config  : {base_train_cfg_path}")
    print(f"  initial_lora       : {initial_lora}")
    print(f"  labels_csv         : {labels_csv}")
    print(f"  mask_dir           : {mask_dir}")
    print(f"  cycles             : {cycles}")
    print(f"  steps_per_cycle    : {steps_per_cycle}")
    print(f"  score_weights      : cross={selector_cross_weight:.3f}, same={selector_same_weight:.3f}")
    print(f"  curriculum         : enabled={curriculum_enabled} include_real_anchor={curriculum_include_real_anchor}")
    print(
        f"  curriculum_filter  : cross_margin>={curriculum_cross_class_min_margin} "
        f"cross_target_prob>={curriculum_cross_class_min_target_prob} "
        f"same_target_prob>={curriculum_same_class_min_target_prob}"
    )
    print(f"  holdout_eval       : {holdout_enabled}")
    print(f"  restart_scratch    : {bool(args.restart_from_scratch)}")

    for cycle_idx in range(1, cycles + 1):
        cycle_name = f"cycle_{cycle_idx:03d}"
        cycle_dir = work_dir / cycle_name
        cycle_out_dir = cycle_dir / "train_output"
        cycle_eval_dir = cycle_dir / "checkpoint_selection_inpaint"
        cycle_cfg_path = cycle_dir / "train_config.generated.yaml"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        cycle_out_dir.mkdir(parents=True, exist_ok=True)

        cycle_cfg = copy.deepcopy(base_train_cfg)
        cycle_cfg["training_mode"] = "lora"
        cycle_cfg["output_dir"] = str(cycle_out_dir)
        cycle_cfg["max_train_steps"] = int(steps_per_cycle)
        cycle_cfg["save_every_n_steps"] = int(save_every_n_steps)
        cycle_cfg["lora_network_weights"] = str(current_start_ckpt)
        cycle_cfg["lora_dim_from_weights"] = bool(lora_dim_from_weights)
        cycle_cfg["lora_use_masked_loss"] = True
        if force_materialize in (True, False):
            cycle_cfg["materialize_from_labels_csv"] = bool(force_materialize)
        elif bool(cycle_cfg.get("materialize_from_labels_csv", False)):
            cycle_cfg["materialize_from_labels_csv"] = False
        if sample_every_override not in (None, "", "null"):
            cycle_cfg["sample_every_n_steps"] = int(sample_every_override)
        if seed_mode == "fixed":
            cycle_cfg["seed"] = int(seed_base)
        else:
            cycle_cfg["seed"] = int(seed_base + (cycle_idx - 1) * seed_step)
        for k, v in train_overrides.items():
            cycle_cfg[k] = v
        for k, v in inpaint_overrides.items():
            cycle_cfg[k] = v

        active_train_dataset = ""
        active_train_masks = ""
        if curriculum_enabled and current_curriculum_dataset is not None and current_curriculum_masks is not None:
            cycle_cfg["dataset_path"] = str(current_curriculum_dataset)
            cycle_cfg["use_existing_dataset_captions"] = True
            cycle_cfg["materialize_from_labels_csv"] = False
            cycle_cfg["lora_mask_dir"] = str(current_curriculum_masks)
            active_train_dataset = str(current_curriculum_dataset)
            active_train_masks = str(current_curriculum_masks)

        mask_dir_raw = cycle_cfg.get("lora_mask_dir")
        lora_mask_mode = str(cycle_cfg.get("lora_mask_mode", "directory")).strip().lower()
        if lora_mask_mode == "directory" and mask_dir_raw in (None, "", "null"):
            raise SystemExit(
                "Phase 3 with lora_mask_mode='directory' requires lora_mask_dir. "
                "Set it in config.inpaint_train_overrides or notebook overrides."
            )

        _write_yaml(cycle_cfg_path, cycle_cfg)

        print(f"\n[cycle {cycle_idx}/{cycles}] start={current_start_ckpt.name}")
        existing_ckpts = _stable_sort_ckpts(list(cycle_out_dir.glob("*.safetensors")))
        if not args.dry_run and reuse_existing_cycle_outputs and existing_ckpts:
            print(f"[cycle {cycle_idx}] reusing existing train_output checkpoints in {cycle_out_dir}")
        else:
            _raise_if_training_running(cycle_out_dir, f"Phase 3 {cycle_name}")
            _run(
                [python_exec, str(finetune_script), "--config", str(cycle_cfg_path)],
                cwd=project_root,
                dry_run=args.dry_run,
            )

        if args.dry_run:
            dry_names_raw = cfg.get("dry_run_checkpoint_names", "last.safetensors")
            if isinstance(dry_names_raw, str):
                dry_names = [x.strip() for x in dry_names_raw.split(",") if x.strip()]
            else:
                dry_names = [str(x).strip() for x in dry_names_raw if str(x).strip()]
            dry_ckpts = [cycle_out_dir / name for name in (dry_names if dry_names else ["last.safetensors"])]
            checkpoint_names = _selector_checkpoint_names(
                dry_ckpts,
                include_last=selector_include_last_checkpoint,
            )
        else:
            produced_ckpts = _stable_sort_ckpts(list(cycle_out_dir.glob("*.safetensors")))
            if not produced_ckpts:
                raise SystemExit(f"No checkpoints produced in {cycle_out_dir}")
            checkpoint_names = _selector_checkpoint_names(
                produced_ckpts,
                include_last=selector_include_last_checkpoint,
            )

        selector_cmd = [
            python_exec,
            str(selector_script),
            "--run-dir",
            str(cycle_out_dir),
            "--base-model",
            str(base_model),
            "--labels-csv",
            str(labels_csv),
            "--label-col",
            label_col,
            "--mask-dir",
            str(mask_dir),
            "--classifier-ckpt",
            str(classifier_ckpt),
            "--classifier-study-config",
            str(classifier_study_config),
            "--device",
            generation_device,
            "--classifier-device",
            classifier_device,
            "--batch-size",
            str(selector_batch_size),
            "--images-per-class",
            str(selector_images_per_class),
            "--max-sources",
            str(selector_max_sources),
            "--steps",
            str(selector_steps),
            "--guidance-scale",
            str(selector_guidance),
            "--strength",
            str(selector_strength),
            "--same-class-strength",
            str(selector_same_class_strength),
            "--cross-class-strength",
            str(selector_cross_class_strength),
            "--lora-scale",
            str(selector_lora_scale),
            "--negative-prompt",
            selector_negative,
            "--width",
            str(selector_width),
            "--height",
            str(selector_height),
            "--mask-feather-radius",
            str(selector_mask_feather),
            "--same-class-mask-dilate-px",
            str(selector_same_class_mask_dilate_px),
            "--cross-class-mask-dilate-px",
            str(selector_cross_class_mask_dilate_px),
            "--seed-base",
            str(selector_seed_base),
            "--cross-class-weight",
            str(selector_cross_weight),
            "--same-class-weight",
            str(selector_same_weight),
            "--prompt-template",
            selector_prompt_template,
            "--tokens",
            ",".join(selector_tokens),
            "--checkpoint-names",
            checkpoint_names,
            "--out-dir",
            str(cycle_eval_dir),
        ]
        if descriptors_path is not None:
            selector_cmd += ["--descriptors-json", str(descriptors_path)]
        if selector_cross_class_target_strengths_arg is not None:
            selector_cmd += ["--cross-class-target-strengths-json", selector_cross_class_target_strengths_arg]
        if selector_cross_class_target_mask_dilate_arg is not None:
            selector_cmd += ["--cross-class-target-mask-dilate-json", selector_cross_class_target_mask_dilate_arg]
        if exclude_last:
            selector_cmd.append("--exclude-last")
        if max_checkpoints > 0:
            selector_cmd += ["--max-checkpoints", str(max_checkpoints)]
        if reuse_selected_sources and global_selected_sources.is_file():
            selector_cmd += ["--selected-sources-csv", str(global_selected_sources)]

        _run(selector_cmd, cwd=project_root, dry_run=args.dry_run)

        if args.dry_run:
            print("dry-run mode: command planning finished.")
            break

        ranking_json = cycle_eval_dir / "checkpoint_ranking.json"
        ranking_csv = cycle_eval_dir / "checkpoint_ranking.csv"
        if not ranking_json.is_file():
            raise SystemExit(f"Expected selector ranking file not found: {ranking_json}")
        ranking = json.loads(ranking_json.read_text(encoding="utf-8"))
        if reuse_selected_sources:
            selected_sources_raw = ranking.get("shared_selected_sources_csv")
            if selected_sources_raw:
                selected_sources_path = Path(selected_sources_raw)
                if selected_sources_path.is_file() and selected_sources_path.resolve() != global_selected_sources.resolve():
                    shutil.copy2(selected_sources_path, global_selected_sources)

        best = ranking.get("best", {}) or {}
        cycle_best_name = str(best.get("checkpoint", "")).strip()
        if not cycle_best_name:
            raise SystemExit(f"Selector ranking has no best checkpoint: {ranking_json}")
        cycle_best_ckpt = cycle_out_dir / cycle_best_name
        if not cycle_best_ckpt.is_file():
            raise SystemExit(f"Best checkpoint listed by selector is missing: {cycle_best_ckpt}")
        cycle_best_score = _as_float(best.get("combined_reward", 0.0), 0.0)
        cycle_best_cross = _as_float(best.get("cross_class_target_top1", 0.0), 0.0)
        cycle_best_same = _as_float(best.get("same_class_target_top1", 0.0), 0.0)
        cycle_best_cross_margin = _as_float(best.get("cross_class_margin_score", 0.0), 0.0)
        cycle_best_cross_margin_mean = _as_float(best.get("cross_class_target_source_prob_margin_mean", 0.0), 0.0)
        cycle_best_same_prob = _as_float(best.get("same_class_target_prob_mean", 0.0), 0.0)
        cycle_best_benchmark_dir = Path(str(best.get("benchmark_dir", cycle_eval_dir / cycle_best_ckpt.stem))).resolve()
        holdout_metrics: Dict[str, Any] = {}

        if holdout_enabled and holdout_images_per_class > 0:
            holdout_dir = cycle_dir / holdout_out_dir_name / cycle_best_ckpt.stem
            benchmark_script = _resolve_path(
                cfg.get("benchmark_script", "scripts/synthetic_data/benchmark_lora_inpaint_with_classifier.py"),
                cfg_dir=cfg_dir,
                project_root=project_root,
            )
            holdout_cmd = [
                python_exec,
                str(benchmark_script),
                "--labels-csv",
                str(labels_csv),
                "--label-col",
                label_col,
                "--mask-dir",
                str(mask_dir),
                "--base-model",
                str(base_model),
                "--model",
                f"holdout={cycle_best_ckpt}",
                "--classifier-ckpt",
                str(classifier_ckpt),
                "--classifier-study-config",
                str(classifier_study_config),
                "--out-dir",
                str(holdout_dir),
                "--device",
                generation_device,
                "--classifier-device",
                classifier_device,
                "--tokens",
                ",".join(selector_tokens),
                "--prompt-template",
                selector_prompt_template,
                "--negative-prompt",
                selector_negative,
                "--width",
                str(holdout_width),
                "--height",
                str(holdout_height),
                "--steps",
                str(holdout_steps),
                "--guidance-scale",
                str(holdout_guidance),
                "--same-class-strength",
                str(holdout_same_strength),
                "--cross-class-strength",
                str(holdout_cross_strength),
                "--lora-scale",
                str(selector_lora_scale),
                "--mask-feather-radius",
                str(selector_mask_feather),
                "--same-class-mask-dilate-px",
                str(holdout_same_mask_dilate),
                "--cross-class-mask-dilate-px",
                str(holdout_cross_mask_dilate),
                "--sources-per-class",
                str(holdout_images_per_class),
                "--max-sources",
                str(holdout_max_sources),
                "--seed-base",
                str(holdout_seed_base),
            ]
            if descriptors_path is not None:
                holdout_cmd += ["--descriptors-json", str(descriptors_path)]
            if holdout_cross_class_target_strengths_arg is not None:
                holdout_cmd += ["--cross-class-target-strengths-json", holdout_cross_class_target_strengths_arg]
            if holdout_cross_class_target_mask_dilate_arg is not None:
                holdout_cmd += ["--cross-class-target-mask-dilate-json", holdout_cross_class_target_mask_dilate_arg]
            if global_holdout_selected_sources.is_file():
                holdout_cmd += ["--selected-sources-csv", str(global_holdout_selected_sources)]
            _run(holdout_cmd, cwd=project_root, dry_run=args.dry_run)
            holdout_selected = holdout_dir / "selected_sources.csv"
            if not global_holdout_selected_sources.is_file() and holdout_selected.is_file():
                shutil.copy2(holdout_selected, global_holdout_selected_sources)
            holdout_summary_path = holdout_dir / "benchmark_summary.json"
            if holdout_summary_path.is_file():
                holdout_summary = json.loads(holdout_summary_path.read_text(encoding="utf-8"))
                holdout_model = (holdout_summary.get("metrics_by_model", {}) or {}).get("holdout", {}) or {}
                holdout_same = holdout_model.get("same_class_target_top1", {}) or {}
                holdout_cross = holdout_model.get("cross_class_target_top1", {}) or {}
                holdout_same_prob = _as_float(holdout_same.get("target_prob_mean", 0.0), 0.0)
                holdout_cross_margin_score = _as_float(holdout_cross.get("target_source_prob_margin_score", 0.0), 0.0)
                holdout_metrics = {
                    "holdout_summary_json": str(holdout_summary_path),
                    "holdout_same_class_target_top1": _as_float(holdout_same.get("top1", 0.0), 0.0),
                    "holdout_cross_class_target_top1": _as_float(holdout_cross.get("top1", 0.0), 0.0),
                    "holdout_same_class_target_prob_mean": holdout_same_prob,
                    "holdout_cross_class_target_prob_margin_mean": _as_float(
                        holdout_cross.get("target_source_prob_margin_mean", 0.0), 0.0
                    ),
                    "holdout_cross_class_margin_score": holdout_cross_margin_score,
                    "holdout_combined_reward": (
                        holdout_same_weight * holdout_same_prob + holdout_cross_weight * holdout_cross_margin_score
                    ),
                }

        improved = cycle_best_score > (global_best_score + epsilon)
        if improved:
            global_best_score = cycle_best_score
            global_best_ckpt = cycle_best_ckpt
            global_best_benchmark_dir = cycle_best_benchmark_dir
            no_improve = 0
        else:
            no_improve += 1

        reverted = False
        if (
            revert_on_drop
            and global_best_ckpt is not None
            and (cycle_best_score + epsilon) < global_best_score
        ):
            next_start_ckpt = global_best_ckpt
            reverted = True
        else:
            next_start_ckpt = cycle_best_ckpt

        if curriculum_enabled:
            benchmark_dir_for_curriculum = global_best_benchmark_dir if reverted and global_best_benchmark_dir is not None else cycle_best_benchmark_dir
            next_curriculum_dir = work_dir / "curriculum" / f"cycle_{cycle_idx:03d}"
            next_dataset_dir = next_curriculum_dir / "dataset"
            next_masks_dir = next_curriculum_dir / "masks"
            _clear_dir(next_dataset_dir)
            _clear_dir(next_masks_dir)
            linked_real = 0
            if curriculum_include_real_anchor:
                linked_real = _link_anchor_dataset(real_anchor_dataset, real_anchor_masks, next_dataset_dir, next_masks_dir)
            pseudo_stats = _append_benchmark_curriculum(
                benchmark_dir=benchmark_dir_for_curriculum,
                dataset_dir=next_dataset_dir,
                masks_dir=next_masks_dir,
                cross_class_repeat=curriculum_cross_class_repeat,
                same_class_repeat=curriculum_same_class_repeat,
                cross_class_min_margin=curriculum_cross_class_min_margin,
                cross_class_min_target_prob=curriculum_cross_class_min_target_prob,
                same_class_min_target_prob=curriculum_same_class_min_target_prob,
                cross_class_target_repeat_map=curriculum_cross_class_target_repeat_map,
            )
            current_curriculum_dataset = next_dataset_dir
            current_curriculum_masks = next_masks_dir
        else:
            linked_real = 0
            pseudo_stats = {}

        row = {
            "cycle": cycle_idx,
            "start_checkpoint": str(current_start_ckpt),
            "train_dataset_path": active_train_dataset,
            "train_mask_dir": active_train_masks,
            "cycle_best_checkpoint": str(cycle_best_ckpt),
            "cycle_best_combined_reward": cycle_best_score,
            "cycle_best_cross_class_target_top1": cycle_best_cross,
            "cycle_best_same_class_target_top1": cycle_best_same,
            "cycle_best_cross_class_margin_score": cycle_best_cross_margin,
            "cycle_best_cross_class_target_prob_margin_mean": cycle_best_cross_margin_mean,
            "cycle_best_same_class_target_prob_mean": cycle_best_same_prob,
            "global_best_checkpoint": str(global_best_ckpt) if global_best_ckpt is not None else "",
            "global_best_combined_reward": global_best_score,
            "improved_global_best": improved,
            "reverted_to_global_best": reverted,
            "next_start_checkpoint": str(next_start_ckpt),
            "ranking_json": str(ranking_json),
            "ranking_csv": str(ranking_csv),
        }
        row.update(real_anchor_stats)
        row.update(pseudo_stats)
        row.update(holdout_metrics)
        if current_curriculum_dataset is not None:
            row["next_curriculum_dataset_path"] = str(current_curriculum_dataset)
        if current_curriculum_masks is not None:
            row["next_curriculum_mask_dir"] = str(current_curriculum_masks)
        row["linked_real_anchor_images"] = linked_real
        history.append(row)
        _dump_cycle_tables(history, work_dir)

        print(
            f"[cycle {cycle_idx}] best={cycle_best_ckpt.name} "
            f"combined={cycle_best_score:.4f} cross_margin={cycle_best_cross_margin_mean:.4f} "
            f"cross_top1={cycle_best_cross:.4f} same_prob={cycle_best_same_prob:.4f} "
            f"same_top1={cycle_best_same:.4f} | global={global_best_score:.4f}"
        )

        current_start_ckpt = next_start_ckpt
        if patience > 0 and no_improve >= patience:
            print(f"Stopping early: no global-best improvement for {no_improve} cycle(s) (patience={patience}).")
            break

    if not args.dry_run and global_best_ckpt is not None:
        best_alias = work_dir / "best.safetensors"
        if best_alias.exists() or best_alias.is_symlink():
            best_alias.unlink()
        try:
            best_alias.symlink_to(global_best_ckpt)
        except Exception:
            shutil.copy2(global_best_ckpt, best_alias)

        summary = {
            "phase3_mode": "inpaint",
            "best_checkpoint": str(global_best_ckpt),
            "best_combined_reward": global_best_score,
            "score_weights": {
                "cross_class_weight": selector_cross_weight,
                "same_class_weight": selector_same_weight,
            },
            "curriculum": {
                "enabled": curriculum_enabled,
                "include_real_anchor": curriculum_include_real_anchor,
                "real_anchor_images_per_class": curriculum_real_anchor_images_per_class,
                "cross_class_repeat": curriculum_cross_class_repeat,
                "same_class_repeat": curriculum_same_class_repeat,
                "cross_class_min_margin": curriculum_cross_class_min_margin,
                "cross_class_min_target_prob": curriculum_cross_class_min_target_prob,
                "same_class_min_target_prob": curriculum_same_class_min_target_prob,
                "cross_class_target_repeat_map": curriculum_cross_class_target_repeat_map,
            },
            "holdout_eval": {
                "enabled": holdout_enabled,
                "images_per_class": holdout_images_per_class,
                "max_sources": holdout_max_sources,
                "seed_base": holdout_seed_base,
                "selected_sources_csv": str(global_holdout_selected_sources) if global_holdout_selected_sources.is_file() else None,
            },
            "selected_sources_csv": str(global_selected_sources) if global_selected_sources.is_file() else None,
            "history_json": str(work_dir / "phase3_history.json"),
            "history_csv": str(work_dir / "phase3_history.csv"),
        }
        (work_dir / "phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\nPhase-3 completed.")
        print(f"Best checkpoint: {global_best_ckpt}")
        print(f"Best combined reward: {global_best_score:.4f}")
        print(f"History CSV: {work_dir / 'phase3_history.csv'}")


if __name__ == "__main__":
    main()
