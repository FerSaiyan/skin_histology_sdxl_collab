#!/usr/bin/env python3
"""
Phase 2 lightweight classifier-guided LoRA loop.

Instead of adversarial per-step backprop, this script does:
1) train LoRA for a short chunk (N steps),
2) score resulting checkpoint(s) with a classifier,
3) pick the best checkpoint by reward metric,
4) continue from that checkpoint for the next chunk.
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


def _normalize_phase2_mode(raw: Any) -> str:
    mode = str(raw or "text2image").strip().lower().replace("-", "").replace("_", "")
    if mode in {"text2image", "txt2img"}:
        return "text2image"
    if mode in {"inpaint", "inpainting"}:
        return "inpaint"
    raise SystemExit("phase2_mode must be one of: text2image | inpaint")


def _stable_sort_ckpts(paths: List[Path]) -> List[Path]:
    def _key(p: Path) -> Tuple[int, float, str]:
        return (0, p.stat().st_mtime, p.name)

    return sorted(paths, key=_key)


def _selector_checkpoint_names(produced_ckpts: List[Path], *, include_last: bool) -> str:
    ordered = _stable_sort_ckpts(produced_ckpts)
    if len(ordered) > 1 and not include_last:
        ordered = [p for p in ordered if p.name != "last.safetensors"] or ordered
    return ",".join(p.name for p in ordered)


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


def _dump_cycle_tables(rows: List[Dict[str, Any]], work_dir: Path) -> None:
    rows_json = work_dir / "phase2_history.json"
    rows_csv = work_dir / "phase2_history.csv"
    rows_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = [
        "cycle",
        "phase2_mode",
        "selector_metric_name",
        "selector_secondary_metric_name",
        "start_checkpoint",
        "cycle_best_checkpoint",
        "cycle_best_primary_score",
        "cycle_best_secondary_score",
        "cycle_best_macro_consistency",
        "cycle_best_consistency_top1",
        "cycle_best_cross_class_target_top1",
        "cycle_best_same_class_target_top1",
        "cycle_best_cross_class_margin_score",
        "cycle_best_cross_class_target_prob_margin_mean",
        "cycle_best_same_class_target_prob_mean",
        "global_best_checkpoint",
        "global_best_primary_score",
        "global_best_macro_consistency",
        "improved_global_best",
        "reverted_to_global_best",
        "next_start_checkpoint",
        "ranking_json",
        "ranking_csv",
    ]
    with rows_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 reward-style LoRA training loop")
    ap.add_argument("--config", required=True, help="YAML/JSON config for the phase-2 loop")
    ap.add_argument("--max-cycles", type=int, default=0, help="Optional override for cycles (>0)")
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
    phase2_mode = _normalize_phase2_mode(cfg.get("phase2_mode", "text2image"))
    train_overrides = cfg.get("train_overrides", {}) or {}
    mode_overrides_key = "inpaint_train_overrides" if phase2_mode == "inpaint" else "text2image_train_overrides"
    mode_train_overrides = cfg.get(mode_overrides_key, {}) or {}
    force_materialize = cfg.get("force_materialize_from_labels_csv", None)

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
    selector_default = (
        "scripts/synthetic_data/select_best_lora_inpaint_checkpoint.py"
        if phase2_mode == "inpaint"
        else "scripts/synthetic_data/select_best_lora_checkpoint.py"
    )
    selector_script = _resolve_path(
        cfg.get("selector_script", selector_default),
        cfg_dir=cfg_dir,
        project_root=project_root,
    )
    if not finetune_script.is_file():
        raise SystemExit(f"finetune_script not found: {finetune_script}")
    if not selector_script.is_file():
        raise SystemExit(f"selector_script not found: {selector_script}")

    selector_cfg = cfg.get("selector", {}) or {}
    base_model = _resolve_path(selector_cfg.get("base_model"), cfg_dir=cfg_dir, project_root=project_root)
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
    selector_label_col = str(selector_cfg.get("label_col", train_overrides.get("label_column", "coarse_label")))
    if not base_model.is_file():
        raise SystemExit(f"selector.base_model not found: {base_model}")
    if not classifier_ckpt.is_file():
        raise SystemExit(f"selector.classifier_ckpt not found: {classifier_ckpt}")
    if not classifier_study_config.is_file():
        raise SystemExit(f"selector.classifier_study_config not found: {classifier_study_config}")

    descriptors_path = _maybe_write_descriptors(
        selector_cfg=selector_cfg,
        work_dir=work_dir,
        cfg_dir=cfg_dir,
        project_root=project_root,
    )

    generation_device = str(selector_cfg.get("generation_device", "auto"))
    classifier_device = str(selector_cfg.get("classifier_device", "auto"))
    selector_batch_size = _as_int(selector_cfg.get("batch_size", 64), 64)
    selector_images_per_class = _as_int(selector_cfg.get("images_per_class", 3), 3)
    selector_steps = _as_int(selector_cfg.get("steps", 35), 35)
    selector_guidance = _as_float(selector_cfg.get("guidance_scale", 5.0), 5.0)
    selector_negative = str(selector_cfg.get("negative_prompt", "lowres,bad anatomy"))
    selector_width = _as_int(selector_cfg.get("width", 512), 512)
    selector_height = _as_int(selector_cfg.get("height", 512), 512)
    selector_seed_base = _as_int(selector_cfg.get("seed_base", 222), 222)
    selector_max_sources = _as_int(selector_cfg.get("max_sources", 0), 0)
    selector_strength = _as_float(selector_cfg.get("strength", 0.55), 0.55)
    selector_lora_scale = _as_float(selector_cfg.get("lora_scale", 0.75), 0.75)
    selector_mask_feather_radius = _as_float(selector_cfg.get("mask_feather_radius", 6.0), 6.0)
    selector_same_class_strength = _as_float(selector_cfg.get("same_class_strength", selector_strength), selector_strength)
    selector_cross_class_strength = _as_float(
        selector_cfg.get("cross_class_strength", selector_strength),
        selector_strength,
    )
    selector_same_class_mask_dilate_px = _as_float(selector_cfg.get("same_class_mask_dilate_px", 0.0), 0.0)
    selector_cross_class_mask_dilate_px = _as_float(selector_cfg.get("cross_class_mask_dilate_px", 0.0), 0.0)
    selector_cross_class_weight = _as_float(selector_cfg.get("cross_class_weight", 0.75), 0.75)
    selector_same_class_weight = _as_float(selector_cfg.get("same_class_weight", 0.25), 0.25)
    selector_reuse_selected_sources = bool(selector_cfg.get("reuse_selected_sources_across_cycles", True))
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

    is_sdxl = bool(selector_cfg.get("is_sdxl", True))
    exclude_last = bool(selector_cfg.get("exclude_last", False))
    max_checkpoints = _as_int(selector_cfg.get("max_checkpoints", 0), 0)
    selector_include_last_checkpoint = bool(cfg.get("selector_include_last_checkpoint", False))
    shared_selected_sources = work_dir / "selected_sources_phase2.csv"

    current_start_ckpt = initial_lora
    global_best_ckpt: Optional[Path] = None
    global_best_score = -1.0
    no_improve = 0
    history: List[Dict[str, Any]] = []

    print("Phase-2 reward loop")
    print(f"  phase2_mode        : {phase2_mode}")
    print(f"  work_dir           : {work_dir}")
    print(f"  base_train_config  : {base_train_cfg_path}")
    print(f"  initial_lora       : {initial_lora}")
    print(f"  cycles             : {cycles}")
    print(f"  steps_per_cycle    : {steps_per_cycle}")

    for cycle_idx in range(1, cycles + 1):
        cycle_name = f"cycle_{cycle_idx:03d}"
        cycle_dir = work_dir / cycle_name
        cycle_out_dir = cycle_dir / "train_output"
        cycle_eval_dir = cycle_dir / "checkpoint_selection"
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
        cycle_cfg["lora_use_masked_loss"] = bool(phase2_mode == "inpaint")
        if phase2_mode == "text2image":
            cycle_cfg["sample_init_image"] = None
            cycle_cfg["sample_mask_image"] = None
            cycle_cfg["sample_denoising_strength"] = None
        if force_materialize in (True, False):
            cycle_cfg["materialize_from_labels_csv"] = bool(force_materialize)
        elif bool(cycle_cfg.get("materialize_from_labels_csv", False)):
            # Phase 2 should usually train on a fixed dataset snapshot.
            cycle_cfg["materialize_from_labels_csv"] = False
        if sample_every_override not in (None, "", "null"):
            cycle_cfg["sample_every_n_steps"] = int(sample_every_override)
        if seed_mode == "fixed":
            cycle_cfg["seed"] = int(seed_base)
        else:
            cycle_cfg["seed"] = int(seed_base + (cycle_idx - 1) * seed_step)
        for k, v in train_overrides.items():
            cycle_cfg[k] = v
        for k, v in mode_train_overrides.items():
            cycle_cfg[k] = v

        if phase2_mode == "inpaint":
            mask_dir_raw = cycle_cfg.get("lora_mask_dir")
            lora_mask_mode = str(cycle_cfg.get("lora_mask_mode", "directory")).strip().lower()
            if lora_mask_mode == "directory" and mask_dir_raw in (None, "", "null"):
                raise SystemExit(
                    "phase2_mode='inpaint' with lora_mask_mode='directory' requires lora_mask_dir. "
                    "Set it in config.inpaint_train_overrides or notebook overrides."
                )
            labels_csv_raw = selector_cfg.get("labels_csv", train_overrides.get("labels_csv_path"))
            if labels_csv_raw in (None, "", "null"):
                raise SystemExit(
                    "phase2_mode='inpaint' requires selector.labels_csv or train_overrides.labels_csv_path."
                )
            selector_labels_csv = _resolve_path(labels_csv_raw, cfg_dir=cfg_dir, project_root=project_root)
            selector_mask_dir_raw = selector_cfg.get("mask_dir", mask_dir_raw)
            if selector_mask_dir_raw in (None, "", "null"):
                raise SystemExit(
                    "phase2_mode='inpaint' requires selector.mask_dir (directory with source masks for selector benchmarking)."
                )
            selector_mask_dir = _resolve_path(selector_mask_dir_raw, cfg_dir=cfg_dir, project_root=project_root)
            if not selector_labels_csv.is_file():
                raise SystemExit(f"selector.labels_csv not found: {selector_labels_csv}")
            if not selector_mask_dir.is_dir():
                raise SystemExit(f"selector.mask_dir not found: {selector_mask_dir}")

        _write_yaml(cycle_cfg_path, cycle_cfg)

        print(f"\n[cycle {cycle_idx}/{cycles}] start={current_start_ckpt.name}")
        _raise_if_training_running(cycle_out_dir, f"Phase 2 {cycle_name}")
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

        if phase2_mode == "inpaint":
            selector_cmd = [
                python_exec,
                str(selector_script),
                "--run-dir",
                str(cycle_out_dir),
                "--base-model",
                str(base_model),
                "--labels-csv",
                str(selector_labels_csv),
                "--label-col",
                selector_label_col,
                "--mask-dir",
                str(selector_mask_dir),
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
                str(selector_mask_feather_radius),
                "--same-class-mask-dilate-px",
                str(selector_same_class_mask_dilate_px),
                "--cross-class-mask-dilate-px",
                str(selector_cross_class_mask_dilate_px),
                "--seed-base",
                str(selector_seed_base),
                "--cross-class-weight",
                str(selector_cross_class_weight),
                "--same-class-weight",
                str(selector_same_class_weight),
                "--prompt-template",
                selector_prompt_template,
                "--tokens",
                ",".join(selector_tokens),
                "--checkpoint-names",
                checkpoint_names,
                "--out-dir",
                str(cycle_eval_dir),
            ]
            if exclude_last:
                selector_cmd.append("--exclude-last")
            if max_checkpoints > 0:
                selector_cmd += ["--max-checkpoints", str(max_checkpoints)]
            if descriptors_path is not None:
                selector_cmd += ["--descriptors-json", str(descriptors_path)]
            if selector_reuse_selected_sources and shared_selected_sources.is_file():
                selector_cmd += ["--selected-sources-csv", str(shared_selected_sources)]
        else:
            selector_cmd = [
                python_exec,
                str(selector_script),
                "--run-dir",
                str(cycle_out_dir),
                "--base-model",
                str(base_model),
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
                "--steps",
                str(selector_steps),
                "--guidance-scale",
                str(selector_guidance),
                "--negative-prompt",
                selector_negative,
                "--width",
                str(selector_width),
                "--height",
                str(selector_height),
                "--seed-base",
                str(selector_seed_base),
                "--prompt-template",
                selector_prompt_template,
                "--tokens",
                ",".join(selector_tokens),
                "--checkpoint-names",
                checkpoint_names,
                "--out-dir",
                str(cycle_eval_dir),
            ]
            if is_sdxl:
                selector_cmd.append("--is-sdxl")
            else:
                selector_cmd.append("--is-sd15")
            if exclude_last:
                selector_cmd.append("--exclude-last")
            if max_checkpoints > 0:
                selector_cmd += ["--max-checkpoints", str(max_checkpoints)]
            if descriptors_path is not None:
                selector_cmd += ["--descriptors-json", str(descriptors_path)]

        _run(selector_cmd, cwd=project_root, dry_run=args.dry_run)

        if args.dry_run:
            print("dry-run mode: command planning finished.")
            break

        ranking_json = cycle_eval_dir / "checkpoint_ranking.json"
        ranking_csv = cycle_eval_dir / "checkpoint_ranking.csv"
        if not ranking_json.is_file():
            raise SystemExit(f"Expected selector ranking file not found: {ranking_json}")
        ranking = json.loads(ranking_json.read_text(encoding="utf-8"))
        if phase2_mode == "inpaint" and selector_reuse_selected_sources:
            selected_sources_raw = ranking.get("shared_selected_sources_csv")
            if selected_sources_raw:
                selected_sources_path = Path(selected_sources_raw)
                if selected_sources_path.is_file() and selected_sources_path.resolve() != shared_selected_sources.resolve():
                    shutil.copy2(selected_sources_path, shared_selected_sources)
        best = ranking.get("best", {}) or {}
        cycle_best_name = str(best.get("checkpoint", "")).strip()
        if not cycle_best_name:
            raise SystemExit(f"Selector ranking has no best checkpoint: {ranking_json}")
        cycle_best_ckpt = cycle_out_dir / cycle_best_name
        if not cycle_best_ckpt.is_file():
            raise SystemExit(f"Best checkpoint listed by selector is missing: {cycle_best_ckpt}")
        if phase2_mode == "inpaint":
            selector_metric_name = "combined_reward"
            selector_secondary_metric_name = "cross_class_margin_score"
            cycle_best_primary = _as_float(best.get("combined_reward", 0.0), 0.0)
            cycle_best_secondary = _as_float(best.get("cross_class_margin_score", 0.0), 0.0)
            cycle_best_cross = _as_float(best.get("cross_class_target_top1", 0.0), 0.0)
            cycle_best_same = _as_float(best.get("same_class_target_top1", 0.0), 0.0)
            cycle_best_cross_margin = _as_float(best.get("cross_class_margin_score", 0.0), 0.0)
            cycle_best_cross_margin_mean = _as_float(best.get("cross_class_target_source_prob_margin_mean", 0.0), 0.0)
            cycle_best_same_prob = _as_float(best.get("same_class_target_prob_mean", 0.0), 0.0)
            cycle_best_macro = cycle_best_primary
            cycle_best_top1 = cycle_best_secondary
        else:
            selector_metric_name = "macro_consistency"
            selector_secondary_metric_name = "consistency_top1"
            cycle_best_primary = _as_float(best.get("macro_consistency", 0.0), 0.0)
            cycle_best_secondary = _as_float(best.get("consistency_top1", 0.0), 0.0)
            cycle_best_cross = ""
            cycle_best_same = ""
            cycle_best_cross_margin = ""
            cycle_best_cross_margin_mean = ""
            cycle_best_same_prob = ""
            cycle_best_macro = cycle_best_primary
            cycle_best_top1 = cycle_best_secondary

        improved = cycle_best_primary > (global_best_score + epsilon)
        if improved:
            global_best_score = cycle_best_primary
            global_best_ckpt = cycle_best_ckpt
            no_improve = 0
        else:
            no_improve += 1

        reverted = False
        if (
            revert_on_drop
            and global_best_ckpt is not None
            and (cycle_best_primary + epsilon) < global_best_score
        ):
            next_start_ckpt = global_best_ckpt
            reverted = True
        else:
            next_start_ckpt = cycle_best_ckpt

        row = {
            "cycle": cycle_idx,
            "start_checkpoint": str(current_start_ckpt),
            "cycle_best_checkpoint": str(cycle_best_ckpt),
            "selector_metric_name": selector_metric_name,
            "selector_secondary_metric_name": selector_secondary_metric_name,
            "cycle_best_primary_score": cycle_best_primary,
            "cycle_best_secondary_score": cycle_best_secondary,
            "cycle_best_macro_consistency": cycle_best_macro,
            "cycle_best_consistency_top1": cycle_best_top1,
            "cycle_best_cross_class_target_top1": cycle_best_cross,
            "cycle_best_same_class_target_top1": cycle_best_same,
            "cycle_best_cross_class_margin_score": cycle_best_cross_margin,
            "cycle_best_cross_class_target_prob_margin_mean": cycle_best_cross_margin_mean,
            "cycle_best_same_class_target_prob_mean": cycle_best_same_prob,
            "global_best_checkpoint": str(global_best_ckpt) if global_best_ckpt is not None else "",
            "global_best_primary_score": global_best_score,
            "global_best_macro_consistency": global_best_score,
            "improved_global_best": improved,
            "reverted_to_global_best": reverted,
            "next_start_checkpoint": str(next_start_ckpt),
            "ranking_json": str(ranking_json),
            "ranking_csv": str(ranking_csv),
            "phase2_mode": phase2_mode,
        }
        history.append(row)
        _dump_cycle_tables(history, work_dir)

        if phase2_mode == "inpaint":
            print(
                f"[cycle {cycle_idx}] best={cycle_best_ckpt.name} combined={cycle_best_primary:.4f} "
                f"cross_margin={cycle_best_cross_margin_mean:.4f} cross_top1={cycle_best_cross:.4f} "
                f"same_prob={cycle_best_same_prob:.4f} same_top1={cycle_best_same:.4f} | "
                f"global={global_best_score:.4f}"
            )
        else:
            print(
                f"[cycle {cycle_idx}] best={cycle_best_ckpt.name} macro={cycle_best_macro:.4f} "
                f"top1={cycle_best_top1:.4f} | global={global_best_score:.4f}"
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
            "phase2_mode": phase2_mode,
            "best_checkpoint": str(global_best_ckpt),
            "best_score_name": "combined_reward" if phase2_mode == "inpaint" else "macro_consistency",
            "best_score": global_best_score,
            "best_macro_consistency": global_best_score,
            "history_json": str(work_dir / "phase2_history.json"),
            "history_csv": str(work_dir / "phase2_history.csv"),
        }
        (work_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\nPhase-2 completed.")
        print(f"Best checkpoint: {global_best_ckpt}")
        if phase2_mode == "inpaint":
            print(f"Best combined reward: {global_best_score:.4f}")
        else:
            print(f"Best macro consistency: {global_best_score:.4f}")
        print(f"History CSV: {work_dir / 'phase2_history.csv'}")


if __name__ == "__main__":
    main()
