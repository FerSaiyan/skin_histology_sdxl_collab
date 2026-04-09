#!/usr/bin/env python3
"""
Generate a fixed class panel for each LoRA checkpoint, score it with the
classifier, and rank checkpoints by class consistency.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch
from diffusers import StableDiffusionPipeline

try:
    from diffusers import StableDiffusionXLPipeline
except Exception:
    StableDiffusionXLPipeline = None


DEFAULT_TOKENS = ["healthy", "benign_lesion", "opmd", "cancer"]
DEFAULT_DESCRIPTORS = {
    "healthy": "normal oral mucosa, no suspicious lesion",
    "benign_lesion": "small benign-appearing oral lesion with smooth borders",
    "opmd": "oral potentially malignant disorder, leukoplakia-like irregular plaque",
    "cancer": "ulcerated malignant-appearing oral lesion with irregular infiltrative margins",
}


def _repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
        if (parent / "dvc.yaml").exists():
            return parent
    return Path.cwd().resolve()


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


def _parse_descriptors(raw: str | None) -> Dict[str, str]:
    if not raw:
        return dict(DEFAULT_DESCRIPTORS)
    p = Path(raw)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(raw)


def _list_checkpoints(run_dir: Path, include_last: bool, max_checkpoints: int | None) -> List[Path]:
    ckpts = sorted(run_dir.glob("*.safetensors"))
    if not ckpts:
        raise SystemExit(f"No .safetensors checkpoints found in: {run_dir}")

    if not include_last:
        ckpts = [p for p in ckpts if p.name != "last.safetensors"]
    if not ckpts:
        raise SystemExit("No checkpoints left after filtering include_last=False.")

    def _key(p: Path):
        m = re.match(r"epoch-(\d+)\.safetensors$", p.name)
        if m:
            return (0, int(m.group(1)))
        if p.name == "last.safetensors":
            return (1, 10**9)
        return (2, p.name)

    ckpts = sorted(ckpts, key=_key)
    if max_checkpoints is not None and max_checkpoints > 0:
        ckpts = ckpts[-max_checkpoints:]
    return ckpts


def _build_pipe(base_model_path: Path, is_sdxl: bool, device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if is_sdxl:
        if StableDiffusionXLPipeline is None:
            raise SystemExit("StableDiffusionXLPipeline is unavailable in current diffusers install.")
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(base_model_path),
            torch_dtype=dtype,
            safety_checker=None,
            use_safetensors=True,
        )
    else:
        pipe = StableDiffusionPipeline.from_single_file(
            str(base_model_path),
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
    return pipe


def _prompt_text(template: str, token: str, descriptors: Dict[str, str]) -> str:
    descriptor = str(descriptors.get(token, "")).strip()
    descriptor_suffix = f", {descriptor}" if descriptor else ""
    return template.format(
        token=token,
        descriptor=descriptor,
        descriptor_suffix=descriptor_suffix,
    ).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Select best LoRA checkpoint using classifier consistency")
    ap.add_argument("--run-dir", required=True, help="LoRA run directory containing *.safetensors")
    ap.add_argument("--base-model", required=True, help="Base SD/SDXL model (.safetensors)")
    ap.add_argument("--classifier-ckpt", required=True, help="Classifier checkpoint (.pth)")
    ap.add_argument("--classifier-study-config", required=True, help="Classifier study YAML")
    ap.add_argument("--is-sdxl", action="store_true", help="Use SDXL pipeline (default)")
    ap.add_argument("--is-sd15", action="store_true", help="Use SD1.x pipeline")
    ap.add_argument("--device", default="auto", help="Generation device: auto|cuda|cpu")
    ap.add_argument("--classifier-device", default="auto", help="Scoring device: auto|cuda|cpu")
    ap.add_argument("--batch-size", type=int, default=64, help="Classifier scoring batch size")
    ap.add_argument("--images-per-class", type=int, default=3, help="Images to generate per class per checkpoint")
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--negative-prompt", default="lowres,bad anatomy")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed-base", type=int, default=222)
    ap.add_argument("--exclude-last", action="store_true", help="Exclude last.safetensors")
    ap.add_argument(
        "--checkpoint-names",
        default="",
        help="Optional comma-separated checkpoint filenames to evaluate (overrides max-checkpoints)",
    )
    ap.add_argument("--max-checkpoints", type=int, default=0, help="If >0, evaluate only latest N checkpoints")
    ap.add_argument(
        "--prompt-template",
        default="clinical intraoral photo, diagnosis: {token}{descriptor_suffix}",
        help="Prompt template with {token} and optional {descriptor_suffix}",
    )
    ap.add_argument("--descriptors-json", default=None, help="JSON string or path with token->descriptor map")
    ap.add_argument("--tokens", default=",".join(DEFAULT_TOKENS), help="Comma-separated class token order")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <run-dir>/checkpoint_selection)",
    )
    args = ap.parse_args()

    repo = _repo_root(Path(__file__))
    run_dir = Path(args.run_dir).resolve()
    base_model = Path(args.base_model).resolve()
    clf_ckpt = Path(args.classifier_ckpt).resolve()
    clf_cfg = Path(args.classifier_study_config).resolve()
    scorer_script = repo / "scripts" / "synthetic_data" / "score_lora_samples_with_classifier.py"
    if not scorer_script.is_file():
        raise SystemExit(f"Could not find scorer script: {scorer_script}")

    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir not found: {run_dir}")
    if not base_model.is_file():
        raise SystemExit(f"--base-model not found: {base_model}")
    if not clf_ckpt.is_file():
        raise SystemExit(f"--classifier-ckpt not found: {clf_ckpt}")
    if not clf_cfg.is_file():
        raise SystemExit(f"--classifier-study-config not found: {clf_cfg}")

    tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    if not tokens:
        raise SystemExit("No tokens provided.")
    descriptors = _parse_descriptors(args.descriptors_json)
    include_last = not bool(args.exclude_last)
    explicit_ckpts = [x.strip() for x in str(args.checkpoint_names).split(",") if x.strip()]
    if explicit_ckpts:
        ckpts = []
        for name in explicit_ckpts:
            p = run_dir / name
            if not p.is_file():
                raise SystemExit(f"Checkpoint listed in --checkpoint-names not found: {p}")
            ckpts.append(p)
    else:
        max_ckpts = args.max_checkpoints if args.max_checkpoints > 0 else None
        ckpts = _list_checkpoints(run_dir, include_last=include_last, max_checkpoints=max_ckpts)

    is_sdxl = not args.is_sd15
    if args.is_sdxl:
        is_sdxl = True
    device = _resolve_device(args.device)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "checkpoint_selection")
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = _build_pipe(base_model, is_sdxl=is_sdxl, device=device)

    rows: List[Dict] = []
    try:
        for ckpt in ckpts:
            ckpt_name = ckpt.name
            ckpt_stem = ckpt.stem
            ckpt_dir = out_dir / ckpt_stem
            images_dir = ckpt_dir / "images"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)

            prompts = [_prompt_text(args.prompt_template, tok, descriptors) for tok in tokens]
            prompts_file = ckpt_dir / "prompts_eval.txt"
            prompts_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")

            print(f"[gen] {ckpt_name}: generating {len(tokens) * args.images_per_class} images")
            pipe.load_lora_weights(str(run_dir), weight_name=ckpt_name)
            try:
                for class_idx, prompt in enumerate(prompts):
                    for rep_idx in range(args.images_per_class):
                        seed = int(args.seed_base + class_idx * 10000 + rep_idx)
                        generator = torch.Generator(device=device.type).manual_seed(seed)
                        result = pipe(
                            prompt=prompt,
                            negative_prompt=args.negative_prompt,
                            width=int(args.width),
                            height=int(args.height),
                            num_inference_steps=int(args.steps),
                            guidance_scale=float(args.guidance_scale),
                            generator=generator,
                        )
                        image = result.images[0]
                        out_name = f"{rep_idx:04d}_{class_idx}_{seed}.png"
                        image.save(images_dir / out_name)
            finally:
                pipe.unload_lora_weights()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            score_json = ckpt_dir / "classifier_eval.json"
            score_csv = ckpt_dir / "classifier_eval.csv"
            score_cmd = [
                sys.executable,
                str(scorer_script),
                "--images-dir",
                str(images_dir),
                "--checkpoint",
                str(clf_ckpt),
                "--study-config",
                str(clf_cfg),
                "--prompts-file",
                str(prompts_file),
                "--batch-size",
                str(args.batch_size),
                "--device",
                str(args.classifier_device),
                "--output-json",
                str(score_json),
                "--output-csv",
                str(score_csv),
            ]
            print("[score]", ckpt_name)
            subprocess.run(score_cmd, check=True, cwd=repo)
            summary = json.loads(score_json.read_text(encoding="utf-8"))

            per_class = summary.get("per_expected_top1", {}) or {}
            macro_consistency = float(sum(per_class.get(tok, 0.0) for tok in tokens) / float(len(tokens)))
            row = {
                "checkpoint": ckpt_name,
                "n_images": int(summary.get("n_images", 0)),
                "consistency_top1": float(summary.get("consistency_top1", 0.0) or 0.0),
                "macro_consistency": macro_consistency,
                "pred_distribution": summary.get("pred_distribution", {}),
                "per_expected_top1": per_class,
                "score_json": str(score_json),
                "score_csv": str(score_csv),
                "images_dir": str(images_dir),
            }
            rows.append(row)
            print(
                f"[done] {ckpt_name}: consistency={row['consistency_top1']:.4f}, "
                f"macro={row['macro_consistency']:.4f}"
            )
    finally:
        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("No checkpoints were evaluated.")

    rows = sorted(rows, key=lambda r: (r["macro_consistency"], r["consistency_top1"]), reverse=True)
    best = rows[0]

    summary_json = out_dir / "checkpoint_ranking.json"
    summary_csv = out_dir / "checkpoint_ranking.csv"
    summary_json.write_text(json.dumps({"best": best, "rows": rows}, indent=2), encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["rank", "checkpoint", "macro_consistency", "consistency_top1", "n_images", "score_json"],
        )
        w.writeheader()
        for i, row in enumerate(rows, start=1):
            w.writerow(
                {
                    "rank": i,
                    "checkpoint": row["checkpoint"],
                    "macro_consistency": row["macro_consistency"],
                    "consistency_top1": row["consistency_top1"],
                    "n_images": row["n_images"],
                    "score_json": row["score_json"],
                }
            )

    print("\nBest checkpoint:")
    print(f"  {best['checkpoint']}")
    print(f"  macro_consistency={best['macro_consistency']:.4f}")
    print(f"  consistency_top1={best['consistency_top1']:.4f}")
    print(f"Saved ranking JSON: {summary_json}")
    print(f"Saved ranking CSV : {summary_csv}")


if __name__ == "__main__":
    main()
