#!/usr/bin/env python3
"""
Select the best LoRA checkpoint using the Grad-CAM inpainting benchmark as the
reward signal.

Each candidate checkpoint is benchmarked on the same fixed source-image panel.
Ranking is based on a weighted combination of:
- normalized cross-class classifier margin p(target) - p(source)
- same-class target-class probability

Top-1 metrics are still recorded and used as tie-breakers.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
        if (parent / "dvc.yaml").exists():
            return parent
    return Path.cwd().resolve()


def _list_checkpoints(run_dir: Path, include_last: bool, max_checkpoints: int | None) -> List[Path]:
    ckpts = sorted(run_dir.glob("*.safetensors"))
    if not ckpts:
        raise SystemExit(f"No .safetensors checkpoints found in: {run_dir}")

    if not include_last:
        ckpts = [p for p in ckpts if p.name != "last.safetensors"]
    if not ckpts:
        raise SystemExit("No checkpoints left after filtering include_last=False.")

    def _key(p: Path) -> Tuple[int, int | str]:
        m = re.match(r"epoch-(\d+)\.safetensors$", p.name)
        if m:
            return (0, int(m.group(1)))
        m = re.match(r"at-step(\d+)\.safetensors$", p.name)
        if m:
            return (1, int(m.group(1)))
        if p.name == "last.safetensors":
            return (2, 10**9)
        return (3, p.name)

    ckpts = sorted(ckpts, key=_key)
    if max_checkpoints is not None and max_checkpoints > 0:
        ckpts = ckpts[-max_checkpoints:]
    return ckpts


def _as_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select best LoRA checkpoint using the Grad-CAM inpainting benchmark")
    ap.add_argument("--run-dir", required=True, help="Directory containing candidate *.safetensors checkpoints")
    ap.add_argument("--base-model", required=True, help="Base SDXL model or inpaint checkpoint")
    ap.add_argument("--labels-csv", required=True, help="CSV with source image_path and label column")
    ap.add_argument("--label-col", default="coarse_label", help="Label column in labels CSV")
    ap.add_argument("--mask-dir", required=True, help="Grad-CAM mask directory")
    ap.add_argument("--classifier-ckpt", required=True, help="Classifier checkpoint (.pth)")
    ap.add_argument("--classifier-study-config", required=True, help="Classifier study YAML")
    ap.add_argument("--device", default="auto", help="Generation device: auto|cuda|cpu")
    ap.add_argument("--classifier-device", default="cpu", help="Classifier device: auto|cuda|cpu")
    ap.add_argument("--batch-size", type=int, default=64, help="Unused compatibility arg for parity with phase2")
    ap.add_argument("--images-per-class", type=int, default=2, help="Source images to sample per class")
    ap.add_argument("--max-sources", type=int, default=0, help="Optional cap after class-balanced sampling")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument("--same-class-strength", type=float, default=None)
    ap.add_argument("--cross-class-strength", type=float, default=None)
    ap.add_argument("--lora-scale", type=float, default=0.75)
    ap.add_argument("--negative-prompt", default="lowres,bad anatomy")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--mask-feather-radius", type=float, default=6.0)
    ap.add_argument("--same-class-mask-dilate-px", type=float, default=0.0)
    ap.add_argument("--cross-class-mask-dilate-px", type=float, default=0.0)
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
    ap.add_argument("--seed-base", type=int, default=222)
    ap.add_argument(
        "--cross-class-weight",
        type=float,
        default=0.75,
        help="Weight for cross-class target accuracy in the combined score.",
    )
    ap.add_argument(
        "--same-class-weight",
        type=float,
        default=0.25,
        help="Weight for same-class target accuracy in the combined score.",
    )
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
    ap.add_argument("--descriptors-json", default=None, help="JSON string or file with token->descriptor map")
    ap.add_argument("--tokens", default="healthy,benign_lesion,opmd,cancer", help="Comma-separated class order")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <run-dir>/checkpoint_selection_inpaint)",
    )
    ap.add_argument(
        "--selected-sources-csv",
        default=None,
        help="Optional fixed selected_sources.csv to reuse across candidate benchmarks.",
    )
    args = ap.parse_args()

    repo = _repo_root(Path(__file__))
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir not found: {run_dir}")

    benchmark_script = repo / "scripts" / "synthetic_data" / "benchmark_lora_inpaint_with_classifier.py"
    if not benchmark_script.is_file():
        raise SystemExit(f"Could not find benchmark script: {benchmark_script}")

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
        ckpts = _list_checkpoints(run_dir, include_last=not bool(args.exclude_last), max_checkpoints=max_ckpts)

    if not ckpts:
        raise SystemExit("No checkpoints selected for evaluation.")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "checkpoint_selection_inpaint")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.selected_sources_csv:
        shared_selected_sources = Path(args.selected_sources_csv).resolve()
        shared_selected_sources.parent.mkdir(parents=True, exist_ok=True)
    else:
        shared_selected_sources = out_dir / "selected_sources_shared.csv"

    rows: List[Dict] = []
    for ckpt in ckpts:
        ckpt_dir = out_dir / ckpt.stem
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        bench_cmd = [
            sys.executable,
            str(benchmark_script),
            "--labels-csv",
            str(Path(args.labels_csv).resolve()),
            "--label-col",
            args.label_col,
            "--mask-dir",
            str(Path(args.mask_dir).resolve()),
            "--base-model",
            str(Path(args.base_model).resolve()),
            "--model",
            f"candidate={ckpt}",
            "--classifier-ckpt",
            str(Path(args.classifier_ckpt).resolve()),
            "--classifier-study-config",
            str(Path(args.classifier_study_config).resolve()),
            "--out-dir",
            str(ckpt_dir),
            "--device",
            args.device,
            "--classifier-device",
            args.classifier_device,
            "--tokens",
            args.tokens,
            "--prompt-template",
            args.prompt_template,
            "--negative-prompt",
            args.negative_prompt,
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--steps",
            str(args.steps),
            "--guidance-scale",
            str(args.guidance_scale),
            "--strength",
            str(args.strength),
            "--same-class-strength",
            str(args.same_class_strength) if args.same_class_strength is not None else str(args.strength),
            "--cross-class-strength",
            str(args.cross_class_strength) if args.cross_class_strength is not None else str(args.strength),
            "--lora-scale",
            str(args.lora_scale),
            "--mask-feather-radius",
            str(args.mask_feather_radius),
            "--same-class-mask-dilate-px",
            str(args.same_class_mask_dilate_px),
            "--cross-class-mask-dilate-px",
            str(args.cross_class_mask_dilate_px),
            "--sources-per-class",
            str(args.images_per_class),
            "--max-sources",
            str(args.max_sources),
            "--seed-base",
            str(args.seed_base),
        ]
        if args.cross_class_target_strengths_json not in (None, "", "null"):
            bench_cmd += ["--cross-class-target-strengths-json", str(args.cross_class_target_strengths_json)]
        if args.cross_class_target_mask_dilate_json not in (None, "", "null"):
            bench_cmd += ["--cross-class-target-mask-dilate-json", str(args.cross_class_target_mask_dilate_json)]
        if args.descriptors_json not in (None, "", "null"):
            bench_cmd += ["--descriptors-json", str(args.descriptors_json)]
        if shared_selected_sources.is_file():
            bench_cmd += ["--selected-sources-csv", str(shared_selected_sources)]

        print(f"[benchmark] {ckpt.name}")
        subprocess.run(bench_cmd, check=True, cwd=repo)

        candidate_selected = ckpt_dir / "selected_sources.csv"
        if not shared_selected_sources.is_file() and candidate_selected.is_file():
            shared_selected_sources.write_text(candidate_selected.read_text(encoding="utf-8"), encoding="utf-8")

        summary_path = ckpt_dir / "benchmark_summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"Expected benchmark summary not found: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = (summary.get("metrics_by_model", {}) or {}).get("candidate", {}) or {}
        same_metrics = (metrics.get("same_class_target_top1", {}) or {})
        cross_metrics = (metrics.get("cross_class_target_top1", {}) or {})
        same_top1 = _as_float(same_metrics.get("top1"), 0.0)
        cross_top1 = _as_float(cross_metrics.get("top1"), 0.0)
        same_target_prob_mean = _as_float(same_metrics.get("target_prob_mean"), 0.0)
        cross_margin_mean = _as_float(cross_metrics.get("target_source_prob_margin_mean"), -1.0)
        cross_margin_score = _as_float(cross_metrics.get("target_source_prob_margin_score"), 0.0)
        combined_score = (
            float(args.same_class_weight) * same_target_prob_mean
            + float(args.cross_class_weight) * cross_margin_score
        )

        row = {
            "checkpoint": ckpt.name,
            "same_class_target_top1": same_top1,
            "cross_class_target_top1": cross_top1,
            "same_class_target_prob_mean": same_target_prob_mean,
            "cross_class_target_source_prob_margin_mean": cross_margin_mean,
            "cross_class_margin_score": cross_margin_score,
            "combined_reward": combined_score,
            "summary_json": str(summary_path),
            "rows_csv": str(ckpt_dir / "benchmark_rows.csv"),
            "panel_index_csv": str(ckpt_dir / "panel_index.csv"),
            "benchmark_dir": str(ckpt_dir),
        }
        rows.append(row)
        print(
            f"[done] {ckpt.name}: combined={combined_score:.4f}, "
            f"cross_margin={cross_margin_mean:.4f}, cross_top1={cross_top1:.4f}, "
            f"same_prob={same_target_prob_mean:.4f}, same_top1={same_top1:.4f}"
        )

    rows = sorted(
        rows,
        key=lambda r: (
            r["combined_reward"],
            r["cross_class_margin_score"],
            r["cross_class_target_top1"],
            r["same_class_target_prob_mean"],
            r["same_class_target_top1"],
        ),
        reverse=True,
    )
    best = rows[0]

    summary_json = out_dir / "checkpoint_ranking.json"
    summary_csv = out_dir / "checkpoint_ranking.csv"
    payload = {
        "best": best,
        "rows": rows,
        "score_weights": {
            "same_class_weight": float(args.same_class_weight),
            "cross_class_weight": float(args.cross_class_weight),
        },
        "primary_metrics": {
            "same_class": "target_prob_mean",
            "cross_class": "target_source_prob_margin_score",
        },
        "shared_selected_sources_csv": str(shared_selected_sources) if shared_selected_sources.is_file() else None,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "checkpoint",
                "combined_reward",
                "cross_class_margin_score",
                "cross_class_target_source_prob_margin_mean",
                "cross_class_target_top1",
                "same_class_target_prob_mean",
                "same_class_target_top1",
                "summary_json",
            ],
        )
        w.writeheader()
        for i, row in enumerate(rows, start=1):
            w.writerow(
                {
                    "rank": i,
                    "checkpoint": row["checkpoint"],
                    "combined_reward": row["combined_reward"],
                    "cross_class_margin_score": row["cross_class_margin_score"],
                    "cross_class_target_source_prob_margin_mean": row["cross_class_target_source_prob_margin_mean"],
                    "cross_class_target_top1": row["cross_class_target_top1"],
                    "same_class_target_prob_mean": row["same_class_target_prob_mean"],
                    "same_class_target_top1": row["same_class_target_top1"],
                    "summary_json": row["summary_json"],
                }
            )

    print("\nBest checkpoint:")
    print(f"  {best['checkpoint']}")
    print(f"  combined_reward={best['combined_reward']:.4f}")
    print(f"  cross_class_margin_score={best['cross_class_margin_score']:.4f}")
    print(f"  cross_class_target_source_prob_margin_mean={best['cross_class_target_source_prob_margin_mean']:.4f}")
    print(f"  cross_class_target_top1={best['cross_class_target_top1']:.4f}")
    print(f"  same_class_target_prob_mean={best['same_class_target_prob_mean']:.4f}")
    print(f"  same_class_target_top1={best['same_class_target_top1']:.4f}")
    print(f"Saved ranking JSON: {summary_json}")
    print(f"Saved ranking CSV : {summary_csv}")


if __name__ == "__main__":
    main()
