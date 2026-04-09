#!/usr/bin/env python
"""
Render runtime phase configs from templates + params.yaml.

Single source of truth for shared model paths:
  params.models.sdxl_base_model
  params.models.classifier_checkpoint
  params.models.kohya_scripts_dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj or {}


def _dump_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _required_path(name: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise SystemExit(f"Missing required params value: {name}")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Render runtime YAML configs from templates and params.")
    ap.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    ap.add_argument("--phase1-template", default="configs/sdxl_lora_phase1_skin_histology.yaml")
    ap.add_argument("--phase2-template", default="configs/sdxl_lora_phase2_reward_skin_histology.yaml")
    ap.add_argument("--phase3-template", default="configs/sdxl_lora_phase3_morph_reward_skin_histology.yaml")
    ap.add_argument("--phase1-out", default="configs/runtime/sdxl_lora_phase1_skin_histology.runtime.yaml")
    ap.add_argument("--phase2-out", default="configs/runtime/sdxl_lora_phase2_reward_skin_histology.runtime.yaml")
    ap.add_argument("--phase3-out", default="configs/runtime/sdxl_lora_phase3_morph_reward_skin_histology.runtime.yaml")
    ap.add_argument("--summary-out", default="data/metadata/runtime_config_sync.json")
    args = ap.parse_args()

    params = _load_yaml(Path(args.params))
    models = params.get("models", {}) or {}
    dataset = params.get("dataset", {}) or {}
    gradcam = params.get("gradcam", {}) or {}

    sdxl_base_model = _required_path("models.sdxl_base_model", models.get("sdxl_base_model"))
    classifier_ckpt = _required_path("models.classifier_checkpoint", models.get("classifier_checkpoint"))
    kohya_dir = _required_path("models.kohya_scripts_dir", models.get("kohya_scripts_dir"))
    labels_csv = _required_path("dataset.pairs_csv", dataset.get("pairs_csv"))
    mask_dir = _required_path("gradcam.output_dir", gradcam.get("output_dir"))

    phase1 = _load_yaml(Path(args.phase1_template))
    phase2 = _load_yaml(Path(args.phase2_template))
    phase3 = _load_yaml(Path(args.phase3_template))

    phase1["base_model_path"] = sdxl_base_model
    phase1["kohya_scripts_dir"] = kohya_dir
    phase1["labels_csv_path"] = labels_csv
    phase1["lora_mask_dir"] = mask_dir

    phase2.setdefault("selector", {})
    phase2["base_train_config"] = args.phase1_out
    phase2["selector"]["base_model"] = sdxl_base_model
    phase2["selector"]["classifier_ckpt"] = classifier_ckpt
    phase2["selector"]["labels_csv"] = labels_csv
    phase2["selector"]["mask_dir"] = mask_dir
    phase2.setdefault("train_overrides", {})
    phase2["train_overrides"]["labels_csv_path"] = labels_csv
    phase2.setdefault("inpaint_train_overrides", {})
    phase2["inpaint_train_overrides"]["lora_mask_dir"] = mask_dir

    phase3.setdefault("selector", {})
    phase3["base_train_config"] = args.phase1_out
    phase3["selector"]["base_model"] = sdxl_base_model
    phase3["selector"]["classifier_ckpt"] = classifier_ckpt
    phase3["selector"]["labels_csv"] = labels_csv
    phase3["selector"]["mask_dir"] = mask_dir
    phase3.setdefault("train_overrides", {})
    phase3["train_overrides"]["labels_csv_path"] = labels_csv
    phase3.setdefault("inpaint_train_overrides", {})
    phase3["inpaint_train_overrides"]["lora_mask_dir"] = mask_dir

    _dump_yaml(Path(args.phase1_out), phase1)
    _dump_yaml(Path(args.phase2_out), phase2)
    _dump_yaml(Path(args.phase3_out), phase3)

    summary = {
        "phase1_out": args.phase1_out,
        "phase2_out": args.phase2_out,
        "phase3_out": args.phase3_out,
        "resolved_paths": {
            "sdxl_base_model": sdxl_base_model,
            "classifier_checkpoint": classifier_ckpt,
            "kohya_scripts_dir": kohya_dir,
            "labels_csv": labels_csv,
            "mask_dir": mask_dir,
            "gradcam_model_config": gradcam.get("model_config"),
        },
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Rendered runtime configs: {args.phase1_out}, {args.phase2_out}, {args.phase3_out}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
