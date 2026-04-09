#!/usr/bin/env python
"""
Run inpainting on extracted ROI patches using SDXL/LoRA models.

This script:
- Loads extracted patches and their masks
- Applies SDXL inpainting with LoRA weights
- Generates edited patches
- Saves generation metadata for provenance

Note: Requires actual SDXL base model and LoRA checkpoint paths.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import torch
    from diffusers import AutoPipelineForInpainting, StableDiffusionXLInpaintPipeline
    from diffusers.utils import load_image

    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False
    torch = None


def _load_inpaint_pipeline(
    base_model: str,
    lora_weights: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "fp16",
) -> Any:
    """Load SDXL inpainting pipeline with optional LoRA."""
    if not HAS_DIFFUSERS:
        raise ImportError("diffusers and torch are required for inpainting")

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32

    try:
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            variant="fp16" if dtype == "fp16" else None,
        )
    except Exception:
        pipe = AutoPipelineForInpainting.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
        )

    if lora_weights:
        pipe.load_lora_weights(lora_weights)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    return pipe


def _generate_prompt(coarse_label: str, template: str) -> str:
    """Generate inpainting prompt from label."""
    label_desc = {
        "cancer": "malignant skin histology pattern with dysplastic morphology",
        "non_cancer": "normal skin tissue architecture, preserved stratification",
    }
    desc = label_desc.get(coarse_label, "skin histology tissue pattern")
    return template.format(token=coarse_label, descriptor_suffix=f", {desc}")


def _inpaint_patch(
    pipe: Any,
    patch: np.ndarray,
    mask: np.ndarray,
    prompt: str,
    negative_prompt: str,
    seed: int,
    num_steps: int = 40,
    guidance_scale: float = 5.0,
    strength: float = 0.55,
) -> np.ndarray:
    """Run inpainting on a single patch."""
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    patch_pil = Image.fromarray(patch)
    mask_pil = Image.fromarray(mask)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=patch_pil,
        mask_image=mask_pil,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generator,
    )

    return np.array(result.images[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run inpainting on ROI patches.")
    ap.add_argument("--metadata-csv", required=True, help="Path to patches_metadata.csv")
    ap.add_argument("--output-dir", required=True, help="Output directory for inpainted patches")
    ap.add_argument("--base-model", required=True, help="Path to SDXL base model")
    ap.add_argument("--lora-weights", default=None, help="Path to LoRA weights")
    ap.add_argument("--device", default="cuda", help="Device for inference")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--num-steps", type=int, default=40)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--prompt-template", default="H&E skin histology slide, tissue pattern: {token}{descriptor_suffix}")
    ap.add_argument("--negative-prompt", default="lowres,blurry,artifact,cartoon,artistic,unrealistic stain")
    ap.add_argument("--max-patches", type=int, default=0, help="Limit patches (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = ap.parse_args()

    import pandas as pd

    meta_df = pd.read_csv(args.metadata_csv)

    if args.max_patches > 0:
        meta_df = meta_df.head(args.max_patches)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Patches to process: {len(meta_df)}")
        print(f"Base model: {args.base_model}")
        print(f"LoRA weights: {args.lora_weights}")
        for idx, row in meta_df.iterrows():
            print(f"  [{idx}] {row['slice_id']}")
        return

    if not HAS_DIFFUSERS:
        print("ERROR: diffusers not installed. Cannot run actual inpainting.")
        print("Install with: pip install diffusers torch accelerate safetensors")
        return

    print(f"Loading pipeline: {args.base_model}")
    pipe = _load_inpaint_pipeline(
        args.base_model,
        lora_weights=args.lora_weights,
        device=args.device,
        dtype=args.dtype,
    )

    gen_metadata: List[Dict[str, Any]] = []

    for idx, row in meta_df.iterrows():
        slice_id = row["slice_id"]
        patch_path = Path(row["patch_image"])
        mask_path = Path(row["patch_mask"])

        if not patch_path.exists() or not mask_path.exists():
            print(f"[WARN] Missing files for {slice_id}")
            continue

        print(f"[{idx+1}/{len(meta_df)}] Inpainting {slice_id}")

        patch = np.array(Image.open(patch_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        seed = args.seed_base + idx
        coarse_label = row.get("coarse_label", "")
        prompt = _generate_prompt(coarse_label, args.prompt_template)

        inpainted = _inpaint_patch(
            pipe,
            patch,
            mask,
            prompt,
            args.negative_prompt,
            seed,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
        )

        out_path = output_dir / "images" / f"{slice_id}_inpainted.png"
        Image.fromarray(inpainted).save(out_path)

        gen_meta = {
            "slice_id": slice_id,
            "source_patch": str(patch_path),
            "source_mask": str(mask_path),
            "inpainted_image": str(out_path),
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "strength": args.strength,
            "base_model": args.base_model,
            "lora_weights": args.lora_weights,
            "coarse_label": coarse_label,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        gen_metadata.append(gen_meta)

    if gen_metadata:
        gen_df = pd.DataFrame(gen_metadata)
        gen_meta_path = output_dir / "metadata" / "inpaint_metadata.csv"
        gen_df.to_csv(gen_meta_path, index=False)
        print(f"Saved generation metadata: {gen_meta_path}")

        stats = {
            "total_patches_processed": len(gen_metadata),
            "base_model": args.base_model,
            "lora_weights": args.lora_weights,
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "strength": args.strength,
        }
        stats_path = output_dir / "metadata" / "inpaint_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
