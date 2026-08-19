#!/usr/bin/env python3
"""
Generate per-tile SDXL LoRA inpainting samples for classification experiments.

One run produces samples for a single LoRA scale and writes them under:

  <output-dir>/str_<scale>x/
    images/
    masks/
    metadata/

Filenames preserve the full source tile stem so patient/slide provenance stays
visible in every generated image.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

try:
    import torch
    from diffusers import AutoPipelineForInpainting, StableDiffusionXLInpaintPipeline
    from diffusers.image_processor import VaeImageProcessor

    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False
    torch = None
    VaeImageProcessor = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def _resolve_device(device_arg: str):
    d = (device_arg or "auto").strip().lower()
    if d == "auto":
        return "cuda" if HAS_DIFFUSERS and torch.cuda.is_available() else "cpu"
    if d == "cuda":
        if not HAS_DIFFUSERS or not torch.cuda.is_available():
            raise SystemExit("--device=cuda requested but CUDA is not available.")
        return "cuda"
    if d == "cpu":
        return "cpu"
    raise SystemExit(f"Unsupported --device value: {device_arg} (expected: auto|cuda|cpu)")


def _strength_folder_name(scale: float) -> str:
    if float(scale).is_integer():
        return f"str_{int(scale)}x"
    text = str(scale).replace(".", "p")
    return f"str_{text}x"


def _load_inpaint_pipeline(base_model: str, device: str, dtype: str):
    if not HAS_DIFFUSERS:
        raise SystemExit("diffusers/torch are required. Install them in the remote environment first.")

    torch_dtype = torch.float16 if dtype == "fp16" and device == "cuda" else torch.float32
    model_path = Path(base_model)
    is_single_file = model_path.is_file() if model_path.exists() else model_path.suffix.lower() in {".safetensors", ".ckpt"}

    if is_single_file:
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            str(base_model),
            torch_dtype=torch_dtype,
            safety_checker=None,
            use_safetensors=True,
        )
    else:
        try:
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                str(base_model),
                torch_dtype=torch_dtype,
                variant="fp16" if torch_dtype == torch.float16 else None,
            )
        except Exception:
            pipe = AutoPipelineForInpainting.from_pretrained(
                str(base_model),
                torch_dtype=torch_dtype,
            )

    pipe = pipe.to(device)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    pipe.set_progress_bar_config(disable=True)

    if VaeImageProcessor is not None:
        pipe.mask_processor = VaeImageProcessor(
            vae_scale_factor=pipe.vae_scale_factor,
            do_normalize=False,
            do_binarize=False,
            do_convert_grayscale=True,
        )

    return pipe


def _iter_tiles(tiles_dir: Path, tile_stems_file: Path | None, tile_offset: int, max_tiles: int) -> List[Path]:
    if tile_stems_file is not None:
        stems = [line.strip() for line in tile_stems_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = [tiles_dir / f"{stem}.png" for stem in stems]
    else:
        paths = sorted(p for p in tiles_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    paths = [p for p in paths if p.exists()]
    if tile_offset > 0:
        paths = paths[tile_offset:]
    if max_tiles > 0:
        paths = paths[:max_tiles]
    return paths


def _source_image_from_stem(stem: str) -> str:
    """Extract the parent source image name from a tile stem.

    Tile stems look like ``A13__jpg__x01024_y03328_s512``.
    The source image is the part before ``__x`` with ``__`` replaced by ``.``
    so ``A13__jpg`` becomes ``A13.jpg``.
    """
    m = re.match(r"(.+?)__x\d+_y\d+_s\d+", stem)
    if m:
        return m.group(1).replace("__", ".")
    return stem


def _sample_tiles_per_source(
    tile_paths: List[Path],
    max_per_source: int,
    seed: int,
) -> List[Path]:
    """Randomly sample up to *max_per_source* tiles per parent source image."""
    import random as _random

    by_source: dict[str, list[Path]] = {}
    for p in tile_paths:
        src = _source_image_from_stem(p.stem)
        by_source.setdefault(src, []).append(p)

    rng = _random.Random(seed)
    selected: list[Path] = []
    for src in sorted(by_source):
        candidates = by_source[src]
        n = min(max_per_source, len(candidates))
        picked = rng.sample(candidates, n)
        selected.extend(picked)

    selected.sort(key=lambda p: p.name)
    return selected


def _read_prompt(tile_path: Path, prompt_fallback: str) -> str:
    caption_path = tile_path.with_suffix(".caption")
    if caption_path.is_file():
        text = caption_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return prompt_fallback


def _keep_largest_connected_component(mask_arr: np.ndarray) -> np.ndarray:
    binary = mask_arr > 0
    if not binary.any():
        return np.zeros_like(mask_arr, dtype=mask_arr.dtype)
    labels, num_labels = ndi.label(binary)
    if num_labels <= 0:
        return np.zeros_like(mask_arr, dtype=mask_arr.dtype)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    out = np.zeros_like(mask_arr, dtype=mask_arr.dtype)
    if largest_label > 0 and counts[largest_label] > 0:
        out[labels == largest_label] = mask_arr[labels == largest_label]
    return out


def _sample_round_blob_mask(*, width: int, height: int, rng: np.random.Generator, min_area_frac: float, max_area_frac: float) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    n_blobs = int(rng.integers(1, 3))
    for _ in range(n_blobs):
        cx = int(rng.integers(width // 6, width - width // 6))
        cy = int(rng.integers(height // 6, height - height // 6))
        r = int(rng.integers(min(width, height) // 8, min(width, height) // 4))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    arr = np.asarray(canvas, dtype=np.uint8)
    frac = float((arr > 0).mean())
    return arr if min_area_frac <= frac <= max_area_frac else np.zeros_like(arr)


def _generate_mask_for_tile(
    *,
    source_rgb: np.ndarray,
    seed: int,
    white_threshold: int,
    tissue_min_overlap: float,
    max_attempts: int,
    min_area_frac: float,
    max_area_frac: float,
    feather_radius: float,
    mask_strength: float,
) -> tuple[Image.Image, Image.Image, int | None, int | None, int | None, int]:
    h, w = source_rgb.shape[:2]
    tissue = np.logical_or.reduce(
        [
            source_rgb[:, :, 0] < white_threshold,
            source_rgb[:, :, 1] < white_threshold,
            source_rgb[:, :, 2] < white_threshold,
        ]
    )

    rng = np.random.default_rng(int(seed))
    mask_arr = None
    chosen_cx = None
    chosen_cy = None
    chosen_r = None
    for _ in range(max(1, int(max_attempts))):
        cand = _sample_round_blob_mask(
            width=w,
            height=h,
            rng=rng,
            min_area_frac=float(min_area_frac),
            max_area_frac=float(max_area_frac),
        )
        if cand.max() == 0:
            continue
        cand = _keep_largest_connected_component(cand)
        if cand.max() == 0:
            continue
        frac = float((cand > 0).mean())
        if frac < min_area_frac or frac > max_area_frac:
            continue
        m = cand > 0
        if tissue_min_overlap > 0:
            overlap = float(np.logical_and(m, tissue).sum()) / float(max(m.sum(), 1))
            if overlap < tissue_min_overlap:
                continue

        ys, xs = np.where(m)
        if len(xs) > 0:
            chosen_cx = int(np.round(xs.mean()))
            chosen_cy = int(np.round(ys.mean()))
            chosen_r = int(np.round(np.sqrt(float(m.sum()) / np.pi)))
        mask_arr = cand
        break

    fallback_used = 0
    if mask_arr is None:
        fallback = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(fallback)
        rx = max(8, w // 6)
        ry = max(8, h // 6)
        cx, cy = w // 2, h // 2
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
        mask_arr = np.asarray(fallback, dtype=np.uint8)
        chosen_cx, chosen_cy = cx, cy
        chosen_r = int(np.round((rx + ry) * 0.5))
        fallback_used = 1

    mask_raw = Image.fromarray(mask_arr)
    mask_soft = mask_raw
    if feather_radius > 0:
        mask_soft = mask_soft.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))
    if abs(mask_strength - 1.0) > 1e-6:
        arr = np.asarray(mask_soft, dtype=np.float32) * float(mask_strength)
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        mask_soft = Image.fromarray(arr)

    return mask_raw, mask_soft, chosen_cx, chosen_cy, chosen_r, fallback_used


def _compose_unmasked(original: np.ndarray, generated: np.ndarray, mask_raw: np.ndarray) -> np.ndarray:
    keep = mask_raw <= 127
    out = generated.copy()
    out[keep] = original[keep]
    return out


def _write_summary(summary_path: Path, payload: dict) -> None:
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate LoRA inpainting tile samples for one strength bucket.")
    ap.add_argument("--tiles-dir", required=True, help="Directory containing source tile PNGs and optional .caption files")
    ap.add_argument("--base-model", required=True, help="SDXL inpainting base model path or single-file checkpoint")
    ap.add_argument("--lora-weights", required=True, help="LoRA weights path (.safetensors)")
    ap.add_argument("--output-dir", required=True, help="Root output directory; script creates str_<scale>x under it")
    ap.add_argument("--lora-scale", required=True, type=float, help="LoRA scale for this run, e.g. 3, 5, 7")
    ap.add_argument("--samples-per-tile", type=int, default=5, help="Random-mask samples per tile")
    ap.add_argument("--tile-offset", type=int, default=0, help="Start from this tile index after sorting")
    ap.add_argument("--max-tiles", type=int, default=0, help="Limit number of tiles processed (0 = all)")
    ap.add_argument("--max-tiles-from-source", type=int, default=0, help="Randomly sample up to N tiles per parent source image (0 = no per-source sampling)")
    ap.add_argument("--source-sample-seed", type=int, default=123, help="Seed for per-source tile sampling")
    ap.add_argument("--tile-stems-file", default=None, help="Optional text file with one tile stem per line")
    ap.add_argument("--seed-base", type=int, default=42, help="Base seed used for deterministic sampling")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance-scale", type=float, default=3.5)
    ap.add_argument("--negative-prompt", default="lowres,blurry,artifact,painting,illustration,cartoon,geometric pattern,abstract art")
    ap.add_argument("--prompt-fallback", default="H&E stained skin histopathology tile, non_cancer, realistic microscopy")
    ap.add_argument("--white-threshold", type=int, default=220)
    ap.add_argument("--tissue-min-overlap", type=float, default=0.75)
    ap.add_argument("--max-attempts", type=int, default=30)
    ap.add_argument("--min-area-frac", type=float, default=0.05)
    ap.add_argument("--max-area-frac", type=float, default=0.16)
    ap.add_argument("--mask-blur-px", type=float, default=0.0, help="Optional Gaussian blur radius for soft mask")
    ap.add_argument("--mask-strength", type=float, default=1.0)
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--skip-existing", action="store_true", help="Skip generations whose output file already exists")
    ap.add_argument("--composite-unmasked", action="store_true", help="Paste unmasked pixels back from the original tile")
    ap.add_argument("--dry-run", action="store_true", help="Print planned work without running generation")
    args = ap.parse_args()

    tiles_dir = Path(args.tiles_dir).resolve()
    if not tiles_dir.is_dir():
        raise SystemExit(f"Tiles directory not found: {tiles_dir}")

    lora_weights = Path(args.lora_weights).resolve()
    if not lora_weights.is_file():
        raise SystemExit(f"LoRA weights file not found: {lora_weights}")

    tile_stems_file = Path(args.tile_stems_file).resolve() if args.tile_stems_file else None
    if tile_stems_file is not None and not tile_stems_file.is_file():
        raise SystemExit(f"Tile stems file not found: {tile_stems_file}")

    device = _resolve_device(args.device)
    scale_dir = Path(args.output_dir).resolve() / _strength_folder_name(float(args.lora_scale))
    images_dir = scale_dir / "images"
    masks_dir = scale_dir / "masks"
    metadata_dir = scale_dir / "metadata"

    tile_paths = _iter_tiles(
        tiles_dir=tiles_dir,
        tile_stems_file=tile_stems_file,
        tile_offset=int(args.tile_offset),
        max_tiles=int(args.max_tiles),
    )
    if not tile_paths:
        raise SystemExit("No tiles were selected.")

    if int(args.max_tiles_from_source) > 0:
        tile_paths = _sample_tiles_per_source(
            tile_paths,
            max_per_source=int(args.max_tiles_from_source),
            seed=int(args.source_sample_seed),
        )
        if not tile_paths:
            raise SystemExit("No tiles remained after per-source sampling.")
        print(f"Per-source sample : up to {args.max_tiles_from_source} tiles per source image (seed={args.source_sample_seed})")

    print(f"Tiles selected    : {len(tile_paths)}")
    if int(args.max_tiles_from_source) > 0:
        sources_seen = {_source_image_from_stem(p.stem) for p in tile_paths}
        print(f"Source images     : {len(sources_seen)}")
    print(f"Samples per tile  : {args.samples_per_tile}")
    print(f"Total generations : {len(tile_paths) * int(args.samples_per_tile)}")
    print(f"Strength folder   : {scale_dir}")
    print(f"LoRA scale        : {args.lora_scale}")
    print(f"Checkpoint        : {lora_weights}")

    if args.dry_run:
        for tile_path in tile_paths[:10]:
            print(f"  tile: {tile_path.name}")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    pipe = _load_inpaint_pipeline(str(args.base_model), device=device, dtype=args.dtype)
    pipe.load_lora_weights(str(lora_weights.parent), weight_name=lora_weights.name)

    rows = []
    try:
        for tile_idx, tile_path in enumerate(tile_paths):
            source_image = Image.open(tile_path).convert("RGB")
            width, height = source_image.size
            prompt = _read_prompt(tile_path, args.prompt_fallback)
            source_np = np.array(source_image)

            for sample_idx in range(int(args.samples_per_tile)):
                seed = int(args.seed_base + tile_idx * 1000 + sample_idx)
                mask_raw_pil, mask_soft_pil, cx, cy, radius, fallback_used = _generate_mask_for_tile(
                    source_rgb=source_np,
                    seed=seed,
                    white_threshold=int(args.white_threshold),
                    tissue_min_overlap=float(args.tissue_min_overlap),
                    max_attempts=int(args.max_attempts),
                    min_area_frac=float(args.min_area_frac),
                    max_area_frac=float(args.max_area_frac),
                    feather_radius=float(args.mask_blur_px),
                    mask_strength=float(args.mask_strength),
                )

                stem = f"{tile_path.stem}__sample{sample_idx:02d}__seed{seed}"
                out_path = images_dir / f"{stem}.png"
                mask_path = masks_dir / f"{stem}__mask.png"

                if args.skip_existing and out_path.is_file() and mask_path.is_file():
                    rows.append(
                        {
                            "tile_stem": tile_path.stem,
                            "source_tile": str(tile_path),
                            "generated_image": str(out_path),
                            "mask_image": str(mask_path),
                            "prompt": prompt,
                            "seed": seed,
                            "sample_index": sample_idx,
                            "mask_center_x": cx,
                            "mask_center_y": cy,
                            "mask_radius_px": radius,
                            "fallback_used": fallback_used,
                            "lora_scale": float(args.lora_scale),
                            "checkpoint": str(lora_weights),
                            "reused_existing": True,
                        }
                    )
                    continue

                generator = torch.Generator(device=device).manual_seed(seed)
                result = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    image=source_image,
                    mask_image=mask_soft_pil,
                    width=width,
                    height=height,
                    num_inference_steps=int(args.steps),
                    guidance_scale=float(args.guidance_scale),
                    generator=generator,
                    cross_attention_kwargs={"scale": float(args.lora_scale)},
                )
                generated = np.array(result.images[0].convert("RGB"))

                if args.composite_unmasked:
                    generated = _compose_unmasked(source_np, generated, np.array(mask_raw_pil))

                Image.fromarray(generated).save(out_path)
                mask_raw_pil.save(mask_path)

                rows.append(
                    {
                        "tile_stem": tile_path.stem,
                        "source_tile": str(tile_path),
                        "generated_image": str(out_path),
                        "mask_image": str(mask_path),
                        "prompt": prompt,
                        "seed": seed,
                        "sample_index": sample_idx,
                        "mask_center_x": cx,
                        "mask_center_y": cy,
                        "mask_radius_px": radius,
                        "fallback_used": fallback_used,
                        "lora_scale": float(args.lora_scale),
                        "checkpoint": str(lora_weights),
                        "reused_existing": False,
                    }
                )

                print(
                    f"[{tile_idx + 1}/{len(tile_paths)}] {tile_path.stem} sample {sample_idx + 1}/{args.samples_per_tile}"
                )
    finally:
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()

    metadata_csv = metadata_dir / "generation_manifest.csv"
    metadata_json = metadata_dir / "generation_summary.json"
    with metadata_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tile_stem",
                "source_tile",
                "generated_image",
                "mask_image",
                "prompt",
                "seed",
                "sample_index",
                "mask_center_x",
                "mask_center_y",
                "mask_radius_px",
                "fallback_used",
                "lora_scale",
                "checkpoint",
                "reused_existing",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    _write_summary(
        metadata_json,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "tiles_dir": str(tiles_dir),
            "selected_tile_count": len(tile_paths),
            "samples_per_tile": int(args.samples_per_tile),
            "total_samples": len(rows),
            "base_model": str(args.base_model),
            "checkpoint": str(lora_weights),
            "lora_scale": float(args.lora_scale),
            "steps": int(args.steps),
            "guidance_scale": float(args.guidance_scale),
            "white_threshold": int(args.white_threshold),
            "tissue_min_overlap": float(args.tissue_min_overlap),
            "max_attempts": int(args.max_attempts),
            "min_area_frac": float(args.min_area_frac),
            "max_area_frac": float(args.max_area_frac),
            "mask_blur_px": float(args.mask_blur_px),
            "mask_strength": float(args.mask_strength),
            "composite_unmasked": bool(args.composite_unmasked),
            "max_tiles_from_source": int(args.max_tiles_from_source),
            "source_sample_seed": int(args.source_sample_seed),
            "device": device,
        },
    )

    print(f"Saved images    : {images_dir}")
    print(f"Saved masks     : {masks_dir}")
    print(f"Saved metadata  : {metadata_csv}")
    print(f"Saved summary   : {metadata_json}")


if __name__ == "__main__":
    main()
