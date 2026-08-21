#!/usr/bin/env python3
"""Benchmark same-class reconstruction and cross-class semantic inpainting.

The SDXL LoRA performs the edit; an independently trained SAM2.1-Hiera semantic
segmenter measures whether the requested target class appears inside the edit
ROI while anatomy outside the ROI remains stable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageFilter

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
for p in (_THIS_DIR, _REPO_ROOT / "scripts" / "segmentation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from generate_lora_inpaint_tile_samples import _compose_unmasked, _load_inpaint_pipeline, _resolve_device  # noqa: E402
from semantic_mask_utils import CLASS_NAME_BY_ID, TARGET_CLASS_IDS, class_prompt  # noqa: E402
from histoseg_tile_dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from sam2_hiera_semantic import SAM2HieraSemanticSegmenter  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/artifacts/semantic_inpaint_v1/semantic_inpaint_manifest.csv")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--mode", default="both", choices=["same", "cross", "both"])
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora-weights", required=True)
    ap.add_argument("--seg-checkpoint", required=True)
    ap.add_argument("--adjacency-json", default="", help="Default: sibling train_class_adjacency.json")
    ap.add_argument("--output-dir", default="outputs/semantic_inpaint_benchmark")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance-scale", type=float, default=4.0)
    ap.add_argument("--mask-blur-px", type=float, default=6.0)
    ap.add_argument("--mask-strength", type=float, default=0.9)
    ap.add_argument("--context-dilate-px", type=int, default=12)
    ap.add_argument("--seeds-per-edit", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--max-samples-per-class", type=int, default=8)
    ap.add_argument(
        "--cross-target-ids",
        default="1,2,3,4,5,6,7,8,9,10,11",
        help="Comma-separated semantic target IDs considered for cross-class edits.",
    )
    ap.add_argument("--negative-prompt", default="lowres,blurry,artifact,painting,illustration,cartoon,geometric pattern,abstract art")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve(raw: str, parent: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (parent / p).resolve()


def _load_segmenter(checkpoint_path: Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    kwargs = dict(ckpt["model_kwargs"])
    model = SAM2HieraSemanticSegmenter(
        num_classes=int(kwargs.get("num_classes", 12)),
        sam2_config=str(kwargs["sam2_config"]),
        sam2_checkpoint=None,
        sam2_model_id=kwargs.get("sam2_model_id"),
        load_pretrained=False,
        decoder_channels=int(kwargs.get("decoder_channels", 192)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(device).eval()
    encoder_size = int(ckpt.get("encoder_input_size", model.encoder_input_size))
    return model, encoder_size


def _segment_image(model, encoder_size: int, image: Image.Image, device: str) -> np.ndarray:
    src_size = image.size
    resized = image.convert("RGB").resize((encoder_size, encoder_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda" if str(device).startswith("cuda") else "cpu",
            dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
            enabled=str(device).startswith("cuda"),
        ):
            logits = model(tensor, output_size=(src_size[1], src_size[0]))
    return torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)


def _soft_mask(raw: Image.Image, blur_px: float, strength: float) -> Image.Image:
    out = raw.convert("L")
    if blur_px > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    if abs(float(strength) - 1.0) > 1e-6:
        arr = np.asarray(out, dtype=np.float32) * float(strength)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    return out


def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return np.asarray(mask, dtype=bool)
    from scipy import ndimage as ndi

    return ndi.binary_dilation(np.asarray(mask, dtype=bool), iterations=int(pixels))


def _metrics(
    original_seg: np.ndarray,
    generated_seg: np.ndarray,
    mask: np.ndarray,
    target_class_id: int,
    context_dilate_px: int,
) -> Dict[str, float]:
    roi = np.asarray(mask, dtype=bool)
    context = _dilate(roi, int(context_dilate_px))
    outside = ~context
    target = int(target_class_id)

    orig_in = float((original_seg[roi] == target).mean()) if roi.any() else 0.0
    gen_in = float((generated_seg[roi] == target).mean()) if roi.any() else 0.0
    orig_out = float((original_seg[outside] == target).mean()) if outside.any() else 0.0
    gen_out = float((generated_seg[outside] == target).mean()) if outside.any() else 0.0
    preservation = float((generated_seg[outside] == original_seg[outside]).mean()) if outside.any() else 1.0
    target_gain = gen_in - orig_in
    leakage_gain = max(0.0, gen_out - orig_out)
    score = 0.60 * gen_in + 0.75 * target_gain + 0.35 * preservation - 0.80 * leakage_gain
    return {
        "target_fraction_inside": gen_in,
        "baseline_target_fraction_inside": orig_in,
        "target_gain_inside": target_gain,
        "target_fraction_outside": gen_out,
        "baseline_target_fraction_outside": orig_out,
        "target_leakage_gain_outside": leakage_gain,
        "outside_segmentation_preservation": preservation,
        "compliance_score": float(score),
    }


def _parse_target_ids(text: str) -> List[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        cid = int(part)
        if cid not in TARGET_CLASS_IDS:
            raise SystemExit(f"Invalid cross target class ID: {cid}")
        out.append(cid)
    if not out:
        raise SystemExit("No cross target IDs supplied")
    return sorted(set(out))


def _adjacency_profile(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    arr = np.asarray(payload["row_probabilities"], dtype=np.float64)
    if arr.shape != (12, 12):
        raise SystemExit(f"Unexpected adjacency profile shape: {arr.shape}")
    return arr


def _row_ring_distribution(row: Dict[str, str]) -> np.ndarray:
    hist = json.loads(row.get("neighbor_class_histogram", "{}") or "{}")
    v = np.zeros(12, dtype=np.float64)
    for key, count in hist.items():
        cid = int(key)
        if 0 <= cid < 12:
            v[cid] = float(count)
    if v.sum() > 0:
        v /= v.sum()
    return v


def _choose_cross_target(row: Dict[str, str], candidates: List[int], adjacency: np.ndarray) -> int | None:
    source = int(row["target_class_id"])
    ring = _row_ring_distribution(row)
    best = None
    best_score = -1.0
    for cid in candidates:
        if cid == source:
            continue
        score = float(np.dot(ring, adjacency[cid]))
        if score > best_score:
            best_score = score
            best = cid
    return best


def _select_rows(rows: List[Dict[str, str]], split: str, max_per_class: int) -> List[Dict[str, str]]:
    by_class = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            by_class[int(row["target_class_id"])].append(row)
    selected = []
    for cid in sorted(by_class):
        candidates = by_class[cid]
        if max_per_class > 0 and len(candidates) > max_per_class:
            idx = np.unique(np.linspace(0, len(candidates) - 1, num=max_per_class, dtype=int))
            candidates = [candidates[int(i)] for i in idx]
        selected.extend(candidates)
    return selected


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    rows = _read_csv(manifest_path)
    selected = _select_rows(rows, args.split, int(args.max_samples_per_class))
    if not selected:
        raise SystemExit(f"No {args.split} samples in {manifest_path}")

    adjacency_path = Path(args.adjacency_json).resolve() if args.adjacency_json else manifest_path.parent / "train_class_adjacency.json"
    adjacency = _adjacency_profile(adjacency_path)
    cross_targets = _parse_target_ids(args.cross_target_ids)
    modes = [args.mode] if args.mode != "both" else ["same", "cross"]

    planned = []
    for row in selected:
        for mode in modes:
            if mode == "same":
                target = int(row["target_class_id"])
            else:
                target = _choose_cross_target(row, cross_targets, adjacency)
                if target is None:
                    continue
            planned.append((row, mode, int(target)))

    print(f"Benchmark examples planned: {len(planned)}")
    if args.dry_run:
        for row, mode, target in planned[:30]:
            print(f"  {mode:5s} {row['sample_id']} -> {target}:{CLASS_NAME_BY_ID[target]}")
        return 0

    device = _resolve_device(args.device)
    lora_path = Path(args.lora_weights).resolve()
    seg_path = Path(args.seg_checkpoint).resolve()
    if not lora_path.is_file():
        raise SystemExit(f"LoRA checkpoint not found: {lora_path}")
    if not seg_path.is_file():
        raise SystemExit(f"Segmentation checkpoint not found: {seg_path}")

    pipe = _load_inpaint_pipeline(args.base_model, device=device, dtype=args.dtype)
    pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
    segmenter, encoder_size = _load_segmenter(seg_path, device)

    out_root = Path(args.output_dir).resolve() / args.split
    images_out = out_root / "images"
    seg_out = out_root / "segmentation"
    images_out.mkdir(parents=True, exist_ok=True)
    seg_out.mkdir(parents=True, exist_ok=True)

    baseline_cache: Dict[str, np.ndarray] = {}
    result_rows: List[Dict[str, object]] = []
    try:
        for edit_index, (row, mode, target) in enumerate(planned):
            image_path = _resolve(row["image_path"], manifest_path.parent)
            mask_path = _resolve(row["mask_path"], manifest_path.parent)
            source = Image.open(image_path).convert("RGB")
            source_np = np.asarray(source, dtype=np.uint8)
            raw_mask_pil = Image.open(mask_path).convert("L")
            raw_mask = np.asarray(raw_mask_pil) > 127
            soft_mask = _soft_mask(raw_mask_pil, args.mask_blur_px, args.mask_strength)

            cache_key = str(image_path)
            if cache_key not in baseline_cache:
                baseline_cache[cache_key] = _segment_image(segmenter, encoder_size, source, device)
            baseline_seg = baseline_cache[cache_key]

            best = None
            prompt = class_prompt(target)
            for seed_offset in range(max(1, int(args.seeds_per_edit))):
                seed = int(args.seed_base) + edit_index * 1009 + seed_offset
                generator = torch.Generator(device=device).manual_seed(seed)
                generated_pil = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    image=source,
                    mask_image=soft_mask,
                    width=source.width,
                    height=source.height,
                    num_inference_steps=int(args.steps),
                    guidance_scale=float(args.guidance_scale),
                    generator=generator,
                    cross_attention_kwargs={"scale": float(args.lora_scale)},
                ).images[0].convert("RGB")
                generated_np = _compose_unmasked(source_np, np.asarray(generated_pil), np.asarray(raw_mask_pil))
                generated_pil = Image.fromarray(generated_np)
                generated_seg = _segment_image(segmenter, encoder_size, generated_pil, device)
                metrics = _metrics(
                    baseline_seg,
                    generated_seg,
                    raw_mask,
                    target,
                    int(args.context_dilate_px),
                )
                candidate = {
                    "seed": seed,
                    "image": generated_pil,
                    "segmentation": generated_seg,
                    "metrics": metrics,
                }
                if best is None or metrics["compliance_score"] > best["metrics"]["compliance_score"]:
                    best = candidate

            assert best is not None
            stem = f"{row['sample_id']}__{mode}__to_c{target:02d}"
            image_out = images_out / f"{stem}.png"
            segmentation_out = seg_out / f"{stem}.npy"
            best["image"].save(image_out)
            np.save(segmentation_out, best["segmentation"])
            rec = {
                "sample_id": row["sample_id"],
                "tile_id": row["tile_id"],
                "slide_id": row["slide_id"],
                "mode": mode,
                "source_class_id": int(row["target_class_id"]),
                "source_class_name": CLASS_NAME_BY_ID[int(row["target_class_id"])],
                "target_class_id": target,
                "target_class_name": CLASS_NAME_BY_ID[target],
                "prompt": prompt,
                "seed": int(best["seed"]),
                "generated_image": str(image_out),
                "generated_segmentation": str(segmentation_out),
                **best["metrics"],
            }
            result_rows.append(rec)
            print(
                f"[{len(result_rows)}/{len(planned)}] {mode} {row['sample_id']} -> "
                f"{CLASS_NAME_BY_ID[target]} target={rec['target_fraction_inside']:.3f} "
                f"gain={rec['target_gain_inside']:+.3f} preserve={rec['outside_segmentation_preservation']:.3f}"
            )
    finally:
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        del pipe
        del segmenter
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    manifest_out = out_root / "benchmark_manifest.csv"
    with manifest_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    aggregates = {}
    for mode in ("same", "cross"):
        subset = [r for r in result_rows if r["mode"] == mode]
        if not subset:
            continue
        aggregates[mode] = {
            "n": len(subset),
            "target_fraction_inside_mean": float(np.mean([r["target_fraction_inside"] for r in subset])),
            "target_gain_inside_mean": float(np.mean([r["target_gain_inside"] for r in subset])),
            "outside_segmentation_preservation_mean": float(np.mean([r["outside_segmentation_preservation"] for r in subset])),
            "target_leakage_gain_outside_mean": float(np.mean([r["target_leakage_gain_outside"] for r in subset])),
            "compliance_score_mean": float(np.mean([r["compliance_score"] for r in subset])),
        }
    summary_out = out_root / "benchmark_summary.json"
    summary_out.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "lora_weights": str(lora_path),
                "segmentation_checkpoint": str(seg_path),
                "adjacency_profile": str(adjacency_path),
                "split": args.split,
                "settings": vars(args),
                "aggregates": aggregates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Benchmark manifest: {manifest_out}")
    print(f"Benchmark summary : {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
