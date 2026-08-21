# Semantic Inpainting v1

This branch replaces the Phase-1 coarse/random-mask objective with a class-aligned histology objective built from `data/artifacts/tiles_for_simulation`.

## Training objective

Each sample contains:

- one oriented 512x512 H&E tile;
- its aligned 12-class semantic label map;
- one target class from IDs 1-11;
- a geometry-aware round/elliptical mask that lies mostly on that target class;
- a natural-language caption such as `H&E stained skin histology, target tissue: papillary dermis, realistic microscopy`.

Background/air is not a target class.

The builder creates both high-purity interior masks and a smaller fraction of boundary-crossing masks. Thin structures use PCA-aligned ellipses rather than being rejected just because a large circle cannot fit.

Training remains same-class reconstruction because it has real ground truth. Cross-class insertion has no paired ground-truth image, so it is evaluated independently after training instead of being treated as supervised truth.

## 1. Build semantic dataset

```bash
python scripts/synthetic_data/build_semantic_inpaint_dataset.py \
  --tiles-root data/artifacts/tiles_for_simulation \
  --output-dir data/artifacts/semantic_inpaint_v1 \
  --overwrite
```

The builder automatically uses `segmentation_splits_v2.csv` when available, otherwise `segmentation_splits.csv`, otherwise derives a deterministic whole-slide train/val/test split. It aborts on slide leakage.

Important outputs:

```text
data/artifacts/semantic_inpaint_v1/
  train/images/*.png + *.caption
  train/masks/*.png
  val/...
  test/...
  semantic_inpaint_manifest.csv
  semantic_inpaint_stats.json
  train_class_adjacency.json
```

Rare classes get more distinct masks per eligible tile/class pair, capped by `--max-masks-per-pair`, instead of duplicating one random mask indefinitely.

## 2. Audit before training

```bash
python scripts/synthetic_data/summarize_semantic_inpaint_dataset.py \
  --manifest data/artifacts/semantic_inpaint_v1/semantic_inpaint_manifest.csv
```

Inspect `data/artifacts/semantic_inpaint_v1/audit/contact_sheets/` before spending GPU time. Each sheet shows source RGB, the sampled mask overlay, and the reference semantic map.

## 3. Train semantic LoRA

```bash
python scripts/synthetic_data/finetune_stable_diffusion_unified.py \
  --config configs/sdxl_lora_semantic_inpaint_v1.yaml
```

The v1 config trains an SDXL UNet LoRA from the base model with rank 16, masked loss, class-specific captions, and feathered masks. It deliberately does not continue from the old binary `cancer/non_cancer` Phase-1 checkpoint so the effect of the new objective remains interpretable.

## 4. Same-class and cross-class benchmark

After training, run the independent SAM2.1-Hiera semantic check:

```bash
python scripts/synthetic_data/benchmark_semantic_inpainting.py \
  --manifest data/artifacts/semantic_inpaint_v1/semantic_inpaint_manifest.csv \
  --split test \
  --mode both \
  --base-model /path/to/sd_xl_base_1.0.safetensors \
  --lora-weights outputs/finetunes/skin_histology_semantic_inpaint_v1/last.safetensors \
  --seg-checkpoint /path/to/best_segmentation_checkpoint.pt
```

For same-class reconstruction, the target class should be recovered inside the held-out mask. For cross-class insertion, the script changes the prompt while keeping the edit ROI, then measures:

- requested-class fraction inside the ROI;
- gain relative to the segmenter's prediction on the untouched source image;
- new requested-class leakage outside the edit context;
- semantic preservation outside the edit context.

Cross-class targets are chosen using a context compatibility score from the train-only class adjacency profile unless the candidate set is restricted with `--cross-target-ids`.

## Why cross-class examples are not used as supervised training data yet

A real tile can supervise `dermis -> reconstruct dermis` because the original pixels are the ground truth. There is no real paired image for `dermis -> SCC in exactly this artificial ROI`. Treating a generated cross-class edit as ground truth would create a self-reinforcing loop. The segmentation model is therefore an evaluator first. A reward-guided or curated cross-class training phase can be added only after the same-class model and independent benchmark are trustworthy.
