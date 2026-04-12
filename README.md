# Skin Histology SDXL Inpainting Collab

This folder is a shareable mini-repo scaffold based on the `Lora_&_Finetune_SDXL.ipynb`
workflow, adapted to skin histology slices (Histo-Seg, Mendeley `vccj8mp2cg`, v2).

It includes:
- a dataset downloader for Mendeley public API,
- a CSV builder that pairs histology `.jpg` images with `.png` segmentation masks,
- vendored synthetic pipeline scripts (`build_roi_masks_gradcam`, `finetune_stable_diffusion_unified`, phase2/phase3 loops),
- SDXL LoRA Phase 1/2/3 config templates for Grad-CAM informed inpainting,
- a DVC pipeline for reproducible data prep and training orchestration.
- an optional tile-index workflow (coordinate CSV only) plus pre-generated random inpaint masks.

## Folder Layout

- `dvc.yaml`: reproducible pipeline stages.
- `params.yaml`: editable knobs and paths.
- `scripts/download_mendeley_histoseg.py`: manifest + downloader.
- `scripts/build_histoseg_pairs_csv.py`: builds Grad-CAM/training CSV.
- `scripts/render_runtime_configs.py`: injects shared model paths from `params.yaml` into runtime phase configs.
- `scripts/run_gradcam_from_params.py`: launches Grad-CAM mask generation from `params.yaml`.
- `scripts/patches/build_tile_index_from_masks.py`: mines 512x512 tile coordinates from mask coverage.
- `scripts/patches/materialize_tile_dataset.py`: materializes selected tiles once for reuse.
- `scripts/patches/generate_random_tile_masks.py`: builds tile-aligned random masks with tissue-overlap constraints.
- `scripts/synthetic_data/`: local copies of the training/selection/benchmark scripts.
- `scripts/setup_kohya_sd_scripts.sh`: helper to clone `kohya-ss/sd-scripts` locally.
- `configs/`: phase config templates adapted for skin histology.
- `notebooks/Skin_Histology_SDXL_Pipeline.ipynb`: lightweight run notebook.

## Quick Start

1) Enter the project folder:

```bash
cd /home/fertroll10/Documents/ML/skin_histology_sdxl_collab
```

If `dvc` is not in `PATH`, use your env binary, for example:

```bash
/home/fertroll10/anaconda3/envs/LocalGPT_llama2/bin/dvc repro fetch_histo_seg_manifest
```

Also make sure `python` points to an environment with this repo requirements.

2) Create manifest + download dataset (stored under `data/raw`, gitignored):

```bash
dvc repro fetch_histo_seg_manifest
dvc repro download_histo_seg_dataset
```

3) Build paired image/mask CSV:

```bash
dvc repro build_histoseg_pairs_csv
```

Optional: build a dense tile index (coordinates only, no image duplication):

```bash
dvc repro build_histoseg_tile_index
```

Current defaults for tile mining (`params.yaml -> tile_mining`):
- `tile_size=512`, `stride=64`
- tissue mask coverage filter: `0.15 <= tile_mask_coverage <= 0.95`
- white-content cap: `tile_white_frac <= 0.30` (`white_threshold=235`)
- per-source cap: `max_tiles_per_image=250`

Important: per-source cap is applied before white filtering, so discarded white-heavy
tiles still count toward that source cap (intentional, to avoid overfilling from a few slides).

Materialize tiles once (recommended before repeated training runs):

```bash
dvc repro materialize_histoseg_tile_dataset
```

This writes `data/artifacts/tiles/materialized_512` and the tile-randommask
configs are set to reuse that directory without rematerializing.

Generate tile-aligned random masks once (recommended):

```bash
dvc repro generate_histoseg_tile_random_masks
```

This writes `data/artifacts/tiles/masks_random_512` and all tile phase configs
are set to reuse that directory.

Default tile-mining settings are in `params.yaml -> tile_mining`.

Current default for `coarse_label` is `filename_group` mode in
`scripts/build_histoseg_pairs_csv.py` (`A -> non_cancer`, `B/C/D -> cancer`).
Use `--coarse-label-mode mask_classes` only if mask-class mapping is validated for your setup.

4) Fill shared model paths in `params.yaml` (single source of truth):
- `models.sdxl_base_model`
- `models.classifier_checkpoint`
- `models.kohya_scripts_dir` (already set to your existing local `sd-scripts` path)

5) Render runtime configs from templates:

```bash
dvc repro render_runtime_configs
```

6) Ensure Kohya scripts are present (required for LoRA training):

```bash
bash scripts/setup_kohya_sd_scripts.sh
```

7) Run Grad-CAM + Phase 1/2/3:

```bash
dvc repro build_gradcam_masks_for_skin
dvc repro train_skin_lora_phase1
dvc repro train_skin_lora_phase2
dvc repro train_skin_lora_phase3
```

## Tile + Random-Mask Training (Optional)

- Phase 1 tile config: `configs/sdxl_lora_phase1_skin_histology_tiles_randommask.yaml`
- Phase 2 tile config: `configs/sdxl_lora_phase2_reward_skin_histology_tiles_randommask.yaml`
- Phase 3 tile config: `configs/sdxl_lora_phase3_morph_reward_skin_histology_tiles_randommask.yaml`
- This mode uses materialized tiles at `data/artifacts/tiles/materialized_512`.
- Masks are pre-generated once and reused from `data/artifacts/tiles/masks_random_512` (`lora_mask_mode: directory`).

See detailed prompt/sampling/training review in `PROMPTS_AND_TRAINING_REVIEW.md`
and the phase tile configs above.

## Exact Repro Commands (Current Tile Run)

Use this exact sequence to reproduce the current training setup:

```bash
conda activate LocalGPT_llama2

# 1) Build full-slide pairs (if not already built)
python scripts/build_histoseg_pairs_csv.py --dataset-dir data/raw/histo_seg_v2 --output-csv data/processed/histoseg_pairs.csv --stats-json data/processed/histoseg_pairs_stats.json

# 2) Build filtered tile index (white cap + per-image cap)
python scripts/patches/build_tile_index_from_masks.py \
  --pairs-csv data/processed/histoseg_pairs.csv \
  --output-csv data/processed/histoseg_tile_pairs.csv \
  --stats-json data/processed/histoseg_tile_pairs_stats.json \
  --tile-size 512 --stride 64 \
  --min-mask-frac 0.15 --max-mask-frac 0.95 \
  --max-white-frac 0.30 --white-threshold 235 \
  --max-tiles-per-image 250 --max-total-tiles 0 \
  --selection-mode random --seed 222 --workers 20

# 3) Materialize filtered tiles once
python scripts/patches/materialize_tile_dataset.py \
  --labels-csv data/processed/histoseg_tile_pairs.csv \
  --output-dir data/artifacts/tiles/materialized_512 \
  --workers 20 --clean \
  --stats-json data/artifacts/tiles/materialized_512_stats.json

# 4) Generate tile-aligned random masks once
python scripts/patches/generate_random_tile_masks.py \
  --image-dir data/artifacts/tiles/materialized_512 \
  --output-dir data/artifacts/tiles/masks_random_512 \
  --workers 20 --seed 222 \
  --white-threshold 235 --tissue-min-overlap 0.60 \
  --max-attempts 20 --min-area-frac 0.12 --max-area-frac 0.40 \
  --min-strokes 1 --max-strokes 4 --min-vertices 3 --max-vertices 8 \
  --min-brush-px 24 --max-brush-px 128 \
  --feather-radius 0.0 --mask-strength 1.0 --overwrite \
  --stats-json data/artifacts/tiles/masks_random_512_stats.json

# 5) Phase 1
python -u scripts/synthetic_data/finetune_stable_diffusion_unified.py \
  --config configs/sdxl_lora_phase1_skin_histology_tiles_randommask.yaml

# 6) Phase 2
python -u scripts/synthetic_data/phase2_reward_guided_lora.py \
  --config configs/sdxl_lora_phase2_reward_skin_histology_tiles_randommask.yaml \
  --max-cycles 5

# 7) Phase 3
python -u scripts/synthetic_data/phase3_morph_reward_guided_lora.py \
  --config configs/sdxl_lora_phase3_morph_reward_skin_histology_tiles_randommask.yaml \
  --max-cycles 5
```

Monitor training progress:

```bash
tail -f outputs/logs/skin_histology_phase1_tiles_randommask/live_phase1.log
nvidia-smi
```

For long stages, you can run only one stage with `--single-item`.

## Notes

- This project is standalone and no longer depends on the oral-lesions repository layout.
- Shared model paths are configured once in `params.yaml` and propagated to runtime configs.
- Core synthetic training scripts are vendored under `scripts/synthetic_data/`.
- Minimal runtime modules are vendored under `src/` (`exp`, `utils`, classifier model factory).
- LoRA training still depends on `kohya_ss/sd-scripts` (bootstrapped with `scripts/setup_kohya_sd_scripts.sh`).
- Large artifacts are marked with `cache: false` in DVC and ignored in Git by default.
