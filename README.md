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

## Phase Behavior Notes (from oral-lesions comparison)

- `Phase 1` (foundation): learns the base inpainting behavior and histology texture adaptation under masked loss.
- `Phase 2` (reward-guided): iterative train/select loop; usually improves realism and class-conditioned edits without over-amplifying style.
- `Phase 3` (morph reward): strongest reward/curriculum pressure; can overshoot and make LoRA edits look exaggerated if not tuned conservatively.

Observed differences versus the oral-lesions notebook overrides (`Lora_&_Finetune_SDXL.ipynb`):
- Oral Grad-CAM run used longer Phase 1 (`max_train_epochs=50`) with ROI masks and inpaint sample controls (`sample_denoising_strength=0.55`, `sample_mask_strength=0.8`, `sample_mask_blur_radius=6.0`).
- In this skin tile workflow, short Phase 1 runs can converge to color-fill behavior inside masks before learning tissue microtexture.
- Oral notebook experiments were strongest in Phase 2; Phase 3 often made the adapter too strong/exaggerated, matching behavior seen here.

Recommended tuning order for skin histology:
- Increase Phase 1 duration first (e.g., 20-50 epochs) before adding stronger reward pressure.
- Prefer smaller/local masks for early runs to force tissue-structure learning instead of broad repainting.
- Keep Phase 2 as the primary optimization stage; use Phase 3 sparingly or with reduced pressure when outputs become over-strong.
- For inpaint sample previews, use a real image/mask pair representative of current mask style (already supported via `sample_init_image` + `sample_mask_image`).

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
  --white-threshold 235 --tissue-min-overlap 0.75 \
  --max-attempts 20 --min-area-frac 0.08 --max-area-frac 0.22 \
  --min-strokes 1 --max-strokes 4 --min-vertices 3 --max-vertices 8 \
  --min-brush-px 24 --max-brush-px 128 \
  --feather-radius 2.0 --mask-strength 1.0 --overwrite \
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

## 3D Coherence Track (MVP)

An extension for Z-coherent volume inpainting from ordered histology slice stacks.

**Goal:** Inpaint all slices in a 3D volume with a consistent (cylindrical) mask
region, then measure Z-coherence with lightweight metrics.

**Constraint:** No large dataset downloads (e.g. MATRICS-A at hundreds of GB).  A
small **Zenodo 8155124** volume benchmark (~700 MB: `cropped_slices.zip` ~635 MB +
`3d_model_10pct.nii` ~64 MB) is optional and does not violate this policy.
MVP uses circular test masks and the existing Histo-Seg dataset by default.

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/3d/propagate_mask_across_slices.py` | Replicate a 2D mask through Z, or generate a centered circular test mask. |
| `scripts/3d/stack_slices_to_nifti.py` | Stack ordered 2D images into a NIfTI volume (`.nii.gz`). |
| `scripts/3d/compute_z_coherence_metrics.py` | Adjacent-slice SSIM + Z-gradient smoothness. |

### Quick test (no data download required)

```bash
# Generate 10 circular masks
python scripts/3d/propagate_mask_across_slices.py \
  --num-slices 10 --height 256 --width 256 --radius 0.3 \
  --output-dir data/artifacts/3d/test_masks \
  --stats-json data/artifacts/3d/test_masks_stats.json

# Stack into NIfTI
python scripts/3d/stack_slices_to_nifti.py \
  --input-glob "data/artifacts/3d/test_masks/mask_slice_*.png" \
  --output-nifti data/artifacts/3d/test_volume.nii.gz \
  --pixdim 1.0 1.0 2.0 \
  --stats-json data/artifacts/3d/test_volume_stats.json

# Compute coherence metrics
python scripts/3d/compute_z_coherence_metrics.py \
  --volume-nifti data/artifacts/3d/test_volume.nii.gz \
  --output-json data/artifacts/3d/test_coherence.json
```

Expected output for identical masks: SSIM = 1.0, Z-gradient = 0.0.

### DVC stages (optional, self-contained)

```bash
dvc repro generate_3d_test_masks
dvc repro stack_3d_test_volume
dvc repro compute_3d_test_coherence
```

---

## Phase B — Volume Inpainting Pipeline (Slice-Stacks → 3D Volume)

Extends the MVP to wire cylindrical masks into the slice-wise inpainting flow,
with automatic metadata building, NIfTI stacking, and Z-coherence scoring.

### Phase B.1 — Tile-First Volume Inpainting (high-res slices)

**Why tile-first?**  Full high-res histology slices (e.g., 5000×5000 px) cannot
be naively resized to 512×512 for inpainting — tissue architecture is lost at
that scale.  Instead, the pipeline:

1. Generates cylindrical masks at **full-slice resolution**.
2. Extracts a local **ROI patch** around the mask (at a configurable
   `--patch-size`, default 512 px in strict mode).
3. Inpaints the patch at **model resolution** (`--target-size`, default 512).
4. Merges the inpainted patch back into the original full-res slice.
5. Stacks edited slices into a NIfTI volume and measures Z-coherence.

This means **tissue outside the mask is never touched**, and the model only
needs to work at 512×512.

#### Strict BBox Mode (Recommended)

The `--strict-bbox` flag enables **no-padding, no-square** mode:

- Raw mask bbox width AND height must be ≤ `--patch-size`.
- No padding expansion around the bbox.
- No square transform.
- Extraction centers a `patch_size × patch_size` tile on the raw bbox center
  and clamps to image boundaries (no zero-padding).
- `--padding-ratio` is ignored (forced to 0).
- `--patch-size` defaults to **512** when strict mode is active without an
  explicit value.

**Use strict mode when your masks are small enough that raw bbox ≤ 512×512.**
This produces clean patches with no synthetic padding, ideal for downstream
inpainting quality.

#### Key Parameters for Tile Geometry

| Flag | Default | Purpose |
|------|---------|---------|
| `--strict-bbox` | (flag) | Enable strict raw-bbox mode (no padding, no square). |
| `--patch-size` | 512 (strict) / 1024 (legacy) | Full-resolution patch crop size. In strict mode: both raw bbox dimensions must be ≤ this. |
| `--target-size` | 512 | Model working resolution for inpainting. The patch is resized to this before model input. |
| `--padding-ratio` | 0.0 (strict) / 0.15 (legacy) | Context margin — ignored in strict mode. |
| `--feather-radius` | 16 | Blending radius when pasting back into full-res slice (px). |

**Strict rule:** `raw_bbox_w ≤ patch_size AND raw_bbox_h ≤ patch_size`

**Legacy rule (without `--strict-bbox`):** the mask diameter should be ≤ 0.6 ×
`--patch-size` to leave room for padding and square transform.

The orchestrator validates mask containment before extraction and errors
out with a helpful message if the rule is violated.

**Dry-run metric caveat:** During `--dry-run`, Step 6 (NIfTI stacking) falls
back to masks when no merged/inpainted slices exist.  Coherence metrics
computed in this mode reflect *mask* consistency (SSIM = 1.0 for identical
cylindrical masks), not actual inpaint quality.  The manifest field
`steps.stacking.mask_fallback: true` (present from v2.1.0+) records when
this fallback occurred so downstream consumers can flag dry-run metrics
appropriately.

#### Tissue-Aware Mask Placement (`--tissue-aware-mask`)

The optional `--tissue-aware-mask` flag enables a smarter mask centering
strategy. Instead of placing the cylindrical mask at the geometric center
of the slice, it selects a center inside **tissue-valid pixels common to
all sequential slices**, avoiding white/black voids.  This is useful when
the tissue does not fill the entire field of view (e.g., small biopsy
samples on a dark background).

Key tissue-aware parameters:
- `--white-threshold` (default 235): pixels ≥ this are white void.
- `--black-threshold` (default 10): pixels ≤ this are black void.
- `--min-mask-tissue-overlap` (default 0.85): minimum fraction of mask
  pixels that must be tissue in each slice.
- `--max-mask-black-frac` (default 0.10): maximum black-void fraction
  allowed inside the mask.
- `--mask-center-seed` (default 42): random seed for center sampling.
- `--mask-center-max-attempts` (default 50): retries to find a valid center.

When enabled, the pipeline saves `tissue_aware_mask_diagnostics.json` and
`.csv` in the run directory with per-slice overlap statistics.  See
`docs/MVP_3D_INPAINTING_PLAN.md` for full details.

### Quick Smoke Test (no model required)

```bash
# Step 1: Generate 10 synthetic gradient slices (256×256)
python scripts/3d/generate_synthetic_slices.py \
  --num-slices 10 --height 256 --width 256 --pattern gradient \
  --output-dir /tmp/test_tile_slices

# Step 2: Run tile-first pipeline in dry-run mode (strict bbox)
python scripts/3d/run_volume_inpaint_pipeline.py \
  --slice-glob "/tmp/test_tile_slices/slice_*.png" \
  --volume-id strict_smoke \
  --coarse-label non_cancer \
  --num-slices 10 \
  --mask-radius 0.15 \
  --strict-bbox \
  --patch-size 128 \
  --target-size 64 \
  --output-dir /tmp/test_strict_pipeline \
  --dry-run
```

This exercises mask generation, pairs CSV building (index-token matching),
extraction metadata (strict_fixed_raw crop mode), inpainting dry-run, merge
dry-run, NIfTI stacking (masks fallback, recorded in manifest), and
coherence metrics — all without requiring a real model.

For non-strict mode with padding + square (legacy):
```bash
python scripts/3d/run_volume_inpaint_pipeline.py \
  --slice-glob "/tmp/test_tile_slices/slice_*.png" \
  --volume-id legacy_smoke \
  --coarse-label non_cancer \
  --num-slices 10 \
  --mask-radius 0.3 \
  --patch-size 256 \
  --target-size 128 \
  --padding-ratio 0.1 \
  --output-dir /tmp/test_legacy_pipeline \
  --dry-run
```

### Full Inpainting Run (requires model paths)

Strict mode (recommended for small masks):
```bash
python scripts/3d/run_volume_inpaint_pipeline.py \
  --slice-glob "data/artifacts/3d/synthetic_slices/slice_*.png" \
  --volume-id my_volume \
  --coarse-label non_cancer \
  --num-slices 10 \
  --mask-radius 0.15 \
  --strict-bbox \
  --patch-size 512 \
  --target-size 512 \
  --output-dir data/artifacts/3d/volume_inpaint_runs \
  --base-model /path/to/sd_xl_base_1.0.safetensors \
  --lora-weights /path/to/lora.safetensors
```

Legacy mode (padding + square):
```bash
python scripts/3d/run_volume_inpaint_pipeline.py \
  --slice-glob "data/artifacts/3d/synthetic_slices/slice_*.png" \
  --volume-id my_volume \
  --coarse-label non_cancer \
  --num-slices 10 \
  --mask-radius 0.3 \
  --patch-size 1024 \
  --target-size 512 \
  --padding-ratio 0.15 \
  --output-dir data/artifacts/3d/volume_inpaint_runs \
  --base-model /path/to/sd_xl_base_1.0.safetensors \
  --lora-weights /path/to/lora.safetensors
```

### Pipeline Steps (8-step tile-first flow)

| Step | Action | Script | Always in dry-run |
|------|--------|--------|-------------------|
| 1 | Generate full-res cylindrical masks | `propagate_mask_across_slices.py` | ✅ Yes |
| 2 | Build pairs CSV (slice↔mask) | inline | ✅ Yes |
| 3 | Extract ROI patches (`strict_fixed_raw` or `fixed`) | `extract_roi_patches.py` | Metadata only |
| 4 | Inpaint patches at model resolution | `inpaint_roi_patches.py` | Plan only |
| 5 | Merge patches back into full-res slices | `merge_inpainted_patches.py` | Plan only |
| 6 | Stack merged slices to NIfTI | `stack_slices_to_nifti.py` | ✅ Yes (masks fallback) |
| 7 | Compute Z-coherence metrics | `compute_z_coherence_metrics.py` | ✅ Yes |
| 8 | Save run manifest JSON | inline | ✅ Yes |

The manifest records `strict_bbox_mode: true/false` and `containment_rule`
detailing which validation rule was applied.

### Dry-Run Behavior

- Heavier steps (3–5) produce metadata/plan but skip actual I/O.
- Steps 6 and 7 fall back to masks as a proxy volume so the full pipeline
  can be exercised end-to-end.
- The manifest clearly marks each step's status.

### DVC Stages (optional, self-contained)

```bash
# Generate synthetic slices once
dvc repro generate_3d_synthetic_slices

# Run the full volume inpainting pipeline in dry-run mode
dvc repro run_3d_volume_inpaint_smoke

# Run tile-first pipeline smoke test (Phase B.1)
dvc repro run_3d_volume_inpaint_tile_smoke
```

### Metadata CSV Format

The pairs CSV (Step 2) maps slice files to mask files with columns:
`slice_id, filename, mask_filename, coarse_label, group_code`.
The extraction step (Step 3) produces `patches_metadata.csv` with bbox
coordinates, which feeds into both `inpaint_roi_patches.py` and
`merge_inpainted_patches.py`.

### Run Manifest

The orchestrator saves a `run_manifest.json` in the run output directory
with all step statuses, paths, and timestamps for provenance.

### Future: YOLO-based detector

To drive mask generation from histological features rather than Grad-CAM:

1. Train a YOLOv8 detector on 512×512 tiles for veins/tumors/inflammation.
2. Run per-slice inference → instance masks → propagation through Z.
3. Use detector masks as the SDXL LoRA inpainting condition.

This is not yet implemented (see `docs/MVP_3D_INPAINTING_PLAN.md` for details).

## Notes

- This project is standalone and no longer depends on the oral-lesions repository layout.
- Shared model paths are configured once in `params.yaml` and propagated to runtime configs.
- Core synthetic training scripts are vendored under `scripts/synthetic_data/`.
- Minimal runtime modules are vendored under `src/` (`exp`, `utils`, classifier model factory).
- LoRA training still depends on `kohya_ss/sd-scripts` (bootstrapped with `scripts/setup_kohya_sd_scripts.sh`).
- Large artifacts are marked with `cache: false` in DVC and ignored in Git by default.
