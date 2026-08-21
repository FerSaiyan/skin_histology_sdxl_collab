# Skin Histology Simulation + SDXL Inpainting

Multi-physics skin simulation pipeline (optical GA, MCX light transport, Pennes bioheat) + SDXL LoRA inpainting for synthetic histology variation.

Based on the Histo-Seg dataset (Mendeley `vccj8mp2cg`, v2).

---

## Quick Start — Murilo's Machine

### 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download Histo-Seg dataset

```bash
python scripts/download_mendeley_histoseg.py --dataset-id vccj8mp2cg --expected-version 2 --manifest-out data/metadata/histo_seg_manifest.json
python scripts/download_mendeley_histoseg.py --manifest-in data/metadata/histo_seg_manifest.json --download --download-dir data/raw/histo_seg_v2 --skip-existing
```

### 3) Build paired CSV

```bash
python scripts/build_histoseg_pairs_csv.py --dataset-dir data/raw/histo_seg_v2 --output-csv data/processed/histoseg_pairs.csv --stats-json data/processed/histoseg_pairs_stats.json
```

### 4) Build oriented simulation tiles

```bash
python scripts/optical_ga/build_tiles_for_simulation.py \
  --mode simulation \
  --pairs-csv data/processed/histoseg_pairs.csv \
  --output-dir data/artifacts/tiles_for_simulation \
  --manifest-csv data/artifacts/tiles_for_simulation/tiles_manifest.csv \
  --tile-size 512 --stride 256
```

Outputs placed in `data/artifacts/tiles_for_simulation/`:
- `rgb/*.png` — oriented RGB tiles
- `label_id/*.npy` — semantic class-ID tiles
- `tiles_manifest.csv`, `tiles_stats.json` — global index

The manifest contains the orientation and class metadata. Derived
`label_vis/*.png`, `epidermis_mask/*.png`, and per-tile JSON are optional; add
`--write-label-vis`, `--write-epidermis-mask`, or `--write-tile-meta` only when
standalone QC files are needed. The QC script derives missing visualizations
and epidermis masks directly from `label_id`.

### 4b) Build dense segmentation tiles

Segmentation mode keeps every 512×512 window with at least 25% labeled tissue.
It does not require epidermis, a surface interface, or normal estimation. The
default 128-pixel stride provides dense coverage. Per-slide HDF5 shards avoid
creating hundreds of thousands of small files; use fast local scratch for
construction when the repository is on a hard disk:

```bash
python scripts/optical_ga/build_tiles_for_simulation.py \
  --mode segmentation --storage hdf5 \
  --staging-dir /tmp/histoseg-tile-staging \
  --pairs-csv data/processed/histoseg_pairs.csv \
  --output-dir data/artifacts/tiles_for_segmentation_hdf5 \
  --tile-size 512 --stride 128 --min-tissue-frac 0.25 \
  --workers 4
```

The segmentation loader reads `shards/*.h5` directly. To preserve an existing
slide split while expanding its tiles:

```bash
python scripts/segmentation/build_segmentation_splits.py \
  --tiles-root data/artifacts/tiles_for_segmentation_hdf5 \
  --reuse-assignment-csv data/artifacts/tiles_for_simulation/segmentation_splits_v2.csv \
  --output-csv data/artifacts/tiles_for_segmentation_hdf5/segmentation_splits_v2_dense.csv \
  --stats-json data/artifacts/tiles_for_segmentation_hdf5/segmentation_splits_v2_dense_stats.json
```

### 5) Optical GA smoke test (fast, surrogate, no MCX)

```bash
python scripts/optical_ga/ga_optimiser_optical.py \
  --generations 3 --population-size 8 \
  --target-L 60 --target-a 10 --target-b 15 \
  --forward-mode surrogate --fitness-mode lab \
  --output-dir outputs/optical_ga/smoke_test --seed 42
```

Expected: `outputs/optical_ga/smoke_test/best_genome.json` + `ga_history.csv`.

### 6) Multi-physics MVP smoke test

```bash
python scripts/simulation/generate_mvp_smoke_volume.py --output data/artifacts/mvp/smoke_label_volume.npy --shape 8 8 8
python scripts/simulation/run_mvp_multiphysics_from_config.py --config configs/mvp_multiphysics_example.yaml --label-volume data/artifacts/mvp/smoke_label_volume.npy --output-dir data/artifacts/mvp/smoke_run
```

### 7) 3D MCX + GA dataset workflow from `tiles_for_simulation`

```bash
# A) Materialize 3D volumes from 2D class tiles
python scripts/simulation/materialize_3d_mcx_volumes_from_tiles.py \
  --tiles-root data/artifacts/tiles_for_simulation \
  --output-dir data/artifacts/3d/mcx_from_tiles \
  --depth 16 --max-volumes 50

# B) Run GA-in-loop + MCX over generated volumes
python scripts/simulation/run_3d_mcx_ga_dataset.py \
  --volumes-manifest data/artifacts/3d/mcx_from_tiles/manifest.json \
  --output-dir data/artifacts/3d/mcx_ga_dataset_run \
  --num-volumes 20 \
  --target-L 60 --target-a 10 --target-b 15 \
  --ga-generations 5 --ga-population-size 12 \
  --ga-forward-mode surrogate \
  --mcx-run --mcx-binary mcx \
  --render-absorption-video --render-reflectance-spectrum
```

Notes:
- `--max-volumes 0` (materializer) and `--num-volumes 0` (dataset runner) mean full dataset.
- GA is class-aware per volume: only present labels contribute priors for parameter bounds.
- MCX build defaults to top-down source from air side (`z=0`, `dir=+z`) and supports `--enforce-air-top` / `--auto-flip-z-to-air-top`.
- Use `python scripts/simulation/view_volume_3d_interactive.py --volume <volume.npy>` to inspect prepared volumes.

---

## External Dependencies for Simulations

### MCX binary (required for `forward-mode realistic`)

```bash
curl -L https://github.com/fangq/mcx/releases/download/v2025.10/mcx-linux-x64 -o ~/.local/bin/mcx
chmod +x ~/.local/bin/mcx
mcx --version
```

Smoke tests use `forward-mode surrogate` and do **not** require MCX.

### PyXOpto (optional — MCX-vs-PyXOpto branch comparison)

```bash
pip install PyXOpto
```

---

## Workflow Reference

### Optical GA scripts (`scripts/optical_ga/`)

| Script | Purpose |
|--------|---------|
| `build_tiles_for_simulation.py` | Fast simulation-oriented or dense segmentation tiles from Histo-Seg slices |
| `estimate_epidermis_normal.py` | PCA-based epidermis normal estimation |
| `select_orient_tile_for_incidence.py` | Align tile to epidermis incidence angle |
| `ga_optimiser_optical.py` | GA for 19-parameter skin genome estimation |
| `genome_encoding_optical.py` | Genome encode/decode |
| `forward_model.py` | Surrogate + realistic forward models |
| `colorimetry.py` | LAB/XYZ colorimetry forward calculations |
| `optical_fitness.py` | GA fitness functions |
| `optical_report.py` | GA run reporting |
| `run_optical_ga_from_labels.py` | Full GA pipeline from Histo-Seg label tiles |
| `run_optical_ga_batch_compare.py` | Multi-seed batch compare |
| `render_best_run_video.py` | GA evolution video |
| `qc_tiles_for_simulation.py` | QC plots for simulation tiles |
| `label_to_optical_priors.py` | Per-class optical property priors |
| `mc_wrapper.py` | MCX subprocess wrapper |

### Multi-physics scripts (`scripts/simulation/`)

| Script | Purpose |
|--------|---------|
| `mcx_build_volume.py` | Build MCX volume from label array |
| `mcx_batch_runner.py` | Batch MCX execution |
| `mcx_extract_fluence.py` | Extract fluence from MCX output |
| `materialize_3d_mcx_volumes_from_tiles.py` | Extrude class-ID tiles into 3D MCX label volumes |
| `run_3d_mcx_ga_dataset.py` | Dataset-scale GA-in-loop + MCX runs |
| `view_volume_3d_interactive.py` | Interactive 3-view slice viewer for 3D volumes |
| `thermal_build_model.py` | Pennes bioheat model builder |
| `thermal_solve.py` | Bioheat equation solver |
| `thermal_visualise.py` | Thermal simulation visualization |
| `run_mvp_multiphysics_from_config.py` | Config-driven pipeline runner |
| `run_mvp_multiphysics_pipeline.py` | Pipeline orchestration |
| `colorimetry_metrics.py` | Tissue colorimetry metrics |
| `generate_mvp_smoke_volume.py` | Generate test label volumes |

---

## SDXL LoRA Inpainting (Akio's PC Only)

The following workflows run only on Akio's PC (not on Murilo's machine). They require:
- SDXL base model checkpoint (`.safetensors`)
- Kohya `sd-scripts` cloned at `params.yaml → models.kohya_scripts_dir`
- GPU with 16+ GB VRAM

### Setup

```bash
# Fill model paths
# Edit params.yaml → models.sdxl_base_model, models.kohya_scripts_dir

# Clone Kohya scripts
bash scripts/setup_kohya_sd_scripts.sh

# Render runtime configs from templates
python scripts/render_runtime_configs.py --params params.yaml
```

### Phase 1/2/3 training

```bash
python scripts/synthetic_data/finetune_stable_diffusion_unified.py --config configs/runtime/sdxl_lora_phase1_skin_histology.runtime.yaml
python scripts/synthetic_data/phase2_reward_guided_lora.py --config configs/runtime/sdxl_lora_phase2_reward_skin_histology.runtime.yaml --max-cycles 5
python scripts/synthetic_data/phase3_morph_reward_guided_lora.py --config configs/runtime/sdxl_lora_phase3_morph_reward_skin_histology.runtime.yaml --max-cycles 5
```

### Tile-based training (alternative)

```bash
# Build tile index
python scripts/patches/build_tile_index_from_masks.py \
  --pairs-csv data/processed/histoseg_pairs.csv \
  --output-csv data/processed/histoseg_tile_pairs.csv \
  --tile-size 512 --stride 64

# Materialize tiles
python scripts/patches/materialize_tile_dataset.py \
  --labels-csv data/processed/histoseg_tile_pairs.csv \
  --output-dir data/artifacts/tiles/materialized_512

# Generate random masks
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
```

## Phase Behavior Notes (from oral-lesions comparison)

- `Phase 1` (foundation) learns base inpainting behavior and histology texture adaptation under masked loss.
- `Phase 2` (reward-guided) iteratively trains and selects checkpoints, usually improving realism and class-conditioned edits.
- `Phase 3` (morph reward) applies the strongest curriculum pressure and can make edits look exaggerated if tuned too aggressively.

Observed differences versus the oral-lesions notebook overrides:
- Oral Grad-CAM runs used longer Phase 1 training with ROI masks and inpaint sample controls.
- Short Phase 1 tile runs can converge to color-fill behavior inside masks before learning tissue microtexture.
- Phase 2 was generally the strongest optimization stage; Phase 3 should be used sparingly when outputs become over-strong.

Recommended tuning order for skin histology:
- Increase Phase 1 duration before adding stronger reward pressure.
- Prefer smaller or local masks for early runs to force tissue-structure learning.
- Keep representative image/mask pairs for inpaint previews.

## Remote H&E Tile Run

The remote H&E tile configs use the tuned round-blob masks and multi-case preview selection:
- `configs/sdxl_lora_phase1_collab_hes_tiles_randommask_w220_remote.yaml` resumes the remote phase-1 run.
- `configs/sdxl_lora_phase1_collab_hes_tiles_randommask_w220_warmstart_remote.yaml` warm-starts from the standard tile checkpoint.
- `scripts/synthetic_data/generate_lora_inpaint_tile_samples.py` generates per-tile samples across LoRA scales.
- Dataset tiles, masks, checkpoints, and generated samples remain local under gitignored directories.

Phase configs: `configs/sdxl_lora_phase*_skin_histology_tiles_randommask.yaml`

---

## 3D Coherence Track (Either Machine)

Self-contained smoke tests (no data download required).

```bash
# Synthetic slices
python scripts/3d/generate_synthetic_slices.py --num-slices 10 --height 256 --width 256 --pattern gradient --output-dir /tmp/test_slices

# Tile-first pipeline dry-run
python scripts/3d/run_volume_inpaint_pipeline.py \
  --slice-glob "/tmp/test_slices/slice_*.png" \
  --volume-id smoke --coarse-label non_cancer \
  --num-slices 10 --mask-radius 0.15 --strict-bbox --dry-run
```

See `docs/MVP_3D_INPAINTING_PLAN.md` for full details and `docs/MVP_MCX_GA_COLOR_THERMAL_PROGRESS.md` for the multi-physics progress report.

---

## DVC Pipeline

DVC is **optional** — use only for submission-time reproducibility.

```bash
pip install dvc

# Murilo's machine (simulations)
dvc repro build_tiles_for_simulation_dataset
dvc repro build_tiles_for_segmentation_dataset
dvc repro run_optical_ga_smoke
dvc repro run_mvp_multiphysics_smoke

# Akio's PC (SDXL LoRA training)
dvc repro render_runtime_configs
dvc repro train_skin_lora_phase1
```

See `dvc.yaml` for all stages organized by workflow domain. Note: `build_gradcam_masks_for_skin` and the old `patch_workflow` stages have been removed — they depended on a deprecated classifier.

---

## SAM2.1 Semantic Segmentation

The POC-v2 Small and Large configs use the same class-aware slide split and
training protocol:

```bash
python scripts/segmentation/train_sam2_hiera_semantic.py \
  --config configs/segmentation/sam2.1_hiera_large_histoseg_poc_v2.yaml
```

Large-v3 keeps that split but guarantees every training tile is visited in each
epoch before weighted rare-class repeats. It also uses paired SAM2-style affine,
zoom, color, and stain augmentation; gradient accumulation; and resumable early
stopping:

```bash
python scripts/segmentation/train_sam2_hiera_semantic.py \
  --config configs/segmentation/sam2.1_hiera_large_histoseg_poc_v3.yaml
```

Training writes `best.pt`, `last.pt`, and `history.json`. To continue a
completed or interrupted v3 run through a higher epoch target, resume from
`last.pt` and set the new total epoch target:

```bash
python scripts/segmentation/train_sam2_hiera_semantic.py \
  --config configs/segmentation/sam2.1_hiera_large_histoseg_poc_v3.yaml \
  --resume outputs/segmentation/sam2.1_hiera_large_poc_v3/last.pt \
  --epochs 20
```

The resume path restores model and optimizer state, preserves history, and
recomputes the cosine schedule against the new total epoch target. Large-v3
also restores its early-stopping patience counter.

---

## Project Layout

```
data/raw/histo_seg_v2/                  # Histo-Seg JPEG + PNG pairs (gitignored)
data/artifacts/tiles_for_simulation/    # Oriented simulation tiles
data/artifacts/tiles_for_segmentation_hdf5/ # Dense per-slide segmentation shards
configs/                                # GA bounds, priors, SDXL phase configs
scripts/
  optical_ga/                           # GA + oriented tile + batch compare
  simulation/                           # MCX, thermal, multi-physics pipeline
  3d/                                   # 3D volume inpainting pipeline
  synthetic_data/                       # SDXL training (Akio's PC only)
  patches/                              # ROI extraction (Akio's PC only)
  segmentation/                         # SAM2.1 semantic segmentation
params.yaml                             # Central config — sectioned per workflow
dvc.yaml                                # Pipeline stages (submission repro)
```

---

## Notes

- This project is standalone (no external repo dependencies).
- Classifier/Grad-CAM tools are deprecated and removed from the pipeline. GT segmentation masks are used instead.
- Large artifacts (data/, outputs/) are gitignored and must be regenerated on each machine.
- The `models.*` paths under `# AKIO'S PC` in `params.yaml` are only needed for SDXL LoRA training on Akio's machine.
