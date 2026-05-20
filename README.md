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
  --pairs-csv data/processed/histoseg_pairs.csv \
  --output-dir data/artifacts/tiles_for_simulation \
  --manifest-csv data/artifacts/tiles_for_simulation/tiles_manifest.csv \
  --tile-size 512 --stride 256
```

Outputs placed in `data/artifacts/tiles_for_simulation/`:
- `rgb/*.png` — oriented RGB tiles
- `label_id/*.npy` — semantic class-ID tiles
- `label_vis/*.png` — class-color visualizations
- `epidermis_mask/*.png` — oriented epidermis masks
- `meta/*.json` — per-tile metadata
- `tiles_manifest.csv`, `tiles_stats.json` — global index

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
| `build_tiles_for_simulation.py` | Oriented 512×512 tiles from Histo-Seg slices |
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
  --output-dir data/artifacts/tiles/masks_random_512
```

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
dvc repro run_optical_ga_smoke
dvc repro run_mvp_multiphysics_smoke

# Akio's PC (SDXL LoRA training)
dvc repro render_runtime_configs
dvc repro train_skin_lora_phase1
```

See `dvc.yaml` for all stages organized by workflow domain. Note: `build_gradcam_masks_for_skin` and the old `patch_workflow` stages have been removed — they depended on a deprecated classifier.

---

## Project Layout

```
data/raw/histo_seg_v2/                  # Histo-Seg JPEG + PNG pairs (gitignored)
data/artifacts/tiles_for_simulation/    # Oriented simulation tiles
configs/                                # GA bounds, priors, SDXL phase configs
scripts/
  optical_ga/                           # GA + oriented tile + batch compare
  simulation/                           # MCX, thermal, multi-physics pipeline
  3d/                                   # 3D volume inpainting pipeline
  synthetic_data/                       # SDXL training (Akio's PC only)
  patches/                              # ROI extraction (Akio's PC only)
params.yaml                             # Central config — sectioned per workflow
dvc.yaml                                # Pipeline stages (submission repro)
```

---

## Notes

- This project is standalone (no external repo dependencies).
- Classifier/Grad-CAM tools are deprecated and removed from the pipeline. GT segmentation masks are used instead.
- Large artifacts (data/, outputs/) are gitignored and must be regenerated on each machine.
- The `models.*` paths under `# AKIO'S PC` in `params.yaml` are only needed for SDXL LoRA training on Akio's machine.
