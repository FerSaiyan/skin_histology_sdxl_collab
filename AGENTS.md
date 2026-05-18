# AGENTS.md - skin_histology_sdxl_collab

## Purpose

Standalone research repo for SDXL inpainting on skin histology slices.

Primary goal:
- build Grad-CAM-informed ROI masks from histology classifier signals,
- train SDXL LoRA inpainting in Phase 1/2/3 loops,
- support synthetic slice variation workflows for downstream 3D volume/simulation studies.

Dataset targets:
- Mendeley Histo-Seg dataset `vccj8mp2cg` (version `2`).
- Histo-Seg is 2D only — for 3D volume MVP, see the `docs/MVP_3D_INPAINTING_PLAN.md`.

**Constraint:** No large dataset downloads (MATRICS-A, hundreds of GB) for now.
MVP development uses only existing Histo-Seg data + synthetic circular masks.


## Repo Layout

- `dvc.yaml`: reproducible pipeline stages.
- `params.yaml`: central path/knob config.
- `configs/`: SDXL phase configs + classifier study config.
- `scripts/download_mendeley_histoseg.py`: dataset manifest/download utility.
- `scripts/build_histoseg_pairs_csv.py`: builds paired CSV from image/mask files.
- `scripts/render_runtime_configs.py`: renders runtime phase configs from templates + params.
- `scripts/run_gradcam_from_params.py`: runs Grad-CAM mask generation from params.
- `scripts/synthetic_data/`: vendored core training/selection/benchmark scripts.
- `scripts/3d/`: 3D slice-stack coherence MVP scripts plus Phase B/B.1 volume inpainting pipeline (propagate mask, stack to NIfTI, Z-coherence metrics, metadata builder, tile-first orchestrator, synthetic slice generator).
- `scripts/setup_kohya_sd_scripts.sh`: helper to clone kohya scripts locally.
- `src/`: minimal modules needed by synthetic scripts.
- `docs/MVP_3D_INPAINTING_PLAN.md`: detailed plan for 3D-coherent inpainting.


## Agent Mission

When assisting users, agents should:
1. Get the project runnable from a fresh clone.
2. Keep all paths/configs inside this repo (no dependency on old oral-lesions repo layout).
3. Avoid destructive actions.
4. Explain required external assets clearly (models/checkpoints).


## First-Run Setup (for new users)

From repo root:

```bash
cd /home/fertroll10/Documents/ML/skin_histology_sdxl_collab
```

### 1) Python environment

- Use a Python env with required packages.
- Install from `requirements.txt`.
- Ensure `python` resolves to that environment before running DVC stages.

### 2) Git + DVC

This repo is expected to be both a Git repo and a DVC repo.

If missing (fresh copy without `.git` or `.dvc`), initialize:

```bash
git init
dvc init
```

### 3) Kohya scripts (required for LoRA training)

```bash
bash scripts/setup_kohya_sd_scripts.sh
```

Expected path after setup:
- `kohya_ss/sd-scripts`

If a user already has `sd-scripts` somewhere else, agents should prefer reusing
that path by setting `params.yaml -> models.kohya_scripts_dir`.


## Required External Assets

Agents must verify these before launching training:

1. SDXL base model checkpoint (`.safetensors`)
   - configured in `params.yaml`: `models.sdxl_base_model`

2. Histology classifier checkpoint (`.pth`)
   - configured in `params.yaml`: `models.classifier_checkpoint`

3. Kohya sd-scripts location
   - configured in `params.yaml`: `models.kohya_scripts_dir`

4. (Phase 2/3 only) previous best LoRA checkpoint paths
   - phase2: `initial_lora_weights`
   - phase3: `initial_lora_weights`


## DVC Pipeline Stages

Main stages in `dvc.yaml`:

1. `fetch_histo_seg_manifest`
2. `download_histo_seg_dataset`
3. `build_histoseg_pairs_csv`
4. `render_runtime_configs`
5. `build_gradcam_masks_for_skin`
6. `train_skin_lora_phase1`
7. `train_skin_lora_phase2`
8. `train_skin_lora_phase3`

`build_histoseg_pairs_csv` currently derives `coarse_label` via filename group
(`A -> non_cancer`, `B/C/D -> cancer`) to guarantee binary classes for Phase 2/3 loops.

Typical run sequence:

```bash
dvc repro fetch_histo_seg_manifest
dvc repro download_histo_seg_dataset
dvc repro build_histoseg_pairs_csv
dvc repro render_runtime_configs
dvc repro build_gradcam_masks_for_skin
dvc repro train_skin_lora_phase1
dvc repro train_skin_lora_phase2
dvc repro train_skin_lora_phase3
```

9. `generate_3d_test_masks` (optional, self-contained)
10. `stack_3d_test_volume` (optional, depends on stage 9)
11. `compute_3d_test_coherence` (optional, depends on stage 10)
12. `generate_3d_synthetic_slices` (optional, Phase B smoke test)
13. `run_3d_volume_inpaint_smoke` (optional, depends on stage 12)
14. `run_3d_volume_inpaint_tile_smoke` (optional, Phase B.1 tile-first smoke test, depends on stage 12)

Phase B.1 is the **preferred flow** for high-res volumes: it avoids full-slice
resizing by extracting ROI patches around the mask, inpainting at model
resolution, and merging back.

**Recommended: use `--strict-bbox` mode** for no-padding/no-square extraction.
This validates raw bbox width AND height ≤ `--patch-size` (default 512 in
strict mode).  The `--padding-ratio` is ignored in strict mode.

Key arguments: `--strict-bbox` (flag), `--patch-size` (crop size),
`--target-size` (model resolution).

For heavy stages, `dvc repro <stage> --single-item` is preferred.


## What Agents Should Check Before Running Heavy Jobs

1. `python` points to intended environment.
2. GPU is available for training/inference stages.
3. Required model paths in `params.yaml` are filled.
4. `kohya_ss/sd-scripts` exists.
5. Disk space is sufficient (dataset + outputs + caches).


## Common Pitfalls

1. **Missing kohya scripts**
   - Error about missing `sdxl_train_network.py` or `merge_captions_to_metadata.py`.
   - Fix: run `bash scripts/setup_kohya_sd_scripts.sh`.

2. **Wrong Python env**
   - Missing modules (`geffnet`, `timm`, `diffusers`, etc.).
   - Fix: activate/install correct env, then rerun.

3. **Unfilled placeholders**
   - Paths like `/absolute/path/to/...` cause failures.
   - Fix: update `params.yaml` model path fields and rerun `dvc repro render_runtime_configs`.

4. **Classifier/config mismatch**
   - If class names/counts differ, Grad-CAM or selector scoring may fail or degrade.
   - Fix: keep classifier study config and checkpoint consistent.


## Agent Editing Rules

- Prefer keeping changes local to this repo.
- Do not reintroduce dependencies on external repo-relative paths.
- Keep `dvc.yaml`, `params.yaml`, and `configs/` synchronized when changing paths.
- Do not commit large generated data unless explicitly requested.


## Minimal Health Check Commands

Useful lightweight checks:

```bash
python scripts/download_mendeley_histoseg.py --dataset-id vccj8mp2cg --expected-version 2 --manifest-out /tmp/histo_manifest.json --max-files 2
python scripts/render_runtime_configs.py --params params.yaml
python scripts/run_gradcam_from_params.py --help
python scripts/synthetic_data/build_roi_masks_gradcam.py --help
python scripts/synthetic_data/finetune_stable_diffusion_unified.py --help
python scripts/synthetic_data/phase2_reward_guided_lora.py --help
python scripts/synthetic_data/phase3_morph_reward_guided_lora.py --help
python scripts/3d/propagate_mask_across_slices.py --help
python scripts/3d/stack_slices_to_nifti.py --help
python scripts/3d/compute_z_coherence_metrics.py --help
python scripts/3d/generate_synthetic_slices.py --help
python scripts/3d/build_volume_inpaint_metadata.py --help
python scripts/3d/run_volume_inpaint_pipeline.py --help
python scripts/patches/extract_roi_patches.py --help
python scripts/patches/merge_inpainted_patches.py --help
```


## Collaboration Notes

- Keep README and AGENTS instructions updated whenever stage names or required paths change.
- If adding new phases or scripts, add them to:
  - `dvc.yaml` stages,
  - `README.md` quickstart,
  - this `AGENTS.md` guide.
