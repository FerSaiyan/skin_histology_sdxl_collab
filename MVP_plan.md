# MVP Plan: Skin Histology Grad-CAM -> SDXL Inpainting -> Slice-Coherent 3D Augmentation

## 1) Mission

Build a practical, reproducible pipeline in this repository to:

1. prepare Histo-Seg slice data,
2. generate classifier-guided Grad-CAM masks,
3. train SDXL LoRA inpainting (Phase 1 -> Phase 2 -> Phase 3),
4. support patch-level editing for large slides,
5. reconstruct/edit full-resolution slices with local patch replacement,
6. later extend to slice-consistent 3D volume augmentation.

This plan is aligned to the current repository structure and scripts.

---

## 2) Current Repository-Accurate Pipeline

These are the active DVC stages in `dvc.yaml`:

1. `fetch_histo_seg_manifest`
2. `download_histo_seg_dataset`
3. `build_histoseg_pairs_csv`
4. `render_runtime_configs`
5. `build_gradcam_masks_for_skin`
6. `train_skin_lora_phase1`
7. `train_skin_lora_phase2`
8. `train_skin_lora_phase3`

Key files controlling behavior:

- `params.yaml`
  - single source of truth for:
    - `models.sdxl_base_model`
    - `models.classifier_checkpoint`
    - `models.kohya_scripts_dir`
- `configs/sdxl_lora_phase*.yaml` (templates)
- `configs/runtime/*.runtime.yaml` (auto-generated runtime configs)
- `scripts/run_gradcam_from_params.py`
- `scripts/render_runtime_configs.py`

---

## 3) Non-Negotiable Rules

1. Never train directly on full WSI resolution for diffusion inpainting.
2. Keep synthetic data out of validation/test folds.
3. Preserve provenance for every generated sample.
4. Prefer local, mask-constrained edits.
5. Reject synthetic outputs that fail classifier or visual QC.

---

## 4) Current Data Status and Labeling Rule

Current status:

- mask RGB palette decoding has been implemented in `scripts/build_histoseg_pairs_csv.py`,
- `coarse_label` currently uses `filename_group` mode by default:
  - `A -> non_cancer`
  - `B/C/D -> cancer`
- current split is balanced (`19 cancer`, `19 non_cancer`) in `data/processed/histoseg_pairs_stats.json`.

Why this rule is used right now:

- it guarantees binary classes for Phase 2/3 loops,
- it avoids ambiguous direct cancer extraction from mixed-class masks until patch-level ROI labeling is fully integrated.

Validation requirement:

- before major training runs, visually verify a sample of A/B/C/D groups against masks and source tissue.

---

## 5) MVP Phases

## Phase A - Environment and configuration lock

Objective:
- ensure everyone can run the same pipeline from clone.

Checklist:
- Python env ready.
- `kohya_ss/sd-scripts` path valid (reuse existing local install when available).
- `params.yaml` model paths filled.

Commands:

```bash
cd /home/fertroll10/Documents/ML/skin_histology_sdxl_collab
dvc repro render_runtime_configs --single-item
```

Acceptance:
- `configs/runtime/*.runtime.yaml` generated.
- `data/metadata/runtime_config_sync.json` reflects real absolute paths.

---

## Phase B - Data acquisition and pairing

Objective:
- reproducibly fetch dataset and build image-mask pair table.

Commands:

```bash
dvc repro fetch_histo_seg_manifest --single-item
dvc repro download_histo_seg_dataset --single-item
dvc repro build_histoseg_pairs_csv --single-item
```

Acceptance:
- 38 `.jpg` + 38 `.png` files in `data/raw/histo_seg_v2`.
- `data/processed/histoseg_pairs.csv` generated.
- `data/processed/histoseg_pairs_stats.json` generated.

---

## Phase C - Grad-CAM masks

Objective:
- generate ROI masks from classifier saliency.

Command:

```bash
dvc repro render_runtime_configs --single-item
dvc repro build_gradcam_masks_for_skin --single-item
```

Acceptance:
- masks in `data/artifacts/roi_masks/histoseg_pairs`.
- sample visual QC confirms lesion/tissue focus is plausible.

---

## Phase D - SDXL LoRA training

Objective:
- run staged LoRA training with classifier-guided selection.

Commands:

```bash
# Phase 1
dvc repro train_skin_lora_phase1 --single-item

# Phase 2
dvc repro render_runtime_configs --single-item
dvc repro train_skin_lora_phase2 --single-item

# Phase 3
dvc repro render_runtime_configs --single-item
dvc repro train_skin_lora_phase3 --single-item
```

Acceptance:
- checkpoints + logs under `outputs/finetunes/...`.
- phase history/summary files exist for phase2/phase3 loops.

---

## 6) Large-Slide Strategy (Patch Inpaint + Paste-Back)

Because histology slides are very large, MVP should edit local patches only.

### Proposed patch workflow

1. For each source slice:
   - use Grad-CAM mask,
   - compute ROI bbox,
   - extract context patch around ROI (e.g., 768 or 1024 square).
2. Resize patch to model working resolution (e.g., 512) for inpainting.
3. Inpaint only masked area.
4. Upscale edited patch back to original patch size.
5. Blend and paste edited patch into original full-resolution slice.
6. Save per-edit metadata (bbox, scale factors, seed, prompt, checkpoint).

### Recommended default hyperparameters

- patch extraction size: 1024 (fallback 768 if VRAM constrained)
- inpaint resolution: 512
- overlap/feather blend: 16-32 px
- denoise strength: 0.45-0.60
- guidance scale: 4.5-6.5

### New scripts to add (next implementation)

- `scripts/patches/extract_roi_patches.py`
- `scripts/patches/inpaint_roi_patches.py`
- `scripts/patches/merge_inpainted_patches.py`
- `scripts/patches/qc_patch_replacement.py`

---

## 7) 3D Coherence Extension (Post-MVP)

Do not start with full 3D diffusion.

First extension:

1. identify slice neighborhoods around anchor slice,
2. propagate target edit parameters to adjacent slices,
3. regularize per-slice edits with neighboring slice similarity constraints,
4. reject inconsistent volumes by morphology + intensity continuity metrics.

Suggested QC metrics:

- adjacent-slice SSIM in non-edited regions,
- connected-component continuity of edited morphology,
- classifier probability trajectory smoothness across slice index.

---

## 8) Minimal QA Checklist Before Any Training Run

1. `params.yaml` model paths are real and accessible.
2. `data/processed/histoseg_pairs_stats.json` has expected class distribution.
3. `dvc repro render_runtime_configs --single-item` succeeds.
4. Grad-CAM stage runs for a small subset first.
5. GPU memory budget checked for selected resolution/batch config.

---

## 9) Short-Term Priority Order

1. Run Phase 1 baseline.
2. Add patch extract -> inpaint -> paste-back scripts.
3. Add patch-level QC and rejection metrics.
4. Run Phase 2/3 reward loops on patch-based pipeline.
5. Add slice-neighborhood coherence checks.

---

## 10) Reusable Agent Prompt (Implementation Milestones)

Use this prompt for future coding agents:

```text
You are implementing the MVP in this repository incrementally.

Repository root:
/home/fertroll10/Documents/ML/skin_histology_sdxl_collab

Read first:
- AGENTS.md
- MVP_plan.md
- dvc.yaml
- params.yaml

Constraints:
1) Do NOT break existing stages:
   - fetch_histo_seg_manifest
   - download_histo_seg_dataset
   - build_histoseg_pairs_csv
   - render_runtime_configs
   - build_gradcam_masks_for_skin
   - train_skin_lora_phase1/2/3
2) Keep paths and configs local to this repo.
3) Make small, reviewable changes.
4) After each milestone, run relevant smoke tests and report exact commands/results.

Current coarse labeling behavior:
- build_histoseg_pairs_csv defaults to filename_group mode
- A -> non_cancer, B/C/D -> cancer

Implement ONLY the requested milestone from the list below:

Milestone 1: ROI patch extraction
- Add scripts/patches/extract_roi_patches.py
- Input: histoseg_pairs.csv + ROI masks
- Output: patch images, patch masks, patch metadata CSV
- Include bbox, scale, source filename, coarse_label, seed placeholder

Milestone 2: Patch inpainting runner
- Add scripts/patches/inpaint_roi_patches.py
- Use existing SDXL/LoRA stack and runtime config paths
- Inpaint masked region on patches only
- Save generated patch + generation metadata

Milestone 3: Paste-back and blending
- Add scripts/patches/merge_inpainted_patches.py
- Paste edited patch back to full-resolution source slice
- Feather seam; preserve outside ROI
- Save edited full-resolution slice and audit metadata

Milestone 4: QC and rejection
- Add scripts/patches/qc_patch_replacement.py
- Compute basic metrics (mask coverage, seam difference, non-ROI drift)
- Emit pass/fail flag and summary report

Milestone 5: DVC integration
- Add optional DVC stages for milestones 1-4 without changing existing stage behavior

Required output format:
1) Files changed
2) Why each change was made
3) Commands executed
4) Test results
5) Next recommended milestone
```
