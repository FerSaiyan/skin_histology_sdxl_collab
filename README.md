# Skin Histology SDXL Inpainting Collab

This folder is a shareable mini-repo scaffold based on the `Lora_&_Finetune_SDXL.ipynb`
workflow, adapted to skin histology slices (Histo-Seg, Mendeley `vccj8mp2cg`, v2).

It includes:
- a dataset downloader for Mendeley public API,
- a CSV builder that pairs histology `.jpg` images with `.png` segmentation masks,
- vendored synthetic pipeline scripts (`build_roi_masks_gradcam`, `finetune_stable_diffusion_unified`, phase2/phase3 loops),
- SDXL LoRA Phase 1/2/3 config templates for Grad-CAM informed inpainting,
- a DVC pipeline for reproducible data prep and training orchestration.

## Folder Layout

- `dvc.yaml`: reproducible pipeline stages.
- `params.yaml`: editable knobs and paths.
- `scripts/download_mendeley_histoseg.py`: manifest + downloader.
- `scripts/build_histoseg_pairs_csv.py`: builds Grad-CAM/training CSV.
- `scripts/render_runtime_configs.py`: injects shared model paths from `params.yaml` into runtime phase configs.
- `scripts/run_gradcam_from_params.py`: launches Grad-CAM mask generation from `params.yaml`.
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

For long stages, you can run only one stage with `--single-item`.

## Notes

- This project is standalone and no longer depends on the oral-lesions repository layout.
- Shared model paths are configured once in `params.yaml` and propagated to runtime configs.
- Core synthetic training scripts are vendored under `scripts/synthetic_data/`.
- Minimal runtime modules are vendored under `src/` (`exp`, `utils`, classifier model factory).
- LoRA training still depends on `kohya_ss/sd-scripts` (bootstrapped with `scripts/setup_kohya_sd_scripts.sh`).
- Large artifacts are marked with `cache: false` in DVC and ignored in Git by default.
