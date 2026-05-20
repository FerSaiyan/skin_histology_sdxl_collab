# AGENTS.md — skin_histology_sdxl_collab

## Purpose

Research repo for skin histology simulation workflows:
- **Optical GA** — estimate skin biophysical parameters from histology tiles using a genetic algorithm over MCX/surrogate forward models
- **MCX light transport** — build volumetric label volumes, run MCX simulations, extract fluence
- **Thermal simulation** — Pennes bioheat solver + visualization
- **Multi-physics MVP** — end-to-end pipeline (label volumes → MCX → GA → thermal → colorimetry)
- **SDXL LoRA inpainting** *(Akio's PC only)* — Phase 1/2/3 Grad-CAM-informed inpainting for synthetic histology variation

Dataset: Mendeley Histo-Seg `vccj8mp2cg` (version 2). 2D only — for 3D volume MVP see `docs/MVP_3D_INPAINTING_PLAN.md`.

**Constraint:** No large dataset downloads (MATRICS-A, hundreds of GB). MVP uses only Histo-Seg + synthetic circular masks.

---

## Repo Layout

```
data/                         # Gitignored — materialized on each machine
  raw/histo_seg_v2/           # Histo-Seg JPG + PNG pairs (downloaded via script)
  processed/                  # Pair CSVs, tile indexes
  artifacts/
    tiles_for_simulation/     # Oriented 512×512 simulation tiles (RGB + labels + epidermis masks)
    mvp/                      # Multi-physics MVP smoke test outputs
    3d/                       # 3D coherence test runs
configs/
  optical_ga_shared_bounds.json
  optical_ga_histoseg_priors.json
  mvp_multiphysics_example.yaml
  sdxl_lora_phase*_skin_histology*.yaml   # SDXL configs (Akio's PC only)
scripts/
  optical_ga/                 # GA + oriented tile + batch compare
  simulation/                 # MCX, thermal, multi-physics pipeline
  3d/                         # 3D volume inpainting pipeline
  synthetic_data/             # SDXL training scripts (Akio's PC only)
  patches/                    # ROI patch extraction + inpainting (Akio's PC only)
  build_histoseg_pairs_csv.py
  download_mendeley_histoseg.py
  render_runtime_configs.py   # SDXL config renderer (Akio's PC only)
src/
  exp/, utils.py              # Shared modules for synthetic training
docs/
  MVP_3D_INPAINTING_PLAN.md
  MVP_MCX_GA_COLOR_THERMAL_PROGRESS.md
dvc.yaml                      # Pipeline stages (repro at submission time)
params.yaml                   # Central config — sectioned per workflow
```

---

## First-Run Setup (Murilo's Machine)

### 1) Clone & enter

```bash
git clone <repo-url> skin_histology_sdxl_collab
cd skin_histology_sdxl_collab
```

### 2) Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use conda:

```bash
conda create -n skin_sim python=3.11
conda activate skin_sim
pip install -r requirements.txt
```

### 3) DVC (optional — only needed for submission-time reproducibility)

```bash
pip install dvc    # already in requirements.txt
dvc init           # if .dvc/ is missing
```

### 4) Configure params.yaml

Open `params.yaml` and update the **simulation_tiles** output paths if needed (defaults are fine for most cases).

### 5) Data: download Histo-Seg dataset

```bash
python scripts/download_mendeley_histoseg.py --dataset-id vccj8mp2cg --expected-version 2 --manifest-out data/metadata/histo_seg_manifest.json
python scripts/download_mendeley_histoseg.py --manifest-in data/metadata/histo_seg_manifest.json --download --download-dir data/raw/histo_seg_v2 --skip-existing
```

### 6) Build paired CSV

```bash
python scripts/build_histoseg_pairs_csv.py --dataset-dir data/raw/histo_seg_v2 --output-csv data/processed/histoseg_pairs.csv --stats-json data/processed/histoseg_pairs_stats.json
```

### 7) Build oriented simulation tiles

```bash
python scripts/optical_ga/build_tiles_for_simulation.py \
  --pairs-csv data/processed/histoseg_pairs.csv \
  --output-dir data/artifacts/tiles_for_simulation \
  --manifest-csv data/artifacts/tiles_for_simulation/tiles_manifest.csv \
  --tile-size 512 --stride 256
```

This writes `data/artifacts/tiles_for_simulation` with:
- RGB tiles (`rgb/*.png`)
- semantic class-ID tiles (`label_id/*.npy`)
- class-color visualizations (`label_vis/*.png`)
- epidermis masks (`epidermis_mask/*.png`)
- per-tile metadata (`meta/*.json`)
- global manifest + stats (`tiles_manifest.csv`, `tiles_stats.json`)

### 8) Run optical GA smoke test (fast, surrogate mode, no MCX needed)

```bash
python scripts/optical_ga/ga_optimiser_optical.py --generations 3 --population-size 8 --target-L 60 --target-a 10 --target-b 15 --forward-mode surrogate --fitness-mode lab --output-dir outputs/optical_ga/smoke_test --seed 42
```

Expected output: `outputs/optical_ga/smoke_test/best_genome.json` and `ga_history.csv`.

### 9) Run multi-physics MVP smoke test (MCX + GA + thermal + colorimetry)

```bash
python scripts/simulation/generate_mvp_smoke_volume.py --output data/artifacts/mvp/smoke_label_volume.npy --shape 8 8 8
python scripts/simulation/run_mvp_multiphysics_from_config.py --config configs/mvp_multiphysics_example.yaml --label-volume data/artifacts/mvp/smoke_label_volume.npy --output-dir data/artifacts/mvp/smoke_run
```

Expected output: multi-file run in `data/artifacts/mvp/smoke_run/`.

---

## Required External Assets (Murilo's Machine)

### MCX binary (required for `forward-mode realistic` and MCX branch of batch compare)

Install from: https://github.com/fangq/mcx

```bash
# Example — adjust for your OS
curl -L https://github.com/fangq/mcx/releases/download/v2025.10/mcx-linux-x64 -o ~/.local/bin/mcx
chmod +x ~/.local/bin/mcx
```

Verify:
```bash
mcx --version
```

### PyXOpto (optional — needed only for MCX vs PyXOpto branch comparison)

Not pre-installed. The repo includes a vendored reference in `external_refs/`. Install via:

```bash
pip install PyXOpto
```

Smoke tests use `forward-mode surrogate` and do **not** require MCX or PyXOpto.

### SDXL base model + classifier (NOT needed for simulations)

The SDXL base model, Kohya scripts, and classifier checkpoint are only used for LoRA inpainting on Akio's PC. The simulation scripts do not reference them.

---

## DVC Pipeline — Reproducibility at Submission Time

DVC is NOT required for daily work. Use it only when you need to reproduce an exact artifact run for a paper submission.

### Simulation stages (Murilo's machine)

```bash
# Data pipeline (one-time)
dvc repro fetch_histo_seg_manifest
dvc repro download_histo_seg_dataset
dvc repro build_histoseg_pairs_csv

# Simulation tile dataset (one-time)
dvc repro build_tiles_for_simulation_dataset

# Optical GA smoke tests
dvc repro run_optical_ga_smoke
dvc repro run_optical_ga_batch_compare_smoke

# Multi-physics MVP smoke test
dvc repro generate_mvp_smoke_label_volume
dvc repro run_mvp_multiphysics_smoke
```

### SDXL stages (Akio's PC only)

```bash
dvc repro render_runtime_configs
dvc repro train_skin_lora_phase1
dvc repro train_skin_lora_phase2
dvc repro train_skin_lora_phase3
```

### 3D coherence stages (either machine — self-contained, no data download)

```bash
dvc repro generate_3d_test_masks
dvc repro stack_3d_test_volume
dvc repro compute_3d_test_coherence
dvc repro generate_3d_synthetic_slices
dvc repro run_3d_volume_inpaint_smoke
dvc repro run_3d_volume_inpaint_tile_smoke
```

---

## Optical GA Workflow Reference

| Script | Description |
|--------|-------------|
| `build_tiles_for_simulation.py` | Build oriented 512×512 simulation tiles (RGB + semantic labels + epidermis masks) |
| `estimate_epidermis_normal.py` | PCA-based epidermis normal estimation |
| `select_orient_tile_for_incidence.py` | Align tile to epidermis incidence angle |
| `ga_optimiser_optical.py` | Genetic algorithm for skin parameter estimation |
| `genome_encoding_optical.py` | 19-parameter skin genome encoding/decoding |
| `forward_model.py` | Surrogate + realistic forward models (MCX/PyXOpto) |
| `colorimetry.py` | Colorimetry (LAB/XYZ) forward calculations |
| `optical_fitness.py` | Fitness functions for GA optimization |
| `optical_report.py` | GA run reporting |
| `run_optical_ga_from_labels.py` | Full GA pipeline from Histo-Seg label tiles |
| `run_optical_ga_batch_compare.py` | Multi-seed batch compare (MCX vs surrogate) |
| `render_best_run_video.py` | Render GA evolution video |
| `qc_tiles_for_simulation.py` | QC plots for simulation tile dataset |
| `label_to_optical_priors.py` | Per-class optical property priors |

## Multi-Physics Simulation Scripts

| Script | Description |
|--------|-------------|
| `mcx_build_volume.py` | Build MCX input volume from label array |
| `mcx_batch_runner.py` | Batch MCX execution for parameter sweeps |
| `mcx_extract_fluence.py` | Extract fluence data from MCX output |
| `thermal_build_model.py` | Build Pennes bioheat model |
| `thermal_solve.py` | Solve bioheat equation |
| `thermal_visualise.py` | Visualize thermal simulation results |
| `run_mvp_multiphysics_from_config.py` | Config-driven multi-physics pipeline runner |
| `run_mvp_multiphysics_pipeline.py` | Pipeline orchestration |
| `colorimetry_metrics.py` | Colorimetry metrics for tissue appearance |

---

## Agent Mission

When assisting users:
1. Get the project runnable from a fresh clone.
2. Keep all paths/configs inside this repo.
3. Avoid destructive actions.
4. Explain required external assets clearly (MCX binary, PyXOpto).
5. Simulation workflow is the primary path; SDXL/LoRA is Akio's-PC-only.

## What Agents Should Check Before Running Heavy Jobs

1. `python` points to intended environment.
2. Required simulation packages installed (`pip install -r requirements.txt`).
3. MCX binary available (if using `forward-mode realistic`).
4. `params.yaml` sections are correct for your workflow (skip AKIO'S PC sections if on Murilo's machine).
5. Disk space sufficient (Histo-Seg download ~3 GB, simulation outputs small).

## Common Pitfalls

1. **Missing MCX binary** — `mc_wrapper.py` calls `mcx` via subprocess. Install MCX or stick to `forward-mode surrogate`.
2. **Wrong Python env** — missing modules. Fix: activate/install correct env.
3. **Missing data** — run the data download + pairs CSV steps before simulation tile building.
4. **Absolute paths in params.yaml** — the `models.*` paths under `# AKIO'S PC` point to Akio's machine. They are NOT needed for simulation work on Murilo's machine.
5. **DVC cache stale** — after git clone, `dvc checkout` will fail (no remote configured). Run scripts directly instead.

## Agent Editing Rules

- Keep changes local to this repo.
- Keep `dvc.yaml`, `params.yaml`, and `configs/` synchronized when changing paths.
- Do not commit large generated data unless explicitly requested.
- Add new simulation/GA scripts to `scripts/optical_ga/` or `scripts/simulation/`.
- Register new DVC stages with clear section headers matching the workflow domain.

## Minimal Health Check Commands (Murilo's Machine)

```bash
python scripts/optical_ga/ga_optimiser_optical.py --help
python scripts/optical_ga/run_optical_ga_from_labels.py --help
python scripts/optical_ga/run_optical_ga_batch_compare.py --help
python scripts/optical_ga/build_tiles_for_simulation.py --help
python scripts/optical_ga/qc_tiles_for_simulation.py --help
python scripts/optical_ga/render_best_run_video.py --help
python scripts/simulation/run_mvp_multiphysics_from_config.py --help
python scripts/simulation/mcx_build_volume.py --help
python scripts/simulation/thermal_build_model.py --help
python scripts/simulation/thermal_solve.py --help
python scripts/3d/propagate_mask_across_slices.py --help
python scripts/3d/stack_slices_to_nifti.py --help
python scripts/3d/compute_z_coherence_metrics.py --help
```

## Collaboration Notes

- Keep README and AGENTS instructions updated whenever stage names or required paths change.
- Sections labelled `AKIO'S PC` → Akio's SDXL/inpainting workflows. `MURILO'S PC` → Murilo's simulation workflows.
- If adding new simulation phases or scripts, add them to: `dvc.yaml` simulation section, `scripts/optical_ga/` or `scripts/simulation/`, this `AGENTS.md` guide, and `README.md`.
