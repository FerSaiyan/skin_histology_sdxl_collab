# MVP Progression: Inpainted Skin Histology Tiles → Multi-Physics Simulation Volumes

## Mission & Scope

This document defines an **incremental MVP roadmap** for the
`skin_histology_sdxl_collab` repository, tracking what is already built, what
can be run today, and the planned extension from **2D tile inpainting** →
**3D coherent volumes** → **multi-physics simulation support** (MCX light
transport, Genetic Algorithm optimisation, colorimetry, and thermal simulation).

### Repository Core Capability

The repo currently delivers an **SDXL LoRA inpainting pipeline** for 2D skin
histology slices (Mendeley Histo-Seg dataset).  The pipeline:

1. Downloads and pairs histology images with segmentation masks.
2. Mines 512×512 tiles with tissue-overlap and white-content filters.
3. Generates tile-aligned random inpainting masks (organic brush strokes
   constrained to tissue regions).
4. Runs SDXL LoRA training in three progressive phases:
   - **Phase 1:** foundation inpainting + histology texture adaptation.
   - **Phase 2:** reward-guided iterative train/select loop.
   - **Phase 3:** morph-reward curriculum (use sparingly).
5. Extends 2D editing to **3D volumes** via cylindrical mask propagation,
   tile-first high-res inpainting, NIfTI stacking, and Z-coherence metrics.

### Scope Note: Classifier Branch Out-of-Scope

The classifier-based branch (training, Grad-CAM masks, classifier-supervised
scoring) is **out of scope for this MVP** due to current quality and results.
This document does not depend on any classifier workflow.

All MVP work uses **pre-generated random tile masks**
(`scripts/patches/generate_random_tile_masks.py`) as the mask source, which
requires no classifier and no external model.  The
**semantic-mask-as-ground-truth** principle guides the design: semantic masks
(from future detectors, manual annotations, or other sources) are the preferred
ground truth when available; random masks serve as the practical fallback for
training and inpainting workflows.

---

## What Is Already Implemented (Can Run Now)

### Data & Tile Pipeline

| Stage / Script | DVC Stage | Status |
|----------------|-----------|--------|
| Mendeley manifest + download | `fetch_histo_seg_manifest`, `download_histo_seg_dataset` | ✅ Implemented |
| Image-mask pair CSV builder | `build_histoseg_pairs_csv` | ✅ Implemented |
| 512×512 tile index mining (coverage+white filters) | `build_histoseg_tile_index` | ✅ Implemented |
| Tile materialisation (copies tiles once) | `materialize_histoseg_tile_dataset` | ✅ Implemented |
| Random inpainting mask generation | `generate_histoseg_tile_random_masks` | ✅ Implemented |
| Runtime config rendering | `render_runtime_configs` | ✅ Implemented |

### Training Pipeline

| Stage / Script | DVC Stage | Status |
|----------------|-----------|--------|
| Phase 1 LoRA (foundation inpainting) | `train_skin_lora_phase1` | ✅ Implemented |
| Phase 2 LoRA (reward-guided) | `train_skin_lora_phase2` | ✅ Implemented |
| Phase 3 LoRA (morph reward) | `train_skin_lora_phase3` | ✅ Implemented |
| Select best checkpoint | `select_best_lora_checkpoint.py` / `select_best_lora_inpaint_checkpoint.py` | ✅ Implemented |

### Patch / Tile-First Workflow (Full-Res Editing)

| Stage / Script | DVC Stage | Status |
|----------------|-----------|--------|
| Extract ROI patches from full slices | `extract_roi_patches` | ✅ Implemented |
| Inpaint patches at model resolution | `inpaint_roi_patches` | ✅ Implemented |
| Merge inpainted patches back | `merge_inpainted_patches` | ✅ Implemented |
| QC patch replacement | `qc_patch_replacement` | ✅ Implemented |
| Build tile index from masks | `scripts/patches/build_tile_index_from_masks.py` | ✅ Implemented |
| Materialise tile dataset | `scripts/patches/materialize_tile_dataset.py` | ✅ Implemented |

### 3D Volume Coherence Pipeline

| Stage / Script | DVC Stage | Status |
|----------------|-----------|--------|
| Cylindrical mask propagation | `generate_3d_test_masks` | ✅ Implemented |
| 2D slice → NIfTI stacking | `stack_3d_test_volume` | ✅ Implemented |
| Z-coherence metrics (SSIM + gradient) | `compute_3d_test_coherence` | ✅ Implemented |
| Synthetic slice generator | `generate_3d_synthetic_slices` | ✅ Implemented |
| Volume inpainting orchestrator (dry-run) | `run_3d_volume_inpaint_smoke` | ✅ Implemented |
| Tile-first volume inpainting (preferred) | `run_3d_volume_inpaint_tile_smoke` | ✅ Implemented |

### Supporting Utilities

| Utility | Path | Status |
|---------|------|--------|
| Dataset manifest/download | `scripts/download_mendeley_histoseg.py` | ✅ Implemented |
| Runtime config injection | `scripts/render_runtime_configs.py` | ✅ Implemented |
| Volume inpaint metadata builder | `scripts/3d/build_volume_inpaint_metadata.py` | ✅ Implemented |
| Run manifest + orchestration | `scripts/3d/run_volume_inpaint_pipeline.py` | ✅ Implemented |
| Analysis: plot run comparison | `scripts/analysis/plot_run_comparison.py` | ✅ Implemented |

### Thermal Simulation

| Script | Path | Status |
|--------|------|--------|
| Thermal model builder | `scripts/simulation/thermal_build_model.py` | ✅ Implemented |
| Pennes bioheat solver | `scripts/simulation/thermal_solve.py` | ✅ Implemented |
| Temperature visualiser | `scripts/simulation/thermal_visualise.py` | ✅ Implemented |

---

## Mask-First Strategy & Semantic-Mask-as-Ground-Truth

The entire pipeline is designed around **mask-constrained inpainting**:

1. **Masks define where editing happens.**  Outside the mask, the original
   tissue is preserved verbatim.  Inside the mask, SDXL inpainting learns to
   produce plausible histology texture consistent with the class label.
2. **Random masks are a valid ground-truth proxy** for the current MVP.
   The masks are generated with tissue-overlap constraints, avoiding white/black
   voids and mimicking the size/shape distribution of real lesions
   (`--min-area-frac 0.08`, `--max-area-frac 0.22`, organic brush strokes).
3. **Semantic masks are the preferred ground-truth target** when available
   (e.g., from future YOLO detectors, manual annotations, or external tissue
   segmenters — see `docs/MVP_3D_INPAINTING_PLAN.md` Phase D).  The pipeline
   **does not change** — only the mask source changes.
4. **3D cylindrical masks** replicate the same mask across Z, forming a
   consistent editing column through the volume.

This design means every intermediate representation (tiles, patches, inpaint
outputs, volumes) is **provenanced**: the mask file, seed, parameters, and
checkpoint are recorded in metadata CSVs and run manifests.

---

## 2D Tile-First Workflow (Current)

The tile-first workflow for editing full-resolution histology slices is the
preferred path (implemented in `scripts/patches/` and orchestrated in the 3D
pipeline):

```
Source slice (e.g. 5000×5000 px)
  │
  ├── 1. Build tile index (512×512 tiles, stride 64)
  │      scripts/patches/build_tile_index_from_masks.py
  │
  ├── 2. Materialize tiles once
  │      scripts/patches/materialize_tile_dataset.py
  │
  ├── 3. Generate random inpainting masks
  │      scripts/patches/generate_random_tile_masks.py
  │
  ├── 4. Train SDXL LoRA (Phase 1 → 2 → 3)
  │      scripts/synthetic_data/finetune_stable_diffusion_unified.py
  │      scripts/synthetic_data/phase2_reward_guided_lora.py
  │      scripts/synthetic_data/phase3_morph_reward_guided_lora.py
  │
  └── 5. For each high-res slice, edit via patch workflow:
         a. Extract ROI patches around mask (scripts/patches/extract_roi_patches.py)
         b. Inpaint patches at 512×512 (scripts/patches/inpaint_roi_patches.py)
         c. Merge patches back (scripts/patches/merge_inpainted_patches.py)
         d. QC replacement (scripts/patches/qc_patch_replacement.py)
```

**Key parameters** (`params.yaml` → `patch_workflow`):
- `patch_size: 1024` — full-resolution crop size
- `target_size: 512` — model working resolution
- `feather_radius: 16` — blending seam
- `strength: 0.55` — denoising strength

### Quick smoke test (no model required):

```bash
# Generate synthetic slices
python scripts/3d/generate_synthetic_slices.py \
  --num-slices 10 --height 256 --width 256 --pattern gradient \
  --output-dir /tmp/test_tile_slices

# Run tile-first pipeline in dry-run mode (strict bbox)
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

---

## 2D → 3D Volume Transition (Current + Next)

### Current (Implemented)

The 3D pipeline treats a volume as an ordered stack of 2D slices:

1. **Generate cylindrical masks** — replicate the same mask across all
   Z-slices (`propagate_mask_across_slices.py`).
2. **Edit slices independently** using the 2D tile-first workflow.
3. **Stack to NIfTI** after editing (`stack_slices_to_nifti.py`).
4. **Measure Z-coherence** (adjacent-slice SSIM, Z-gradient smoothness)
   via `compute_z_coherence_metrics.py`.
5. **Full orchestrator** (`run_volume_inpaint_pipeline.py`) ties it all
   together with run manifests, metadata CSV, and dry-run mode.

The **preferred flow** (Phase B.1) uses `--strict-bbox` mode: raw mask
bounding box must fit within the patch size, eliminating padding and square
transforms.

### Next: Volume Augmentation & Multi-Slice Consistency

| Step | Description | Target |
|------|-------------|--------|
| V1 | Run a real inpainting pass on a slice stack with a trained LoRA (requires model paths) | Current phase |
| V2 | Add Z-coherence thresholding — reject volumes where SSIM drops below configurable threshold | Next implementation |
| V3 | Relax rigid cylindrical masks with per-slice affine warping (small random perturbations between adjacent slices) | Next implementation |
| V4 | Add Z-smoothness regularisation to Phase 2/3 reward functions | Future |

---

## Multi-Physics Simulation Roadmap

The following sections outline how the inpainting pipeline can serve as a data
engine for physics-based simulation studies.  These are **planned extensions**,
not yet implemented.

### MCX (Monte Carlo eXtreme) — Light Transport Simulation

**Goal:** Generate volumes suitable as input to MCX for GPU-accelerated photon
transport simulation.

**MCX** (http://mcx.space/) simulates photon migration in turbid media like
tissue.  It requires:
- A 3D voxel volume defining optical properties per voxel (absorption μa,
  scattering μs, anisotropy g, refractive index n).
- A source position and detector geometry.

**How the repurposed pipeline contributes:**

| MCX Input | How Inpainting Pipeline Provides It |
|-----------|-------------------------------------|
| 3D tissue geometry (voxel grid) | NIfTI volume from `stack_slices_to_nifti.py` after inpainting |
| Tissue-type labels per voxel | Mask regions define one tissue type; unedited background is native tissue. Multi-class masks (future) enable per-label optical properties. |
| Optical property maps | Derived from inpainted RGB → optical property lookup tables (future: `scripts/simulation/mcx_optical_map.py`) |
| Batch of volumes for Monte Carlo studies | Pipeline can generate many variants by varying random mask seeds, LoRA checkpoints, and prompt parameters |

**Phase B — MCX Scaffolding (implemented):**

| Script | Purpose |
|--------|---------|
| `scripts/simulation/mcx_build_volume.py` | Convert NIfTI / .npy volume + optional label map → MCX-compatible JSON config + binary volume + media table |
| `scripts/simulation/mcx_batch_runner.py` | Dry-run or execute MCX simulations over a list of config files |
| `scripts/simulation/mcx_extract_fluence.py` | Extract summary statistics and optional axial projection PNG from MCX output |

**Example commands (smoke test, no MCX binary required):**

```bash
# 1. Create a tiny synthetic volume
python -c "
import numpy as np
vol = np.random.randint(0, 4, size=(16, 16, 8)).astype(np.uint8)
np.save('/tmp/mcx_smoke_volume.npy', vol)
"

# 2. Build MCX artifacts
python scripts/simulation/mcx_build_volume.py \
  --volume /tmp/mcx_smoke_volume.npy \
  --output-dir /tmp/mcx_smoke_output

# 3. Dry-run batch runner over the generated config
python scripts/simulation/mcx_batch_runner.py \
  /tmp/mcx_smoke_output/mcx_config.json \
  --dry-run \
  --output-dir /tmp/mcx_smoke_batch

# 4. Extract fluence from a synthetic fluence array
python scripts/simulation/mcx_extract_fluence.py \
  --input /tmp/mcx_smoke_volume.npy \
  --output-json /tmp/mcx_smoke_fluence_stats.json \
  --output-png /tmp/mcx_smoke_fluence_mip.png
```

**Acceptance criteria for MCX integration:**

- A single inpainted NIfTI volume can be exported to `mcx --input-format jds` format.
- MCX simulation runs on the exported volume (using default tissue optical
  properties from literature, e.g., Jacques 2013).
- Fluence maps are written back as PNG overlays on the original histology slices.

---

### Genetic Algorithm (GA) — Mask & Parameter Optimisation

**Goal:** Use GA to search the inpainting parameter space (mask shape, location,
strength, prompt tokens, seed) for optimised outputs against a reward function.

**How it fits:**

| GA Component | Implementation |
|--------------|----------------|
| Genome | Encodes: mask center (x,y), mask radius, mask shape type, denoising strength, guidance scale, random seed |
| Population | Set of parameter combinations, each producing one inpainting run |
| Fitness function | Combined score from Z-coherence metrics + inpainting fidelity (mask-edge blending quality) |
| Selection + crossover | Standard GA operators (tournament selection, uniform crossover) |
| Mutation | Perturb mask parameters ±10%, randomise seed |

**GA scripts (implemented skeleton + extensible fitness wiring):**

| Script | Purpose |
|--------|---------|
| `scripts/ga/genome_encoding.py` | Encode/decode inpainting parameters as a fixed-length genome |
| `scripts/ga/run_inpaint_fitness.py` | Run one inpainting + scoring cycle for a genome |
| `scripts/ga/ga_optimiser.py` | Main GA loop: population init → fitness → selection → crossover → mutation → repeat (supports `--mock` and `--fitness-json`) |
| `scripts/ga/ga_report.py` | Analyse convergence, pareto front, best genomes |

**Acceptance criteria for GA integration:**

- A configurable genome produces reproducible inpainting runs.
- GA converges to lower mask-edge RMSE over ≤50 generations.
- Pareto front shows trade-off between inpainting fidelity and Z-coherence.

---

### Colorimetry — Quantitative Colour Analysis

**Goal:** Validate inpainted regions against target colour distributions
derived from real histology, using CIELAB colour space metrics.

**How it fits:**

| Colour Metric | Purpose | Implementation |
|---------------|---------|----------------|
| CIELAB ΔE* | Per-pixel colour difference between inpainted region and reference tissue | `scripts/analysis/plot_run_comparison.py` (extend with CIELAB) |
| Histogram correlation | Compare RGB/HSV/La*b* histograms inside inpainted vs outside mask | New `scripts/simulation/colorimetry_metrics.py` |
| Stain colour deconvolution | Match Haematoxylin & Eosin stain vectors | Future, requires reference stain matrix |
| Colour constancy | Ensure inpainted regions do not introduce systematic colour shift relative to unmasked tissue | Already partially covered by `qc_patch_replacement.py` seam diff |

Colorimetry is the **lightest-weight extension** — most metrics can be added
as additional columns in the existing QC metadata CSV
(`data/artifacts/qc/metadata/qc_metadata.csv`).

**Acceptance criteria for colorimetry:**

- CIELAB ΔE* between inpainted and reference patches is computed and reported
  in QC output.
- Inpainted regions do not systematically deviate from unmasked tissue colour
  distribution (two-sample KS test on La*b* channels, p > 0.05).
- Stain colour deconvolution (H&E) works on tile patches.

---

### Thermal Simulation — Heat Transfer in Tissue Volumes

**Goal:** Use inpainted 3D volumes as input to finite-difference or finite-element
thermal simulation (e.g., Pennes bioheat equation).

**How it fits:**

| Thermal Component | Relationship to Pipeline |
|-------------------|-------------------------|
| Tissue geometry | NIfTI volume from `stack_slices_to_nifti.py` defines the 3D domain |
| Initial temperature | Uniform body temperature (37 °C) — no modification needed |
| Tissue thermal properties | Assigned per label region (mask = tumour/label, background = healthy tissue). Literature values for skin (37 °C) and tumour (slightly higher) |
| Heat source | External laser or probe positioned at the mask center (future parameterisation) |
| Boundary conditions | Fixed temperature at volume boundaries or convective cooling at skin surface |

**Phase C — Thermal Baseline Utilities (implemented):**

| Script | Purpose |
|--------|---------|
| `scripts/simulation/thermal_build_model.py` | Convert NIfTI / .npy label volume to thermal property arrays (rho, c, k, wb, qmet) with built-in defaults for labels 0/1/2 |
| `scripts/simulation/thermal_solve.py` | Explicit finite-difference solver for the Pennes bioheat equation; supports `none` and `spherical` heat source modes; NaN/Inf fail-fast |
| `scripts/simulation/thermal_visualise.py` | Export 2D temperature slice PNG (matplotlib optional) + summary JSON with min/max/mean statistics |

**Example commands (smoke test, no GPU required):**

```bash
# 1. Create a tiny synthetic label volume
python -c "
import numpy as np
vol = np.ones((8, 10, 10), dtype=np.int32)
vol[3:6, 3:7, 3:7] = 2   # lesion core
vol[:2, :, :] = 0         # air layer at top
np.save('/tmp/thermal_labels.npy', vol)
"

# 2. Build thermal model
python scripts/simulation/thermal_build_model.py \
  --label-volume /tmp/thermal_labels.npy \
  --output-dir /tmp/thermal_model_out

# 3. Solve Pennes equation (small steps)
python scripts/simulation/thermal_solve.py \
  --model-npz /tmp/thermal_model_out/thermal_model.npz \
  --output-dir /tmp/thermal_solve_out \
  --dt 0.01 --num-steps 50 \
  --source-mode spherical \
  --source-center 4 5 5 \
  --source-radius-vox 2.5 \
  --source-power 5e5

# 4. Visualise central axial slice
python scripts/simulation/thermal_visualise.py \
  --temperature-npy /tmp/thermal_solve_out/temperature_final.npy \
  --output-dir /tmp/thermal_vis_out \
  --slice-axis 0 --slice-index 4
```

**Acceptance criteria for thermal simulation:**

- A 3D NIfTI volume (inpainted or test) is converted to a thermal grid with property arrays.
- Pennes bioheat equation is solved for a short heating pulse at a user-defined location.
- 2D temperature slices are exported as PNG overlays (when matplotlib available).
- Summary JSON is always written regardless of matplotlib availability.

---

## Practical Execution Order

The following is the recommended ordering, from what works today to future
extensions.  Each stage should be verified before moving to the next.

### Stage 0 — Verify Baseline (Today)

```bash
# 1. Data fetch + tile pipeline
dvc repro fetch_histo_seg_manifest
dvc repro download_histo_seg_dataset
dvc repro build_histoseg_pairs_csv
dvc repro build_histoseg_tile_index
dvc repro materialize_histoseg_tile_dataset
dvc repro generate_histoseg_tile_random_masks

# 2. Smoke test 3D pipeline
python scripts/3d/generate_synthetic_slices.py \
  --num-slices 10 --height 256 --width 256 --pattern gradient \
  --output-dir /tmp/test_tile_slices

python scripts/3d/run_volume_inpaint_pipeline.py \
  --slice-glob "/tmp/test_tile_slices/slice_*.png" \
  --volume-id baseline_smoke \
  --coarse-label non_cancer \
  --num-slices 10 \
  --mask-radius 0.15 \
  --strict-bbox \
  --patch-size 128 \
  --target-size 64 \
  --output-dir /tmp/test_baseline \
  --dry-run
```

**Acceptance:** All 8 pipeline steps report success.  Run manifest JSON is
saved.  Coherence metrics contain `adjacent_ssim` and `z_gradient_smoothness`.

### Stage 1 — Train + Real Inpainting (Requires SDXL Base Model + Kohya Scripts)

```bash
# 3. Render runtime configs (fills model paths)
dvc repro render_runtime_configs

# 4. Phase 1 training (tile + random mask mode)
dvc repro train_skin_lora_phase1

# 5. Phase 2 reward-guided optimisation
dvc repro train_skin_lora_phase2

# 6. Run real volume inpainting with trained LoRA
python scripts/3d/run_volume_inpaint_pipeline.py \
  --slice-glob "data/raw/histo_seg_v2/*.jpg" \
  --volume-id my_inpaint_run \
  --coarse-label non_cancer \
  --num-slices 5 \
  --mask-radius 0.15 \
  --strict-bbox \
  --patch-size 512 \
  --target-size 512 \
  --output-dir data/artifacts/3d/volume_inpaint_runs
```

**Acceptance:** Inpainted slices are generated.  NIfTI volume stacks correctly.
QC metadata confirms no seam artifacts.

### Stage 2 — Colorimetry Extension

- Implement `scripts/simulation/colorimetry_metrics.py` (new directory).
- Extend `qc_patch_replacement.py` to compute CIELAB ΔE* and KS statistics.
- Integrate into the existing QC CSV output.

### Stage 3 — GA Optimisation

- Implement `scripts/ga/` scripts.
- Run GA on a subset of parameters (mask center + strength as first genome).
- Verify convergence with geometry-only fitness function.

### Stage 4 — MCX Export

- Validate `scripts/simulation/mcx_build_volume.py`,
  `scripts/simulation/mcx_batch_runner.py`, and
  `scripts/simulation/mcx_extract_fluence.py` on representative volumes.
- Run MCX on a test volume (requires MCX binary installed separately).
- Verify fluence map overlay on histology.

### Stage 5 — Thermal Simulation (Implemented)

- `scripts/simulation/thermal_build_model.py` — builds thermal property arrays from a label volume.
- `scripts/simulation/thermal_solve.py` — explicit finite-difference Pennes solver with spherical heat source.
- `scripts/simulation/thermal_visualise.py` — temperature slice PNG + summary JSON.
- Example commands in the Thermal section above provide a full smoke test.
- Verify temperature maps export as PNG (requires matplotlib).

## Config-Driven Execution Layer

The multi-physics pipeline can also be driven from a YAML configuration file,
enabling reproducible, version-controlled runs without CLI boilerplate.

### Script

- `scripts/simulation/run_mvp_multiphysics_from_config.py` — reads a YAML config
  and invokes `run_mvp_multiphysics_pipeline.py` with correctly mapped arguments.

### Config file

- `configs/mvp_multiphysics_example.yaml` — documented example with classifier-free
  defaults for all GA, MCX, and thermal options.

### Usage

```bash
# Dry-run to inspect the resolved command
python scripts/simulation/run_mvp_multiphysics_from_config.py \
    --config configs/mvp_multiphysics_example.yaml --dry-run

# Full run with config defaults (requires a label volume at the path in config)
python scripts/simulation/run_mvp_multiphysics_from_config.py \
    --config configs/mvp_multiphysics_example.yaml

# Override label volume and output directory on CLI
python scripts/simulation/run_mvp_multiphysics_from_config.py \
    --config configs/mvp_multiphysics_example.yaml \
    --label-volume /tmp/my_labels.npy \
    --output-dir /tmp/my_run

# Fail-fast override
python scripts/simulation/run_mvp_multiphysics_from_config.py \
    --config configs/mvp_multiphysics_example.yaml --fail-fast

# Quick smoke test: create a tiny volume then run from config with overrides
python -c "
import numpy as np
vol = np.array([[[0,0,0,0],[0,1,1,0],[0,1,2,0],[0,0,0,0]]], dtype=np.int32)
np.save('/tmp/mvp_smoke_labels.npy', vol)
"
python scripts/simulation/run_mvp_multiphysics_from_config.py \
    --config configs/mvp_multiphysics_example.yaml \
    --label-volume /tmp/mvp_smoke_labels.npy \
    --output-dir /tmp/mvp_config_run
```

### Behavior

1. The config is loaded and validated (requires `label_volume` and `output_dir`).
2. CLI overrides (`--label-volume`, `--output-dir`, `--fail-fast`) take precedence
   over config file values.
3. Resolved config key-value pairs are mapped to CLI flags of the pipeline script:
   - `ga.enabled` → `--skip-ga` (inverted)
   - `ga.mock` → `--ga-mock`
   - `ga.fitness_json` → `--ga-fitness-json`
   - `ga.generations` → `--ga-generations`
   - `ga.population_size` → `--ga-population-size`
   - `ga.mutation_rate` → `--ga-mutation-rate`
   - `ga.mutation_strength` → `--ga-mutation-strength`
   - `ga.elite_fraction` → `--ga-elite-fraction`
   - `ga.tournament_size` → `--ga-tournament-size`
   - `ga.seed` → `--ga-seed`
   - `mcx.enabled` → `--skip-mcx` (inverted)
   - `mcx.mode` → `--mcx-run` (when `"run"`)
   - `mcx.mcx_binary` → `--mcx-binary`
   - `mcx.photons` → `--mcx-photons`
   - `thermal.enabled` → `--skip-thermal` (inverted)
   - `thermal.dt` → `--thermal-dt`
   - `thermal.num_steps` → `--thermal-num-steps`
   - `thermal.source.*` → `--thermal-source-*`
4. In `--dry-run` mode, the resolved command and config snapshot are printed but
   nothing is executed.
5. In run mode, a config snapshot JSON (`mvp_config_snapshot.json`) is written
   into `{output_dir}/manifests/` alongside the pipeline run manifest.
6. The pipeline script `run_mvp_multiphysics_pipeline.py` now accepts additional
   optional CLI arguments for thermal solver parameters and MCX binary path,
   enabling clean parameterisation from the config layer.

### Stage 6 — Integrated Multi-Physics Orchestrator

- `scripts/simulation/run_mvp_multiphysics_pipeline.py` — single-command orchestrator chaining GA + MCX + Thermal.
- Creates subdirs: `ga/`, `mcx/`, `thermal/`, `manifests/` under a single output root.
- Writes `mvp_run_manifest.json` with per-step status, command, returncode, and artifact paths.
- Supports `--fail-fast`, `--skip-ga`, `--skip-mcx`, `--skip-thermal`, `--mcx-run` flags.
- GA defaults to mock mode (--ga-mock) with 5 generations / 8 population for lightweight smoke use.
- MCX defaults to dry-run mode (pass `--mcx-run` to execute with a real MCX binary).
- Thermal solver uses conservative defaults (dt=0.01s, 50 steps, spherical source).

**Example smoke run:**
```bash
# Create a tiny label volume (e.g. 4×4×4)
python -c "
import numpy as np
vol = np.array([[[0,0,0,0],[0,1,1,0],[0,1,2,0],[0,0,0,0]]], dtype=np.int32)
np.save('/tmp/mvp_smoke_labels.npy', vol)
"

# Run the full orchestrator (GA mock + MCX dry-run + thermal)
python scripts/simulation/run_mvp_multiphysics_pipeline.py \
    --label-volume /tmp/mvp_smoke_labels.npy \
    --output-dir /tmp/mvp_smoke_run

# Verify manifest
python -c "
import json
m = json.load(open('/tmp/mvp_smoke_run/manifests/mvp_run_manifest.json'))
print('Status:', m['summary']['has_errors'])
for sid, s in m['steps'].items():
    print(f'  {sid}: {s[\"status\"]}')
"
```

**Thermal-only run:**
```bash
python scripts/simulation/run_mvp_multiphysics_pipeline.py \
    --label-volume /tmp/mvp_smoke_labels.npy \
    --output-dir /tmp/mvp_thermal_only \
    --skip-ga --skip-mcx
```


---

## Acceptance Criteria (Overall)

### This MVP document is satisfied when:

1. **Baseline pipeline** runs end-to-end: data fetch → tile mining → random mask
   generation → training → patch inpainting → merge → 3D stacking → coherence
   metrics.
2. **Dry-run smoke tests** succeed without any external model or dataset.
3. **Real inpainting** produces plausible histology texture inside masks
   (requires SDXL base model + trained LoRA).
4. **Colorimetry metrics** are integrated into the QC output and confirm
   inpainted/non-inpainted colour distribution consistency.
5. **GA optimisation** converges over a single-parameter search space (e.g.,
   mask radius × denoising strength).
6. **MCX export** produces a runnable simulation input from an inpainted volume.
7. **Thermal simulation** solves Pennes bioheat equation on the same volume.

### Non-goals (Explicitly Out of Scope)

- Full 3D diffusion model training (no 3D U-Net, no volumetric diffusion).
- MATRICS-A dataset download (hundreds of GB — excluded by policy).
- Real-time interactive inpainting.
- Deploying a web service or API.

---

## Key File References

| Purpose | Path |
|---------|------|
| Existing 3D inpainting plan | `docs/MVP_3D_INPAINTING_PLAN.md` |
| Original MVP plan (full pipeline) | `MVP_plan.md` |
| DVC pipeline definition | `dvc.yaml` |
| Central configuration | `params.yaml` |
| Runtime configs (auto-generated) | `configs/runtime/*.runtime.yaml` |
| Phase 1 config (tiles + random masks) | `configs/sdxl_lora_phase1_skin_histology_tiles_randommask.yaml` |
| Phase 2 config (tiles + random masks) | `configs/sdxl_lora_phase2_reward_skin_histology_tiles_randommask.yaml` |
| Phase 3 config (tiles + random masks) | `configs/sdxl_lora_phase3_morph_reward_skin_histology_tiles_randommask.yaml` |
| Tile index builder | `scripts/patches/build_tile_index_from_masks.py` |
| Tile materialiser | `scripts/patches/materialize_tile_dataset.py` |
| Random mask generator | `scripts/patches/generate_random_tile_masks.py` |
| Patch extraction | `scripts/patches/extract_roi_patches.py` |
| Patch inpainting | `scripts/patches/inpaint_roi_patches.py` |
| Patch merge | `scripts/patches/merge_inpainted_patches.py` |
| Patch QC | `scripts/patches/qc_patch_replacement.py` |
| Cylindrical mask propagation | `scripts/3d/propagate_mask_across_slices.py` |
| NIfTI stacking | `scripts/3d/stack_slices_to_nifti.py` |
| Z-coherence metrics | `scripts/3d/compute_z_coherence_metrics.py` |
| Synthetic slice generator | `scripts/3d/generate_synthetic_slices.py` |
| Volume inpaint metadata builder | `scripts/3d/build_volume_inpaint_metadata.py` |
| Pipeline orchestrator | `scripts/3d/run_volume_inpaint_pipeline.py` |
| Analysis: run comparison | `scripts/analysis/plot_run_comparison.py` |
| Agent instructions | `AGENTS.md` |

---

*Last updated: 2026-05-17*
