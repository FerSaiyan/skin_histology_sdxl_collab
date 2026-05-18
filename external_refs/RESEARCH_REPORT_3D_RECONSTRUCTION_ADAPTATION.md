# Adaptation Report: Extending SDXL Inpainting Pipeline for 3D Reconstruction

**Date:** 2026-05-15  
**Scope:** Assessment of 5 external resources for integration with `skin_histology_sdxl_collab`  
**Prepared for:** orchestrator / agent planning

---

## A) Findings Per Resource

---

### A1. MATRICS-A (HuBMAP Consortium)

**Repo:** `https://github.com/hubmapconsortium/MATRICS-A`  
**Paper:** Ghose et al., "Human Digital Twin: 3D Atlas Reconstruction of Skin…" (bioRxiv 2022, Nature Communications Biology 2023)  
**License:** BSD 2-Clause

**Purpose & Pipeline Stages:**
1. **Segmentation** (`SCRIPTS/Segmentation.sh`):  
   - DAPI nuclei segmentation via deep learning (`hubmap_segmentation/DAPI_Seg2.py`)  
   - GMM-based probabilistic cell-type classification for CD3/4/8/FOXP3/CD68/CD31/AE1/P53/KI67/DDB2  
   - Cell segmentation (connected components filtering with DAPI)  
   - Tissue mask generation via Otsu thresholding + morphological closing/erosion + connected-component hole-filling

2. **Registration** (`SCRIPTS/Registration.sh`):  
   - 2D AF images resampled 16× down (`Resample/resample_AF_2D.py` using `scipy.ndimage.zoom`)  
   - Affine registration via NiftyReg `reg_aladin`  
   - B-Spline non-rigid registration via NiftyReg `reg_f3d`  
   - Transform applied to all biomarker channels  
   - Registered 2D slices stacked into 3D volumes via ITK `TileImageFilter` (2D→3D stacking along Z axis)

3. **3D Coordinate Extraction** (`SCRIPTS/3DCoordinates.sh`):  
   - Connected-component labeling per registered slice  
   - CSV output with 3D centroid coordinates per cell type

**Skin Data Support:** **Direct skin pipeline.** 26 serial sections from 12 donors, 6 anatomical regions. 18-plex immunofluorescence (AF, DAPI, CD31, CD3/4/8, FOXP3, CD68, AE1, P53, KI67, DDB2).

**Data Formats:**  
- Input: 2D TIFF (multi-channel immunofluorescence, 16-bit, pyr16 pyramid)  
- Intermediate: NIfTI (.nii.gz) per slice per channel  
- Output: 3D NIfTI volume per channel, CSV centroid coordinates  

**Reuse potential for this repo:**
- **Registration pipeline** (NiftyReg affine + B-spline) is directly applicable for aligning adjacent histology slices before/after inpainting. This is the most valuable component.  
- **Tissue masking** (Otsu + morphological cleanup + connected components) can be adapted as an alternative mask generation strategy.  
- **Volume stacking** (ITK TileImageFilter) is trivially implementable in Python with numpy/nibabel. No need to build C++ code.  
- **Cell-type GMM classification** is not directly needed (your pipeline is about inpainting, not cell typing), but the cell detection logic could inform downstream metrics (e.g., whether inpainted regions preserve expected nuclear patterns).

**Dependency Stack:**  
- NiftyReg (C++ registration library, included as submodule in `PROG/NiftyReg/`)  
- ITK (for image I/O and tile filter)  
- nibabel + scipy (for Python resampling)  
- TensorFlow/Keras (for DAPI segmentation model)  
- Docker deployment (`hubmap/gehc:skin`)

**Key Files Referenced:**  
- `SCRIPTS/Segmentation.sh` — full 2D segmentation pipeline  
- `SCRIPTS/Registration.sh` — full 2D→3D registration + volume creation  
- `SCRIPTS/3DCoordinates.sh` — 3D coordinate extraction  
- `PROG/CreateVolume/CreateVolume.cxx` — ITK-based 2D slice stacking into 3D NIfTI  
- `PROG/NiftyReg/README.txt` — NiftyReg documentation  
- `PROG/Resample/resample_AF_2D.py` — 16× downsampling via `scipy.ndimage.zoom`  
- `PROG/TissueMask/OtsuThresh.py` — Otsu tissue mask  

---

### A2. vccf-visualization-2022 (HuBMAP Consortium)

**Repo:** `https://github.com/hubmapconsortium/vccf-visualization-2022`  
**License:** MIT (Copyright Yingnan Ju)

**Purpose:** Companion visualization website and analysis scripts for the MATRICS-A skin 3D reconstruction paper. Not a pipeline — a **visualization and analysis toolkit** for the reconstructed 3D skin data.

**Key Components:**
- `visualization/` — Python scripts for:  
  - 3D scatter plots of immune cell positions (`GE_visualization_3D_tcell.py`, `GE_visualization_3D.py`)  
  - 3D immune cluster density plots (`GE_visualization_3D_cluster.py`)  
  - Violin plots for cell-to-vasculature distances (`GE_violin_all_region.py`)  
  - 3D mesh generation from cell coordinates (`mesh2stl.py`)  
  - Interactive Bokeh distance plots (`distance_bokeh.py`)  

- `vitessce/` — OME-TIFF generation for Vitessce viewer integration (`ome_tiff_generator.py`), cell set definitions  

- `utils/` — Distance calculation scripts (`GE_nuclei_vessel_calculate_3d.py`), CSV manipulation, centroid extraction  

- `docs/` — Tissue block metadata spreadsheet, VCCF registration documentation  

**Skin Data Support:** **Directly built for skin visualizations.** Same 12-region dataset.

**Reuse potential for this repo:**
- The **3D plotting and analysis functions** (`GE_visualization_3D*.py`, `mesh2stl.py`, `distance_bokeh.py`) can be adapted for evaluating 3D coherence of inpainted volumes.  
- The **cluster density visualization** approach (heatmap bubbles over 3D cell positions) could be repurposed as a quality metric for inpainting consistency along Z.  
- The **OME-TIFF generation** (`vitessce/ome_tiff_generator.py`) could be useful if you want to load inpainted volumes into a standard viewer.  

**Dependency Stack:** Python (`skimage`, `pandas`, `matplotlib`, `bokeh`, `numpy`, `vitessce`)

---

### A3. CODA (Kiemen Lab, Johns Hopkins)

**Repo:** `https://github.com/ashleylk/CODA`  
**Paper:** Kiemen et al., Nature Methods 2022  
**License:** Not specified (no LICENSE file found). Based on Nature Methods publication, assume academic/non-commercial use.

**Purpose & Pipeline Stages (from `README.txt`):**
1. **Downsampling** (`create_downsampled_tif_images`): Whole-slide NDPI/SVS → 10×/5×/1× TIFF  
2. **Registration** (`calculate_image_registration`, `calculate_tissue_ws`):  
   - Multi-resolution global (affine-like via FFT correlation) + elastic registration via displacement fields  
   - Hierarchical: register adjacent pairs outward from a central reference slice  
   - Outputs: global transforms (`.mat`), elastic displacement fields (`.mat`)  
3. **Segmentation** (`train_image_segmentation`, `apply_image_registration`): DeepLab-based semantic segmentation of tissue structures, then warped via registration transforms  
4. **3D Tissue Volume** (`build_tissue_volume`): Stack registered segmentation maps into 3D volume  
5. **Nuclear Coordinate Detection** (`cell_detection`, `register_cell_coordinates`, `build_cell_volume`):  
   - Hematoxylin channel deconvolution → cell detection → coordinate registration → 3D coordinate matrix  

**Skin Data Support:** **General organ pipeline.** Originally developed for pancreatic cancer tissue. The `train_image_segmentation_lung.m` file indicates lung support. No skin-specific code, but the registration and 3D volume building are organ-agnostic.

**Data Formats:**  
- Input: Whole-slide images (NDPI, SVS), TIFF  
- Intermediate: JPG for registration, MATLAB `.mat` for transforms  
- Output: Registered TIFF/JPG, 3D tissue volume (MATLAB array or NIfTI), cell coordinate CSV  

**Reuse potential for this repo:**
- **Registration approach** is the most directly applicable:  
  - Adjacent-pair registration from center slice outward  
  - Global (FFT correlation) + elastic (displacement field) — similar to NiftyReg's affine + B-spline  
  - The reference image strategy (choose middle slice, register outwards) is optimal for serial section alignment  
- **Cell detection and coordinate tracking** across slices is relevant if you later integrate cell-level metrics.  
- **DeepLab segmentation** approach could be replaced by your existing classifier-based Grad-CAM ROI generation.  
- **MATLAB dependency** is a limitation — it cannot be directly imported into Python pipeline. However, the algorithm descriptions are sufficient for a Python reimplementation.

**Dependency Stack:**  
- MATLAB (core), MATLAB Image Processing Toolbox  
- DeepLab (for semantic segmentation)  
- Base functions in `update 12-13-2023/image registration base functions/` — all `.m` files

**Key Files Referenced:**  
- `update 12-13-2023/calculate_image_registration.m` — main registration function  
- `update 12-13-2023/image registration base functions/calculate_global_reg.m` — FFT correlation  
- `update 12-13-2023/image registration base functions/calculate_elastic_registration.m` — elastic warp  
- `update 12-13-2023/build_tissue_volume.m` — 3D volume construction  
- `update 12-13-2023/build_cell_volume.m` — 3D cell coordinate matrix  

---

### A4. Zenodo Record 7565670 — MATRICS-A Skin 3D Reconstruction Dataset

**DOI:** `10.1101/2022.03.30.486438` (linked to bioRxiv preprint)  
**License:** Data is open-access (no specific license in Zenodo metadata; associated with HuBMAP which uses CC0/CC-BY)  
**Total size:** ~50 GB (13 ZIP files)

**Contents (13 ZIP files):**

| File | Size | Description |
|------|------|-------------|
| `AF.zip` | 4.99 GB | Autofluorescence images (grayscale, used as registration reference) |
| `AE.zip` | 2.44 GB | AE1 (cytokeratin) channel — keratinocytes |
| `DAPI.zip` | 5.35 GB | Nuclear stain |
| `CD31.zip` | 1.13 GB | Blood vessel marker |
| `CD3.zip` | 5.44 GB | Pan T-cell marker |
| `CD4.zip` | 3.32 GB | T helper marker |
| `CD8.zip` | 4.31 GB | T killer marker |
| `CD68.zip` | 3.64 GB | Macrophage marker |
| `FOXP3.zip` | 2.86 GB | T regulatory marker |
| `P53.zip` | 3.13 GB | DNA damage marker |
| `KI67.zip` | 4.17 GB | Proliferation marker |
| `DDB2.zip` | 5.44 GB | DNA repair marker |
| `Original_single_cell_data_for_interactive_plots.zip` | 0.51 GB | Interactive plot data |

**Data format:** 16-bit TIFF pyramids at pyr16 (16× downsampled from 20× original). Single region 007 for the downloadable dataset (26 serial sections, ~20-30 TIFFs per channel).  

**Skin relevance:** **Direct match.** This is skin tissue from 12 donors, but only region 007 is in the open dataset. The full dataset (~314 tissue sections from 12 regions) requires HuBMAP portal access.

**Practical download strategy:**  
- Download only `AF.zip` (registration reference) + one or two marker channels for prototyping (~6-10 GB).  
- The interactive plots zip is small (0.5 GB) and provides pre-computed cell positions.  
- Use Python `requests` with resume capability.  

---

### A5. Zenodo Record 8155124 — Melanoma 3D Histology Model

**DOI:** `10.5281/zenodo.8155124`  
**License:** CC-BY 4.0  
**Total size:** ~699 MB (very manageable)

**Contents:**
- `cropped_slices.zip` — 635 MB: 66 sequential H&E slices of human melanoma, cropped and aligned  
- `3d_model_10pct.nii` — 64 MB: Pre-built 3D NIfTI volume at 10% resolution

**Skin relevance:** **Direct match (melanoma skin cancer).** 66 sequential slices from a melanoma biopsy. This is the only fully downloadable 2D→3D histology dataset with both raw slices and reconstructed volume.

**Practical use:**  
- Download the full 700 MB in ~5 minutes.  
- Use the 66 slices as a **benchmark dataset** for evaluating SDXL inpainting → 3D reconstruction without needing to download the 50 GB MATRICS-A dataset.  
- The pre-built `3d_model_10pct.nii` serves as a **ground truth 3D volume** for evaluating 3D coherence metrics.  
- Both H&E (this dataset) and IF (MATRICS-A) can be used to validate the pipeline on two different staining modalities.  

---

## B) Comparative Matrix

| Feature | MATRICS-A | vccf-visualization-2022 | CODA | Zenodo 7565670 | Zenodo 8155124 |
|---------|-----------|------------------------|------|----------------|----------------|
| **Primary function** | 3D reconstruction pipeline | Visualization & analysis | 3D reconstruction pipeline | Raw skin IF dataset | Raw melanoma H&E dataset |
| **Tissue type** | Skin | Skin | General (pancreas, lung) | Skin (IF, 18-plex) | Skin (melanoma, H&E) |
| **Code language** | C++ (ITK), Python, Bash | Python | MATLAB | N/A (data only) | N/A (data only) |
| **License** | BSD 2-Clause | MIT | Unspecified | Open access | CC-BY 4.0 |
| **Registration method** | NiftyReg (affine + B-spline) | None (uses MATRICS-A output) | FFT correlation + elastic displacement | N/A (raw data) | N/A (pre-aligned) |
| **3D volume format** | NIfTI (.nii.gz) | CSV + HTML (visualization) | MATLAB array | N/A (2D TIFF slices) | NIfTI (.nii) |
| **Segmentation method** | GMM + DAPI deep learning | N/A | DeepLab | N/A | N/A |
| **Skin-specific?** | Yes | Yes | No | Yes (skin IF) | Yes (melanoma) |
| **Download size** | ~50 GB (raw) | N/A (code only) | N/A (code only) | ~50 GB | ~700 MB |
| **Reusable Python code** | Yes (resample scripts) | Yes (visualization) | No (MATLAB) | N/A | N/A |
| **GPU needed?** | Yes (DAPI seg, registration) | No | No (MATLAB CPU) | N/A | N/A |
| **Direct value for inpainting** | Registration + masking | 3D evaluation metrics | Registration algorithm design | Training/eval data | Benchmark data |
| **Slice count** | ~26 per region | 10 regions × 26 | Variable (user data) | 26 (region 007) | 66 |

---

## C) Recommended Integration Architecture

### C.1 Overview

The core idea: extend the existing `skin_histology_sdxl_collab` 2D LoRA inpainting pipeline to **sequential-slice (2.5D)** processing, then to **full 3D reconstruction**. The pipeline becomes:

```
[Raw serial slices] → [Registration/Alignment] → [2.5D tiling + masks] → [SDXL sequential inpainting]
    → [De-tiling] → [3D volume assembly] → [3D coherence evaluation]
```

### C.2 Registration Module (New)

**Source:** CODA strategy + NiftyReg (from MATRICS-A)

```
scripts/registration/
├── register_serial_sections.py      # Main entry point
├── registration_config.yaml          # Parameters
├── feature_based_align.py            # SIFT/ORB feature matching
├── intensity_based_align.py          # Mutual information / NCC
├── niftyreg_wrapper.py               # Python subprocess wrapper for NiftyReg
├── elastic_warp.py                   # B-spline / displacement field warp
└── transforms.py                     # Transform accumulation + inverse mapping
```

**Strategy (from CODA):**
1. Choose the middle slice as reference.  
2. Register adjacent pairs outward: (ref→ref+1), (ref→ref-1), (ref+1→ref+2), etc.  
3. Accumulate transforms.  
4. For elastic refinement: compute displacement fields between globally aligned images.  

**For inpainting:** the registration transforms also define the **mask correspondence** between adjacent slices — if slice N has a Grad-CAM ROI, slice N+1's corresponding ROI can be predicted via the warp field and used as a "target" mask.

### C.3 2.5D Tiling Strategy

**Current state:** 2D tiles extracted independently per slice using `build_tile_index_from_masks.py`.

**Extended to 2.5D:**

```
scripts/patches/
├── extract_serial_tile_blocks.py     # NEW — extracts aligned tile blocks across Z
```

**Algorithm:**
1. Register all slices to a common reference space.
2. Define a grid of tile positions in the reference space.
3. For each tile position, extract the corresponding patch from each registered slice → produces a **tile block** of shape `(Z, H, W, C)`.
4. Generate inpainting masks: either the Grad-CAM mask at each slice, or a **union mask** across the block for consistency.

**Tile overlap strategy (for inter-slice alignment):**
- Tiles must use a **fixed grid** in registered coordinates (not per-slice random offsets).  
- Overlap of 25-50% in XY ensures reconstruction can be blended.  
- In Z, tiles are full-slice height (no Z tiling in early phase).

### C.4 Mask Generation for Adjacent-Slice Consistency

Three strategies, in increasing sophistication:

1. **Independent Grad-CAM** (current): Run existing `build_roi_masks_gradcam.py` on each slice independently. Fast, but masks may jitter between slices.

2. **Warped Grad-CAM** (recommended for Phase 1):  
   - Compute Grad-CAM on reference slice.  
   - Warp mask to adjacent slices using registration transforms.  
   - This ensures the inpainting region moves coherently with tissue.

3. **Propagated Classifier Score** (Phase 2):  
   - Run classifier on all slices.  
   - Use registration to propagate "high-cancer-score" regions across Z.  
   - Mask only regions that are consistently high across a Z block.

### C.5 3D Reconstruction After Slice-Wise Inpainting

Three options:

| Option | Method | Complexity | Output |
|--------|--------|------------|--------|
| **Option A: Simple stack** | Stack inpainted 2D slices with identity Z spacing | Low | NIfTI volume |
| **Option B: Registered stack** | Apply inverse registration transforms, then stack | Medium | Aligned NIfTI volume |
| **Option C: Interpolated volume** | Register + interpolate between slices (cubic/nearest) | Medium | Smooth NIfTI volume |

**Recommendation:** Option B as default, Option C for visualization.  
**Implementation:** `scripts/reconstruction/stack_to_volume.py` using nibabel + scipy.ndimage.

### C.6 Evaluation Metrics for 3D Coherence

| Metric | What it measures | Source | Implementation |
|--------|-----------------|--------|----------------|
| **Slice-to-slice SSIM** | Adjacent slice similarity | Image quality | `skimage.metrics.structural_similarity` along Z |
| **Z-gradient smoothness** | Intensity continuity along Z | Image quality | `numpy.gradient` + L2 norm |
| **Tissue mask IoU overlap** | Segmentations match across slices | Structure | Intersection over union of registered masks |
| **Mutual information (Z)** | Alignment quality | Alignment | `sklearn.metrics.mutual_info_score` |
| **Dice on registered structures** | Whether inpainted regions preserve anatomy | Structure | Compare registered inpainting masks |
| **Cell density continuity** | Cell count per slice should be smooth | Biology | Nuclei detection per slice + Z gradient |
| **Frechet Inception Distance (FID)** | Visual quality per slice | GAN quality | Standard FID between inpainted and real slices |
| **3D structural similarity (3D-SSIM)** | Full volume quality | Volume | Extension of SSIM to 3D patches |

### C.7 Phased Roadmap

#### Quick-Win (1-2 weeks)

1. **Download Zenodo 8155124** (melanoma dataset, ~700 MB). This is your instant testbed.  
2. **Implement basic registration** for the 66 melanoma slices using OpenCV's `findTransformECC` or SIFT + RANSAC.  
3. **Stack into a NIfTI volume** and verify with 3D Slicer / napari.  
4. **Run existing per-slice SDXL inpainting** on the 66 slices, then stack the results poorly (no registration) to establish a baseline.  
5. **Implement slice-to-slice SSIM + Z-gradient** metrics.  

**Commands:**
```bash
# Download melanoma dataset
curl -L -o /tmp/zenodo_8155124.zip https://zenodo.org/records/8155124/files/cropped_slices.zip
mkdir -p data/benchmarks/melanoma_3d/slices
unzip /tmp/zenodo_8155124.zip -d data/benchmarks/melanoma_3d/slices/
```

#### Medium (1-2 months)

1. **Build registration module** (`scripts/registration/`) with:  
   - Feature-based alignment (OpenCV SIFT + RANSAC)  
   - Intensity-based refinement (OpenCV ECC / NiftyReg)  
   - Transform accumulation (cascade for N+1→N→N-1 consistency)  

2. **Extend tiling to 2.5D** (`extract_serial_tile_blocks.py`) with fixed-grid overlap.  

3. **Add mask propagation** (warp Grad-CAM masks between slices).  

4. **Phase 4 pipeline** (`dvc.yaml` stage): `register_slices → extract_tile_blocks → inpaint_blocks → merge_tile_blocks → stack_volume`.  

5. **Implement all 3D evaluation metrics** from C.6.  

6. **Test on Zenodo 7565670** (MATRICS-A skin IF data) — download just AF + one or two channels.  

7. **Run the full loop on Mendeley Histo-Seg dataset** (this repo's primary data). Since Histo-Seg is NOT serial sections (it's random patches), the 2.5D extension will require a **pseudo-serial** strategy:  
   - Group images by `coarse_label` and augment with elastic deformations to create "synthetic serial sections."  
   - Or source a true serial histology dataset (the melanoma dataset fills this gap).

#### Advanced (3-6 months)

1. **JEPA-like latent consistency training**:  
   - Train a latent space where adjacent inpainted slices should have similar representations.  
   - Could be implemented as an additional loss term during LoRA training, or as a post-hoc regularizer.  
   - Based on I-JEPA (Image Joint Embedding Predictive Architecture) — predict one slice's latent from its neighbors.  

2. **Full 3D GAN/DiT inpainting** (replace 2D SDXL with 3D-aware model):  
   - This is a major research effort and should only be pursued after the 2.5D pipeline is solid.  
   - Options: 3D U-Net + diffusion, latent video diffusion (Stable Video Diffusion), or 3D DiTs.

3. **Spatial transcriptomics integration**: If your data ever includes spatial gene expression (Visium, MERFISH), the registered 3D volume becomes a map for molecular inquiry (similar to CODA's Visium integration scripts).

---

## D) Concrete Next Steps & Commands

### Step 1: Download benchmark datasets

```bash
# Melanoma dataset (small, quick)
mkdir -p data/benchmarks/melanoma_3d
python3 -c "
import requests, zipfile, io
r = requests.get('https://zenodo.org/records/8155124/files/cropped_slices.zip', stream=True)
r.raise_for_status()
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall('data/benchmarks/melanoma_3d/slices')
print('Done')
"

# Also fetch the NIfTI volume
curl -L -o data/benchmarks/melanoma_3d/3d_model_10pct.nii https://zenodo.org/records/8155124/files/3d_model_10pct.nii

# MATRICS-A skin IF (just AF channel for registration prototyping, ~5 GB)
curl -L -o data/benchmarks/matrics_skin_af.zip https://zenodo.org/records/7565670/files/AF.zip
mkdir -p data/benchmarks/matrics_skin/af
unzip data/benchmarks/matrics_skin_af.zip -d data/benchmarks/matrics_skin/af/
```

### Step 2: Create registration script skeleton

Create `scripts/registration/register_serial_sections.py` with:
```python
# Register a stack of 2D serial sections into a common coordinate space.
# 1. Load slices (TIFF, PNG, or read from pairs CSV)
# 2. Feature-based alignment (SIFT/ORB) for initial transform
# 3. Intensity-based refinement (ECC) or NiftyReg subprocess call
# 4. Warp all slices to reference
# 5. Save registered slices + transform CSVs
```

### Step 3: Add DVC stages

Add to `dvc.yaml`:
```yaml
  register_melanoma_slices:
    cmd: python scripts/registration/register_serial_sections.py ...
    # ...

  build_3d_volume:
    cmd: python scripts/reconstruction/stack_to_volume.py ...
    # ...

  eval_3d_coherence:
    cmd: python scripts/evaluation/eval_3d_coherence.py ...
    # ...
```

### Step 4: Run baseline inpainting → 3D stack

```bash
# Use existing Phase 1 LoRA to inpaint the 66 melanoma slices independently
python scripts/patches/inpaint_roi_patches.py \
  --metadata-csv data/benchmarks/melanoma_3d/slices_metadata.csv \
  --output-dir data/benchmarks/melanoma_3d/inpainted \
  --base-model <path_to_sdxl> \
  --lora-weights outputs/finetunes/skin_histology_phase1/last.safetensors

# Stack inpainted slices (no registration — baseline)
python scripts/reconstruction/stack_to_volume.py \
  --input-dir data/benchmarks/melanoma_3d/inpainted \
  --output data/benchmarks/melanoma_3d/baseline_volume.nii.gz
```

### Step 5: Implement 3D evaluation

```bash
python scripts/evaluation/eval_3d_coherence.py \
  --volume data/benchmarks/melanoma_3d/baseline_volume.nii.gz \
  --ground-truth data/benchmarks/melanoma_3d/3d_model_10pct.nii \
  --output data/benchmarks/melanoma_3d/baseline_metrics.json
```

---

## E) Open Questions & Risks

### Risks

1. **Serial sections vs. random patches:** The Mendeley Histo-Seg dataset (this repo's primary data) consists of random tissue patches, not serial sections. The 2.5D/3D pipeline can only be validated on the melanoma dataset or MATRICS-A data. Pseudo-serial augmentation (elastic deformations of single patches to simulate serial sections) is a workaround but not a replacement.

2. **No ground truth for Histo-Seg:** There is no 3D ground truth for the primary dataset. Evaluation metrics will be limited to slice-to-slice consistency and visual quality — not structural accuracy.

3. **SDXL inpainting is 2D-native:** The LoRA was trained on 2D slices. Applying it per-slice on registered serial sections may produce temporally inconsistent results (content "pops" between slices). Adjacency-aware conditioning (feeding neighboring slices as context) would require architecture changes.

4. **Registration quality varies:** Serial sections can have warping, tearing, and folding artifacts. No registration algorithm is perfect, and registration errors will compound in the 3D volume.

5. **Computational cost:** Full-resolution 20× histology slices are huge (10k×10k+ pixels). The existing tiling strategy handles this for 2D, but 2.5D tile blocks multiply memory by Z count.

### Unknowns

1. **Is inter-slice consistency actually needed?** For the downstream use case (synthetic slice variation for 3D volume simulation), maybe independent per-slice inpainting is acceptable. The 3D coherence metrics will answer this.

2. **What is the optimal tile size for 2.5D?** Current 512×512 may be too small for meaningful Z context. 1024×1024 may be better but increases GPU memory.

3. **Best adjacent-context strategy:** Should neighboring slices be (a) concatenated as extra channels, (b) used as cross-attention conditioning, or (c) only used for mask propagation? This needs ablative experiments.

### First 3 Experiments

| # | Experiment | Expected Duration | Key Question |
|---|-----------|-------------------|--------------|
| 1 | Register 66 melanoma slices with OpenCV ECC, stack to volume, compute SSIM and Z-gradient | 1-2 days | Is registration alone enough to produce a coherent inpainted-looking volume? |
| 2 | Inpaint each slice independently (current pipeline), then register + stack. Compare 3D metrics to experiment 1. | 2-3 days | How bad is per-slice independent inpainting in 3D? |
| 3 | Implement mask propagation (warp Grad-CAM mask from slice N to N+1), inpaint with propagated mask. Compare to independent mask inpainting. | 3-5 days | Does mask propagation improve Z-consistency of inpainted regions? |

---

## Summary

| Resource | Direct Value | Integration Effort |
|----------|-------------|-------------------|
| MATRICS-A (code) | Registration + masking pipeline | Medium (Python wrapper around NiftyReg) |
| MATRICS-A (data, Zenodo 7565670) | Skin IF serial sections | High (50 GB download, single region) |
| vccf-visualization-2022 | 3D evaluation + visualization scripts | Low (adapt Python scripts directly) |
| CODA | Registration algorithm design | Low (conceptual, MATLAB not reusable) |
| Melanoma dataset (Zenodo 8155124) | **Best testbed** — small, complete, has ground truth | **Low** (700 MB, 66 slices, CC-BY-4.0) |

**Recommendation:** Start with Zenodo 8155124 (melanoma) for rapid prototyping, then graduate to MATRICS-A skin IF data for domain-relevant validation. Commit to the registration module as the first deliverable — without it, the 3D pipeline cannot exist.
