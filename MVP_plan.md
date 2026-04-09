# MVP Plan: Histology Classifier + GradCAM-Guided 2D Inpainting with 3D-Coherent Volume Reconstruction

## 1. Mission

Build an MVP pipeline for skin histology whole-slide data that does all of the following:

1. trains a patch classifier on Histo-Seg patches,
2. generates class-conditional GradCAM-style ROI masks,
3. uses those masks to drive **small, localized** SDXL inpainting edits on real 2D slices,
4. reconstructs edited 3D histology volumes from slice-wise edits while preserving **slice-to-slice coherence**,
5. outputs an auditable synthetic-augmentation dataset plus quality-control reports.

This plan is designed for the Histo-Seg dataset, which currently contains **38 H&E whole-slide images with corresponding masks across 12 classes**. The public dataset description lists classes spanning skin layers, skin tissues, skin cancers, and background. citeturn779699search0

---

## 2. Why this MVP is structured this way

The core constraint is that the dataset is small and the slides are high-resolution. That makes **patch-based transfer learning** the correct first implementation, rather than training a large WSI model from scratch. Public pathology foundation models such as **UNI** are now strong choices for feature extraction and downstream fine-tuning, and TRIDENT already supports patch/slide processing with public pathology encoders including UNI and CONCH. citeturn609297search0turn779699search3turn779699search11

For GradCAM localization, CNN-style backbones remain the easiest engineering target because Grad-CAM was introduced for CNN feature maps, whereas ViTs and Swin models require target-layer selection and reshape transforms to map token outputs back into spatial grids. The `pytorch-grad-cam` documentation explicitly notes the need for reshape transforms and careful layer choice for Vision Transformers and Swin. citeturn822744search3turn609297search5turn609297search17

For 3D coherence, do **not** edit every slice independently and then stack them naively. The plan below adapts ideas from **temporally consistent video diffusion** and **slice-consistent medical image translation/synthesis** into a practical engineering workflow for histology volumes. Video-diffusion work has shown that explicit consistency guidance improves coherence across frames, and medical-imaging work has similarly emphasized shape and histogram consistency across slices. citeturn645132search5turn645132search4turn645132search6

---

## 3. Primary decisions

### Backbone choices
Implement these models in this order:

1. **UNI** = main classifier backbone and default feature extractor.
2. **ConvNeXt-Tiny** = explainability-first baseline and fallback.
3. **CONCH** = optional auxiliary scorer for later text-label / reward-loop experiments.

Rationale:
- **UNI** is a pathology foundation encoder pretrained on a very large histopathology corpus and is publicly released for research use. citeturn609297search0turn609297search4
- **ConvNeXt-Tiny** is easier to debug for GradCAM and fully open through `timm`. citeturn822744search10turn609297search17
- **CONCH** is relevant later if the reward loop needs image-text alignment because it is a vision-language pathology model trained on **1.17M image-caption pairs**. citeturn779699search2turn779699search6turn779699search22

### Generative stack choices
Use:
- **SDXL inpainting** via Hugging Face Diffusers for the first inpainting MVP. Diffusers provides official SDXL inpainting guidance and an SDXL inpainting model card. citeturn822744search11turn822744search0
- Optional structure guidance via **ControlNet-compatible conditioning** if shape drift is too high. ControlNet was introduced as a way to preserve extra conditioning signals during diffusion generation. citeturn822744search8turn822744search1

### 3D coherence decision
For the MVP, do **not** train a full native 3D diffusion model first. Instead:

- perform **slice-wise 2D inpainting**,
- anchor edits to a small subset of key slices,
- propagate edits outward with **registration-aware constraints**,
- enforce **adjacent-slice latent/style consistency**,
- reject volumes that fail a post-hoc coherence audit.

This is lower-risk and much cheaper than training a full 3D latent diffusion model from scratch. Full 3D medical diffusion is possible, and MONAI provides 3D latent diffusion examples, but that should be a later phase rather than the MVP baseline. citeturn822744search17turn822744search2turn609297search3

---

## 4. Non-negotiable implementation rules

1. **Group all training/validation splits by WSI or by original 3D volume.**
   Never split by patch alone.

2. **Never use synthetic slices in the validation or test sets.**
   Synthetic data is train-only augmentation.

3. **Only make small local edits.**
   Inpainting should alter morphology locally while preserving slide identity, stain style, and global tissue layout.

4. **Keep a full provenance record per synthetic slice.**
   For every edited output, save:
   - original source volume ID,
   - slice index,
   - patch coordinates,
   - source class,
   - target label prompt,
   - mask used,
   - seed,
   - denoise strength,
   - guidance scale,
   - reference/anchor slice ID,
   - coherence metrics.

5. **Reject aggressively.**
   If an edited slice or volume looks plausible but fails quantitative checks, discard it.

---

## 5. Repository deliverables

Create the following deliverables:

```text
repo/
  data/
    raw/
      histoseg_wsi/
      histoseg_masks/
      volumes_3d/
    processed/
      tissue_masks/
      patches/
      patch_labels.parquet
      folds.csv
      synthetic/
        edited_slices/
        edited_volumes/
        metadata.csv
  models/
    classifiers/
      uni/
      convnext_tiny/
    cams/
    generative/
  src/
    data/
      extract_patches.py
      build_patch_labels.py
      build_3d_volumes.py
      sample_training_manifest.py
    train/
      train_classifier.py
      validate_classifier.py
    explain/
      generate_cams.py
      calibrate_cam_thresholds.py
      evaluate_cam_localization.py
    synth/
      build_edit_masks.py
      run_sdxl_inpaint.py
      propagate_slice_edits.py
      reconstruct_volume.py
      qc_volume_consistency.py
      export_synthetic_manifest.py
    utils/
      stain.py
      registration.py
      metrics.py
      prompts.py
  configs/
    data.yaml
    uni.yaml
    convnext.yaml
    synth.yaml
    qc.yaml
  reports/
    classifier/
    cam/
    synthetic/
  MVP_plan.md
```

---

## 6. Phase-by-phase execution plan

## Phase 0 — Environment setup

### Objective
Create a reproducible environment for WSI reading, patch extraction, training, GradCAM, and SDXL inpainting.

### Install core packages
Use a clean Python environment and install:

- `torch`, `torchvision`, `timm`
- `transformers`, `huggingface_hub`
- `pytorch-grad-cam`
- `openslide-python`
- `opencv-python`, `scikit-image`, `scikit-learn`, `pandas`, `numpy`
- `albumentations`
- `diffusers`, `accelerate`, `safetensors`
- `monai` and optionally `monai-generative`
- `SimpleITK` or another registration library
- `trident` if using the Mahmood Lab pipeline directly

OpenSlide is the standard Python library for whole-slide image reading, and TRIDENT is a public toolkit for large-scale WSI processing with support for many pathology encoders. citeturn779699search19turn779699search3

### Acceptance criteria
- Can read a sample WSI.
- Can extract and save patches.
- Can load UNI weights after access approval if required.
- Can run a test SDXL inpainting call on a small dummy patch.

---

## Phase 1 — Data curation and patch dataset creation

### Objective
Turn WSIs and masks into a clean patch classification dataset.

### Steps
1. Read each WSI and corresponding segmentation mask.
2. Select the working magnification:
   - target **20x** or nearest native level,
   - patch size **256x256**,
   - training stride **128**,
   - validation/inference stride **256**.
3. Compute a simple tissue mask and discard low-tissue patches.
4. Assign labels to patches by mask occupancy:
   - if one non-background class occupies at least **60%** of the patch, assign that class,
   - otherwise mark as ambiguous and exclude from classifier training.
5. Also create a binary label:
   - `cancer = {BCC, SCC, IEC}`
   - `non_cancer = remaining tissue classes except background`

The public dataset description lists these cancer-related classes explicitly. citeturn779699search0

### Outputs
- `patch_labels.parquet`
- `folds.csv` with grouped splits by WSI / 3D volume
- summary plots of class counts by fold

### Acceptance criteria
- No patch leakage across train/val folds.
- At least one class histogram and fold report saved.
- Manual visual spot-check confirms label generation is sane.

---

## Phase 2 — Baseline classifier training

### Objective
Train the two initial backbones.

### Models to train
1. `UNI`
2. `ConvNeXt-Tiny`

### Training recipe
For **UNI**:
- Stage A: freeze encoder, train classifier head for 5–10 epochs.
- Stage B: unfreeze last 1–2 transformer blocks, fine-tune 20–30 epochs.
- Optimizer: AdamW
- Head LR: `3e-4`
- Backbone LR: `1e-5`
- Weight decay: `1e-4`
- Batch size: as high as VRAM allows
- Use mixed precision

For **ConvNeXt-Tiny**:
- Stage A: train head 3–5 epochs
- Stage B: unfreeze final stage
- Stage C: optionally full fine-tune if validation remains stable

### Augmentations
Use moderate pathology-safe augmentations:
- horizontal and vertical flips
- 90-degree rotations
- mild brightness/contrast
- mild hue/saturation
- Gaussian blur
- JPEG compression
- stain perturbation

For stain handling, use one of:
- **Macenko normalization** to a fold-specific reference slide, or
- stain augmentation alone if normalization produces artifacts

### Class balancing
Use:
- weighted cross-entropy,
- weighted sampler,
- binary auxiliary head with 50/50 cancer-vs-non-cancer mini-batches.

### Metrics
Report:
- macro-F1
- balanced accuracy
- macro-AUROC
- per-class precision/recall
- cancer-vs-non-cancer AUROC and AUPRC

### Acceptance criteria
- Both models train end-to-end.
- Best checkpoint per fold is saved.
- UNI and ConvNeXt metrics are summarized in a single report.

---

## Phase 3 — GradCAM generation and calibration

### Objective
Turn classifier predictions into class-conditional ROI proposals.

### Method
For **ConvNeXt-Tiny**:
- Use Grad-CAM and HiResCAM on the last convolutional stage.

For **UNI**:
- Use `pytorch-grad-cam` with a **reshape transform**.
- Select a target layer **before** the final transformer block.
- Save both raw and normalized CAM maps.

The Grad-CAM paper localizes influential image regions via gradients flowing into the final convolutional layer, while the `pytorch-grad-cam` ViT/Swin guidance documents the extra reshape-transform step needed for token-based models. citeturn822744search3turn609297search5turn609297search17

### Calibration
1. For each class, compute CAM thresholds on the validation set.
2. Convert CAMs into binary masks.
3. Evaluate against ground-truth segmentation masks using:
   - IoU
   - Dice
   - pointing accuracy
4. Save per-class threshold recommendations.

### CAM selection rule
For the synthetic pipeline, use the classifier/method pair with the best combined score:

`selection_score = 0.6 * macro_F1 + 0.4 * CAM_IoU`

If the project prioritizes localization over classification, flip the weights to `0.4 / 0.6`.

### Acceptance criteria
- Per-class CAM metrics saved.
- Example overlay gallery saved for both UNI and ConvNeXt.
- One model is selected as the synthesis mask generator.

---

## Phase 4 — Build edit masks from CAMs

### Objective
Convert CAMs into safe inpainting masks.

### Rules
1. Use only CAMs from correctly classified patches.
2. Threshold CAM to produce a base mask.
3. Clean the mask:
   - remove tiny connected components,
   - fill small holes,
   - optionally smooth borders slightly.
4. Constrain area:
   - default edit area between **3% and 15%** of patch area,
   - reject if too small or too large.
5. Add a ring or feathered boundary so inpainting blends at the edges.

### Mask categories
Create three edit-mask modes:
- `micro_edit`: very small structure perturbation
- `meso_edit`: moderate gland/keratin/cancer nest perturbation
- `shape_preserve`: same ROI, stronger texture-only perturbation

### Acceptance criteria
- Random visual inspection confirms masks are local and plausible.
- Mask metadata saved for all selected patches.

---

## Phase 5 — SDXL inpainting MVP

### Objective
Generate localized histology edits with minimal structure drift.

### Base model
Use an SDXL inpainting pipeline from Diffusers. Diffusers documents both the general inpainting API and SDXL inpainting usage. citeturn822744search11turn822744search0turn822744search7

### Prompting strategy
Do **not** use artistic prompts.
Use short pathology-style prompts tied to the label taxonomy, for example:

- `"H&E histology, basal cell carcinoma morphology, preserve stain, preserve surrounding tissue architecture"`
- `"H&E histology, squamous cell carcinoma focus, realistic nuclei and keratinization, preserve neighboring tissue"`
- `"H&E histology, mild inflammatory infiltrate, preserve skin layer boundaries"`
- `"H&E histology, glandular structure variation, preserve global architecture"`

Also define negative prompts, for example:
- `"cartoon, artistic, watercolor, oversmoothed, duplicated nuclei, broken tissue, unrealistic stain, synthetic look"`

### Generation settings
Start narrow:
- denoise strength: **0.15–0.35**
- guidance scale: **4–7**
- 20–40 inference steps
- fixed seed per experiment group

### Reference preservation
Pass the original patch as the image input and restrict edits to the binary mask.
Do not allow generation outside the mask.

### Optional structure conditioning
If shape drift remains too high, add structure guidance:
- edge map,
- stain-deconvolved structure map,
- or ControlNet-like conditioning.

ControlNet was introduced to preserve external conditioning during diffusion generation. citeturn822744search8turn822744search1

### Acceptance criteria
- Edited patches remain globally similar to the source patch.
- Local changes are visible inside the mask.
- At least 80% of a 100-sample audit set passes a quick human plausibility screen.

---

## Phase 6 — 3D coherence strategy for slice-wise inpainting

## This is the most important phase for your actual use case.

### Goal
Edit 2D slices one at a time **without** breaking the continuity of the real 3D histological volume.

### Core principle
Treat the slice stack like a short video or ordered volume, not like unrelated 2D images.

Video diffusion literature shows that explicit consistency guidance improves coherence over independently edited frames, and medical slice-based synthesis work has proposed mechanisms to maintain shape and histogram consistency across neighboring slices. This MVP adapts those ideas for histology. citeturn645132search5turn645132search4turn645132search6

### MVP coherence design
Implement the following 6-part strategy:

#### 6.1. Select anchor slices
For each 3D volume:
- identify slices where the target tissue/cancer region is best defined,
- choose one central anchor slice and optionally two secondary anchors.

Selection rule:
- highest classifier confidence,
- strongest CAM localization overlap,
- good tissue quality,
- not too close to volume boundaries.

Only anchor slices are edited freely at first.

#### 6.2. Register neighboring slices to the anchor
Before propagating edits:
- compute rigid or affine registration between adjacent slices,
- optionally refine with light deformable registration if distortions are small and stable.

Use registration to warp:
- CAM masks,
- edit masks,
- and optionally low-frequency appearance guides
from slice `z` to slice `z+1` and `z-1`.

The purpose is simple: corresponding tissue structures should receive related edits across slices whenever they represent the same 3D structure.

#### 6.3. Propagate edit intent, not exact pixels
Do **not** copy the generated patch directly across slices.
Instead propagate:
- the ROI mask,
- target class intent,
- local stain/style statistics,
- and optional latent noise seed schedule.

This avoids obvious copy-paste artifacts while preserving anatomical continuity.

#### 6.4. Use bidirectional slice editing
For each anchor slice:
- edit outward in the `+z` direction,
- edit outward in the `-z` direction,
- then reconcile overlaps.

At slice `z+1`, condition the edit on:
- the real current slice,
- the propagated mask from `z`,
- the previous edited slice as a soft reference,
- and a low-weight consistency penalty to stay near the previous slice’s local style.

This is conceptually similar to temporal consistency guidance in video editing, but applied to ordered histology slices. citeturn645132search5turn645132search1

#### 6.5. Enforce local slice-to-slice consistency metrics
After each generated slice, compute:

- **masked SSIM** between edited slice `z` and registered edited slice `z+1`
- **masked LPIPS or feature distance** in the edited ROI
- **stain histogram distance** in ROI and in a surrounding ring
- **boundary continuity score**: compare the ROI contour position after registration
- **classifier consistency**: neighboring slices should not flip labels erratically

If any metric exceeds tolerance:
- reduce denoise strength,
- shrink mask,
- regenerate from the same reference chain,
- or discard the sequence.

#### 6.6. Final volume reconciliation pass
After editing all candidate slices in the stack:
- smooth only the **edit parameters and latent/style statistics**, not the image directly,
- regenerate any outlier slices using the smoothed neighboring context,
- then run a final QC pass on the whole volume.

This is important because direct image smoothing would damage histological texture.

---

## 7. Concrete 3D coherence implementation recipe

### Recommended slice propagation algorithm

For each volume `V`:

1. detect target slices with confident CAM ROIs,
2. choose anchor slice `z0`,
3. generate edit on `z0`,
4. for `z = z0+1` to `z_max`:
   - register `slice[z-1] -> slice[z]`
   - warp previous ROI mask to current slice
   - derive current edit mask from:
     - warped previous mask,
     - current CAM,
     - current tissue mask
   - run SDXL inpainting with:
     - current original slice as base image
     - current mask
     - same semantic prompt
     - slightly jittered seed
     - reduced denoise strength if change is large
   - compute coherence metrics
   - accept or retry
5. repeat from `z0-1` down to `z_min`
6. reconcile the two directional chains if there are multiple anchors
7. run full-volume QC

### Practical parameter policy
- anchor slice denoise: `0.25–0.35`
- neighbor slice denoise: `0.12–0.22`
- farther slices: denoise decays with distance from anchor
- prompt stays fixed within one volume edit
- seed evolves deterministically with slice index, e.g. `seed(z) = base_seed + z`

This keeps morphology changes correlated across slices without making them identical.

---

## 8. Quality control and rejection pipeline

### Slice-level QC
Reject a synthetic slice if any of these happen:
- classifier confidence drops below threshold for target class,
- CAM after editing no longer localizes inside the intended ROI,
- stain histogram drifts too far from the source slide,
- boundary artifacts are visible at the inpaint edge,
- duplicate-nuclei or repeated texture artifacts appear.

### Volume-level QC
Reject or partially roll back a synthetic volume if:
- neighboring slices show discontinuous lesion motion,
- ROI area changes too abruptly across `z`,
- classifier label oscillates unrealistically across adjacent slices,
- volume fails masked SSIM / feature-similarity tolerance,
- a human reviewer flags it as non-biological.

### Human review
For the MVP, create a simple audit interface that shows:
- original slice,
- CAM,
- edit mask,
- edited slice,
- previous and next slices,
- and a slider to compare original vs edited.

Use it to audit a random sample of accepted and rejected edits.

---

## 9. Synthetic data usage policy

### Safe augmentation policy
Use synthetic examples only for:
- training augmentation,
- classifier robustness experiments,
- reward-loop experiments.

Do **not**:
- mix them into validation or test,
- over-represent one synthetic phenotype,
- create entirely synthetic volumes without provenance.

### Mix ratio
Start with:
- `70–80%` real patches
- `20–30%` synthetic patches

Only increase synthetic proportion if real-only baseline is already stable.

### Curriculum
Use a curriculum:
1. train on real-only,
2. train on real + small local synthetic edits,
3. optionally add harder synthetic cases later.

---

## 10. Experiments to run in order

### Experiment A — classifier benchmark
- UNI vs ConvNeXt-Tiny
- choose best checkpoint by grouped CV

### Experiment B — CAM benchmark
- GradCAM / HiResCAM for ConvNeXt
- transformer CAM for UNI
- select best localization pipeline

### Experiment C — 2D inpainting benchmark
- evaluate prompts, denoise strength, and mask sizes on isolated patches

### Experiment D — 3D coherence benchmark
Compare:
1. naive independent slice editing
2. anchor + registration propagation
3. anchor + registration + bidirectional consistency

The hypothesis is that version 3 will produce the best slice continuity because it explicitly carries context across ordered slices, which is consistent with temporal-consistency ideas in video diffusion and slice-consistency ideas in medical translation. citeturn645132search5turn645132search4

### Experiment E — augmentation utility
Train classifier on:
1. real only
2. real + 2D synthetic
3. real + 3D-coherent synthetic

Compare macro-F1, balanced accuracy, AUROC, and CAM IoU.

---

## 11. Suggested checkpoints, tool links, and licenses

### Classifier / encoder checkpoints
- **UNI** model card and checkpoint: https://huggingface.co/MahmoodLab/UNI  
  Research-use release; model card lists a **CC-BY-NC-ND-4.0** license and Hugging Face access controls. citeturn609297search0
- **CONCH** model card and checkpoint: https://huggingface.co/MahmoodLab/CONCH  
  Pathology vision-language model; model card is public and gated. citeturn779699search2
- **ConvNeXt-Tiny** in `timm`: https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k citeturn609297search17

### WSI / pathology tooling
- **TRIDENT** repo: https://github.com/mahmoodlab/TRIDENT citeturn779699search3
- **OpenSlide Python**: https://openslide.org/api/python/ citeturn779699search19

### Explainability tooling
- **pytorch-grad-cam** repo: https://github.com/jacobgil/pytorch-grad-cam citeturn609297search1
- ViT/Swin CAM notes: https://jacobgil.github.io/pytorch-gradcam-book/vision_transformers.html citeturn609297search5

### Generative tooling
- **SDXL inpainting model card**: https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1 citeturn822744search0
- **Diffusers inpainting guide**: https://huggingface.co/docs/diffusers/using-diffusers/inpaint citeturn822744search11
- **SDXL guide**: https://huggingface.co/docs/diffusers/using-diffusers/sdxl citeturn822744search7
- **MONAI Generative Models**: https://github.com/Project-MONAI/GenerativeModels  
  Public repo under **Apache-2.0**. citeturn822744search2

---

## 12. Acceptance criteria for the whole MVP

The MVP is complete only if all of the following are true:

1. A grouped-CV classifier benchmark exists for UNI and ConvNeXt.
2. A CAM benchmark exists with localization metrics against masks.
3. At least one SDXL inpainting pipeline runs end-to-end on selected patches.
4. The 3D coherence pipeline can reconstruct at least one edited volume from slice-wise edits.
5. Every synthetic output has provenance metadata.
6. A volume-level QC report exists.
7. An augmentation experiment compares:
   - real-only
   - real + 2D synthetic
   - real + 3D-coherent synthetic

---

## 13. What the agent should do first

Execute in this exact order:

1. set up environment,
2. download / verify Histo-Seg files,
3. build grouped patch dataset,
4. train ConvNeXt baseline,
5. train UNI baseline,
6. benchmark CAM localization,
7. select best CAM generator,
8. build inpainting masks,
9. run 2D SDXL inpainting pilot,
10. implement anchor-based 3D slice propagation,
11. run volume QC,
12. run augmentation experiments,
13. write final report with plots and failure cases.

Do not skip directly to synthetic generation before the classifier and CAM pipeline are quantitatively validated.

---

## 14. Final engineering recommendation

For the first MVP release:

- use **UNI** as the main classifier backbone,
- keep **ConvNeXt-Tiny** as the GradCAM sanity-check baseline,
- use **SDXL inpainting** only for **small, local edits**,
- enforce 3D coherence through:
  - anchor slices,
  - registration-based mask propagation,
  - bidirectional slice generation,
  - adjacency-aware coherence metrics,
  - hard rejection of incoherent volumes.

If the slice-wise coherence pipeline works, then the next step is to explore a more native volumetric generator or latent-consistency model. Medical imaging libraries and recent literature show that 3D diffusion is feasible, but for this repo the slice-consistent MVP above is the more practical first system. citeturn822744search17turn645132search6turn609297search3
