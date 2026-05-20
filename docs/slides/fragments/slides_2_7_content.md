# Slides 2–7: Content Fragment

---

## motivation

**Headline:** Why build synthetic histology variants via SDXL inpainting?

- **Clinical data scarcity** — High-quality 3D histology volumes are rare, expensive, and time-consuming to acquire. Generating synthetic variants from existing 2D slices is a practical path to larger, more diverse datasets.
- **Controllable mask-constrained editing** — Inpaint only inside a user-defined mask region; preserve original tissue everywhere else. The mask is the _ground-truth anchor_ — what you edit is what the mask defines.
- **Multi-physics bridge** — Inpainted volumes feed downstream simulation pipelines:
  - Light transport (MCX Monte Carlo on GPU, PyXOpto MCML on CPU)
  - Biophysical parameter estimation (genetic algorithm on optical properties)
  - Thermal simulation (finite-difference Pennes bioheat equation)
- **Single pipeline, multiple outputs** — The same LoRA inpainting engine serves 2D tile editing, 3D coherent volumes, and simulation-ready property maps.

**Speaker notes:**  
Emphasise that this is fundamentally a _data-side_ project: we are not training a medical classifier; we are generating plausible histological variants from limited source data, then using those variants as input to physics simulation stacks.

---

## dataset-landscape

**Headline:** What data sources are available / in scope?

| Dataset | Type | Size | Labels | Status | Role |
|---------|------|------|--------|--------|------|
| **Histo-Seg** (vccj8mp2cg, v2) | 2D full-slide images + semantic masks | ~150 image/mask pairs | 12-class semantic (epidermis, dermis, BCC, SCC, …) | **Downloaded, processed** | Primary training & evaluation |
| **Zenodo 8155124** (melanoma) | 2D slice stack (cryosection series) | ~700 MB, **66 sequential slices** in full HR stack | Melanoma patches (coarse region annotations) | **Optional** — manual download | 3D volume benchmark, Z-coherence testing |
| **MATRICS-A** | Full 3D block-face histology | Hundreds of GB | Full volumetric histology | **Out of scope for MVP** | Future scale-up |
| **Synthetic circular masks** | Programmatic 2D masks | Configurable count | Binary (tissue / void) | **Generated in pipeline** | MVP fallback when semantic masks unavailable |

### ⚠️ Critical caution: Zenodo slice subsets

The repo contains several **derived subsets** that are _not_ the full 66-slice Zenodo sequence:

- `data/benchmarks/melanoma_3d/original_hr_prepped_tl/slice_*.png` — a **20-slice subset** (images `2_01_a.png` through `2_20_a.png`) padded to a common top-left canvas. High-resolution but **not** the full sequential volume.
- `data/benchmarks/melanoma_3d/slices/slice_*.png` — **legacy low-res stack**. Do not use for new tile inpainting, comparisons, or QC plots.

**Correct workflow:** For true sequential HR analysis, build a fixed-canvas stack from **all 66** raw `cropped_slices/*.png` files in source order. Use `scripts/3d/build_sequential_slice_stack.py` — it validates source order and rejects filtered non-contiguous globs.

**Key constraint:** No large dataset downloads (MATRICS-A) for the MVP phase. Everything must work with Histo-Seg + synthetic masks + optionally the ~700 MB Zenodo benchmark.

**Speaker notes:**  
The dataset landscape is deliberately constrained. We are _not_ trying to match the scale of TCGA or MATRICS-A. The point is to prove the editing-to-simulation bridge works on small curated data. The derived-subset warning is important — if someone grabs the prepped 20-slice stack and runs analysis, they may make sequential-volume claims that don't hold for the full 66-slice sequence.

---

## e2e-flowchart

**Headline:** End-to-end pipeline — data and geometry focus.

```
Histo-Seg pairs   →  Tile mining (512², stride 64)  →  Grad-CAM / random masks
       ↓                          ↓                              ↓
 LoRA Phase 1/2/3        ROI patch extraction             Cylindrical Z propagation
       ↓                          ↓                              ↓
 Inpainted tiles          Merge back to full slice         NIfTI stack + Z-coherence
       ↓                          ↓                              ↓
 Tile-level QC           Full-slice QC                   Volume-level metrics
       ↓                          ↓                              ↓
                MCX light transport · Optical GA · Pennes thermal
```

**Key geometry stages:**

1. **Data acquisition** — Download Histo-Seg manifest, verify version 2, build paired CSV with coarse labels (`A → non_cancer`, `B/C/D → cancer`).
2. **Tile mining** — Slide a 512×512 window across each full-res slice with stride 64. Filter candidates by:
   - Tissue coverage (mask fraction ≥ 15% foreground)
   - White cap (near-white pixels ≤ 30%, threshold 235)
   - Per-image cap (max 250 tiles)
3. **Mask generation** — Two paths:
   - **Grad-CAM** (classifier-driven, requires trained checkpoint)
   - **Random brush masks** (organic strokes, MVP default)
4. **LoRA training** — Three-phase reward-guided pipeline:
   - Phase 1: vanilla fine-tune
   - Phase 2: reward-guided selection (classifier score)
   - Phase 3: morph-aware reward (tissue structure preservation)
5. **Patch inpainting** — Extract ROI patches around mask bbox, inpaint at model resolution (512), merge back feather-stitched. **Strict bbox mode** preferred: validates raw bbox ≤ patch size, zero padding.
6. **Volume assembly** — Stack inpainted slices → NIfTI → Z-coherence metrics (adjacent SSIM, Z-gradient smoothness).
7. **Simulation bridge** — Assign optical/thermal properties per tissue label, run forward models.

**Speaker notes:**  
The flow diagram emphasises data and geometry because the bottleneck is mask quality, not model architecture. Every stage before LoRA training is about getting clean, well-filtered, geometrically consistent tiles. The simulation bridge is what makes this more than an image-editing exercise.

---

## histoseg-first

**Headline:** Why start with the Histo-Seg dataset?

### Rationale

- **Curated skin histology with ground-truth masks** — 12-class semantic segmentation (epidermis, reticular/papillary dermis, BCC, SCC, inflammation, glands, hair follicles, …). The masks are _not_ synthetic — they are manually annotated.
- **Manageable size** — ~150 image/mask pairs. Fits within the no-large-download constraint. Enables rapid iteration on the pipeline without waiting for terabytes of data.
- **Binary coarse labels from filename groups** — Group `A` → `non_cancer`, groups `B`/`C`/`D` → `cancer`. Guarantees two balanced classes for Phase 2/3 reward loops.
- **Immediate tile-mining** — Each full slice (e.g. 5000×5000 px) yields hundreds of 512² tile candidates after filtering. Materialised once, then reused across all three LoRA phases.

### Pipeline integration points

| Stage | Script / Step | What it produces |
|-------|---------------|------------------|
| Pairs | `build_histoseg_pairs_csv.py` | CSV with `image_path`, `mask_path`, `coarse_label`, `slice_id` |
| Tile index | `build_tile_index_from_masks.py` | CSV of 512² tile coordinates filtered by mask coverage + white cap |
| Materialise | `materialize_histoseg_tile_dataset` | Extracted 512² tile PNGs on disk |
| Random masks | `generate_random_masks.py` | Organic brush masks on tiles (MVP fallback) |
| Grad-CAM masks | `build_roi_masks_gradcam.py` | Classifier-saliency masks (when checkpoint available) |

### Why not start with Zenodo or MATRICS-A?

- **Zenodo 8155124** — No semantic masks; only coarse region annotations. The 66-slice HR stack is useful for Z-coherence benchmarking, not for training tissue-aware inpainting.
- **MATRICS-A** — Hundreds of GB, out of scope. The pipeline design is mask-agnostic so that when better data arrives, nothing else changes.

**Speaker notes:**  
HistoSeg is the _proving ground_. The 12-class masks let us evaluate how mask quality affects inpainting output without waiting for a YOLO detector or manual annotations. The coarse binary labels are a simple heuristic that works surprisingly well for reward-guided LoRA training.

---

## tile-geometry

**Headline:** How tiles are extracted, filtered, and oriented.

### Tile mining parameters (from `params.yaml`)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Tile size | 512 × 512 | Matches SDXL working resolution, no downsampling needed |
| Stride | 64 px | Dense coverage: ~94% positional overlap for 5000×5000 slices |
| Min mask fraction | 0.15 (15%) | Reject tiles with insufficient tissue content |
| Max mask fraction | 0.95 | Reject tiles that are nearly 100% mask (likely artefact) |
| Max white fraction | 0.30 | Drop tiles where ≥30% pixels are near-white (empty slide, glare) |
| White threshold | 235 | All three RGB channels ≥ 235 → counted as "white void" |
| Max tiles per image | 250 | Budget cap — prevents a single large slice from dominating |
| Selection mode | random | Stochastic coverage over top-coverage bias |

### Tissue-importance filtering

Tiles are ranked by `mask_coverage` and filtered through a white-void cap, not a tissue-type classifier. This means:

- **Good:** Tiles with moderate tissue content (15–95% mask), low white fraction → kept.
- **Bad:** Tiles that are mostly air, background, or near-white → dropped.
- **Trade-off:** No tissue-class awareness at tile-selection time. A tile of pure dermis and a tile of BCC with the same coverage fraction are equally likely to be selected.

### Epidermis-air interface orientation

For optical simulation, the **epidermis-air normal** must point consistently (toward the top of the output tile). This is handled by:

- `estimate_epidermis_normal.py` — Fits a boundary from the segmentation mask (air = class 0, epidermis = class 1), computes the outward normal vector.
- `select_orient_tile_for_incidence.py` — Rotates and crops a 512×512 tile so the normal points to the tile top edge. No padded canvas; the affine warp samples directly from source coordinates.

**Convention:** The outward normal points from air → epidermis. Rotation aligns this normal to image −y (up).

### Materialisation strategy

1. Build tile _index_ CSV (coordinates only — ~KB).
2. Materialise tiles _once_ into `data/artifacts/tiles/materialized_512/`.
3. All three LoRA phases and mask generation reuse the materialised tiles.
4. Torch datasets in `src/` load from disk on demand.

**Speaker notes:**  
The tile geometry pipeline is deliberately simple and stateless. We do not try to align tiles to anatomical structures at this stage — the orientation step runs later for optical GA. The key insight is that tile filtering is a _rejection_ pipeline: we define clear exclusion criteria (too white, too much mask, too little tissue) and randomise within the pass set.

---

## geometry-limitation

**Headline:** Why geometry-only masks are not sufficient.

### Current gap

| Limitation | Impact |
|-----------|--------|
| **LoRA learns texture inpainting, not semantics** | The model inpaints plausible-looking tissue, but cannot distinguish "should this be dermis or BCC?" inside the mask |
| **Random masks are geometry-only** | They define _where_ to inpaint, but not _what_ tissue class should appear there |
| **Grad-CAM branch faces quality issues** | Classifier saliency maps are noisy; thresholding and dilation are brittle |
| **No per-tissue optical properties in masks** | A geometry-only mask cannot tell the optical GA "this region is melanin-rich" vs "this is collagen" |
| **Z-coherence is measured, not enforced** | After generation we compute SSIM across slices, but the model does not optimise for it |

### Why this matters for physics simulation

- **Optical/thermal sims need per-label property maps.** MCX requires a voxel grid where each label index maps to (μₐ, μₛ, g, n). A geometry mask without class information produces a homogenous phantom — physically unrealistic.
- **Orientation matters for light transport.** The epidermis-air normal angle affects Fresnel reflection. A tile rotated arbitrarily (not aligned to the normal) will give different MCX reflectance than real tissue.
- **Cylindrical mask propagation ignores Z morphology.** Real histology features (e.g., a gland) change shape across slices. An identical (x,y) mask on every slice creates an artificial cylinder.

### Mitigations (pipeline is mask-agnostic by design)

| Mitigation | Status | Description |
|-----------|--------|-------------|
| Semantic masks as preferred GT | **Future** | YOLO detector or manual annotations → per-class masks |
| Phase 2/3 reward loops | **Implemented** | Reward functions guide toward classifier-preferred output |
| Tile filters (tissue ≥ 15%, white ≤ 30%) | **Implemented** | Constrain which tiles are eligible for training |
| Optical GA label bridge | **Implemented** | `build_histoseg_label_volume.py` + `label_to_optical_priors.py` assign priors per label class |
| Affine Z-warping | **Planned** | Relax cylindrical masks via per-slice affine perturbation |
| Z-smoothness reward | **Future** | Add Z-coherence to the LoRA reward function |

### Key insight

> The pipeline is mask-agnostic — better masks directly yield better results.  
> The current geometry-only regime is a starting point, not a ceiling.

Every stage (tile mining, LoRA training, patch extraction, volume stacking) works with any binary mask. Replacing random masks with semantic masks requires _zero pipeline changes_, only better upstream mask generation.

**Speaker notes:**  
This is the most important slide to convey honestly. Geometry-only masks are a pragmatic starting point. The audience should understand that for downstream multi-physics simulation, class-aware masks are essential — and the pipeline is designed to accommodate them with no refactoring. The optical GA bridge already has the scaffolding to convert label volumes to property maps; the missing piece is the masks themselves.
