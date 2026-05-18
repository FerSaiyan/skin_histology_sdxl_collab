# MVP Plan: 3D-Coherent Slice Inpainting Pipeline

## Objective / Scope

Extend the existing 2D SDXL LoRA inpainting pipeline to produce **Z-coherent
3D volumes** from ordered histology slice stacks.  The immediate focus is on
**cylindrical mask propagation** (same 2D mask replicated through Z) and
**lightweight coherence metrics** — not full 3D diffusion.

A downstream goal is to support synthetic volume generation for 3D
reconstruction / simulation studies without requiring massive new datasets.

---

## Current Constraint (Explicit)

**No full MATRICS-A dataset download.**  
The MATRICS-A dataset is large (~hundreds of GB).  This plan explicitly avoids
it for now.  All MVP development and testing uses:

- The already-downloaded **Mendeley Histo-Seg** (vccj8mp2cg, v2) dataset
  (full-slide 2D slices, not 3D volumes).
- Optionally the **Zenodo 8155124 melanoma dataset** (small volume benchmark,
  see below) for lightweight 3D volume testing if slice-stack information
  is present.

### Zenodo 8155124 (Optional Small Benchmark)

**Identifier:** `10.5281/zenodo.8155124`  
**Content:** Melanoma histology image patches with annotations.  
**Size:** ~700 MB total — `cropped_slices.zip` (~635 MB) + `3d_model_10pct.nii` (~64 MB).
  This is a small, focused volume benchmark and does **not** trigger the
  no-large-download policy (which targets MATRICS-A at hundreds of GB).  
**Command (manual, not automated):**

```bash
# Download manually from https://zenodo.org/records/8155124
# or via:
# wget https://zenodo.org/records/8155124/files/cropped_slices.zip
# wget https://zenodo.org/records/8155124/files/3d_model_10pct.nii
unzip cropped_slices.zip -d data/raw/zenodo_melanoma/
```

This is purely optional — the MVP scripts work on any ordered PNG/JPEG stack.

> **Note on external references:** This plan and its sibling scripts may
> occasionally mention "oral-lesions" paths or notebooks for comparison.
> Those references are legacy/contextual only — the `skin_histology_sdxl_collab`
> repo is fully standalone and does not require the oral-lesions repository.

---

## Cylindrical Mask Strategy

For the MVP, mask propagation across Z is deliberately simple:

1. **Take a single 2D binary mask** (e.g., from Grad-CAM, from a manual ROI,
   or a synthetic circular mask).
2. **Replicate it identically** for all slices in a volume.
3. **Use as inpainting condition** for each slice independently during
   SDXL LoRA generation.

This means the inpainted region is a **cylinder** in the 3D volume — same
(x,y) footprint on every slice.

### Why this approach?

- Zero additional complexity in the diffusion step.
- The inpainting model still learns slice-local textures inside the mask.
- After generation, Z-coherence can be measured.
- Future work can relax the rigid propagation (e.g., affine-warp the mask
  between adjacent slices, or train a YOLO detector per slice).

---

## Phased Roadmap

### Phase A — Quick Win (This Implementation)

| Task | Script | What it does |
|------|--------|-------------|
| A1 | `scripts/3d/propagate_mask_across_slices.py` | Replicate a 2D mask across N slices, or generate a centered circular/elliptical test mask. |
| A2 | `scripts/3d/stack_slices_to_nifti.py` | Stack ordered 2D images (JPEG/PNG) into a NIfTI volume (`.nii.gz`). |
| A3 | `scripts/3d/compute_z_coherence_metrics.py` | Compute adjacent-slice SSIM and Z-gradient smoothness from a NIfTI volume. |

**Acceptance criteria:**
- Each script runs with `--help` and produces valid output.
- A circular mask can be generated across 10 slices, stacked into a NIfTI,
  and coherence metrics computed on the resulting volume.
- All paths are local to this repo.

### Phase B — Workflow Integration (Phase A + B + B.1 Implemented)

| Task | Script | What it does |
|------|--------|-------------|
| B1 | `scripts/3d/generate_synthetic_slices.py` | Generate synthetic test slice images (gradient/noise/checkerboard) for smoke testing. |
| B2 | `scripts/3d/build_volume_inpaint_metadata.py` | Build metadata CSV from slice glob + mask directory, compatible with `inpaint_roi_patches.py`. |
| B3 | `scripts/3d/run_volume_inpaint_pipeline.py` | Full tile-first orchestration: mask gen → pairs CSV → extraction → inpainting → merge → NIfTI stack → coherence → manifest. |
| B4 | DVC stages `generate_3d_synthetic_slices` + `run_3d_volume_inpaint_smoke` + `run_3d_volume_inpaint_tile_smoke` | Optional self-contained stages in `dvc.yaml`. |

### Phase B.1 — Tile-First Volume Inpainting (High-Res Slices)

**Rationale:** Full high-res histology slices (e.g., 5000×5000 px) cannot be
naively resized to 512×512 for inpainting — tissue architecture is lost.
Phase B.1 replaces the old full-slice-resize flow with a tile/patch-first flow:

1. Generate cylindrical masks at **full-slice resolution**.
2. Extract a local **ROI patch** around the mask (configurable `--patch-size`).
3. Inpaint the patch at **model resolution** (`--target-size`, default 512).
4. Merge the inpainted patch back into the original full-res slice.
5. Stack edited slices into a NIfTI volume and measure Z-coherence.

### Geometry Guidance for Tile-First Flow

There are **two containment modes**, selected by `--strict-bbox`:

#### Strict BBox Mode (`--strict-bbox`, preferred for B.1)

The strict mode eliminates padding and square transforms entirely:

**Rule:** raw mask bounding box width AND height must each be ≤ `--patch-size` (no padding, no square transform).

```
raw_bbox_w ≤ patch_size  AND  raw_bbox_h ≤ patch_size
```

- `--padding-ratio` is **ignored** (forced to 0.0).
- `--patch-size` defaults to **512** when `--strict-bbox` is set without explicit value.
- Extraction uses `crop-mode=strict_fixed_raw`: center a `patch_size × patch_size` tile on the raw bbox center, clamp to image boundaries (no zero-padding).
- **Use this mode when your masks are small enough that the raw bbox fits in 512×512.** For such cases, it guarantees clean patches with no synthetic padding.

**Example (512×512 tiles for a 256×256 raw bbox):**

```
slice_dim = 5000×5000, mask_radius = 0.05
→ mask_diameter ≈ 250 px
→ raw bbox ≈ 250×250 ≤ 512  ✓
```

#### Non-Strict Mode (legacy, padding + square)

The key constraint is that the **mask bounding box (after padding + square) must fit within the extraction patch**:

```
mask diameter  →  bbox + padding  →  make square  →  patch_size
```

Default values:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--mask-radius` | 0.3 | Fraction of smaller slice dimension |
| `--padding-ratio` | 0.15 | Context margin as fraction of bbox |
| `--patch-size` | 1024 | Full-res crop (must cover mask + padding) |
| `--target-size` | 512 | Model working resolution |

**Rule of thumb:** `mask_diameter_px ≤ 0.6 × patch_size`

For a 5000×5000 slice with `--mask-radius 0.3`:
- mask diameter ≈ 1500 px
- with padding 0.15: ≈ 1725 px
- squared: ≈ 1725 px
- patch_size needed: ≥ 1725 px (default 1024 is too small!)

So for large slices, use either a smaller mask or larger patch.  If you get
a "Mask bbox exceeds configured patch_size" error, the diagnostics message
will suggest specific values.

**The orchestrator validates mask containment before extraction and errors
out with a helpful message and parameter suggestions if the rule is violated.**

### Tissue-Aware Mask Placement (`--tissue-aware-mask`)

By default, the cylindrical mask is centered on the slice (or at a manually
specified `--center-x`/`--center-y`).  With **`--tissue-aware-mask`**, the
pipeline selects a mask center that lies inside **tissue-valid regions common
to all sequential slices**, avoiding white/black voids.

**How it works:**
1. Each slice is classified pixel-wise: a pixel is "white void" if all RGB
   channels ≥ `--white-threshold` (default 235), "black void" if all channels
   ≤ `--black-threshold` (default 10), and "tissue-valid" otherwise.
2. The common intersection of tissue-valid pixels across all N slices is
   computed.
3. A random center is sampled from this intersection (controlled by
   `--mask-center-seed`, default 42).
4. The cylindrical mask centered at this point is validated:
   - raw mask bbox ≤ `--patch-size`
   - tissue overlap fraction ≥ `--min-mask-tissue-overlap` (default 0.85)
   - black-void fraction ≤ `--max-mask-black-frac` (default 0.10)
   - encompassing tile (centered and clamped) is not entirely void
5. Up to `--mask-center-max-attempts` (default 50) random samples are tried
   until all constraints pass.

**CLI options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--tissue-aware-mask` | (flag) | Enable tissue-aware center selection |
| `--white-threshold` | 235 | White-void threshold (all channels ≥ this) |
| `--black-threshold` | 10 | Black-void threshold (all channels ≤ this) |
| `--min-mask-tissue-overlap` | 0.85 | Min fraction of mask in tissue per slice |
| `--max-mask-black-frac` | 0.10 | Max black-void fraction inside mask |
| `--mask-center-seed` | 42 | RNG seed for center sampling |
| `--mask-center-max-attempts` | 50 | Max retries to find valid center |

**Important:** For very sparse-tissue images (e.g., tiny tissue islands on a
black background), the defaults may be too strict.  Adjust `--black-threshold`
downward (e.g., to 0) to treat any non-zero pixel as tissue, and lower
`--min-mask-tissue-overlap` to match the achievable overlap given the mask size
vs. tissue area.

**Output:** When enabled, the pipeline saves `tissue_aware_mask_diagnostics.json`
and `.csv` in the run directory, containing per-slice tissue/white/black overlap
fractions inside the mask.

**Backward compatibility:** Without `--tissue-aware-mask`, behavior is
unchanged (centered geometric mask).

### Phase B.1 Quick Smoke Test

```bash
# Generate synthetic slices (256x256 is a low-res volume for testing)
python scripts/3d/generate_synthetic_slices.py \
  --num-slices 10 --height 256 --width 256 --pattern gradient \
  --output-dir /tmp/test_tile_slices

# Run tile-first pipeline in dry-run mode
python scripts/3d/run_volume_inpaint_pipeline.py \
  --slice-glob "/tmp/test_tile_slices/slice_*.png" \
  --volume-id tile_smoke \
  --coarse-label non_cancer \
  --num-slices 10 \
  --mask-radius 0.3 \
  --patch-size 256 \
  --target-size 128 \
  --padding-ratio 0.1 \
  --output-dir /tmp/test_tile_pipeline \
  --dry-run
```

Expected: all 8 steps complete; mask generation, pairs CSV, stacking, and
coherence run in full; extraction/inpainting/merge report dry-run status.

**Real inpainting run (requires model paths):**
```bash
python scripts/3d/run_volume_inpaint_pipeline.py \
  --slice-glob "data/artifacts/3d/synthetic_slices/slice_*.png" \
  --volume-id my_volume \
  --coarse-label non_cancer \
  --num-slices 10 \
  --mask-radius 0.3 \
  --patch-size 1024 \
  --target-size 512 \
  --padding-ratio 0.15 \
  --output-dir data/artifacts/3d/volume_inpaint_runs \
  --base-model /path/to/sd_xl_base_1.0.safetensors \
  --lora-weights /path/to/lora.safetensors
```

**DVC:**
```bash
dvc repro generate_3d_synthetic_slices
dvc repro run_3d_volume_inpaint_smoke
dvc repro run_3d_volume_inpaint_tile_smoke
```

### Phase C — Advanced: Z-Coherence Hardening

- Reject or re-sample inpainted volumes where Z-coherence metrics fall
  below a threshold.
- Add mask-warping between adjacent slices (affine transform with small
  random perturbation) rather than identical replication.
- Add Z-smoothness regularisation to the reward function in Phase 2/3 loops.

### Phase D — Future: YOLO-Based Detector (512×512 Tiles)

**Goal:** Replace manual / Grad-CAM mask sources with an object detector that
identifies histological features (veins, tumors, glands, inflammation zones)
on 512×512 tiles.

**Plan:**
1. Annotate or source bounding-box labels for features of interest.
2. Train a **YOLOv8** (or YOLOv11) detector on 512×512 tiles.
3. Run detector inference per slice to produce instance segmentation masks.
4. Use detector-derived masks as the inpainting condition (instead of or in
   addition to Grad-CAM masks).
5. Propagate detector masks through Z using the same cylindrical strategy,
   or with per-slice detection for varying masks.

**Benefits:**
- Feature-specific inpainting (inpaint only tumor regions, preserve veins).
- Multi-class masks enable conditional inpainting (diffusion model sees
  which feature type to generate).
- Detector can run on new volumes without needing the classifier pipeline.

**Implementation checklist (when ready):**
1. `scripts/detection/prepare_yolo_dataset.py` — convert mask pairs to YOLO
   format.
2. `scripts/detection/train_yolo_detector.py` — train on 512×512 tiles.
3. `scripts/detection/run_yolo_to_masks.py` — inference → binary masks.
4. Update `propagate_mask_across_slices.py` to accept per-slice masks from
   detector output.

---

## Implementation Checklist (Concrete File Targets)

### New files to create

| File | Purpose |
|------|---------|
| `docs/MVP_3D_INPAINTING_PLAN.md` | This plan |
| `scripts/3d/__init__.py` | Package marker |
| `scripts/3d/propagate_mask_across_slices.py` | Cylindrical mask replication |
| `scripts/3d/stack_slices_to_nifti.py` | 2D → NIfTI stacking |
| `scripts/3d/compute_z_coherence_metrics.py` | SSIM + Z-gradient metrics |
| `scripts/3d/generate_synthetic_slices.py` | Synthetic test slice images (Phase B) |
| `scripts/3d/build_volume_inpaint_metadata.py` | Metadata CSV adapter (Phase B) |
| `scripts/3d/run_volume_inpaint_pipeline.py` | Pipeline orchestrator (Phase B) |

### Files to modify

| File | Change |
|------|--------|
| `README.md` | Add "3D Coherence Track" section with commands; add "Phase B — Volume Inpainting Pipeline" section |
| `AGENTS.md` | Add guidance for 3D MVP and no-large-download policy; update DVC stage list and health checks |
| `MVP_plan.md` | Cross-reference 3D plan / update scope |
| `params.yaml` | Add `volume_pipeline` configuration block |
| `dvc.yaml` | (Optional) Add lightweight 3D stage stubs + Phase B volume inpainting stages |

---

## Acceptance Criteria & Stop Conditions

### This implementation is complete when:

1. `python scripts/3d/propagate_mask_across_slices.py --help` prints usage.
2. `python scripts/3d/stack_slices_to_nifti.py --help` prints usage.
3. `python scripts/3d/compute_z_coherence_metrics.py --help` prints usage.
4. Running the following end-to-end test succeeds:

```bash
# Generate 10 circular masks
python scripts/3d/propagate_mask_across_slices.py \
  --num-slices 10 --height 256 --width 256 --radius 0.3 \
  --output-dir /tmp/test_cyl_masks

# Stack into NIfTI
python scripts/3d/stack_slices_to_nifti.py \
  --input-glob "/tmp/test_cyl_masks/mask_slice_*.png" \
  --output-nifti /tmp/test_cyl_volume.nii.gz

# Compute coherence metrics
python scripts/3d/compute_z_coherence_metrics.py \
  --volume-nifti /tmp/test_cyl_volume.nii.gz \
  --output-json /tmp/test_cyl_metrics.json
```

5. Metrics JSON contains `adjacent_ssim` and `z_gradient_smoothness` blocks.
6. All output paths are under `data/` or `/tmp/` — no system modifications.

### Phase B acceptance criteria (additional):

7. `python scripts/3d/generate_synthetic_slices.py --help` prints usage.
8. `python scripts/3d/build_volume_inpaint_metadata.py --help` prints usage.
9. `python scripts/3d/run_volume_inpaint_pipeline.py --help` prints usage.
10. Running the Phase B smoke flow (below) succeeds with dry-run mode:
    - generates synthetic slices,
    - generates cylindrical masks,
    - builds metadata CSV,
    - runs inpaint dry-run,
    - stacks inpainted (or placeholder) slices into NIfTI,
    - computes coherence metrics,
    - saves run manifest JSON.
11. Run manifest contains all 5 steps with status values.
12. Coherence metrics JSON contains `adjacent_ssim` and `z_gradient_smoothness` blocks.

### Stop conditions (do NOT proceed if):

- Any script requires a model download or dataset >100 MB to test.
- Any script modifies existing DVC stages.
- Any script introduces a new external dependency not in `requirements.txt`.
- The plan file exceeds 50 KB (keep it concise).

---

## File Targets (Detailed)

### `scripts/3d/propagate_mask_across_slices.py`

```
Usage:
  python scripts/3d/propagate_mask_across_slices.py \
    --mask path/to/reference_mask.png \
    --num-slices 20 \
    --output-dir data/artifacts/3d/propagated_masks

  # Or generate centered circular mask:
  python scripts/3d/propagate_mask_across_slices.py \
    --height 512 --width 512 --num-slices 10 --radius 0.3 \
    --output-dir data/artifacts/3d/propagated_masks

Output:
  - mask_slice_0000.png ... mask_slice_0009.png  (binary PNG, same content)
  - stats JSON with foreground fraction and dimensions
```

### `scripts/3d/stack_slices_to_nifti.py`

```
Usage:
  python scripts/3d/stack_slices_to_nifti.py \
    --input-glob "data/artifacts/3d/propagated_masks/mask_slice_*.png" \
    --output-nifti data/artifacts/3d/volumes/mask_volume.nii.gz

Output:
  - NIfTI file (.nii.gz), shape (H, W, Z)
  - Optional stats JSON
```

### `scripts/3d/compute_z_coherence_metrics.py`

```
Usage:
  python scripts/3d/compute_z_coherence_metrics.py \
    --volume-nifti data/artifacts/3d/volumes/mask_volume.nii.gz \
    --output-json data/artifacts/3d/metrics/coherence.json

Metrics:
  - adjacent_ssim: per-slice-pair SSIM, also aggregated (mean, std, min, max)
  - z_gradient_smoothness: mean absolute intensity gradient between slices
```

---

## Dependencies

All scripts use only:
- `numpy` (already in `requirements.txt`)
- `Pillow` (already in `requirements.txt`)
- `nibabel` (for NIfTI I/O — add to `requirements.txt` if not present)
- `scipy.ndimage` (for SSIM Gaussian filter — `scipy` already in env)

No new large downloads, no external model files.

---

## DVC Stage Stubs (Optional, Low-Risk)

If added to `dvc.yaml`, the new stages would look like:

```yaml
  # --- 3D Coherence Pipeline (optional) ---

  generate_3d_test_masks:
    cmd: python scripts/3d/propagate_mask_across_slices.py --num-slices 10 --height 256 --width 256 --radius 0.3 --output-dir data/artifacts/3d/test_masks --stats-json data/artifacts/3d/test_masks_stats.json
    outs:
      - data/artifacts/3d/test_masks:
          cache: false
      - data/artifacts/3d/test_masks_stats.json

  stack_3d_test_volume:
    cmd: python scripts/3d/stack_slices_to_nifti.py --input-glob "data/artifacts/3d/test_masks/mask_slice_*.png" --output-nifti data/artifacts/3d/test_volume.nii.gz --stats-json data/artifacts/3d/test_volume_stats.json
    deps:
      - data/artifacts/3d/test_masks
    outs:
      - data/artifacts/3d/test_volume.nii.gz
      - data/artifacts/3d/test_volume_stats.json

  compute_3d_test_coherence:
    cmd: python scripts/3d/compute_z_coherence_metrics.py --volume-nifti data/artifacts/3d/test_volume.nii.gz --output-json data/artifacts/3d/test_coherence.json
    deps:
      - data/artifacts/3d/test_volume.nii.gz
    outs:
      - data/artifacts/3d/test_coherence.json
```

These stages are fully self-contained (no external data needed), run in
~1 second, and are safe to add.

---

## Next Steps (Immediate Execution Checklist)

The strict bbox mode (`--strict-bbox`) is now the recommended default for
Phase B.1 tile-first volume inpainting.  The following tasks are ready to
execute:

- [ ] **Run dry-run smoke test with strict mode** to verify the flow:
  ```bash
  python scripts/3d/generate_synthetic_slices.py \
    --num-slices 10 --height 256 --width 256 --pattern gradient \
    --output-dir /tmp/test_strict_slices

  python scripts/3d/run_volume_inpaint_pipeline.py \
    --slice-glob "/tmp/test_strict_slices/slice_*.png" \
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

- [ ] **Verify strict validation catches oversized masks:**
  ```bash
  # Create a mask with bbox > 128x128 (radius 0.4 on 256x256 slice = ~102px diameter
  # but we use patch-size 64 to trigger fail)
  python scripts/3d/run_volume_inpaint_pipeline.py \
    --slice-glob "/tmp/test_strict_slices/slice_*.png" \
    --volume-id strict_fail_test \
    --coarse-label non_cancer \
    --num-slices 5 \
    --mask-radius 0.35 \
    --strict-bbox \
    --patch-size 64 \
    --target-size 64 \
    --output-dir /tmp/test_strict_fail \
    --dry-run 2>&1 | grep -i "FAILED\|Error\|exceed"
  # Expected: validation error about raw bbox > patch-size
  ```

- [ ] **Update DVC stage `run_3d_volume_inpaint_tile_smoke`** to use `--strict-bbox` by default.

- [ ] **Benchmark real histology slices** with small masks (bbox ≤ 512×512)
      and run a full inpainting pass with `--strict-bbox`.

- [ ] **Document mask generation strategy** so users know to size their masks
      such that raw bbox ≤ 512×512 (or the chosen `--patch-size`).

### Known Caveats (Strict Mode)

1. **Boundary clamping** — If the centered tile extends beyond the image
   edge, the extracted patch is smaller than `patch_size` on the clamped
   side.  This is acceptable (no zero-padding), but downstream resize fills
   to `target_size × target_size`.  The actual crop dimensions are recorded
   in metadata (`bbox_height`, `bbox_width`).
2. **No padding = less context** — The model sees exactly the mask region
   plus surrounding tissue within `patch_size`.  If the mask touches the
   edge of the tile, the model has less boundary context.  This is the
   trade-off for avoiding synthetic padding.
3. **Mask sizing responsibility** — The user/upstream mask generator must
   ensure raw bbox ≤ `patch_size`.  The pipeline rejects oversized masks
   with a clear error.
4. **Downstream merge compatibility** — The merge script reads actual
   bbox dimensions from metadata and resizes inpainted output accordingly.
   Variable-sized patches are handled correctly.
