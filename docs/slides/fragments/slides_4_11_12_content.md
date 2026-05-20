# Slide Content Fragments — Slides 4, 11, 12

---

## e2e-flowchart

### Improved Lane Labels & Node Text

**Layout:** Five horizontal swimlanes, left-to-right flow. Each lane has 3–5 nodes with connecting arrows. A vertical timeline bar on the left shows "MVP → Production" progression.

---

#### Lane 1: Data
| Node | Label | Detail |
|------|-------|--------|
| **N1** | Histo-Seg Acquisition | Mendeley vccj8mp2cg, v2. ~150 image+mask pairs. Fits < 1 GB. |
| **N2** | Pairs CSV | `build_histoseg_pairs_csv.py`. Groups filenames A → non_cancer, B/C/D → cancer. |
| **N3** | Tile Mining | 512² tiles, stride 64. Filters: tissue ≥ 15 %, white cap ≤ 30 %. Max 250 tiles per source. |
| **N4** | HR Source Guardrail | For melanoma / Zenodo stacks: verify `run_manifest.json` points to `cropped_slices/*.png`, not legacy `melanoma_3d/slices/`. See AGENTS.md. |

*Speaker hint:* Emphasise that the data lane is deliberately small and constrained — we are not chasing large datasets. The HR guardrail is a practical check against mixing low-res and high-res sources during 3D runs.

---

#### Lane 2: Geometry

| Node | Label | Detail |
|------|-------|--------|
| **N1** | Mask Sources | Three options: Grad-CAM ROI (from classifier), random organic brush (MVP fallback), tissue-aware center selection (common tissue intersection across Z). |
| **N2** | Cylindrical Propagation | Identical 2D mask replicated through all Z slices. MVP simplicity; no mask deformation between adjacent slices. |
| **N3** | Strict BBox Validation | `--strict-bbox`: raw mask bbox width AND height ≤ patch_size. Rejects oversized masks with diagnostic error. No padding, no square transform. |
| **N4** | Patch Extraction | `extract_roi_patches.py` — crop-mode `strict_fixed_raw` or `fixed`. Clamps to image boundaries (no zero-padding). |

*Speaker hint:* The geometry lane is the anchor for all editing — mask is ground truth. Strict bbox mode is the recommended default for Phase B.1. It eliminates synthetic padding artifacts at the cost of requiring small masks.

---

#### Lane 3: Synthesis (SDXL LoRA Inpainting)

| Node | Label | Detail |
|------|-------|--------|
| **N1** | Phase 1 — Vanilla LoRA | SDXL LoRA fine-tune on masked tile pairs. Classifier-free guidance, default loss. |
| **N2** | Phase 2 — Reward Guided | `phase2_reward_guided_lora.py`. Reward score from histology classifier selects better checkpoints. |
| **N3** | Phase 3 — Morph Reward | `phase3_morph_reward_guided_lora.py`. Morphological diversity term added to reward. |
| **N4** | Patch Inpainting | Run trained LoRA on extracted ROI patches at target_size (e.g. 512). Strength 0.55, steps 40. |
| **N5** | Merge & QC | `merge_inpainted_patches.py` — feather-stitch patches back into full-res slice. QC: validate seam diff, compute CIELAB ΔE*. |

*Speaker hint:* The three-phase LoRA loop is the core research contribution. Phase 2/3 incorporate classifier feedback during training. The patch inpainting step is where the trained model actually edits tissue.

---

#### Lane 4: Physics (Simulation Bridge)

| Node | Label | Detail |
|------|-------|--------|
| **N1** | Volume Stacking | `stack_slices_to_nifti.py` — ordered 2D PNG/JPEG → NIfTI (.nii.gz). |
| **N2** | Optical GA | Genetic algorithm estimating 19 biophysical parameters (melanin, blood, O₂ sat, scattering, thickness). Forward model: surrogate (lightweight) or MCX/PyXOpto (realistic). |
| **N3** | MCX Light Transport | GPU-accelerated Monte Carlo. Requires binary voxel volume + JSON config. |
| **N4** | Thermal Pennes Solver | Finite-difference bioheat equation. Label-dependent ρ, c, k, w_b, q_met. Source modes: none / spherical. |

*Speaker hint:* The physics lane turns an inpainted volume into simulation inputs. Only the surrogate GA and thermal solver are provenanced today. MCX is scaffolded but not yet validated end-to-end.

---

#### Lane 5: Validation

| Node | Label | Detail |
|------|-------|--------|
| **N1** | Z-Coherence Metrics | Adjacent-slice SSIM, Z-gradient smoothness. Aggregated: mean, std, min, max across slice pairs. |
| **N2** | Sequential Integrity Check | Verify slice IDs are contiguous in original Zenodo order. Use `build_sequential_slice_stack.py` — rejects non-contiguous globs (e.g. `*_a.png`). |
| **N3** | HR Source Provenance | Run manifest records `steps.pairs_csv.image_dir`. QC plots flag if path contains `melanoma_3d/slices/` (legacy low-res). |
| **N4** | Tissue-Aware Diagnostics | When `--tissue-aware-mask` is enabled: per-slice tissue/white/black overlap fractions saved to JSON + CSV. |
| **N5** | Comparative Analysis | `plot_run_comparison.py` — scatter/box plots comparing strict vs non-strict, tissue-aware vs centered, HR vs low-res. |

*Speaker hint:* Validation is under-emphasised in current slides — this lane makes it explicit. The sequential integrity check and HR source guardrail directly address concerns about mixing up resolution or slice ordering during 3D benchmarks.

---

### Speaker Notes (Slide 4, Full)

"This pipeline has five lanes, read left to right.

**Data** — We start with the Histo-Seg dataset, build pairs CSV with coarse binary labels, mine 512×512 tiles, and always verify our HR source paths before 3D runs.

**Geometry** — Masks come from Grad-CAM, random brushes, or tissue-aware selection. We propagate them identically through Z (cylindrical). Strict bbox mode validates that raw mask dimensions fit within the extraction patch — no synthetic padding.

**Synthesis** — Three-phase LoRA loop: vanilla, reward-guided, morph reward. Then patch extraction, SDXL inpainting at model resolution, and feather-stitch merging back into the full slice.

**Physics** — Stack slices to NIfTI, then branch to optical parameter estimation (genetic algorithm with MCX or PyXOpto) and thermal bioheat simulation.

**Validation** — Z-coherence metrics, sequential ordering checks, HR source provenance logging, tissue-aware diagnostics, and cross-run comparison plots.

Key principle throughout: the mask is the ground truth anchor for all downstream editing and simulation."

---

## synthetic-roadmap

### Roadmap: From 2D Tiles → 3D Coherent Volumes

| Phase | Label | Status | Key Components |
|-------|-------|--------|----------------|
| **A** | Quick Win — 3D Scaffold | ✅ Complete | Cylindrical mask propagation (`propagate_mask_across_slices.py`), NIfTI stacking (`stack_slices_to_nifti.py`), Z-coherence metrics (`compute_z_coherence_metrics.py`). Self-contained, no model required. |
| **B** | Volume Inpainting Integration | 🟡 In Progress | `generate_synthetic_slices.py` (smoke test images), `build_volume_inpaint_metadata.py` (CSV adapter), `run_volume_inpaint_pipeline.py` (8-step orchestrator: mask → pairs → extract → inpaint → merge → stack → coherence → manifest). |
| **B.1** | Tile-First High-Res (Preferred) | 🟡 In Progress | Avoids full-slice resize. Strict bbox mode, tissue-aware mask placement, ROI patch extraction at full resolution, inpaint at model res, merge back. DVC stage: `run_3d_volume_inpaint_tile_smoke`. |
| **C** | Multi-Physics Bridge | 🟡 Scaffolded | MCX volume builder + batch runner, optical GA with surrogate/realistic forward models, thermal Pennes solver. Smoke tests pass. End-to-end validation pending. |
| **D** | Advanced Z-Coherence + Semantics | 🔴 Future | Affine mask warping between adjacent slices (relaxed cylinders). Z-smoothness reward in Phase 2/3 loop. YOLO-based tissue detector (512×512 tiles) replacing manual / random mask sources. Multi-class conditional inpainting. |

### Key Design Decisions (Conservative Claims)

1. **Cylindrical masks are a deliberate simplification.** Identical mask on every Z-slice. Zero added complexity in diffusion. Z-coherence is *measured* but not *enforced* during generation. Future work (Phase D) can relax this.

2. **Tile-first is strictly preferred over full-slice resize.** Rationale: high-res histology (e.g. 5000×5000) cannot be naively downsampled to 512×512 without losing tissue architecture. The patch extraction → inpaint → merge flow preserves original resolution everywhere outside the edited region.

3. **HR source guardrails are explicit.** The pipeline validates that `run_manifest.json → pairs_csv.image_dir` points to a valid HR source (`cropped_slices/*.png`) and that slice IDs are truly sequential. Legacy `melanoma_3d/slices/` (low-res stack) is explicitly excluded from new benchmark runs.

4. **No large dataset downloads.** MATRICS-A (hundreds of GB) remains out of scope. All MVP work uses Histo-Seg (~150 images) and optionally Zenodo 8155124 (~700 MB, 66 slices).

### Speaker Notes (Slide 11)

"Phase A is done — we can propagate cylindrical masks, stack to NIfTI, and measure adjacent-slice SSIM.

Phase B is the current focus. The tile-first B.1 sub-flow is preferred because it preserves full-resolution tissue architecture. Strict bbox mode ensures no synthetic padding is introduced.

Phase C bridges to physics — MCX, optical GA, and thermal simulation are scaffolded but not yet validated against real measurements.

Phase D is future work: relaxed mask propagation, Z-smoothness rewards, and a YOLO detector for semantic mask sources.

The roadmap is deliberately incremental: each phase adds capability without breaking the mask-agnostic pipeline design."

---

## open-questions

### Immediate Milestones (Next 1–2 Cycles)

| # | Milestone | Dependencies | Acceptance |
|---|-----------|-------------|------------|
| M1 | Run real LoRA inpainting pass on a sequential HR slice stack | Trained LoRA checkpoint (Phase 1/2/3), base SDXL model, small Zenodo stack | Inpainted NIfTI volume + Z-coherence metrics + run manifest |
| M2 | Z-coherence thresholding — auto-reject volumes where mean adjacent SSIM < threshold | M1 output, threshold parameter in `compute_z_coherence_metrics.py` | Rejection log + fallback slice selection |
| M3 | Per-slice affine mask warping (relaxed cylinder) | M1, `propagate_mask_across_slices.py` | Warped mask stack with configurable perturbation magnitude |
| M4 | CIELAB ΔE* in QC output for merged patches | `merge_inpainted_patches.py` | ΔE* per patch + summary in QC JSON |
| M5 | MCX vs PyXOpto batch comparison across N seeds | Batch compare script, shared bounds | Convergence scatter plot + summary table |

### Open Research Questions

1. **Semantic mask encoding.** The current pipeline is mask-agnostic — masks are binary. How should tissue-class information (e.g. 12-class Histo-Seg labels) be encoded in the inpainting condition? Concatenated class channels? Conditioned cross-attention? Separate LoRA per class?

2. **Minimal Z-slice count for meaningful coherence.** Is 10 slices enough to detect Z-coherence degradation? 20? Does the metric saturate? This affects how we size test volumes.

3. **Optical prior assignment strategy.** Should biophysical priors (μa, μs, g, n) be assigned per label region (coarse) or per voxel (fine)? The coarse approach is simpler and matches typical MCX input format; the fine approach may overfit sparse data.

4. **Validation against real histology.** Can we compare inpainted tissue against real H&E stain vectors (e.g. Macenko normalisation)? If the model generates plausible but non-real stain colours, does that matter for downstream simulation?

5. **Cylindrical validity as Z-spacing increases.** For thick-section datasets (e.g. > 5 μm between slices), does the rigid cylindrical mask assumption break down? What is the maximum Z-spacing before affine warping becomes necessary?

6. **Classifier reward generalisation.** Phase 2/3 reward signals come from a binary histology classifier trained on Histo-Seg. Does this classifier reward generalise to held-out tissue types or stain variations? What is the reward-hacking risk?

### Guiding Principles

- **Mask-agnostic pipeline** — the editing workflow does not depend on mask source. Better masks directly yield better results.
- **Semantic-mask-as-ground-truth** — the long-term goal is class-aware masks, but the pipeline works with binary masks today.
- **Provenanced intermediates** — every step logs its inputs, parameters, and outputs in the run manifest. Enables retrospective QC and comparison.

### Speaker Notes (Slide 12)

"We have six near-term milestones that tighten the pipeline, and six open research questions that guide our long-term direction.

The biggest open questions are around semantic encoding — how to feed tissue class information into the inpainting model — and validation — how to know if an inpainted patch is histologically realistic.

Our three guiding principles keep the design robust: mask-agnosticism means we can swap in better mask sources without rewriting the pipeline; provenanced intermediates mean every run is auditable."
