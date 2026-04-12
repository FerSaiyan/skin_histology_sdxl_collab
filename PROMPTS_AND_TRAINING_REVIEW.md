# Tile Inpaint Prompt & Training Review

This note documents the current tile-based Phase 1/2/3 setup and why prompts,
sampling, and mask strategy are configured this way.

## Current Data/Mask Assumptions

- Training images are materialized 512x512 tiles in `data/artifacts/tiles/materialized_512`.
- Training masks are pre-generated random masks in `data/artifacts/tiles/masks_random_512`.
- Tile mining uses a white-content cap (`tile_white_frac <= 0.30`) plus tissue coverage filter.

Why this matters:
- Random inpaint masks are useful only if tiles contain enough tissue.
- White-heavy tiles push edits into background and harm learning signal.

## Phase 1 (Base LoRA Warmup)

Config: `configs/sdxl_lora_phase1_skin_histology_tiles_randommask.yaml`

### Prompting
- Caption template (training):
  - `H&E stained skin histopathology tile, diagnosis: {token}{descriptor_suffix}, realistic microscopy`
- Class tokens:
  - `non_cancer`, `cancer`
- Per-class descriptors are enabled.

### Sample preview prompts (during training)
- Same template as above.
- Negative prompt includes anti-style terms:
  - `painting, illustration, cartoon, geometric pattern, abstract art`
- `sample_init_image` + `sample_mask_image` are set, so preview sampling is inpaint-like (not pure text-to-image).

### Training behavior
- LoRA rank/alpha: `8/8`
- Resolution: `512x512`
- Batch/accum: `1 x 4`
- Epochs: `5`
- Masked-loss inpainting is enabled and uses directory masks (`lora_mask_mode: directory`).

## Phase 2 (Reward-Guided Selection)

Config: `configs/sdxl_lora_phase2_reward_skin_histology_tiles_randommask.yaml`

### Prompting for checkpoint selection
- Selector prompt template:
  - `H&E stained skin histopathology tile, diagnosis: {token}{descriptor_suffix}, realistic microscopy`
- Same anti-style negative prompt as Phase 1.
- Same/cross class strengths:
  - same: `0.55`, cross: `0.75`

### Training behavior
- `5` cycles, `250` steps per cycle.
- Each cycle trains from the current best checkpoint.
- Selector benchmarks cycle checkpoints and picks best by weighted reward:
  - cross-class performance is weighted more than same-class (`0.75 / 0.25`).

## Phase 3 (Morphology Reward + Curriculum)

Config: `configs/sdxl_lora_phase3_morph_reward_skin_histology_tiles_randommask.yaml`

### Prompting
- Same selector prompt/negative style as Phase 2 for consistency.

### Training behavior
- `5` cycles, `250` steps per cycle.
- Morphology-focused reward loop with curriculum enabled.
- Curriculum keeps real anchors and emphasizes cross-class edits.
- Holdout evaluation is enabled at each cycle best checkpoint.

## Are These Prompts Appropriate for Random Masks?

Yes, mostly. They are diagnosis-conditioned prompts with explicit microscopy style
anchors and anti-art negatives, which is a good fit for random-mask inpainting.

What to watch:
- If outputs drift into stylized artifacts, strengthen negatives or reduce cross-class strength.
- If edits are too weak, increase cross-class strength slightly or raise LoRA scale during selector benchmarking.

## Suggested Next Prompt Tweaks (Optional)

If needed, test one variable at a time:

1. Add stronger histology anchors:
   - `high-detail H&E whole-slide crop, pathology microscopy`
2. Add stronger anti-style negatives:
   - `oil painting, watercolor, sketch, 3d render`
3. Lower selector cross-class strength from `0.75` to `0.65` if edits are over-aggressive.
