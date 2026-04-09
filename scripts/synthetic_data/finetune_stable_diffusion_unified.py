#!/usr/bin/env python3
# Unified Stable Diffusion finetune script (full finetune / LoRA) with .env, YAML config, MLflow tracking

import os, sys, subprocess, argparse, csv, json, shutil, warnings
from pathlib import Path
import unicodedata as ud
from unidecode import unidecode
import re
import numpy as np
from PIL import Image, ImageFilter
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
_ws_re = re.compile(r"\s+")

# Make src importable and load .env
ROOT = Path(__file__).resolve().parents[2]  # project root
PROJECT_ROOT = ROOT
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.utils import load_dotenv, getenv_path
from src.exp.config import load_config, coalesce_path, ensure_dir
from src.exp.tracking import mlflow_run

load_dotenv()


def run(cmd, cwd=None, env=None):
    """Wrapper que imprime o comando e aborta se retornar erro."""
    print(">>", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)


def run_with_nan_retry_for_latents(
    cmd,
    *,
    dataset_path: Path,
    cwd=None,
    env=None,
    max_nan_retries: int = 0,
    quarantine_dir: Path | None = None,
):
    """
    Run prepare_buckets_latents command and optionally recover from known
    "NaN detected in latents: <path>" failures by removing the offending image
    and its paired .caption file, then retrying.
    """
    max_nan_retries = int(max_nan_retries)
    attempt = 0
    while True:
        print(">>", " ".join(map(str, cmd)))
        proc = subprocess.run(
            list(map(str, cmd)),
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        if proc.returncode == 0:
            return

        combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        bad_path_str = None
        matches = re.findall(r"NaN detected in latents:\s*(.+)", combined)
        for cand in reversed(matches):
            cand = str(cand).strip().strip("'\"")
            if "{" in cand or "}" in cand:
                continue
            if cand:
                bad_path_str = cand
                break
        if bad_path_str is None or attempt >= max_nan_retries:
            raise subprocess.CalledProcessError(proc.returncode, list(map(str, cmd)))

        bad_path = Path(bad_path_str)
        try:
            bad_resolved = bad_path.resolve()
        except Exception:
            bad_resolved = bad_path
        try:
            dataset_resolved = dataset_path.resolve()
            if dataset_resolved not in [bad_resolved, *bad_resolved.parents]:
                raise RuntimeError(
                    f"Refusing to delete NaN image outside dataset_path: {bad_resolved}"
                )
        except Exception:
            # If safety check fails due resolution issues, proceed cautiously only
            # when original path is directly under dataset_path.
            if bad_path.parent != dataset_path:
                raise

        if quarantine_dir is None:
            quarantine_dir = dataset_path / "_nan_quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        def _quarantine_one(path: Path) -> Path | None:
            if not path.exists():
                return None
            dest = quarantine_dir / path.name
            if dest.exists():
                i = 1
                while True:
                    alt = quarantine_dir / f"{path.stem}__{i}{path.suffix}"
                    if not alt.exists():
                        dest = alt
                        break
                    i += 1
            shutil.move(str(path), str(dest))
            return dest

        cap_path = bad_path.with_suffix(".caption")
        moved_img = _quarantine_one(bad_path)
        moved_cap = _quarantine_one(cap_path)
        attempt += 1
        print(
            f"Quarantined NaN image: {bad_path} -> {moved_img} "
            f"(caption -> {moved_cap}, retry {attempt}/{max_nan_retries})"
        )

def stem(p: str) -> str:          # “IMG_123.JPG” → “img_123”
    return Path(p).stem.lower()

def normalise_cat(txt: str) -> str:
    """
    - strip()                 → remove leading/trailing spaces
    - Unicode NFD + ASCII     → drop accents
    - casefold()              → lower-case (unicode-aware)
    - collapse \t, newlines   → single space
    """
    txt = txt.strip()
    txt = ud.normalize("NFD", txt)
    txt = txt.encode("ascii", "ignore").decode("ascii")
    txt = txt.casefold()
    txt = _ws_re.sub(" ", txt)          # <── collapse whitespace
    return txt


def _coerce_optional_int(raw) -> int | None:
    if raw in (None, "", "null"):
        return None
    return int(raw)


def _coerce_prompt_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw if str(x).strip()]
    return [str(raw)]


def _extract_diagnosis_token(text: str) -> str:
    m = re.search(r"diagnosis:\s*([A-Za-z0-9_]+)", text)
    if m:
        return m.group(1)
    parts = text.split()
    return parts[-1] if parts else "benign_lesion"


def _as_optional_path(raw, base_dir: Path) -> Path | None:
    if raw in (None, "", "null"):
        return None
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _prepare_soft_masks_for_training(
    *,
    dataset_path: Path,
    source_mask_dir: Path,
    output_mask_dir: Path,
    feather_radius: float,
    mask_strength: float,
) -> Path:
    """Prepare a one-to-one mask directory aligned to the materialized training images."""
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
    dataset_images = [p for p in dataset_path.iterdir() if p.suffix.lower() in image_exts]
    if not dataset_images:
        raise SystemExit(f"No training images found in dataset_path: {dataset_path}")

    source_masks_by_stem = {}
    for p in source_mask_dir.iterdir():
        if p.suffix.lower() in image_exts:
            source_masks_by_stem[p.stem.lower()] = p

    missing_masks = [img.name for img in dataset_images if img.stem.lower() not in source_masks_by_stem]
    if missing_masks:
        preview = ", ".join(missing_masks[:8])
        raise SystemExit(
            f"Missing masks for {len(missing_masks)} training images in {source_mask_dir}. "
            f"Examples: {preview}"
        )

    if output_mask_dir.exists():
        for p in output_mask_dir.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()
    else:
        output_mask_dir.mkdir(parents=True, exist_ok=True)

    for img in dataset_images:
        src_mask = source_masks_by_stem[img.stem.lower()]
        dst_mask = output_mask_dir / src_mask.name
        with Image.open(src_mask) as pil_mask:
            mask = pil_mask.convert("L")
            if feather_radius > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))
            if abs(mask_strength - 1.0) > 1e-6:
                arr = np.asarray(mask, dtype=np.float32) * float(mask_strength)
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                mask = Image.fromarray(arr, mode="L")
            mask.save(dst_mask)

    print(
        f"Prepared soft masks for masked-loss training in {output_mask_dir} "
        f"(feather_radius={feather_radius}, mask_strength={mask_strength})."
    )
    return output_mask_dir

def main(args):
    # Load config if provided
    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    # Core identifiers
    project_path = Path(args.project_path) if args.project_path else Path(cfg.get('project_path') or PROJECT_ROOT)
    seed_name    = args.seed_name or str(cfg.get('seed_name'))
    fold_number  = args.fold_number or str(cfg.get('fold_number'))
    dataset_subdir = args.dataset_subdir or cfg.get('dataset_subdir')
    if not seed_name or not fold_number:
        raise SystemExit("seed_name and fold_number must be provided via CLI or config")

    # Paths and external tool locations
    base_model_path = args.base_model_path or cfg.get('base_model_path') or getenv_path('DIFFUSION_BASE_MODEL_PATH')
    if not base_model_path:
        raise SystemExit("base_model_path must be provided via CLI, config, or DIFFUSION_BASE_MODEL_PATH in .env")
    kohya_dir = cfg.get('kohya_scripts_dir') or getenv_path('KOHYA_DIR') or str(project_path / 'kohya_ss' / 'sd-scripts')
    SD = Path(kohya_dir).resolve()
    # Validate kohya path; fallback to project-local if env/config is invalid
    mk_caption = SD / 'finetune' / 'merge_captions_to_metadata.py'
    if not mk_caption.exists():
        sd_default = (project_path / 'kohya_ss' / 'sd-scripts').resolve()
        if (sd_default / 'finetune' / 'merge_captions_to_metadata.py').exists():
            SD = sd_default
        else:
            raise SystemExit(
                f"Cannot locate kohya sd-scripts. Tried: {mk_caption} and {sd_default}.\n"
                f"Set a valid path via config.kohya_scripts_dir or KOHYA_DIR in .env"
            )
    print(f"Using kohya sd-scripts at: {SD}")

    # Training mode: full vs LoRA
    training_mode = str(cfg.get('training_mode', 'full')).lower()  # 'full' | 'lora'

    # SDXL support (kohya provides separate SDXL train scripts)
    is_sdxl = bool(cfg.get('is_sdxl', cfg.get('sdxl', False)))

    # Hyperparameters (with sensible defaults, overridable via config)
    UNET_LR = float(cfg.get('unet_lr', 3e-5))
    TE_LR   = float(cfg.get('text_encoder_lr', 5e-6))
    resolution = str(cfg.get('resolution', '512,512'))
    train_batch_size = int(cfg.get('train_batch_size', 16))
    gradient_accumulation_steps = int(cfg.get('gradient_accumulation_steps', 1))
    dataset_repeats = int(cfg.get('dataset_repeats', 1))
    sample_every_n_steps = int(cfg.get('sample_every_n_steps', 80))
    sample_sampler = str(cfg.get('sample_sampler', 'k_dpm_2_a'))
    sample_steps = int(cfg.get("sample_steps", 35))
    sample_guidance_scale = float(cfg.get("sample_guidance_scale", 7.0))
    sample_negative_prompt = str(cfg.get("sample_negative_prompt", "lowres,bad anatomy"))
    sample_prompt_template = str(
        cfg.get(
            "sample_prompt_template",
            "clinical photo of an oral lesion, diagnosis: {token}{descriptor_suffix}",
        )
    )
    sample_prompts_per_class = int(cfg.get("sample_prompts_per_class", 1))
    if sample_prompts_per_class < 1:
        raise SystemExit("sample_prompts_per_class must be >= 1")
    sample_prompt_templates = _coerce_prompt_list(cfg.get("sample_prompt_templates"))
    sample_prompts_by_token_cfg = cfg.get("sample_prompts_by_token", None)
    sample_use_class_descriptors = bool(cfg.get("sample_use_class_descriptors", True))
    sample_class_descriptors_cfg = cfg.get("sample_class_descriptors", {}) or {}
    sample_seed_mode = str(cfg.get("sample_seed_mode", "per_class")).strip().lower()
    if sample_seed_mode not in {"fixed", "per_class", "random"}:
        raise SystemExit("sample_seed_mode must be one of: fixed, per_class, random")
    sample_seed = _coerce_optional_int(cfg.get("sample_seed", None))
    sample_seed_base = int(cfg.get("sample_seed_base", cfg.get("seed", 222)))
    sample_denoising_strength_raw = cfg.get("sample_denoising_strength", None)
    sample_denoising_strength = (
        float(sample_denoising_strength_raw)
        if sample_denoising_strength_raw not in (None, "", "null")
        else None
    )
    if sample_denoising_strength is not None and not (0.0 <= sample_denoising_strength <= 1.0):
        raise SystemExit("sample_denoising_strength must be in [0.0, 1.0]")
    sample_mask_strength = float(cfg.get("sample_mask_strength", 1.0))
    if sample_mask_strength <= 0:
        raise SystemExit("sample_mask_strength must be > 0")
    sample_mask_blur_radius = float(cfg.get("sample_mask_blur_radius", 0.0))
    mixed_precision = str(cfg.get('mixed_precision', 'fp16'))
    prepare_latents_nan_retries = int(cfg.get("prepare_latents_nan_retries", 0))
    if prepare_latents_nan_retries < 0:
        raise SystemExit("prepare_latents_nan_retries must be >= 0")
    prepare_latents_mixed_precision = str(cfg.get("prepare_latents_mixed_precision", mixed_precision)).strip().lower()
    if prepare_latents_mixed_precision not in {"no", "fp16", "bf16"}:
        raise SystemExit("prepare_latents_mixed_precision must be one of: no, fp16, bf16")
    prepare_latents_nan_quarantine_dir = _as_optional_path(
        cfg.get("prepare_latents_nan_quarantine_dir"),
        project_path,
    )
    max_train_epochs = int(cfg.get('max_train_epochs', 50))
    # Optional: cap by steps (useful for short sanity runs). Note: in kohya scripts,
    # max_train_epochs overrides max_train_steps if both are set.
    max_train_steps = cfg.get('max_train_steps', None)
    max_train_steps = int(max_train_steps) if max_train_steps not in (None, "", "null") else None

    lr_warmup_steps = int(cfg.get('lr_warmup_steps', 200))

    save_every_n_epochs = int(cfg.get('save_every_n_epochs', 5))
    save_every_n_steps = cfg.get('save_every_n_steps', None)
    save_every_n_steps = int(save_every_n_steps) if save_every_n_steps not in (None, "", "null") else None
    noise_offset = float(cfg.get('noise_offset', 0.05))
    optimizer_type = str(cfg.get('optimizer_type', 'AdamW8bit'))
    gradient_checkpointing = bool(cfg.get('gradient_checkpointing', True))
    min_snr_gamma = float(cfg.get('min_snr_gamma', 5))
    lr_scheduler = str(cfg.get('lr_scheduler', 'constant_with_warmup'))
    seed = int(cfg.get('seed', 222))

    # Attention backend (mutually exclusive in kohya: prefer one)
    attn_backend = str(cfg.get('attn_backend', 'xformers')).lower()
    if attn_backend in ("xformers", "xformer"):
        attn_flags = ["--xformers"]
    elif attn_backend == "sdpa":
        attn_flags = ["--sdpa"]
    elif attn_backend in ("mem_eff_attn", "mem_eff", "memory_efficient"):
        attn_flags = ["--mem_eff_attn"]
    elif attn_backend in ("none", ""):
        attn_flags = []
    else:
        raise SystemExit(
            f"Unknown attn_backend='{attn_backend}' (expected xformers|sdpa|mem_eff_attn|none)"
        )

    # LoRA inpainting / masked-loss specific toggles (no-op for full FT)
    lora_use_masked_loss: bool = bool(cfg.get("lora_use_masked_loss", False))
    lora_mask_dir_raw = cfg.get("lora_mask_dir")
    lora_mask_dir = Path(lora_mask_dir_raw).resolve() if lora_mask_dir_raw else None
    lora_mask_feather_radius = float(cfg.get("lora_mask_feather_radius", 0.0))
    lora_mask_strength = float(cfg.get("lora_mask_strength", 1.0))
    if lora_mask_strength <= 0:
        raise SystemExit("lora_mask_strength must be > 0")
    lora_mask_processed_dir = _as_optional_path(cfg.get("lora_mask_processed_dir"), project_path)
    # ───────────────────── 1. Paths e pastas ─────────────────────
    materialize_from_csv = bool(cfg.get('materialize_from_labels_csv', False))
    dataset_path = Path(
        cfg.get('dataset_path') or
        project_path / 'data' / 'artifacts' / f"syntetic_custom_base_{seed_name}" /
        (dataset_subdir or fold_number)
    ).resolve()
    if materialize_from_csv:
        # Avoid mixing old images (e.g., full multisource) with a new
        # class-balanced materialization: clear the folder first.
        if dataset_path.exists():
            for p in dataset_path.iterdir():
                try:
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                except Exception:
                    # best-effort cleanup; continue even if a file cannot be removed
                    pass
        else:
            dataset_path.mkdir(parents=True, exist_ok=True)
    elif not dataset_path.is_dir():
        raise SystemExit(f"Dataset '{dataset_path}' não encontrado.")

    meta_dir  = SD / "sd_datasets" / "small_test" / f"syntetic_custom_base_{seed_name}" / fold_number
    out_dir   = Path(cfg.get('output_dir') or project_path / 'finetunes' / 'oral_lesions' /
                     f"syntetic_custom_base_{seed_name}___{fold_number}").resolve()
    logs_dir  = Path(cfg.get('logs_dir') or SD / 'logs')

    for p in (meta_dir, out_dir, logs_dir):
        p.mkdir(parents=True, exist_ok=True)

    meta_cap_json = meta_dir / "meta_cap_v1.json"
    meta_lat_json = meta_dir / "meta_lat.json"

    # When we re-materialize the dataset from a labels CSV (e.g., switching
    # from full multisource → balanced subset), any existing metadata JSONs
    # may still contain entries for images that no longer exist under
    # dataset_path. The kohya helpers merge into existing metadata, so we
    # explicitly clear these files in that case to keep things in sync.
    if materialize_from_csv:
        for p in (meta_cap_json, meta_lat_json):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SD) + os.pathsep + env.get("PYTHONPATH", "")
    # Reduce CUDA memory fragmentation spikes on long SD/SDXL jobs.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ───────────────────── 2. Geração de captions ─────────────────────
    use_existing_dataset_captions = bool(cfg.get("use_existing_dataset_captions", False))
    labels_csv = cfg.get('labels_csv_path') or getenv_path('LABELS_CSV')
    CSV_PATH = Path(labels_csv) if labels_csv else None
    if use_existing_dataset_captions and materialize_from_csv:
        raise SystemExit(
            "use_existing_dataset_captions=True is not compatible with "
            "materialize_from_labels_csv=True."
        )
    if not use_existing_dataset_captions and CSV_PATH is None:
        raise SystemExit("labels_csv_path not provided and LABELS_CSV not set in .env")

    # Map textual labels (Portuguese fine taxonomies or 4-class coarse labels)
    # into 4 coarse tokens used in captions/prompts.
    _RAW = {
        # 4-class coarse labels (already in target space)
        "healthy": "healthy",
        "benign_lesion": "benign_lesion",
        "benign lesion": "benign_lesion",
        "opmd": "opmd",
        "cancer": "cancer",
        # Legacy internal fine labels (English)
        "malignant": "cancer",
        "potentially_malignant": "opmd",
        "infectious": "benign_lesion",
        "reactive_inflammatory": "benign_lesion",
        "other": "benign_lesion",
        # Legacy Portuguese diagnosis_categories from transfer_learning_labels.csv
        "Doenças infecciosas": "benign_lesion",
        "Lesões inflamatórias reativas": "benign_lesion",
        "Neoplasias malignas": "cancer",
        "Doenças potencialmente malignas": "opmd",
        "Outras": "benign_lesion",
    }
    CAT2TOK = {normalise_cat(k): v for k, v in _RAW.items()}

    label_dict = {}
    filename_map = {}
    image_path_map = {}
    if not use_existing_dataset_captions:
        assert CSV_PATH is not None
        with CSV_PATH.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise SystemExit(f"No columns found in labels CSV: {CSV_PATH}")
            fieldnames = [fn.strip() for fn in reader.fieldnames]

            # Allow overriding label column via config; otherwise prefer coarse_label, then diagnosis_categories.
            preferred = []
            cfg_label_col = str(cfg.get("label_column") or "").strip()
            if cfg_label_col:
                preferred.append(cfg_label_col)
            preferred.extend(["coarse_label", "diagnosis_categories", "label", "diagnosis category"])
            label_col = None
            for cand in preferred:
                if cand in fieldnames:
                    label_col = cand
                    break
            if label_col is None:
                raise SystemExit(
                    f"Could not find a label column in {CSV_PATH} "
                    f"(looked for {preferred})."
                )
            if "filename" not in fieldnames:
                raise SystemExit(f"Column 'filename' is required in {CSV_PATH}")
            if materialize_from_csv and "image_path" not in fieldnames:
                raise SystemExit(
                    f"materialize_from_labels_csv=True but 'image_path' column "
                    f"is missing in {CSV_PATH}"
                )

            # Build label and path dictionaries keyed by filename stem
            for row in reader:
                raw_cat = row.get(label_col, "")
                category = normalise_cat(str(raw_cat))
                token = CAT2TOK.get(category)
                if token is None:
                    # If the label already looks like a coarse token, keep it;
                    # otherwise fall back to benign_lesion.
                    coarse = category.replace(" ", "_")
                    if coarse in {"healthy", "benign_lesion", "opmd", "cancer"}:
                        token = coarse
                    else:
                        token = "benign_lesion"
                fn = row["filename"]
                key = stem(fn)
                label_dict[key] = token
                filename_map[key] = fn
                if materialize_from_csv:
                    image_path_map[key] = row.get("image_path", "")

    # Optionally materialize the training dataset from the labels CSV
    # (e.g., using the full multisource_train.csv with absolute image_path).
    if materialize_from_csv:
        missing_src = 0
        created = 0
        for key, fn in filename_map.items():
            src_path = image_path_map.get(key)
            if not src_path:
                continue
            src = Path(src_path)
            if not src.is_file():
                missing_src += 1
                continue
            dst = dataset_path / fn
            if dst.exists():
                continue
            try:
                os.symlink(src, dst)
            except OSError:
                # Fallback if symlinks are not available
                shutil.copy2(src, dst)
            created += 1
        print(f"Materialized dataset in {dataset_path} from {CSV_PATH}: {created} files (missing sources: {missing_src})")

    EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif", ".jxl")
    default_class_descriptors = {
        "healthy": "intact oral mucosa without visible lesion",
        "benign_lesion": "well-circumscribed benign oral lesion",
        "opmd": "oral potentially malignant disorder with heterogeneous plaque-like changes",
        "cancer": "irregular invasive oral lesion with high-risk malignant appearance",
    }
    caption_template = str(
        cfg.get("caption_template", "clinical photo of an oral lesion, diagnosis: {token}{descriptor_suffix}")
    )
    caption_use_class_descriptors = bool(cfg.get("caption_use_class_descriptors", False))
    caption_class_descriptors_cfg = cfg.get("caption_class_descriptors", {}) or {}

    if use_existing_dataset_captions:
        missing_caps = []
        for img in dataset_path.iterdir():
            if img.suffix.lower() not in EXTS:
                continue
            cap_path = img.with_suffix(".caption")
            if not cap_path.is_file():
                missing_caps.append(img.name)
        if missing_caps:
            preview = ", ".join(missing_caps[:8])
            raise SystemExit(
                f"use_existing_dataset_captions=True but {len(missing_caps)} images "
                f"have no .caption file in {dataset_path}. Examples: {preview}"
            )
        print(f"Using existing .caption files from {dataset_path} (caption generation skipped).")
    else:
        synth_name_re = re.compile(r"^synth_(healthy|benign_lesion|opmd|cancer)(?:_|$)")
        for img in dataset_path.iterdir():
            if img.suffix.lower() not in EXTS:
                continue

            img_stem = stem(img.name)
            tok = label_dict.get(img_stem)
            if tok is None and img_stem.startswith("real_"):
                tok = label_dict.get(img_stem[len("real_"):])
            if tok is None and img_stem.startswith("synth_"):
                m = synth_name_re.match(img_stem)
                if m:
                    tok = m.group(1)
            if tok is None:
                tok = "benign_lesion"

            descriptor = str(caption_class_descriptors_cfg.get(tok, default_class_descriptors.get(tok, ""))).strip()
            descriptor_suffix = f", {descriptor}" if (caption_use_class_descriptors and descriptor) else ""
            try:
                caption_text = caption_template.format(
                    token=tok,
                    descriptor=descriptor,
                    descriptor_suffix=descriptor_suffix,
                )
            except Exception as e:
                raise SystemExit(
                    f"Invalid caption_template='{caption_template}'. "
                    "Expected placeholders like {token} or {descriptor_suffix}."
                ) from e
            cap_path = img.with_suffix(".caption")
            if cap_path.is_symlink():
                # Avoid mutating source-dataset captions through symlinked files.
                cap_path.unlink()
            cap_path.write_text(caption_text.strip(), encoding="utf-8")


    # quick sanity-check (remove or comment out later)
    from collections import Counter, defaultdict

    token_counts = Counter()
    examples      = defaultdict(list)

    for img in dataset_path.iterdir():
        if img.suffix.lower() not in EXTS:
            continue
        token = _extract_diagnosis_token(img.with_suffix(".caption").read_text())
        token_counts[token] += 1
        if len(examples[token]) < 3:
            examples[token].append(img.name)

    print("Token distribution:", token_counts)
    for tok, ex in examples.items():
        print(f"{tok:8} -> {', '.join(ex)}")

    # ───────────────────── 3. Prompts de amostragem ────────────────────
    prompts = SD / "prompts_to_check.txt"
    coarse_tokens = ["healthy", "benign_lesion", "opmd", "cancer"]
    sample_class_descriptors = {
        tok: str(sample_class_descriptors_cfg.get(tok, default_class_descriptors.get(tok, ""))).strip()
        for tok in coarse_tokens
    }

    def _format_prompt(template: str, tok: str) -> str:
        descriptor = sample_class_descriptors.get(tok, "")
        descriptor_suffix = f", {descriptor}" if (sample_use_class_descriptors and descriptor) else ""
        try:
            out = template.format(
                token=tok,
                descriptor=descriptor,
                descriptor_suffix=descriptor_suffix,
            )
        except Exception as e:
            raise SystemExit(
                f"Invalid sample prompt template: '{template}'. "
                "Expected placeholders like {token} or {descriptor_suffix}."
            ) from e
        out = str(out).strip()
        if not out:
            raise SystemExit("Empty sample prompt generated; check sample prompt templates in config.")
        return out

    def _prompts_for_token(tok: str) -> list[str]:
        if isinstance(sample_prompts_by_token_cfg, dict):
            tok_prompts = _coerce_prompt_list(sample_prompts_by_token_cfg.get(tok))
            if tok_prompts:
                return [_format_prompt(p, tok) for p in tok_prompts]

        templates = sample_prompt_templates if sample_prompt_templates else [sample_prompt_template]
        repeated = [templates[i % len(templates)] for i in range(sample_prompts_per_class)]
        return [_format_prompt(tpl, tok) for tpl in repeated]

    def _seed_for_prompt(token_idx: int, prompt_idx: int) -> int | None:
        if sample_seed_mode == "random":
            return None
        if sample_seed_mode == "fixed":
            return sample_seed if sample_seed is not None else sample_seed_base
        # per_class: stable but distinct seeds across classes/prompts.
        base = sample_seed if sample_seed is not None else sample_seed_base
        return int(base + token_idx * 100 + prompt_idx)

    sample_init_image = _as_optional_path(cfg.get("sample_init_image"), project_path)
    sample_mask_image = _as_optional_path(cfg.get("sample_mask_image"), project_path)
    if sample_init_image is not None and not sample_init_image.is_file():
        raise SystemExit(f"sample_init_image does not exist: {sample_init_image}")
    if sample_mask_image is not None and not sample_mask_image.is_file():
        raise SystemExit(f"sample_mask_image does not exist: {sample_mask_image}")
    if sample_mask_image is not None and sample_init_image is None:
        raise SystemExit("sample_mask_image requires sample_init_image to be set")
    # Some samplers (notably k_dpm_2_a) are unstable for img2img/inpaint sampling in kohya.
    # When sample_init_image is set, force a safer sampler for sample previews.
    if sample_init_image is not None and sample_sampler.lower() in {"k_dpm_2_a", "k_dpm_2"}:
        print(
            f"sample_sampler='{sample_sampler}' is unstable for inpaint/img2img sample previews; "
            "overriding to 'ddim'."
        )
        sample_sampler = "ddim"

    # Derive W/H from the configured training resolution (defaults: 512 for SD1.x, 1024 for SDXL)
    try:
        w_str, h_str = [s.strip() for s in resolution.split(",")[:2]]
        w, h = int(w_str), int(h_str)
    except Exception:
        w = h = 1024 if is_sdxl else 512

    sample_suffix_parts = [
        f"--n {sample_negative_prompt}",
        f"--w {w}",
        f"--h {h}",
        f"--l {sample_guidance_scale}",
        f"--s {sample_steps}",
    ]
    if sample_init_image is not None:
        sample_suffix_parts.append(f"--i {sample_init_image}")
    if sample_mask_image is not None:
        sample_suffix_parts.append(f"--m {sample_mask_image}")
    if sample_denoising_strength is not None:
        sample_suffix_parts.append(f"--t {sample_denoising_strength}")
    if abs(sample_mask_strength - 1.0) > 1e-6:
        sample_suffix_parts.append(f"--ms {sample_mask_strength}")
    if sample_mask_blur_radius > 0:
        sample_suffix_parts.append(f"--mbr {sample_mask_blur_radius}")
    prompt_lines = 0
    with prompts.open("w") as f:
        for token_idx, tok in enumerate(coarse_tokens):
            for prompt_idx, prompt_text in enumerate(_prompts_for_token(tok)):
                line_parts = [prompt_text, *sample_suffix_parts]
                prompt_seed = _seed_for_prompt(token_idx, prompt_idx)
                if prompt_seed is not None:
                    line_parts.append(f"--d {prompt_seed}")
                f.write(" ".join(line_parts) + "\n")
                prompt_lines += 1
    print(
        f"Wrote {prompt_lines} sample prompts to {prompts} "
        f"(seed_mode={sample_seed_mode}, prompts_per_class={sample_prompts_per_class})."
    )

    # ───────────────────── 4. merge captions → metadata ────────────────
    # For masked-loss LoRA we may train via DreamBooth/ControlNet dataset
    # config without using the fine-tuning metadata/latents pipeline.
    use_ft_latents = not (training_mode == "lora" and lora_use_masked_loss)

    if use_ft_latents:
        run(
            [sys.executable, "finetune/merge_captions_to_metadata.py", str(dataset_path), meta_cap_json],
            cwd=SD,
            env=env,
        )

        # ───────────────────── 5. buckets & latents ────────────────────
        run_with_nan_retry_for_latents(
            [
                sys.executable,
                "finetune/prepare_buckets_latents.py",
                str(dataset_path),
                meta_cap_json,
                meta_lat_json,
                base_model_path,
                "--batch_size",
                str(cfg.get("latent_batch_size", 8)),
                "--max_resolution",
                resolution,
                "--mixed_precision",
                prepare_latents_mixed_precision,
                "--bucket_reso_steps",
                str(cfg.get("bucket_reso_steps", 64)),
            ],
            dataset_path=dataset_path,
            cwd=SD,
            env=env,
            max_nan_retries=prepare_latents_nan_retries,
            quarantine_dir=prepare_latents_nan_quarantine_dir,
        )

    # ───────────────────── 6. Finetuning ─────────────────────────
    # Training length / checkpoint cadence flags
    max_train_flag = (
        f"--max_train_steps={max_train_steps}" if max_train_steps is not None else f"--max_train_epochs={max_train_epochs}"
    )
    save_every_flag = (
        f"--save_every_n_steps={save_every_n_steps}" if save_every_n_steps is not None else f"--save_every_n_epochs={save_every_n_epochs}"
    )
    if training_mode == "full":
        train_script = SD / "fine_tune.py"
        accel = [
            sys.executable, "-m", "accelerate.commands.launch",
            "--num_processes", "1",
            "--mixed_precision", mixed_precision,
            "--num_cpu_threads_per_process", "2",
            train_script,

            f"--pretrained_model_name_or_path={base_model_path}",
            f"--in_json={meta_lat_json}",
            f"--train_data_dir={str(dataset_path)}",
            f"--output_dir={str(out_dir)}",
            f"--resolution={resolution}",
            f"--train_batch_size={train_batch_size}",
            f"--gradient_accumulation_steps={gradient_accumulation_steps}",
            f"--dataset_repeats={dataset_repeats}",
            "--enable_bucket",
            "--keep_tokens=77",
            f"--sample_every_n_steps={sample_every_n_steps}",
            f"--sample_sampler={sample_sampler}",
            f"--sample_prompts={prompts}",
            f"--learning_rate={UNET_LR}",
            f"--learning_rate_te={TE_LR}",
            "--max_grad_norm=1",
            max_train_flag,
            f"--lr_warmup_steps={lr_warmup_steps}",
            "--train_text_encoder",
            save_every_flag,
            f"--noise_offset={noise_offset}",
            "--save_model_as=safetensors",
            f"--optimizer_type={optimizer_type}",
            "--gradient_checkpointing" if gradient_checkpointing else "",
            f"--min_snr_gamma={min_snr_gamma}",
            f"--lr_scheduler={lr_scheduler}",
            f"--logging_dir={logs_dir}",
            *attn_flags,
            f"--seed={seed}",
        ]
    elif training_mode == "lora":
        # SDXL LoRA training uses a separate kohya entrypoint.
        train_script = SD / ("sdxl_train_network.py" if is_sdxl else "train_network.py")
        network_dim = int(cfg.get('lora_rank', 16))
        network_alpha = int(cfg.get('lora_alpha', network_dim))
        lora_network_weights_raw = cfg.get("lora_network_weights", None)
        lora_network_weights = (
            _as_optional_path(lora_network_weights_raw, project_path)
            if lora_network_weights_raw not in (None, "", "null")
            else None
        )
        if lora_network_weights is not None and not lora_network_weights.is_file():
            raise SystemExit(f"Configured lora_network_weights does not exist: {lora_network_weights}")
        lora_dim_from_weights = bool(cfg.get("lora_dim_from_weights", False))
        # LoRA training often uses a single LR
        lora_lr = float(cfg.get('lora_lr', UNET_LR))
        accel = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            "1",
            "--mixed_precision",
            mixed_precision,
            "--num_cpu_threads_per_process",
            "2",
            train_script,
            f"--pretrained_model_name_or_path={base_model_path}",
        ]

        if lora_use_masked_loss:
            if lora_mask_dir is None:
                raise SystemExit(
                    "lora_use_masked_loss=True but 'lora_mask_dir' is not set in the config. "
                    "Point it to the ROI mask directory with one PNG per training image."
                )
            if not lora_mask_dir.is_dir():
                raise SystemExit(f"Configured lora_mask_dir does not exist or is not a directory: {lora_mask_dir}")
            effective_lora_mask_dir = lora_mask_dir
            if lora_mask_feather_radius > 0 or abs(lora_mask_strength - 1.0) > 1e-6:
                soft_mask_dir = (
                    lora_mask_processed_dir
                    if lora_mask_processed_dir is not None
                    else (meta_dir / "masks_soft_for_masked_loss")
                )
                effective_lora_mask_dir = _prepare_soft_masks_for_training(
                    dataset_path=dataset_path,
                    source_mask_dir=lora_mask_dir,
                    output_mask_dir=soft_mask_dir,
                    feather_radius=lora_mask_feather_radius,
                    mask_strength=lora_mask_strength,
                )

            # Build a minimal ControlNet-style dataset config (DreamBooth method with masks).
            # Each training image under dataset_path must have a corresponding mask PNG in lora_mask_dir.
            import toml  # lazy import to avoid dependency when not needed

            dataset_cfg = {
                "datasets": [
                    {
                        "subsets": [
                            {
                                "image_dir": str(dataset_path),
                                "conditioning_data_dir": str(effective_lora_mask_dir),
                                "caption_extension": ".caption",
                                "num_repeats": int(dataset_repeats),
                            }
                        ]
                    }
                ]
            }
            dataset_cfg_path = meta_dir / "dataset_masked_loss_lora.toml"
            dataset_cfg_path.parent.mkdir(parents=True, exist_ok=True)
            toml.dump(dataset_cfg, dataset_cfg_path.open("w", encoding="utf-8"))

            accel += [
                f"--dataset_config={str(dataset_cfg_path)}",
                "--masked_loss",
            ]
        else:
            # Standard fine-tuning-style dataset using precomputed latents.
            accel += [
                f"--in_json={meta_lat_json}",
                f"--train_data_dir={str(dataset_path)}",
            ]

        accel += [
            f"--output_dir={str(out_dir)}",
            f"--resolution={resolution}",
            f"--train_batch_size={train_batch_size}",
            f"--gradient_accumulation_steps={gradient_accumulation_steps}",
            f"--dataset_repeats={dataset_repeats}",
            "--enable_bucket",
            "--keep_tokens=77",
            f"--sample_every_n_steps={sample_every_n_steps}",
            f"--sample_sampler={sample_sampler}",
            f"--sample_prompts={prompts}",
            f"--learning_rate={lora_lr}",
            "--max_grad_norm=1",
            max_train_flag,
            f"--lr_warmup_steps={lr_warmup_steps}",
            save_every_flag,
            f"--noise_offset={noise_offset}",
            "--save_model_as=safetensors",
            f"--optimizer_type={optimizer_type}",
            "--gradient_checkpointing" if gradient_checkpointing else "",
            f"--min_snr_gamma={min_snr_gamma}",
            f"--lr_scheduler={lr_scheduler}",
            f"--logging_dir={logs_dir}",
            *attn_flags,
            f"--seed={seed}",
            # LoRA-specific
            "--network_module=networks.lora",
            f"--network_dim={network_dim}",
            f"--network_alpha={network_alpha}",
        ]
        if lora_network_weights is not None:
            accel += [f"--network_weights={str(lora_network_weights)}"]
            if lora_dim_from_weights:
                accel += ["--dim_from_weights"]
        if bool(cfg.get('lora_train_text_encoder', True)):
            accel += ["--network_train_text_encoder_only"] if bool(cfg.get('lora_te_only', False)) else []
        if bool(cfg.get('lora_train_unet_only', False)):
            accel += ["--network_train_unet_only"]
    else:
        raise SystemExit(f"Unknown training_mode='{training_mode}' (expected 'full' or 'lora')")
    # Remove any empty args from optional flags
    accel = [a for a in accel if a]

    # Optional passthrough of extra kohya training args from YAML (use with care).
    # Example: extra_train_args: ["--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk"]
    extra_args = cfg.get('extra_train_args', None)
    if extra_args is None:
        extra_args = cfg.get('train_extra_args', None)
    if extra_args:
        if isinstance(extra_args, str):
            extra_args = [extra_args]
        if isinstance(extra_args, (list, tuple)):
            accel += list(map(str, extra_args))

    wandb_key = args.wandb_api_key or cfg.get('wandb_api_key') or os.environ.get('WANDB_API_KEY')
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key
        accel += ["--log_with=wandb", f"--wandb_run_name={out_dir.name}"]
    else:
        accel += ["--log_with=tensorboard"]

    # MLflow tracking (optional)
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'file:./mlruns')
    experiment_name = cfg.get('mlflow_experiment', os.environ.get('MLFLOW_EXPERIMENT', 'synthetic_finetune'))
    run_name = f"ft_{seed_name}___{fold_number}"
    params_log = {
        'seed_name': seed_name,
        'fold_number': fold_number,
        'dataset_path': str(dataset_path),
        'base_model_path': base_model_path,
        'out_dir': str(out_dir),
        'kohya_dir': str(SD),
        'UNET_LR': UNET_LR,
        'TE_LR': TE_LR,
        'resolution': resolution,
        'train_batch_size': train_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'dataset_repeats': dataset_repeats,
        'latent_batch_size': int(cfg.get('latent_batch_size', 8)),
        'prepare_latents_mixed_precision': prepare_latents_mixed_precision,
        'prepare_latents_nan_retries': prepare_latents_nan_retries,
        'sample_every_n_steps': sample_every_n_steps,
        'sample_steps': sample_steps,
        'sample_guidance_scale': sample_guidance_scale,
        'sample_prompts_per_class': sample_prompts_per_class,
        'sample_seed_mode': sample_seed_mode,
        'sample_seed': sample_seed,
        'sample_seed_base': sample_seed_base,
        'sample_denoising_strength': sample_denoising_strength,
        'sample_mask_strength': sample_mask_strength,
        'sample_mask_blur_radius': sample_mask_blur_radius,
        'sample_sampler': sample_sampler,
        'sample_use_class_descriptors': sample_use_class_descriptors,
        'caption_use_class_descriptors': caption_use_class_descriptors,
        'max_train_epochs': max_train_epochs,
        'max_train_steps': max_train_steps,
        'lr_warmup_steps': lr_warmup_steps,
        'save_every_n_epochs': save_every_n_epochs,
        'save_every_n_steps': save_every_n_steps,
        'noise_offset': noise_offset,
        'lora_mask_feather_radius': lora_mask_feather_radius,
        'lora_mask_strength': lora_mask_strength,
        'optimizer_type': optimizer_type,
        'gradient_checkpointing': gradient_checkpointing,
        'min_snr_gamma': min_snr_gamma,
        'lr_scheduler': lr_scheduler,
        'mixed_precision': mixed_precision,
        'seed': seed,
    }
    with mlflow_run(enabled=True, tracking_uri=tracking_uri, experiment_name=experiment_name,
                    run_name=run_name, params=params_log, tags={'task': 'sd_finetune'}) as active_run:
        # Log token distribution and prompts as small artifacts
        try:
            token_counts_json = json.dumps({k: int(v) for k, v in token_counts.items()}, indent=2)
            tmp_meta = out_dir / 'run_info.json'
            ensure_dir(out_dir)
            tmp_meta.write_text(json.dumps({
                'seed_name': seed_name,
                'fold_number': fold_number,
                'dataset_path': str(dataset_path),
                'meta_cap_json': str(meta_cap_json),
                'meta_lat_json': str(meta_lat_json),
                'labels_csv': str(CSV_PATH),
            }, indent=2), encoding='utf-8')
            if active_run is not None:
                import mlflow
                mlflow.log_text(token_counts_json, 'token_counts.json')
                if prompts.exists():
                    mlflow.log_artifact(str(prompts), artifact_path='prompts')
                mlflow.log_artifact(str(tmp_meta))
        except Exception:
            pass

        # Launch training
        run(accel, cwd=SD, env=env)

        # Optionally log produced model filenames (not the heavy artifacts themselves)
        try:
            if active_run is not None:
                import mlflow
                ckpts = [p.name for p in out_dir.glob('*.safetensors')]
                mlflow.log_dict({'checkpoints': ckpts}, 'checkpoints_index.json')
        except Exception:
            pass


# ───────────────────── CLI ─────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SD/SDXL full finetuning with .env, YAML config, MLflow")
    ap.add_argument("--project_path", required=False, help="Project root (defaults to repo root)")
    ap.add_argument("--seed_name", required=False)
    ap.add_argument("--fold_number", required=False)
    ap.add_argument("--base_model_path", required=False, help="Path to base model checkpoint")
    ap.add_argument("--dataset_subdir", required=False, help="Override subfolder under data/artifacts/…")
    ap.add_argument("--wandb_api_key", required=False)
    ap.add_argument("--config", required=False, help="Path to YAML/JSON config")
    main(ap.parse_args())
