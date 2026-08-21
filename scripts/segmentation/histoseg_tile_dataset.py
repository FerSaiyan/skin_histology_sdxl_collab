#!/usr/bin/env python3
"""Dataset helpers for Histo-Seg simulation-tile semantic segmentation."""

from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

try:
    import h5py
except Exception:  # pragma: no cover - only required by sharded manifests
    h5py = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAME_BY_ID: Dict[int, str] = {
    0: "background",
    1: "epidermis",
    2: "reticular_dermis",
    3: "papillary_dermis",
    4: "dermis",
    5: "keratin",
    6: "inflammation",
    7: "hair_follicles",
    8: "glands",
    9: "basal_cell_carcinoma",
    10: "squamous_cell_carcinoma",
    11: "intraepidermal_carcinoma",
}


def _resolve(root: Path, path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else root / p


def load_split_rows(splits_csv: Path, split: str) -> List[Dict[str, str]]:
    with splits_csv.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("split") == split]
    if not rows:
        raise RuntimeError(f"No rows for split={split!r} in {splits_csv}")
    return rows


def _jitter_rgb(image: Image.Image, rng: random.Random, strength: float) -> Image.Image:
    """Controlled H&E-like appearance perturbation without changing anatomy."""
    s = max(0.0, float(strength))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(1.0 - 0.15 * s, 1.0 + 0.15 * s))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(1.0 - 0.12 * s, 1.0 + 0.12 * s))
    image = ImageEnhance.Color(image).enhance(rng.uniform(1.0 - 0.18 * s, 1.0 + 0.18 * s))
    arr = np.asarray(image, dtype=np.float32)
    od = -np.log((arr + 1.0) / 256.0)
    gains = np.array([rng.uniform(1.0 - 0.12 * s, 1.0 + 0.12 * s) for _ in range(3)], dtype=np.float32)
    od *= gains.reshape(1, 1, 3)
    arr = np.clip(np.exp(-od) * 256.0 - 1.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _color_jitter(image: Image.Image, rng: random.Random, *, brightness: float,
                  contrast: float, saturation: float) -> Image.Image:
    operations = [
        lambda value: ImageEnhance.Brightness(value).enhance(rng.uniform(1.0 - brightness, 1.0 + brightness)),
        lambda value: ImageEnhance.Contrast(value).enhance(rng.uniform(1.0 - contrast, 1.0 + contrast)),
        lambda value: ImageEnhance.Color(value).enhance(rng.uniform(1.0 - saturation, 1.0 + saturation)),
    ]
    rng.shuffle(operations)
    for operation in operations:
        image = operation(image)
    return image


def _sam2_finetune_augment(image: Image.Image, labels: np.ndarray, rng: random.Random,
                           config: Dict[str, object]) -> tuple[Image.Image, np.ndarray]:
    """Adapt Meta's paired SAM2 transforms to a single semantic tile."""
    if rng.random() < float(config.get("horizontal_flip_prob", 0.5)):
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        labels = np.ascontiguousarray(np.fliplr(labels))

    if rng.random() < float(config.get("affine_probability", 1.0)):
        degrees = float(config.get("affine_degrees", 25.0))
        shear = float(config.get("affine_shear", 20.0))
        translate = float(config.get("affine_translate", 0.0))
        scale_min = float(config.get("scale_min", 1.0))
        scale_max = float(config.get("scale_max", 1.0))
        angle, translations, scale, shear_values = T.RandomAffine.get_params(
            degrees=(-degrees, degrees),
            translate=(translate, translate),
            scale_ranges=(scale_min, scale_max),
            shears=(-shear, shear),
            img_size=image.size,
        )
        fill = tuple(int(value) for value in config.get("image_fill", [123, 116, 103]))
        image = TF.affine(
            image,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=shear_values,
            interpolation=InterpolationMode.BILINEAR,
            fill=fill,
        )
        label_image = Image.fromarray(labels.astype(np.uint8), mode="L")
        label_image = TF.affine(
            label_image,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=shear_values,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        labels = np.asarray(label_image, dtype=np.int64)

    image = _color_jitter(
        image,
        rng,
        brightness=float(config.get("brightness", 0.1)),
        contrast=float(config.get("contrast", 0.03)),
        saturation=float(config.get("saturation", 0.03)),
    )
    image = _color_jitter(
        image,
        rng,
        brightness=float(config.get("secondary_brightness", 0.1)),
        contrast=float(config.get("secondary_contrast", 0.05)),
        saturation=float(config.get("secondary_saturation", 0.05)),
    )
    if rng.random() < float(config.get("grayscale_prob", 0.05)):
        image = ImageOps.grayscale(image).convert("RGB")

    stain_strength = float(config.get("stain_strength", 0.0))
    if stain_strength > 0:
        arr = np.asarray(image, dtype=np.float32)
        optical_density = -np.log((arr + 1.0) / 256.0)
        gains = np.array(
            [rng.uniform(1.0 - stain_strength, 1.0 + stain_strength) for _ in range(3)],
            dtype=np.float32,
        )
        optical_density *= gains.reshape(1, 1, 3)
        arr = np.clip(np.exp(-optical_density) * 256.0 - 1.0, 0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(arr, mode="RGB")
    return image, np.ascontiguousarray(labels)


class HistosegSimulationTileDataset(Dataset):
    """Aligned RGB + semantic label tiles from ``tiles_for_simulation``."""

    def __init__(self, *, tiles_root: str | Path, splits_csv: str | Path, split: str, encoder_size: int = 1024,
                 augment: bool = False, seed: int = 42, augmentation_strength: float = 1.0,
                 augmentation_config: Dict[str, object] | None = None) -> None:
        self.tiles_root = Path(tiles_root).resolve()
        self.splits_csv = Path(splits_csv).resolve()
        self.split = split
        self.encoder_size = int(encoder_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.augmentation_strength = float(augmentation_strength)
        self.augmentation_config = dict(augmentation_config or {})
        self.rows = load_split_rows(self.splits_csv, split)
        self._shards: Dict[str, object] = {}
        if self.encoder_size <= 0:
            raise ValueError("encoder_size must be > 0")

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shards"] = {}
        return state

    def _open_shard(self, path: Path):
        if h5py is None:
            raise RuntimeError("HDF5 tile loading requires h5py; install requirements-segmentation.txt")
        key = str(path)
        shard = self._shards.get(key)
        if shard is None:
            shard = h5py.File(path, "r")
            self._shards[key] = shard
        return shard

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        if row.get("storage") == "hdf5":
            shard_path = _resolve(self.tiles_root, row["shard_path"])
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            shard_index = int(row["shard_index"])
            shard = self._open_shard(shard_path)
            if "rgb_bytes" in shard:
                offsets = shard["rgb_offsets"]
                start, end = int(offsets[shard_index]), int(offsets[shard_index + 1])
                encoded = np.asarray(shard["rgb_bytes"][start:end], dtype=np.uint8).tobytes()
            else:
                encoded = np.asarray(shard["rgb_jpeg"][shard_index], dtype=np.uint8).tobytes()
            image = Image.open(io.BytesIO(encoded)).convert("RGB")
            labels = np.asarray(shard["labels"][shard_index], dtype=np.int64)
            rgb_path = label_path = Path(f"{shard_path}#{shard_index}")
        else:
            rgb_path = _resolve(self.tiles_root, row["rgb_path"])
            label_path = _resolve(self.tiles_root, row["label_id_npy_path"])
            if not rgb_path.is_file():
                raise FileNotFoundError(rgb_path)
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            image = Image.open(rgb_path).convert("RGB")
            labels = np.asarray(np.load(label_path), dtype=np.int64)
        if labels.ndim != 2:
            raise ValueError(f"Expected 2D label map, got {labels.shape} at {label_path}")
        if self.augment:
            # PyTorch seeds Python's random module independently in each worker.
            # Using the advancing worker RNG means repeated tiles receive fresh
            # transforms across epochs instead of index-locked augmentation.
            rng = random
            if self.augmentation_config.get("profile") == "sam2_finetune":
                image, labels = _sam2_finetune_augment(image, labels, rng, self.augmentation_config)
            else:
                if rng.random() < 0.5:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    labels = np.ascontiguousarray(np.fliplr(labels))
                image = _jitter_rgb(image, rng, self.augmentation_strength)
        image = image.resize((self.encoder_size, self.encoder_size), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        label_tensor = torch.from_numpy(np.ascontiguousarray(labels)).long()
        return {"image": image_tensor, "label": label_tensor, "tile_id": row["tile_id"], "slide_id": row["slide_id"],
                "rgb_path": str(rgb_path), "label_path": str(label_path)}


def compute_pixel_class_counts(*, tiles_root: str | Path, splits_csv: str | Path, split: str, num_classes: int = 12) -> np.ndarray:
    root = Path(tiles_root).resolve()
    rows = load_split_rows(Path(splits_csv).resolve(), split)
    counts = np.zeros(int(num_classes), dtype=np.int64)
    for row in rows:
        histogram = row.get("class_histogram", "")
        if histogram:
            class_counts = ((int(cid), int(count)) for cid, count in json.loads(histogram).items())
        else:
            labels = np.load(_resolve(root, row["label_id_npy_path"]), mmap_mode="r")
            uniq, tile_counts = np.unique(labels, return_counts=True)
            class_counts = zip(uniq.tolist(), tile_counts.tolist())
        for cid, count in class_counts:
            cid = int(cid)
            if 0 <= cid < num_classes:
                counts[cid] += int(count)
    return counts


def frequency_class_weights(counts: np.ndarray, *, power: float = 0.25, min_weight: float = 0.5, max_weight: float = 2.5) -> torch.Tensor:
    """Gentle inverse-frequency weights intended to complement oversampling."""
    counts = np.asarray(counts, dtype=np.float64)
    present = counts > 0
    weights = np.zeros_like(counts, dtype=np.float64)
    if present.any():
        total = float(counts[present].sum())
        weights[present] = np.power(total / counts[present], float(power))
        weights[present] /= weights[present].mean()
        weights[present] = np.clip(weights[present], float(min_weight), float(max_weight))
    return torch.tensor(weights, dtype=torch.float32)


def sqrt_inverse_frequency_weights(counts: np.ndarray) -> torch.Tensor:
    """Backward-compatible v1 weighting."""
    return frequency_class_weights(counts, power=0.5, min_weight=0.0, max_weight=float("inf"))


def compute_tile_sampling_weights(*, tiles_root: str | Path, splits_csv: str | Path, split: str,
                                  class_pixel_counts: np.ndarray, num_classes: int = 12,
                                  rarity_power: float = 0.25, rare_strength: float = 1.5,
                                  min_class_pixels_in_tile: int = 64, max_tile_weight: float = 4.0) -> torch.Tensor:
    """Pixel-aware tile weights for rare-class exposure.

    Rare pixels only boost a tile in proportion to sqrt(pixel fraction), so a few
    noisy pixels cannot force maximum oversampling.
    """
    root = Path(tiles_root).resolve()
    rows = load_split_rows(Path(splits_csv).resolve(), split)
    global_counts = np.asarray(class_pixel_counts, dtype=np.float64)
    present = global_counts > 0
    rarity = np.ones(num_classes, dtype=np.float64)
    if present.any():
        median = float(np.median(global_counts[present]))
        rarity[present] = np.maximum(1.0, np.power(median / global_counts[present], float(rarity_power)))
    rarity[0] = 1.0
    weights = np.ones(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        histogram = row.get("class_histogram", "")
        if histogram:
            parsed = {int(cid): int(count) for cid, count in json.loads(histogram).items()}
            class_counts = parsed.items()
            tile_pixels = float(sum(parsed.values()))
        else:
            labels = np.load(_resolve(root, row["label_id_npy_path"]), mmap_mode="r")
            uniq, counts = np.unique(labels, return_counts=True)
            class_counts = zip(uniq.tolist(), counts.tolist())
            tile_pixels = float(labels.size)
        boost = 0.0
        for cid, count in class_counts:
            cid = int(cid)
            count = int(count)
            if cid <= 0 or cid >= num_classes or count < int(min_class_pixels_in_tile):
                continue
            frac_term = np.sqrt(float(count) / tile_pixels)
            boost = max(boost, frac_term * max(0.0, rarity[cid] - 1.0))
        weights[i] = min(float(max_tile_weight), 1.0 + float(rare_strength) * boost)
    weights /= max(float(weights.mean()), 1e-12)
    return torch.tensor(weights, dtype=torch.double)
