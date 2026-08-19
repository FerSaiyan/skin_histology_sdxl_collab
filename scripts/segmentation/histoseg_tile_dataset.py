#!/usr/bin/env python3
"""Dataset helpers for Histo-Seg simulation-tile semantic segmentation."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

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


def _jitter_rgb(image: Image.Image, rng: random.Random) -> Image.Image:
    # Mild H&E appearance augmentation without changing anatomy.
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.90, 1.10))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.92, 1.08))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.90, 1.10))
    return image


class HistosegSimulationTileDataset(Dataset):
    """Aligned RGB + semantic label tiles from ``tiles_for_simulation``.

    RGB is resized to the SAM2 encoder resolution (1024 by default) using
    bilinear interpolation. Ground-truth labels stay at their native 512x512
    resolution; the semantic head resizes logits back to the label shape.
    """

    def __init__(
        self,
        *,
        tiles_root: str | Path,
        splits_csv: str | Path,
        split: str,
        encoder_size: int = 1024,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.tiles_root = Path(tiles_root).resolve()
        self.splits_csv = Path(splits_csv).resolve()
        self.split = split
        self.encoder_size = int(encoder_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.rows = load_split_rows(self.splits_csv, split)

        if self.encoder_size <= 0:
            raise ValueError("encoder_size must be > 0")

    def __len__(self) -> int:
        return len(self.rows)

    def _rng(self, index: int) -> random.Random:
        # Deterministic per sample/worker; epoch-level variation comes from shuffled order.
        worker = torch.utils.data.get_worker_info()
        worker_seed = worker.seed if worker is not None else self.seed
        return random.Random(int(worker_seed) + int(index) * 1_000_003)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
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

        rng = self._rng(index)
        if self.augment:
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                labels = np.ascontiguousarray(np.fliplr(labels))
            image = _jitter_rgb(image, rng)

        image = image.resize((self.encoder_size, self.encoder_size), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        label_tensor = torch.from_numpy(np.ascontiguousarray(labels)).long()

        return {
            "image": image_tensor,
            "label": label_tensor,
            "tile_id": row["tile_id"],
            "slide_id": row["slide_id"],
            "rgb_path": str(rgb_path),
            "label_path": str(label_path),
        }


def compute_pixel_class_counts(
    *,
    tiles_root: str | Path,
    splits_csv: str | Path,
    split: str,
    num_classes: int = 12,
) -> np.ndarray:
    root = Path(tiles_root).resolve()
    rows = load_split_rows(Path(splits_csv).resolve(), split)
    counts = np.zeros(int(num_classes), dtype=np.int64)
    for row in rows:
        labels = np.load(_resolve(root, row["label_id_npy_path"]), mmap_mode="r")
        uniq, c = np.unique(labels, return_counts=True)
        for cid, count in zip(uniq.tolist(), c.tolist()):
            cid = int(cid)
            if 0 <= cid < num_classes:
                counts[cid] += int(count)
    return counts


def sqrt_inverse_frequency_weights(counts: np.ndarray) -> torch.Tensor:
    """Stable class weights for heavily imbalanced histology labels.

    Uses sqrt(total / class_count), normalized so present classes have mean 1.
    Classes absent from the training set receive weight 0.
    """
    counts = np.asarray(counts, dtype=np.float64)
    present = counts > 0
    weights = np.zeros_like(counts, dtype=np.float64)
    if present.any():
        total = float(counts[present].sum())
        weights[present] = np.sqrt(total / counts[present])
        weights[present] /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)
