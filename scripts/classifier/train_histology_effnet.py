#!/usr/bin/env python
"""
Train a binary histology classifier (cancer vs non_cancer) using EfficientNet.

This checkpoint is compatible with build_roi_masks_gradcam.py via create_model().
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exp.config import load_config
from src.oral_lesions.models.factory import create_model


Image.MAX_IMAGE_PIXELS = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HistologySliceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: Path, transform: transforms.Compose):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["filename"]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = 1 if str(row["coarse_label"]) == "cancer" else 0
        return x, y


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def split_by_volume(df: pd.DataFrame, seed: int, val_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    groups = df["volume_id"].astype(str)
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_y: List[int] = []
    all_p: List[int] = []
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0.0
    n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            preds = torch.argmax(logits, dim=1)

            bs = x.size(0)
            loss_sum += float(loss.item()) * bs
            n += bs

            all_y.extend(y.cpu().tolist())
            all_p.extend(preds.cpu().tolist())

    acc = float(np.mean(np.array(all_y) == np.array(all_p))) if all_y else 0.0
    f1 = float(f1_score(all_y, all_p, average="binary", zero_division=0)) if all_y else 0.0
    return {"val_loss": loss_sum / max(n, 1), "val_acc": acc, "val_f1": f1}


def train(args) -> None:
    cfg = load_config(args.model_config)
    model_cfg = cfg["models"][0]
    img_size = int(model_cfg.get("img_size", args.img_size))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    df = pd.read_csv(args.csv)
    train_df, val_df = split_by_volume(df, seed=args.seed, val_fraction=args.val_fraction)

    train_tf, val_tf = build_transforms(img_size)
    train_ds = HistologySliceDataset(train_df, Path(args.image_root), train_tf)
    val_ds = HistologySliceDataset(val_df, Path(args.image_root), val_tf)

    train_labels = np.array([1 if str(x) == "cancer" else 0 for x in train_df["coarse_label"].tolist()])
    class_counts = np.bincount(train_labels, minlength=2).astype(float)
    class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(weights=torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(args.batch_size, 2), shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = create_model(model_cfg, num_classes=2, device=device).to(device)

    # Warmup stage: head-only
    if hasattr(model, "enet"):
        for p in model.enet.parameters():
            p.requires_grad = False

    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=args.head_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and args.mixed_precision))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / args.checkpoint_name
    history_path = out_dir / "train_history.json"

    best_f1 = -1.0
    history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        # Unfreeze backbone after warmup
        if epoch == args.head_warmup_epochs and hasattr(model, "enet"):
            for p in model.enet.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(
                [
                    {"params": model.enet.parameters(), "lr": args.backbone_lr},
                    {"params": model.myfc.parameters(), "lr": args.head_lr},
                ],
                weight_decay=1e-4,
            )

        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda" and args.mixed_precision)):
                logits = model(x)
                loss = criterion(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            bs = x.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs

        tr_loss = running_loss / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": float(epoch + 1),
            "train_loss": float(tr_loss),
            **{k: float(v) for k, v in val_metrics.items()},
        }
        history.append(row)
        print(f"[Epoch {epoch+1}/{args.epochs}] train_loss={tr_loss:.4f} val_loss={val_metrics['val_loss']:.4f} val_acc={val_metrics['val_acc']:.4f} val_f1={val_metrics['val_f1']:.4f}")

        if val_metrics["val_f1"] > best_f1:
            best_f1 = val_metrics["val_f1"]
            torch.save(model.state_dict(), best_ckpt)

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    split_meta = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_volumes": sorted(train_df["volume_id"].astype(str).unique().tolist()),
        "val_volumes": sorted(val_df["volume_id"].astype(str).unique().tolist()),
        "best_val_f1": float(best_f1),
        "checkpoint": str(best_ckpt),
    }
    (out_dir / "split_and_metrics.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint: {best_ckpt}")
    print(f"Best val_f1: {best_f1:.4f}")


def parse_args():
    ap = argparse.ArgumentParser(description="Train histology binary classifier (EffNet transfer learning)")
    ap.add_argument("--csv", default="data/processed/histoseg_pairs.csv")
    ap.add_argument("--image-root", default="data/raw/histo_seg_v2")
    ap.add_argument("--model-config", default="configs/studies/histoseg_effnet_b0_binary.yaml")
    ap.add_argument("--output-dir", default="outputs/classifiers")
    ap.add_argument("--checkpoint-name", default="histoseg_effnet_b0_binary.pth")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--head-warmup-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed-precision", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
