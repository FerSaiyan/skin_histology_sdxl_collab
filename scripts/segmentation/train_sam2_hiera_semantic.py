#!/usr/bin/env python3
"""Train a SAM2.1 Hiera Small semantic segmenter on simulation tiles."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from histoseg_tile_dataset import (  # noqa: E402
    CLASS_NAME_BY_ID,
    HistosegSimulationTileDataset,
    compute_pixel_class_counts,
    compute_tile_sampling_weights,
    frequency_class_weights,
)
from sam2_hiera_semantic import SAM2HieraSemanticSegmenter  # noqa: E402
from segmentation_metrics import combined_ce_dice_loss, metrics_from_confusion, update_confusion_matrix  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/segmentation/sam2.1_hiera_small_histoseg_poc.yaml")
    ap.add_argument("--max-train-batches", type=int, default=0)
    ap.add_argument("--max-val-batches", type=int, default=0)
    return ap.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_config(value: str) -> torch.device:
    value = str(value or "auto").lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Config requested CUDA but torch.cuda.is_available() is False")
    return torch.device(value)


def _make_loader(dataset, *, batch_size, num_workers, shuffle, device, sampler=None, seed=42):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
        generator=generator,
    )


def _lr_lambda(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    denom = max(1, total_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / denom))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _run_epoch(*, model, loader, device, optimizer, class_weights, num_classes, ce_weight, dice_weight,
               exclude_background_from_dice, supported_min_gt_pixels, amp, grad_clip_norm, max_batches, desc) -> Dict[str, object]:
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    total_loss = total_ce = total_dice = 0.0
    batches = 0
    iterator = tqdm(loader, desc=desc, leave=False)
    for batch_idx, batch in enumerate(iterator):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        output_size = (int(labels.shape[-2]), int(labels.shape[-1]))
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast_enabled = bool(amp and device.type == "cuda")
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                                enabled=autocast_enabled):
                logits = model(images, output_size=output_size)
                loss, parts = combined_ce_dice_loss(
                    logits, labels, class_weights=class_weights, num_classes=num_classes,
                    ce_weight=ce_weight, dice_weight=dice_weight,
                    exclude_background_from_dice=exclude_background_from_dice,
                )
            if training:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        update_confusion_matrix(confusion, logits.detach(), labels, num_classes=num_classes)
        total_loss += float(loss.detach())
        total_ce += float(parts["ce"])
        total_dice += float(parts["dice_loss"])
        batches += 1
        iterator.set_postfix(loss=f"{total_loss / batches:.4f}")
    metrics = metrics_from_confusion(
        confusion,
        exclude_background_from_macro=True,
        supported_min_gt_pixels=int(supported_min_gt_pixels),
    )
    metrics.update({
        "loss": total_loss / max(batches, 1),
        "ce_loss": total_ce / max(batches, 1),
        "dice_loss": total_dice / max(batches, 1),
        "batches": batches,
    })
    return metrics


def _save_checkpoint(path: Path, *, model, optimizer, epoch, config, class_weights, val_metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_kwargs": model.checkpoint_model_kwargs(),
        "encoder_input_size": int(model.encoder_input_size),
        "class_names": CLASS_NAME_BY_ID,
        "class_weights": class_weights.detach().cpu(),
        "training_config": config,
        "val_metrics": val_metrics,
    }, path)


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    config: Dict[str, object] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = dict(config["data"])
    model_cfg = dict(config["model"])
    train_cfg = dict(config["training"])
    sampling_cfg = dict(config.get("sampling", {}))
    weighting_cfg = dict(config.get("class_weighting", {}))
    aug_cfg = dict(config.get("augmentation", {}))
    metrics_cfg = dict(config.get("metrics", {}))

    seed = int(train_cfg.get("seed", 42))
    _seed_everything(seed)
    device = _device_from_config(str(train_cfg.get("device", "auto")))
    num_classes = int(model_cfg.get("num_classes", 12))
    print(f"Device: {device}")
    print(f"SAM2 model: {model_cfg.get('sam2_model_id', 'local checkpoint')}")

    model = SAM2HieraSemanticSegmenter(
        num_classes=num_classes,
        sam2_config=str(model_cfg.get("sam2_config", "configs/sam2.1/sam2.1_hiera_s.yaml")),
        sam2_checkpoint=model_cfg.get("sam2_checkpoint") or None,
        sam2_model_id=model_cfg.get("sam2_model_id") or None,
        load_pretrained=True,
        decoder_channels=int(model_cfg.get("decoder_channels", 192)),
    )
    encoder_size = int(model.encoder_input_size)
    if int(data_cfg.get("encoder_size", encoder_size)) != encoder_size:
        print(f"[WARN] Config encoder_size={data_cfg.get('encoder_size')} but official SAM2 model uses {encoder_size}; using {encoder_size}.")

    train_split = str(data_cfg.get("train_split", "train"))
    val_split = str(data_cfg.get("val_split", "val"))
    train_ds = HistosegSimulationTileDataset(
        tiles_root=data_cfg["tiles_root"], splits_csv=data_cfg["splits_csv"], split=train_split,
        encoder_size=encoder_size, augment=True, seed=seed,
        augmentation_strength=float(aug_cfg.get("strength", 1.0)),
    )
    val_ds = HistosegSimulationTileDataset(
        tiles_root=data_cfg["tiles_root"], splits_csv=data_cfg["splits_csv"], split=val_split,
        encoder_size=encoder_size, augment=False, seed=seed,
    )

    counts = compute_pixel_class_counts(
        tiles_root=data_cfg["tiles_root"], splits_csv=data_cfg["splits_csv"],
        split=train_split, num_classes=num_classes,
    )
    class_weights = frequency_class_weights(
        counts,
        power=float(weighting_cfg.get("power", 0.25)),
        min_weight=float(weighting_cfg.get("min_weight", 0.5)),
        max_weight=float(weighting_cfg.get("max_weight", 2.5)),
    ).to(device)
    print("Training class pixel counts / CE weights:")
    for cid in range(num_classes):
        print(f"  {cid:2d} {CLASS_NAME_BY_ID.get(cid, f'class_{cid}'):<28} pixels={int(counts[cid]):>12} weight={float(class_weights[cid]):.3f}")

    sampler = None
    if bool(sampling_cfg.get("enabled", False)):
        sample_weights = compute_tile_sampling_weights(
            tiles_root=data_cfg["tiles_root"], splits_csv=data_cfg["splits_csv"], split=train_split,
            class_pixel_counts=counts, num_classes=num_classes,
            rarity_power=float(sampling_cfg.get("rarity_power", 0.25)),
            rare_strength=float(sampling_cfg.get("rare_strength", 1.5)),
            min_class_pixels_in_tile=int(sampling_cfg.get("min_class_pixels_in_tile", 64)),
            max_tile_weight=float(sampling_cfg.get("max_tile_weight", 4.0)),
        )
        num_samples = max(1, int(round(len(train_ds) * float(sampling_cfg.get("epoch_multiplier", 1.0)))))
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,
            generator=torch.Generator().manual_seed(seed + 101),
        )
        print(
            "Rare-class sampler: "
            f"num_samples={num_samples}, weight[min/mean/max]="
            f"{float(sample_weights.min()):.3f}/{float(sample_weights.mean()):.3f}/{float(sample_weights.max()):.3f}"
        )

    batch_size = int(train_cfg.get("batch_size", 1))
    workers = int(train_cfg.get("num_workers", 4))
    train_loader = _make_loader(train_ds, batch_size=batch_size, num_workers=workers,
                                shuffle=sampler is None, device=device, sampler=sampler, seed=seed)
    val_loader = _make_loader(val_ds, batch_size=batch_size, num_workers=workers,
                              shuffle=False, device=device, sampler=None, seed=seed + 1)

    model = model.to(device)
    freeze_encoder_epochs = int(train_cfg.get("freeze_encoder_epochs", 1))
    model.set_encoder_trainable(freeze_encoder_epochs <= 0)
    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder_parameters()), "lr": float(train_cfg.get("encoder_lr", 1e-5))},
        {"params": list(model.decoder_parameters()), "lr": float(train_cfg.get("decoder_lr", 1e-4))},
    ], weight_decay=float(train_cfg.get("weight_decay", 0.01)))

    epochs = int(train_cfg.get("epochs", 5))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: _lr_lambda(e, epochs, warmup_epochs)
    )
    output_dir = Path(str(train_cfg.get("output_dir", "outputs/segmentation/sam2.1_hiera_small_poc"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    supported_min_gt_pixels = int(metrics_cfg.get("supported_min_gt_pixels", 1024))
    checkpoint_metric = str(metrics_cfg.get("checkpoint_metric", "macro_dice_supported"))
    history = []
    best_score = -float("inf")
    for epoch in range(epochs):
        if epoch == freeze_encoder_epochs and freeze_encoder_epochs > 0:
            model.set_encoder_trainable(True)
            print(f"Epoch {epoch + 1}: unfroze SAM2.1 Hiera image encoder")
        train_metrics = _run_epoch(
            model=model, loader=train_loader, device=device, optimizer=optimizer,
            class_weights=class_weights, num_classes=num_classes,
            ce_weight=float(train_cfg.get("ce_weight", 1.0)),
            dice_weight=float(train_cfg.get("dice_weight", 1.0)),
            exclude_background_from_dice=bool(train_cfg.get("exclude_background_from_dice", True)),
            supported_min_gt_pixels=supported_min_gt_pixels,
            amp=bool(train_cfg.get("amp", True)), grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
            max_batches=int(args.max_train_batches), desc=f"train {epoch + 1}/{epochs}",
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model=model, loader=val_loader, device=device, optimizer=None,
                class_weights=class_weights, num_classes=num_classes,
                ce_weight=float(train_cfg.get("ce_weight", 1.0)),
                dice_weight=float(train_cfg.get("dice_weight", 1.0)),
                exclude_background_from_dice=bool(train_cfg.get("exclude_background_from_dice", True)),
                supported_min_gt_pixels=supported_min_gt_pixels,
                amp=bool(train_cfg.get("amp", True)), grad_clip_norm=0.0,
                max_batches=int(args.max_val_batches), desc=f"val   {epoch + 1}/{epochs}",
            )
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "encoder_trainable": any(p.requires_grad for p in model.encoder_parameters()),
            "lr_encoder": optimizer.param_groups[0]["lr"],
            "lr_decoder": optimizer.param_groups[1]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        score = float(val_metrics.get(checkpoint_metric, val_metrics["macro_dice_present"]))
        print(
            f"Epoch {epoch + 1:02d}: train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"present={val_metrics['macro_dice_present']:.4f} fixed={val_metrics['macro_dice_fixed']:.4f} "
            f"supported={val_metrics['macro_dice_supported']:.4f} select({checkpoint_metric})={score:.4f}"
        )
        _save_checkpoint(output_dir / "last.pt", model=model, optimizer=optimizer,
                         epoch=epoch + 1, config=config, class_weights=class_weights, val_metrics=val_metrics)
        if score > best_score:
            best_score = score
            _save_checkpoint(output_dir / "best.pt", model=model, optimizer=optimizer,
                             epoch=epoch + 1, config=config, class_weights=class_weights, val_metrics=val_metrics)

    print(f"Best validation {checkpoint_metric}: {best_score:.4f}")
    print(f"Checkpoints: {output_dir / 'best.pt'} and {output_dir / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
