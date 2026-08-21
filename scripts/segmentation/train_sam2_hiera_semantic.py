#!/usr/bin/env python3
"""Train a SAM2.1 Hiera semantic segmenter on simulation tiles."""

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
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
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
    ap.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Override training.epochs (use with --resume to extend a run)",
    )
    ap.add_argument(
        "--resume",
        default="",
        metavar="CHECKPOINT",
        help="Resume from last.pt/best.pt; increase training.epochs to extend a completed run",
    )
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


class FullCoverageWeightedSampler(Sampler[int]):
    """Visit every tile once, then mix in weighted rare-class extras."""

    def __init__(self, weights: torch.Tensor, *, num_samples: int, seed: int) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double).cpu()
        self.dataset_size = int(self.weights.numel())
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        if self.dataset_size <= 0:
            raise ValueError("weights must not be empty")
        if self.num_samples < self.dataset_size:
            raise ValueError("num_samples must be >= dataset size for full coverage")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        base = torch.randperm(self.dataset_size, generator=generator)
        extra_count = self.num_samples - self.dataset_size
        if extra_count > 0:
            extras = torch.multinomial(
                self.weights,
                extra_count,
                replacement=True,
                generator=generator,
            )
            indices = torch.cat([base, extras])
        else:
            indices = base
        order = torch.randperm(indices.numel(), generator=generator)
        return iter(indices[order].tolist())

    def __len__(self) -> int:
        return self.num_samples


def _lr_lambda(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    denom = max(1, total_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / denom))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_resumed_learning_rates(optimizer, scheduler, *, completed_epochs: int,
                                total_epochs: int, warmup_epochs: int) -> None:
    """Set the LR for the next epoch using the current target epoch count.

    Recomputing the factor is important when a completed cosine schedule is
    extended, because the checkpoint's optimizer LR is otherwise zero.
    """
    factor = _lr_lambda(completed_epochs, total_epochs, warmup_epochs)
    for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
        group["lr"] = float(base_lr) * factor
    scheduler.last_epoch = int(completed_epochs)
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]


def _resume_history(output_dir: Path, completed_epochs: int) -> list[Dict[str, object]]:
    history_path = output_dir / "history.json"
    if not history_path.is_file():
        return []
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return [record for record in history if int(record.get("epoch", 0)) <= completed_epochs]


def _run_epoch(*, model, loader, device, optimizer, class_weights, num_classes, ce_weight, dice_weight,
               exclude_background_from_dice, supported_min_gt_pixels, amp, grad_clip_norm,
               grad_accum_steps, max_batches, desc) -> Dict[str, object]:
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    total_loss = total_ce = total_dice = 0.0
    batches = 0
    iterator = tqdm(loader, desc=desc, leave=False)
    accumulation = max(1, int(grad_accum_steps))
    effective_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(iterator):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        output_size = (int(labels.shape[-2]), int(labels.shape[-1]))
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
                group_start = (batch_idx // accumulation) * accumulation
                group_size = min(accumulation, effective_batches - group_start)
                (loss / group_size).backward()
                should_step = (batch_idx + 1) % accumulation == 0 or (batch_idx + 1) == effective_batches
                if should_step:
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
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
        "optimizer_steps": math.ceil(batches / accumulation) if training else 0,
    })
    return metrics


def _save_checkpoint(path: Path, *, model, optimizer, scheduler, epoch, config, class_weights,
                     val_metrics, best_score, early_stopping_bad_epochs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_kwargs": model.checkpoint_model_kwargs(),
        "encoder_input_size": int(model.encoder_input_size),
        "class_names": CLASS_NAME_BY_ID,
        "class_weights": class_weights.detach().cpu(),
        "training_config": config,
        "val_metrics": val_metrics,
        "best_score": float(best_score),
        "early_stopping_bad_epochs": int(early_stopping_bad_epochs),
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

    resume_path = Path(args.resume).resolve() if args.resume else None
    resume_checkpoint = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise SystemExit(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)

    seed = int(train_cfg.get("seed", 42))
    _seed_everything(seed)
    device = _device_from_config(str(train_cfg.get("device", "auto")))
    num_classes = int(model_cfg.get("num_classes", 12))
    print(f"Device: {device}")
    print(f"SAM2 model: {model_cfg.get('sam2_model_id', 'local checkpoint')}")

    configured_model_kwargs = {
        "num_classes": num_classes,
        "sam2_config": str(model_cfg.get("sam2_config", "configs/sam2.1/sam2.1_hiera_s.yaml")),
        "sam2_model_id": model_cfg.get("sam2_model_id") or None,
        "decoder_channels": int(model_cfg.get("decoder_channels", 192)),
    }
    model_kwargs = configured_model_kwargs
    if resume_checkpoint is not None:
        model_kwargs = dict(resume_checkpoint["model_kwargs"])
        for key in ("num_classes", "sam2_config", "sam2_model_id", "decoder_channels"):
            if model_kwargs.get(key) != configured_model_kwargs.get(key):
                raise SystemExit(
                    f"Resume checkpoint model mismatch for {key}: "
                    f"checkpoint={model_kwargs.get(key)!r}, config={configured_model_kwargs.get(key)!r}"
                )
    model = SAM2HieraSemanticSegmenter(
        num_classes=int(model_kwargs["num_classes"]),
        sam2_config=str(model_kwargs["sam2_config"]),
        sam2_checkpoint=None if resume_checkpoint is not None else model_cfg.get("sam2_checkpoint") or None,
        sam2_model_id=model_kwargs.get("sam2_model_id"),
        load_pretrained=resume_checkpoint is None,
        decoder_channels=int(model_kwargs["decoder_channels"]),
    )
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"], strict=True)
    encoder_size = int(model.encoder_input_size)
    if int(data_cfg.get("encoder_size", encoder_size)) != encoder_size:
        print(f"[WARN] Config encoder_size={data_cfg.get('encoder_size')} but official SAM2 model uses {encoder_size}; using {encoder_size}.")

    train_split = str(data_cfg.get("train_split", "train"))
    val_split = str(data_cfg.get("val_split", "val"))
    train_ds = HistosegSimulationTileDataset(
        tiles_root=data_cfg["tiles_root"], splits_csv=data_cfg["splits_csv"], split=train_split,
        encoder_size=encoder_size, augment=True, seed=seed,
        augmentation_strength=float(aug_cfg.get("strength", 1.0)),
        augmentation_config=aug_cfg,
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
        num_samples = max(len(train_ds), int(round(len(train_ds) * float(sampling_cfg.get("epoch_multiplier", 1.0)))))
        full_coverage = bool(sampling_cfg.get("full_coverage", False))
        if full_coverage:
            sampler = FullCoverageWeightedSampler(sample_weights, num_samples=num_samples, seed=seed + 101)
        else:
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=num_samples,
                replacement=True,
                generator=torch.Generator().manual_seed(seed + 101),
            )
        print(
            f"Rare-class sampler ({'full coverage + weighted extras' if full_coverage else 'weighted replacement'}): "
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
    start_epoch = int(resume_checkpoint.get("epoch", 0)) if resume_checkpoint is not None else 0
    model.set_encoder_trainable(start_epoch >= freeze_encoder_epochs)
    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder_parameters()), "lr": float(train_cfg.get("encoder_lr", 1e-5))},
        {"params": list(model.decoder_parameters()), "lr": float(train_cfg.get("decoder_lr", 1e-4))},
    ], weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])

    epochs = int(args.epochs) if int(args.epochs) > 0 else int(train_cfg.get("epochs", 5))
    train_cfg["epochs"] = epochs
    config["training"] = train_cfg
    warmup_epochs = int(train_cfg.get("warmup_epochs", 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: _lr_lambda(e, epochs, warmup_epochs)
    )
    output_dir = Path(str(train_cfg.get("output_dir", "outputs/segmentation/sam2.1_hiera_small_poc"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if start_epoch >= epochs:
        raise SystemExit(
            f"Checkpoint already completed epoch {start_epoch}; set training.epochs above {start_epoch} to continue"
        )
    if resume_checkpoint is not None:
        if "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        _set_resumed_learning_rates(
            optimizer,
            scheduler,
            completed_epochs=start_epoch,
            total_epochs=epochs,
            warmup_epochs=warmup_epochs,
        )
        print(
            f"Resumed {resume_path} after epoch {start_epoch}; "
            f"continuing through epoch {epochs} with encoder_lr={optimizer.param_groups[0]['lr']:.3e} "
            f"decoder_lr={optimizer.param_groups[1]['lr']:.3e}"
        )

    supported_min_gt_pixels = int(metrics_cfg.get("supported_min_gt_pixels", 1024))
    checkpoint_metric = str(metrics_cfg.get("checkpoint_metric", "macro_dice_supported"))
    history = _resume_history(output_dir, start_epoch)
    historical_scores = [
        float(record["val"].get(checkpoint_metric, record["val"]["macro_dice_present"]))
        for record in history
    ]
    best_score = max(historical_scores, default=float(resume_checkpoint.get("best_score", -float("inf")))
                     if resume_checkpoint is not None else -float("inf"))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    bad_epochs = int(resume_checkpoint.get("early_stopping_bad_epochs", 0)) if resume_checkpoint is not None else 0
    for epoch in range(start_epoch, epochs):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
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
            grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
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
                grad_accum_steps=1,
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
        improved = score > best_score + early_stopping_min_delta
        if improved:
            best_score = score
            bad_epochs = 0
        else:
            bad_epochs += 1
        _save_checkpoint(output_dir / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                          epoch=epoch + 1, config=config, class_weights=class_weights,
                          val_metrics=val_metrics, best_score=best_score,
                          early_stopping_bad_epochs=bad_epochs)
        if improved:
            _save_checkpoint(output_dir / "best.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                              epoch=epoch + 1, config=config, class_weights=class_weights,
                              val_metrics=val_metrics, best_score=best_score,
                              early_stopping_bad_epochs=bad_epochs)
        if early_stopping_patience > 0 and bad_epochs >= early_stopping_patience:
            print(
                f"Early stopping after {bad_epochs} epochs without a "
                f"{early_stopping_min_delta:g} improvement"
            )
            break

    print(f"Best validation {checkpoint_metric}: {best_score:.4f}")
    print(f"Checkpoints: {output_dir / 'best.pt'} and {output_dir / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
