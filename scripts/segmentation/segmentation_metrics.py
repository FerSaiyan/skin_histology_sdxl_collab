#!/usr/bin/env python3
"""Losses and metrics for multiclass histology segmentation."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, *, num_classes: int,
                          exclude_background: bool = True, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).to(probs.dtype)
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    present = one_hot.sum(dims) > 0
    if exclude_background and num_classes > 1:
        present[0] = False
    if not torch.any(present):
        return logits.new_tensor(0.0)
    return 1.0 - dice[present].mean()


def combined_ce_dice_loss(logits: torch.Tensor, target: torch.Tensor, *, class_weights: torch.Tensor | None,
                          num_classes: int, ce_weight: float = 1.0, dice_weight: float = 1.0,
                          exclude_background_from_dice: bool = True) -> tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy(logits, target, weight=class_weights)
    dice = dice_loss_from_logits(logits, target, num_classes=num_classes,
                                 exclude_background=exclude_background_from_dice)
    total = float(ce_weight) * ce + float(dice_weight) * dice
    return total, {"ce": float(ce.detach()), "dice_loss": float(dice.detach())}


@torch.no_grad()
def update_confusion_matrix(confusion: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, *, num_classes: int) -> None:
    pred = torch.argmax(logits, dim=1)
    valid = (target >= 0) & (target < num_classes)
    flat = target[valid].to(torch.int64) * num_classes + pred[valid].to(torch.int64)
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    confusion += counts.reshape(num_classes, num_classes).to(confusion.device)


def metrics_from_confusion(confusion: torch.Tensor, *, exclude_background_from_macro: bool = True,
                           supported_min_gt_pixels: int = 1024) -> Dict[str, object]:
    cm = confusion.detach().double().cpu()
    tp = torch.diag(cm)
    gt = cm.sum(dim=1)
    pred = cm.sum(dim=0)
    union = gt + pred - tp
    dice = torch.where(gt + pred > 0, 2.0 * tp / (gt + pred), torch.nan)
    iou = torch.where(union > 0, tp / union, torch.nan)
    precision = torch.where(pred > 0, tp / pred, torch.nan)
    recall = torch.where(gt > 0, tp / gt, torch.nan)

    foreground = torch.ones_like(gt, dtype=torch.bool)
    if exclude_background_from_macro and len(foreground) > 1:
        foreground[0] = False
    present = foreground & (gt > 0)
    supported = foreground & (gt >= int(supported_min_gt_pixels))
    fixed_values = torch.nan_to_num(dice[foreground], nan=0.0)

    macro_present = float(torch.nanmean(dice[present])) if torch.any(present) else float("nan")
    macro_fixed = float(fixed_values.mean()) if torch.any(foreground) else float("nan")
    macro_supported = float(torch.nanmean(dice[supported])) if torch.any(supported) else float("nan")

    return {
        "macro_dice": macro_present,
        "macro_dice_present": macro_present,
        "macro_dice_fixed": macro_fixed,
        "macro_dice_supported": macro_supported,
        "supported_min_gt_pixels": int(supported_min_gt_pixels),
        "supported_class_ids": [int(i) for i in torch.where(supported)[0].tolist()],
        "macro_iou": float(torch.nanmean(iou[present])) if torch.any(present) else float("nan"),
        "pixel_accuracy": float(tp.sum() / cm.sum()) if cm.sum() > 0 else float("nan"),
        "per_class": {
            str(i): {
                "dice": float(dice[i]) if not torch.isnan(dice[i]) else None,
                "iou": float(iou[i]) if not torch.isnan(iou[i]) else None,
                "precision": float(precision[i]) if not torch.isnan(precision[i]) else None,
                "recall": float(recall[i]) if not torch.isnan(recall[i]) else None,
                "gt_pixels": int(gt[i]),
                "pred_pixels": int(pred[i]),
                "present_in_gt": bool(gt[i] > 0),
                "supported": bool(supported[i]),
            }
            for i in range(cm.shape[0])
        },
        "confusion_matrix": cm.to(torch.int64).tolist(),
    }
