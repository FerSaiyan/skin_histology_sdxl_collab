#!/usr/bin/env python3
"""Evaluate a trained SAM2.1 Hiera semantic checkpoint on val/test tiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from histoseg_tile_dataset import (  # noqa: E402
    CLASS_NAME_BY_ID,
    HistosegSimulationTileDataset,
)
from sam2_hiera_semantic import SAM2HieraSemanticSegmenter  # noqa: E402
from segmentation_metrics import metrics_from_confusion, update_confusion_matrix  # noqa: E402

CLASS_ID_TO_COLOR = {
    0: (0, 0, 0),
    1: (224, 224, 224),
    2: (96, 96, 96),
    3: (150, 150, 0),
    4: (127, 255, 255),
    5: (255, 156, 0),
    6: (255, 0, 255),
    7: (0, 255, 0),
    8: (0, 156, 255),
    9: (127, 96, 255),
    10: (112, 48, 160),
    11: (0, 0, 128),
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="best.pt or last.pt from training")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tiles-root", default="", help="Override path stored in checkpoint config")
    ap.add_argument("--splits-csv", default="", help="Override path stored in checkpoint config")
    ap.add_argument("--output-dir", default="", help="Default: <checkpoint-dir>/eval_<split>")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--max-batches", type=int, default=0, help="0=all")
    ap.add_argument("--save-predictions", action="store_true")
    return ap.parse_args()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    return torch.device(value)


def _label_vis(labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in CLASS_ID_TO_COLOR.items():
        out[labels == cid] = color
    return out


def main() -> int:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_kwargs = dict(checkpoint["model_kwargs"])
    num_classes = int(model_kwargs.get("num_classes", 12))

    # Recreate only the SAM2 architecture; our checkpoint supplies the trained
    # encoder + decoder weights, so there is no need to re-download Meta weights.
    model = SAM2HieraSemanticSegmenter(
        num_classes=num_classes,
        sam2_config=str(model_kwargs["sam2_config"]),
        sam2_checkpoint=None,
        sam2_model_id=model_kwargs.get("sam2_model_id"),
        load_pretrained=False,
        decoder_channels=int(model_kwargs.get("decoder_channels", 192)),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)

    training_config = dict(checkpoint.get("training_config", {}))
    data_cfg = dict(training_config.get("data", {}))
    tiles_root = args.tiles_root or data_cfg.get("tiles_root")
    splits_csv = args.splits_csv or data_cfg.get("splits_csv")
    if not tiles_root or not splits_csv:
        raise SystemExit("tiles_root/splits_csv missing; pass CLI overrides")

    encoder_size = int(checkpoint.get("encoder_input_size", model.encoder_input_size))
    dataset = HistosegSimulationTileDataset(
        tiles_root=tiles_root,
        splits_csv=splits_csv,
        split=args.split,
        encoder_size=encoder_size,
        augment=False,
    )
    device = _device(args.device)
    model = model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else checkpoint_path.parent / f"eval_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "predictions"
    vis_dir = output_dir / "predictions_vis"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    seen = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"eval {args.split}")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=device.type == "cuda",
            ):
                logits = model(images, output_size=(int(labels.shape[-2]), int(labels.shape[-1])))
            update_confusion_matrix(confusion, logits, labels, num_classes=num_classes)

            if args.save_predictions:
                pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
                for i, tile_id in enumerate(batch["tile_id"]):
                    np.save(pred_dir / f"{tile_id}.npy", pred[i])
                    Image.fromarray(_label_vis(pred[i])).save(vis_dir / f"{tile_id}.png")
            seen += int(images.shape[0])

    metrics = metrics_from_confusion(confusion, exclude_background_from_macro=True)
    metrics["split"] = args.split
    metrics["tiles_evaluated"] = seen
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["class_names"] = {str(k): v for k, v in CLASS_NAME_BY_ID.items()}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Tiles evaluated: {seen}")
    print(f"Macro Dice     : {metrics['macro_dice']:.4f}")
    print(f"Macro IoU      : {metrics['macro_iou']:.4f}")
    for cid in range(num_classes):
        cls = metrics["per_class"][str(cid)]
        dice = cls["dice"]
        print(
            f"  {cid:2d} {CLASS_NAME_BY_ID.get(cid, f'class_{cid}'):<28} "
            f"Dice={'n/a' if dice is None else f'{dice:.4f}'}"
        )
    print(f"Metrics: {output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
