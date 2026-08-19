#!/usr/bin/env python3
"""SAM2.1 Hiera image encoder + lightweight multiclass semantic decoder.

This deliberately does not use SAM2's prompt encoder/mask decoder. Histo-Seg
provides dense class-ID masks, so the proof of concept reuses the pretrained
Hiera/FPN image encoder and learns a conventional semantic segmentation head.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SAM21_HIERA_SMALL_MODEL_ID = "facebook/sam2.1-hiera-small"
SAM21_HIERA_SMALL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


class SemanticFPNDecoder(nn.Module):
    """Fuse SAM2 FPN levels and predict dense class logits."""

    def __init__(
        self,
        *,
        num_classes: int,
        fpn_channels: int = 256,
        decoder_channels: int = 192,
        num_levels: int = 3,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = int(num_classes)
        self.num_levels = int(num_levels)
        in_channels = int(fpn_channels) * self.num_levels
        hidden = int(decoder_channels)

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=hidden),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(hidden, self.num_classes, kernel_size=1)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        *,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        if len(features) < self.num_levels:
            raise ValueError(
                f"Need at least {self.num_levels} FPN features, got {len(features)}"
            )
        selected = list(features[: self.num_levels])
        target_hw = selected[0].shape[-2:]
        resized = [selected[0]]
        for feat in selected[1:]:
            resized.append(
                F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
            )
        x = self.fuse(torch.cat(resized, dim=1))
        logits = self.classifier(x)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


def _build_official_sam2(
    *,
    sam2_config: str,
    sam2_checkpoint: str | None,
    sam2_model_id: str | None,
    load_pretrained: bool,
):
    try:
        from sam2.build_sam import build_sam2, build_sam2_hf
    except ImportError as exc:  # pragma: no cover - depends on external package
        raise RuntimeError(
            "SAM2 is not installed. Install requirements-segmentation.txt first."
        ) from exc

    if load_pretrained:
        if sam2_checkpoint:
            return build_sam2(
                sam2_config,
                sam2_checkpoint,
                device="cpu",
                mode="eval",
                apply_postprocessing=False,
            )
        if sam2_model_id:
            return build_sam2_hf(
                sam2_model_id,
                device="cpu",
                mode="eval",
                apply_postprocessing=False,
            )
        raise ValueError("Need sam2_checkpoint or sam2_model_id when load_pretrained=True")

    # Used when restoring our own semantic checkpoint: instantiate the official
    # architecture without downloading/loading the original SAM2 weights first.
    return build_sam2(
        sam2_config,
        ckpt_path=None,
        device="cpu",
        mode="eval",
        apply_postprocessing=False,
    )


class SAM2HieraSemanticSegmenter(nn.Module):
    """Dense 12-class segmenter built on the official SAM2.1 Hiera encoder."""

    def __init__(
        self,
        *,
        num_classes: int = 12,
        sam2_config: str = SAM21_HIERA_SMALL_CONFIG,
        sam2_checkpoint: str | None = None,
        sam2_model_id: str | None = SAM21_HIERA_SMALL_MODEL_ID,
        load_pretrained: bool = True,
        decoder_channels: int = 192,
    ) -> None:
        super().__init__()
        sam2_model = _build_official_sam2(
            sam2_config=sam2_config,
            sam2_checkpoint=sam2_checkpoint,
            sam2_model_id=sam2_model_id,
            load_pretrained=load_pretrained,
        )
        self.image_encoder = sam2_model.image_encoder
        self.encoder_input_size = int(getattr(sam2_model, "image_size", 1024))
        del sam2_model

        self.decoder = SemanticFPNDecoder(
            num_classes=int(num_classes),
            fpn_channels=256,
            decoder_channels=int(decoder_channels),
            num_levels=3,
        )
        self.num_classes = int(num_classes)
        self.sam2_config = str(sam2_config)
        self.sam2_model_id = sam2_model_id
        self.decoder_channels = int(decoder_channels)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = bool(trainable)

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.image_encoder.parameters()

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.decoder.parameters()

    def forward(self, images: torch.Tensor, *, output_size: Tuple[int, int]) -> torch.Tensor:
        encoded: Dict[str, object] = self.image_encoder(images)
        features = encoded.get("backbone_fpn")
        if not isinstance(features, (list, tuple)):
            raise RuntimeError("SAM2 image_encoder did not return backbone_fpn features")
        return self.decoder(features, output_size=output_size)

    def checkpoint_model_kwargs(self) -> Dict[str, object]:
        return {
            "num_classes": self.num_classes,
            "sam2_config": self.sam2_config,
            "sam2_model_id": self.sam2_model_id,
            "decoder_channels": self.decoder_channels,
        }
