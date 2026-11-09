"""Binary SegFormer adapter with the same logit interface as U-Net."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

SEGFORMER_SOURCES = {
    "b2": ("nvidia/segformer-b2-finetuned-ade-512-512", "de01bae28967510f9ddd496c60a969357195400c"),
    "b1": ("nvidia/segformer-b1-finetuned-ade-512-512", "a775b4f5dcb2179e57006dca0e268281306412ef"),
    "b0": ("nvidia/segformer-b0-finetuned-ade-512-512", "489d5cd81a0b59fab9b7ea758d3548ebe99677da"),
}


class BinarySegFormer(nn.Module):
    """Accept normalized grayscale patches and return full-size binary logits."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("SegFormer expects BCHW grayscale input")
        rgb = images.repeat(1, 3, 1, 1)
        logits = self.network(pixel_values=rgb).logits
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


def build_binary_segformer(
    variant: str = "b2", *, pretrained: bool = False,
    hf_config: dict[str, Any] | None = None,
) -> tuple[BinarySegFormer, dict[str, Any]]:
    """Load a pinned ADE20K SegFormer, or rebuild exactly from checkpoint config."""
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    if variant not in SEGFORMER_SOURCES:
        raise ValueError(f"Unsupported SegFormer variant: {variant}")
    repo, revision = SEGFORMER_SOURCES[variant]
    if pretrained:
        network = SegformerForSemanticSegmentation.from_pretrained(
            repo, revision=revision, local_files_only=True, use_safetensors=False
        )
        source_config = network.config.to_dict()
    else:
        if hf_config is None:
            raise ValueError("Checkpoint loading requires embedded SegFormer config")
        source_config = hf_config
        network = SegformerForSemanticSegmentation(SegformerConfig.from_dict(source_config))
    in_features = network.decode_head.classifier.in_channels
    network.decode_head.classifier = nn.Conv2d(in_features, 1, kernel_size=1)
    network.config.num_labels = 1
    network.config.id2label = {0: "filament"}
    network.config.label2id = {"filament": 0}
    return BinarySegFormer(network), {
        "architecture": "SegFormer", "variant": variant, "pretrained_repo": repo,
        "pretrained_revision": revision, "hf_config": source_config,
        "input_channels": 1, "encoder_channels": 3, "classes": 1,
        "input_adapter": "repeat grayscale channel 3 times; no additional normalization",
    }
