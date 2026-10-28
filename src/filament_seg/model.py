"""Deep learning segmentation model architecture initialization."""

from __future__ import annotations

from typing import Any

from filament_seg.utils import get_logger

logger = get_logger(__name__)


def build_segmentation_model(
    architecture: str = "Unet",
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    activation: str | None = None,
) -> Any:
    """Initialize a PyTorch segmentation model using segmentation_models_pytorch (smp).

    Args:
        architecture: Segmentation head name ('Unet', 'UnetPlusPlus', 'FPN', etc.).
        encoder_name: Backbone encoder ('resnet34', 'efficientnet-b0', etc.).
        encoder_weights: Pretrained weights name ('imagenet' or None).
        in_channels: Number of input image channels (e.g. 1 for grayscale, 3 for RGB).
        classes: Number of target output classes (1 for binary/instance mask logits).
        activation: Optional activation ('sigmoid', 'softmax', or None for raw logits).

    Returns:
        PyTorch nn.Module instance.

    Raises:
        ImportError: If segmentation_models_pytorch is not installed.
        ValueError: If the requested architecture is unsupported.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation_models_pytorch is required to instantiate segmentation models. "
            "Install it via `pip install segmentation-models-pytorch` or through `environment.yml`."
        ) from exc

    if not hasattr(smp, architecture):
        supported = [attr for attr in dir(smp) if attr[0].isupper() and not attr.startswith("_")]
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported architectures include: {supported[:8]}..."
        )

    model_class = getattr(smp, architecture)
    model = model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    logger.info(
        "Initialized %s model with backbone %s (weights: %s, in_channels: %d, classes: %d)",
        architecture,
        encoder_name,
        encoder_weights,
        in_channels,
        classes,
    )
    return model
