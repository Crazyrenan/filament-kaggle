"""Deep learning segmentation model architecture initialization."""

from __future__ import annotations

from typing import Any

from filament_seg.utils import get_logger

logger = get_logger(__name__)


def build_segmentation_model(
    architecture: str = "Unet",
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
    in_channels: int = 1,
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

    weights_path = None
    if encoder_weights is not None:
        from huggingface_hub import try_to_load_from_cache
        from segmentation_models_pytorch.encoders import encoders

        settings = encoders[encoder_name]["pretrained_settings"][encoder_weights]
        for filename in ("config.json", "model.safetensors"):
            cached = try_to_load_from_cache(
                settings["repo_id"], filename, revision=settings["revision"]
            )
            if not isinstance(cached, str):
                raise FileNotFoundError(
                    f"{encoder_weights} {encoder_name} weights are not cached ({filename}); "
                    "download them explicitly before requesting pretrained weights"
                )
            if filename == "model.safetensors":
                weights_path = cached

    model_class = getattr(smp, architecture)
    model = model_class(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3 if weights_path is not None else in_channels,
        classes=classes,
        activation=activation,
    )
    if weights_path is not None:
        from safetensors.torch import load_file

        model.encoder.load_state_dict(load_file(weights_path, device="cpu"))
        model.encoder.set_in_channels(in_channels, pretrained=True)
    logger.info(
        "Initialized %s model with backbone %s (weights: %s, in_channels: %d, classes: %d)",
        architecture,
        encoder_name,
        encoder_weights,
        in_channels,
        classes,
    )
    return model


def build_model(
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    pretrained: bool = False,
    in_channels: int = 1,
    classes: int = 1,
) -> Any:
    """Build a raw-logit U-Net without implicitly downloading encoder weights.

    SMP adapts cached RGB encoder weights to one channel by summing the first
    convolution's input-channel weights when ``in_channels=1``.
    """
    if architecture.lower() != "unet":
        raise ValueError("Stage 3.2 supports only the U-Net baseline")

    return build_segmentation_model(
        architecture="Unet",
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
