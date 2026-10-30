"""Segmentation loss functions including Dice Loss and combined BCE + Dice Loss."""

from __future__ import annotations

from typing import Any

from filament_seg.utils import get_logger

logger = get_logger(__name__)


class DiceLoss:
    """Soft Dice Loss for binary segmentation masks.

    Computes:
        Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
        Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = True) -> None:
        """Initialize DiceLoss.

        Args:
            smooth: Smoothing factor to prevent division by zero and smooth gradients.
            from_logits: Whether input predictions are unnormalized logits.
        """
        self.smooth = smooth
        self.from_logits = from_logits

    def __call__(self, pred: Any, target: Any) -> Any:
        """Compute soft Dice loss.

        Args:
            pred: Predicted tensor of shape (B, 1, H, W) or (B, H, W).
            target: Ground truth binary tensor of matching shape.

        Returns:
            Scalar loss tensor.
        """
        import torch

        if pred.ndim != 4 or pred.shape != target.shape or pred.shape[1] != 1:
            raise ValueError("pred and target must have matching [B, 1, H, W] shapes")
        if self.smooth <= 0:
            raise ValueError("smooth must be positive")

        probabilities = torch.sigmoid(pred) if self.from_logits else pred
        probabilities = probabilities.float().flatten(1)
        targets = target.float().flatten(1)
        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        # An empty target approaches zero loss only when predictions are empty.
        return (1.0 - dice).mean()


class BCEDiceLoss:
    """Combined Binary Cross-Entropy (BCE) and Dice Loss.

    Provides balanced gradient signals for semantic segmentation under class imbalance.
    Loss = bce_weight * BCE + dice_weight * Dice
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        """Initialize BCEDiceLoss.

        Args:
            bce_weight: Weight assigned to Binary Cross-Entropy loss.
            dice_weight: Weight assigned to soft Dice loss.
            smooth: Smoothing factor for Dice denominator.
        """
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        if pos_weight is not None and pos_weight <= 0:
            raise ValueError("pos_weight must be positive")
        self.pos_weight = pos_weight
        self.dice_fn = DiceLoss(smooth=smooth, from_logits=True)

    def __call__(self, pred: Any, target: Any) -> Any:
        """Compute weighted sum of BCE and Dice losses.

        Args:
            pred: Unnormalized logits tensor of shape (B, 1, H, W) or (B, H, W).
            target: Ground truth binary tensor with values in {0, 1}.

        Returns:
            Combined scalar loss tensor.
        """
        import torch
        import torch.nn.functional as F

        if pred.ndim != 4 or pred.shape != target.shape or pred.shape[1] != 1:
            raise ValueError("pred and target must have matching [B, 1, H, W] shapes")
        target_float = target.float()
        positive_weight = (
            pred.new_tensor(self.pos_weight) if self.pos_weight is not None else None
        )
        bce = F.binary_cross_entropy_with_logits(
            pred, target_float, pos_weight=positive_weight
        )
        dice = self.dice_fn(pred, target_float)

        return self.bce_weight * bce + self.dice_weight * dice
