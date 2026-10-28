"""Quantitative evaluation metrics for semantic and instance segmentation."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_iou(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Compute Intersection over Union (Jaccard Index) for binary masks.

    Args:
        pred_mask: Boolean or binary array representing predicted mask.
        target_mask: Boolean or binary array representing ground-truth mask.
        eps: Small epsilon to prevent division by zero when both masks are empty.

    Returns:
        IoU score as float in [0.0, 1.0].
    """
    pred_bool = pred_mask.astype(bool)
    target_bool = target_mask.astype(bool)

    intersection = np.logical_and(pred_bool, target_bool).sum()
    union = np.logical_or(pred_bool, target_bool).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    if intersection == 0:
        return 0.0

    return float(intersection / union)


def compute_dice(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Compute Dice Coefficient (F1-score) for binary masks.

    Args:
        pred_mask: Boolean or binary array representing predicted mask.
        target_mask: Boolean or binary array representing ground-truth mask.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Dice score as float in [0.0, 1.0].
    """
    pred_bool = pred_mask.astype(bool)
    target_bool = target_mask.astype(bool)

    intersection = np.logical_and(pred_bool, target_bool).sum()
    total_area = pred_bool.sum() + target_bool.sum()

    if total_area == 0:
        return 1.0 if intersection == 0 else 0.0

    if intersection == 0:
        return 0.0

    return float((2.0 * intersection) / total_area)


def compute_precision_recall(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    eps: float = 1e-7,
) -> tuple[float, float]:
    """Compute pixel-level Precision and Recall.

    Args:
        pred_mask: Predicted binary mask.
        target_mask: Ground-truth binary mask.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Tuple of (precision, recall).
    """
    pred_bool = pred_mask.astype(bool)
    target_bool = target_mask.astype(bool)

    tp = np.logical_and(pred_bool, target_bool).sum()
    fp = np.logical_and(pred_bool, ~target_bool).sum()
    fn = np.logical_and(~pred_bool, target_bool).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return precision, recall


def evaluate_semantic_predictions(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate a batch/collection of semantic predictions against ground truth.

    Args:
        predictions: List of probability arrays (values in [0.0, 1.0]).
        targets: List of binary ground-truth arrays.
        threshold: Decision threshold for binarizing probabilities.

    Returns:
        Dictionary containing mean IoU, Dice, Precision, and Recall.
    """
    ious: list[float] = []
    dices: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    for pred, target in zip(predictions, targets):
        bin_pred = (pred >= threshold).astype(np.uint8)
        ious.append(compute_iou(bin_pred, target))
        dices.append(compute_dice(bin_pred, target))
        p, r = compute_precision_recall(bin_pred, target)
        precisions.append(p)
        recalls.append(r)

    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
    }
