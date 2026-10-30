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
    """Compute diagnostic foreground metrics, not competition Panoptic Quality.

    Probabilities are foreground at ``>= threshold``. Dice and IoU are 1 when
    both binary masks are empty; precision and recall are 0 in that case.

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


def diagnostic_panoptic_quality(
    ground_truth: list[np.ndarray],
    predictions: list[np.ndarray],
    iou_threshold: float = 0.5,
    inclusive: bool = False,
) -> dict[str, float | int]:
    """Match individual binary masks and compute diagnostic PQ, SQ, and RQ.

    The default is strict IoU > 0.5. Matching maximizes TP count, then summed
    IoU, with deterministic input-order tie handling. This is not the verified
    organizer scorer. Both sets empty are defined as PQ=SQ=RQ=1.
    """
    from scipy.optimize import linear_sum_assignment

    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    all_masks = []
    shape = None
    for mask in [*ground_truth, *predictions]:
        array = np.asarray(mask)
        if array.ndim != 2 or array.size == 0:
            raise ValueError("each instance must be a nonempty 2D binary mask")
        if shape is None:
            shape = array.shape
        if array.shape != shape or not np.isin(array, (0, 1)).all() or not array.any():
            raise ValueError("instances must share shape and contain binary foreground")
        all_masks.append(array.astype(bool, copy=False))

    n_gt, n_pred = len(ground_truth), len(predictions)
    if n_gt == n_pred == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0}

    ious = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i, truth in enumerate(all_masks[:n_gt]):
        for j, prediction in enumerate(all_masks[n_gt:]):
            overlap = np.count_nonzero(truth & prediction)
            if overlap:
                union = np.count_nonzero(truth | prediction)
                ious[i, j] = overlap / union

    eligible = ious >= iou_threshold if inclusive else ious > iou_threshold
    scores = np.zeros((n_gt + n_pred, n_gt + n_pred), dtype=np.float64)
    scores[:n_gt, :n_pred] = np.where(eligible, 1.0 + ious, -1.0)
    rows, columns = linear_sum_assignment(scores, maximize=True)
    matched_iou = sum(
        ious[row, column]
        for row, column in zip(rows, columns)
        if row < n_gt and column < n_pred and eligible[row, column]
    )
    tp = sum(
        row < n_gt and column < n_pred and eligible[row, column]
        for row, column in zip(rows, columns)
    )
    fp, fn = n_pred - tp, n_gt - tp
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": float(matched_iou / denominator) if denominator else 1.0,
        "sq": float(matched_iou / tp) if tp else 0.0,
        "rq": float(tp / denominator) if denominator else 1.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }
