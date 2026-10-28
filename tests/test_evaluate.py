"""Unit tests for evaluation metrics."""

import numpy as np

from filament_seg.evaluate import (
    compute_dice,
    compute_iou,
    compute_precision_recall,
    evaluate_semantic_predictions,
)


def test_compute_iou():
    m1 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    m2 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    assert np.isclose(compute_iou(m1, m2), 1.0)

    m3 = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    assert np.isclose(compute_iou(m1, m3), 0.0)


def test_compute_dice():
    m1 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    m2 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    assert np.isclose(compute_dice(m1, m2), 1.0)

    m3 = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    assert np.isclose(compute_dice(m1, m3), 0.0)


def test_compute_precision_recall():
    target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    precision, recall = compute_precision_recall(pred, target)
    assert np.isclose(precision, 1.0)
    assert np.isclose(recall, 0.5)


def test_evaluate_semantic_predictions():
    targets = [np.array([[1, 1], [0, 0]], dtype=np.uint8)]
    preds = [np.array([[0.9, 0.8], [0.1, 0.2]], dtype=np.float32)]
    metrics = evaluate_semantic_predictions(preds, targets, threshold=0.5)
    assert metrics["mean_iou"] > 0.99
    assert metrics["mean_dice"] > 0.99
