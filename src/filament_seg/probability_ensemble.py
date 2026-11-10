"""Weighted probability-map ensemble and organizer-compatible RLE scoring."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from pycocotools import mask as coco_mask

from filament_seg.inference import iter_probability_instances


@dataclass(frozen=True, order=True)
class EnsembleConfig:
    v2_weight: float
    threshold: float
    min_area: int
    mean_confidence: float

    def validate(self) -> None:
        if not 0 <= self.v2_weight <= 1 or not 0 <= self.threshold <= 1:
            raise ValueError("Weight and mask threshold must be in [0, 1]")
        if self.min_area < 1 or not 0 <= self.mean_confidence <= 1:
            raise ValueError("Minimum area or confidence is invalid")


def blend(v2: np.ndarray, v3: np.ndarray, v2_weight: float) -> np.ndarray:
    if v2.shape != v3.shape or v2.dtype != np.float32 or v3.dtype != np.float32:
        raise ValueError("Probability maps must have identical shape and float32 dtype")
    if not 0 <= v2_weight <= 1:
        raise ValueError("v2_weight must be in [0, 1]")
    if not np.isfinite(v2).all() or not np.isfinite(v3).all() or min(v2.min(), v3.min()) < 0 or max(v2.max(), v3.max()) > 1:
        raise ValueError("Probability maps must be finite and in [0, 1]")
    return np.add(np.multiply(v2, np.float32(v2_weight), dtype=np.float32),
                  np.multiply(v3, np.float32(1 - v2_weight), dtype=np.float32), dtype=np.float32)


def candidate_components(probability: np.ndarray, threshold: float, min_area: int) -> list[tuple[int, float, np.ndarray]]:
    """Use the existing eight-connected extractor, retaining area and mean confidence."""
    candidates = []
    for mask in iter_probability_instances(probability, threshold=threshold, min_area=min_area):
        foreground = mask.astype(bool)
        candidates.append((int(foreground.sum()), float(probability[foreground].mean()), mask))
    return candidates


def select_components(candidates: list[tuple[int, float, np.ndarray]], config: EnsembleConfig) -> list[np.ndarray]:
    config.validate()
    return [mask for area, confidence, mask in candidates
            if area >= config.min_area and confidence >= config.mean_confidence]


def encode_masks(masks: list[np.ndarray]) -> list[dict]:
    return [coco_mask.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in masks]


def score_encoded_sets(gt_sets: list[list[dict]], predicted: list[dict]) -> dict[str, float | int]:
    """Match the public organizer notebook: count every IoU > 0.5 pair."""
    tp = fp = fn = 0
    sum_iou = 0.0
    for gt in gt_sets:
        if gt and predicted:
            ious = coco_mask.iou(predicted, gt, [0] * len(gt))
            hits = ious > 0.5
            tp += int(hits.sum())
            fp += len(predicted) - int(hits.any(axis=1).sum())
            fn += len(gt) - int(hits.any(axis=0).sum())
            sum_iou += float(ious[hits].sum())
        else:
            fp += len(predicted)
            fn += len(gt)
    denominator = tp + .5 * (fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "sum_iou": sum_iou,
            "pq": sum_iou / denominator if denominator else 0.0,
            "sq": sum_iou / tp if tp else 0.0,
            "rq": tp / denominator if denominator else 0.0}


def summarize(rows: list[dict]) -> dict:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    sum_iou = sum(row["sum_iou"] for row in rows)
    denominator = tp + .5 * (fp + fn)
    return {"images": len(rows), "global_pq": sum_iou / denominator if denominator else 0.0,
            "sq": sum_iou / tp if tp else 0.0, "rq": tp / denominator if denominator else 0.0,
            "tp": tp, "fp": fp, "fn": fn}
