"""Reusable, inference-free post-processing comparison on cached probability maps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from filament_seg.evaluate import diagnostic_panoptic_quality
from filament_seg.inference import iter_probability_instances


def score_cached_predictions(
    records: Iterable[Mapping[str, Any]],
    *,
    threshold: float,
    min_area: int,
    connectivity: int = 8,
) -> dict[str, float | int]:
    """Score one configuration without rerunning the model.

    Each record contains a `probability_map` and a list of `gt_instances`.
    The caller controls the fixed image subset and annotation-set policy.
    """
    totals = {"tp": 0, "fp": 0, "fn": 0}
    matched_iou = 0.0
    per_image_pq: list[float] = []
    for record in records:
        predictions = list(iter_probability_instances(
            np.asarray(record["probability_map"]),
            threshold=threshold,
            min_area=min_area,
            connectivity=connectivity,
        ))
        result = diagnostic_panoptic_quality(list(record["gt_instances"]), predictions)
        per_image_pq.append(float(result["pq"]))
        matched_iou += float(result["sq"]) * int(result["tp"])
        for key in totals:
            totals[key] += int(result[key])
    if not per_image_pq:
        raise ValueError("At least one cached image is required")
    denominator = totals["tp"] + 0.5 * (totals["fp"] + totals["fn"])
    return {
        "threshold": threshold,
        "min_area": min_area,
        "connectivity": connectivity,
        "images": len(per_image_pq),
        "global_pq": matched_iou / denominator if denominator else 1.0,
        "macro_pq": float(np.mean(per_image_pq)),
        "sq": matched_iou / totals["tp"] if totals["tp"] else 0.0,
        "rq": totals["tp"] / denominator if denominator else 1.0,
        **totals,
    }
