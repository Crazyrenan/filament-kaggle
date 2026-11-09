"""Memory-efficient parity implementation of organizer self-evaluation v6 PQ.

Source: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
Archived source: reports/reference/organizer_self_evaluation_v6.ipynb
The public notebook evaluates every annotator-image entry independently. This
matches that notebook, but does not establish parity with Kaggle's hidden scorer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools import mask as coco_mask


def organizer_pq_image(
    ground_truth: list[np.ndarray], predictions: list[np.ndarray], *,
    return_details: bool = False,
) -> dict[str, Any]:
    """Count all IoU > 0.5 hits, not exclusive one-to-one assignments."""
    n_gt, n_pred = len(ground_truth), len(predictions)
    matches = []
    hit_gt: set[int] = set()
    hit_pred: set[int] = set()
    if n_gt and n_pred:
        gt_bool = [np.asarray(mask, dtype=bool) for mask in ground_truth]
        pred_bool = [np.asarray(mask, dtype=bool) for mask in predictions]
        shape = gt_bool[0].shape
        if any(mask.shape != shape for mask in gt_bool + pred_bool):
            raise ValueError("GT and prediction masks must share one shape")
        gt_rles = [coco_mask.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in gt_bool]
        pred_rles = [coco_mask.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in pred_bool]
        ious = coco_mask.iou(pred_rles, gt_rles, [0] * n_gt).T
        for i, j in np.argwhere(ious > 0.5):
            matches.append({"gt_index": int(i), "pred_index": int(j), "iou": float(ious[i, j])})
            hit_gt.add(int(i))
            hit_pred.add(int(j))
    tp = len(matches)
    fp = n_pred - len(hit_pred)
    fn = n_gt - len(hit_gt)
    sum_iou = sum(item["iou"] for item in matches)
    denominator = tp + 0.5 * fp + 0.5 * fn
    result = {
        "pq": sum_iou / denominator if denominator else 0.0,
        "sq": sum_iou / tp if tp else 0.0,
        "rq": tp / denominator if denominator else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "sum_iou": sum_iou,
    }
    if return_details:
        result.update({"matches": matches,
                       "unmatched_gt": [i for i in range(n_gt) if i not in hit_gt],
                       "unmatched_pred": [j for j in range(n_pred) if j not in hit_pred]})
    return result
