"""Cached post-processing and instance-detail regressions."""

import numpy as np

from filament_seg.evaluate import diagnostic_panoptic_quality
from filament_seg.experiments import score_cached_predictions


def test_cached_scoring_and_match_details():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:7, 2:7] = 1
    probs = mask.astype(np.float32)
    records = [{"probability_map": probs, "gt_instances": [mask]}]
    perfect = score_cached_predictions(records, threshold=0.5, min_area=1)
    missed = score_cached_predictions(records, threshold=0.5, min_area=26)
    assert (perfect["global_pq"], perfect["tp"], perfect["fp"], perfect["fn"]) == (1.0, 1, 0, 0)
    assert (missed["global_pq"], missed["tp"], missed["fp"], missed["fn"]) == (0.0, 0, 0, 1)
    details = diagnostic_panoptic_quality([mask], [mask], return_details=True)
    assert details["matches"] == [{"gt_index": 0, "pred_index": 0, "iou": 1.0}]
    assert details["unmatched_gt"] == details["unmatched_pred"] == []
