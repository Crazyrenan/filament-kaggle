"""Parity checks against the archived organizer self-evaluation notebook v6."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pycocotools import mask as mask_util

from filament_seg.official_metric import organizer_pq_image
from filament_seg.submission import encode_coco_counts


def organizer_functions():
    source = Path(__file__).resolve().parents[1] / "reports/reference/organizer_self_evaluation_v6.ipynb"
    notebook = json.loads(source.read_text(encoding="utf-8"))
    scope = {"pd": pd, "np": np, "torch": torch, "mask_util": mask_util}
    exec("".join(notebook["cells"][3]["source"]) + "\n" +
         "".join(notebook["cells"][4]["source"]), scope)
    original_decode = scope["rles_to_layers"]
    scope["rles_to_layers"] = lambda rles: original_decode(rles, height=8, width=8)
    return scope


def mask(y0, y1, x0, x1):
    result = np.zeros((8, 8), dtype=np.uint8)
    result[y0:y1, x0:x1] = 1
    return result


@pytest.mark.parametrize("truth,predicted", [
    ([mask(1, 4, 1, 4)], [mask(1, 4, 1, 4)]),
    ([mask(1, 4, 1, 4)], [mask(1, 4, 1, 4), mask(1, 4, 1, 4)]),
    ([mask(1, 4, 1, 4)], [mask(5, 8, 5, 8)]),
    ([mask(1, 4, 1, 4)], []),
    ([], [mask(1, 4, 1, 4)]),
])
def test_pair_counting_matches_public_notebook(truth, predicted):
    scope = organizer_functions()
    gt = pd.DataFrame([{"filament_id": f"annotator-20260101_{i}",
                        "segmentation_rle": encode_coco_counts(item)}
                       for i, item in enumerate(truth, 1)],
                      columns=["filament_id", "segmentation_rle"])
    pred = pd.DataFrame([{"filament_id": f"20260101_{i}",
                          "segmentation_rle": encode_coco_counts(item)}
                         for i, item in enumerate(predicted, 1)],
                        columns=["filament_id", "segmentation_rle"])
    if truth:
        official = scope["get_pq_score"](scope["get_overlap_df"](gt, pred))
    else:
        official = 0.0  # Organizer has no GT entry to iterate in this case.
    local = organizer_pq_image(truth, predicted)
    assert local["pq"] == pytest.approx(official)
    if len(truth) == 1 and len(predicted) == 2:
        assert (local["tp"], local["fp"], local["fn"]) == (2, 0, 0)


def test_multiple_annotator_entries_are_scored_separately():
    scope = organizer_functions()
    first = mask(1, 4, 1, 4)
    second = mask(5, 8, 5, 8)
    gt = pd.DataFrame([
        {"filament_id": "a-20260101_1", "segmentation_rle": encode_coco_counts(first)},
        {"filament_id": "b-20260101_1", "segmentation_rle": encode_coco_counts(second)},
    ])
    pred = pd.DataFrame([{"filament_id": "20260101_1", "segmentation_rle": encode_coco_counts(first)}])
    official = scope["get_pq_score"](scope["get_overlap_df"](gt, pred))
    sets = [organizer_pq_image([first], [first]), organizer_pq_image([second], [first])]
    tp, fp, fn = (sum(row[key] for row in sets) for key in ("tp", "fp", "fn"))
    local = sum(row["sum_iou"] for row in sets) / (tp + 0.5 * fp + 0.5 * fn)
    assert local == pytest.approx(official)
    assert (tp, fp, fn) == (1, 1, 1)
