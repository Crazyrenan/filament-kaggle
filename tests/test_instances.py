"""Instance extraction, separation, PQ, and COCO submission contracts."""

import json

import cv2
import numpy as np
import pytest

from filament_seg.dataset import SolarFilamentDataset
from filament_seg.evaluate import diagnostic_panoptic_quality
from filament_seg.inference import iter_probability_instances
from filament_seg.submission import (
    encode_coco_counts,
    decode_coco_counts,
    validate_submission_rows,
)


def test_instance_polygons_overlap_and_alternative_sets(tmp_path):
    cv2.imwrite(str(tmp_path / "solar.jpeg"), np.zeros((12, 12), dtype=np.uint8))
    coco = {
        "images": [
            {"id": "set-a", "file_name": "solar.jpeg", "height": 12, "width": 12},
            {"id": "set-b", "file_name": "solar.jpeg", "height": 12, "width": 12},
        ],
        "annotations": [
            {"id": "a1", "image_id": "set-a", "category_id": 1,
             "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4], [8, 8, 10, 8, 10, 10, 8, 10]]},
            {"id": "a2", "image_id": "set-a", "category_id": 2,
             "segmentation": [[3, 3, 6, 3, 6, 6, 3, 6]]},
            {"id": "b1", "image_id": "set-b", "category_id": 3,
             "segmentation": [[0, 8, 2, 8, 2, 10, 0, 10]]},
        ],
    }
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(coco), encoding="utf-8")
    dataset = SolarFilamentDataset(tmp_path, annotation_path)

    first = list(dataset.iter_instances(0))
    alternative = list(dataset.iter_instances(0, 1))
    assert len(dataset) == 1
    assert len(first) == 2 and len(alternative) == 1
    assert [(x["annotation_id"], x["category_id"], x["image_id"], x["filename"])
            for x in first] == [("a1", 1, "set-a", "solar.jpeg"),
                               ("a2", 2, "set-a", "solar.jpeg")]
    assert first[0]["mask"].shape == (12, 12)
    assert first[0]["mask"][9, 9] == 1  # second polygon of one annotation
    assert first[0]["mask"][3, 3] == first[1]["mask"][3, 3] == 1
    assert alternative[0]["image_id"] == "set-b"
    assert alternative[0]["mask"][3, 3] == 0
    with pytest.raises(IndexError):
        list(dataset.iter_instances(0, 2))


def test_connected_components_threshold_area_connectivity_and_empty():
    probabilities = np.zeros((6, 6), dtype=np.float32)
    probabilities[1, 1] = probabilities[2, 2] = 0.5
    assert len(list(iter_probability_instances(probabilities, threshold=0.5,
                                               min_area=1, connectivity=4))) == 2
    masks = list(iter_probability_instances(probabilities, threshold=0.5,
                                            min_area=2, connectivity=8))
    assert len(masks) == 1 and masks[0].sum() == 2
    assert masks[0].shape == probabilities.shape
    assert list(iter_probability_instances(probabilities, threshold=0.9)) == []
    with pytest.raises(ValueError):
        list(iter_probability_instances(probabilities, connectivity=6))


def test_diagnostic_pq_perfect_missing_extra_overlap_and_boundary():
    one = np.zeros((4, 4), dtype=np.uint8)
    one[0, 0:2] = 1
    other = np.zeros_like(one)
    other[2:4, 2:4] = 1
    half = np.zeros_like(one)
    half[0, 0] = 1

    perfect = diagnostic_panoptic_quality([one], [one])
    assert (perfect["pq"], perfect["sq"], perfect["rq"], perfect["tp"]) == (1, 1, 1, 1)
    missing = diagnostic_panoptic_quality([one, other], [one])
    assert (missing["tp"], missing["fn"], missing["fp"]) == (1, 1, 0)
    assert missing["pq"] == pytest.approx(2 / 3)
    extra = diagnostic_panoptic_quality([one], [one, other])
    assert (extra["tp"], extra["fn"], extra["fp"]) == (1, 0, 1)
    assert extra["pq"] == pytest.approx(2 / 3)
    assert diagnostic_panoptic_quality([one], [half])["pq"] == 0
    inclusive = diagnostic_panoptic_quality([one], [half], inclusive=True)
    assert inclusive["pq"] == 0.5 and inclusive["tp"] == 1

    overlapping_truth = [one, one]
    matched = diagnostic_panoptic_quality(overlapping_truth, [one])
    assert (matched["tp"], matched["fn"]) == (1, 1)
    assert diagnostic_panoptic_quality([], []) == {
        "pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0
    }
    assert diagnostic_panoptic_quality([], [one])["fp"] == 1


def test_compressed_coco_rle_small_and_full_resolution():
    for shape in ((8, 9), (2048, 2048)):
        original = np.zeros(shape, dtype=np.uint8)
        original[1:4, 2:6] = 1
        counts = encode_coco_counts(original)
        assert isinstance(counts, str)
        assert np.array_equal(decode_coco_counts(counts, shape), original)
    with pytest.raises(ValueError):
        encode_coco_counts(np.full((3, 3), 2))
    with pytest.raises(ValueError):
        decode_coco_counts("1 2", (8, 9))


def test_submission_schema_is_structural_only():
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[10:12, 20:23] = 1
    row = {"filament_id": "20150125172714Mh_1",
           "segmentation_rle": encode_coco_counts(mask)}
    assert validate_submission_rows([row], ["20150125172714Mh.jpeg"]) == 1
    with pytest.raises(ValueError):
        validate_submission_rows([row, row], ["20150125172714Mh.jpeg"])
    with pytest.raises(ValueError):
        validate_submission_rows([{"id": row["filament_id"],
                                   "segmentation_rle": row["segmentation_rle"]}],
                                 ["20150125172714Mh.jpeg"])
    with pytest.raises(ValueError):
        validate_submission_rows([row], ["different.jpeg"])
