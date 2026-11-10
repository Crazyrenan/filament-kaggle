import numpy as np
import pytest

from filament_seg.official_metric import organizer_pq_image
from filament_seg.probability_ensemble import (
    EnsembleConfig, blend, candidate_components, encode_masks, score_encoded_sets,
    select_components,
)


def test_weighted_blend_requires_aligned_finite_float32_maps():
    first = np.array([[0.0, 1.0]], dtype=np.float32)
    second = np.array([[1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(blend(first, second, .25), np.array([[.75, .25]], dtype=np.float32))
    with pytest.raises(ValueError):
        blend(first, second.T, .5)
    with pytest.raises(ValueError):
        blend(first, second.astype(np.float64), .5)
    with pytest.raises(ValueError):
        blend(first, np.array([[np.nan, 0]], dtype=np.float32), .5)


def test_component_selection_uses_existing_eight_connected_masks():
    probability = np.zeros((16, 16), dtype=np.float32)
    probability[2:5, 2:5] = .98
    probability[10:13, 10:13] = .91
    candidates = candidate_components(probability, threshold=.8, min_area=5)
    assert [area for area, _, _ in candidates] == [9, 9]
    config = EnsembleConfig(.5, .8, 8, .95)
    selected = select_components(candidates, config)
    assert len(selected) == 1
    assert selected[0][3, 3] == 1 and selected[0][11, 11] == 0


def test_encoded_scorer_parity_with_public_organizer_logic_and_alternatives():
    gt1 = np.zeros((32, 32), dtype=np.uint8)
    gt2 = np.zeros_like(gt1)
    pred1 = np.zeros_like(gt1)
    pred2 = np.zeros_like(gt1)
    gt1[2:8, 2:8] = 1
    gt2[20:26, 20:26] = 1
    pred1[2:8, 2:8] = 1
    pred2[11:14, 11:14] = 1
    sets = [[gt1], [gt2]]
    predicted = [pred1, pred2]
    direct = [organizer_pq_image(gt, predicted) for gt in sets]
    encoded = score_encoded_sets([encode_masks(gt) for gt in sets], encode_masks(predicted))
    assert (encoded["tp"], encoded["fp"], encoded["fn"]) == (
        sum(row["tp"] for row in direct), sum(row["fp"] for row in direct),
        sum(row["fn"] for row in direct))
    assert encoded["sum_iou"] == pytest.approx(sum(row["sum_iou"] for row in direct))
