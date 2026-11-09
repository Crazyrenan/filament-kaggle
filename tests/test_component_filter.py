import numpy as np

from filament_seg.component_filter import ComponentFilter, extract_filtered
from filament_seg.inference import iter_probability_instances


def test_baseline_filter_matches_submission_extractor():
    probability = np.zeros((80, 80), dtype=np.float32)
    probability[5:35, 5:35] = .92
    probability[45:75, 45:75] = .98
    expected = list(iter_probability_instances(probability, threshold=.8, min_area=500))
    actual = extract_filtered(probability, ComponentFilter("baseline"))
    assert len(expected) == len(actual) == 2
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))


def test_confidence_filter_removes_weak_component():
    probability = np.zeros((80, 80), dtype=np.float32)
    probability[5:35, 5:35] = .92
    probability[45:75, 45:75] = .98
    actual = extract_filtered(probability, ComponentFilter("confidence", min_mean_probability=.95))
    assert len(actual) == 1
    assert actual[0][50, 50] == 1
