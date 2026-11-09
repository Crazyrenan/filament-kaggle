import numpy as np

from filament_seg.adaptive_instance_filter import AdaptiveFilterConfig, extract_adaptive
from filament_seg.component_filter import ComponentFilter, extract_filtered


def test_adaptive_baseline_equals_official_confidence_baseline():
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[5:35, 5:35] = .96
    probability[55:85, 55:85] = .98
    expected = extract_filtered(probability, ComponentFilter("confidence", min_mean_probability=.97))
    actual = extract_adaptive(probability, AdaptiveFilterConfig("baseline"))
    assert len(actual) == len(expected) == 1
    assert np.array_equal(actual[0], expected[0])


def test_area_elongation_rule_recovers_long_low_confidence_component():
    probability = np.zeros((130, 130), dtype=np.float32)
    probability[10:20, 5:125] = .96
    probability[50:90, 50:90] = .96
    result = extract_adaptive(probability, AdaptiveFilterConfig(
        "rescue", rescue_enabled=True, rescue_min_area=1100, rescue_min_elongation=4.5))
    assert len(result) == 1
    assert result[0][15, 60] == 1
