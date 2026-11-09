import numpy as np
import pytest

from filament_seg.inference import iter_probability_instances
from filament_seg.postprocessing_experiments import PostprocessingConfig, extract_instances


def test_baseline_exactly_matches_existing_extractor():
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[3:14, 3:14] = .9
    probability[31:44, 40:50] = .95
    config = PostprocessingConfig("baseline", threshold=.8, min_area=100)
    actual = extract_instances(probability, config)
    expected = list(iter_probability_instances(probability, threshold=.8, min_area=100))
    assert len(actual) == len(expected) == 2
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))


def test_opening_removes_thin_bridge_and_split_separates_lobes():
    probability = np.zeros((96, 96), dtype=np.float32)
    probability[20:40, 10:30] = .9
    probability[20:40, 50:70] = .9
    probability[29, 30:50] = .9
    baseline = extract_instances(probability, PostprocessingConfig("baseline", min_area=100))
    opened = extract_instances(probability, PostprocessingConfig("open", min_area=100, operation="open", kernel=3))
    split = extract_instances(probability, PostprocessingConfig("split", min_area=100, operation="split", kernel=3))
    assert len(baseline) == 1
    assert len(opened) == 2
    assert len(split) == 2
    assert not np.any(split[0] & split[1])


def test_shape_filter_rejects_compact_blob_and_keeps_long_filament():
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[5:25, 5:25] = .9
    probability[40:45, 5:85] = .9
    result = extract_instances(probability, PostprocessingConfig(
        "shape", min_area=100, operation="shape", min_elongation=2.0))
    assert len(result) == 1
    assert result[0][42, 50] == 1


def test_invalid_configuration_rejected():
    probability = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        extract_instances(probability, PostprocessingConfig("bad", operation="open", kernel=2))
