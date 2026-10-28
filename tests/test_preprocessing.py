"""Unit tests for preprocessing and normalization."""

import numpy as np
import pytest

from filament_seg.preprocessing import normalize_image


def test_normalize_minmax():
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    norm = normalize_image(arr, method="minmax")
    assert np.isclose(norm.min(), 0.0)
    assert np.isclose(norm.max(), 1.0)


def test_normalize_percentile():
    arr = np.linspace(0, 100, 100, dtype=np.float32)
    norm = normalize_image(arr, method="percentile")
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_normalize_unknown_method():
    arr = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        normalize_image(arr, method="unsupported_method")
