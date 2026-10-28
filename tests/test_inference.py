"""Unit tests for inference and post-processing utilities."""

import numpy as np

from filament_seg.inference import mask_to_rle, separate_instances_cc


def test_separate_instances_cc():
    mask = np.zeros((20, 20), dtype=np.uint8)
    # Filament 1
    mask[2:6, 2:6] = 1
    # Filament 2
    mask[12:16, 12:16] = 1

    labeled, count = separate_instances_cc(mask, min_area=5)
    assert count == 2
    assert labeled.max() == 2


def test_mask_to_rle():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 0] = 1
    # Fortran order: pixels (0,0) and (1,0) are 1-based indices 1 and 2
    rle = mask_to_rle(mask)
    assert rle == "1 2"
