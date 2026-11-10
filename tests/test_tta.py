import numpy as np
import pytest

from filament_seg.tta import CONFIGURATIONS, DIHEDRAL8, average_aligned


def test_all_dihedral_transforms_are_unique_and_invert_exactly():
    image = np.arange(49, dtype=np.float32).reshape(7, 7)
    images = [transform.apply(image) for transform in DIHEDRAL8]
    assert len({item.tobytes() for item in images}) == 8
    for transform, transformed in zip(DIHEDRAL8, images):
        np.testing.assert_array_equal(transform.invert(transformed), image)


@pytest.mark.parametrize("configuration", list(CONFIGURATIONS))
def test_aligned_average_restores_asymmetric_probability_map(configuration):
    image = np.arange(25, dtype=np.float32).reshape(5, 5) / 25
    aligned = {transform.name: transform.invert(transform.apply(image))
               for transform in CONFIGURATIONS[configuration]}
    np.testing.assert_allclose(average_aligned(aligned, configuration), image, atol=1e-7)


def test_average_rejects_missing_or_misaligned_maps():
    image = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        average_aligned({"identity": image}, "flip4")
    with pytest.raises(ValueError):
        DIHEDRAL8[0].apply(np.zeros((5, 6), dtype=np.float32))
