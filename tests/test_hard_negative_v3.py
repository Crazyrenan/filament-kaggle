"""Training-only hard-negative crop and provenance guards."""
import numpy as np
import pytest
import torch

from filament_seg.hard_negative import HardNegativePatchDataset


class TinyDataset:
    filenames = [f"train_{i}.jpeg" for i in range(10)] + ["validation.jpeg"]

    def __getitem__(self, index):
        image = torch.linspace(0, 1, 64).reshape(1, 8, 8)
        mask = torch.zeros((1, 8, 8))
        if index != 0:
            mask[:, 2:4, 2:4] = 1
        return {"image": image, "mask": mask}


def test_mixed_sampling_preserves_binary_alignment_and_empty_hard_negatives():
    base = TinyDataset()
    patches = HardNegativePatchDataset(base, list(range(10)),
                                       {"train_0.jpeg": [{"top": 0, "left": 0}]},
                                       patch_size=4)
    patches.set_epoch(1)
    samples = [patches[i] for i in range(10)]
    assert {sample["sampling_mode"] for sample in samples} == {"hard_negative", "foreground", "random"}
    for sample in samples:
        assert sample["image"].shape == sample["mask"].shape == (1, 4, 4)
        assert sample["image"].dtype == sample["mask"].dtype == torch.float32
        assert sample["image"].min() >= 0 and sample["image"].max() <= 1
        assert set(sample["mask"].unique().tolist()) <= {0, 1}
        if sample["sampling_mode"] == "hard_negative":
            assert sample["filename"] == "train_0.jpeg"
            assert not sample["mask"].any()
    assert sum(patches.sampling_counts.values()) == 10
    patches.set_epoch(2)
    assert not patches.sampling_counts


def test_mined_filename_cannot_be_validation_filename():
    with pytest.raises(ValueError, match="training subset"):
        HardNegativePatchDataset(TinyDataset(), list(range(10)),
                                 {"validation.jpeg": [{"top": 0, "left": 0}]}, patch_size=4)


def test_mined_patch_fails_if_annotation_enters_crop():
    patches = HardNegativePatchDataset(TinyDataset(), list(range(10)),
                                       {"train_1.jpeg": [{"top": 0, "left": 0}]}, patch_size=4)
    patches.set_epoch(1)
    with pytest.raises(ValueError, match="contains a training annotation"):
        _ = patches[7]
