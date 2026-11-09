"""Deterministic mixed foreground, random, and mined-negative patch sampling."""

from __future__ import annotations

from typing import Any
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset


class HardNegativePatchDataset(Dataset):
    """One 512px crop per training index; hard negatives use training-only mines."""

    def __init__(self, base_dataset: Any, train_indices: list[int],
                 patches_by_filename: dict[str, list[dict]], *, seed: int = 43,
                 patch_size: int = 512) -> None:
        if not train_indices or len(train_indices) != len(set(train_indices)):
            raise ValueError("Training indices must be nonempty and unique")
        training_filenames = {base_dataset.filenames[index] for index in train_indices}
        if not patches_by_filename or not set(patches_by_filename).issubset(training_filenames):
            raise ValueError("Hard-negative patch filenames must be a nonempty training subset")
        self.base = base_dataset
        self.train_indices = list(train_indices)
        self.patch_size = patch_size
        self.seed = seed
        self.epoch = 0
        self.mined = patches_by_filename
        self.mined_filenames = sorted(patches_by_filename)
        self.filename_to_index = {name: index for index, name in enumerate(base_dataset.filenames)}
        self.sampling_counts: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self.train_indices)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError("Epoch must be positive")
        self.epoch = epoch
        self.sampling_counts.clear()

    def __getitem__(self, position: int) -> dict:
        if not 0 <= position < len(self):
            raise IndexError(position)
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + position)
        slot = (position + self.epoch * 3) % 10
        mode = "hard_negative" if slot < 3 else "foreground" if slot < 7 else "random"
        if mode == "hard_negative":
            filename = self.mined_filenames[int(rng.integers(len(self.mined_filenames)))]
            index = self.filename_to_index[filename]
            choices = self.mined[filename]
            selected = choices[int(rng.integers(len(choices)))]
            top, left = int(selected["top"]), int(selected["left"])
        else:
            index = self.train_indices[position]
            filename = self.base.filenames[index]
            top = left = -1
        sample = self.base[index]
        image = sample["image"].squeeze(0).numpy()
        mask = sample["mask"].squeeze(0).numpy()
        height, width = image.shape
        size = self.patch_size
        if height < size or width < size:
            raise ValueError("Source image is smaller than patch")
        if mode == "foreground":
            points = np.argwhere(mask > .5)
            if len(points):
                cy, cx = points[int(rng.integers(len(points)))]
                top = int(np.clip(cy - size // 2, 0, height - size))
                left = int(np.clip(cx - size // 2, 0, width - size))
            else:
                mode = "random"
        if mode == "random":
            top = int(rng.integers(0, height - size + 1))
            left = int(rng.integers(0, width - size + 1))
        if not 0 <= top <= height - size or not 0 <= left <= width - size:
            raise ValueError("Mined crop is out of bounds")
        image = image[top:top + size, left:left + size]
        mask = mask[top:top + size, left:left + size]
        if mode == "hard_negative" and np.any(mask):
            raise ValueError("Mined hard-negative patch contains a training annotation")
        if rng.random() < .5:
            image, mask = np.fliplr(image), np.fliplr(mask)
        if rng.random() < .5:
            image, mask = np.flipud(image), np.flipud(mask)
        turns = int(rng.integers(4))
        image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        self.sampling_counts[mode] += 1
        return {"image": torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32)),
                "mask": torch.from_numpy(np.ascontiguousarray(mask[None], dtype=np.float32)),
                "filename": filename, "sampling_mode": mode}
