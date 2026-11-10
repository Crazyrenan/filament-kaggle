"""Deterministic dihedral test-time transforms for square probability maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpatialTransform:
    name: str
    quarter_turns: int = 0
    horizontal_flip: bool = False
    vertical_flip: bool = False

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2 or image.shape[0] != image.shape[1]:
            raise ValueError("TTA requires a square 2D image")
        result = np.rot90(image, self.quarter_turns)
        if self.horizontal_flip:
            result = np.fliplr(result)
        if self.vertical_flip:
            result = np.flipud(result)
        return np.ascontiguousarray(result)

    def invert(self, probability: np.ndarray) -> np.ndarray:
        if probability.ndim != 2 or probability.shape[0] != probability.shape[1]:
            raise ValueError("TTA requires a square 2D probability map")
        result = probability
        if self.vertical_flip:
            result = np.flipud(result)
        if self.horizontal_flip:
            result = np.fliplr(result)
        return np.ascontiguousarray(np.rot90(result, -self.quarter_turns))


IDENTITY = SpatialTransform("identity")
HORIZONTAL = SpatialTransform("horizontal", horizontal_flip=True)
VERTICAL = SpatialTransform("vertical", vertical_flip=True)
ROT90 = SpatialTransform("rot90", quarter_turns=1)
ROT180 = SpatialTransform("rot180", quarter_turns=2)
ROT270 = SpatialTransform("rot270", quarter_turns=3)
DIAGONAL = SpatialTransform("diagonal", quarter_turns=1, horizontal_flip=True)
ANTIDIAGONAL = SpatialTransform("antidiagonal", quarter_turns=3, horizontal_flip=True)
FLIP4 = (IDENTITY, HORIZONTAL, VERTICAL, ROT180)
ROT4 = (IDENTITY, ROT90, ROT180, ROT270)
DIHEDRAL8 = (IDENTITY, HORIZONTAL, VERTICAL, ROT180, ROT90, ROT270, DIAGONAL, ANTIDIAGONAL)
CONFIGURATIONS = {"flip4": FLIP4, "rot4": ROT4, "dihedral8": DIHEDRAL8}


def average_aligned(probabilities: dict[str, np.ndarray], configuration: str) -> np.ndarray:
    transforms = CONFIGURATIONS[configuration]
    if set(probabilities) != {transform.name for transform in transforms}:
        raise ValueError("Expected exactly one probability map per TTA transform")
    shape = probabilities[transforms[0].name].shape
    result = np.zeros(shape, dtype=np.float32)
    for transform in transforms:
        item = probabilities[transform.name]
        if item.shape != shape or item.dtype != np.float32 or not np.isfinite(item).all():
            raise ValueError("TTA probability maps must be finite, aligned float32 arrays")
        result += item
    result /= np.float32(len(transforms))
    return result
