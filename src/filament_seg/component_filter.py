"""Conservative confidence and radial filters on baseline CC instances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from filament_seg.inference import iter_probability_instances


@dataclass(frozen=True)
class ComponentFilter:
    name: str
    min_mean_probability: float = 0.0
    max_radial_distance: float = 2.0


def extract_filtered(probability: np.ndarray, config: ComponentFilter) -> list[np.ndarray]:
    if not 0 <= config.min_mean_probability <= 1 or config.max_radial_distance <= 0:
        raise ValueError("Invalid component filter")
    height, width = probability.shape
    output = []
    for mask in iter_probability_instances(probability, threshold=.8, min_area=500):
        y, x = np.nonzero(mask)
        if float(np.mean(probability[y, x])) < config.min_mean_probability:
            continue
        radius = float(np.hypot(np.mean(x) - width / 2, np.mean(y) - height / 2) / (min(height, width) / 2))
        if radius > config.max_radial_distance:
            continue
        output.append(mask)
    return output
