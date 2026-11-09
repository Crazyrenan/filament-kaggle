"""Small, frozen family of shape-aware rescues for low-confidence components."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from filament_seg.inference import iter_probability_instances


@dataclass(frozen=True)
class AdaptiveFilterConfig:
    name: str
    min_mean_probability: float = 0.97
    rescue_min_area: int = 0
    rescue_min_elongation: float = 0.0
    rescue_min_q25: float = 0.0
    rescue_max_compactness: float = 2.0
    rescue_enabled: bool = False


def extract_adaptive(probability: np.ndarray, config: AdaptiveFilterConfig) -> list[np.ndarray]:
    """Extract original 0.8/500 components and optionally rescue selected rejects."""
    if not 0 <= config.min_mean_probability <= 1 or config.rescue_min_area < 0:
        raise ValueError("Invalid adaptive filter configuration")
    result = []
    for mask in iter_probability_instances(probability, threshold=.8, min_area=500):
        y, x = np.nonzero(mask)
        values = probability[y, x]
        if float(values.mean()) >= config.min_mean_probability:
            result.append(mask)
            continue
        if not config.rescue_enabled or len(y) < config.rescue_min_area:
            continue
        if float(np.quantile(values, .25)) < config.rescue_min_q25:
            continue
        if config.rescue_min_elongation > 0:
            eigen = np.linalg.eigvalsh(np.cov(np.stack((y, x))))
            elongation = float(np.sqrt(eigen[-1] / max(eigen[0], 1e-6)))
            if elongation < config.rescue_min_elongation:
                continue
        if config.rescue_max_compactness < 2:
            y0, y1, x0, x1 = int(y.min()), int(y.max()) + 1, int(x.min()), int(x.max()) + 1
            crop = np.ascontiguousarray(mask[y0:y1, x0:x1].astype(np.uint8))
            contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
            compactness = float(4 * np.pi * len(y) / perimeter**2) if perimeter > 0 else 2.0
            if compactness > config.rescue_max_compactness:
                continue
        result.append(mask)
    return result
