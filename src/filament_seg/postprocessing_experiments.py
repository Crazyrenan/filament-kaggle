"""Frozen, local post-processing candidates for cached foreground probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class PostprocessingConfig:
    name: str
    threshold: float = 0.8
    min_area: int = 500
    operation: str = "baseline"
    kernel: int = 0
    min_elongation: float = 0.0


def extract_instances(probability: np.ndarray, config: PostprocessingConfig) -> list[np.ndarray]:
    """Return disjoint uint8 instance masks without altering the probability map."""
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("Expected a finite 2D probability map")
    if not 0 <= config.threshold <= 1 or config.min_area < 1:
        raise ValueError("Invalid threshold or minimum area")
    binary = (probability >= config.threshold).astype(np.uint8)
    if config.operation in {"open", "close"}:
        if config.kernel < 3 or config.kernel % 2 != 1:
            raise ValueError("Morphology kernel must be odd and at least three")
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.kernel, config.kernel))
        code = cv2.MORPH_OPEN if config.operation == "open" else cv2.MORPH_CLOSE
        binary = cv2.morphologyEx(binary, code, element)
    elif config.operation not in {"baseline", "shape", "split"}:
        raise ValueError(f"Unknown operation: {config.operation}")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output: list[np.ndarray] = []
    for component_id in range(1, count):
        x, y, width, height, area = map(int, stats[component_id])
        if area < config.min_area:
            continue
        roi = (labels[y:y + height, x:x + width] == component_id).astype(np.uint8)
        if config.operation == "shape":
            # Covariance-axis ratio tolerates curved filaments better than a box ratio.
            rows, columns = np.nonzero(roi)
            if len(rows) < 2:
                continue
            eigenvalues = np.linalg.eigvalsh(np.cov(np.stack((rows, columns))))
            elongation = float(np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-6)))
            if elongation < config.min_elongation:
                continue
        pieces = [roi]
        if config.operation == "split":
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.kernel, config.kernel))
            seeds = cv2.erode(roi, element)
            seed_count, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
            keep = [seed_id for seed_id in range(1, seed_count)
                    if seed_stats[seed_id, cv2.CC_STAT_AREA] >= max(50, config.min_area // 5)]
            if 1 < len(keep) <= 20:
                # Assign foreground pixels to the nearest eroded core, within this component.
                cores = np.zeros_like(roi, dtype=np.int32)
                for new_id, seed_id in enumerate(keep, start=1):
                    cores[seed_labels == seed_id] = new_id
                nearest = distance_transform_edt(cores == 0, return_distances=False, return_indices=True)
                assigned = cores[nearest[0], nearest[1]]
                pieces = [(roi & (assigned == seed_id)).astype(np.uint8)
                          for seed_id in range(1, len(keep) + 1)]
        for piece in pieces:
            if int(piece.sum()) < config.min_area:
                continue
            full = np.zeros(binary.shape, dtype=np.uint8)
            full[y:y + height, x:x + width] = piece
            output.append(full)
    return output
