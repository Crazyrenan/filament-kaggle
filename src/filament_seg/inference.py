"""Inference pipeline, test-time post-processing, and instance separation."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from filament_seg.utils import get_logger

logger = get_logger(__name__)


def predict_probability_map(
    model: Any,
    image_tensor: Any,
    device: Any,
) -> np.ndarray:
    """Run model forward pass to compute probability map.

    Args:
        model: Trained PyTorch segmentation model.
        image_tensor: Normalized input tensor of shape (1, C, H, W) or (C, H, W).
        device: Device to execute inference on.

    Returns:
        Probability map as a 2D float32 NumPy array in range [0.0, 1.0].
    """
    import torch

    model.eval()
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        prob_map = probs.squeeze().cpu().numpy()

    return prob_map.astype(np.float32)


def separate_instances_cc(
    binary_mask: np.ndarray,
    min_area: int = 50,
) -> tuple[np.ndarray, int]:
    """Extract individual instances using Connected Component Analysis.

    Args:
        binary_mask: Binary mask of shape (H, W).
        min_area: Minimum pixel area required to retain a connected component.

    Returns:
        Tuple of (labeled_mask, instance_count), where labeled_mask contains
        0 for background and integer labels 1..N for instances.
    """
    from scipy.ndimage import label

    labeled, num_features = label(binary_mask > 0)
    if num_features == 0:
        return np.zeros_like(binary_mask, dtype=np.int32), 0

    # Filter small components
    filtered_labeled = np.zeros_like(labeled, dtype=np.int32)
    current_id = 1
    for instance_id in range(1, num_features + 1):
        component = labeled == instance_id
        if component.sum() >= min_area:
            filtered_labeled[component] = current_id
            current_id += 1

    return filtered_labeled, current_id - 1


def iter_probability_instances(
    probability_map: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 50,
    connectivity: int = 8,
) -> Iterator[np.ndarray]:
    """Yield binary masks for connected foreground components in scan order.

    Touching filaments remain one component; this is only a baseline separator.
    Each yielded mask retains the input resolution, including 2048 x 2048.
    """
    from scipy.ndimage import label

    probabilities = np.asarray(probability_map)
    if probabilities.ndim != 2 or not all(probabilities.shape) or not np.isfinite(probabilities).all():
        raise ValueError("probability_map must be a finite 2D array")
    if not 0 <= threshold <= 1 or not 0 <= probabilities.min() or probabilities.max() > 1:
        raise ValueError("probabilities and threshold must be in [0, 1]")
    if min_area < 1 or connectivity not in (4, 8):
        raise ValueError("min_area must be positive and connectivity must be 4 or 8")

    structure = (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        if connectivity == 4
        else np.ones((3, 3), dtype=np.uint8)
    )
    labels, count = label(probabilities >= threshold, structure=structure)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    for component_id in range(1, count + 1):
        if areas[component_id] >= min_area:
            yield (labels == component_id).astype(np.uint8)


def separate_instances_watershed(
    probability_map: np.ndarray,
    core_threshold: float = 0.7,
    mask_threshold: float = 0.3,
    min_area: int = 50,
) -> tuple[np.ndarray, int]:
    """Separate touching solar filaments using marker-controlled Watershed segmentation.

    Args:
        probability_map: Continuous probability array in [0.0, 1.0].
        core_threshold: High confidence threshold to define sure foreground seeds/cores.
        mask_threshold: Lower threshold to define the extent of candidate filament bodies.
        min_area: Minimum pixel count per separated instance.

    Returns:
        Tuple of (labeled_mask, instance_count).
    """
    from scipy.ndimage import distance_transform_edt, label
    from skimage.segmentation import watershed

    # 1. Mask representing the full filament extent
    extent_mask = probability_map >= mask_threshold

    # 2. Seed cores with high confidence
    core_mask = probability_map >= core_threshold
    markers, num_markers = label(core_mask)

    if num_markers == 0:
        return separate_instances_cc(extent_mask.astype(np.uint8), min_area=min_area)

    # 3. Distance transform on extent mask
    distance = distance_transform_edt(extent_mask)

    # 4. Watershed guided by negative distance transform and seeds
    labels = watershed(-distance, markers, mask=extent_mask)

    # 5. Filter out instances smaller than min_area
    filtered_labels = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for inst_id in np.unique(labels):
        if inst_id == 0:
            continue
        comp = labels == inst_id
        if comp.sum() >= min_area:
            filtered_labels[comp] = next_id
            next_id += 1

    return filtered_labels, next_id - 1


def mask_to_rle(mask: np.ndarray) -> str:
    """Run-length encode a binary mask (standard Kaggle format).

    Flattens the mask in column-major (Fortran) order.
    Returns space-delimited string of 1-based (start, length) pairs.

    Args:
        mask: 2D binary array (1 for foreground, 0 for background).

    Returns:
        Space-delimited string of pairs: 'start length start length...'
    """
    pixels = mask.T.flatten()
    padded = np.concatenate([[0], pixels, [0]])
    runs = np.where(padded[1:] != padded[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)
