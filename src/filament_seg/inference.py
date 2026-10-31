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
    """Run model forward pass to compute probability map on a single tensor.

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
        with torch.amp.autocast(device_type=torch.device(device).type, enabled=torch.device(device).type == "cuda"):
            logits = model(image_tensor)
            probs = torch.sigmoid(logits)
            prob_map = probs.squeeze().cpu().numpy()

    return prob_map.astype(np.float32)


def predict_tiled(
    model: Any,
    image: Any,
    tile_size: int = 512,
    overlap: int = 128,
    device: Any = "cuda",
    batch_size: int = 4,
) -> np.ndarray:
    """Run full-resolution tiled inference with Hann window blending.

    Eliminates patch boundary and seam artifacts by weighting overlapping
    predictions with a 2D Hann/cosine tapering window.

    Args:
        model: Trained PyTorch segmentation model.
        image: Input image as torch.Tensor (1, 1, H, W) or (1, H, W) or (H, W),
            or NumPy array (H, W). Normalized to [0, 1].
        tile_size: Height and width of square tiles. Default is 512.
        overlap: Pixel overlap between adjacent tiles. Default is 128.
        device: Execution device (e.g. 'cuda' or 'cpu').
        batch_size: Batch size of tiles processed simultaneously on GPU.

    Returns:
        Reconstructed 2D float32 probability map of shape (H, W) in [0.0, 1.0].
    """
    import torch
    import torch.nn.functional as F

    dev = torch.device(device)
    model.eval()

    # Convert to standard 4D tensor (1, 1, H, W)
    if isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, np.newaxis, ...]
        elif arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        tensor = torch.from_numpy(arr).to(dev)
    elif isinstance(image, torch.Tensor):
        tensor = image.to(dev, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    H, W = tensor.shape[2], tensor.shape[3]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_size must be strictly greater than overlap")

    # Construct 2D Hann window for smooth cosine taper blending
    hann_1d = np.hanning(tile_size).astype(np.float32)
    # Floor window at 1e-3 to prevent zero division near patch boundaries
    window_2d = np.outer(hann_1d, hann_1d)
    window_2d = np.maximum(window_2d, 1e-3)
    window_tensor = torch.from_numpy(window_2d).to(device=dev, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Calculate padding to ensure all edge regions are fully covered by tiles
    pad_h = (stride - ((H - tile_size) % stride)) % stride if H > tile_size else max(0, tile_size - H)
    pad_w = (stride - ((W - tile_size) % stride)) % stride if W > tile_size else max(0, tile_size - W)

    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    pad_H, pad_W = padded.shape[2], padded.shape[3]

    prob_accum = torch.zeros((1, 1, pad_H, pad_W), device=dev, dtype=torch.float32)
    weight_accum = torch.zeros((1, 1, pad_H, pad_W), device=dev, dtype=torch.float32)

    y_coords = list(range(0, pad_H - tile_size + 1, stride))
    x_coords = list(range(0, pad_W - tile_size + 1, stride))

    tile_coords = [(y, x) for y in y_coords for x in x_coords]

    use_amp = dev.type == "cuda"
    with torch.no_grad():
        for batch_start in range(0, len(tile_coords), batch_size):
            batch_indices = tile_coords[batch_start : batch_start + batch_size]
            batch_tiles = torch.cat(
                [padded[:, :, y : y + tile_size, x : x + tile_size] for y, x in batch_indices],
                dim=0,
            )

            with torch.amp.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(batch_tiles)
                probs = torch.sigmoid(logits)

            for i, (y, x) in enumerate(batch_indices):
                prob_accum[:, :, y : y + tile_size, x : x + tile_size] += probs[i : i + 1] * window_tensor
                weight_accum[:, :, y : y + tile_size, x : x + tile_size] += window_tensor

    blended = prob_accum / torch.clamp(weight_accum, min=1e-6)
    result = blended[:, :, :H, :W].squeeze().cpu().numpy()
    return result.astype(np.float32)


def predict_full_resolution(
    model: Any,
    image: Any,
    device: Any = "cuda",
    use_tiled: bool = True,
    tile_size: int = 512,
    overlap: int = 128,
    batch_size: int = 4,
) -> np.ndarray:
    """Predict continuous probability map for an image at native resolution.

    Args:
        model: Trained segmentation model.
        image: Single image array or tensor.
        device: Target execution device.
        use_tiled: Whether to use tiled inference with Hann window blending.
        tile_size: Size of square tiles if tiled.
        overlap: Tile overlap in pixels if tiled.
        batch_size: Batch size of tiles processed in parallel.

    Returns:
        Float32 probability map of shape (H, W) in [0.0, 1.0].
    """
    if use_tiled:
        return predict_tiled(
            model=model,
            image=image,
            tile_size=tile_size,
            overlap=overlap,
            device=device,
            batch_size=batch_size,
        )

    # Direct full image prediction
    import torch

    dev = torch.device(device)
    if isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, np.newaxis, ...]
        elif arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        tensor = torch.from_numpy(arr)
    else:
        tensor = image
    return predict_probability_map(model, tensor, dev)


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

    Touching filaments remain one component; this is a reliable baseline separator.
    Each yielded mask retains the input resolution (e.g. 2048 x 2048).

    Args:
        probability_map: 2D array of predicted foreground probabilities in [0, 1].
        threshold: Binarization decision threshold.
        min_area: Minimum pixel count per instance. Smaller components are dropped.
        connectivity: Pixel neighborhood connectivity (4 or 8).

    Yields:
        Binary 2D numpy array of shape (H, W) and dtype np.uint8 for each instance.
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
    if count == 0:
        return

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

    extent_mask = probability_map >= mask_threshold

    core_mask = probability_map >= core_threshold
    markers, num_markers = label(core_mask)

    if num_markers == 0:
        return separate_instances_cc(extent_mask.astype(np.uint8), min_area=min_area)

    distance = distance_transform_edt(extent_mask)
    labels = watershed(-distance, markers, mask=extent_mask)

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


def postprocess_binary_mask(
    mask: np.ndarray,
    fill_holes: bool = True,
    close_kernel_size: int = 0,
) -> np.ndarray:
    """Apply non-destructive morphological refinement to a binary mask.

    Fills internal cavities without eroding fine, thin filament tips.

    Args:
        mask: 2D binary mask array.
        fill_holes: Whether to fill interior background holes.
        close_kernel_size: Optional morphological closing kernel size (0 to disable).

    Returns:
        Refined binary mask.
    """
    from scipy.ndimage import binary_fill_holes
    import cv2

    result = mask.astype(bool)
    if fill_holes:
        result = binary_fill_holes(result)

    result_uint8 = result.astype(np.uint8)
    if close_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        result_uint8 = cv2.morphologyEx(result_uint8, cv2.MORPH_CLOSE, kernel)

    return result_uint8


def mask_to_rle(mask: np.ndarray) -> str:
    """Run-length encode a binary mask (standard legacy Kaggle format).

    Flattens the mask in column-major (Fortran) order.
    Returns space-delimited string of 1-based (start, length) pairs.
    Note: For official competition submissions, use encode_coco_counts from submission.py.

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
