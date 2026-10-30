"""Astronomical image preprocessing, normalization, and augmentation pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import albumentations as A


def load_image(image_path: str | Path, as_rgb: bool = True) -> np.ndarray:
    """Load an astronomical image from disk.

    Supports common formats (PNG, JPG, TIFF) via OpenCV or fallback to PIL.

    Args:
        image_path: Path to the image file.
        as_rgb: If True, convert BGR/grayscale to RGB (3-channel).

    Returns:
        Image as a NumPy array.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be decoded.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at: {path.resolve()}")

    try:
        import cv2

        flags = cv2.IMREAD_COLOR if as_rgb else cv2.IMREAD_UNCHANGED
        img = cv2.imread(str(path), flags)
        if img is not None:
            if as_rgb and len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    except ImportError:
        pass

    # Fallback to PIL
    from PIL import Image

    with Image.open(path) as pil_img:
        if as_rgb:
            pil_img = pil_img.convert("RGB")
        return np.array(pil_img)


def normalize_image(
    image: np.ndarray,
    method: str = "minmax",
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Normalize pixel intensities for scientific / deep-learning inputs.

    Args:
        image: Input image array.
        method: Normalization method:
            - 'minmax': Scales values to [0.0, 1.0].
            - 'standardize': (x - mean) / std using ImageNet or custom statistics.
            - 'percentile': Scales between 1st and 99th percentiles (robust to solar flares/hot pixels).
        mean: Channel means for 'standardize'.
        std: Channel standard deviations for 'standardize'.

    Returns:
        Normalized float32 NumPy array.
    """
    arr = image.astype(np.float32)

    if method == "minmax":
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val)
        else:
            arr = np.zeros_like(arr)
        return arr

    if method == "percentile":
        p_low = np.percentile(arr, 1.0)
        p_high = np.percentile(arr, 99.0)
        if p_high > p_low:
            arr = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            arr = np.zeros_like(arr)
        return arr

    if method == "standardize":
        # Scale to [0, 1] first if in [0, 255]
        if np.max(arr) > 1.0:
            arr = arr / 255.0
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr = np.array(std, dtype=np.float32)
        return (arr - mean_arr) / std_arr

    raise ValueError(f"Unknown normalization method: {method}")


def get_transforms(
    image_size: tuple[int, int] = (512, 512),
    is_train: bool = True,
) -> Any:
    """Build Albumentations transform pipeline for training or validation.

    Args:
        image_size: Target (height, width) for resizing.
        is_train: If True, include geometric and photometric augmentations.

    Returns:
        Albumentations Compose pipeline (or callable stub if albumentations is missing).
    """
    try:
        import albumentations as A

        if is_train:
            return A.Compose(
                [
                    A.Resize(image_size[0], image_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=45,
                        p=0.5,
                    ),
                ]
            )
        return A.Compose([A.Resize(image_size[0], image_size[1])])
    except ImportError:
        # Stub callable when albumentations is not installed yet
        return lambda **kwargs: kwargs
    
def get_train_transform(patch_size=512):
    return A.Compose([
        A.RandomCrop(
            height=patch_size,
            width=patch_size,
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])

def get_val_transform(patch_size=512):
    return A.Compose([
        A.CenterCrop(
            height=patch_size,
            width=patch_size,
            p=1.0,
        ),
    ])


import numpy as np
import albumentations as A


class ForegroundAwareCrop:
    def __init__(self, patch_size=512, foreground_prob=0.7):
        self.patch_size = patch_size
        self.foreground_prob = foreground_prob

    def __call__(self, image, mask, **kwargs):
        height, width = mask.shape[:2]
        size = self.patch_size

        if height < size or width < size:
            raise ValueError("Image is smaller than patch size")

        foreground_y, foreground_x = np.where(mask > 0)

        use_foreground = (
            len(foreground_y) > 0
            and np.random.random() < self.foreground_prob
        )

        if use_foreground:
            index = np.random.randint(len(foreground_y))

            center_y = int(foreground_y[index])
            center_x = int(foreground_x[index])

            top = np.clip(center_y - size // 2, 0, height - size)
            left = np.clip(center_x - size // 2, 0, width - size)
        else:
            top = np.random.randint(0, height - size + 1)
            left = np.random.randint(0, width - size + 1)

        top = int(top)
        left = int(left)

        return {
            "image": image[top:top + size, left:left + size],
            "mask": mask[top:top + size, left:left + size],
        }


def get_train_transform(patch_size=512):
    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])

    crop = ForegroundAwareCrop(
        patch_size=patch_size,
        foreground_prob=0.7,
    )

    def transform(image, mask):
        cropped = crop(image=image, mask=mask)
        return augmentation(**cropped)

    return transform
