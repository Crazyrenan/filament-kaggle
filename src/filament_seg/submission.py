"""Compressed COCO masks and structural checks for competition CSV rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

SUBMISSION_COLUMNS = ("filament_id", "segmentation_rle")
COMPETITION_SHAPE = (2048, 2048)


def encode_coco_counts(instance_mask: np.ndarray) -> str:
    """Encode one 2D binary instance as compressed COCO RLE counts."""
    array = np.asarray(instance_mask)
    if array.ndim != 2 or not all(array.shape) or not np.isin(array, (0, 1)).all():
        raise ValueError("instance_mask must be a nonempty 2D binary array")
    if not array.any():
        raise ValueError("a filament instance must contain foreground pixels")
    encoded = coco_mask.encode(np.asfortranarray(array.astype(np.uint8)))
    return encoded["counts"].decode("utf-8")


def decode_coco_counts(counts: str, shape: tuple[int, int]) -> np.ndarray:
    """Decode compressed counts using the explicit height and width."""
    if (
        not isinstance(counts, str)
        or not counts
        or not counts.isascii()
        or '"' in counts
        or "'" in counts
        or len(shape) != 2
        or any(dimension <= 0 for dimension in shape)
    ):
        raise ValueError("invalid compressed COCO RLE counts or shape")
    try:
        decoded = coco_mask.decode(
            {"size": list(shape), "counts": counts.encode("utf-8")}
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid compressed COCO RLE counts") from exc
    if decoded.shape != shape or not decoded.any():
        raise ValueError("RLE must decode to a nonempty mask of the requested shape")
    return decoded.astype(np.uint8, copy=False)


def validate_submission_rows(
    rows: Iterable[Mapping[str, str]], image_filenames: Iterable[str | Path]
) -> int:
    """Validate CSV-like rows against known test filenames, without submitting.

    This is a structural check, not proof of acceptance by Kaggle's scorer.
    """
    image_ids = {Path(name).stem for name in image_filenames}
    if not image_ids:
        raise ValueError("image_filenames must contain the test image identifiers")
    seen: set[str] = set()
    count = 0
    for row in rows:
        if tuple(row) != SUBMISSION_COLUMNS:
            raise ValueError(f"columns must be {SUBMISSION_COLUMNS}")
        filament_id = row["filament_id"]
        if not isinstance(filament_id, str) or "_" not in filament_id:
            raise ValueError("filament_id must contain an image ID and unique suffix")
        image_id, suffix = filament_id.rsplit("_", 1)
        if image_id not in image_ids or not suffix or filament_id in seen:
            raise ValueError("unknown or duplicate filament_id")
        decode_coco_counts(row["segmentation_rle"], COMPETITION_SHAPE)
        seen.add(filament_id)
        count += 1
    return count
