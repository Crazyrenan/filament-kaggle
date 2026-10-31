"""Compressed COCO masks and structural checks for competition CSV rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask

from filament_seg.utils import get_logger

logger = get_logger(__name__)

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

    This is a structural check verifying column headers, image IDs, uniqueness,
    and valid compressed COCO RLE decode to 2048x2048.
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
            raise ValueError(f"unknown or duplicate filament_id: {filament_id}")
        decode_coco_counts(row["segmentation_rle"], COMPETITION_SHAPE)
        seen.add(filament_id)
        count += 1
    return count


def generate_submission(
    model: Any,
    test_images: list[Path] | list[str],
    output_csv_path: str | Path | None = None,
    device: Any = "cuda",
    threshold: float = 0.6,
    min_area: int = 200,
    tile_size: int = 512,
    overlap: int = 128,
    batch_size: int = 4,
) -> pd.DataFrame:
    """Generate competition submission rows from test images using tiled inference.

    Args:
        model: Trained PyTorch segmentation model.
        test_images: List of file paths to test images.
        output_csv_path: Optional path to save CSV submission.
        device: Execution device.
        threshold: Probability threshold for foreground.
        min_area: Minimum instance component area.
        tile_size: Tile size for inference.
        overlap: Overlap between adjacent tiles.
        batch_size: Parallel tile batch size.

    Returns:
        DataFrame containing columns ['filament_id', 'segmentation_rle'].
    """
    import cv2
    from filament_seg.inference import iter_probability_instances, predict_tiled

    rows: list[dict[str, str]] = []
    test_paths = [Path(p) for p in test_images]

    logger.info("Generating predictions for %d test images...", len(test_paths))

    for img_path in test_paths:
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Could not read test image: {img_path}")
        image_norm = image.astype(np.float32) / 255.0

        prob_map = predict_tiled(
            model=model,
            image=image_norm,
            tile_size=tile_size,
            overlap=overlap,
            device=device,
            batch_size=batch_size,
        )

        instances = list(iter_probability_instances(prob_map, threshold=threshold, min_area=min_area))
        stem = img_path.stem

        for idx, inst_mask in enumerate(instances, start=1):
            filament_id = f"{stem}_{idx}"
            rle = encode_coco_counts(inst_mask)
            rows.append({
                "filament_id": filament_id,
                "segmentation_rle": rle,
            })

    # Validate generated rows against test stems
    if rows:
        validate_submission_rows(rows, test_paths)

    df = pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))
    logger.info("Generated %d filament instances across %d test images", len(df), len(test_paths))

    if output_csv_path:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        logger.info("Saved submission CSV to %s", out_p)

    return df
