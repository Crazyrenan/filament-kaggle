"""Read-only audit of the exact Kaggle-submitted CSV and its provenance."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_seg.submission import decode_coco_counts

CSV = ROOT / "outputs/submission_provisional_unet_resnet34_v2_20261104T094726Z.csv"
PROVENANCE = CSV.with_suffix(".provenance.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    provenance = json.loads(PROVENANCE.read_text())
    assert sha256(CSV) == provenance["csv_sha256"]
    assert sha256(Path(provenance["checkpoint"])) == provenance["checkpoint_sha256"]
    images = sorted((ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images").glob("*.jpeg"))
    assert len(images) == 180
    image_ids = {path.stem for path in images}
    frame = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    assert frame.columns.tolist() == ["filament_id", "segmentation_rle"]
    assert len(frame) == provenance["predicted_filaments"]
    assert frame["filament_id"].is_unique
    indices: dict[str, list[int]] = defaultdict(list)
    areas = []
    boxes = []
    orientations = []
    for row in frame.itertuples(index=False):
        match = re.fullmatch(r"(.+)_([1-9]\d*)", row.filament_id)
        assert match is not None and match.group(1) in image_ids
        stem, number = match.group(1), int(match.group(2))
        indices[stem].append(number)
        decoded = decode_coco_counts(row.segmentation_rle, (2048, 2048))
        encoded = coco_mask.encode(np.asfortranarray(decoded))
        assert encoded["counts"].decode("ascii") == row.segmentation_rle
        assert encoded["size"] == [2048, 2048]
        area = int(coco_mask.area(encoded))
        assert area >= provenance["postprocessing"]["min_area"]
        box = coco_mask.toBbox(encoded).tolist()
        assert 0 <= box[0] < 2048 and 0 <= box[1] < 2048
        assert box[0] + box[2] <= 2048 and box[1] + box[3] <= 2048
        areas.append(area)
        boxes.append(box)
        orientations.append((int(np.count_nonzero(decoded[0])), int(np.count_nonzero(decoded[-1]))))
    for stem, numbers in indices.items():
        assert numbers == list(range(1, len(numbers) + 1)), stem
    predicted_stems = set(indices)
    counts = {stem: len(indices.get(stem, [])) for stem in image_ids}
    result = {
        "kaggle_submission_ref": 58150471,
        "reported_public_score": 0.31,
        "csv": str(CSV), "csv_sha256": sha256(CSV),
        "provenance": str(PROVENANCE), "checkpoint_sha256": provenance["checkpoint_sha256"],
        "rows": len(frame), "test_images": len(images),
        "images_with_predictions": len(predicted_stems), "images_without_predictions": len(image_ids - predicted_stems),
        "row_count_per_image": {"min": min(counts.values()), "median": float(np.median(list(counts.values()))), "max": max(counts.values())},
        "instance_area_px": {"min": min(areas), "median": float(np.median(areas)), "p95": float(np.percentile(areas, 95)), "max": max(areas)},
        "all_rles_round_trip": True, "all_masks_2048_square": True,
        "all_instance_indices_consecutive_per_image": True,
        "all_identifiers_match_test_stems": True,
        "orientation_note": "COCO Fortran-order decode/re-encode is exact. Because masks are square, model-output alignment needs a separate image comparison.",
    }
    destination = ROOT / "reports/submitted_csv_audit_20261109_v2.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
