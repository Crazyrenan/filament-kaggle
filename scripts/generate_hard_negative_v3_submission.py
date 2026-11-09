"""Create a validated v3 submission only after a frozen positive internal check."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_seg.benchmark import load_candidate, sha256_file
from filament_seg.component_filter import ComponentFilter, extract_filtered
from filament_seg.inference import predict_tiled
from filament_seg.submission import encode_coco_counts, validate_submission_rows


def main(internal_path: Path) -> None:
    internal = json.loads(internal_path.read_text(encoding="utf-8"))
    if internal["decision"] != "submission_supported" or internal["phase"] != "internal_scored_once":
        raise ValueError("Frozen v3 candidate did not pass the submission decision gate")
    frozen_path = Path(internal["frozen_path"])
    if sha256_file(frozen_path) != internal["frozen_sha256"]:
        raise ValueError("Frozen tuning artifact changed")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    checkpoint = ROOT / "checkpoints/unet_resnet34_v3_hard_negative_01/best_model.pth"
    original = ROOT / "checkpoints/unet_resnet34_v2/best_model.pth"
    previous = ROOT / "outputs/submission_provisional_v2_confidence_0p97_20261109T044350Z.csv"
    historical = {str(Path(row["path"]).resolve()): row["sha256"] for row in
                  json.loads((ROOT / "reports/historical_checkpoint_hashes.json").read_text(encoding="utf-8"))}
    old_hashes = {path: sha256_file(path) for path in [original, previous]}
    if old_hashes[original] != historical[str(original.resolve())] or \
            old_hashes[original] != frozen["v2_checkpoint_sha256"] or \
            sha256_file(checkpoint) != frozen["v3_checkpoint_sha256"]:
        raise ValueError("Checkpoint or original submission integrity failed")
    if sha256_file(ROOT / "src/filament_seg/component_filter.py") != frozen["filter_source_sha256"]:
        raise ValueError("Post-processing source changed after freeze")
    model, metadata = load_candidate(checkpoint, device="cuda")
    config = ComponentFilter(**frozen["v3_selected_postprocessing"])
    test_dir = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    images = sorted(test_dir.glob("*.jpeg"))
    if len(images) != 180 or len({path.stem for path in images}) != 180:
        raise ValueError("Expected 180 unique physical test images")
    rows = []
    zero_images = 0
    for position, path in enumerate(images, 1):
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != (2048, 2048):
            raise ValueError(f"Invalid test image: {path}")
        probability = predict_tiled(model, raw.astype(np.float32) / 255.0,
                                    device="cuda", tile_size=512, overlap=128, batch_size=4)
        masks = extract_filtered(probability, config)
        zero_images += not masks
        for index, mask in enumerate(masks, 1):
            rows.append({"filament_id": f"{path.stem}_{index}",
                         "segmentation_rle": encode_coco_counts(mask)})
        if position % 20 == 0:
            print(f"Validated inference {position}/180 test images", flush=True)
    if not rows or validate_submission_rows(rows, images) != len(rows):
        raise ValueError("Generated rows failed compressed COCO RLE validation")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "outputs" / f"submission_provisional_unet_resnet34_v3_hard_negative_01_{stamp}.csv"
    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    frame.to_csv(destination, index=False, mode="x")
    saved = pd.read_csv(destination, dtype=str)
    if saved.shape != frame.shape or list(saved.columns) != ["filament_id", "segmentation_rle"]:
        raise ValueError("Saved submission CSV differs from validated rows")
    if any(sha256_file(path) != previous_hash for path, previous_hash in old_hashes.items()):
        raise ValueError("Historical checkpoint or official-baseline submission changed")
    manifest = {"status": "provisional", "experiment_id": metadata["experiment_id"],
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(checkpoint), "checkpoint_sha256": metadata["checkpoint_sha256"],
                "parent_checkpoint_sha256": old_hashes[original], "old_official_csv_sha256": old_hashes[previous],
                "frozen_tuning_report": str(frozen_path), "frozen_tuning_report_sha256": sha256_file(frozen_path),
                "internal_check_report": str(internal_path), "internal_check_report_sha256": sha256_file(internal_path),
                "postprocessing": vars(config), "postprocessing_source_sha256": frozen["filter_source_sha256"],
                "inference_source_sha256": sha256_file(ROOT / "src/filament_seg/inference.py"),
                "submission_source_sha256": sha256_file(Path(__file__)),
                "normalization": "grayscale divided by 255 exactly once", "tiling": {"tile_size": 512, "overlap": 128,
                "batch_size": 4}, "test_images_processed": len(images), "zero_prediction_images": zero_images,
                "rows": len(rows), "csv": str(destination), "csv_sha256": sha256_file(destination),
                "validation": "all compressed COCO RLE rows round-trip decoded at 2048x2048; IDs and schema checked",
                "uploaded_to_kaggle": False,
                "limitation": "Local organizer-notebook-compatible score only; internal check historically exposed."}
    manifest_path = destination.with_suffix(".provenance.json")
    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print("SUBMISSION_COMPLETE", destination, "rows", len(rows), "manifest", manifest_path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal", required=True, type=Path)
    main(parser.parse_args().internal)
