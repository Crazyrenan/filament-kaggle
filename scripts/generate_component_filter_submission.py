"""Generate a separate locally validated provisional CSV from the frozen filter."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_seg.benchmark import code_revision, load_candidate, sha256_file
from filament_seg.component_filter import ComponentFilter, extract_filtered
from filament_seg.inference import predict_tiled
from filament_seg.submission import encode_coco_counts, validate_submission_rows

FROZEN = ROOT / "reports/component_filter_frozen_20261109T043725Z.json"
INTERNAL = ROOT / "reports/component_filter_internal_20261109T044117Z.json"
PREVIOUS = ROOT / "outputs/submission_provisional_unet_resnet34_v2_20261104T094726Z.csv"


def bootstrap_interval(baseline: list[dict], selected: list[dict]) -> list[float]:
    assert [row["index"] for row in baseline] == [row["index"] for row in selected]
    groups: dict[str, list[tuple[dict, dict]]] = {}
    for first, second in zip(baseline, selected):
        groups.setdefault(first["filename"][:8], []).append((first, second))
    keys = list(groups)
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(5000):
        sampled = rng.choice(keys, len(keys), replace=True)
        paired = [pair for key in sampled for pair in groups[key]]
        scores = []
        for side in (0, 1):
            rows = [pair[side] for pair in paired]
            tp = sum(row["tp"] for row in rows)
            fp = sum(row["fp"] for row in rows)
            fn = sum(row["fn"] for row in rows)
            scores.append(sum(row["sum_iou"] for row in rows) / (tp + .5 * (fp + fn)))
        differences.append(scores[1] - scores[0])
    return [float(value) for value in np.percentile(differences, (2.5, 97.5))]


def main() -> None:
    frozen = json.loads(FROZEN.read_text())
    internal = json.loads(INTERNAL.read_text())
    if frozen["phase"] != "frozen_after_tuning" or internal["frozen_sha256"] != sha256_file(FROZEN):
        raise ValueError("Internal result is not linked to the frozen tuning artifact")
    if internal["selected_candidate"] != frozen["selected_candidate"] or internal["decision"] != "improved":
        raise ValueError("No supported improvement from the tuning-selected candidate")
    if frozen["implementation_sha256"] != sha256_file(ROOT / "src/filament_seg/component_filter.py"):
        raise ValueError("Frozen filter implementation changed")
    selected = next(row for row in internal["candidate_results"] if row["config"]["name"] == frozen["selected_candidate"])
    baseline = internal["candidate_results"][0]
    interval = bootstrap_interval(baseline["per_image"], selected["per_image"])
    if interval[0] <= 0 or internal["selected_delta_global_pq"] <= .005:
        raise ValueError("Improvement is not robust enough for local submission generation")
    old_hash = sha256_file(PREVIOUS)
    selection = json.loads((ROOT / "reports/selected_model_provisional_20261104T094701Z.json").read_text())
    checkpoint = Path(selection["checkpoint"])
    if sha256_file(checkpoint) != frozen["checkpoint_sha256"] or checkpoint != Path(selection["checkpoint"]):
        raise ValueError("Checkpoint mismatch")
    if code_revision(ROOT)["source_sha256"] != selection["code_revision"]["source_sha256"]:
        raise ValueError("Original inference source changed")
    model, _ = load_candidate(checkpoint, device="cuda")
    config = ComponentFilter(**selected["config"])
    test_dir = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    images = sorted(test_dir.glob("*.jpeg"))
    if len(images) != 180:
        raise ValueError(f"Expected 180 test images, found {len(images)}")
    rows = []
    counts = {}
    tiling = selection["tiling"]
    for position, path in enumerate(images, 1):
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != (2048, 2048):
            raise ValueError(f"Invalid test image: {path}")
        probability = predict_tiled(model, raw.astype(np.float32) / 255.0, device="cuda",
                                    tile_size=tiling["tile_size"], overlap=tiling["overlap"],
                                    batch_size=tiling["batch_size"])
        masks = extract_filtered(probability, config)
        counts[path.stem] = len(masks)
        for number, mask in enumerate(masks, 1):
            rows.append({"filament_id": f"{path.stem}_{number}",
                         "segmentation_rle": encode_coco_counts(mask)})
        if position % 20 == 0 or position == len(images):
            print(f"Generated {position}/{len(images)} test images", flush=True)
    if not rows or validate_submission_rows(rows, images) != len(rows):
        raise ValueError("Submission validation failed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "outputs" / f"submission_provisional_v2_confidence_0p97_{stamp}.csv"
    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    frame.to_csv(destination, index=False, mode="x")
    saved = pd.read_csv(destination, dtype=str)
    if saved.shape != frame.shape or saved.columns.tolist() != ["filament_id", "segmentation_rle"]:
        raise ValueError("Saved CSV differs from validated rows")
    provenance = {"status": "provisional", "experiment_id": "unet_resnet34_v2_confidence_0p97",
                  "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                  "original_selection_manifest": str(ROOT / "reports/selected_model_provisional_20261104T094701Z.json"),
                  "original_selection_manifest_sha256": sha256_file(ROOT / "reports/selected_model_provisional_20261104T094701Z.json"),
                  "frozen_tuning_artifact": str(FROZEN), "frozen_tuning_sha256": sha256_file(FROZEN),
                  "internal_check_artifact": str(INTERNAL), "internal_check_sha256": sha256_file(INTERNAL),
                  "component_filter": selected["config"], "component_filter_source_sha256": sha256_file(ROOT / "src/filament_seg/component_filter.py"),
                  "original_inference_source": selection["code_revision"], "tiling": tiling,
                  "normalization": "grayscale divide by 255 exactly once", "evaluator": selection["evaluator"],
                  "internal_check_baseline_global_pq": baseline["summary"]["global_pq"],
                  "internal_check_candidate_global_pq": selected["summary"]["global_pq"],
                  "paired_capture_date_bootstrap_95pct_delta_interval": interval,
                  "csv": str(destination), "csv_sha256": sha256_file(destination),
                  "test_images_processed": len(images), "rows": len(rows),
                  "zero_prediction_images": sum(value == 0 for value in counts.values()),
                  "validation": "all rows compressed COCO RLE round-trip decoded at 2048x2048; identifiers and CSV schema checked locally",
                  "existing_official_submission_csv_sha256_unchanged": old_hash == sha256_file(PREVIOUS),
                  "uploaded_to_kaggle": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    manifest = destination.with_suffix(".provenance.json")
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    print(f"PASS {destination} rows={len(rows)} provenance={manifest} CI={interval}", flush=True)


if __name__ == "__main__":
    main()
