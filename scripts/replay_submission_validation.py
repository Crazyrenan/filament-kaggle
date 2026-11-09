"""Replay the exact submission generator on internal validation, without saving a CSV."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_seg.benchmark import coco_annotation_instances, load_candidate, sha256_file
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.inference import iter_probability_instances, predict_tiled
from filament_seg.official_metric import organizer_pq_image
from filament_seg.submission import decode_coco_counts, generate_submission


def main() -> None:
    provenance = json.loads((ROOT / "outputs/submission_provisional_unet_resnet34_v2_20261104T094726Z.provenance.json").read_text())
    manifest = json.loads(Path(provenance["selection_manifest"]).read_text())
    checkpoint = Path(provenance["checkpoint"])
    assert sha256_file(checkpoint) == provenance["checkpoint_sha256"]
    model, _ = load_candidate(checkpoint)
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images", base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    partition = json.loads((ROOT / "reports/validation_partition_v1.json").read_text())
    indices = partition["internal_check_indices"]
    paths = [dataset.image_dir / dataset.filenames[index] for index in indices]
    post = manifest["postprocessing"]
    tiling = manifest["tiling"]
    generated = generate_submission(model, paths, output_csv_path=None, device="cuda",
                                    threshold=post["threshold"], min_area=post["min_area"],
                                    tile_size=tiling["tile_size"], overlap=tiling["overlap"],
                                    batch_size=tiling["batch_size"])
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in generated.itertuples(index=False):
        grouped[row.filament_id.rsplit("_", 1)[0]].append(row.segmentation_rle)
    tp = fp = fn = 0
    total_iou = 0.0
    for position, index in enumerate(indices, start=1):
        filename = dataset.filenames[index]
        predictions = [decode_coco_counts(rle, (2048, 2048)) for rle in grouped[Path(filename).stem]]
        for set_index in range(len(dataset.images_by_filename[filename])):
            gt = coco_annotation_instances(dataset, index, set_index)
            result = organizer_pq_image(gt, predictions)
            tp += result["tp"]
            fp += result["fp"]
            fn += result["fn"]
            total_iou += result["sum_iou"]
        if position % 20 == 0 or position == len(indices):
            print(f"Scored {position}/{len(indices)} validation images", flush=True)
    denominator = tp + .5 * (fp + fn)
    score = total_iou / denominator
    benchmark = json.loads((ROOT / "reports/model_benchmark_20261104T094701Z.json").read_text())
    original = next(row for row in benchmark["results"] if row["model"]["experiment_id"] == "unet_resnet34_v2")
    expected = original["tuned_internal"]["summary"]
    assert abs(score - expected["global_pq"]) < 1e-5, (score, expected["global_pq"])
    assert (tp, fp, fn) == (expected["tp"], expected["fp"], expected["fn"])

    # Independently check submitted test masks against fresh model output for
    # the first and last images with predictions. This catches transpose/flip errors.
    submitted = ROOT / "outputs/submission_provisional_unet_resnet34_v2_20261104T094726Z.csv"
    import pandas as pd
    frame = pd.read_csv(submitted, dtype=str)
    test_dir = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    checked = []
    for stem in (frame.iloc[0].filament_id.rsplit("_", 1)[0],
                 frame.iloc[-1].filament_id.rsplit("_", 1)[0]):
        path = test_dir / f"{stem}.jpeg"
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        probability = predict_tiled(model, raw.astype(np.float32) / 255,
                                    device="cuda", tile_size=tiling["tile_size"],
                                    overlap=tiling["overlap"], batch_size=tiling["batch_size"])
        fresh = list(iter_probability_instances(probability, threshold=post["threshold"],
                                                 min_area=post["min_area"]))
        rows = frame[frame.filament_id.str.startswith(stem + "_")]
        assert len(fresh) == len(rows)
        assert all(np.array_equal(mask, decode_coco_counts(rle, (2048, 2048)))
                   for mask, rle in zip(fresh, rows.segmentation_rle))
        checked.append(stem)
    result = {"checkpoint_sha256": provenance["checkpoint_sha256"],
              "submission_csv_sha256": sha256_file(submitted),
              "validation_images": len(indices), "generated_rows": len(generated),
              "global_pq": score, "sq": total_iou / tp, "rq": tp / denominator,
              "tp": tp, "fp": fp, "fn": fn,
              "benchmark_global_pq": expected["global_pq"],
              "absolute_difference": abs(score - expected["global_pq"]),
              "test_image_orientation_checks": checked,
              "note": "Exact generate_submission path replayed without writing a new CSV."}
    destination = ROOT / "reports/submission_validation_replay_20261109.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
