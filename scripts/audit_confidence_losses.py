"""Describe confidence-filter decisions; internal labels are diagnostic only."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_seg.benchmark import coco_annotation_instances, sha256_file
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.inference import iter_probability_instances
from filament_seg.official_metric import organizer_pq_image


def features(probability: np.ndarray, mask: np.ndarray) -> dict:
    y, x = np.nonzero(mask)
    values = probability[y, x]
    y0, y1, x0, x1 = int(y.min()), int(y.max()) + 1, int(x.min()), int(x.max()) + 1
    height, width = y1 - y0, x1 - x0
    eigen = np.linalg.eigvalsh(np.cov(np.stack((y, x))))
    elongation = float(np.sqrt(eigen[-1] / max(eigen[0], 1e-6)))
    crop = np.ascontiguousarray(mask[y0:y1, x0:x1].astype(np.uint8))
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, closed=True) for contour in contours)
    area = len(y)
    return {"mean_probability": float(values.mean()),
            "q10_probability": float(np.quantile(values, .1)),
            "q25_probability": float(np.quantile(values, .25)),
            "q50_probability": float(np.quantile(values, .5)),
            "q90_probability": float(np.quantile(values, .9)),
            "area_px": area, "elongation": elongation,
            "bbox_width": width, "bbox_height": height,
            "bbox_aspect": max(width, height) / min(width, height),
            "bbox_extent": area / (width * height),
            "compactness": float(4 * np.pi * area / perimeter**2) if perimeter > 0 else None,
            "radial_distance": float(np.hypot(x.mean() - 1024, y.mean() - 1024) / 1024)}


def describe(rows: list[dict]) -> dict:
    keys = ("mean_probability", "q10_probability", "q25_probability", "q50_probability",
            "q90_probability", "area_px", "elongation", "bbox_aspect", "bbox_extent",
            "compactness", "radial_distance", "matched_iou")
    result = {"events": len(rows)}
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            result[key] = {"median": float(np.median(values)),
                           "p10": float(np.percentile(values, 10)),
                           "p90": float(np.percentile(values, 90))}
    return result


def main() -> None:
    benchmark = json.loads((ROOT / "reports/model_benchmark_20261104T094701Z.json").read_text())
    original = next(row for row in benchmark["results"] if row["model"]["experiment_id"] == "unet_resnet34_v2")
    cache = Path(original["cache"])
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images", base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    groups: dict[str, dict[str, list[dict]]] = {}
    for split in ("tuning", "internal_check"):
        categories: dict[str, list[dict]] = defaultdict(list)
        indices = benchmark["partition"][f"{split}_indices"]
        for position, index in enumerate(indices, 1):
            filename = dataset.filenames[index]
            probability = np.load(cache / f"{index:04d}.npy", mmap_mode="r")
            predictions = list(iter_probability_instances(probability, threshold=.8, min_area=500))
            component_features = [features(probability, mask) for mask in predictions]
            for set_index in range(len(dataset.images_by_filename[filename])):
                gt = coco_annotation_instances(dataset, index, set_index)
                result = organizer_pq_image(gt, predictions, return_details=True)
                by_pred: dict[int, list[float]] = defaultdict(list)
                for match in result["matches"]:
                    by_pred[match["pred_index"]].append(match["iou"])
                for pred_index, descriptor in enumerate(component_features):
                    retained = descriptor["mean_probability"] >= .97
                    matched = pred_index in by_pred
                    category = ("retained" if retained else "rejected") + ("_matched" if matched else "_fp")
                    categories[category].append({"filename": filename, "set_index": set_index,
                                                 "pred_index": pred_index,
                                                 "matched_iou": max(by_pred[pred_index]) if matched else None,
                                                 **descriptor})
            if position % 20 == 0 or position == len(indices):
                print(f"{split}: {position}/{len(indices)}", flush=True)
        groups[split] = categories
    summary = {split: {category: describe(rows) for category, rows in categories.items()}
               for split, categories in groups.items()}
    internal = summary["internal_check"]
    assert internal["rejected_matched"]["events"] == 56
    assert internal["rejected_fp"]["events"] == 185
    payload = {"scope": "tuning feature analysis; internal outcomes diagnostic only, never used to set candidate parameters",
               "checkpoint_sha256": original["model"]["checkpoint_sha256"],
               "cache_manifest_sha256": sha256_file(cache / "cache_manifest.json"),
               "split_fingerprint": benchmark["validation_fingerprint"],
               "event_note": "Each prediction is counted once per annotation set; matched_iou is the best qualifying IoU for that prediction.",
               "summary": summary,
               "records": groups}
    destination = ROOT / "reports/confidence_loss_audit_20261109.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row_axes, split in zip(axes, ("tuning", "internal_check")):
        for axis, key in zip(row_axes, ("area_px", "elongation", "q10_probability")):
            for category in ("rejected_matched", "rejected_fp"):
                values = [item[key] for item in groups[split][category]]
                axis.hist(values, bins=25, density=True, alpha=.5, label=category)
            axis.set_title(f"{split}: {key}")
            axis.legend(fontsize=8)
    fig.tight_layout()
    figure = ROOT / "reports/figures/confidence_loss_features_20261109.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    print(f"Saved {destination} and {figure}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
