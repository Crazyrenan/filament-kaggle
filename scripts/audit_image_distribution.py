"""Compare unlabeled image appearance and prediction distributions across splits."""

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
import pandas as pd
from pycocotools import mask as coco_mask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.inference import iter_probability_instances


def image_stats(path: Path) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (2048, 2048):
        raise ValueError(f"Invalid image: {path}")
    sample = image[::4, ::4]
    _, disk_candidate = cv2.threshold(sample, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(disk_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    disk_mask = np.zeros(sample.shape, dtype=np.uint8)
    cv2.drawContours(disk_mask, [largest], -1, 1, cv2.FILLED)
    pixels = sample[disk_mask != 0]
    p5, p50, p95 = np.percentile(pixels, (5, 50, 95))
    return {"filename": path.name, "disk_p5": float(p5), "disk_median": float(p50),
            "disk_p95": float(p95), "disk_contrast_5_95": float(p95 - p5),
            "disk_std": float(pixels.std()), "disk_area_fraction": float(disk_mask.mean()),
            "disk_center_x_px": float(cx * 4), "disk_center_y_px": float(cy * 4),
            "disk_radius_px": float(radius * 4),
            "global_mean": float(sample.mean()), "global_saturated_fraction": float(np.mean(sample >= 250))}


def summarize(rows: list[dict]) -> dict:
    keys = ("disk_p5", "disk_median", "disk_p95", "disk_contrast_5_95", "disk_std",
            "disk_area_fraction", "disk_center_x_px", "disk_center_y_px", "disk_radius_px",
            "global_mean", "global_saturated_fraction")
    return {"images": len(rows), **{key: {"median": float(np.median([row[key] for row in rows])),
                                     "p05": float(np.percentile([row[key] for row in rows], 5)),
                                     "p95": float(np.percentile([row[key] for row in rows], 95))}
                                for key in keys}}


def main() -> None:
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    train = base / "train"
    dataset = SolarFilamentDataset(train / "train_images", train / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    permutation = np.random.default_rng(42).permutation(707)
    groups = {"train": [dataset.image_dir / dataset.filenames[i] for i in permutation[:565]],
              "validation": [dataset.image_dir / dataset.filenames[i] for i in permutation[565:]],
              "test": sorted((base / "test/test_images").glob("*.jpeg"))}
    assert [len(paths) for paths in groups.values()] == [565, 142, 180]
    details = {}
    for group, paths in groups.items():
        rows = []
        for index, path in enumerate(paths, start=1):
            rows.append(image_stats(path))
            if index % 100 == 0 or index == len(paths):
                print(f"{group}: {index}/{len(paths)}", flush=True)
        details[group] = rows
    summary = {group: summarize(rows) for group, rows in details.items()}

    benchmark = json.loads((ROOT / "reports/model_benchmark_20261104T094701Z.json").read_text())
    baseline = next(item for item in benchmark["results"] if item["model"]["experiment_id"] == "unet_resnet34_v2")
    cache = Path(baseline["cache"])
    val_counts = []
    val_areas = []
    for index in permutation[565:]:
        probability = np.load(cache / f"{index:04d}.npy", mmap_mode="r")
        masks = list(iter_probability_instances(probability, threshold=.8, min_area=500))
        val_counts.append(len(masks))
        val_areas.extend(int(mask.sum()) for mask in masks)
    submitted = pd.read_csv(ROOT / "outputs/submission_provisional_unet_resnet34_v2_20261104T094726Z.csv", dtype=str)
    test_counts = defaultdict(int)
    test_areas = []
    for row in submitted.itertuples(index=False):
        test_counts[row.filament_id.rsplit("_", 1)[0]] += 1
        test_areas.append(int(coco_mask.area({"size": [2048, 2048], "counts": row.segmentation_rle.encode("ascii")})))
    test_counts_all = [test_counts[path.stem] for path in groups["test"]]
    prediction_summary = {
        "validation": {"images": 142, "instances": len(val_areas), "count_median": float(np.median(val_counts)),
                       "count_p95": float(np.percentile(val_counts, 95)), "zero_count_images": int(np.count_nonzero(np.array(val_counts) == 0)),
                       "area_median_px": float(np.median(val_areas)), "area_p95_px": float(np.percentile(val_areas, 95))},
        "test": {"images": 180, "instances": len(test_areas), "count_median": float(np.median(test_counts_all)),
                 "count_p95": float(np.percentile(test_counts_all, 95)), "zero_count_images": int(np.count_nonzero(np.array(test_counts_all) == 0)),
                 "area_median_px": float(np.median(test_areas)), "area_p95_px": float(np.percentile(test_areas, 95))},
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    metrics = ("disk_median", "disk_contrast_5_95", "disk_std", "disk_radius_px", "disk_center_x_px", "disk_center_y_px")
    for axis, key in zip(axes.flat, metrics):
        axis.boxplot([[row[key] for row in details[group]] for group in groups], tick_labels=list(groups), showfliers=False)
        axis.set_title(key)
        axis.grid(alpha=.2)
    fig.tight_layout()
    figure = ROOT / "reports/figures/image_distribution_20261109.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    payload = {"method": "grayscale JPEG, every fourth pixel; threshold>30 largest contour for approximate solar disk; no test labels",
               "split_counts": {group: len(paths) for group, paths in groups.items()},
               "image_summary": summary, "prediction_summary": prediction_summary,
               "figure": str(figure), "caveat": "Disk contour is an appearance proxy, not a calibrated solar-limb measurement."}
    destination = ROOT / "reports/image_distribution_audit_20261109.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
