"""Render paired instance-error examples for the benchmarked candidates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.dataset import SolarFilamentDataset
from filament_seg.benchmark import coco_annotation_instances
from filament_seg.experiment_tracking import instance_error_overlay
from filament_seg.inference import iter_probability_instances
from filament_seg.official_metric import organizer_pq_image


def main(report_path: Path) -> list[Path]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report["results"]
    if not results:
        raise ValueError("Benchmark has no candidates")
    by_filename = {}
    for item in results:
        for row in item["tuned_internal"]["per_image"]:
            by_filename.setdefault(row["filename"], []).append(row)
    candidates = {
        "false_positives": max(by_filename, key=lambda name: sum(row["fp"] for row in by_filename[name])),
        "missed_instances": max(by_filename, key=lambda name: sum(row["fn"] for row in by_filename[name])),
        "model_disagreement": max(by_filename, key=lambda name: max(row["pq"] for row in by_filename[name]) - min(row["pq"] for row in by_filename[name])),
    }
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images",
                                   base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outputs = []
    for label, filename in candidates.items():
        index = dataset.filenames.index(filename)
        raw = cv2.imread(str(dataset.image_dir / filename), cv2.IMREAD_GRAYSCALE)
        gt_instances = coco_annotation_instances(dataset, index, 0)
        gt_union = np.maximum.reduce(gt_instances) if gt_instances else np.zeros_like(raw)
        columns = 2 + 2 * len(results)
        fig, axes = plt.subplots(1, columns, figsize=(3.4 * columns, 4.2))
        axes[0].imshow(raw[::4, ::4], cmap="gray", vmin=0, vmax=255)
        axes[0].set_title(f"Image\n{filename}")
        axes[1].imshow(gt_union[::4, ::4], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"GT set 0\n{len(gt_instances)} instances")
        for position, item in enumerate(results):
            probability = np.load(Path(item["cache"]) / f"{index:04d}.npy", mmap_mode="r")
            parameters = item["selected_postprocessing"]
            predicted = list(iter_probability_instances(
                probability, threshold=parameters["threshold"], min_area=parameters["min_area"]
            ))
            pq = organizer_pq_image(gt_instances, predicted, return_details=True)
            overlay = instance_error_overlay(gt_instances, predicted, pq)
            axes[2 + position * 2].imshow(probability[::4, ::4], cmap="magma", vmin=0, vmax=1)
            axes[2 + position * 2].set_title(f"{item['model']['experiment_id']}\nprobability")
            axes[3 + position * 2].imshow(raw[::4, ::4], cmap="gray", vmin=0, vmax=255)
            axes[3 + position * 2].imshow(overlay[::4, ::4], alpha=0.65)
            axes[3 + position * 2].set_title(
                f"Set-0 PQ {pq['pq']:.3f} TP {pq['tp']} FP {pq['fp']} FN {pq['fn']}\n"
                "green matched, red FP, blue FN"
            )
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        target = ROOT / "reports/figures" / f"comparison_{label}_{stamp}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=140)
        plt.close(fig)
        outputs.append(target)
        print(f"Saved {target}", flush=True)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    main(args.report)
