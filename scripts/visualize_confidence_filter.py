"""Render tuning-only gains and losses for the frozen confidence filter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_seg.benchmark import coco_annotation_instances
from filament_seg.component_filter import ComponentFilter, extract_filtered
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.experiment_tracking import instance_error_overlay
from filament_seg.official_metric import organizer_pq_image


def pq(row):
    denominator = row["tp"] + .5 * (row["fp"] + row["fn"])
    return row["sum_iou"] / denominator if denominator else 0.


def main() -> None:
    frozen = json.loads((ROOT / "reports/component_filter_frozen_20261109T043725Z.json").read_text())
    baseline = frozen["candidate_results"][0]["per_image"]
    selected = next(row for row in frozen["candidate_results"] if row["config"]["name"] == frozen["selected_candidate"])
    config = ComponentFilter(**selected["config"])
    deltas = [(pq(new) - pq(old), old["index"], old["filename"])
              for old, new in zip(baseline, selected["per_image"])]
    cases = {"gain": max(deltas), "loss": min(deltas)}
    benchmark = json.loads((ROOT / "reports/model_benchmark_20261104T094701Z.json").read_text())
    model = next(row for row in benchmark["results"] if row["model"]["experiment_id"] == "unet_resnet34_v2")
    cache = Path(model["cache"])
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images", base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    from filament_seg.inference import iter_probability_instances
    for label, (delta, index, filename) in cases.items():
        raw = cv2.imread(str(dataset.image_dir / filename), cv2.IMREAD_GRAYSCALE)
        probability = np.load(cache / f"{index:04d}.npy", mmap_mode="r")
        gt = coco_annotation_instances(dataset, index, 0)
        original = list(iter_probability_instances(probability, threshold=.8, min_area=500))
        filtered = extract_filtered(probability, config)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(raw[::4, ::4], cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("grayscale image")
        axes[1].imshow(probability[::4, ::4], cmap="magma", vmin=0, vmax=1)
        axes[1].set_title("probability")
        for axis, masks, title in ((axes[2], original, "baseline"), (axes[3], filtered, "confidence 0.97")):
            detail = organizer_pq_image(gt, masks, return_details=True)
            axis.imshow(raw[::4, ::4], cmap="gray", vmin=0, vmax=255)
            if gt or masks:
                overlay = instance_error_overlay(gt, masks, detail)
                axis.imshow(overlay[::4, ::4], alpha=.65)
            axis.set_title(f"{title}: {len(masks)} predictions; set-0 PQ {detail['pq']:.3f}")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(f"Tuning {label}: {filename}; all-set per-image PQ change {delta:+.3f}")
        fig.tight_layout()
        destination = ROOT / "reports/figures" / f"confidence_filter_tuning_{label}_20261109.png"
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        print(destination, flush=True)


if __name__ == "__main__":
    main()
