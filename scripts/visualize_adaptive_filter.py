"""Render tuning-only gains and losses for the frozen adaptive rescue rule."""

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
from filament_seg.adaptive_instance_filter import AdaptiveFilterConfig, extract_adaptive
from filament_seg.benchmark import coco_annotation_instances
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.experiment_tracking import instance_error_overlay
from filament_seg.official_metric import organizer_pq_image


def pq(row: dict) -> float:
    denominator = row["tp"] + .5 * (row["fp"] + row["fn"])
    return row["sum_iou"] / denominator if denominator else 0.0


def main() -> None:
    frozen = json.loads((ROOT / "reports/adaptive_filter_tuning_frozen_20261109T052841Z.json").read_text())
    original_rows = frozen["candidate_results"][0]["per_image"]
    chosen = next(row for row in frozen["candidate_results"] if row["config"]["name"] == frozen["selected_candidate"])
    candidate_rows = chosen["per_image"]
    configuration = AdaptiveFilterConfig(**chosen["config"])
    changes = [(pq(after) - pq(before), before["index"], before["filename"])
               for before, after in zip(original_rows, candidate_rows)]
    cases = {"gain": max(changes), "loss": min(changes)}
    benchmark = json.loads((ROOT / "reports/model_benchmark_20261104T094701Z.json").read_text())
    model = next(row for row in benchmark["results"] if row["model"]["experiment_id"] == "unet_resnet34_v2")
    cache = Path(model["cache"])
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images", base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    for label, (delta, index, filename) in cases.items():
        image = cv2.imread(str(dataset.image_dir / filename), cv2.IMREAD_GRAYSCALE)
        probability = np.load(cache / f"{index:04d}.npy", mmap_mode="r")
        gt = coco_annotation_instances(dataset, index, 0)
        baseline = extract_adaptive(probability, AdaptiveFilterConfig("confidence_0p97"))
        candidate = extract_adaptive(probability, configuration)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(image[::4, ::4], cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("image")
        axes[1].imshow(probability[::4, ::4], cmap="magma", vmin=0, vmax=1)
        axes[1].set_title("probability")
        for axis, masks, title in ((axes[2], baseline, "confidence 0.97"),
                                   (axes[3], candidate, "adaptive rescue")):
            detail = organizer_pq_image(gt, masks, return_details=True)
            axis.imshow(image[::4, ::4], cmap="gray", vmin=0, vmax=255)
            if gt or masks:
                overlay = instance_error_overlay(gt, masks, detail)
                axis.imshow(overlay[::4, ::4], alpha=.65)
            axis.set_title(f"{title}: {len(masks)} masks; set-0 PQ {detail['pq']:.3f}")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(f"Tuning {label}: {filename}; all-set per-image PQ change {delta:+.3f}")
        fig.tight_layout()
        destination = ROOT / "reports/figures" / f"adaptive_filter_tuning_{label}_20261109.png"
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        print(destination, flush=True)


if __name__ == "__main__":
    main()
