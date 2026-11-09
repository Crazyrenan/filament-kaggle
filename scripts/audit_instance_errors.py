"""Quantify v2 instance errors on tuning images without touching internal check."""

from __future__ import annotations

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
from filament_seg.benchmark import coco_annotation_instances, sha256_file
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.inference import iter_probability_instances
from filament_seg.official_metric import organizer_pq_image


def overlap_patterns(gt: list[np.ndarray], pred: list[np.ndarray]) -> tuple[int, int]:
    """Count GT split across predictions and predictions spanning GT, using 10% overlaps."""
    if not gt or not pred:
        return 0, 0
    intersections = np.zeros((len(gt), len(pred)), dtype=np.int64)
    for i, truth in enumerate(gt):
        for j, candidate in enumerate(pred):
            intersections[i, j] = np.count_nonzero(truth & candidate.astype(bool))
    gt_area = np.array([np.count_nonzero(mask) for mask in gt])
    pred_area = np.array([np.count_nonzero(mask) for mask in pred])
    fragmented = int(np.count_nonzero(np.sum(intersections >= .1 * gt_area[:, None], axis=1) >= 2))
    merged = int(np.count_nonzero(np.sum(intersections >= .1 * pred_area[None, :], axis=0) >= 2))
    return fragmented, merged


def main() -> None:
    report = json.loads((ROOT / "reports/model_benchmark_20261104T094701Z.json").read_text())
    result = next(row for row in report["results"] if row["model"]["experiment_id"] == "unet_resnet34_v2")
    partition = report["partition"]
    indices = partition["tuning_indices"]
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images", base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    folder = Path(result["cache"])
    band = np.zeros((2048, 2048), dtype=bool)
    for coordinate in (384, 768, 1152, 1536):
        band[max(0, coordinate - 16):coordinate + 16, :] = True
        band[:, max(0, coordinate - 16):coordinate + 16] = True
    rows = []
    near_fn = near_gt = far_fn = far_gt = 0
    for position, index in enumerate(indices, start=1):
        probability = np.load(folder / f"{index:04d}.npy", mmap_mode="r")
        filename = dataset.filenames[index]
        predictions = list(iter_probability_instances(probability, threshold=.8, min_area=500))
        scores = []
        fragmented = merged = 0
        first_gt = None
        for set_index in range(len(dataset.images_by_filename[filename])):
            gt = coco_annotation_instances(dataset, index, set_index)
            if first_gt is None:
                first_gt = gt
            detail = organizer_pq_image(gt, predictions, return_details=True)
            scores.append(detail)
            frag, merge = overlap_patterns(gt, predictions)
            fragmented += frag
            merged += merge
        gt_union = np.zeros((2048, 2048), dtype=bool)
        for mask in first_gt or []:
            gt_union |= mask
        false_negative = gt_union & (probability < .8)
        near_fn += int(np.count_nonzero(false_negative & band))
        near_gt += int(np.count_nonzero(gt_union & band))
        far_fn += int(np.count_nonzero(false_negative & ~band))
        far_gt += int(np.count_nonzero(gt_union & ~band))
        rows.append({"index": index, "filename": filename,
                     "annotation_sets": len(scores),
                     "tp": sum(item["tp"] for item in scores),
                     "fp": sum(item["fp"] for item in scores),
                     "fn": sum(item["fn"] for item in scores),
                     "fragmented_gt_heuristic": fragmented,
                     "merged_predictions_heuristic": merged})
        if position % 10 == 0 or position == len(indices):
            print(f"Audited {position}/{len(indices)}", flush=True)
    summary = {name: sum(row[name] for row in rows)
               for name in ("tp", "fp", "fn", "fragmented_gt_heuristic", "merged_predictions_heuristic")}
    summary.update({"images": len(rows), "annotation_sets": sum(row["annotation_sets"] for row in rows),
                    "seam_band_half_width_px": 16,
                    "seam_gt_false_negative_rate": near_fn / near_gt if near_gt else None,
                    "off_seam_gt_false_negative_rate": far_fn / far_gt if far_gt else None,
                    "seam_gt_pixels": near_gt, "off_seam_gt_pixels": far_gt,
                    "interpretation": "Pixel error enrichment near tile starts is descriptive, not causal."})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "reports" / f"instance_error_audit_tuning_{stamp}.json"
    payload = {"status": "tuning-only", "baseline_checkpoint_sha256": result["model"]["checkpoint_sha256"],
               "cache_manifest_sha256": sha256_file(folder / "cache_manifest.json"),
               "partition_fingerprint": partition["validation_fingerprint"],
               "definitions": {"fragmentation": "one GT overlaps at least two predictions by >=10% of GT area",
                               "merge": "one prediction overlaps at least two GT by >=10% of prediction area",
                               "seam": "GT foreground pixel false-negative rate within 16px of starts of 512px tiles at 128px overlap"},
               "summary": summary, "per_image": rows}
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    figure_paths = []
    for key in ("fp", "fn", "fragmented_gt_heuristic", "merged_predictions_heuristic"):
        row = max(rows, key=lambda item: item[key])
        index = row["index"]
        raw = cv2.imread(str(dataset.image_dir / row["filename"]), cv2.IMREAD_GRAYSCALE)
        probability = np.load(folder / f"{index:04d}.npy", mmap_mode="r")
        gt = coco_annotation_instances(dataset, index, 0)
        pred = list(iter_probability_instances(probability, threshold=.8, min_area=500))
        detail = organizer_pq_image(gt, pred, return_details=True)
        from filament_seg.experiment_tracking import instance_error_overlay
        overlay = instance_error_overlay(gt, pred, detail) if gt or pred else np.zeros((*raw.shape, 3), dtype=np.uint8)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(raw[::4, ::4], cmap="gray", vmin=0, vmax=255)
        axes[1].imshow(probability[::4, ::4], cmap="magma", vmin=0, vmax=1)
        axes[2].imshow(raw[::4, ::4], cmap="gray", vmin=0, vmax=255)
        axes[2].imshow(overlay[::4, ::4], alpha=.65)
        for axis, title in zip(axes, ("image", "probability", "GT/pred error overlay")):
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(f"{key}: {row['filename']} ({row[key]})")
        fig.tight_layout()
        target = ROOT / "reports/figures" / f"instance_error_{key}_{stamp}.png"
        fig.savefig(target, dpi=140)
        plt.close(fig)
        figure_paths.append(str(target))
    print(json.dumps({"audit": str(destination), "figures": figure_paths, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
