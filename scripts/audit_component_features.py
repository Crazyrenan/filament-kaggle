"""Compare matched and false-positive component features on tuning images only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_seg.benchmark import coco_annotation_instances
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.inference import iter_probability_instances
from filament_seg.official_metric import organizer_pq_image


def summary(values: list[float]) -> dict:
    return {"count": len(values), "p10": float(np.percentile(values, 10)),
            "median": float(np.median(values)), "p90": float(np.percentile(values, 90))} if values else {"count": 0}


def main() -> None:
    benchmark = json.loads((ROOT / "reports/model_benchmark_20261104T094701Z.json").read_text())
    model = next(item for item in benchmark["results"] if item["model"]["experiment_id"] == "unet_resnet34_v2")
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images", base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    cache = Path(model["cache"])
    features = {"matched": {"mean_probability": [], "area_px": [], "radial_distance": []},
                "false_positive": {"mean_probability": [], "area_px": [], "radial_distance": []}}
    for position, index in enumerate(benchmark["partition"]["tuning_indices"], 1):
        filename = dataset.filenames[index]
        probability = np.load(cache / f"{index:04d}.npy", mmap_mode="r")
        predicted = list(iter_probability_instances(probability, threshold=.8, min_area=500))
        component_values = []
        for mask in predicted:
            y, x = np.nonzero(mask)
            component_values.append({"mean_probability": float(np.mean(probability[y, x])),
                                     "area_px": len(y),
                                     "radial_distance": float(np.hypot(np.mean(x)-1024, np.mean(y)-1024) / 1024)})
        for set_index in range(len(dataset.images_by_filename[filename])):
            gt = coco_annotation_instances(dataset, index, set_index)
            detail = organizer_pq_image(gt, predicted, return_details=True)
            unmatched = set(detail["unmatched_pred"])
            for pred_index, values in enumerate(component_values):
                group = "false_positive" if pred_index in unmatched else "matched"
                for key, value in values.items():
                    features[group][key].append(value)
        if position % 20 == 0 or position == 71:
            print(f"Feature audit {position}/71", flush=True)
    payload = {"scope": "tuning only, each prediction counted for each annotation set",
               "threshold": .8, "min_area": 500,
               "features": {group: {key: summary(values) for key, values in columns.items()}
                            for group, columns in features.items()}}
    destination = ROOT / "reports/component_feature_audit_20261109.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    figure = ROOT / "reports/figures/component_features_20261109.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, key in zip(axes, ("mean_probability", "area_px", "radial_distance")):
        for group in ("matched", "false_positive"):
            axis.hist(features[group][key], bins=30, alpha=.5, density=True, label=group)
        axis.set_title(key)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
