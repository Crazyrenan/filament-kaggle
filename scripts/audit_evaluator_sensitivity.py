"""Measure local PQ sensitivity to GT rasterization and annotation aggregation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_seg.benchmark import coco_annotation_instances
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.inference import iter_probability_instances
from filament_seg.official_metric import organizer_pq_image


def summarize(rows: list[dict]) -> dict:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    total_iou = sum(row["sum_iou"] for row in rows)
    denominator = tp + .5 * (fp + fn)
    return {"entries": len(rows), "global_pq": total_iou / denominator if denominator else 0.,
            "macro_entry_pq": float(np.mean([row["pq"] for row in rows])),
            "sq": total_iou / tp if tp else 0., "rq": tp / denominator if denominator else 0.,
            "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    benchmark = json.loads((ROOT / "reports/model_benchmark_20261104T094701Z.json").read_text())
    reference = next(row for row in benchmark["results"] if row["model"]["experiment_id"] == "unet_resnet34_v2")
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images", base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    cache = Path(reference["cache"])
    partition = benchmark["partition"]
    output = {}
    for split in ("tuning", "internal_check"):
        indices = partition[f"{split}_indices"]
        variants = {name: [] for name in ("pycoco_all_sets", "pycoco_first_set", "opencv_all_sets", "opencv_first_set")}
        many_to_one_entries = 0
        for position, index in enumerate(indices, 1):
            filename = dataset.filenames[index]
            probability = np.load(cache / f"{index:04d}.npy", mmap_mode="r")
            predicted = list(iter_probability_instances(probability, threshold=.8, min_area=500))
            for set_index in range(len(dataset.images_by_filename[filename])):
                py_masks = coco_annotation_instances(dataset, index, set_index)
                cv_masks = [item["mask"] for item in dataset.iter_instances(index, set_index)]
                py_score = organizer_pq_image(py_masks, predicted, return_details=True)
                cv_score = organizer_pq_image(cv_masks, predicted)
                if len({match["gt_index"] for match in py_score["matches"]}) < py_score["tp"] or len({match["pred_index"] for match in py_score["matches"]}) < py_score["tp"]:
                    many_to_one_entries += 1
                variants["pycoco_all_sets"].append(py_score)
                variants["opencv_all_sets"].append(cv_score)
                if set_index == 0:
                    variants["pycoco_first_set"].append(py_score)
                    variants["opencv_first_set"].append(cv_score)
            if position % 20 == 0 or position == len(indices):
                print(f"{split}: {position}/{len(indices)}", flush=True)
        output[split] = {name: summarize(rows) for name, rows in variants.items()}
        output[split]["entries_with_nonexclusive_matches"] = many_to_one_entries
    expected = reference["tuned_internal"]["summary"]
    assert abs(output["internal_check"]["pycoco_all_sets"]["global_pq"] - expected["global_pq"]) < 1e-10
    payload = {"checkpoint_sha256": reference["model"]["checkpoint_sha256"],
               "prediction_cache": str(cache), "threshold": .8, "min_area": 500,
               "variants": output,
               "limitations": "These are sensitivity analyses on known validation annotations, not evidence of hidden Kaggle scorer behavior. The public notebook uses all annotator-image entries and all IoU>0.5 pairs."}
    destination = ROOT / "reports/evaluator_sensitivity_20261109.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
