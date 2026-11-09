"""Tune on saved tuning images, then score frozen candidates once on internal check."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_seg.benchmark import coco_annotation_instances, sha256_file
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.official_metric import organizer_pq_image
from filament_seg.postprocessing_experiments import PostprocessingConfig, extract_instances

BENCHMARK = ROOT / "reports/model_benchmark_20261104T094701Z.json"
CONFIGS = (
    PostprocessingConfig("baseline"),
    PostprocessingConfig("morph_open3", operation="open", kernel=3),
    PostprocessingConfig("morph_close3", operation="close", kernel=3),
    PostprocessingConfig("shape_1p5", operation="shape", min_elongation=1.5),
    PostprocessingConfig("shape_2", operation="shape", min_elongation=2.0),
    PostprocessingConfig("split_erode3", operation="split", kernel=3),
    PostprocessingConfig("split_erode5", operation="split", kernel=5),
)


def context():
    report = json.loads(BENCHMARK.read_text())
    baseline = next(item for item in report["results"] if item["model"]["experiment_id"] == "unet_resnet34_v2")
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images", base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    folder = Path(baseline["cache"])
    cache_manifest = json.loads((folder / "cache_manifest.json").read_text())
    if cache_manifest["checkpoint_sha256"] != baseline["model"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint/cache hash mismatch")
    if cache_manifest["validation_fingerprint"] != report["validation_fingerprint"]:
        raise ValueError("Validation/cache mismatch")
    return report, baseline, dataset, folder


def score(dataset, folder: Path, indices: list[int], config: PostprocessingConfig) -> dict:
    rows = []
    processing_seconds = 0.0
    for position, index in enumerate(indices, start=1):
        probability = np.load(folder / f"{index:04d}.npy", mmap_mode="r")
        started = time.perf_counter()
        predictions = extract_instances(probability, config)
        processing_seconds += time.perf_counter() - started
        filename = dataset.filenames[index]
        set_scores = [organizer_pq_image(coco_annotation_instances(dataset, index, set_index), predictions)
                      for set_index in range(len(dataset.images_by_filename[filename]))]
        rows.append({"index": index, "filename": filename,
                     "sum_iou": sum(item["sum_iou"] for item in set_scores),
                     "tp": sum(item["tp"] for item in set_scores),
                     "fp": sum(item["fp"] for item in set_scores),
                     "fn": sum(item["fn"] for item in set_scores),
                     "predictions": len(predictions)})
        if position % 20 == 0 or position == len(indices):
            print(f"{config.name}: {position}/{len(indices)}", flush=True)
    tp = sum(item["tp"] for item in rows)
    fp = sum(item["fp"] for item in rows)
    fn = sum(item["fn"] for item in rows)
    iou_sum = sum(item["sum_iou"] for item in rows)
    denominator = tp + .5 * (fp + fn)
    return {"config": vars(config), "summary": {
        "images": len(indices), "global_pq": iou_sum / denominator if denominator else 0.0,
        "sq": iou_sum / tp if tp else 0.0, "rq": tp / denominator if denominator else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "postprocessing_seconds_per_image": processing_seconds / len(indices),
    }, "per_image": rows}


def tune() -> Path:
    report, baseline, dataset, folder = context()
    indices = report["partition"]["tuning_indices"]
    results = []
    for config in CONFIGS:
        result = score(dataset, folder, indices, config)
        results.append(result)
        print(f"TUNING {config.name} PQ={result['summary']['global_pq']:.6f}", flush=True)
    best = max(results, key=lambda row: (row["summary"]["global_pq"],
                                          row["summary"]["sq"],
                                          -row["summary"]["postprocessing_seconds_per_image"]))
    baseline_tuning = results[0]["summary"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / "reports" / f"postprocessing_tuning_frozen_{stamp}.json"
    payload = {"phase": "frozen_after_tuning", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "benchmark_report": str(BENCHMARK), "benchmark_sha256": sha256_file(BENCHMARK),
               "checkpoint_sha256": baseline["model"]["checkpoint_sha256"],
               "cache_manifest_sha256": sha256_file(folder / "cache_manifest.json"),
               "implementation_sha256": sha256_file(ROOT / "src/filament_seg/postprocessing_experiments.py"),
               "partition_fingerprint": report["validation_fingerprint"],
               "tuning_indices": indices, "internal_indices": report["partition"]["internal_check_indices"],
               "selection_metric": "tuning Global PQ; SQ then lower postprocessing latency tie breaks",
               "candidate_results": results, "selected_candidate": best["config"]["name"],
               "baseline_tuning_global_pq": baseline_tuning["global_pq"]}
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"FROZEN {best['config']['name']} artifact={path}", flush=True)
    return path


def internal(frozen_path: Path) -> Path:
    frozen = json.loads(frozen_path.read_text())
    report, baseline, dataset, folder = context()
    if frozen["phase"] != "frozen_after_tuning" or frozen["benchmark_sha256"] != sha256_file(BENCHMARK):
        raise ValueError("Invalid frozen tuning artifact")
    if frozen["implementation_sha256"] != sha256_file(ROOT / "src/filament_seg/postprocessing_experiments.py"):
        raise ValueError("Candidate implementation changed after freeze")
    if frozen["checkpoint_sha256"] != baseline["model"]["checkpoint_sha256"] or frozen["cache_manifest_sha256"] != sha256_file(folder / "cache_manifest.json"):
        raise ValueError("Checkpoint or cache changed after freeze")
    if frozen["partition_fingerprint"] != report["validation_fingerprint"] or frozen["internal_indices"] != report["partition"]["internal_check_indices"]:
        raise ValueError("Internal partition changed")
    if [row["config"] for row in frozen["candidate_results"]] != [vars(config) for config in CONFIGS]:
        raise ValueError("Candidate configurations changed")
    indices = frozen["internal_indices"]
    results = []
    for config in CONFIGS:
        result = score(dataset, folder, indices, config)
        results.append(result)
        print(f"INTERNAL {config.name} PQ={result['summary']['global_pq']:.6f}", flush=True)
    selected = next(row for row in results if row["config"]["name"] == frozen["selected_candidate"])
    baseline_result = results[0]
    delta = selected["summary"]["global_pq"] - baseline_result["summary"]["global_pq"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / "reports" / f"postprocessing_internal_once_{stamp}.json"
    payload = {"phase": "internal_scored_once", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "frozen_tuning_artifact": str(frozen_path.resolve()),
               "frozen_tuning_sha256": sha256_file(frozen_path),
               "checkpoint_sha256": baseline["model"]["checkpoint_sha256"],
               "evaluator": report["evaluator"], "candidate_results": results,
               "selected_candidate": frozen["selected_candidate"],
               "selected_delta_global_pq": delta,
               "inference_seconds_per_image": baseline["inference"]["seconds_per_image"],
               "inference_peak_gpu_memory_mib": baseline["inference"]["peak_gpu_memory_mib"],
               "decision": "improved" if delta > .005 else "no_defensible_improvement",
               "limitation": "The internal check is not independent: historical v2 model selection used these images. Hidden evaluator parity is unverified."}
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"INTERNAL_REPORT {path} selected={frozen['selected_candidate']} delta={delta:.6f} decision={payload['decision']}", flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("tune", "internal"))
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args()
    if args.phase == "tune":
        tune()
    elif args.frozen is None:
        parser.error("--frozen is required for internal scoring")
    else:
        internal(args.frozen)
