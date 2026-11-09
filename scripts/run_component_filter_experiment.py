"""Tune a small component-filter grid, freeze it, then score internal once."""

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
from filament_seg.component_filter import ComponentFilter, extract_filtered
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.official_metric import organizer_pq_image

REPORT = ROOT / "reports/model_benchmark_20261104T094701Z.json"
SOURCE = ROOT / "src/filament_seg/component_filter.py"
CONFIGS = (
    ComponentFilter("baseline"),
    ComponentFilter("confidence_0p95", min_mean_probability=.95),
    ComponentFilter("confidence_0p96", min_mean_probability=.96),
    ComponentFilter("confidence_0p97", min_mean_probability=.97),
    ComponentFilter("confidence_0p975", min_mean_probability=.975),
    ComponentFilter("radius_0p85", max_radial_distance=.85),
    ComponentFilter("radius_0p80", max_radial_distance=.80),
    ComponentFilter("confidence_0p96_radius_0p85", min_mean_probability=.96, max_radial_distance=.85),
    ComponentFilter("confidence_0p97_radius_0p85", min_mean_probability=.97, max_radial_distance=.85),
)


def context():
    report = json.loads(REPORT.read_text())
    model = next(row for row in report["results"] if row["model"]["experiment_id"] == "unet_resnet34_v2")
    cache = Path(model["cache"])
    manifest = json.loads((cache / "cache_manifest.json").read_text())
    assert manifest["checkpoint_sha256"] == model["model"]["checkpoint_sha256"]
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images", base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    return report, model, cache, dataset


def score(indices: list[int], config: ComponentFilter, cache: Path, dataset) -> dict:
    records = []
    seconds = 0.0
    for position, index in enumerate(indices, 1):
        probability = np.load(cache / f"{index:04d}.npy", mmap_mode="r")
        start = time.perf_counter()
        predictions = extract_filtered(probability, config)
        seconds += time.perf_counter() - start
        filename = dataset.filenames[index]
        set_scores = [organizer_pq_image(coco_annotation_instances(dataset, index, j), predictions)
                      for j in range(len(dataset.images_by_filename[filename]))]
        records.append({"index": index, "filename": filename,
                        "sum_iou": sum(item["sum_iou"] for item in set_scores),
                        "tp": sum(item["tp"] for item in set_scores),
                        "fp": sum(item["fp"] for item in set_scores),
                        "fn": sum(item["fn"] for item in set_scores)})
        if position % 20 == 0 or position == len(indices):
            print(f"{config.name}: {position}/{len(indices)}", flush=True)
    tp = sum(row["tp"] for row in records)
    fp = sum(row["fp"] for row in records)
    fn = sum(row["fn"] for row in records)
    iou = sum(row["sum_iou"] for row in records)
    denominator = tp + .5 * (fp + fn)
    return {"config": vars(config), "summary": {"images": len(indices),
            "global_pq": iou / denominator, "sq": iou / tp, "rq": tp / denominator,
            "tp": tp, "fp": fp, "fn": fn,
            "postprocessing_seconds_per_image": seconds / len(indices)},
            "per_image": records}


def tune():
    report, model, cache, dataset = context()
    indices = report["partition"]["tuning_indices"]
    results = [score(indices, config, cache, dataset) for config in CONFIGS]
    best = max(results, key=lambda row: (row["summary"]["global_pq"], row["summary"]["sq"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "reports" / f"component_filter_frozen_{stamp}.json"
    artifact = {"phase": "frozen_after_tuning", "created_utc": datetime.now(timezone.utc).isoformat(),
                "benchmark_sha256": sha256_file(REPORT),
                "checkpoint_sha256": model["model"]["checkpoint_sha256"],
                "cache_manifest_sha256": sha256_file(cache / "cache_manifest.json"),
                "implementation_sha256": sha256_file(SOURCE),
                "validation_fingerprint": report["validation_fingerprint"],
                "tuning_indices": indices, "internal_indices": report["partition"]["internal_check_indices"],
                "candidate_results": results, "selected_candidate": best["config"]["name"],
                "selection_metric": "tuning Global PQ; SQ tie break"}
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    print(f"FROZEN {best['config']['name']} PQ={best['summary']['global_pq']:.6f} {destination}", flush=True)


def internal(frozen_path: Path):
    frozen = json.loads(frozen_path.read_text())
    report, model, cache, dataset = context()
    if frozen["phase"] != "frozen_after_tuning" or frozen["benchmark_sha256"] != sha256_file(REPORT):
        raise ValueError("Benchmark changed after freeze")
    if frozen["implementation_sha256"] != sha256_file(SOURCE):
        raise ValueError("Implementation changed after freeze")
    if frozen["checkpoint_sha256"] != model["model"]["checkpoint_sha256"] or frozen["cache_manifest_sha256"] != sha256_file(cache / "cache_manifest.json"):
        raise ValueError("Checkpoint/cache changed after freeze")
    if frozen["validation_fingerprint"] != report["validation_fingerprint"] or frozen["internal_indices"] != report["partition"]["internal_check_indices"]:
        raise ValueError("Partition changed after freeze")
    if [row["config"] for row in frozen["candidate_results"]] != [vars(config) for config in CONFIGS]:
        raise ValueError("Candidate grid changed after freeze")
    indices = frozen["internal_indices"]
    results = [score(indices, config, cache, dataset) for config in CONFIGS]
    selected = next(row for row in results if row["config"]["name"] == frozen["selected_candidate"])
    baseline = results[0]
    delta = selected["summary"]["global_pq"] - baseline["summary"]["global_pq"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "reports" / f"component_filter_internal_{stamp}.json"
    artifact = {"phase": "internal_scored_once", "frozen_artifact": str(frozen_path.resolve()),
                "frozen_sha256": sha256_file(frozen_path), "candidate_results": results,
                "selected_candidate": frozen["selected_candidate"], "selected_delta_global_pq": delta,
                "inference_seconds_per_image": model["inference"]["seconds_per_image"],
                "peak_inference_vram_mib": model["inference"]["peak_gpu_memory_mib"],
                "decision": "improved" if delta > .005 else "no_defensible_improvement",
                "limitation": "Internal check is not independent of historical original-v2 decisions; hidden scorer parity unknown."}
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    print(f"INTERNAL selected={frozen['selected_candidate']} delta={delta:.6f} decision={artifact['decision']} report={destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("tune", "internal"))
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args()
    if args.phase == "tune":
        tune()
    elif args.frozen is None:
        parser.error("--frozen is required")
    else:
        internal(args.frozen)
