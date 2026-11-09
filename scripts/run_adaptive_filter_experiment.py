"""Select one predefined rescue rule on tuning, freeze, then score internal once."""

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
from filament_seg.adaptive_instance_filter import AdaptiveFilterConfig, extract_adaptive
from filament_seg.benchmark import coco_annotation_instances, sha256_file
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.official_metric import organizer_pq_image

BENCHMARK = ROOT / "reports/model_benchmark_20261104T094701Z.json"
SOURCE = ROOT / "src/filament_seg/adaptive_instance_filter.py"
PREVIOUS_PROVENANCE = ROOT / "outputs/submission_provisional_v2_confidence_0p97_20261109T044350Z.provenance.json"
CONFIGS = (
    AdaptiveFilterConfig("confidence_0p97"),
    AdaptiveFilterConfig("rescue_area_elong", rescue_enabled=True,
                         rescue_min_area=1100, rescue_min_elongation=4.5),
    AdaptiveFilterConfig("rescue_area_q25", rescue_enabled=True,
                         rescue_min_area=1100, rescue_min_q25=.945),
    AdaptiveFilterConfig("rescue_area_elong_q25", rescue_enabled=True,
                         rescue_min_area=1000, rescue_min_elongation=4.5, rescue_min_q25=.94),
    AdaptiveFilterConfig("rescue_area_compact_q25", rescue_enabled=True,
                         rescue_min_area=1000, rescue_max_compactness=.27, rescue_min_q25=.94),
)


def context():
    benchmark = json.loads(BENCHMARK.read_text())
    original = next(row for row in benchmark["results"] if row["model"]["experiment_id"] == "unet_resnet34_v2")
    cache = Path(original["cache"])
    manifest = json.loads((cache / "cache_manifest.json").read_text())
    provenance = json.loads(PREVIOUS_PROVENANCE.read_text())
    if manifest["checkpoint_sha256"] != provenance["checkpoint_sha256"] or provenance["component_filter"]["min_mean_probability"] != .97:
        raise ValueError("Cached predictions do not match the official 0.33 baseline")
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images", base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    return benchmark, original, cache, dataset


def score(indices: list[int], config: AdaptiveFilterConfig, cache: Path, dataset) -> dict:
    rows = []
    postprocess_seconds = 0.0
    for position, index in enumerate(indices, 1):
        probability = np.load(cache / f"{index:04d}.npy", mmap_mode="r")
        started = time.perf_counter()
        predictions = extract_adaptive(probability, config)
        postprocess_seconds += time.perf_counter() - started
        filename = dataset.filenames[index]
        scores = [organizer_pq_image(coco_annotation_instances(dataset, index, set_index), predictions)
                  for set_index in range(len(dataset.images_by_filename[filename]))]
        rows.append({"index": index, "filename": filename,
                     "tp": sum(item["tp"] for item in scores),
                     "fp": sum(item["fp"] for item in scores),
                     "fn": sum(item["fn"] for item in scores),
                     "sum_iou": sum(item["sum_iou"] for item in scores),
                     "predictions": len(predictions)})
        if position % 20 == 0 or position == len(indices):
            print(f"{config.name}: {position}/{len(indices)}", flush=True)
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    iou = sum(row["sum_iou"] for row in rows)
    denominator = tp + .5 * (fp + fn)
    return {"config": vars(config), "summary": {
        "images": len(indices), "global_pq": iou / denominator if denominator else 0.0,
        "sq": iou / tp if tp else 0.0, "rq": tp / denominator if denominator else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "postprocess_seconds_per_image": postprocess_seconds / len(indices)},
        "per_image": rows}


def tune() -> Path:
    benchmark, original, cache, dataset = context()
    indices = benchmark["partition"]["tuning_indices"]
    results = [score(indices, config, cache, dataset) for config in CONFIGS]
    best = max(results, key=lambda item: (item["summary"]["global_pq"], item["summary"]["sq"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "reports" / f"adaptive_filter_tuning_frozen_{stamp}.json"
    artifact = {"phase": "frozen_after_tuning", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "selection_metric": "tuning Global PQ, SQ tie break",
                "benchmark_sha256": sha256_file(BENCHMARK),
                "official_0p33_submission_provenance_sha256": sha256_file(PREVIOUS_PROVENANCE),
                "checkpoint_sha256": original["model"]["checkpoint_sha256"],
                "cache_manifest_sha256": sha256_file(cache / "cache_manifest.json"),
                "implementation_sha256": sha256_file(SOURCE),
                "validation_fingerprint": benchmark["validation_fingerprint"],
                "tuning_indices": indices, "internal_indices": benchmark["partition"]["internal_check_indices"],
                "candidate_results": results, "selected_candidate": best["config"]["name"]}
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    print(f"FROZEN {best['config']['name']} PQ={best['summary']['global_pq']:.6f} artifact={destination}", flush=True)
    return destination


def internal(frozen_path: Path) -> Path:
    frozen = json.loads(frozen_path.read_text())
    benchmark, original, cache, dataset = context()
    if frozen["phase"] != "frozen_after_tuning" or frozen["benchmark_sha256"] != sha256_file(BENCHMARK):
        raise ValueError("Benchmark changed after freeze")
    if frozen["official_0p33_submission_provenance_sha256"] != sha256_file(PREVIOUS_PROVENANCE):
        raise ValueError("Official baseline provenance changed")
    if frozen["checkpoint_sha256"] != original["model"]["checkpoint_sha256"] or frozen["cache_manifest_sha256"] != sha256_file(cache / "cache_manifest.json"):
        raise ValueError("Checkpoint or cache changed after freeze")
    if frozen["implementation_sha256"] != sha256_file(SOURCE):
        raise ValueError("Candidate implementation changed after freeze")
    if frozen["validation_fingerprint"] != benchmark["validation_fingerprint"] or frozen["internal_indices"] != benchmark["partition"]["internal_check_indices"]:
        raise ValueError("Internal partition changed")
    if [row["config"] for row in frozen["candidate_results"]] != [vars(config) for config in CONFIGS]:
        raise ValueError("Candidate grid changed")
    indices = frozen["internal_indices"]
    selected_config = next(config for config in CONFIGS if config.name == frozen["selected_candidate"])
    internal_configs = [CONFIGS[0]] if selected_config == CONFIGS[0] else [CONFIGS[0], selected_config]
    results = [score(indices, config, cache, dataset) for config in internal_configs]
    selected = next(row for row in results if row["config"]["name"] == frozen["selected_candidate"])
    baseline = results[0]
    delta = selected["summary"]["global_pq"] - baseline["summary"]["global_pq"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "reports" / f"adaptive_filter_internal_once_{stamp}.json"
    artifact = {"phase": "internal_scored_once", "frozen_artifact": str(frozen_path.resolve()),
                "frozen_sha256": sha256_file(frozen_path), "candidate_results": results,
                "selected_candidate": frozen["selected_candidate"], "selected_delta_global_pq": delta,
                "cached_inference_seconds_per_image": original["inference"]["seconds_per_image"],
                "cached_peak_inference_vram_mib": original["inference"]["peak_gpu_memory_mib"],
                "limitation": "Original v2 was historically evaluated on these images; hidden scorer parity unverified."}
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    print(f"INTERNAL selected={frozen['selected_candidate']} delta={delta:.6f} report={destination}", flush=True)
    return destination


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
