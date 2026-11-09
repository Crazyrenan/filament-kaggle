"""Benchmark completed models and select a provisional internal-check winner."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.benchmark import (
    cache_probability_maps, code_revision, load_candidate, make_partition,
    score_probability_maps, sha256_file,
)
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.experiment_tracking import validation_fingerprint

THRESHOLDS = (0.60, 0.70, 0.75, 0.80)
MIN_AREAS = (300, 500, 750)
FIXED = {"threshold": 0.70, "min_area": 500}


def candidates() -> list[Path]:
    result = [ROOT / "checkpoints/unet_resnet34_v2/best_model.pth"]
    for pattern in ("unet_resnet34_v2_retrain_*", "segformer_*"):
        for directory in sorted((ROOT / "checkpoints").glob(pattern)):
            best = directory / "best_model.pth"
            metrics_file = directory / "metrics.json"
            if not best.is_file() or not metrics_file.is_file():
                continue
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            if metrics.get("status") in {"completed", "early_stopped"}:
                result.append(best)
    return result


def timing_summary(folder: Path, indices: list[int]) -> dict:
    samples = []
    for index in indices:
        path = folder / f"{index:04d}.timing.json"
        if path.is_file():
            samples.append(json.loads(path.read_text(encoding="utf-8")))
    return {"measured_images": len(samples),
            "seconds_per_image": float(np.mean([x["seconds"] for x in samples])) if samples else None,
            "peak_gpu_memory_mib": max((x["peak_gpu_memory_mib"] for x in samples), default=None)}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark run")
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(
        base / "train_images", base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    assert len(dataset) == 707
    indices = np.random.default_rng(42).permutation(707)[565:].tolist()
    partition = make_partition(dataset, indices, ROOT / "reports/validation_partition_v1.json")
    tuning = partition["tuning_indices"]
    internal = partition["internal_check_indices"]
    assert set(tuning).isdisjoint(internal) and len(tuning) + len(internal) == 142
    prepared = []
    for checkpoint in candidates():
        model, metadata = load_candidate(checkpoint, device="cuda")
        batch_size = 4
        config_path = checkpoint.parent / "config.yaml"
        if config_path.is_file():
            import yaml
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            batch_size = min(4, int(config.get("training", {}).get("batch_size", 4)))
        folder, cache_manifest = cache_probability_maps(
            model, metadata, dataset, indices, ROOT / "outputs/probability_cache",
            device="cuda", tile_size=512, overlap=128, batch_size=batch_size,
        )
        del model
        torch.cuda.empty_cache()
        prepared.append((metadata, folder, cache_manifest, timing_summary(folder, indices)))
    results = []
    for metadata, folder, cache_manifest, timing in prepared:
        print(f"Scoring {metadata['experiment_id']} fixed protocol", flush=True)
        fixed_full = score_probability_maps(dataset, indices, folder, **FIXED)
        fixed_internal = score_probability_maps(dataset, internal, folder, **FIXED)
        trials = []
        for threshold in THRESHOLDS:
            for area in MIN_AREAS:
                scored = score_probability_maps(dataset, tuning, folder,
                                                threshold=threshold, min_area=area)
                trials.append(scored["summary"])
                print(f"{metadata['experiment_id']} tuning threshold={threshold} area={area} "
                      f"PQ={scored['summary']['global_pq']:.6f}", flush=True)
        selected = max(trials, key=lambda row: (row["global_pq"], row["sq"],
                                                 -row["threshold"], -row["min_area"]))
        tuned_internal = score_probability_maps(
            dataset, internal, folder, threshold=selected["threshold"],
            min_area=selected["min_area"],
        )
        results.append({
            "model": metadata, "cache": str(folder), "cache_manifest": cache_manifest,
            "inference": timing, "fixed_full_142": fixed_full,
            "fixed_internal": fixed_internal, "tuning_trials": trials,
            "selected_postprocessing": {"threshold": selected["threshold"],
                                        "min_area": selected["min_area"],
                                        "instance_method": "connected_components"},
            "tuning_best": selected, "tuned_internal": tuned_internal,
        })
        print(f"{metadata['experiment_id']} internal PQ={tuned_internal['summary']['global_pq']:.6f}", flush=True)
    leaderboard = sorted(results, key=lambda row: (
        row["tuned_internal"]["summary"]["global_pq"],
        row["tuned_internal"]["summary"]["sq"],
        -(row["inference"]["seconds_per_image"] or float("inf")),
    ), reverse=True)
    winner = leaderboard[0]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = ROOT / "reports" / f"model_benchmark_{stamp}.json"
    manifest_path = ROOT / "reports" / f"selected_model_provisional_{stamp}.json"
    report = {
        "status": "provisional", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_fingerprint": validation_fingerprint(dataset, indices),
        "partition": partition, "fixed_protocol": FIXED,
        "historical_original_v2_references_unverified": {
            "diagnostic_pq_threshold_0_75_area_300": 0.386869,
            "diagnostic_pq_threshold_0_70_area_500": 0.400488,
            "note": "Previously reported with a different local matcher and annotation-set protocol; not directly comparable to public-notebook PQ.",
        },
        "tuning_grid": {"thresholds": THRESHOLDS, "min_areas": MIN_AREAS},
        "evaluator": {"implementation": "filament_seg.official_metric.organizer_pq_image",
                      "iou_threshold": 0.5, "inclusive": False,
                      "matching": "all qualifying GT/prediction pairs, no one-to-one assignment",
                      "annotation_sets": "all recorded annotator-image sets per physical image",
                      "ground_truth_decoding": "pycocotools.frPyObjects plus merge on source COCO polygons",
                      "public_notebook_url": "https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook",
                      "public_notebook_version": 6,
                      "public_notebook_sha256": sha256_file(ROOT / "reports/reference/organizer_self_evaluation_v6.ipynb"),
                      "hidden_kaggle_scorer_parity": "unverified",
                      "limitations": "Pair counting is parity-tested against the public self-evaluation notebook on synthetic cases. The hidden scorer's GT RLE generation remains unverified. Original v2 previously used all 142 validation images; internal check is not independent."},
        "code_revision": code_revision(ROOT), "results": results,
        "ranking": [row["model"]["experiment_id"] for row in leaderboard],
    }
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    selected = {
        "status": "provisional", "experiment_id": winner["model"]["experiment_id"],
        "architecture": winner["model"]["architecture"],
        "checkpoint": winner["model"]["checkpoint"],
        "checkpoint_sha256": winner["model"]["checkpoint_sha256"],
        "model_config": winner["model"]["model_config"],
        "preprocessing": {"input": "grayscale", "normalization": "divide_by_255",
                          "input_adapter": winner["model"]["model_config"].get("input_adapter")},
        "tiling": {"tile_size": 512, "overlap": 128,
                   "batch_size": next(item[2]["batch_size"] for item in prepared
                                      if item[0]["experiment_id"] == winner["model"]["experiment_id"])},
        "postprocessing": winner["selected_postprocessing"],
        "evaluation_protocol": "public notebook v6 PQ across all annotation sets; tuning-only optimization; existing validation internal check",
        "internal_check_global_pq": winner["tuned_internal"]["summary"]["global_pq"],
        "internal_check_sq": winner["tuned_internal"]["summary"]["sq"],
        "internal_check_rq": winner["tuned_internal"]["summary"]["rq"],
        "evaluator": report["evaluator"], "validation_fingerprint": report["validation_fingerprint"],
        "code_revision": report["code_revision"],
        "benchmark_report": str(report_path), "benchmark_report_sha256": sha256_file(report_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)
    print(f"PROVISIONAL winner={selected['experiment_id']} "
          f"PQ={selected['internal_check_global_pq']:.6f} manifest={manifest_path} report={report_path}", flush=True)


if __name__ == "__main__":
    main()
