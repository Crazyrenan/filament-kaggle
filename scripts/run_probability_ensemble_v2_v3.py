"""Controlled, tuning-only v2/v3 probability ensemble with one internal check."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from filament_seg.benchmark import coco_annotation_instances, load_candidate, sha256_file
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.inference import predict_tiled
from filament_seg.official_metric import organizer_pq_image
from filament_seg.probability_ensemble import (
    EnsembleConfig, blend, candidate_components, encode_masks, score_encoded_sets,
    select_components, summarize,
)
from filament_seg.submission import encode_coco_counts, validate_submission_rows
from generate_component_filter_submission import bootstrap_interval

EXPERIMENT = "ensemble_v2_v3_01"
V2 = ROOT / "checkpoints/unet_resnet34_v2/best_model.pth"
V3 = ROOT / "checkpoints/unet_resnet34_v3_hard_negative_01/best_model.pth"
V2_CACHE = ROOT / "outputs/probability_cache/unet_resnet34_v2/f9422022d65c0de7"
V3_CACHE = ROOT / "outputs/probability_cache/unet_resnet34_v3_hard_negative_01/302aa58783fce9c4"
PARTITION = ROOT / "reports/validation_partition_v1.json"
BASELINE_INTERNAL = ROOT / "reports/component_filter_internal_20261109T044117Z.json"
OLD_SUBMISSION = ROOT / "outputs/submission_provisional_v2_confidence_0p97_20261109T044350Z.csv"
PREFLIGHT = ROOT / f"reports/{EXPERIMENT}_preflight.json"
GRID = tuple(EnsembleConfig(weight, threshold, area, confidence)
             for weight in (.25, .50, .75)
             for threshold in (.75, .80)
             for area in (300, 500)
             for confidence in (.96, .97))


def dataset_and_partition():
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images",
        base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    partition = json.loads(PARTITION.read_text(encoding="utf-8"))
    tuning, internal = partition["tuning_indices"], partition["internal_check_indices"]
    if len(dataset) != 707 or len(tuning) != 71 or len(internal) != 71 or set(tuning) & set(internal):
        raise ValueError("Physical validation partition is invalid")
    return dataset, partition


def audit(*, scan_maps: bool = False) -> dict:
    dataset, partition = dataset_and_partition()
    historical_rows = json.loads((ROOT / "reports/historical_checkpoint_hashes.json").read_text(encoding="utf-8"))
    historical = {str(Path(row["path"]).resolve()): row for row in historical_rows}
    for path in (V2, V3):
        if not path.is_file():
            raise FileNotFoundError(path)
    for row in historical_rows:
        path = Path(row["path"])
        if sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Historical checkpoint changed: {path}")
    if sha256_file(V2) != historical[str(V2.resolve())]["sha256"]:
        raise ValueError("Original v2 hash changed")
    v3_registry = json.loads((ROOT / "reports/experiment_registry.json").read_text(encoding="utf-8"))["experiments"]["unet_resnet34_v3_hard_negative_01"]
    if sha256_file(V3) != v3_registry["best_checkpoint_sha256"]:
        raise ValueError("V3 checkpoint differs from experiment registry")
    manifests = [json.loads((folder / "cache_manifest.json").read_text(encoding="utf-8"))
                 for folder in (V2_CACHE, V3_CACHE)]
    indices = set(partition["tuning_indices"] + partition["internal_check_indices"])
    for manifest, checkpoint, folder in zip(manifests, (V2, V3), (V2_CACHE, V3_CACHE)):
        if manifest["checkpoint_sha256"] != sha256_file(checkpoint) or set(manifest["indices"]) != indices:
            raise ValueError("Cache checkpoint or image index mismatch")
        if len(manifest["indices"]) != len(indices) or any(
            dataset.filenames[index] != name for index, name in zip(manifest["indices"], manifest["filenames"])):
            raise ValueError("Cached image filename alignment mismatch")
        if (manifest["tile_size"], manifest["overlap"], manifest["batch_size"], manifest["shape"], manifest["dtype"]) != (
            512, 128, 4, [2048, 2048], "float32"):
            raise ValueError("Inference configuration mismatch")
        for index in indices:
            map_path = folder / f"{index:04d}.npy"
            timing_path = folder / f"{index:04d}.timing.json"
            if not map_path.is_file() or not timing_path.is_file():
                raise FileNotFoundError(f"Cache map or timing missing for {index}: {folder}")
            probability = np.load(map_path, mmap_mode="r")
            if probability.shape != (2048, 2048) or probability.dtype != np.float32:
                raise ValueError(f"Cache map shape/dtype invalid: {map_path}")
            if scan_maps and (not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1):
                raise ValueError(f"Cache map range/nonfinite invalid: {map_path}")
    if not OLD_SUBMISSION.is_file():
        raise FileNotFoundError(OLD_SUBMISSION)
    return {"experiment_id": EXPERIMENT, "status": "preflight_passed", "checkpoint_sha256": {
        "v2": sha256_file(V2), "v3": sha256_file(V3)}, "historical_checkpoints_verified": len(historical_rows),
        "partition_sha256": sha256_file(PARTITION),
        "tuning_indices": partition["tuning_indices"], "internal_indices": partition["internal_check_indices"],
        "cache_manifest_sha256": {"v2": sha256_file(V2_CACHE / "cache_manifest.json"),
                                  "v3": sha256_file(V3_CACHE / "cache_manifest.json")},
        "cache_index_sets_identical": True, "cache_filenames_aligned": True,
        "cache_maps_verified": 284 if scan_maps else None,
        "cache_order_fingerprints_differ_only_due_to_index_order": manifests[0]["validation_fingerprint"] != manifests[1]["validation_fingerprint"],
        "preprocessing": "same source grayscale uint8 / 255 once, native 2048x2048, 512/128 Hann tiling, batch 4",
        "original_submission_sha256": sha256_file(OLD_SUBMISSION),
        "config_sha256": {"v3": sha256_file(ROOT / "configs/unet_resnet34_v3_hard_negative_01.yaml")},
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in [
            Path(__file__), ROOT / "src/filament_seg/probability_ensemble.py",
            ROOT / "src/filament_seg/inference.py", ROOT / "src/filament_seg/official_metric.py",
            ROOT / "src/filament_seg/benchmark.py", ROOT / "src/filament_seg/submission.py",
            ROOT / "scripts/generate_component_filter_submission.py"]}}


def verify_preflight() -> dict:
    saved = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    current = audit(scan_maps=False)
    if saved != current or saved["cache_maps_verified"] != 284:
        # The full scan marker differs from the lightweight audit by design.
        current["cache_maps_verified"] = saved.get("cache_maps_verified")
        if saved != current or saved["cache_maps_verified"] != 284:
            raise ValueError("Preflight provenance changed")
    return saved


def profile_two_models(dataset: SolarFilamentDataset, indices: list[int]) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for actual two-model profiling")
    model2, _ = load_candidate(V2, device="cuda")
    model3, _ = load_candidate(V3, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    max_cache_difference = 0.0
    for index in indices[:5]:
        raw = cv2.imread(str(dataset.image_dir / dataset.filenames[index]), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != (2048, 2048):
            raise ValueError("Invalid profiling source image")
        image = raw.astype(np.float32) / 255.0
        torch.cuda.synchronize()
        start = time.perf_counter()
        prob2 = predict_tiled(model2, image, device="cuda", tile_size=512, overlap=128, batch_size=4)
        prob3 = predict_tiled(model3, image, device="cuda", tile_size=512, overlap=128, batch_size=4)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
        max_cache_difference = max(max_cache_difference,
            float(np.max(np.abs(prob2 - np.load(V2_CACHE / f"{index:04d}.npy", mmap_mode="r")))),
            float(np.max(np.abs(prob3 - np.load(V3_CACHE / f"{index:04d}.npy", mmap_mode="r")))))
    if max_cache_difference > 1e-5:
        raise ValueError(f"Live predictions do not align with cached probabilities: {max_cache_difference}")
    result = {"sample_images": len(elapsed), "combined_inference_seconds_per_image": float(np.mean(elapsed)),
              "inference_seconds_each": elapsed, "peak_allocated_vram_mib": torch.cuda.max_memory_allocated()/2**20,
              "peak_reserved_vram_mib": torch.cuda.max_memory_reserved()/2**20,
              "max_abs_live_vs_cache_difference": max_cache_difference}
    del model2, model3
    torch.cuda.empty_cache()
    return result


def probability(index: int, weight: float) -> np.ndarray:
    return blend(np.load(V2_CACHE / f"{index:04d}.npy", mmap_mode="r"),
                 np.load(V3_CACHE / f"{index:04d}.npy", mmap_mode="r"), weight)


def gt_encoded(dataset: SolarFilamentDataset, index: int) -> list[list[dict]]:
    filename = dataset.filenames[index]
    return [encode_masks(coco_annotation_instances(dataset, index, alternative))
            for alternative in range(len(dataset.images_by_filename[filename]))]


def direct_score(dataset: SolarFilamentDataset, indices: list[int], config: EnsembleConfig) -> dict:
    """Score one configuration through organizer_pq_image, not the grid shortcut."""
    rows = []
    processing_seconds = 0.0
    for position, index in enumerate(indices, 1):
        started = time.perf_counter()
        ensemble = probability(index, config.v2_weight)
        masks = select_components(candidate_components(ensemble, config.threshold, config.min_area), config)
        processing_seconds += time.perf_counter() - started
        filename = dataset.filenames[index]
        scores = [organizer_pq_image(coco_annotation_instances(dataset, index, alternative), masks)
                  for alternative in range(len(dataset.images_by_filename[filename]))]
        rows.append({"index": index, "filename": filename,
                     "tp": sum(score["tp"] for score in scores),
                     "fp": sum(score["fp"] for score in scores),
                     "fn": sum(score["fn"] for score in scores),
                     "sum_iou": sum(score["sum_iou"] for score in scores)})
        if position % 20 == 0 or position == len(indices):
            print(f"direct {position}/{len(indices)}", flush=True)
    summary = summarize(rows)
    summary["cpu_blend_and_postprocess_seconds_per_image"] = processing_seconds / len(indices)
    return {"config": asdict(config), "summary": summary, "per_image": rows}


def tune() -> Path:
    preflight = verify_preflight()
    dataset, partition = dataset_and_partition()
    indices = partition["tuning_indices"]
    results = {config: [] for config in GRID}
    for position, index in enumerate(indices, 1):
        gt_sets = gt_encoded(dataset, index)
        filename = dataset.filenames[index]
        for weight in (.25, .50, .75):
            ensemble = probability(index, weight)
            for threshold in (.75, .80):
                candidates = candidate_components(ensemble, threshold, 300)
                encoded = [(area, confidence, encode_masks([mask])[0]) for area, confidence, mask in candidates]
                for area in (300, 500):
                    for confidence in (.96, .97):
                        config = EnsembleConfig(weight, threshold, area, confidence)
                        selected = [rle for component_area, mean, rle in encoded
                                    if component_area >= area and mean >= confidence]
                        row = score_encoded_sets(gt_sets, selected)
                        results[config].append({"index": index, "filename": filename,
                                                **{key: row[key] for key in ("tp", "fp", "fn", "sum_iou")}})
        if position % 10 == 0 or position == len(indices):
            print(f"Tuning grid {position}/{len(indices)}", flush=True)
    summaries = [{"config": asdict(config), "summary": summarize(rows)} for config, rows in results.items()]
    ranked = sorted(summaries, key=lambda row: (
        -row["summary"]["global_pq"], -row["summary"]["sq"], row["summary"]["fp"],
        -row["config"]["v2_weight"], -row["config"]["mean_confidence"],
        -row["config"]["min_area"], -row["config"]["threshold"]))
    selected_config = EnsembleConfig(**ranked[0]["config"])
    direct = direct_score(dataset, indices, selected_config)
    for key in ("global_pq", "sq", "rq", "tp", "fp", "fn"):
        if not np.isclose(direct["summary"][key], ranked[0]["summary"][key], atol=1e-8):
            raise ValueError(f"Grid scorer disagrees with organizer implementation: {key}")
    profile = profile_two_models(dataset, indices)
    frozen = {"phase": "frozen_after_tuning", "status": "provisional", "experiment_id": EXPERIMENT,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "preflight_path": str(PREFLIGHT.resolve()), "preflight_sha256": sha256_file(PREFLIGHT),
              "checkpoint_sha256": preflight["checkpoint_sha256"],
              "source_sha256": preflight["source_sha256"],
              "grid": [asdict(config) for config in GRID], "grid_results": summaries,
              "selection_rule": "tuning Global PQ descending, SQ descending, FP ascending, then v2 weight/confidence/area/threshold descending",
              "selected_config": asdict(selected_config), "selected_tuning_direct": direct,
              "tuning_indices": indices, "internal_indices": partition["internal_check_indices"],
              "two_model_cuda_profile": profile,
              "limitation": "The original v2 was historically evaluated on the internal images; this is provisional and hidden evaluator parity is unverified."}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "reports" / f"{EXPERIMENT}_frozen_{timestamp}.json"
    with destination.open("x", encoding="utf-8") as file:
        json.dump(frozen, file, indent=2)
    print("FROZEN", destination, "selected", asdict(selected_config),
          "tuning_Global_PQ", direct["summary"]["global_pq"], flush=True)
    return destination


def internal(frozen_path: Path) -> Path:
    if list((ROOT / "reports").glob(f"{EXPERIMENT}_internal_*.json")):
        raise FileExistsError("Internal-check score already exists; only one evaluation is allowed")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    preflight = verify_preflight()
    if frozen["phase"] != "frozen_after_tuning" or frozen["preflight_sha256"] != sha256_file(PREFLIGHT):
        raise ValueError("Frozen artifact or preflight changed")
    if frozen["source_sha256"] != preflight["source_sha256"] or frozen["checkpoint_sha256"] != preflight["checkpoint_sha256"]:
        raise ValueError("Source code or checkpoints changed after tuning")
    dataset, partition = dataset_and_partition()
    if frozen["internal_indices"] != partition["internal_check_indices"]:
        raise ValueError("Internal partition changed")
    config = EnsembleConfig(**frozen["selected_config"])
    candidate = direct_score(dataset, frozen["internal_indices"], config)
    historical = json.loads(BASELINE_INTERNAL.read_text(encoding="utf-8"))
    baseline = next(row for row in historical["candidate_results"]
                    if row["config"]["name"] == "confidence_0p97")
    if not np.isclose(baseline["summary"]["global_pq"], .4099034711305698) or \
            [row["index"] for row in baseline["per_image"]] != frozen["internal_indices"]:
        raise ValueError("Historical original-v2 internal baseline mismatch")
    interval = bootstrap_interval(baseline["per_image"], candidate["per_image"])
    delta = candidate["summary"]["global_pq"] - baseline["summary"]["global_pq"]
    supported = (delta > .005 and interval[0] > 0 and
                 candidate["summary"]["tp"] >= .9 * baseline["summary"]["tp"])
    report = {"phase": "internal_scored_once", "status": "provisional", "experiment_id": EXPERIMENT,
              "frozen_path": str(frozen_path.resolve()), "frozen_sha256": sha256_file(frozen_path),
              "baseline_source": str(BASELINE_INTERNAL), "baseline_source_sha256": sha256_file(BASELINE_INTERNAL),
              "baseline_v2": baseline, "ensemble": candidate, "delta_global_pq": delta,
              "paired_capture_date_bootstrap_95pct_delta_interval": interval,
              "bootstrap_seed": 42, "bootstrap_replicates": 5000,
              "two_model_cuda_profile": frozen["two_model_cuda_profile"],
              "decision": "submission_supported" if supported else "retain_official_v2_0p33",
              "decision_rule": "Global PQ gain >0.005, paired bootstrap lower bound >0, TP >=90% of baseline",
              "limitation": "Historically exposed internal check; no independent verification or hidden evaluator parity."}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "reports" / f"{EXPERIMENT}_internal_{timestamp}.json"
    with destination.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print("INTERNAL", destination, "baseline", baseline["summary"],
          "ensemble", candidate["summary"], "CI", interval, "decision", report["decision"], flush=True)
    return destination


def submit(internal_path: Path) -> Path:
    report = json.loads(internal_path.read_text(encoding="utf-8"))
    if report["decision"] != "submission_supported":
        raise ValueError("No defensible local improvement; submission generation refused")
    frozen_path = Path(report["frozen_path"])
    if sha256_file(frozen_path) != report["frozen_sha256"]:
        raise ValueError("Frozen selection changed")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    preflight = verify_preflight()
    if frozen["source_sha256"] != preflight["source_sha256"] or frozen["checkpoint_sha256"] != preflight["checkpoint_sha256"]:
        raise ValueError("Source or checkpoint changed")
    old_hash = sha256_file(OLD_SUBMISSION)
    model2, _ = load_candidate(V2, device="cuda")
    model3, _ = load_candidate(V3, device="cuda")
    config = EnsembleConfig(**frozen["selected_config"])
    test_dir = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    images = sorted(test_dir.glob("*.jpeg"))
    if len(images) != 180 or len({path.stem for path in images}) != 180:
        raise ValueError("Expected exactly 180 unique test image identifiers")
    rows = []
    zero_count = 0
    for position, path in enumerate(images, 1):
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != (2048, 2048):
            raise ValueError(f"Invalid test image: {path}")
        image = raw.astype(np.float32) / 255.0
        probability_map = blend(
            predict_tiled(model2, image, device="cuda", tile_size=512, overlap=128, batch_size=4),
            predict_tiled(model3, image, device="cuda", tile_size=512, overlap=128, batch_size=4),
            config.v2_weight)
        masks = select_components(candidate_components(probability_map, config.threshold, config.min_area), config)
        zero_count += not masks
        for number, mask in enumerate(masks, 1):
            rows.append({"filament_id": f"{path.stem}_{number}", "segmentation_rle": encode_coco_counts(mask)})
        if position % 20 == 0:
            print(f"Test inference {position}/180", flush=True)
    if not rows or validate_submission_rows(rows, images) != len(rows):
        raise ValueError("Submission rows fail compressed COCO RLE validation")
    if sha256_file(OLD_SUBMISSION) != old_hash or audit()["checkpoint_sha256"] != preflight["checkpoint_sha256"]:
        raise ValueError("Historical artifact changed during test inference")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = ROOT / "outputs" / f"submission_provisional_{EXPERIMENT}_{timestamp}.csv"
    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    frame.to_csv(csv_path, mode="x", index=False)
    saved = pd.read_csv(csv_path, dtype=str)
    if list(saved.columns) != ["filament_id", "segmentation_rle"] or saved.shape != frame.shape:
        raise ValueError("Saved CSV schema or row count mismatch")
    provenance = {"status": "provisional", "uploaded_to_kaggle": False, "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path), "rows": len(rows), "test_images_processed": 180,
        "zero_prediction_images": zero_count, "postprocessing": asdict(config),
        "checkpoint_sha256": preflight["checkpoint_sha256"], "source_sha256": preflight["source_sha256"],
        "frozen_path": str(frozen_path), "frozen_sha256": sha256_file(frozen_path),
        "internal_path": str(internal_path), "internal_sha256": sha256_file(internal_path),
        "old_official_submission_sha256": old_hash, "validation": "all rows compressed COCO RLE decoded at 2048x2048; unique IDs and CSV schema checked"}
    manifest = csv_path.with_suffix(".provenance.json")
    with manifest.open("x", encoding="utf-8") as file:
        json.dump(provenance, file, indent=2)
    print("SUBMISSION_COMPLETE", csv_path, manifest, flush=True)
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "tune", "internal", "submit"))
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--internal", type=Path)
    args = parser.parse_args()
    if args.phase == "preflight":
        record = audit(scan_maps=True)
        with PREFLIGHT.open("x", encoding="utf-8") as file:
            json.dump(record, file, indent=2)
        print("PREFLIGHT_PASS", PREFLIGHT, flush=True)
    elif args.phase == "tune":
        tune()
    elif args.phase == "internal":
        if args.frozen is None:
            parser.error("--frozen is required")
        internal(args.frozen)
    else:
        if args.internal is None:
            parser.error("--internal is required")
        submit(args.internal)
