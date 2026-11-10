"""Controlled v2 test-time augmentation, tuning freeze, and one internal check."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
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
from filament_seg.probability_ensemble import candidate_components, encode_masks, score_encoded_sets, summarize
from filament_seg.submission import encode_coco_counts, validate_submission_rows
from filament_seg.tta import CONFIGURATIONS, DIHEDRAL8, average_aligned
from generate_component_filter_submission import bootstrap_interval

EXPERIMENT = "tta_unet_resnet34_v2_01"
V2 = ROOT / "checkpoints/unet_resnet34_v2/best_model.pth"
BASE_CACHE = ROOT / "outputs/probability_cache/unet_resnet34_v2/f9422022d65c0de7"
CACHE = ROOT / "outputs/tta_cache" / EXPERIMENT
PARTITION = ROOT / "reports/validation_partition_v1.json"
PREFLIGHT = ROOT / "reports" / f"{EXPERIMENT}_preflight.json"
BASELINE_INTERNAL = ROOT / "reports/component_filter_internal_20261109T044117Z.json"
OLD_CSV = ROOT / "outputs/submission_provisional_v2_confidence_0p97_20261109T044350Z.csv"


@dataclass(frozen=True)
class TtaConfig:
    tta: str
    threshold: float
    min_area: int
    mean_confidence: float


GRID = tuple(TtaConfig(name, threshold, area, confidence)
             for name in ("flip4", "rot4", "dihedral8")
             for threshold in (.75, .80)
             for area in (300, 500)
             for confidence in (.96, .97))


def context():
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images",
                                   base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    partition = json.loads(PARTITION.read_text(encoding="utf-8"))
    tune, internal = partition["tuning_indices"], partition["internal_check_indices"]
    if len(dataset) != 707 or len(tune) != 71 or len(internal) != 71 or set(tune) & set(internal):
        raise ValueError("Validation partition is invalid")
    return dataset, partition


def preflight() -> dict:
    dataset, partition = context()
    historical = json.loads((ROOT / "reports/historical_checkpoint_hashes.json").read_text(encoding="utf-8"))
    for row in historical:
        path = Path(row["path"])
        if sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Historical checkpoint changed: {path}")
    expected = next(row["sha256"] for row in historical if Path(row["path"]).resolve() == V2.resolve())
    if sha256_file(V2) != expected:
        raise ValueError("Original v2 checkpoint changed")
    manifest = json.loads((BASE_CACHE / "cache_manifest.json").read_text(encoding="utf-8"))
    indices = set(partition["tuning_indices"] + partition["internal_check_indices"])
    if (manifest["checkpoint_sha256"] != expected or set(manifest["indices"]) != indices or
        manifest["tile_size"] != 512 or manifest["overlap"] != 128 or manifest["batch_size"] != 4 or
        manifest["shape"] != [2048, 2048] or manifest["dtype"] != "float32"):
        raise ValueError("Baseline probability cache provenance differs")
    if any(dataset.filenames[index] != filename for index, filename in zip(manifest["indices"], manifest["filenames"])):
        raise ValueError("Baseline cached image IDs are misaligned")
    for index in indices:
        probability = np.load(BASE_CACHE / f"{index:04d}.npy", mmap_mode="r")
        if probability.shape != (2048, 2048) or probability.dtype != np.float32 or \
                not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
            raise ValueError(f"Invalid baseline probability map: {index}")
    return {"experiment_id": EXPERIMENT, "checkpoint_sha256": expected,
        "historical_checkpoints_verified": len(historical), "partition_sha256": sha256_file(PARTITION),
        "base_cache_manifest_sha256": sha256_file(BASE_CACHE / "cache_manifest.json"),
        "base_cache_maps_verified": len(indices), "tuning_indices": partition["tuning_indices"],
        "internal_indices": partition["internal_check_indices"], "old_csv_sha256": sha256_file(OLD_CSV),
        "preprocessing": "grayscale uint8 divided by 255 once; native 2048x2048; 512/128 Hann tiles, batch 4",
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in [
            Path(__file__), ROOT / "src/filament_seg/tta.py", ROOT / "src/filament_seg/inference.py",
            ROOT / "src/filament_seg/probability_ensemble.py", ROOT / "src/filament_seg/official_metric.py",
            ROOT / "src/filament_seg/benchmark.py", ROOT / "src/filament_seg/submission.py",
            ROOT / "scripts/generate_component_filter_submission.py"]},
        "deadline_utc": "2026-11-15T06:00:00Z", "deadline_wib": "2026-11-15T13:00:00+07:00",
        "deadline_source": "Kaggle competitions list, verified 2026-11-10"}


def verify_preflight() -> dict:
    saved = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if saved != preflight():
        raise ValueError("Preflight source, checkpoint, cache, or partition changed")
    return saved


def image_array(dataset: SolarFilamentDataset, index: int) -> np.ndarray:
    raw = cv2.imread(str(dataset.image_dir / dataset.filenames[index]), cv2.IMREAD_GRAYSCALE)
    if raw is None or raw.shape != (2048, 2048):
        raise ValueError(f"Invalid source image at index {index}")
    return raw.astype(np.float32) / 255.0


def tta_probabilities(model, image: np.ndarray, names: set[str], *, identity_cache: np.ndarray | None = None) -> dict[str, np.ndarray]:
    aligned = {}
    for transform in DIHEDRAL8:
        if transform.name not in names:
            continue
        if transform.name == "identity" and identity_cache is not None:
            aligned[transform.name] = np.array(identity_cache, dtype=np.float32)
        else:
            transformed = transform.apply(image)
            prediction = predict_tiled(model, transformed, device="cuda", tile_size=512, overlap=128, batch_size=4)
            aligned[transform.name] = transform.invert(prediction)
    if set(aligned) != names or any(item.shape != image.shape for item in aligned.values()):
        raise ValueError("TTA inverse alignment failed")
    return aligned


def write_map(path: Path, probability: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = np.load(path, mmap_mode="r")
        if existing.shape != probability.shape or existing.dtype != probability.dtype or not np.array_equal(existing, probability):
            raise ValueError(f"Existing TTA map differs: {path}")
        return
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, probability)
    os.replace(temporary, path)


def generate_tuning_maps(dataset, indices):
    model, _ = load_candidate(V2, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for position, index in enumerate(indices, 1):
        image = image_array(dataset, index)
        identity = np.load(BASE_CACHE / f"{index:04d}.npy", mmap_mode="r")
        start = time.perf_counter()
        aligned = tta_probabilities(model, image, {item.name for item in DIHEDRAL8}, identity_cache=identity)
        elapsed.append(time.perf_counter() - start)
        for name in CONFIGURATIONS:
            averaged = average_aligned({item.name: aligned[item.name] for item in CONFIGURATIONS[name]}, name)
            if not np.isfinite(averaged).all() or averaged.min() < 0 or averaged.max() > 1:
                raise ValueError("Averaged TTA map is invalid")
            write_map(CACHE / "tuning" / name / f"{index:04d}.npy", averaged)
        if position % 10 == 0 or position == len(indices):
            print(f"TTA tuning maps {position}/{len(indices)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return {"images": len(indices), "seven_new_transform_inferences_seconds_per_image": float(np.mean(elapsed)),
            "peak_allocated_vram_mib": torch.cuda.max_memory_allocated()/2**20}


def profile_selected(model, dataset, indices, name: str) -> dict:
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    difference = 0.0
    names = {item.name for item in CONFIGURATIONS[name]}
    for index in indices[:5]:
        image = image_array(dataset, index)
        torch.cuda.synchronize()
        start = time.perf_counter()
        aligned = tta_probabilities(model, image, names)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
        live = average_aligned(aligned, name)
        cached = np.load(CACHE / "tuning" / name / f"{index:04d}.npy", mmap_mode="r")
        difference = max(difference, float(np.max(np.abs(live - cached))))
    if difference > 1e-5:
        raise ValueError(f"Live TTA differs from cached TTA: {difference}")
    return {"sample_images": len(elapsed), "transform_count": len(names),
            "seconds_per_image": float(np.mean(elapsed)), "seconds_each": elapsed,
            "peak_allocated_vram_mib": torch.cuda.max_memory_allocated()/2**20,
            "peak_reserved_vram_mib": torch.cuda.max_memory_reserved()/2**20,
            "max_abs_live_vs_cache_difference": difference}


def gt_encoded(dataset, index):
    filename = dataset.filenames[index]
    return [encode_masks(coco_annotation_instances(dataset, index, alternative))
            for alternative in range(len(dataset.images_by_filename[filename]))]


def map_path(subset: str, name: str, index: int) -> Path:
    return CACHE / subset / name / f"{index:04d}.npy"


def direct_score(dataset, indices, config: TtaConfig, *, subset: str, model=None) -> dict:
    rows = []
    inference_seconds = 0.0
    processing_seconds = 0.0
    for position, index in enumerate(indices, 1):
        if model is not None:
            image = image_array(dataset, index)
            started = time.perf_counter()
            aligned = tta_probabilities(model, image, {item.name for item in CONFIGURATIONS[config.tta]})
            torch.cuda.synchronize()
            probability = average_aligned(aligned, config.tta)
            inference_seconds += time.perf_counter() - started
            write_map(map_path(subset, config.tta, index), probability)
        else:
            probability = np.load(map_path(subset, config.tta, index), mmap_mode="r")
        started = time.perf_counter()
        components = candidate_components(probability, config.threshold, config.min_area)
        masks = [mask for area, mean, mask in components if mean >= config.mean_confidence]
        processing_seconds += time.perf_counter() - started
        filename = dataset.filenames[index]
        scores = [organizer_pq_image(coco_annotation_instances(dataset, index, alternative), masks)
                  for alternative in range(len(dataset.images_by_filename[filename]))]
        rows.append({"index": index, "filename": filename,
                     "tp": sum(score["tp"] for score in scores), "fp": sum(score["fp"] for score in scores),
                     "fn": sum(score["fn"] for score in scores), "sum_iou": sum(score["sum_iou"] for score in scores)})
        if position % 20 == 0 or position == len(indices):
            print(f"TTA direct {subset} {position}/{len(indices)}", flush=True)
    summary = summarize(rows)
    summary["cpu_postprocess_seconds_per_image"] = processing_seconds / len(indices)
    if model is not None:
        summary["tta_inference_seconds_per_image"] = inference_seconds / len(indices)
    return {"config": asdict(config), "summary": summary, "per_image": rows}


def tune() -> Path:
    preflight_data = verify_preflight()
    dataset, partition = context()
    indices = partition["tuning_indices"]
    map_stats = generate_tuning_maps(dataset, indices)
    rows_by_config = {config: [] for config in GRID}
    for position, index in enumerate(indices, 1):
        ground_truth = gt_encoded(dataset, index)
        filename = dataset.filenames[index]
        for name in CONFIGURATIONS:
            probability = np.load(map_path("tuning", name, index), mmap_mode="r")
            for threshold in (.75, .80):
                candidates = candidate_components(probability, threshold, 300)
                encoded = [(area, mean, encode_masks([mask])[0]) for area, mean, mask in candidates]
                for area in (300, 500):
                    for confidence in (.96, .97):
                        config = TtaConfig(name, threshold, area, confidence)
                        predicted = [rle for component_area, mean, rle in encoded
                                     if component_area >= area and mean >= confidence]
                        score = score_encoded_sets(ground_truth, predicted)
                        rows_by_config[config].append({"index": index, "filename": filename,
                            **{key: score[key] for key in ("tp", "fp", "fn", "sum_iou")}})
        if position % 10 == 0 or position == len(indices):
            print(f"TTA tuning grid {position}/{len(indices)}", flush=True)
    summaries = [{"config": asdict(config), "summary": summarize(rows)} for config, rows in rows_by_config.items()]
    ranking = sorted(summaries, key=lambda row: (
        -row["summary"]["global_pq"], -row["summary"]["sq"], row["summary"]["fp"],
        len(CONFIGURATIONS[row["config"]["tta"]]), -row["config"]["mean_confidence"],
        -row["config"]["min_area"], -row["config"]["threshold"], row["config"]["tta"]))
    selected = TtaConfig(**ranking[0]["config"])
    direct = direct_score(dataset, indices, selected, subset="tuning")
    for key in ("global_pq", "sq", "rq", "tp", "fp", "fn"):
        if not np.isclose(direct["summary"][key], ranking[0]["summary"][key], atol=1e-8):
            raise ValueError(f"Encoded grid scorer differs from organizer scorer: {key}")
    model, _ = load_candidate(V2, device="cuda")
    profile = profile_selected(model, dataset, indices, selected.tta)
    del model
    torch.cuda.empty_cache()
    frozen = {"phase": "frozen_after_tuning", "status": "provisional", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_path": str(PREFLIGHT.resolve()), "preflight_sha256": sha256_file(PREFLIGHT),
        "checkpoint_sha256": preflight_data["checkpoint_sha256"],
        "source_sha256": preflight_data["source_sha256"],
        "grid": [asdict(config) for config in GRID], "grid_results": summaries,
        "selection_rule": "tuning Global PQ, SQ, fewer FP, fewer transforms, then confidence/area/threshold descending and name",
        "selected_config": asdict(selected), "selected_tuning_direct": direct,
        "tuning_indices": indices, "internal_indices": partition["internal_check_indices"],
        "tuning_map_generation": map_stats, "selected_live_cuda_profile": profile,
        "limitation": "Original v2 was historically scored on internal check; hidden scorer parity unverified."}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "reports" / f"{EXPERIMENT}_frozen_{stamp}.json"
    with destination.open("x", encoding="utf-8") as file:
        json.dump(frozen, file, indent=2)
    print("FROZEN", destination, "selected", asdict(selected),
          "tuning_Global_PQ", direct["summary"]["global_pq"], flush=True)
    return destination


def internal(frozen_path: Path) -> Path:
    if list((ROOT / "reports").glob(f"{EXPERIMENT}_internal_*.json")):
        raise FileExistsError("TTA internal check already exists; only one score is allowed")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    preflight_data = verify_preflight()
    if frozen["phase"] != "frozen_after_tuning" or frozen["preflight_sha256"] != sha256_file(PREFLIGHT) or \
            frozen["source_sha256"] != preflight_data["source_sha256"] or \
            frozen["checkpoint_sha256"] != preflight_data["checkpoint_sha256"]:
        raise ValueError("Frozen TTA source or checkpoint changed")
    dataset, partition = context()
    if frozen["internal_indices"] != partition["internal_check_indices"]:
        raise ValueError("Frozen internal partition changed")
    model, _ = load_candidate(V2, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    selected = TtaConfig(**frozen["selected_config"])
    result = direct_score(dataset, frozen["internal_indices"], selected, subset="internal", model=model)
    result["summary"]["peak_allocated_vram_mib"] = torch.cuda.max_memory_allocated()/2**20
    del model
    torch.cuda.empty_cache()
    historical = json.loads(BASELINE_INTERNAL.read_text(encoding="utf-8"))
    baseline = next(row for row in historical["candidate_results"] if row["config"]["name"] == "confidence_0p97")
    if not np.isclose(baseline["summary"]["global_pq"], .4099034711305698) or \
            [row["index"] for row in baseline["per_image"]] != frozen["internal_indices"]:
        raise ValueError("Historical v2 baseline mismatch")
    interval = bootstrap_interval(baseline["per_image"], result["per_image"])
    delta = result["summary"]["global_pq"] - baseline["summary"]["global_pq"]
    supported = delta > .005 and interval[0] > 0 and result["summary"]["tp"] >= .9 * baseline["summary"]["tp"]
    report = {"phase": "internal_scored_once", "status": "provisional", "experiment_id": EXPERIMENT,
        "frozen_path": str(frozen_path.resolve()), "frozen_sha256": sha256_file(frozen_path),
        "baseline_source": str(BASELINE_INTERNAL), "baseline_source_sha256": sha256_file(BASELINE_INTERNAL),
        "baseline_v2": baseline, "tta_candidate": result, "delta_global_pq": delta,
        "paired_capture_date_bootstrap_95pct_delta_interval": interval,
        "bootstrap_seed": 42, "bootstrap_replicates": 5000,
        "decision_rule": "Global PQ gain >0.005, paired bootstrap lower bound >0, TP >=90% of baseline",
        "decision": "submission_supported" if supported else "retain_official_v2_0p33",
        "limitation": "Internal check historically exposed to original v2; hidden scorer parity unverified."}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "reports" / f"{EXPERIMENT}_internal_{stamp}.json"
    with destination.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print("INTERNAL", destination, "baseline", baseline["summary"],
          "tta", result["summary"], "CI", interval, "decision", report["decision"], flush=True)
    return destination


def submit(internal_path: Path) -> Path:
    report = json.loads(internal_path.read_text(encoding="utf-8"))
    if report["decision"] != "submission_supported":
        raise ValueError("TTA did not meet the submission gate")
    frozen_path = Path(report["frozen_path"])
    if sha256_file(frozen_path) != report["frozen_sha256"]:
        raise ValueError("Frozen TTA report changed")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    preflight_data = verify_preflight()
    if frozen["source_sha256"] != preflight_data["source_sha256"] or frozen["checkpoint_sha256"] != preflight_data["checkpoint_sha256"]:
        raise ValueError("Source or checkpoint changed")
    original_csv_hash = sha256_file(OLD_CSV)
    model, _ = load_candidate(V2, device="cuda")
    config = TtaConfig(**frozen["selected_config"])
    test_dir = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    images = sorted(test_dir.glob("*.jpeg"))
    if len(images) != 180 or len({path.stem for path in images}) != 180:
        raise ValueError("Expected exactly 180 unique test images")
    rows = []
    zero_images = 0
    for position, path in enumerate(images, 1):
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != (2048, 2048):
            raise ValueError(f"Invalid test image: {path}")
        image = raw.astype(np.float32) / 255.0
        aligned = tta_probabilities(model, image, {item.name for item in CONFIGURATIONS[config.tta]})
        probability = average_aligned(aligned, config.tta)
        masks = [mask for area, mean, mask in candidate_components(probability, config.threshold, config.min_area)
                 if mean >= config.mean_confidence]
        zero_images += not masks
        for number, mask in enumerate(masks, 1):
            rows.append({"filament_id": f"{path.stem}_{number}", "segmentation_rle": encode_coco_counts(mask)})
        if position % 20 == 0:
            print(f"TTA test images {position}/180", flush=True)
    if not rows or validate_submission_rows(rows, images) != len(rows):
        raise ValueError("TTA submission rows failed COCO RLE or ID validation")
    if sha256_file(OLD_CSV) != original_csv_hash or preflight() != preflight_data:
        raise ValueError("Historical artifact changed during test inference")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "outputs" / f"submission_provisional_{EXPERIMENT}_{stamp}.csv"
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(destination, mode="x", index=False)
    saved = pd.read_csv(destination, dtype=str)
    if list(saved.columns) != ["filament_id", "segmentation_rle"] or len(saved) != len(rows):
        raise ValueError("Saved TTA CSV failed schema or row-count validation")
    manifest = {"status": "provisional", "uploaded_to_kaggle": False, "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "csv": str(destination),
        "csv_sha256": sha256_file(destination), "rows": len(rows), "test_images_processed": 180,
        "zero_prediction_images": zero_images, "configuration": asdict(config),
        "checkpoint_sha256": preflight_data["checkpoint_sha256"], "source_sha256": preflight_data["source_sha256"],
        "frozen_report": str(frozen_path), "frozen_report_sha256": sha256_file(frozen_path),
        "internal_report": str(internal_path), "internal_report_sha256": sha256_file(internal_path),
        "old_submission_sha256": original_csv_hash,
        "validation": "180 images; compressed COCO RLE round-trip decoded at 2048x2048; IDs and CSV schema checked"}
    manifest_path = destination.with_suffix(".provenance.json")
    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print("SUBMISSION_COMPLETE", destination, manifest_path, flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "tune", "internal", "submit"))
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--internal", type=Path)
    args = parser.parse_args()
    if args.phase == "preflight":
        record = preflight()
        with PREFLIGHT.open("x", encoding="utf-8") as file:
            json.dump(record, file, indent=2)
        print("PREFLIGHT_PASS", PREFLIGHT, flush=True)
    elif args.phase == "tune":
        tune()
    elif args.phase == "internal":
        if args.frozen is None:
            parser.error("--frozen required")
        internal(args.frozen)
    else:
        if args.internal is None:
            parser.error("--internal required")
        submit(args.internal)
