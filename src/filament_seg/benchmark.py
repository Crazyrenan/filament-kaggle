"""Reproducible probability caching and local instance-level model comparison."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask

from filament_seg.inference import iter_probability_instances, predict_tiled
from filament_seg.model import build_model
from filament_seg.official_metric import organizer_pq_image
from filament_seg.train import load_checkpoint
from filament_seg.transformer import build_binary_segformer


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coco_annotation_instances(dataset: Any, index: int, annotation_set_index: int) -> list[np.ndarray]:
    """Decode source COCO polygons with pycocotools for evaluation GT masks."""
    filename = dataset.filenames[index]
    record = dataset.images_by_filename[filename][annotation_set_index]
    height, width = int(record["height"]), int(record["width"])
    instances = []
    for annotation in dataset.annotations_by_image_id.get(record["id"], []):
        polygons = annotation["segmentation"]
        if not polygons:
            continue
        encoded = coco_mask.frPyObjects(polygons, height, width)
        combined = coco_mask.merge(encoded)
        instances.append(coco_mask.decode(combined).astype(bool))
    return instances


def load_candidate(checkpoint: str | Path, device: str = "cuda") -> tuple[torch.nn.Module, dict[str, Any]]:
    path = Path(checkpoint)
    if path.name != "best_model.pth" or not path.is_file():
        raise FileNotFoundError(f"Trained best checkpoint required: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("epoch", 0) < 1 or state.get("purpose") == "smoke_test":
        raise ValueError(f"Not a trained checkpoint: {path}")
    model_config = state.get("model_config", {})
    architecture = model_config.get("architecture", "Unet")
    if architecture == "SegFormer":
        model, _ = build_binary_segformer(
            model_config["variant"], pretrained=False, hf_config=model_config["hf_config"]
        )
    elif architecture.lower() == "unet":
        model = build_model(pretrained=False)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    load_checkpoint(path, model, map_location="cpu")
    model.to(device).eval()
    metadata = {
        "experiment_id": path.parent.name, "architecture": architecture,
        "checkpoint": str(path.resolve()), "checkpoint_sha256": sha256_file(path),
        "epoch": state["epoch"], "model_config": model_config,
    }
    return model, metadata


def make_partition(dataset: Any, indices: list[int], path: str | Path, *, seed: int = 42) -> dict[str, Any]:
    """Persist a date-grouped tuning/internal-check split of existing validation."""
    from filament_seg.experiment_tracking import validation_fingerprint

    target = Path(path)
    fingerprint = validation_fingerprint(dataset, indices)
    if target.is_file():
        saved = json.loads(target.read_text(encoding="utf-8"))
        if saved["validation_fingerprint"] != fingerprint or sorted(saved["tuning_indices"] + saved["internal_check_indices"]) != sorted(indices):
            raise ValueError("Saved validation partition is incompatible with current data")
        return saved
    groups: dict[str, list[int]] = {}
    for index in indices:
        filename = dataset.filenames[index]
        groups.setdefault(filename[:8], []).append(index)
    ordered_dates = sorted(groups, key=lambda date: hashlib.sha256(f"{seed}:{date}".encode()).hexdigest())
    tuning: list[int] = []
    for date in ordered_dates:
        if len(tuning) >= len(indices) // 2:
            break
        tuning.extend(groups[date])
    internal = [index for index in indices if index not in set(tuning)]
    if not tuning or not internal:
        raise ValueError("Could not create two nonempty validation partitions")
    result = {
        "seed": seed, "method": "SHA256(seed:capture_date), groups kept together",
        "validation_fingerprint": fingerprint, "tuning_indices": tuning,
        "internal_check_indices": internal,
        "tuning_filenames": [dataset.filenames[i] for i in tuning],
        "internal_check_filenames": [dataset.filenames[i] for i in internal],
        "limitation": "Original v2 was historically evaluated on these images; internal check is not independent.",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def cache_probability_maps(
    model: torch.nn.Module, metadata: dict[str, Any], dataset: Any,
    indices: list[int], root: str | Path, *, device: str = "cuda",
    tile_size: int = 512, overlap: int = 128, batch_size: int = 4,
) -> tuple[Path, dict[str, Any]]:
    from filament_seg.experiment_tracking import validation_fingerprint

    fingerprint = validation_fingerprint(dataset, indices)
    folder = Path(root) / metadata["experiment_id"] / metadata["checkpoint_sha256"][:16]
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "cache_manifest.json"
    expected = {
        "experiment_id": metadata["experiment_id"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "validation_fingerprint": fingerprint, "indices": indices,
        "filenames": [dataset.filenames[i] for i in indices],
        "tile_size": tile_size, "overlap": overlap, "batch_size": batch_size,
        "dtype": "float32", "shape": [2048, 2048],
    }
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved != expected:
            raise ValueError(f"Probability cache provenance mismatch: {folder}")
    for position, index in enumerate(indices):
        destination = folder / f"{index:04d}.npy"
        if destination.is_file():
            existing = np.load(destination, mmap_mode="r")
            if existing.shape != (2048, 2048) or existing.dtype != np.float32:
                raise ValueError(f"Invalid cached probability map: {destination}")
            continue
        image_path = dataset.image_dir / dataset.filenames[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f"Invalid source image: {image_path}")
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        probability = predict_tiled(model, image.astype(np.float32) / 255.0,
                                    device=device, tile_size=tile_size, overlap=overlap,
                                    batch_size=batch_size)
        if probability.shape != image.shape or not np.isfinite(probability).all():
            raise ValueError(f"Invalid model probability map: {image_path}")
        temporary = destination.with_suffix(".tmp.npy")
        np.save(temporary, probability)
        os.replace(temporary, destination)
        timing = destination.with_suffix(".timing.json")
        with timing.open("x", encoding="utf-8") as handle:
            json.dump({"seconds": time.perf_counter() - started,
                       "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20
                       if torch.cuda.is_available() and torch.device(device).type == "cuda" else 0}, handle)
        if (position + 1) % 10 == 0 or position + 1 == len(indices):
            print(f"{metadata['experiment_id']} cached {position + 1}/{len(indices)} "
                  f"last_sec={time.perf_counter() - started:.2f}", flush=True)
    if not manifest.is_file():
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2)
    return folder, expected


def score_probability_maps(
    dataset: Any, indices: list[int], cache_folder: str | Path, *,
    threshold: float, min_area: int, annotation_set_index: int | None = None,
) -> dict[str, Any]:
    if not indices:
        raise ValueError("At least one image is required for scoring")
    records = []
    total_tp = total_fp = total_fn = total_inter = total_gt = total_pred = total_union = 0
    matched_iou = 0.0
    for index in indices:
        probability = np.load(Path(cache_folder) / f"{index:04d}.npy", mmap_mode="r")
        filename = dataset.filenames[index]
        number_of_sets = len(dataset.images_by_filename[filename]) if hasattr(dataset, "images_by_filename") else 1
        annotation_sets = range(number_of_sets) if annotation_set_index is None else (annotation_set_index,)
        all_gt_instances = [
            coco_annotation_instances(dataset, index, set_index)
            if hasattr(dataset, "annotations_by_image_id")
            else [item["mask"].astype(bool) for item in dataset.iter_instances(index, set_index)]
            for set_index in annotation_sets
        ]
        gt_instances = all_gt_instances[0]
        gt_union = np.zeros(probability.shape, dtype=bool)
        for mask in gt_instances:
            gt_union |= mask
        predicted = list(iter_probability_instances(probability, threshold=threshold, min_area=min_area))
        per_set = [organizer_pq_image(instances, predicted) for instances in all_gt_instances]
        image_tp = sum(score["tp"] for score in per_set)
        image_fp = sum(score["fp"] for score in per_set)
        image_fn = sum(score["fn"] for score in per_set)
        image_iou_sum = sum(score["sum_iou"] for score in per_set)
        image_denominator = image_tp + 0.5 * (image_fp + image_fn)
        image_pq = image_iou_sum / image_denominator if image_denominator else 0.0
        image_sq = image_iou_sum / image_tp if image_tp else 0.0
        image_rq = image_tp / image_denominator if image_denominator else 0.0
        semantic = probability >= threshold
        intersection = int(np.count_nonzero(semantic & gt_union))
        gt_pixels = int(np.count_nonzero(gt_union))
        pred_pixels = int(np.count_nonzero(semantic))
        union = gt_pixels + pred_pixels - intersection
        dice = 2 * intersection / (gt_pixels + pred_pixels) if gt_pixels + pred_pixels else 1.0
        iou = intersection / union if union else 1.0
        records.append({
            "index": index, "filename": filename, "pq": image_pq,
            "sq": image_sq, "rq": image_rq, "tp": image_tp, "fp": image_fp, "fn": image_fn,
            "annotation_sets": len(all_gt_instances),
            "gt_instances": sum(len(instances) for instances in all_gt_instances),
            "pred_instances": len(predicted),
            "dice": dice, "iou": iou,
        })
        total_tp += image_tp
        total_fp += image_fp
        total_fn += image_fn
        matched_iou += image_iou_sum
        total_inter += intersection
        total_gt += gt_pixels
        total_pred += pred_pixels
        total_union += union
    denominator = total_tp + 0.5 * (total_fp + total_fn)
    summary = {
        "num_images": len(indices), "threshold": threshold, "min_area": min_area,
        "global_pq": matched_iou / denominator if denominator else 0.0,
        "sq": matched_iou / total_tp if total_tp else 0.0,
        "rq": total_tp / denominator if denominator else 0.0,
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "macro_pq": float(np.mean([row["pq"] for row in records])),
        "macro_dice": float(np.mean([row["dice"] for row in records])),
        "micro_dice": 2 * total_inter / (total_gt + total_pred) if total_gt + total_pred else 1.0,
        "macro_iou": float(np.mean([row["iou"] for row in records])),
        "micro_iou": total_inter / total_union if total_union else 1.0,
    }
    return {"summary": summary, "per_image": records}


def code_revision(root: str | Path) -> dict[str, str]:
    root = Path(root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    files = [root / "src/filament_seg" / name for name in
             ("benchmark.py", "evaluate.py", "official_metric.py", "inference.py",
              "submission.py", "dataset.py", "transformer.py")]
    return {"git_commit": commit, "source_sha256": {str(path.relative_to(root)): sha256_file(path) for path in files}}


def verify_independent_holdout(
    holdout_filenames: list[str], development_filenames: list[str], test_filenames: list[str]
) -> dict[str, int]:
    """Require physical and capture-date disjointness before verified selection."""
    holdout = set(holdout_filenames)
    development = set(development_filenames)
    test = set(test_filenames)
    if not holdout or len(holdout) != len(holdout_filenames):
        raise ValueError("Holdout physical filenames must be nonempty and unique")
    if holdout & (development | test):
        raise ValueError("Independent holdout overlaps development or competition test images")
    holdout_dates = {name[:8] for name in holdout}
    seen_dates = {name[:8] for name in development | test}
    if holdout_dates & seen_dates:
        raise ValueError("Independent holdout capture dates overlap development or test images")
    return {"images": len(holdout), "capture_dates": len(holdout_dates)}
