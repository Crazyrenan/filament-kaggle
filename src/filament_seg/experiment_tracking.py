"""Collision-safe experiment allocation, metadata, and fixed-protocol comparison."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import pandas as pd
import numpy as np
import torch
import yaml

from filament_seg.evaluate import evaluate_full_resolution_dataset
from filament_seg.model import build_model
from filament_seg.train import load_checkpoint
from filament_seg.utils import project_root

RETRAIN_PREFIX = "unet_resnet34_v2_retrain"
COMPARISON_METRICS = {
    "Macro Dice": ("semantic_metrics", "macro_dice"),
    "Micro Dice": ("semantic_metrics", "micro_dice"),
    "Macro IoU": ("semantic_metrics", "macro_iou"),
    "Precision": ("semantic_metrics", "macro_precision"),
    "Recall": ("semantic_metrics", "macro_recall"),
    "Global PQ": ("instance_metrics", "global_pq"),
    "Macro PQ": ("instance_metrics", "macro_pq"),
    "SQ": ("instance_metrics", "global_sq"),
    "RQ": ("instance_metrics", "global_rq"),
    "TP": ("instance_metrics", "total_tp"),
    "FP": ("instance_metrics", "total_fp"),
    "FN": ("instance_metrics", "total_fn"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _registry_lock(path: Path, timeout: float = 10.0) -> Iterator[None]:
    """Serialize JSON registry updates between notebook processes."""
    lock = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Experiment registry lock is busy: {lock}")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.rmdir()


def _update_registry(path: Path, experiment_id: str, fields: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _registry_lock(path):
        registry = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"experiments": {}}
        entries = registry.setdefault("experiments", {})
        entries[experiment_id] = {**entries.get(experiment_id, {}), **fields}
        _atomic_json(path, registry)


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root(), capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def validation_fingerprint(dataset: Any, indices: list[int]) -> str:
    """Hash the COCO source and ordered physical validation filenames."""
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Validation indices must be nonempty and unique")
    digest = hashlib.sha256()
    annotation_path = Path(dataset.annotation_path)
    with annotation_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    for index in indices:
        digest.update(str(dataset.filenames[index]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_fresh_experiment(
    base_config: dict[str, Any],
    *,
    prefix: str = RETRAIN_PREFIX,
    checkpoint_root: str | Path | None = None,
    registry_path: str | Path | None = None,
    dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Reserve a never-used version and save its effective config and initial metadata.

    A pre-existing directory, even an empty one, is reserved and never reused.
    The caller receives a deep copy; `base_config` remains unchanged.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]*", prefix):
        raise ValueError("prefix must be a safe lowercase directory stem")
    root = Path(checkpoint_root) if checkpoint_root is not None else project_root() / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    registry = Path(registry_path) if registry_path is not None else project_root() / "reports" / "experiment_registry.json"
    existing = [int(match.group(1)) for child in root.iterdir()
                if (match := re.fullmatch(re.escape(prefix) + r"_(\d+)", child.name))]
    version = max(existing, default=0) + 1
    while True:
        experiment_id = f"{prefix}_{version:02d}"
        directory = root / experiment_id
        try:
            directory.mkdir()
            break
        except FileExistsError:
            version += 1
    effective = copy.deepcopy(base_config)
    effective.setdefault("project", {})["experiment_name"] = experiment_id.replace("_", "-")
    effective.setdefault("training", {})["checkpoint_dir"] = str(directory.resolve())
    effective["training"]["track_experiment"] = True
    effective["training"]["registry_path"] = str(registry.resolve())
    created_at = _utc_now()
    config_path = directory / "config.yaml"
    with config_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(effective, handle, sort_keys=False)
    _atomic_json(directory / "history.json", [])
    metrics = {
        "status": "allocated", "best_epoch": None, "best_val_dice": None,
        "global_pq": None, "sq": None, "rq": None, "evaluation_settings": None,
        "dataset_fingerprint": dataset_fingerprint,
    }
    _atomic_json(directory / "metrics.json", metrics)
    _update_registry(registry, experiment_id, {
        "experiment_id": experiment_id,
        "experiment_name": effective["project"]["experiment_name"],
        "architecture": effective["model"]["architecture"],
        "encoder_name": effective["model"]["encoder_name"],
        "encoder_initialization": effective["model"].get("encoder_weights"),
        "seed": effective["project"].get("seed"),
        "training_config": str(config_path),
        "git_commit": _git_revision(),
        "checkpoint_dir": str(directory.resolve()),
        "best_checkpoint": str(directory / "best_model.pth"),
        "latest_checkpoint": str(directory / "latest.pth"),
        "status": "allocated", "best_epoch": None, "best_val_dice": None,
        "global_pq": None, "sq": None, "rq": None,
        "evaluation_settings": None, "dataset_fingerprint": dataset_fingerprint,
        "created_at": created_at,
    })
    return effective


def load_experiment_config(
    experiment_id: str, *, checkpoint_root: str | Path | None = None
) -> dict[str, Any]:
    """Read a previously allocated experiment without creating a new version."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", experiment_id):
        raise ValueError("Invalid experiment identifier")
    root = Path(checkpoint_root) if checkpoint_root is not None else project_root() / "checkpoints"
    directory = root / experiment_id
    path = directory / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config is missing: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if Path(config["training"]["checkpoint_dir"]).resolve() != directory.resolve():
        raise ValueError("Experiment config checkpoint directory mismatch")
    if config["project"]["experiment_name"] != experiment_id.replace("_", "-"):
        raise ValueError("Experiment identifier mismatch")
    return config


def record_training_progress(
    config: dict[str, Any], history: list[dict[str, Any]], *, status: str
) -> None:
    """Persist checkpoint-derived history and summary after each successful save."""
    train_cfg = config["training"]
    if not train_cfg.get("track_experiment"):
        return
    directory = Path(train_cfg["checkpoint_dir"])
    registry = Path(train_cfg["registry_path"])
    if not (directory / "config.yaml").is_file():
        raise FileNotFoundError("Tracked experiment config is missing")
    _atomic_json(directory / "history.json", history)
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = max(history, key=lambda row: row.get("val_dice", float("-inf"))) if history else None
    metrics.update({"status": status, "best_epoch": best.get("epoch") if best else None,
                    "best_val_dice": best.get("val_dice") if best else None})
    _atomic_json(metrics_path, metrics)
    _update_registry(registry, directory.name, {"status": status,
                     "best_epoch": metrics["best_epoch"], "best_val_dice": metrics["best_val_dice"]})


def record_evaluation(
    config: dict[str, Any], summary: dict[str, Any], *,
    settings: dict[str, Any], dataset_fingerprint: str,
) -> None:
    """Record diagnostic validation results without touching model weights."""
    train_cfg = config["training"]
    if not train_cfg.get("track_experiment"):
        raise ValueError("Evaluation tracking requires an allocated experiment")
    directory = Path(train_cfg["checkpoint_dir"])
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    recorded_fingerprint = metrics.get("dataset_fingerprint")
    if recorded_fingerprint is not None and recorded_fingerprint != dataset_fingerprint:
        raise ValueError("Evaluation dataset fingerprint differs from the allocated experiment")
    instance = summary["instance_metrics"]
    fields = {"global_pq": instance["global_pq"], "sq": instance["global_sq"],
              "rq": instance["global_rq"], "evaluation_settings": settings,
              "dataset_fingerprint": dataset_fingerprint, "evaluated_at": _utc_now(),
              "metric_kind": "diagnostic_pq"}
    metrics.update(fields)
    _atomic_json(metrics_path, metrics)
    _update_registry(Path(train_cfg["registry_path"]), directory.name, fields)


def compare_trained_experiments(
    original_checkpoint: str | Path,
    retrained_checkpoint: str | Path,
    dataset: Any,
    validation_indices: list[int],
    *, device: Any = "cpu", threshold: float = 0.70, min_area: int = 500,
    tile_size: int = 512, overlap: int = 128,
    expected_dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Evaluate two genuine best checkpoints on the same images and matching rules."""
    fingerprint = validation_fingerprint(dataset, validation_indices)
    if expected_dataset_fingerprint is not None and fingerprint != expected_dataset_fingerprint:
        raise ValueError("Comparison validation dataset fingerprint differs from the retrain experiment")
    paths = [Path(original_checkpoint), Path(retrained_checkpoint)]
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError("Comparison requires two distinct checkpoints")
    results = []
    for path in paths:
        results.append(evaluate_trained_experiment(
            path, dataset, validation_indices, device=device, threshold=threshold,
            min_area=min_area, tile_size=tile_size, overlap=overlap,
        ))
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    rows = []
    first, second = (result["summary"] for result in results)
    for label, (section, key) in COMPARISON_METRICS.items():
        a, b = first[section][key], second[section][key]
        rows.append({"Metric": label, "Original v2": a, "Retrained v2": b, "Delta": b - a})
    return {
        "table": pd.DataFrame(rows), "original": results[0], "retrained": results[1],
        "validation_fingerprint": fingerprint,
        "settings": {"threshold": threshold, "min_area": min_area,
                     "tile_size": tile_size, "overlap": overlap, "annotation_set_index": 0},
    }


def evaluate_trained_experiment(
    best_checkpoint: str | Path, dataset: Any, validation_indices: list[int], *,
    device: Any = "cpu", threshold: float = 0.70, min_area: int = 500,
    tile_size: int = 512, overlap: int = 128,
) -> dict[str, Any]:
    """Read a trained best checkpoint and evaluate without writing model weights."""
    path = Path(best_checkpoint)
    if path.name != "best_model.pth" or not path.is_file():
        raise FileNotFoundError(f"Trained best checkpoint required: {path}")
    model = build_model(pretrained=False).to(device)
    state = load_checkpoint(path, model, map_location=device)
    if state.get("epoch", 0) < 1 or state.get("purpose") == "smoke_test" or any(
        row.get("smoke_test") for row in state.get("history", [])
    ):
        raise ValueError(f"Smoke or untrained checkpoint: {path}")
    return evaluate_full_resolution_dataset(
        model=model, dataset=dataset, sample_indices=list(validation_indices), device=device,
        threshold=threshold, min_area=min_area, tile_size=tile_size, overlap=overlap,
        annotation_set_index=0,
    )


def qualitative_case_candidates(comparison: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Rank instance-level error candidates from identical per-image comparisons.

    Fragmentation/merging labels are hypotheses based on counts; inspect masks
    before treating them as established causes.
    """
    original = comparison["original"]["per_image"]
    retrained = comparison["retrained"]["per_image"]
    if [row["filename"] for row in original] != [row["filename"] for row in retrained]:
        raise ValueError("Per-image comparison rows are not aligned")
    paired = [{"filename": a["filename"], "index": a["index"],
               "original_pq": a["pq"], "retrained_pq": b["pq"],
               "original": a, "retrained": b}
              for a, b in zip(original, retrained)]
    return {
        "retrained_improves": sorted((p for p in paired if p["retrained_pq"] > p["original_pq"]),
                                    key=lambda p: p["retrained_pq"] - p["original_pq"], reverse=True),
        "original_better": sorted((p for p in paired if p["original_pq"] > p["retrained_pq"]),
                                  key=lambda p: p["original_pq"] - p["retrained_pq"], reverse=True),
        "both_fail": sorted((p for p in paired if p["original_pq"] == 0 and p["retrained_pq"] == 0),
                            key=lambda p: p["filename"]),
        "fragmentation_candidates": [p for p in paired if p["retrained"]["pred_instances"] > p["retrained"]["gt_instances"]],
        "merging_candidates": [p for p in paired if p["retrained"]["pred_instances"] < p["retrained"]["gt_instances"]],
    }


def instance_error_overlay(
    gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], matching: dict[str, Any],
) -> np.ndarray:
    """Color matched instances green, missed GT blue, unmatched predictions red."""
    if not gt_masks and not pred_masks:
        raise ValueError("At least one instance mask is needed to determine image shape")
    shape = np.asarray((gt_masks or pred_masks)[0]).shape
    overlay = np.zeros((*shape, 3), dtype=np.uint8)
    for pair in matching["matches"]:
        overlay[np.asarray(gt_masks[pair["gt_index"]], dtype=bool)] = (0, 180, 0)
        overlay[np.asarray(pred_masks[pair["pred_index"]], dtype=bool)] = (0, 220, 0)
    for index in matching["unmatched_gt"]:
        overlay[np.asarray(gt_masks[index], dtype=bool)] = (0, 80, 255)
    for index in matching["unmatched_pred"]:
        overlay[np.asarray(pred_masks[index], dtype=bool)] = (255, 30, 30)
    return overlay
