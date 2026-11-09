"""Versioned retraining and fixed-protocol comparison safety checks."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import filament_seg.experiment_tracking as tracking
import filament_seg.train as train


def config(tmp_path):
    return {
        "project": {"experiment_name": "original", "seed": 42},
        "model": {"architecture": "Unet", "encoder_name": "resnet34", "encoder_weights": "imagenet",
                  "in_channels": 1, "classes": 1},
        "training": {"epochs": 2, "batch_size": 2, "learning_rate": 1e-3, "weight_decay": 0.0,
                     "checkpoint_dir": str(tmp_path / "original"), "mixed_precision": False,
                     "early_stopping_patience": 5},
    }


def paths(tmp_path):
    return tmp_path / "checkpoints", tmp_path / "registry.json"


def test_atomic_versioning_preserves_existing_and_base_config(tmp_path):
    root, registry = paths(tmp_path)
    root.mkdir()
    (root / "unet_resnet34_v2_retrain_01").mkdir()  # even an empty reserved directory counts
    old = root / "unet_resnet34_v2"
    old.mkdir()
    original = old / "best_model.pth"
    original.write_bytes(b"historical weights")
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    base = config(tmp_path)
    snapshot = copy.deepcopy(base)
    first = tracking.prepare_fresh_experiment(base, checkpoint_root=root, registry_path=registry,
                                               dataset_fingerprint="dataset-sha")
    second = tracking.prepare_fresh_experiment(base, checkpoint_root=root, registry_path=registry)
    assert base == snapshot
    assert Path(first["training"]["checkpoint_dir"]).name == "unet_resnet34_v2_retrain_02"
    assert Path(second["training"]["checkpoint_dir"]).name == "unet_resnet34_v2_retrain_03"
    assert first["model"] == base["model"] and first["project"]["seed"] == 42
    assert tracking.load_experiment_config("unet_resnet34_v2_retrain_02", checkpoint_root=root) == first
    assert json.loads((root / "unet_resnet34_v2_retrain_02/metrics.json").read_text())["dataset_fingerprint"] == "dataset-sha"
    entry = json.loads(registry.read_text())["experiments"]["unet_resnet34_v2_retrain_02"]
    assert entry["encoder_initialization"] == "imagenet" and entry["git_commit"]
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before


def test_allocator_rejects_unsafe_name_and_resume_requires_latest(tmp_path):
    root, registry = paths(tmp_path)
    with pytest.raises(ValueError):
        tracking.prepare_fresh_experiment(config(tmp_path), prefix="../escape", checkpoint_root=root, registry_path=registry)
    effective = tracking.prepare_fresh_experiment(config(tmp_path), checkpoint_root=root, registry_path=registry)
    with pytest.raises(FileNotFoundError, match="latest.pth"):
        train.train_v2_model(effective, [], [], None, [], device="cpu", resume=True)


def test_tracked_training_and_selected_resume(tmp_path, monkeypatch):
    root, registry = paths(tmp_path)
    first = tracking.prepare_fresh_experiment(config(tmp_path), checkpoint_root=root, registry_path=registry)
    second = tracking.prepare_fresh_experiment(config(tmp_path), checkpoint_root=root, registry_path=registry)
    monkeypatch.setattr(train, "build_model", lambda **kwargs: nn.Conv2d(1, 1, 1))
    samples = [{"image": torch.rand(1, 8, 8), "mask": (torch.rand(1, 8, 8) > 0.5).float()} for _ in range(2)]
    loader = DataLoader(samples, batch_size=2)
    result = train.train_v2_model(first, loader, loader, None, [], device="cpu", resume=False, max_epochs=1)
    first_dir = Path(first["training"]["checkpoint_dir"])
    second_dir = Path(second["training"]["checkpoint_dir"])
    assert (first_dir / "latest.pth").is_file() and (first_dir / "best_model.pth").is_file()
    assert json.loads((first_dir / "history.json").read_text())[0]["epoch"] == 1
    assert json.loads((first_dir / "metrics.json").read_text())["best_epoch"] == 1
    assert not (second_dir / "latest.pth").exists()
    with pytest.raises(FileExistsError):
        train.train_v2_model(first, loader, loader, None, [], device="cpu", resume=False, max_epochs=1)
    resumed = train.train_v2_model(
        tracking.load_experiment_config(first_dir.name, checkpoint_root=root),
        loader, loader, None, [], device="cpu", resume=True, max_epochs=2,
    )
    assert len(resumed["history"]) == 2
    assert torch.load(first_dir / "latest.pth", map_location="cpu", weights_only=False)["epoch"] == 2
    assert not (second_dir / "latest.pth").exists()


def test_smoke_weights_refused_for_production_resume(tmp_path, monkeypatch):
    root, registry = paths(tmp_path)
    effective = tracking.prepare_fresh_experiment(config(tmp_path), checkpoint_root=root, registry_path=registry)
    monkeypatch.setattr(train, "build_model", lambda **kwargs: nn.Conv2d(1, 1, 1))
    directory = Path(effective["training"]["checkpoint_dir"])
    torch.save({"epoch": 0, "purpose": "smoke_test", "model_state_dict": nn.Conv2d(1, 1, 1).state_dict(),
                "history": [{"smoke_test": True}]}, directory / "latest.pth")
    with pytest.raises(ValueError, match="smoke"):
        train.train_v2_model(effective, [], [], None, [], device="cpu", resume=True)


def test_evaluation_and_comparison_do_not_mutate_weights(tmp_path, monkeypatch):
    class Dataset:
        filenames = ["a.jpeg", "b.jpeg", "c.jpeg"]
        annotation_path = tmp_path / "annotations.json"

    dataset = Dataset()
    dataset.annotation_path.write_text("{}", encoding="utf-8")
    model_a, model_b = nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)
    dirs = [tmp_path / "original", tmp_path / "retrained"]
    for directory, model in zip(dirs, [model_a, model_b]):
        directory.mkdir()
        torch.save({"epoch": 1, "model_state_dict": model.state_dict(), "history": []}, directory / "best_model.pth")
    paths_ = [directory / "best_model.pth" for directory in dirs]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths_]
    monkeypatch.setattr(tracking, "build_model", lambda **kwargs: nn.Conv2d(1, 1, 1))
    calls = []

    def fake_evaluate(**kwargs):
        calls.append({k: v for k, v in kwargs.items() if k != "model"})
        value = float(kwargs["model"].weight.detach().mean())
        return {"summary": {
            "semantic_metrics": {key: value for section, key in tracking.COMPARISON_METRICS.values() if section == "semantic_metrics"},
            "instance_metrics": {key: value for section, key in tracking.COMPARISON_METRICS.values() if section == "instance_metrics"},
        }, "per_image": [{"filename": "a.jpeg", "pq": value}]}

    monkeypatch.setattr(tracking, "evaluate_full_resolution_dataset", fake_evaluate)
    comparison = tracking.compare_trained_experiments(paths_[0], paths_[1], dataset, [0, 2], device="cpu")
    assert len(comparison["table"]) == 12
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[0]["sample_indices"] == [0, 2]
    assert calls[0]["threshold"] == 0.70 and calls[0]["min_area"] == 500
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths_] == before
    with pytest.raises(ValueError, match="distinct"):
        tracking.compare_trained_experiments(paths_[0], paths_[0], dataset, [0], device="cpu")


def test_record_evaluation_metadata(tmp_path):
    root, registry = paths(tmp_path)
    effective = tracking.prepare_fresh_experiment(config(tmp_path), checkpoint_root=root, registry_path=registry)
    summary = {"instance_metrics": {"global_pq": 0.42, "global_sq": 0.7, "global_rq": 0.6}}
    settings = {"threshold": 0.7, "min_area": 500, "image_count": 2}
    tracking.record_evaluation(effective, summary, settings=settings, dataset_fingerprint="abc123")
    directory = Path(effective["training"]["checkpoint_dir"])
    metrics = json.loads((directory / "metrics.json").read_text())
    entry = json.loads(registry.read_text())["experiments"][directory.name]
    assert metrics["global_pq"] == entry["global_pq"] == 0.42
    assert metrics["evaluation_settings"] == settings and entry["dataset_fingerprint"] == "abc123"
    with pytest.raises(ValueError, match="fingerprint"):
        tracking.record_evaluation(effective, summary, settings=settings, dataset_fingerprint="different")
    assert json.loads((directory / "metrics.json").read_text())["dataset_fingerprint"] == "abc123"


def test_comparison_rejects_unexpected_dataset_fingerprint(tmp_path):
    class Dataset:
        filenames = ["a.jpeg"]
        annotation_path = tmp_path / "annotations.json"

    Dataset.annotation_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        tracking.compare_trained_experiments(
            tmp_path / "old/best_model.pth", tmp_path / "new/best_model.pth",
            Dataset(), [0], expected_dataset_fingerprint="wrong",
        )


def test_qualitative_candidates_require_identical_images():
    def row(name, index, pq, predicted, gt):
        return {"filename": name, "index": index, "pq": pq,
                "pred_instances": predicted, "gt_instances": gt}

    comparison = {"original": {"per_image": [row("a", 0, 0.2, 1, 2), row("b", 1, 0, 1, 3)]},
                  "retrained": {"per_image": [row("a", 0, 0.5, 3, 2), row("b", 1, 0, 1, 3)]}}
    cases = tracking.qualitative_case_candidates(comparison)
    assert cases["retrained_improves"][0]["filename"] == "a"
    assert cases["both_fail"][0]["filename"] == "b"
    assert cases["fragmentation_candidates"][0]["filename"] == "a"
    bad = copy.deepcopy(comparison)
    bad["retrained"]["per_image"][1]["filename"] = "other"
    with pytest.raises(ValueError, match="aligned"):
        tracking.qualitative_case_candidates(bad)

    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    matching = {"matches": [{"gt_index": 0, "pred_index": 0, "iou": 1.0}],
                "unmatched_gt": [], "unmatched_pred": []}
    overlay = tracking.instance_error_overlay([mask], [mask], matching)
    assert tuple(overlay[1, 1]) == (0, 220, 0)
