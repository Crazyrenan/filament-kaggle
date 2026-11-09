"""Train or resume the next isolated U-Net v2 retrain experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.dataset import SolarFilamentDataset
from filament_seg.experiment_tracking import (
    RETRAIN_PREFIX,
    load_experiment_config,
    prepare_fresh_experiment,
    validation_fingerprint,
)
from filament_seg.preprocessing import get_train_transform, get_val_transform
from filament_seg.train import train_v2_model
from filament_seg.utils import load_config, set_seed


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested local retraining")
    template = load_config(ROOT / "configs/unet_resnet34_v2_retrain.yaml")
    set_seed(int(template["project"]["seed"]))
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    image_dir = base / "train_images"
    annotation_path = base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    train_dataset = SolarFilamentDataset(image_dir, annotation_path, transform=get_train_transform(512))
    val_dataset = SolarFilamentDataset(image_dir, annotation_path, transform=get_val_transform(512))
    assert len(train_dataset) == len(val_dataset) == 707
    indices = np.random.default_rng(42).permutation(707)
    train_indices, val_indices = indices[:565].tolist(), indices[565:].tolist()
    assert {train_dataset.filenames[i] for i in train_indices}.isdisjoint(
        {val_dataset.filenames[i] for i in val_indices}
    )
    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=4, shuffle=False, num_workers=0)
    fingerprint = validation_fingerprint(val_dataset, val_indices)
    candidates = sorted((ROOT / "checkpoints").glob(f"{RETRAIN_PREFIX}_*"))
    active = []
    for directory in candidates:
        if not (directory / "config.yaml").is_file():
            continue
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if metrics.get("dataset_fingerprint") != fingerprint:
            continue
        if (directory / "best_model.pth").is_file() and metrics.get("status") in {"completed", "early_stopped"}:
            print(f"Completed retrain exists: {directory}; training skipped", flush=True)
            return
        if (directory / "latest.pth").is_file():
            active.append(directory)
    if len(active) > 1:
        raise RuntimeError(f"Multiple resumable retrains need an explicit choice: {active}")
    if active:
        config = load_experiment_config(active[0].name)
        resume = True
    else:
        config = prepare_fresh_experiment(template, dataset_fingerprint=fingerprint)
        resume = False
    directory = Path(config["training"]["checkpoint_dir"])
    print(f"CNN experiment={directory.name} resume={resume} checkpoint_dir={directory}", flush=True)
    print(f"dataset_fingerprint={fingerprint} train=565 validation=142", flush=True)
    torch.cuda.reset_peak_memory_stats()
    result = train_v2_model(config, train_loader, val_loader, val_dataset, val_indices,
                            device="cuda", resume=resume)
    print(f"CNN result best_dice={result['best_val_dice']} epochs={len(result['history'])} "
          f"peak_allocated_MiB={torch.cuda.max_memory_allocated() / 2**20:.1f}", flush=True)
    print(f"best_checkpoint={result['best_checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
