"""Isolated, resumable v2-initialized hard-negative fine-tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.hard_negative import HardNegativePatchDataset
from filament_seg.losses import BCEDiceLoss
from filament_seg.model import build_model
from filament_seg.preprocessing import get_val_transform
from filament_seg.train import load_checkpoint, save_checkpoint, train_one_epoch, validate_one_epoch
from filament_seg.experiment_tracking import _update_registry

EXPERIMENT = "unet_resnet34_v3_hard_negative_01"
CONFIG = ROOT / "configs" / f"{EXPERIMENT}.yaml"
DIRECTORY = ROOT / "checkpoints" / EXPERIMENT


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_data(config: dict) -> tuple[HardNegativePatchDataset, Subset, dict]:
    source = config["source"]
    mining_path = ROOT / source["mining_artifact"]
    mining = json.loads(mining_path.read_text(encoding="utf-8"))
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    annotation = base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    image_dir = base / "train_images"
    dataset = SolarFilamentDataset(image_dir, annotation)
    validation = SolarFilamentDataset(image_dir, annotation, transform=get_val_transform(512))
    partition = json.loads((ROOT / "reports/validation_partition_v1.json").read_text(encoding="utf-8"))
    training = np.random.default_rng(42).permutation(len(dataset))[:565].tolist()
    tuning = partition["tuning_indices"]
    internal = partition["internal_check_indices"]
    train_names = {dataset.filenames[i] for i in training}
    excluded_names = {dataset.filenames[i] for i in tuning + internal}
    test_dir = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    test_names = {path.name for path in test_dir.glob("*") if path.is_file()}
    if len(training) != 565 or len(tuning) != 71 or len(internal) != 71 or train_names & (excluded_names | test_names):
        raise ValueError("Training, tuning, internal-check, or test filename overlap")
    if set(tuning + internal) != set(range(len(dataset))) - set(training):
        raise ValueError("Validation partition differs from the mining training split")
    if mining["training_indices"] != training or set(mining["training_filenames"]) != train_names:
        raise ValueError("Mining artifact does not match the training split")
    if set(mining["validation_filenames_excluded"]) != excluded_names:
        raise ValueError("Mining exclusion list differs from validation")
    if digest(annotation) != mining["annotation_sha256"] or mining["source_checkpoint_sha256"] != source["expected_sha256"]:
        raise ValueError("Mining annotation or parent checkpoint provenance changed")
    minimum_radius = float(config["data"]["sampling"]["hard_negative_min_radial_distance"])
    selected = {filename: [row for row in rows if math.hypot(row["centroid_x"] - 1024,
                 row["centroid_y"] - 1024) / 1024 >= minimum_radius]
                for filename, rows in mining["patches_by_filename"].items()}
    selected = {name: rows for name, rows in selected.items() if rows}
    if not selected or not set(selected).issubset(train_names):
        raise ValueError("No training-only outer-disk hard negatives")
    patches = HardNegativePatchDataset(dataset, training, selected,
                                       seed=int(config["project"]["seed"]), patch_size=512)
    return patches, Subset(validation, tuning), {"mining_sha256": digest(mining_path),
        "annotation_sha256": digest(annotation), "partition_sha256": digest(ROOT / "reports/validation_partition_v1.json"),
        "training_images": len(training), "tuning_images": len(tuning),
        "hard_negative_images": len(selected), "hard_negative_patches": sum(map(len, selected.values()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this experiment")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["project"]["experiment_id"] != EXPERIMENT:
        raise ValueError("Experiment ID mismatch")
    parent = ROOT / config["source"]["checkpoint"]
    parent_hash = digest(parent)
    if parent_hash != config["source"]["expected_sha256"]:
        raise ValueError("Historical v2 checkpoint hash changed")
    train_data, val_data, provenance = prepare_data(config)
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in [CONFIG,
        ROOT / "scripts/run_hard_negative_v3.py", ROOT / "src/filament_seg/hard_negative.py",
        ROOT / "src/filament_seg/train.py", ROOT / "src/filament_seg/dataset.py"]}
    provenance.update({"parent_checkpoint_sha256": parent_hash, "source_hashes": source_hashes})
    if args.resume:
        latest = DIRECTORY / "latest.pth"
        if not latest.is_file():
            raise FileNotFoundError(f"Genuine latest checkpoint missing: {latest}")
        saved_provenance = json.loads((DIRECTORY / "provenance.json").read_text(encoding="utf-8"))
        if saved_provenance != provenance:
            raise ValueError("Resume refused: source, config, annotation, or mining provenance changed")
    else:
        DIRECTORY.mkdir(parents=True, exist_ok=False)
        (DIRECTORY / "config.yaml").write_bytes(CONFIG.read_bytes())
        atomic_json(DIRECTORY / "provenance.json", provenance)
    seed = int(config["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(pretrained=False).cuda()
    groups = [{"params": list(model.encoder.parameters()), "lr": float(config["training"]["encoder_learning_rate"])},
              {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
               "lr": float(config["training"]["decoder_learning_rate"])}]
    training_cfg = config["training"]
    optimizer = torch.optim.AdamW(groups, weight_decay=float(training_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training_cfg["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training_cfg["mixed_precision"]))
    criterion = BCEDiceLoss(bce_weight=float(training_cfg["bce_weight"]),
                            dice_weight=float(training_cfg["dice_weight"]),
                            pos_weight=float(training_cfg["positive_class_weight"]))
    if args.resume:
        state = load_checkpoint(DIRECTORY / "latest.pth", model, optimizer, scheduler, scaler, map_location="cuda")
        if state["provenance"] != provenance or state["experiment_id"] != EXPERIMENT:
            raise ValueError("Resume checkpoint provenance mismatch")
        torch.set_rng_state(state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])
        start = int(state["epoch"]) + 1
        best = float(state["best_val_dice"])
        best_epoch = int(state["best_epoch"])
        stale = int(state["epochs_without_improvement"])
        history = state["history"]
    else:
        load_checkpoint(parent, model, map_location="cuda")
        start, best, best_epoch, stale, history = 1, -1.0, 0, 0, []
    def checkpoint(epoch: int, *, improved: bool = False) -> None:
        state = {"epoch": epoch, "experiment_id": EXPERIMENT, "purpose": "training",
                 "model_config": {"architecture": "Unet", "encoder_name": "resnet34", "in_channels": 1, "classes": 1},
                 "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                 "best_val_dice": best, "best_epoch": best_epoch, "epochs_without_improvement": stale,
                 "history": history, "provenance": provenance,
                 "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
                 "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate()}
        save_checkpoint(state, DIRECTORY / "latest.pth", is_best=improved)
        atomic_json(DIRECTORY / "history.json", history)
        atomic_json(DIRECTORY / "metrics.json", {"status": "running", "epoch": epoch,
                    "best_epoch": best_epoch, "best_val_dice": best, **provenance})
    if not args.resume:
        checkpoint(0)
    validation_loader = DataLoader(val_data, batch_size=4, num_workers=0, pin_memory=True)
    for epoch in range(start, int(training_cfg["epochs"]) + 1):
        train_data.set_epoch(epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        train_loader = DataLoader(train_data, batch_size=int(training_cfg["batch_size"]), shuffle=True,
                                  generator=generator, num_workers=0, pin_memory=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss = train_one_epoch(model, train_loader, criterion, optimizer, "cuda", scaler,
                               float(training_cfg["clip_grad_norm"]))
        metrics = validate_one_epoch(model, validation_loader, criterion, "cuda")
        scheduler.step()
        improved = metrics["val_dice"] > best + 1e-5
        if improved:
            best, best_epoch, stale = metrics["val_dice"], epoch, 0
        else:
            stale += 1
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=False)
        row = {"epoch": epoch, "train_loss": loss, **metrics, "duration_seconds": time.perf_counter()-t0,
               "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
               "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
               "gpu_utilization_and_memory_end_epoch": gpu.stdout.strip() if gpu.returncode == 0 else None,
               "sampling_counts": dict(train_data.sampling_counts), "encoder_lr": optimizer.param_groups[0]["lr"],
               "decoder_lr": optimizer.param_groups[1]["lr"]}
        history.append(row)
        checkpoint(epoch, improved=improved)
        _update_registry(ROOT / "reports/experiment_registry.json", EXPERIMENT,
                         {"status": "training", "checkpoint_dir": str(DIRECTORY), "best_epoch": best_epoch,
                          "best_val_dice": best, "parent_checkpoint_sha256": parent_hash,
                          "latest_checkpoint": str(DIRECTORY / "latest.pth"),
                          "best_checkpoint": str(DIRECTORY / "best_model.pth")})
        print("EPOCH_COMPLETE", json.dumps(row), flush=True)
        if stale >= int(training_cfg["early_stopping_patience"]):
            print(f"EARLY_STOP epoch={epoch}", flush=True)
            break
    atomic_json(DIRECTORY / "metrics.json", {"status": "training_complete", "epochs_run": len(history),
                "best_epoch": best_epoch, "best_val_dice": best, **provenance})
    _update_registry(ROOT / "reports/experiment_registry.json", EXPERIMENT,
                     {"status": "training_complete", "best_epoch": best_epoch, "best_val_dice": best,
                      "best_checkpoint_sha256": digest(DIRECTORY / "best_model.pth")})
    print(f"TRAINING_COMPLETE best_epoch={best_epoch} best_dice={best:.6f}", flush=True)


if __name__ == "__main__":
    main()
