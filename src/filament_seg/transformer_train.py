"""Resumable SegFormer training under the existing binary-mask protocol."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from filament_seg.experiment_tracking import load_experiment_config, record_training_progress
from filament_seg.losses import BCEDiceLoss
from filament_seg.train import load_checkpoint, save_checkpoint, validate_one_epoch
from filament_seg.transformer import build_binary_segformer


def train_segformer(
    config: dict[str, Any], train_loader: Any, val_loader: Any, *,
    device: str = "cuda", resume: bool = False,
) -> dict[str, Any]:
    cfg = config["training"]
    directory = Path(cfg["checkpoint_dir"])
    saved = load_experiment_config(directory.name, checkpoint_root=directory.parent)
    if saved != config:
        raise ValueError("Effective SegFormer config differs from the allocated config")
    latest, best = directory / "latest.pth", directory / "best_model.pth"
    if resume and not latest.is_file():
        raise FileNotFoundError(f"Cannot resume without {latest}")
    if not resume and (latest.exists() or best.exists()):
        raise FileExistsError(f"Fresh training would overwrite {directory}")
    dev = torch.device(device)
    if resume:
        raw = torch.load(latest, map_location="cpu", weights_only=False)
        if raw.get("purpose") != "full_training" or raw.get("experiment_name") != config["project"]["experiment_name"]:
            raise ValueError("Resume checkpoint is not from this full training experiment")
        model, model_config = build_binary_segformer(
            cfg.get("variant", config["model"]["variant"]), pretrained=False,
            hf_config=raw["model_config"]["hf_config"],
        )
    else:
        model, model_config = build_binary_segformer(config["model"]["variant"], pretrained=True)
    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    scheduler_cfg = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(scheduler_cfg["t_max"]), eta_min=float(scheduler_cfg["eta_min"])
    )
    use_amp = bool(cfg["mixed_precision"]) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_cfg = cfg["loss"]
    criterion = BCEDiceLoss(
        bce_weight=float(loss_cfg["bce_weight"]), dice_weight=float(loss_cfg["dice_weight"]),
        smooth=float(loss_cfg["smooth"]), pos_weight=float(loss_cfg["pos_weight"]),
    )
    history: list[dict[str, Any]] = []
    best_dice, stale_epochs, start_epoch = 0.0, 0, 1
    if resume:
        state = load_checkpoint(latest, model, optimizer, scheduler, scaler, map_location=dev)
        history = state["history"]
        best_dice = float(state["best_val_dice"])
        stale_epochs = int(state["epochs_without_improvement"])
        start_epoch = int(state["epoch"]) + 1
        if best_dice > 0 and not best.is_file():
            raise FileNotFoundError(f"Recorded best checkpoint is missing: {best}")
    epochs = int(cfg["epochs"])
    accumulation = int(cfg["gradient_accumulation_steps"])
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    patience = int(cfg["early_stopping_patience"])
    for epoch in range(start_epoch, epochs + 1):
        started = time.monotonic()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for batch_index, batch in enumerate(train_loader):
            images = batch["image"].to(dev, non_blocking=True)
            masks = batch["mask"].to(dev, non_blocking=True)
            with torch.amp.autocast(device_type=dev.type, enabled=use_amp):
                loss = criterion(model(images), masks)
            running += float(loss.detach())
            scaled_loss = loss / min(accumulation, len(train_loader) - (batch_index // accumulation) * accumulation)
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip_grad_norm"]))
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = validate_one_epoch(model, val_loader, criterion, device=dev)
        dice = float(validation["val_dice"])
        improved = not best.is_file() or dice > best_dice
        if improved:
            best_dice, stale_epochs = dice, 0
        else:
            stale_epochs += 1
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        entry = {
            "epoch": epoch, "train_loss": running / len(train_loader),
            "val_loss": float(validation["val_loss"]), "val_dice": dice,
            "val_iou": float(validation["val_iou"]), "lr": lr,
            "time_sec": time.monotonic() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else 0,
        }
        history.append(entry)
        state = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
            "history": history, "best_val_dice": best_dice,
            "epochs_without_improvement": stale_epochs, "purpose": "full_training",
            "experiment_name": config["project"]["experiment_name"],
            "model_config": model_config, "checkpoint_dir": str(directory),
        }
        save_checkpoint(state, latest, is_best=improved)
        record_training_progress(config, history, status="running")
        print(f"SegFormer epoch {epoch}/{epochs} Dice={dice:.5f} best={best_dice:.5f} "
              f"loss={entry['train_loss']:.5f} seconds={entry['time_sec']:.1f} "
              f"peak_MiB={entry['peak_allocated_mib']:.1f}", flush=True)
        if stale_epochs >= patience:
            break
    status = "early_stopped" if stale_epochs >= patience else "completed"
    record_training_progress(config, history, status=status)
    return {"status": status, "history": history, "best_checkpoint": str(best),
            "latest_checkpoint": str(latest), "best_val_dice": best_dice}
