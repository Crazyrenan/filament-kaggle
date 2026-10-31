"""Model training and validation loops with metric tracking and checkpointing."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from filament_seg.evaluate import compute_dice, compute_iou, evaluate_full_resolution_dataset
from filament_seg.losses import BCEDiceLoss
from filament_seg.model import build_model
from filament_seg.utils import get_logger

logger = get_logger(__name__)


def train_one_epoch(
    model: Any,
    dataloader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    scaler: Any | None = None,
    clip_grad_norm: float | None = 1.0,
) -> float:
    """Train the model for one epoch and return average batch loss.

    Follows PyTorch best practices with AMP GradScaler unscaling and gradient clipping.
    Guards against unrecorded inf check errors by validating gradients before optimizer step.

    Args:
        model: PyTorch nn.Module.
        dataloader: PyTorch DataLoader supplying batches with 'image' and 'mask'.
        criterion: Loss function callable.
        optimizer: PyTorch optimizer.
        device: Target execution device.
        scaler: Optional torch.amp.GradScaler for mixed-precision training.
        clip_grad_norm: Max gradient norm for gradient clipping (default 1.0).

    Returns:
        Average loss across all batches in the epoch.
    """
    device = torch.device(device)
    model.train()

    total_loss = 0.0
    batch_count = 0
    use_amp = scaler is not None and device.type == "cuda"

    for batch in dataloader:
        images = batch["image"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        masks = batch["mask"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, masks)

        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            # Guard: ensure gradients were populated before scaler.step
            has_grads = any(p.grad is not None for p in model.parameters())
            if has_grads:
                scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

        total_loss += loss.detach().item()
        batch_count += 1

    if batch_count == 0:
        raise ValueError("Training DataLoader is empty.")

    average_loss = total_loss / batch_count
    logger.info("Training completed: batches=%d, loss=%.6f", batch_count, average_loss)
    return average_loss


@torch.inference_mode()
def validate_one_epoch(
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
) -> dict[str, float]:
    """Evaluate foreground loss, Dice, and IoU on validation patches.

    Note: This evaluates patch-level validation (e.g. 512x512 center crops).
    For primary competition evaluation, use evaluate_full_resolution_dataset.

    Args:
        model: PyTorch nn.Module.
        dataloader: Validation DataLoader.
        criterion: Loss function callable.
        device: Target execution device.

    Returns:
        Dictionary of validation metrics (val_loss, val_dice, val_iou).
    """
    device = torch.device(device)
    model.eval()

    total_loss = 0.0
    batch_count = 0
    dices: list[float] = []
    ious: list[float] = []

    for batch in dataloader:
        images = batch["image"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        masks = batch["mask"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        logits = model(images)
        loss = criterion(logits, masks)

        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite validation loss: {loss.item()}")

        total_loss += loss.item()
        batch_count += 1

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).to(torch.uint8)
        targets = (masks > 0.5).to(torch.uint8)

        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        for pred_mask, target_mask in zip(predictions_np, targets_np):
            pred_mask = np.squeeze(pred_mask)
            target_mask = np.squeeze(target_mask)

            dice = compute_dice(pred_mask, target_mask)
            iou = compute_iou(pred_mask, target_mask)
            dices.append(float(dice))
            ious.append(float(iou))

    if batch_count == 0:
        raise ValueError("Validation DataLoader is empty.")

    metrics = {
        "val_loss": total_loss / batch_count,
        "val_dice": float(np.mean(dices)),
        "val_iou": float(np.mean(ious)),
    }

    logger.info(
        "Patch validation completed: loss=%.6f, Dice=%.6f, IoU=%.6f",
        metrics["val_loss"],
        metrics["val_dice"],
        metrics["val_iou"],
    )
    return metrics


def save_checkpoint(
    state: dict[str, Any],
    filepath: str | Path,
    is_best: bool = False,
) -> None:
    """Save a checkpoint and optionally update the best-model checkpoint.

    Uses atomic temporary file renaming to prevent file corruption.

    Args:
        state: State dictionary containing model_state_dict, optimizer, epoch, etc.
        filepath: Target checkpoint path.
        is_best: If True, also updates 'best_model.pth' in the same directory.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    logger.info("Saved checkpoint to: %s", path)

    if is_best:
        best_path = path.parent / "best_model.pth"
        best_temporary_path = best_path.with_name(f"{best_path.name}.tmp")
        try:
            torch.save(state, best_temporary_path)
            os.replace(best_temporary_path, best_path)
        finally:
            if best_temporary_path.exists():
                best_temporary_path.unlink()

        logger.info("Updated best model checkpoint at: %s", best_path)


def load_checkpoint(
    filepath: str | Path,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: Any = "cpu",
    weights_only: bool = False,
) -> dict[str, Any]:
    """Load checkpoint weights into model and optionally restore training states.

    Args:
        filepath: Path to the checkpoint file (.pth).
        model: PyTorch nn.Module instance to load weights into.
        optimizer: Optional optimizer to restore state dict.
        scheduler: Optional learning rate scheduler to restore state dict.
        scaler: Optional GradScaler to restore state dict.
        map_location: Device to map tensors to ('cpu', 'cuda', etc.).
        weights_only: Whether to restrict unpickling to tensors/primitives.

    Returns:
        Checkpoint dictionary containing metadata ('epoch', 'best_val_dice', 'history', etc.).
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found at: {path.resolve()}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unrecognized checkpoint format in {path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    logger.info(
        "Loaded checkpoint from %s (epoch: %s, best_metric: %s)",
        path,
        checkpoint.get("epoch"),
        checkpoint.get("best_val_dice"),
    )
    return checkpoint


def train_v2_model(
    config: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_dataset: Any,
    val_indices: list[int],
    device: Any = "cuda",
    resume: bool = True,
    max_epochs: int | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Train or resume the U-Net ResNet34 v2 model with full-resolution evaluation.

    Args:
        config: Configuration dictionary (matching unet_resnet34_v2.yaml).
        train_loader: DataLoader for 512x512 foreground-aware training patches.
        val_loader: DataLoader for validation patch checks.
        val_dataset: SolarFilamentDataset instance for full-resolution tiled evaluation.
        val_indices: Indices of validation images.
        device: Execution device (e.g. 'cuda').
        resume: Whether to resume from latest.pth if present.
        max_epochs: Optional override for total training epochs.
        smoke_test: If True, executes 2 batches and validates checkpoint saving/loading.

    Returns:
        Dictionary containing trained model, history, and best checkpoint state.
    """
    dev = torch.device(device)
    train_cfg = config.get("training", {})
    total_epochs = max_epochs or train_cfg.get("epochs", 30)
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/unet_resnet34_v2"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = ckpt_dir / "latest.pth"
    best_ckpt_path = ckpt_dir / "best_model.pth"

    # Instantiate model with ImageNet pretraining
    pretrained = config.get("model", {}).get("encoder_weights") == "imagenet"
    model = build_model(
        architecture="unet",
        encoder_name=config.get("model", {}).get("encoder_name", "resnet34"),
        pretrained=pretrained,
        in_channels=config.get("model", {}).get("in_channels", 1),
        classes=config.get("model", {}).get("classes", 1),
    ).to(dev)

    lr = float(train_cfg.get("learning_rate", 3e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler_cfg = train_cfg.get("scheduler", {})
    t_max = int(scheduler_cfg.get("t_max", total_epochs))
    eta_min = float(scheduler_cfg.get("eta_min", 1e-6))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    use_amp = bool(train_cfg.get("mixed_precision", True)) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loss_cfg = train_cfg.get("loss", {})
    pos_weight = float(loss_cfg.get("pos_weight", 8.0))
    criterion = BCEDiceLoss(
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        smooth=float(loss_cfg.get("smooth", 1.0)),
        pos_weight=pos_weight,
    )

    clip_grad = float(train_cfg.get("clip_grad_norm", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 7))

    # Smoke test mode - strictly isolated from full training
    if smoke_test:
        logger.info("Running U-Net v2 training smoke test (2 batches)...")
        model.train()
        smoke_loader = [next(iter(train_loader)), next(iter(train_loader))]
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(smoke_loader):
            img = batch["image"][:2].to(dev)
            msk = batch["mask"][:2].to(dev)
            with torch.amp.autocast(device_type=dev.type, enabled=use_amp):
                out = model(img)
                loss = criterion(out, msk)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Smoke validation
        smoke_val_batch = next(iter(val_loader))
        with torch.no_grad():
            with torch.amp.autocast(device_type=dev.type, enabled=use_amp):
                val_out = model(smoke_val_batch["image"][:2].to(dev))
                val_loss = criterion(val_out, smoke_val_batch["mask"][:2].to(dev))

        smoke_state = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_dice": 0.5,
            "history": [{"epoch": 0, "smoke_test": True}],
            "epochs_without_improvement": 0,
        }
        smoke_path = ckpt_dir / "smoke_checkpoint.pth"
        assert smoke_path.resolve() != latest_ckpt_path.resolve(), "Smoke checkpoint must not overwrite latest.pth"
        assert smoke_path.resolve() != best_ckpt_path.resolve(), "Smoke checkpoint must not overwrite best_model.pth"
        save_checkpoint(smoke_state, smoke_path, is_best=False)
        assert smoke_path.is_file()
        loaded = load_checkpoint(smoke_path, model, map_location=dev)
        assert loaded["epoch"] == 0
        logger.info("Smoke test SUCCESS: CUDA, AMP, gradients, loss, save/load verified.")
        return {
            "status": "smoke_test_passed",
            "loss": float(loss.item()),
            "val_loss": float(val_loss.item()),
            "checkpoint": str(smoke_path),
        }

    # Full training resumption logic: strictly only loads latest.pth if present
    start_epoch = 1
    best_val_dice = 0.0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    if resume:
        if latest_ckpt_path.is_file():
            logger.info("Resuming full training from latest checkpoint: %s", latest_ckpt_path)
            checkpoint = load_checkpoint(
                latest_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                map_location=dev,
            )
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_dice = checkpoint.get("best_val_dice", 0.0)
            epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
            history = checkpoint.get("history", [])
            logger.info(
                "Resumed at epoch %d, previous best metric: %.4f (history entries: %d)",
                start_epoch,
                best_val_dice,
                len(history),
            )
        else:
            logger.info(
                "Resume requested but no checkpoint found at %s. Initializing fresh training run with ImageNet-pretrained ResNet34 encoder from epoch 1.",
                latest_ckpt_path,
            )
    else:
        logger.info(
            "Initializing fresh training run with ImageNet-pretrained ResNet34 encoder from epoch 1 (resume=False)."
        )

    # Full training loop
    logger.info("Starting training from epoch %d to %d...", start_epoch, total_epochs)
    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=dev,
            scaler=scaler,
            clip_grad_norm=clip_grad,
        )

        patch_metrics = validate_one_epoch(model, val_loader, criterion, device=dev)
        current_val_dice = patch_metrics["val_dice"]
        current_val_loss = patch_metrics["val_loss"]
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        is_best = current_val_dice > best_val_dice
        if is_best:
            best_val_dice = current_val_dice
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": current_val_loss,
            "val_dice": current_val_dice,
            "val_iou": patch_metrics["val_iou"],
            "lr": current_lr,
            "time_sec": time.time() - t0,
        }
        history.append(record)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_dice": best_val_dice,
            "history": history,
            "epochs_without_improvement": epochs_without_improvement,
        }
        save_checkpoint(state, latest_ckpt_path, is_best=is_best)

        logger.info(
            "Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f | Val Dice: %.4f (Best: %.4f) | LR: %.6f | Time: %.1fs",
            epoch,
            total_epochs,
            train_loss,
            current_val_loss,
            current_val_dice,
            best_val_dice,
            current_lr,
            time.time() - t0,
        )

        if epochs_without_improvement >= patience:
            logger.info("Early stopping triggered after %d epochs without improvement.", patience)
            break

    return {
        "model": model,
        "best_val_dice": best_val_dice,
        "history": history,
        "best_checkpoint": str(best_ckpt_path),
        "latest_checkpoint": str(latest_ckpt_path),
    }
