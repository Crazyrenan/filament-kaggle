"""Model training and validation loops with metric tracking and checkpointing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from filament_seg.evaluate import compute_dice, compute_iou
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
