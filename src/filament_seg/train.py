
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
) -> float:
    """Train the model for one epoch and return average batch loss."""

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
            raise FloatingPointError(
                f"Non-finite training loss: {loss.item()}"
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.detach().item()
        batch_count += 1

    if batch_count == 0:
        raise ValueError("Training DataLoader is empty.")

    average_loss = total_loss / batch_count

    logger.info(
        "Training completed: batches=%d, loss=%.6f",
        batch_count,
        average_loss,
    )

    return average_loss


@torch.inference_mode()
def validate_one_epoch(
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
) -> dict[str, float]:
    """Evaluate foreground loss, Dice, and IoU on validation patches."""

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
            raise FloatingPointError(
                f"Non-finite validation loss: {loss.item()}"
            )

        total_loss += loss.item()
        batch_count += 1

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).to(torch.uint8)

        targets = (
            masks > 0.5
        ).to(torch.uint8)

        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        for pred_mask, target_mask in zip(
            predictions_np,
            targets_np,
        ):
            pred_mask = np.squeeze(pred_mask)
            target_mask = np.squeeze(target_mask)

            dice = compute_dice(
                pred_mask,
                target_mask,
            )
            iou = compute_iou(
                pred_mask,
                target_mask,
            )

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
        "Validation completed: loss=%.6f, Dice=%.6f, IoU=%.6f",
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
    """Save a checkpoint and optionally update the best-model checkpoint."""

    path = Path(filepath)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f"{path.name}.tmp"
    )

    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    logger.info(
        "Saved checkpoint to: %s",
        path,
    )

    if is_best:
        best_path = path.parent / "best_model.pth"
        best_temporary_path = best_path.with_name(
            f"{best_path.name}.tmp"
        )

        try:
            torch.save(state, best_temporary_path)
            os.replace(best_temporary_path, best_path)
        finally:
            if best_temporary_path.exists():
                best_temporary_path.unlink()

        logger.info(
            "Updated best model checkpoint at: %s",
            best_path,
        )
