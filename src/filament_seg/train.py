"""Model training and validation loops with metric tracking and checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    """Execute a single training epoch across the dataset.

    Args:
        model: PyTorch nn.Module.
        dataloader: PyTorch DataLoader supplying (images, masks).
        criterion: Loss function callable.
        optimizer: PyTorch optimizer.
        device: Target execution device.
        scaler: Optional torch.cuda.amp.GradScaler for mixed-precision.

    Returns:
        Average loss across all batches in the epoch.
    """
    import torch

    model.train()
    total_loss = 0.0
    batch_count = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(images)
                loss = criterion(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        batch_count += 1

    return total_loss / max(batch_count, 1)


def validate_one_epoch(
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
) -> dict[str, float]:
    """Evaluate model on the validation dataset.

    Args:
        model: PyTorch nn.Module.
        dataloader: Validation PyTorch DataLoader.
        criterion: Loss function callable.
        device: Target execution device.

    Returns:
        Dictionary of validation metrics (loss, mean dice, mean iou).
    """
    import torch

    from filament_seg.evaluate import compute_dice, compute_iou

    model.eval()
    total_loss = 0.0
    ious: list[float] = []
    dices: list[float] = []
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            preds = model(images)
            loss = criterion(preds, masks)
            total_loss += float(loss.item())
            batch_count += 1

            probs = torch.sigmoid(preds).cpu().numpy()
            targets = masks.cpu().numpy()

            for p_arr, t_arr in zip(probs, targets):
                p_bin = (p_arr > 0.5).astype(int)
                t_bin = (t_arr > 0.5).astype(int)
                ious.append(compute_iou(p_bin, t_bin))
                dices.append(compute_dice(p_bin, t_bin))

    return {
        "val_loss": total_loss / max(batch_count, 1),
        "val_iou": float(sum(ious) / max(len(ious), 1)),
        "val_dice": float(sum(dices) / max(len(dices), 1)),
    }


def save_checkpoint(
    state: dict[str, Any],
    filepath: str | Path,
    is_best: bool = False,
) -> None:
    """Persist model checkpoint to disk.

    Args:
        state: State dictionary containing model_state_dict, optimizer, epoch, etc.
        filepath: Target checkpoint path.
        is_best: If True, also save a copy to 'best_model.pth'.
    """
    import torch

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info("Saved checkpoint to: %s", path)

    if is_best:
        best_path = path.parent / "best_model.pth"
        torch.save(state, best_path)
        logger.info("Updated best model checkpoint at: %s", best_path)
