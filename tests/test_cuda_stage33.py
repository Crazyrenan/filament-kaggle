"""Bounded real-data CUDA optimization checks for the U-Net baseline."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from filament_seg.dataset import SolarFilamentDataset
from filament_seg.losses import BCEDiceLoss
from filament_seg.model import build_model
from filament_seg.preprocessing import get_train_transform


@pytest.mark.cuda
@pytest.mark.parametrize("batch_size", [2, 4])
def test_mixed_precision_real_batch_step(batch_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    torch.manual_seed(42)
    root = Path(__file__).resolve().parents[1]
    train_dir = root / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(
        train_dir / "train_images",
        train_dir / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        transform=get_train_transform(512),
    )
    batch = next(iter(DataLoader(dataset, batch_size=batch_size, num_workers=0)))
    assert batch["image"].shape == (batch_size, 1, 512, 512)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    model = build_model(pretrained=False).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    criterion = BCEDiceLoss(pos_weight=30)
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    before = next(model.parameters()).detach().clone()

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(images)
        loss = criterion(logits, masks)
    assert logits.shape == masks.shape
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, next(model.parameters()).detach())
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    print(f"CUDA batch={batch_size} loss={loss.item():.6f} "
          f"peak_allocated={allocated:.1f} MiB peak_reserved={reserved:.1f} MiB")
    del batch, dataset, images, masks, logits, loss, before, model, optimizer, scaler
    torch.cuda.empty_cache()
