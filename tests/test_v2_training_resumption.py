"""Regression tests for U-Net ResNet34 v2 checkpoint resumption and smoke test isolation."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from filament_seg.model import build_model
from filament_seg.train import load_checkpoint, save_checkpoint, train_v2_model


class DummyFilamentDataset:
    def __init__(self, size: int = 4):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return {
            "image": torch.randn(1, 64, 64, dtype=torch.float32),
            "mask": (torch.rand(1, 64, 64) > 0.8).float(),
            "filename": f"dummy_{idx}.jpeg",
        }


@pytest.fixture
def dummy_loaders():
    train_ds = DummyFilamentDataset(4)
    val_ds = DummyFilamentDataset(2)
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)
    return train_loader, val_loader, val_ds


@pytest.fixture
def base_config(tmp_path):
    ckpt_dir = tmp_path / "checkpoints_v2"
    return {
        "project": {"name": "test", "experiment_name": "test-v2", "seed": 42},
        "model": {
            "architecture": "Unet",
            "encoder_name": "resnet34",
            "encoder_weights": None,  # Avoid downloading weights during fast unit tests
            "in_channels": 1,
            "classes": 1,
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "loss": {"bce_weight": 0.5, "dice_weight": 0.5, "smooth": 1.0, "pos_weight": 8.0},
            "scheduler": {"name": "CosineAnnealingLR", "t_max": 2, "eta_min": 1e-6},
            "mixed_precision": False,
            "clip_grad_norm": 1.0,
            "early_stopping_patience": 5,
            "checkpoint_dir": str(ckpt_dir),
        },
    }


def test_smoke_test_isolation_and_no_overwrite(dummy_loaders, base_config, tmp_path):
    """Verify smoke test only writes smoke_checkpoint.pth and never creates or modifies latest.pth or best_model.pth."""
    train_loader, val_loader, val_ds = dummy_loaders
    ckpt_dir = Path(base_config["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create a synthetic latest.pth
    latest_pth = ckpt_dir / "latest.pth"
    best_pth = ckpt_dir / "best_model.pth"
    sentinel_content = {"sentinel": "do_not_overwrite", "epoch": 99}
    torch.save(sentinel_content, latest_pth)
    torch.save(sentinel_content, best_pth)

    # Run smoke test with resume=True
    res = train_v2_model(
        config=base_config,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_ds,
        val_indices=[0, 1],
        device="cpu",
        resume=True,
        smoke_test=True,
    )

    assert res["status"] == "smoke_test_passed"
    smoke_ckpt = ckpt_dir / "smoke_checkpoint.pth"
    assert smoke_ckpt.is_file()

    # Verify latest.pth and best_model.pth were completely untouched
    loaded_latest = torch.load(latest_pth)
    loaded_best = torch.load(best_pth)
    assert loaded_latest.get("sentinel") == "do_not_overwrite"
    assert loaded_best.get("sentinel") == "do_not_overwrite"


def test_fresh_training_ignores_smoke_checkpoint(dummy_loaders, base_config):
    """Verify full training with resume=True initializes freshly when latest.pth does not exist, even if smoke_checkpoint.pth exists."""
    train_loader, val_loader, val_ds = dummy_loaders
    ckpt_dir = Path(base_config["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Place a smoke checkpoint in the directory
    smoke_ckpt = ckpt_dir / "smoke_checkpoint.pth"
    smoke_state = {
        "epoch": 0,
        "best_val_dice": 0.5,
        "history": [{"epoch": 0, "smoke_test": True}],
    }
    torch.save(smoke_state, smoke_ckpt)

    # Run 1 epoch with resume=True
    res = train_v2_model(
        config=base_config,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_ds,
        val_indices=[0, 1],
        device="cpu",
        resume=True,
        max_epochs=1,
        smoke_test=False,
    )

    latest_pth = ckpt_dir / "latest.pth"
    assert latest_pth.is_file()

    loaded = torch.load(latest_pth)
    # Must have trained epoch 1, NOT resumed smoke_test epoch 0
    assert loaded["epoch"] == 1
    assert len(loaded["history"]) == 1
    assert loaded["history"][0]["epoch"] == 1
    assert "smoke_test" not in loaded["history"][0]


def test_genuine_resumption_restores_all_training_components(dummy_loaders, base_config):
    """Verify that when latest.pth exists, full training resumes from that exact epoch and state."""
    train_loader, val_loader, val_ds = dummy_loaders
    ckpt_dir = Path(base_config["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy model and optimizer to get real state dicts
    temp_model = build_model(pretrained=False, in_channels=1, classes=1)
    temp_opt = torch.optim.AdamW(temp_model.parameters(), lr=1e-3)
    temp_sched = torch.optim.lr_scheduler.CosineAnnealingLR(temp_opt, T_max=10)
    temp_scaler = torch.amp.GradScaler("cpu", enabled=False)

    # Step scheduler twice to advance state
    temp_sched.step()
    temp_sched.step()

    prior_state = {
        "epoch": 2,
        "model_state_dict": temp_model.state_dict(),
        "optimizer_state_dict": temp_opt.state_dict(),
        "scheduler_state_dict": temp_sched.state_dict(),
        "scaler_state_dict": temp_scaler.state_dict(),
        "best_val_dice": 0.777,
        "epochs_without_improvement": 1,
        "history": [
            {"epoch": 1, "train_loss": 0.5, "val_dice": 0.70},
            {"epoch": 2, "train_loss": 0.4, "val_dice": 0.777},
        ],
    }
    latest_pth = ckpt_dir / "latest.pth"
    torch.save(prior_state, latest_pth)

    # Resume for 1 additional epoch (epoch 3)
    res = train_v2_model(
        config=base_config,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_ds,
        val_indices=[0, 1],
        device="cpu",
        resume=True,
        max_epochs=3,
        smoke_test=False,
    )

    loaded = torch.load(latest_pth)
    assert loaded["epoch"] == 3
    # Previous best metric of 0.777 was retained
    assert loaded["best_val_dice"] >= 0.777
    assert len(loaded["history"]) == 3
    assert loaded["history"][0]["epoch"] == 1
    assert loaded["history"][1]["epoch"] == 2
    assert loaded["history"][2]["epoch"] == 3


def test_load_checkpoint_restores_parameters_and_states(tmp_path):
    """Verify load_checkpoint faithfully restores weights, optimizer, scheduler, and scaler."""
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    sched.step()
    sched.step()
    sched.step()

    orig_weights = {k: v.clone() for k, v in model.state_dict().items()}
    state = {
        "epoch": 3,
        "model_state_dict": orig_weights,
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_dice": 0.8123,
    }
    ckpt_file = tmp_path / "checkpoint.pth"
    save_checkpoint(state, ckpt_file, is_best=False)

    # Initialize separate instances with different values
    new_model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
    with torch.no_grad():
        for p in new_model.parameters():
            p.zero_()
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-2)
    new_sched = torch.optim.lr_scheduler.CosineAnnealingLR(new_opt, T_max=20)
    new_scaler = torch.amp.GradScaler("cpu", enabled=False)

    loaded = load_checkpoint(
        ckpt_file,
        model=new_model,
        optimizer=new_opt,
        scheduler=new_sched,
        scaler=new_scaler,
        map_location="cpu",
    )

    assert loaded["epoch"] == 3
    assert loaded["best_val_dice"] == 0.8123
    for k, v in orig_weights.items():
        assert torch.equal(new_model.state_dict()[k], v)
    assert new_sched.last_epoch == 3
