"""Checks for the foreground U-Net baseline's learning contract."""

import torch

from filament_seg.evaluate import evaluate_semantic_predictions
from filament_seg.losses import BCEDiceLoss, DiceLoss
from filament_seg.model import build_model, build_segmentation_model


def test_unet_grayscale_logits_and_gradient():
    torch.set_num_threads(4)
    model = build_model(pretrained=False)
    inputs = torch.rand(1, 1, 64, 64)
    targets = torch.zeros_like(inputs)
    targets[:, :, 16:48, 16:48] = 1

    logits = model(inputs)
    loss = BCEDiceLoss()(logits, targets)
    loss.backward()

    assert logits.shape == targets.shape
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss) and loss >= 0
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_losses_improve_with_correct_logits_and_empty_targets():
    criterion = BCEDiceLoss()
    dice = DiceLoss()
    targets = torch.zeros(2, 1, 32, 32)
    targets[0, :, 8:24, 8:24] = 1
    wrong = torch.zeros_like(targets)
    perfect = torch.where(targets.bool(), 20.0, -20.0)

    assert 0 <= criterion(perfect, targets) < criterion(wrong, targets)
    assert criterion(perfect, targets) < 1e-4
    assert dice(perfect, targets) < 1e-4

    empty = torch.zeros(1, 1, 32, 32)
    assert criterion(torch.full_like(empty, -20.0), empty) < 1e-4
    assert criterion(torch.full_like(empty, 20.0), empty) > 0
    weighted = BCEDiceLoss(bce_weight=1, dice_weight=0, pos_weight=30)
    plain = BCEDiceLoss(bce_weight=1, dice_weight=0)
    assert weighted(torch.full_like(empty, -2.0), torch.ones_like(empty)) > plain(
        torch.full_like(empty, -2.0), torch.ones_like(empty)
    )


def test_cached_rgb_encoder_is_adapted_to_grayscale_without_download(monkeypatch):
    import huggingface_hub
    import safetensors.torch

    rgb_model = build_segmentation_model(in_channels=3)
    state = rgb_model.encoder.state_dict()
    expected = state["conv1.weight"].sum(dim=1, keepdim=True)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: "cached")
    monkeypatch.setattr(safetensors.torch, "load_file", lambda *a, **k: state)

    grayscale_model = build_model(pretrained=True)
    actual = grayscale_model.encoder.conv1.weight.detach()
    assert actual.shape[1] == 1
    assert torch.equal(actual, expected)


def test_foreground_metrics_are_diagnostic_and_define_empty_case():
    import numpy as np

    empty = np.zeros((4, 4), dtype=np.uint8)
    scores = evaluate_semantic_predictions([empty.astype(np.float32)], [empty])
    assert scores["mean_dice"] == scores["mean_iou"] == 1.0
    assert scores["mean_precision"] == scores["mean_recall"] == 0.0

    target = empty.copy()
    target[0, 0] = 1
    probabilities = np.zeros((4, 4), dtype=np.float32)
    probabilities[0, 0] = 0.5
    scores = evaluate_semantic_predictions([probabilities], [target], threshold=0.5)
    assert scores["mean_dice"] == scores["mean_iou"] == 1.0
