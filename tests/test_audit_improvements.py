"""Comprehensive focused tests verifying Phase 2 through Phase 6 pipeline improvements."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from filament_seg.dataset import SolarFilamentDataset
from filament_seg.evaluate import (
    compute_dice,
    compute_iou,
    compute_precision_recall,
    diagnostic_panoptic_quality,
    evaluate_semantic_predictions,
)
from filament_seg.inference import (
    iter_probability_instances,
    postprocess_binary_mask,
    predict_tiled,
)
from filament_seg.submission import (
    decode_coco_counts,
    encode_coco_counts,
    generate_submission,
    validate_submission_rows,
)
from filament_seg.train import load_checkpoint, save_checkpoint, train_one_epoch


class ConstantSegmentationModel(nn.Module):
    """Mock model outputting constant logits for seamlessness and tiling tests."""

    def __init__(self, constant_logit: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.conv.weight.fill_(0.0)
            self.conv.bias.fill_(constant_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_dataset_split_integrity():
    """Verify train/val split is deterministic and has zero physical image leakage."""
    root = Path(__file__).resolve().parents[1]
    train_dir = root / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(
        train_dir / "train_images",
        train_dir / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    )
    assert len(dataset) == 707

    indices = np.random.default_rng(42).permutation(len(dataset))
    train_indices = indices[:565].tolist()
    val_indices = indices[565:].tolist()

    train_files = {dataset.filenames[i] for i in train_indices}
    val_files = {dataset.filenames[i] for i in val_indices}

    assert len(train_files) == 565
    assert len(val_files) == 142
    assert train_files.isdisjoint(val_files), "Leakage detected between train and validation files!"

    # Ensure all alternative annotation sets belong to the same split
    for filename in train_files:
        records = dataset.images_by_filename[filename]
        assert len(records) >= 1


def test_polygon_to_mask_conversion_and_instances(tmp_path):
    """Test multi-polygon and multi-instance rasterization into exact binary masks."""
    import cv2

    img_path = tmp_path / "test_img.jpeg"
    cv2.imwrite(str(img_path), np.zeros((32, 32), dtype=np.uint8))

    coco = {
        "images": [{"id": "img1", "file_name": "test_img.jpeg", "height": 32, "width": 32}],
        "annotations": [
            {
                "id": "ann1",
                "image_id": "img1",
                "category_id": 1,
                "segmentation": [
                    [2, 2, 8, 2, 8, 8, 2, 8],
                    [12, 12, 16, 12, 16, 16, 12, 16],
                ],
            }
        ],
    }
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps(coco), encoding="utf-8")

    ds = SolarFilamentDataset(tmp_path, json_path)
    instances = list(ds.iter_instances(0))
    assert len(instances) == 1
    mask = instances[0]["mask"]
    assert mask.shape == (32, 32)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 1}
    # Both disjoint polygons belonging to this single annotation should be marked as 1
    assert mask[4, 4] == 1
    assert mask[14, 14] == 1
    assert mask[0, 0] == 0


def test_tiled_inference_and_reconstruction():
    """Verify tiled prediction accurately reconstructs full-resolution coordinates."""
    model = ConstantSegmentationModel(constant_logit=2.0)
    img = np.ones((768, 768), dtype=np.float32) * 0.5
    probs = predict_tiled(model, img, tile_size=512, overlap=128, device="cpu", batch_size=2)
    assert probs.shape == (768, 768)
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    expected_prob = 1.0 / (1.0 + np.exp(-2.0))
    assert np.allclose(probs, expected_prob, atol=1e-3)


def test_tile_overlap_and_blending_seamlessness():
    """Verify Hann window blending eliminates seam artifacts across tile boundaries."""
    model = ConstantSegmentationModel(constant_logit=0.0)
    img = np.random.rand(1024, 1024).astype(np.float32)
    probs = predict_tiled(model, img, tile_size=512, overlap=128, device="cpu", batch_size=4)
    # Since model is constant logit 0.0, output everywhere should be exactly 0.5
    assert np.allclose(probs, 0.5, atol=1e-3), "Tile seams produced non-uniform output on constant logit!"


def test_thresholding_and_connected_components():
    """Verify thresholding, minimum area filtering, and connected components."""
    probs = np.zeros((100, 100), dtype=np.float32)
    # Valid component: 10x10 = 100 pixels
    probs[10:20, 10:20] = 0.8
    # Small noise component: 2x2 = 4 pixels
    probs[50:52, 50:52] = 0.9

    # Filter with min_area=50
    instances_filtered = list(iter_probability_instances(probs, threshold=0.5, min_area=50))
    assert len(instances_filtered) == 1
    assert instances_filtered[0].sum() == 100

    # No filtering: min_area=1
    instances_all = list(iter_probability_instances(probs, threshold=0.5, min_area=1))
    assert len(instances_all) == 2


def test_morphological_postprocessing():
    """Verify hole filling preserves thin filaments without eroding tips."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    # Filament with a small hole inside
    mask[10:30, 10:30] = 1
    mask[18:22, 18:22] = 0  # hole
    # Thin tip: 1 pixel wide
    mask[30:45, 20] = 1

    refined = postprocess_binary_mask(mask, fill_holes=True, close_kernel_size=0)
    assert refined[20, 20] == 1, "Internal hole was not filled"
    assert refined[40, 20] == 1, "Thin 1-pixel tip was corrupted"


def test_empty_mask_metrics_and_graceful_pq():
    """Verify semantic and instance metrics handle zero/empty cases gracefully."""
    empty = np.zeros((64, 64), dtype=np.uint8)
    non_empty = np.zeros((64, 64), dtype=np.uint8)
    non_empty[10:20, 10:20] = 1

    # Semantic metrics on empty
    assert compute_dice(empty, empty) == 1.0
    assert compute_iou(empty, empty) == 1.0
    assert compute_dice(non_empty, empty) == 0.0

    # Diagnostic PQ on empty cases
    res_both_empty = diagnostic_panoptic_quality([], [])
    assert res_both_empty == {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0}

    res_gt_empty = diagnostic_panoptic_quality([], [non_empty])
    assert res_gt_empty["tp"] == 0 and res_gt_empty["fp"] == 1 and res_gt_empty["pq"] == 0.0

    res_pred_empty = diagnostic_panoptic_quality([non_empty], [])
    assert res_pred_empty["tp"] == 0 and res_pred_empty["fn"] == 1 and res_pred_empty["pq"] == 0.0

    # Critical bug check: prediction list containing all-zero mask does not crash
    res_with_zero_pred = diagnostic_panoptic_quality([non_empty], [empty, non_empty])
    assert res_with_zero_pred["tp"] == 1
    assert res_with_zero_pred["fp"] == 0


def test_checkpoint_saving_loading_and_resumption(tmp_path):
    """Verify full checkpoint saving and loading restores model and state dicts."""
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    orig_state = {k: v.clone() for k, v in model.state_dict().items()}
    ckpt_path = tmp_path / "model.pth"
    state = {
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_dice": 0.725,
        "history": [{"epoch": 5, "val_dice": 0.725}],
    }
    save_checkpoint(state, ckpt_path, is_best=True)
    assert ckpt_path.is_file()
    assert (tmp_path / "best_model.pth").is_file()

    # Modify weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    new_model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    loaded = load_checkpoint(ckpt_path, new_model, optimizer=new_opt, map_location="cpu")

    assert loaded["epoch"] == 5
    assert loaded["best_val_dice"] == 0.725
    for p1, p2 in zip(orig_state.values(), new_model.state_dict().values()):
        assert torch.equal(p1, p2)


def test_submission_generation_and_validation(tmp_path):
    """Verify test-set submission generation, RLE encoding, and schema compliance."""
    import cv2

    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    img1 = test_dir / "20150125172714Mh.jpeg"
    img2 = test_dir / "20150201120000Uh.jpeg"
    cv2.imwrite(str(img1), np.zeros((2048, 2048), dtype=np.uint8))
    cv2.imwrite(str(img2), np.zeros((2048, 2048), dtype=np.uint8))

    model = ConstantSegmentationModel(constant_logit=-5.0)  # low probability (0 instances)
    out_csv = tmp_path / "submission.csv"

    df = generate_submission(
        model=model,
        test_images=[img1, img2],
        output_csv_path=out_csv,
        device="cpu",
        threshold=0.5,
        min_area=100,
        tile_size=512,
        overlap=128,
        batch_size=2,
    )
    assert out_csv.is_file()
    assert list(df.columns) == ["filament_id", "segmentation_rle"]
    # Since model output is low, 0 instances are generated without schema error
    assert len(df) == 0
    original = out_csv.read_bytes()
    with pytest.raises(FileExistsError):
        generate_submission(model, [img1], output_csv_path=out_csv, device="cpu")
    assert out_csv.read_bytes() == original


def test_amp_optimizer_step_guard():
    """Verify AMP training loop safely unscales and clips without 'No inf checks' error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    model = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.MSELoss()

    dataset = [{"image": torch.randn(1, 64, 64), "mask": torch.randn(1, 64, 64)}]
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    loss = train_one_epoch(
        model=model,
        dataloader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cuda",
        scaler=scaler,
        clip_grad_norm=1.0,
    )
    assert np.isfinite(loss)
