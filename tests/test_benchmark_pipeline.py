"""Integration checks for Transformer logits and provisional scoring boundaries."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from filament_seg.benchmark import (
    coco_annotation_instances, make_partition, score_probability_maps,
    verify_independent_holdout,
)
from filament_seg.experiment_tracking import prepare_fresh_experiment
from filament_seg.transformer import build_binary_segformer
import filament_seg.transformer_train as transformer_train


def test_segformer_grayscale_logits_and_checkpoint_rebuild():
    from transformers import SegformerConfig

    config = SegformerConfig(
        num_labels=1, depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 32, 64],
        num_attention_heads=[1, 2, 4, 8], sr_ratios=[8, 4, 2, 1],
        decoder_hidden_size=16,
    ).to_dict()
    model, metadata = build_binary_segformer("b2", pretrained=False, hf_config=config)
    image = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        logits = model(image)
    assert logits.shape == image.shape and torch.isfinite(logits).all()
    assert metadata["input_adapter"].startswith("repeat grayscale")
    with pytest.raises(ValueError, match="grayscale"):
        model(torch.rand(1, 3, 64, 64))


def test_date_grouped_partition_and_cached_pq(tmp_path):
    class Dataset:
        filenames = ["20260101a.jpeg", "20260101b.jpeg", "20260102a.jpeg", "20260103a.jpeg"]
        annotation_path = tmp_path / "annotations.json"

        def iter_instances(self, index, annotation_set_index=0):
            assert annotation_set_index == 0
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:5, 2:5] = 1
            yield {"mask": mask}

    Dataset.annotation_path.write_text("{}", encoding="utf-8")
    dataset = Dataset()
    partition_path = tmp_path / "partition.json"
    partition = make_partition(dataset, [0, 1, 2, 3], partition_path)
    assert partition == make_partition(dataset, [0, 1, 2, 3], partition_path)
    assert set(partition["tuning_indices"]).isdisjoint(partition["internal_check_indices"])
    assert (0 in partition["tuning_indices"]) == (1 in partition["tuning_indices"])
    with pytest.raises(ValueError, match="incompatible"):
        make_partition(dataset, [0, 1, 2], partition_path)
    probability = np.zeros((8, 8), dtype=np.float32)
    probability[2:5, 2:5] = 0.9
    np.save(tmp_path / "0000.npy", probability)
    result = score_probability_maps(dataset, [0], tmp_path, threshold=0.7, min_area=1)
    assert result["summary"]["global_pq"] == 1.0
    assert result["summary"]["sq"] == result["summary"]["rq"] == 1.0
    assert (result["summary"]["tp"], result["summary"]["fp"], result["summary"]["fn"]) == (1, 0, 0)


def test_transformer_training_isolated_and_resumable(tmp_path, monkeypatch):
    monkeypatch.setattr(transformer_train, "build_binary_segformer",
                        lambda *args, **kwargs: (nn.Conv2d(1, 1, 1),
                                                 {"architecture": "SegFormer", "variant": "b2", "hf_config": {}}))
    base = {
        "project": {"experiment_name": "template", "seed": 42},
        "model": {"architecture": "SegFormer", "encoder_name": "mit_b2", "variant": "b2",
                  "encoder_weights": "pinned"},
        "training": {"checkpoint_dir": None, "epochs": 2, "batch_size": 2,
                     "gradient_accumulation_steps": 1, "learning_rate": 1e-3,
                     "weight_decay": 0, "scheduler": {"t_max": 2, "eta_min": 1e-6},
                     "mixed_precision": False, "loss": {"bce_weight": .5, "dice_weight": .5,
                                                         "smooth": 1, "pos_weight": 8},
                     "clip_grad_norm": 1, "early_stopping_patience": 5},
    }
    config = prepare_fresh_experiment(base, prefix="segformer_b2",
                                      checkpoint_root=tmp_path / "checkpoints",
                                      registry_path=tmp_path / "registry.json")
    samples = [{"image": torch.rand(1, 8, 8), "mask": (torch.rand(1, 8, 8) > .5).float()}
               for _ in range(2)]
    loader = DataLoader(samples, batch_size=2)
    config["training"]["epochs"] = 1
    # Keep the allocated YAML as the source of truth when changing total epochs.
    import yaml
    config_path = tmp_path / "checkpoints/segformer_b2_01/config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    first = transformer_train.train_segformer(config, loader, loader, device="cpu")
    assert first["status"] == "completed" and len(first["history"]) == 1
    with pytest.raises(FileExistsError):
        transformer_train.train_segformer(config, loader, loader, device="cpu", resume=False)
    config["training"]["epochs"] = 2
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    second = transformer_train.train_segformer(config, loader, loader, device="cpu", resume=True)
    assert second["status"] == "completed" and len(second["history"]) == 2


def test_verified_holdout_requires_physical_and_date_disjointness():
    assert verify_independent_holdout(["20270101a.jpeg"], ["20260101a.jpeg"], ["20260201a.jpeg"])["images"] == 1
    with pytest.raises(ValueError, match="overlaps"):
        verify_independent_holdout(["20260101a.jpeg"], ["20260101a.jpeg"], [])
    with pytest.raises(ValueError, match="capture dates"):
        verify_independent_holdout(["20260101b.jpeg"], ["20260101a.jpeg"], [])


def test_scoring_uses_coco_polygons_and_all_annotator_sets(tmp_path):
    class Dataset:
        filenames = ["20260101a.jpeg"]
        images_by_filename = {"20260101a.jpeg": [
            {"id": "a-20260101a", "height": 8, "width": 8},
            {"id": "b-20260101a", "height": 8, "width": 8},
        ]}
        annotations_by_image_id = {
            "a-20260101a": [{"segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]]}],
            "b-20260101a": [{"segmentation": [[5, 5, 7, 5, 7, 7, 5, 7]]}],
        }

    dataset = Dataset()
    first = coco_annotation_instances(dataset, 0, 0)[0]
    second = coco_annotation_instances(dataset, 0, 1)[0]
    assert first.shape == second.shape == (8, 8)
    assert not np.any(first & second)
    probability = first.astype(np.float32)
    np.save(tmp_path / "0000.npy", probability)
    result = score_probability_maps(dataset, [0], tmp_path, threshold=0.7, min_area=1)
    assert result["summary"]["global_pq"] == pytest.approx(0.5)
    assert (result["summary"]["tp"], result["summary"]["fp"], result["summary"]["fn"]) == (1, 1, 1)
    assert result["per_image"][0]["annotation_sets"] == 2
