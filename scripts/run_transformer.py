"""Profile B2 on local CUDA, then train/resume an isolated SegFormer."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.dataset import SolarFilamentDataset
from filament_seg.experiment_tracking import (
    load_experiment_config, prepare_fresh_experiment, validation_fingerprint,
)
from filament_seg.losses import BCEDiceLoss
from filament_seg.preprocessing import get_train_transform, get_val_transform
from filament_seg.transformer import SEGFORMER_SOURCES, build_binary_segformer
from filament_seg.transformer_train import train_segformer
from filament_seg.utils import load_config, set_seed


def ensure_pretrained(variant: str) -> str:
    from huggingface_hub import hf_hub_download

    repo, revision = SEGFORMER_SOURCES[variant]
    hf_hub_download(repo, "config.json", revision=revision)
    file = hf_hub_download(repo, "pytorch_model.bin", revision=revision)
    digest = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    return digest


def profile(variant: str, microbatch: int) -> dict:
    model, _ = build_binary_segformer(variant, pretrained=True)
    model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=8.0)
    torch.cuda.reset_peak_memory_stats()
    try:
        image = torch.rand(microbatch, 1, 512, 512, device="cuda")
        target = (torch.rand_like(image) > 0.98).float()
        with torch.amp.autocast("cuda"):
            loss = criterion(model(image), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        return {"variant": variant, "microbatch": microbatch,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "loss": float(loss.detach())}
    finally:
        del model, optimizer, criterion
        gc.collect()
        torch.cuda.empty_cache()


def choose_profile() -> tuple[dict, list[dict]]:
    attempts = []
    for variant in ("b2", "b1", "b0"):
        weights_hash = ensure_pretrained(variant)
        for microbatch in (4, 2, 1):
            try:
                result = profile(variant, microbatch)
                result["pretrained_sha256"] = weights_hash
                attempts.append(result)
                return result, attempts
            except torch.cuda.OutOfMemoryError as exc:
                attempts.append({"variant": variant, "microbatch": microbatch,
                                 "error": "CUDA out of memory", "detail": str(exc)[:400]})
                gc.collect()
                torch.cuda.empty_cache()
    raise RuntimeError(f"No SegFormer variant fits CUDA: {attempts}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; requested Transformer training cannot start")
    template = load_config(ROOT / "configs/segformer_b2.yaml")
    set_seed(int(template["project"]["seed"]))
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    images = base / "train_images"
    annotations = base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    train_dataset = SolarFilamentDataset(images, annotations, transform=get_train_transform(512))
    val_dataset = SolarFilamentDataset(images, annotations, transform=get_val_transform(512))
    assert len(train_dataset) == len(val_dataset) == 707
    indices = np.random.default_rng(42).permutation(707)
    train_indices, val_indices = indices[:565].tolist(), indices[565:].tolist()
    assert {train_dataset.filenames[i] for i in train_indices}.isdisjoint(
        {val_dataset.filenames[i] for i in val_indices}
    )
    fingerprint = validation_fingerprint(val_dataset, val_indices)
    candidates = sorted((ROOT / "checkpoints").glob("segformer_*"))
    resumable = []
    for directory in candidates:
        if not (directory / "config.yaml").is_file():
            continue
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if metrics.get("dataset_fingerprint") != fingerprint:
            continue
        if (directory / "best_model.pth").is_file() and metrics.get("status") in {"completed", "early_stopped"}:
            print(f"Completed Transformer exists: {directory}; training skipped", flush=True)
            return
        if (directory / "latest.pth").is_file():
            resumable.append(directory)
    if len(resumable) > 1:
        raise RuntimeError(f"Multiple resumable Transformers need an explicit choice: {resumable}")
    if resumable:
        config = load_experiment_config(resumable[0].name)
        resume = True
        print(f"Resuming {resumable[0]}", flush=True)
    else:
        chosen, attempts = choose_profile()
        variant = chosen["variant"]
        repo, revision = SEGFORMER_SOURCES[variant]
        template["model"].update({"variant": variant, "encoder_name": f"mit_{variant}",
                                  "encoder_weights": f"{repo}@{revision}",
                                  "pretrained_sha256": chosen["pretrained_sha256"]})
        template["training"]["batch_size"] = chosen["microbatch"]
        template["training"]["gradient_accumulation_steps"] = math.ceil(4 / chosen["microbatch"])
        config = prepare_fresh_experiment(template, prefix=f"segformer_{variant}",
                                          dataset_fingerprint=fingerprint)
        report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "gpu": torch.cuda.get_device_name(0), "total_vram_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
                  "attempts": attempts, "selected": chosen,
                  "fallback_reason": None if variant == "b2" else "B2 did not fit at microbatch 1"}
        with (Path(config["training"]["checkpoint_dir"]) / "cuda_profile.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        resume = False
        print(f"CUDA profile selected {chosen}; attempts={attempts}", flush=True)
    microbatch = int(config["training"]["batch_size"])
    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=microbatch,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=microbatch,
                            shuffle=False, num_workers=0)
    directory = Path(config["training"]["checkpoint_dir"])
    print(f"Transformer experiment={directory.name} resume={resume} train=565 val=142", flush=True)
    result = train_segformer(config, train_loader, val_loader, device="cuda", resume=resume)
    print(f"Transformer result={result['status']} best_dice={result['best_val_dice']} "
          f"epochs={len(result['history'])} best_checkpoint={result['best_checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
