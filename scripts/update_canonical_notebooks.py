"""Script to update canonical notebooks 01_eda.ipynb, 02_training.ipynb, and 03_submission.ipynb in place."""

import json
from pathlib import Path


def make_cell(cell_type, source_lines):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source_lines[:-1]] + ([source_lines[-1]] if source_lines else []),
    }


def update_02_training():
    cells = [
        make_cell("markdown", [
            "# 02 - Model Training & Validation Orchestration (U-Net ResNet34 v2)",
            "",
            "**Competition:** Solar Filament Segmentation Challenge 2026  ",
            "**Experiment:** U-Net with ImageNet-pretrained ResNet34 Backbone (v2 - EDA-Informed)  ",
            "**Key Improvements from Audit:**",
            "- Pretrained ImageNet weights adapted to 1-channel grayscale input.",
            "- Calibrated positive loss weighting (`pos_weight=8.0` vs. 30.0) to eliminate massive false positives.",
            "- Foreground-aware 512x512 patch cropping (`foreground_prob=0.7`).",
            "- Mixed precision with PyTorch AMP GradScaler, unscaling, and gradient clipping (`max_norm=1.0`).",
            "- Separate experiment directory: `checkpoints/unet_resnet34_v2/` preserving original baseline weights.",
            "- Resumable training loop from `checkpoints/unet_resnet34_v2/latest.pth`.",
            "- Full-resolution tiled evaluation and baseline comparison workflow.",
            "",
            "> **Principle:** Kaggle notebooks import modular Python components from `src/filament_seg/`."
        ]),
        make_cell("code", [
            "from pathlib import Path",
            "import sys",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import torch",
            "from torch.utils.data import DataLoader, Subset",
            "",
            "cwd = Path.cwd().resolve()",
            "project_root = cwd if (cwd / 'src' / 'filament_seg' / 'dataset.py').is_file() else cwd.parent",
            "assert (project_root / 'src' / 'filament_seg' / 'dataset.py').is_file(), cwd",
            "if str(project_root / 'src') not in sys.path:",
            "    sys.path.insert(0, str(project_root / 'src'))",
            "",
            "from filament_seg.utils import set_seed, load_config, get_device, get_logger",
            "from filament_seg.model import build_model",
            "from filament_seg.losses import BCEDiceLoss",
            "from filament_seg.evaluate import compute_dice, compute_iou, evaluate_full_resolution_dataset, compare_models",
            "from filament_seg.dataset import SolarFilamentDataset",
            "from filament_seg.preprocessing import ForegroundAwareCrop, get_train_transform, get_val_transform",
            "from filament_seg.train import load_checkpoint, save_checkpoint, train_v2_model, validate_one_epoch",
            "",
            "# Load Configuration & Set Seed",
            "config_path = project_root / 'configs' / 'unet_resnet34_v2.yaml'",
            "config = load_config(config_path if config_path.is_file() else project_root / 'configs' / 'baseline.yaml')",
            "set_seed(config['project']['seed'])",
            "device = get_device(config['training'].get('device'))",
            "print(f'Python: {sys.executable}')",
            "print(f'Project root: {project_root}')",
            "print(f'Device selected: {device}')",
            "print(f'Experiment: {config[\"project\"][\"experiment_name\"]}')"
        ]),
        make_cell("markdown", [
            "## 1. Dataset & DataLoader Setup (Deterministic 565/142 Physical Split)",
            "",
            "Split 707 physical filenames once (seed 42, 80/20: 565 train / 142 val) ensuring zero leakage across splits.",
            "- Training dataset uses foreground-aware crop (512x512, foreground_prob=0.7).",
            "- Validation dataset uses center crop for epoch-level loss tracking.",
            "- Evaluation dataset preserves full 2048x2048 resolution for competition-relevant evaluation."
        ]),
        make_cell("code", [
            "train_dir = project_root / 'data' / 'filament-segmentation-2026' / 'MAGFiLO_1.0_Kaggle_2026' / 'train'",
            "image_dir = train_dir / 'train_images'",
            "annotation_path = train_dir / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'",
            "",
            "train_dataset = SolarFilamentDataset(image_dir, annotation_path, transform=get_train_transform(512))",
            "val_dataset = SolarFilamentDataset(image_dir, annotation_path, transform=get_val_transform(512))",
            "val_eval_dataset = SolarFilamentDataset(image_dir, annotation_path)",
            "assert len(train_dataset) == len(val_dataset) == len(val_eval_dataset) == 707",
            "",
            "indices = np.random.default_rng(config['project']['seed']).permutation(len(train_dataset))",
            "train_indices = indices[:565].tolist()",
            "val_indices = indices[565:].tolist()",
            "",
            "train_names = {train_dataset.filenames[i] for i in train_indices}",
            "val_names = {val_dataset.filenames[i] for i in val_indices}",
            "assert len(train_names) == 565 and len(val_names) == 142",
            "assert train_names.isdisjoint(val_names)",
            "",
            "train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=4, shuffle=True, num_workers=0)",
            "val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=4, shuffle=False, num_workers=0)",
            "print(f'Train images: {len(train_indices)}, batches: {len(train_loader)}')",
            "print(f'Validation images: {len(val_indices)}, batches: {len(val_loader)}')"
        ]),
        make_cell("markdown", [
            "## 2. CUDA Model Initialization & Pretrained ResNet34 Encoder",
            "",
            "Initialize U-Net with ImageNet-pretrained ResNet34 encoder adapted to single-channel grayscale input on CUDA."
        ]),
        make_cell("code", [
            "pretrained = config['model'].get('encoder_weights') == 'imagenet'",
            "model = build_model(pretrained=pretrained, in_channels=1, classes=1).to(device)",
            "total_params = sum(p.numel() for p in model.parameters())",
            "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)",
            "print(f'Model Backbone: {config[\"model\"][\"encoder_name\"]} (Pretrained: {pretrained})')",
            "print(f'Device: {next(model.parameters()).device}')",
            "print(f'Total parameters: {total_params:,} | Trainable: {trainable_params:,}')"
        ]),
        make_cell("markdown", [
            "## 3. v2 Loss, Optimizer & Scheduler Configuration",
            "",
            "Configure calibrated positive loss weighting (`pos_weight=8.0`), AdamW optimizer, and CosineAnnealingLR scheduler."
        ]),
        make_cell("code", [
            "train_cfg = config['training']",
            "loss_cfg = train_cfg.get('loss', {})",
            "criterion = BCEDiceLoss(",
            "    bce_weight=float(loss_cfg.get('bce_weight', 0.5)),",
            "    dice_weight=float(loss_cfg.get('dice_weight', 0.5)),",
            "    smooth=float(loss_cfg.get('smooth', 1.0)),",
            "    pos_weight=float(loss_cfg.get('pos_weight', 8.0)),",
            ")",
            "lr = float(train_cfg.get('learning_rate', 3e-4))",
            "wd = float(train_cfg.get('weight_decay', 1e-4))",
            "optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)",
            "sched_cfg = train_cfg.get('scheduler', {})",
            "t_max = int(sched_cfg.get('t_max', train_cfg.get('epochs', 30)))",
            "eta_min = float(sched_cfg.get('eta_min', 1e-6))",
            "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)",
            "print(f'BCEDiceLoss configured: pos_weight={loss_cfg.get(\"pos_weight\", 8.0)}')",
            "print(f'Optimizer: AdamW (lr={lr}, weight_decay={wd})')",
            "print(f'Scheduler: CosineAnnealingLR (T_max={t_max}, eta_min={eta_min})')"
        ]),
        make_cell("markdown", [
            "## 4. AMP Setup & Gradient Clipping",
            "",
            "Configure automatic mixed precision (AMP) with PyTorch GradScaler and gradient norm clipping (`max_norm=1.0`) to prevent gradient explosion and inf checks errors."
        ]),
        make_cell("code", [
            "use_amp = config['training'].get('amp', True) and device.type == 'cuda'",
            "scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp)",
            "clip_grad = float(config['training'].get('clip_grad_norm', 1.0))",
            "print(f'AMP enabled: {use_amp}')",
            "print(f'GradScaler initialized for device: {device.type}')",
            "print(f'Gradient norm clipping threshold: {clip_grad}')"
        ]),
        make_cell("markdown", [
            "## 5. Training Pipeline Smoke Test",
            "",
            "Execute a 2-batch smoke test to verify CUDA execution, AMP autocast, gradient flow, loss finiteness, checkpoint saving to `checkpoints/unet_resnet34_v2/smoke_checkpoint.pth`, and resumption."
        ]),
        make_cell("code", [
            "smoke_result = train_v2_model(",
            "    config=config,",
            "    train_loader=train_loader,",
            "    val_loader=val_loader,",
            "    val_dataset=val_dataset,",
            "    val_indices=val_indices,",
            "    device=device,",
            "    smoke_test=True,",
            ")",
            "print('Smoke Test Status:', smoke_result['status'])",
            "print(f'Verified checkpoint saved at: {smoke_result[\"checkpoint\"]}')"
        ]),
        make_cell("markdown", [
            "## 6. Manual Full-Training Toggle & Execution",
            "",
            "**Ready for manual execution:** Running the cell below trains the v2 model for 30 epochs (or resumes from `checkpoints/unet_resnet34_v2/latest.pth` if interrupted).",
            "To execute full training manually, set `RUN_FULL_TRAINING = True`."
        ]),
        make_cell("code", [
            "RUN_FULL_TRAINING = False  # Set to True when ready to execute full 30-epoch training run",
            "",
            "if RUN_FULL_TRAINING:",
            "    print('Starting U-Net ResNet34 v2 training run...')",
            "    train_result = train_v2_model(",
            "        config=config,",
            "        train_loader=train_loader,",
            "        val_loader=val_loader,",
            "        val_dataset=val_dataset,",
            "        val_indices=val_indices,",
            "        device=device,",
            "        resume=True,",
            "        smoke_test=False,",
            "    )",
            "    print('Training completed!')",
            "    print('Best checkpoint:', train_result['best_checkpoint'])",
            "else:",
            "    print('RUN_FULL_TRAINING is False. Full training is prepared and ready for manual execution.')"
        ]),
        make_cell("markdown", [
            "## 7. Checkpoint Resumption Logic & Verification",
            "",
            "Demonstrate checkpoint restoration logic:",
            "- Full training with `resume=True` checks ONLY `checkpoints/unet_resnet34_v2/latest.pth`.",
            "- If `latest.pth` exists, it restores model weights, optimizer, scheduler, scaler, epoch, and validation history.",
            "- If `latest.pth` does not exist, training initializes cleanly from ImageNet-pretrained ResNet34 weights (Epoch 1).",
            "- `smoke_checkpoint.pth` is strictly isolated and never used as an automatic resume source."
        ]),
        make_cell("code", [
            "v2_latest_path = project_root / 'checkpoints' / 'unet_resnet34_v2' / 'latest.pth'",
            "",
            "if v2_latest_path.is_file():",
            "    resumed_state = load_checkpoint(",
            "        v2_latest_path,",
            "        model=model,",
            "        optimizer=optimizer,",
            "        scheduler=scheduler,",
            "        scaler=scaler,",
            "        map_location=device,",
            "    )",
            "    print(f'Resumed from latest training checkpoint: {v2_latest_path.name}')",
            "    print(f'Next Epoch to run: {resumed_state.get(\"epoch\", 0) + 1}')",
            "    print(f'Best Val Metric: {resumed_state.get(\"best_val_dice\", 0.0):.4f}')",
            "else:",
            "    print(f'No full-training checkpoint found at: {v2_latest_path.name}')",
            "    print('A full training run will initialize cleanly from ImageNet-pretrained ResNet34 weights (Epoch 1).')",
            "    print('Smoke-test checkpoints are strictly isolated and will never be loaded for full training.')"
        ]),
        make_cell("markdown", [
            "## 8. Full-Resolution Validation",
            "",
            "Perform full-resolution 2048x2048 validation using tiled prediction with 2D Hann window blending, calculating semantic Dice, IoU, Precision, Recall, and Panoptic Quality."
        ]),
        make_cell("code", [
            "val_eval_indices = val_indices[:5]  # demonstration on first 5 validation images",
            "print(f'Evaluating full-resolution tiled inference on {len(val_eval_indices)} validation images...')",
            "val_results = evaluate_full_resolution_dataset(",
            "    model=model,",
            "    dataset=val_eval_dataset,",
            "    sample_indices=val_eval_indices,",
            "    device=device,",
            "    threshold=0.75,",
            "    min_area=300,",
            "    use_tiled=True,",
            "    tile_size=512,",
            "    overlap=128,",
            ")",
            "summary = val_results['summary']",
            "sem = summary['semantic_metrics']",
            "inst = summary['instance_metrics']",
            "print('--- Full-Resolution Validation Summary ---')",
            "print(f'Macro Dice: {sem[\"macro_dice\"]:.4f} | Micro Dice: {sem[\"micro_dice\"]:.4f}')",
            "print(f'Macro IoU:  {sem[\"macro_iou\"]:.4f}')",
            "print(f'Precision:  {sem[\"macro_precision\"]:.4f} | Recall: {sem[\"macro_recall\"]:.4f}')",
            "print(f'Global PQ:  {inst[\"global_pq\"]:.4f} (TP={inst[\"total_tp\"]}, FP={inst[\"total_fp\"]}, FN={inst[\"total_fn\"]})')"
        ]),
        make_cell("markdown", [
            "## 9. Baseline v1 versus v2 Model Comparison",
            "",
            "Evaluates and compares the original baseline (`checkpoints/unet_resnet34/best_model.pth`) and v2 (`checkpoints/unet_resnet34_v2/best_model.pth` or initialized v2) on the exact same validation images using full-resolution tiled inference."
        ]),
        make_cell("code", [
            "v2_best_path = project_root / 'checkpoints' / 'unet_resnet34_v2' / 'best_model.pth'",
            "# Load Baseline Model (v1)",
            "model_v1 = build_model(pretrained=False).to(device)",
            "load_checkpoint(project_root / 'checkpoints' / 'unet_resnet34' / 'best_model.pth', model_v1, map_location=device)",
            "model_v1.eval()",
            "",
            "# Load v2 Model if trained (or compare against current initialized v2 model)",
            "model_v2 = build_model(pretrained=True).to(device)",
            "if v2_best_path.is_file():",
            "    load_checkpoint(v2_best_path, model_v2, map_location=device)",
            "    model_v2_name = 'U-Net ResNet34 v2 (Trained)'",
            "else:",
            "    model_v2_name = 'U-Net ResNet34 v2 (Pretrained Init)'",
            "model_v2.eval()",
            "",
            "# Compare on sample validation images (or full validation set)",
            "comp_indices = val_indices[:5]  # demonstration on first 5 validation images",
            "comp_results = compare_models(",
            "    model_a=model_v1,",
            "    model_b=model_v2,",
            "    dataset=val_eval_dataset,",
            "    sample_indices=comp_indices,",
            "    device=device,",
            "    threshold_a=0.75,",
            "    min_area_a=300,",
            "    threshold_b=0.75,",
            "    min_area_b=300,",
            "    model_a_name='Baseline U-Net (v1)',",
            "    model_b_name=model_v2_name,",
            ")",
            "",
            "print('--- Model Comparison Table ---')",
            "display(comp_results['comparison_table'])"
        ])
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Solar Filament (.conda)",
                "language": "python",
                "name": "solar-filament-local",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }

    out_path = Path("notebooks/02_training.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated notebooks/02_training.ipynb in place.")


if __name__ == "__main__":
    update_02_training()
