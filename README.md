# Solar Filament Segmentation

A reproducible PyTorch baseline for the [Kaggle Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking). The competition asks for **one instance mask per filament**. This repository trains a one-channel U-Net/ResNet34 foreground model, separates connected components as a baseline, computes **diagnostic** PQ, and encodes masks as compressed COCO RLE. The local evaluator has not been proven identical to the organizer's scorer; see [competition target audit](reports/competition_target_audit.md).

## Data and environment

The local MAGFiLO training set has 707 physical JPEGs, 1,154 COCO image records, and 8,199 annotation records. Multiple records for a filename represent alternative annotation sets. `SolarFilamentDataset` groups by physical filename; the seed-42 split has 565 training and 142 validation images with no exact filename overlap. Thirteen validation filenames share a capture date with a training filename, so the random-image split is not a temporal holdout. Raw data and checkpoints are ignored by Git.

Use Python 3.11 and the dependencies in `environment.yml` or `requirements.txt`. On this Windows checkout, the tested interpreter is `.conda\python.exe`:

```powershell
& '.\.conda\python.exe' -m pip install -e .
& '.\.conda\python.exe' -m pytest -q
```

The canonical notebooks assert this interpreter and resolve the project root from either the repository root or `notebooks/` working directory. For a new environment, register an ipykernel pointing to that environment and update the notebooks' kernel selection if its name differs.

## Canonical workflow

| Order | Notebook | Purpose |
|---|---|---|
| 1 | [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Data inventory, polygons, annotators, intensity, crop bias, split checks, representative masks |
| 2 | [`02_training.ipynb`](notebooks/02_training.ipynb) | Pre-training checks; opt-in smoke/full training; checkpoint and history review |
| 3 | [`04_instance_error_analysis.ipynb`](notebooks/04_instance_error_analysis.ipynb) | Trained-v2 instance matches, TP/FP/FN, qualitative overlays |
| 4 | [`05_postprocessing_optimization.ipynb`](notebooks/05_postprocessing_optimization.ipynb) | Cache full-resolution probability maps and compare threshold/area on disjoint tuning and holdout subsets |
| 5 | [`03_submission.ipynb`](notebooks/03_submission.ipynb) | Independently load trained weights, preview inference, opt-in full-test CSV generation |

Run cells in order within each notebook. `RUN_SMOKE_TEST`, `RUN_FULL_TRAINING`, `RUN_POST_TRAINING_EVAL`, `RUN_MODEL_COMPARISON`, and `RUN_SUBMISSION` default to `False`. Pretrained ImageNet encoder initialization is not a trained checkpoint. The two research notebooks run bounded samples by default; their subset scores are **not** full-validation benchmarks. Increase the selected validation indices deliberately and preserve a holdout before selecting final parameters. No notebook starts a full training run by default.

Reusable code lives in `src/filament_seg/`: dataset/COCO parsing, preprocessing, model, losses, training/checkpoints, native-resolution tiling, diagnostic evaluation, cached post-processing scoring, and submission validation. `configs/unet_resnet34_v2.yaml` records the v2 experiment. The current trainer selects the best checkpoint by **patch validation Dice**; native-resolution PQ is a separate diagnostic, not the checkpoint selection metric.

## Checkpoints and resume

`project_path()` resolves configured relative paths against the repository root, independent of notebook working directory. Production v2 files are `checkpoints/unet_resnet34_v2/latest.pth` (resume state) and `best_model.pth` (inference/evaluation). Smoke runs create uniquely named files under `checkpoints/unet_resnet34_v2/smoke_runs/`; existing smoke checkpoints remain untouched. `resume=True` reads only `latest.pth`, rejects smoke metadata and a mismatched experiment name when present, and returns `already_completed` when the saved epoch has reached the configured maximum. `resume=False` refuses to overwrite existing production weights; use a new experiment checkpoint directory for a fresh run. Do not copy a smoke checkpoint to `latest.pth`.

The local v1 best checkpoint is epoch 11; v2 best is epoch 29 and v2 latest is epoch 30. These files are **not** in Git. For post-training evaluation, load `best_model.pth` explicitly and verify its epoch. See the [consolidation audit](reports/consolidation_audit.md) for checkpoint locations and preserved hashes.

## Evaluation and submission

`predict_tiled` reconstructs 2048 × 2048 probability maps from 512 × 512 tiles, with 128-pixel overlap by default. `iter_probability_instances` thresholds and separates connected components. Touching filaments can merge, limiting this baseline. `diagnostic_panoptic_quality` uses one-to-one IoU matching and can return matched/unmatched indices for error analysis. Empty-image conventions and alternative annotator aggregation may differ from the official scorer.

The competition CSV requires `filament_id,segmentation_rle`, one row per predicted filament, with compressed COCO counts for full-resolution masks. `generate_submission()` validates rows and refuses to overwrite an existing CSV. `03_submission.ipynb` generates the entire 180-image file only when explicitly enabled; it reports images with no predicted instances. The pre-existing 33-row, five-image artifact is retained as [`historical_partial_submission_5_images.csv`](outputs/historical_partial_submission_5_images.csv) and is **not** a complete competition submission. No Kaggle acceptance test has been performed.

## Research status

The [Stage 3.3 report](reports/stage_3_3_verification.md) covers CUDA and instance infrastructure. The [historical pipeline audit](reports/pipeline_audit_report.md) and [failure analysis](reports/error_and_failure_analysis.md) contain v1 results and hypotheses. Historical v2 PQ values in the research notebooks are unverified references until recomputed with a fixed full-validation protocol. The most valuable next work is scorer parity, full-validation v2 error analysis, then controlled instance-separation and tiling experiments. Preserve a tuning/holdout distinction; do not compare scores from different subsets as though they were the same benchmark.

The older feature branches and executed verification notebooks remain recoverable from Git history. See [consolidation audit](reports/consolidation_audit.md) for their disposition.
