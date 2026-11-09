# Solar Filament Segmentation

A reproducible PyTorch pipeline for the [Kaggle Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking). The competition asks for **one instance mask per filament**. This repository trains U-Net/ResNet34 and SegFormer-B2 foreground models, separates connected components, evaluates with a local implementation of the organizer's public self-evaluation notebook, and encodes masks as compressed COCO RLE. Public-notebook parity has been tested on synthetic cases; hidden Kaggle scorer parity and an independent holdout remain unverified.

## Data and environment

The local MAGFiLO training set has 707 physical JPEGs, 1,154 COCO image records, and 8,199 annotation records. Multiple records for a filename represent alternative annotation sets. `SolarFilamentDataset` groups by physical filename; the seed-42 split has 565 training and 142 validation images with no exact filename overlap. Thirteen validation filenames share a capture date with a training filename, so the random-image split is not a temporal holdout. Raw data and checkpoints are ignored by Git.

Use Python 3.11 and the dependencies in `environment.yml` or `requirements.txt`. On this Windows checkout, the tested interpreter is `.conda\python.exe`:

```powershell
& '.\.conda\python.exe' -m pip install -e .
& '.\.conda\python.exe' -m pytest -q
```

The canonical notebooks assert this interpreter and resolve the project root from either the repository root or `notebooks/` working directory. For a new environment, register an ipykernel pointing to that environment and update the notebooks' kernel selection if its name differs. `transformers==4.57.1` is required for SegFormer; the exact local package set is recorded in [`reports/environment_snapshot_20261104.txt`](reports/environment_snapshot_20261104.txt).

## Canonical workflow

| Order | Notebook | Purpose |
|---|---|---|
| 1 | [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Data inventory, polygons, annotators, intensity, crop bias, split checks, representative masks |
| 2 | [`02_training.ipynb`](notebooks/02_training.ipynb) | Pre-training checks; opt-in smoke/full training; checkpoint and history review |
| 3 | [`04_instance_error_analysis.ipynb`](notebooks/04_instance_error_analysis.ipynb) | Trained-v2 instance matches, TP/FP/FN, qualitative overlays |
| 4 | [`05_postprocessing_optimization.ipynb`](notebooks/05_postprocessing_optimization.ipynb) | Cache full-resolution probability maps and compare threshold/area on disjoint tuning and holdout subsets |
| 5 | [`03_submission.ipynb`](notebooks/03_submission.ipynb) | Independently load trained weights, preview inference, opt-in full-test CSV generation |

Run cells in order within each notebook. `RUN_SMOKE_TEST`, `RUN_FULL_TRAINING`, `RUN_POST_TRAINING_EVAL`, `RUN_MODEL_COMPARISON`, and `RUN_SUBMISSION` default to `False`. Pretrained ImageNet encoder initialization is not a trained checkpoint. The two research notebooks run bounded samples by default; their subset scores are **not** full-validation benchmarks. Increase the selected validation indices deliberately and preserve a holdout before selecting final parameters. No notebook starts a full training run by default.

The reproducible command-line path for the three-model comparison is:

```powershell
& '.\.conda\python.exe' scripts\run_cnn_retrain.py
& '.\.conda\python.exe' scripts\run_transformer.py
& '.\.conda\python.exe' scripts\run_benchmark.py
& '.\.conda\python.exe' scripts\generate_selected_submission.py reports\selected_model_provisional_<timestamp>.json
```

The training commands skip a completed experiment and resume a matching genuine `latest.pth` after interruption. Benchmarking writes a deterministic capture-date-grouped tuning/internal-check partition, probability caches, fixed and tuned results, and a unique provisional selection manifest. The internal-check set is **not independent** for original v2, which was evaluated on all 142 validation images previously. The generated CSV is local only; no command uploads to Kaggle.

Reusable code lives in `src/filament_seg/`: dataset/COCO parsing, preprocessing, model, losses, training/checkpoints, native-resolution tiling, diagnostic evaluation, cached post-processing scoring, and submission validation. `configs/unet_resnet34_v2.yaml` records the v2 experiment. The current trainer selects the best checkpoint by **patch validation Dice**; native-resolution PQ is a separate diagnostic, not the checkpoint selection metric.

## Checkpoints and resume

### Versioned v2 retraining

The original `checkpoints/unet_resnet34_v2/` is preserved. To launch a new ImageNet-initialized ResNet34 training run, use `scripts/run_cnn_retrain.py` or open `02_training.ipynb` and set `TRAINING_MODE='fresh'`, `EXPERIMENT_NAME='unet_resnet34_v2_retrain'`, `RESUME_TRAINING=False`, and `RUN_FULL_TRAINING=True`. The workflow loads [`configs/unet_resnet34_v2_retrain.yaml`](configs/unet_resnet34_v2_retrain.yaml), allocates an unused numbered directory atomically, and saves its effective `config.yaml`, `history.json`, and `metrics.json`. The empty `_retrain_01` directory was reserved; the completed retrain is `_retrain_02`. No directory is allocated while notebook full training remains disabled.

For an interrupted run, set `TRAINING_MODE='resume'`, the exact allocated ID in `EXPERIMENT_NAME`, `RESUME_TRAINING=True`, and `RUN_FULL_TRAINING=True`. Resume requires that directory's genuine `latest.pth` and restores optimizer, scheduler, GradScaler, best metric, early stopping, and history. For read-only model evaluation, use `TRAINING_MODE='evaluate'`, an exact trained ID, and leave `RUN_FULL_TRAINING=False`; `RUN_FULL_VALIDATION=True` runs a fixed 0.70/500 full-resolution diagnostic evaluation. `RUN_MODEL_COMPARISON=True` compares the selected retrain with original v2 on the same 142 image IDs and matching rules. `RUN_QUALITATIVE_COMPARISON=True` adds instance-level overlays after the quantitative comparison. These evaluation flags require trained weights and are not prerequisites for training.

[`reports/experiment_registry.json`](reports/experiment_registry.json) tracks the original v2 with unknown historical fields marked `null`. New experiments record their ID, seed, encoder initialization, Git revision, dataset fingerprint, checkpoint paths, completion status, patch Dice, and diagnostic full-resolution PQ/SQ/RQ when evaluated. Registry metadata may change after an explicitly enabled run; model weights remain ignored by Git.

`project_path()` resolves configured relative paths against the repository root, independent of notebook working directory. Production v2 files are `checkpoints/unet_resnet34_v2/latest.pth` (resume state) and `best_model.pth` (inference/evaluation). The notebook's optional smoke mode writes uniquely named files under ignored `outputs/checkpoints/smoke_v2_retrain/`; existing smoke checkpoints remain untouched. `resume=True` requires `latest.pth`, rejects smoke metadata and a mismatched experiment name when present, and returns `already_completed` when the saved epoch has reached the configured maximum. `resume=False` refuses to overwrite existing production weights; use a new experiment checkpoint directory for a fresh run. Do not copy a smoke checkpoint to `latest.pth`.

The local v1 best checkpoint is epoch 11; v2 best is epoch 29 and v2 latest is epoch 30. These files are **not** in Git. For post-training evaluation, load `best_model.pth` explicitly and verify its epoch. See the [consolidation audit](reports/consolidation_audit.md) for checkpoint locations and preserved hashes.

## Evaluation and submission

`predict_tiled` reconstructs 2048 × 2048 probability maps from 512 × 512 tiles, with 128-pixel overlap by default. `iter_probability_instances` thresholds and separates connected components. Touching filaments can merge, limiting this baseline. The legacy `diagnostic_panoptic_quality` uses one-to-one matching; the new benchmark uses `organizer_pq_image`, which counts all IoU > 0.5 pairs and scores every alternative annotation set separately, matching the public [organizer self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) on tested synthetic cases. Evaluation ground truth is rasterized from source COCO polygons with `pycocotools`; training masks retain their original implementation. The archived notebook is in [`reports/reference/organizer_self_evaluation_v6.ipynb`](reports/reference/organizer_self_evaluation_v6.ipynb). The hidden Kaggle scorer's ground-truth conversion remains unverified.

The competition CSV requires `filament_id,segmentation_rle`, one row per predicted filament, with compressed COCO counts for full-resolution masks. `generate_submission()` validates rows and refuses to overwrite an existing CSV. `03_submission.ipynb` loads an explicit selected-model manifest and generates the entire 180-image file only when enabled. The initial complete v2 fallback CSV and the pre-existing 33-row, five-image [`historical_partial_submission_5_images.csv`](outputs/historical_partial_submission_5_images.csv) are retained. No Kaggle upload or acceptance test has been performed. [The Kaggle cutoff and separate report deadline](reports/competition_deadline_check.md) were checked on the live competition page.

## Research status

The [Stage 3.3 report](reports/stage_3_3_verification.md) covers CUDA and instance infrastructure. The [historical pipeline audit](reports/pipeline_audit_report.md) and [failure analysis](reports/error_and_failure_analysis.md) contain v1 results and hypotheses. The [provisional three-model benchmark](reports/model_benchmark_20261104T094701Z.md) reports fixed full-validation and tuning-selected internal-check PQ under one local protocol; earlier research-notebook PQ values used a different protocol and are not directly comparable. Next work is hidden-scorer parity, an independent labeled holdout, and controlled instance-separation and tiling experiments.

The older feature branches and executed verification notebooks remain recoverable from Git history. See [consolidation audit](reports/consolidation_audit.md) for their disposition.
