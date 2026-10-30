# Stage 3.3 — Instance-aware evaluation and local CUDA verification

Audited 2026-10-31 in `C:\Users\Vinny\Project\Kaggle\Solar Filament Segmentation`. **Status: PASS for Stage 3.3 infrastructure and bounded GPU checks.** No full epoch, Kaggle submission, or model-weight download was performed.

## Scope and files

Inspected: `src/filament_seg/{dataset,preprocessing,model,losses,train,evaluate,inference,utils}.py`, `configs/baseline.yaml`, `notebooks/{02_training,03_submission}.ipynb`, `reports/competition_target_audit.md`, the existing tests, and the MAGFiLO COCO JSON. `src/filament_seg/submission.py` did not exist before this stage.

Changed for Stage 3.3:

- [`dataset.py`](../src/filament_seg/dataset.py): added lazy `iter_instances(idx, annotation_set_index=0)` for separate full-resolution masks and metadata. Existing binary `__getitem__` remains unchanged.
- [`inference.py`](../src/filament_seg/inference.py): added configurable probability threshold, minimum area, and 4/8-connectivity component-mask iterator. The earlier labeled-mask API remains available.
- [`evaluate.py`](../src/filament_seg/evaluate.py): added deterministic one-to-one **diagnostic** PQ/SQ/RQ. Existing foreground Dice and IoU diagnostics remain unchanged.
- [`submission.py`](../src/filament_seg/submission.py): added compressed COCO RLE encode/decode and structural CSV-row validation. No submission writer or test-set predictions were created.
- [`02_training.ipynb`](../notebooks/02_training.ipynb): accepts project-root or `notebooks` working directory, checks the interpreter, and honors an optional configured device; its bounded Stage 3.2 check remains intact.
- [`03_submission.ipynb`](../notebooks/03_submission.ipynb): runs a synthetic full-resolution RLE/schema check without writing a submission.
- [`tests/test_instances.py`](../tests/test_instances.py) and [`tests/test_cuda_stage33.py`](../tests/test_cuda_stage33.py): added boundary and real-CUDA smoke tests; registered the `cuda` marker in [`pyproject.toml`](../pyproject.toml).
- [`requirements.txt`](../requirements.txt), [`environment.yml`](../environment.yml), and [`pyproject.toml`](../pyproject.toml): added the required `pycocotools>=2.0.11`. Installed `pycocotools 2.0.11` into the project environment. PyTorch was not reinstalled during this stage.

The earlier Stage 3.2 edits to `model.py`, `losses.py`, `train.py`, `configs/baseline.yaml`, and `tests/test_baseline.py` were present before this stage and were preserved. The existing Stage 3.2 verified notebook was moved to [`02_training_verified_stage32.ipynb`](../notebooks/02_training_verified_stage32.ipynb) before placing the new verified run at the requested filename.

## CUDA device and model

`nvidia-smi` identified **NVIDIA GeForce RTX 3060**, **12,288 MiB** VRAM, driver **617.42**. The exact interpreter was `C:\Users\Vinny\Project\Kaggle\Solar Filament Segmentation\.conda\python.exe` (Python **3.11.18**), with PyTorch **2.14.1+cu126**, CUDA runtime **12.6**, and `torch.cuda.is_available() == True`. A tensor multiply/sum succeeded on `cuda:0`; `torch.cuda.get_device_name(0)` matched the RTX 3060. [Device selection code](../src/filament_seg/utils.py), [baseline model](../src/filament_seg/model.py).

`get_device()` chooses CUDA when no preference is supplied; `get_device("cpu")` still returns CPU. The training notebook now passes optional `training.device` from config, retaining explicit override behavior. The prior CPU result came from the previous `+cpu` PyTorch environment; the current `+cu126` build and fresh kernels both select CUDA. The U-Net/ResNet34 model moved to `cuda:0`. [Notebook initialization](../notebooks/02_training.ipynb), [configuration](../configs/baseline.yaml).

## Real-batch mixed-precision optimization

[`tests/test_cuda_stage33.py`](../tests/test_cuda_stage33.py) loaded real MAGFiLO training images at **512 × 512**, ran one U-Net forward/backward/AdamW step with FP16 autocast and `torch.amp.GradScaler("cuda")`, verified finite logits, loss, **all computed gradients**, a nonzero gradient, and changed parameters. CUDA peak counters were reset per case and cache was cleared between cases. No resolution or batch size was reduced.

| Batch size | Result | Loss | Peak allocated | Peak reserved |
|---:|---|---:|---:|---:|
| 2 | PASS | 1.203646 | 588.8 MiB | 610.0 MiB |
| 4 | PASS | 1.122724 | 1,032.6 MiB | 1,080.0 MiB |

These are one-step peaks, **not** an epoch-long VRAM bound. Batch size 8 from the current config was not tested; batch size 4 is the largest verified local smoke setting. GPU tests carry the `cuda` marker and skip only when `torch.cuda.is_available()` is false.

## Instance ground truth and separation

The COCO JSON has **707 physical training images**, **1,154 image records** (alternative annotation sets), and **8,199 annotations**. `iter_instances` selects a set by deterministic image-record order and yields one `uint8` mask per polygon annotation, with `annotation_id`, `category_id`, `image_id`, and `filename`. It supports multiple polygons per annotation and retains overlapping instances as separate masks. A real first-instance check yielded a **2048 × 2048** mask with 885 foreground pixels and matching metadata. [Training JSON `images` and `annotations`](../data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json), [implementation](../src/filament_seg/dataset.py), [synthetic tests](../tests/test_instances.py).

`iter_probability_instances` keeps the input resolution and yields individual binary component masks. Tests passed for threshold equality, minimum area, 4- versus 8-connectivity, and an all-background probability map. **Touching filaments merge into one component**, so connected components are a baseline separator, not a reliable final instance method. [Implementation](../src/filament_seg/inference.py), [tests](../tests/test_instances.py).

## Diagnostic PQ, SQ, and RQ

The local matcher uses a one-to-one assignment that prioritizes the number of eligible matches and then summed IoU. Default matching is **strict IoU > 0.5**; `inclusive=True` uses `>=`. It computes `PQ = sum(matched IoU)/(TP + 0.5 FP + 0.5 FN)`, `SQ = sum(matched IoU)/TP`, and `RQ = TP/(TP + 0.5 FP + 0.5 FN)`. Both instance sets empty are defined locally as PQ=SQ=RQ=1; with no matches and a nonempty set, SQ=0. These empty conventions are diagnostic choices. Tests passed for perfect, missing, extra, partial, threshold-boundary, overlapping, and empty cases. [Implementation](../src/filament_seg/evaluate.py), [tests](../tests/test_instances.py).

The [official competition page](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking) confirms the PQ formula but does not specify the exact matching and multi-annotator aggregation behavior. **This implementation is not a verified replica of official leaderboard PQ.** See the [Stage 3.1 audit](competition_target_audit.md).

## Compressed COCO RLE and submission schema

`pycocotools.mask` encodes Fortran-contiguous binary masks and returns UTF-8 compressed counts. Decode uses explicit `[height, width]`; small and **2048 × 2048** round trips were exact. Malformed masks and a space-separated RLE string were rejected. The old [`mask_to_rle`](../src/filament_seg/inference.py) still produces space-delimited start/length pairs for backward compatibility and must **not** be used for competition output. [New codec](../src/filament_seg/submission.py), [tests](../tests/test_instances.py).

The [official submission specification](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking) requires exactly `filament_id,segmentation_rle`, one row per predicted filament, compressed counts only, and 2048 × 2048 masks. It defines image-stem-plus-unique-suffix identifiers. `validate_submission_rows` checks column order/names, known test-image stems, identifier uniqueness, and decoded full-resolution nonempty masks. Duplicate IDs, unknown stems, and wrong columns were rejected in tests. **Structural validation passed; Kaggle acceptance was not tested.** There is no local sample-submission file. The submission notebook uses one explicitly synthetic mask and does not create a CSV.

## Fresh-kernel notebook execution and suite

The kernel specification was checked to invoke the project's `.conda\python.exe`. Both notebooks ran from fresh kernels with **each** working directory: repository root and `notebooks`. The `02_training` run executed seven code cells (setup, split, preprocessing, DataLoaders, model forward/backward, bounded 50-step fixed-patch check, plot) and selected CUDA in both locations. Root execution recorded train-mode loss **1.081589 → 0.312230** and foreground Dice **0.0427 → 0.8570**; `notebooks`-directory execution recorded loss **1.085348 → 0.299618** and Dice **0.0409 → 0.8959**. The full-training section is markdown and did not run. The `03_submission` notebook executed its one code cell in both locations, selected CUDA, and passed synthetic RLE/schema checks. No `NameError`, import error, assertion failure, or notebook error output occurred.

Requested outputs: [`02_training_verified.ipynb`](../notebooks/02_training_verified.ipynb) and [`03_submission_verified.ipynb`](../notebooks/03_submission_verified.ipynb). Additional `*_verified_stage33_notebooks.ipynb` copies record the second working-directory check. Existing experiment outputs were not overwritten.

Full suite: **29 passed** with `.conda\python.exe -m pytest -q`, including the Stage 3.2 baseline regressions and both CUDA tests. `py_compile` and `git diff --check` passed. `pycocotools` emitted a NumPy 2 `__array__` deprecation warning on decode; round trips and tests still passed.

## Remaining uncertainties and blockers before full training

1. Official scorer parity remains unverified: exact instance matching, tie handling, empty-image behavior, and aggregation over alternate annotator sets need comparison with the organizer's self-evaluation notebook. [Official PQ description](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking).
2. No model predictions or CSV were submitted to Kaggle. Structural format checks cannot prove official acceptance.
3. Batch size 8 in `baseline.yaml` and epoch-long VRAM use are unverified. Start full training at batch size 4 or run a separate capacity check before using 8; no full run was started here.
4. The 50-step fixed-patch notebook remains a learning diagnostic, not validation performance. Evaluation-mode Dice after that short run varied between **0.3133** and **0.0838** across working-directory runs; monitor BatchNorm behavior during actual training.
5. The filename split prevents exact image overlap, but the [Stage 3.1 audit](competition_target_audit.md) records same-day capture overlap and recommends a grouped holdout for stronger generalization evidence.
