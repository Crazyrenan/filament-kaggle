# Pipeline Audit Report: Solar Filament Segmentation Challenge 2026

**Date:** 2026-10-31  
**Target Environment:** Python 3.11.18 (`.conda`), PyTorch 2.14.1+cu126, NVIDIA RTX 3060 (12 GB VRAM)  
**Baseline Checkpoint:** `checkpoints/unet_resnet34/best_model.pth` (Epoch 11, Val Dice ~0.6299, Val IoU ~0.5426) & `latest.pth` (Epoch 18, Early Stopping at Patience 7)

---

## Executive Summary

This audit evaluates the end-to-end solar filament segmentation pipeline across 11 key components: dataset handling, multi-annotator structure, data splitting, cropping & validation transforms, normalization, loss formulations, AMP/GradScaler stability, checkpointing, metric aggregation, instance extraction/PQ, and submission compliance.

The investigation distinguishes between **Confirmed Defects** (proven via code analysis or reproduction), **Potential Risks** (factors that could bias evaluation or generalization), and **Design Trade-offs** (deliberate simplifications).

---

## Phase 1 — Comprehensive Audit Findings

### 1. Dataset Parsing and COCO Polygon Handling
* **Status:** Verified working with 1 design limitation.
* **Analysis:**
  - `SolarFilamentDataset` correctly parses `MAGFiLO_1.0_Annotations_kaggle2026_train.json` containing 1,154 image records and 8,199 annotations.
  - Floating-point polygons (`[x1, y1, x2, y2, ...]`) are rounded with `np.rint(points).astype(np.int32)` and rasterized using `cv2.fillPoly`.
  - All 8,199 annotations contain valid polygon coordinates with $\ge 3$ vertices; zero empty or corrupted annotations exist.
  - Multi-polygon annotations (multi-segment filaments) are correctly accumulated into the same instance mask in `iter_instances`.
* **Design Limitation:** In `SolarFilamentDataset.__getitem__`, all individual filament polygons for a physical image are unioned into a single binary semantic mask (`dtype=np.uint8`, values in $\{0, 1\}$). Individual instance boundaries and identities are discarded during training data generation.

### 2. Treatment of Alternative Annotations for the Same Physical Image
* **Status:** Confirmed Design Limitation / Evaluation Risk.
* **Analysis:**
  - The dataset has 707 physical image files but 1,154 COCO image records: 411 images have 1 annotation set, 145 have 2 sets, and 151 have 3 sets (annotator passes prefixed e.g. `050101-`, `050102-`, `050103-`).
  - During training and basic `__getitem__`, `dataset.py` strictly selects `self.images_by_filename[filename][0]["id"]` (`annotation_policy="first"`).
  - Alternative annotations (representing expert annotator variance and boundary uncertainty) are completely disregarded in the default training target and in previous validation loops.
  - In Stage 3.3, `iter_instances(idx, annotation_set_index=0)` was introduced, enabling multi-annotator access, but aggregate validation has hitherto evaluated only against set 0.

### 3. Train/Validation Splitting and Potential Filename Leakage
* **Status:** Verified split integrity; Confirmed Temporal Correlation Risk.
* **Analysis:**
  - Splitting is performed on the 707 unique physical filenames with seed 42 (80% train = 565 images, 20% validation = 142 images).
  - **Zero Filename Leakage:** Exact filename sets are strictly disjoint (`train_names.isdisjoint(val_names) == True`). Alternative annotation records for the same physical image cannot cross split boundaries.
  - **Temporal Correlation Risk:** Filename timestamps (`YYYYMMDDHHMMSS[telescope]h.jpeg`) reveal that 13 validation images share an exact observation calendar date with training images (and 33 test images share dates with training images). Because solar active regions persist across hours/days, random holdout may be slightly optimistic compared to a strict time-blocked split.

### 4. Foreground-Aware Cropping and Validation Transforms
* **Status:** Confirmed Major Defect in Previous Validation Pipeline.
* **Analysis:**
  - `ForegroundAwareCrop(patch_size=512, foreground_prob=0.7)` in training effectively samples around foreground coordinates, mitigating patch-level class imbalance.
  - **Major Defect:** `get_val_transform(512)` applied `A.CenterCrop(512, 512)` to the 2048x2048 images.
  - **Empirical Proof:** In our audit of the 142 validation images, **48 images (33.8%) contain ZERO ground-truth filament pixels in the 512x512 center crop**, whereas **100% of full 2048x2048 images contain filaments (0% empty)**.
  - Because `compute_dice` assigns 1.0 when both prediction and ground truth are empty, and 0.0 if even a single false-positive pixel is predicted on an empty patch, 33.8% of the validation set in the previous baseline was scoring either 1.0 or 0.0 based on background noise. This severely distorted the reported validation Dice of 0.6299.
  - **Fix:** Full-resolution 2048x2048 evaluation (via tiled inference with window blending) is mandatory for primary validation; center-crop remains strictly an auxiliary diagnostic.

### 5. Image Normalization and Pretrained Encoder Preprocessing
* **Status:** Confirmed Design Inconsistency / Optimization Opportunity.
* **Analysis:**
  - Raw images are 8-bit grayscale JPEGs with values in `[0, 255]`.
  - `dataset.py` normalizes images simply by `image = image.astype(np.float32) / 255.0` (`[0, 1]` range).
  - `configs/baseline.yaml` specifies `normalization: "minmax"`.
  - The model architecture uses a ResNet34 backbone. Standard ImageNet pretraining expects inputs standardized with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. While SMP sums the input channels when `in_channels=1`, feeding unstandardized `[0, 1]` inputs creates a distribution shift from ImageNet features.
  - In `baseline.yaml`, `encoder_weights: null` indicates the baseline was trained without ImageNet weights or adapted from scratch.

### 6. Loss Implementation and Positive Weighting (`pos_weight=30.0`)
* **Status:** Confirmed Model Calibration Shift.
* **Analysis:**
  - Loss combines `0.5 * BCEWithLogits + 0.5 * SoftDice`.
  - `pos_weight=30.0` was applied to BCE. Mathematically, this penalizes false negatives 30 times more heavily than false positives:
    $$\mathcal{L}_{\text{BCE}} = -\left[ 30 \cdot y \log \sigma(x) + (1 - y) \log (1 - \sigma(x)) \right]$$
  - The theoretical logit shift induced by `pos_weight=30` is $\Delta \approx \ln(30) \approx +3.40$.
  - **Consequence:** Raw model sigmoid probabilities are shifted heavily towards foreground. At the default threshold of 0.5, the model predicts numerous low-confidence false positives, hurting precision and PQ.
  - **Empirical Validation:** When testing probability thresholds on `best_model.pth`, Dice and PQ increase substantially as threshold is raised from 0.4 to 0.7 (e.g. from 0.52 to 0.63+ on test samples).

### 7. AMP and GradScaler Implementation (`AssertionError: No inf checks`)
* **Status:** Confirmed Root Cause and Defect.
* **Analysis:**
  - The error `AssertionError: No inf checks were recorded for this optimizer.` originates in `torch.amp.grad_scaler.GradScaler.step(optimizer)`.
  - Specifically, line 475 of PyTorch's `grad_scaler.py` checks:
    ```python
    if len(optimizer_state["found_inf_per_device"]) == 0:
        raise AssertionError("No inf checks were recorded for this optimizer.")
    ```
  - `found_inf_per_device` is populated during `unscale_()`. If `optimizer` contains no parameters with non-None gradients (e.g., if `loss.backward()` did not compute gradients for `optimizer.param_groups`, if `requires_grad=False`, if `optimizer.zero_grad()` was invoked between backward and step, or if `scaler.step` was called without a preceding scaled backward), the dictionary is empty, triggering the assertion.
  - **Fix:** In `train.py`, verify gradients are populated, correctly sequence `scaler.scale(loss).backward()`, `scaler.unscale_(optimizer)`, `clip_grad_norm_`, `scaler.step(optimizer)`, and `scaler.update()`.

### 8. Checkpoint Saving, Loading, and Resumption
* **Status:** Confirmed Missing Functionality.
* **Analysis:**
  - `train.py` contained `save_checkpoint(state, filepath, is_best=False)` using atomic temporary file renaming.
  - However, there was **no `load_checkpoint` function** in `src/filament_seg/train.py`.
  - Notebooks and evaluation scripts lacked a standardized, robust checkpoint-loading utility that handles device remapping (`map_location`), `weights_only=True` security, and full state restoration (model, optimizer, scheduler, scaler, epoch, metrics).

### 9. Semantic Dice/IoU Calculation and Metric Aggregation
* **Status:** Verified with Macro/Micro Aggregation Gap.
* **Analysis:**
  - `compute_dice` and `compute_iou` in `evaluate.py` correctly handle the zero-ground-truth / zero-prediction edge case by returning 1.0 when both are empty, and 0.0 when one is empty and the other is not.
  - However, `evaluate_semantic_predictions` computed only **macro-averaged** metrics (mean of per-image scores).
  - When evaluating segmentation, **micro-averaged (global foreground)** Dice/IoU provides an essential complementary view that weights images proportional to their filament area, avoiding disproportionate influence from tiny edge-case masks. Both macro and micro metrics must be reported.

### 10. Instance Extraction, COCO RLE Encoding, and Local PQ Implementation
* **Status:** Confirmed Edge-Case Bug in `diagnostic_panoptic_quality`.
* **Analysis:**
  - **Edge-Case Crash:** In `evaluate.py`, line 161:
    `if array.shape != shape or not np.isin(array, (0, 1)).all() or not array.any(): raise ValueError("instances must share shape and contain binary foreground")`
    If postprocessing or filtering produces an empty mask (0 foreground pixels), calling `diagnostic_panoptic_quality` crashes with `ValueError` instead of gracefully treating it as a non-instance or zero prediction.
  - **Connected Components Limitation:** `scipy.ndimage.label` cannot separate filaments that touch or cross; they are lumped into a single instance, yielding lower Panoptic Quality due to matched IoU degradation and false negative penalization.
  - **Watershed Limitation:** While Watershed can split touching filaments, it tends to over-fragment elongated thin filaments unless core seeds are carefully calibrated.
  - **COCO RLE:** `pycocotools.mask.encode(np.asfortranarray(mask))['counts'].decode('utf-8')` produces standard compressed string counts, matching official competition specs.

### 11. Submission Schema and Competition Alignment
* **Status:** Verified Schema; Documented Scorer Uncertainties.
* **Analysis:**
  - Submission schema requires CSV with headers `filament_id,segmentation_rle`.
  - Format: `<test_image_stem>_<unique_suffix>`, e.g., `20150125172714Mh_1`.
  - 180 test images in `MAGFiLO_1.0_Kaggle_2026/test/test_images/`.
  - Official rules specify "one row per predicted filament". If an image has no predicted filaments, no rows are submitted for that image.
  - **Uncertainty:** The official competition self-evaluation notebook is not publicly viewable (404), so exact tie-breaking, multi-annotator validation aggregation, and matching thresholds on the official Kaggle backend remain unverified. Our local PQ serves as a reliable diagnostic proxy based on the published formula.

---

## Severity Matrix

| Component | Issue Identified | Severity | Status |
| :--- | :--- | :--- | :--- |
| **Validation Transform** | CenterCrop evaluates on empty patches in 33.8% of images | **High** | Fixed in Phase 2 |
| **Checkpoint Utility** | Missing `load_checkpoint` in `src/filament_seg/train.py` | **Medium** | Fixed in Phase 2 |
| **PQ Edge Case** | Empty prediction mask crashes `diagnostic_panoptic_quality` | **Medium** | Fixed in Phase 2 |
| **AMP Step** | Unhandled empty gradients cause `No inf checks` assertion | **Medium** | Fixed in Phase 2 |
| **Loss Calibration** | `pos_weight=30` makes threshold 0.5 overpredict false positives | **Medium** | Addressed in Phase 3 |
| **Tiled Inference** | Previous pipeline lacked seamless full-res reconstruction | **High** | Implemented in Phase 2 |
| **Multi-Annotator** | Validation evaluated only against first annotator | **Low** | Enhanced in Phase 4 |
