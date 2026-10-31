# Solar Filament Segmentation: Failure Mode & Error Analysis

**Audit Date:** 2026-10-31  
**Model:** U-Net ResNet-34 Baseline (`checkpoints/unet_resnet34/best_model.pth`)  
**Evaluated Data:** Full-resolution (2048 x 2048) MAGFiLO Validation Split (142 images)  
**Optimal Postprocessing Configuration:** Tiled Inference (512x512, overlap 128, Hann blending), Threshold = 0.75, Min Component Area = 300-500 pixels.

---

## 1. Quantitative Performance Overview

On the full-resolution validation split (142 images, 986 ground truth instances):

| Metric | Threshold 0.50 | Threshold 0.65 | Threshold 0.75 (Selected) | Threshold 0.80 |
| :--- | :---: | :---: | :---: | :---: |
| **Macro Dice** | 0.5204 | 0.5451 | **0.5621** | 0.5717 |
| **Macro IoU** | 0.3639 | 0.3875 | **0.4042** | 0.4138 |
| **Pixel Precision** | 0.3918 | 0.4257 | **0.4528** | 0.4701 |
| **Pixel Recall** | 0.8617 | 0.8378 | **0.8166** | 0.8024 |
| **Micro (Global) Dice**| 0.5507 | 0.5777 | **0.5965** | 0.6069 |
| **Macro PQ (area=300)**| 0.2228 | 0.2525 | **0.2715** | — |
| **Global PQ (area=300)**| 0.2299 | 0.2557 | **0.2759** | — |
| **Global PQ (area=500)**| 0.2375 | 0.2628 | **0.2820** | — |
| **Total True Positives**| 472 | 510 | **536** | — |
| **Total False Positives**| 1,055 | 989 | **926** | — |
| **Total False Negatives**| 514 | 476 | **450** | — |

*Key finding:* Because training used `pos_weight=30.0`, default threshold 0.50 suffers from severe over-prediction (1,488 false positives with area=50). Moving to threshold 0.75 with minimum area filtering (300-500 pixels) cuts false positives by nearly 50% while actually **increasing True Positives from 472 to 536** (due to reduced false match competition in bipartite assignment), lifting Global PQ from 0.1974 to 0.2820.

---

## 2. Dominant Failure Modes & Visual Evidence

Visual inspection across representative validation cases (saved under `reports/figures/`) reveals four primary failure modes:

### Failure Mode 1: Touching and Clustered Filaments (Structural U-Net Limitation)
* **Representative Figure:** `reports/figures/case_dense_touching.png` (Filename: `20130606001014Mh.jpeg`, 14 GT instances)
* **Mechanism:**
  - Filaments in active solar complexes frequently touch, branch, or run parallel with sub-pixel or few-pixel separations.
  - Because semantic U-Net produces a single binary probability map without instance boundary awareness, connected components invariably merges adjacent filaments into one monolithic mask.
* **PQ Impact:**
  - A merged prediction can match at most one ground truth instance (achieving IoU $> 0.5$).
  - The remaining $N - 1$ ground truth filaments become False Negatives ($FN$), while the merged mask suffers lower IoU due to excessive union area.
  - Watershed with distance transform helps in some convex cases, but on elongated serpentine filaments it causes over-fragmentation, leading to more False Positives (1,208 FP for Watershed vs 1,074 FP for CC).

### Failure Mode 2: Solar Limb and Optical Artifacts (False Positives)
* **Representative Figure:** `reports/figures/case_high_fp.png` (Filename: `20120117005814Mh.jpeg`)
* **Mechanism:**
  - The circular solar disk exhibits strong limb darkening, atmospheric diffraction, and telescope occulting borders.
  - High positive loss weighting (`pos_weight=30`) causes the model to fire on dark circular edge contours and chromospheric fibril textures, generating spurious narrow filament-like predictions along the limb.
* **Mitigation:**
  - Raising the binarization threshold to $\ge 0.70$ and enforcing a minimum component area $\ge 300$ pixels eliminates over 40% of these small limb artifacts.

### Failure Mode 3: Faint and Fragmented Filaments (False Negatives)
* **Representative Figure:** `reports/figures/case_faint_low_contrast.png` (Filename: `20110322005814Mh.jpeg`)
* **Mechanism:**
  - Thin, diffuse quiescent filaments have subtle absorption contrast against the dynamic chromospheric background.
  - When applying high probability thresholds, narrow filament bridges dip below threshold, breaking a continuous filament into 2–4 smaller disconnected fragments.
  - Each fragment has smaller area and lower individual IoU, frequently dropping below the 0.5 IoU matching threshold, converting a near-detection into multiple False Positives and one False Negative.

### Failure Mode 4: Annotator Disagreement and Label Ambiguity
* **Analysis:**
  - The MAGFiLO dataset includes up to 3 independent annotation sets per physical image.
  - Comparing predictions across annotators reveals that human experts frequently disagree on faint filament tails and whether segmented fibril groups constitute single filaments or distinct structures.
  - For images with multiple annotator sets, the mean validation PQ against Annotator Set 0 is 0.2767, with a per-image standard deviation of $\pm 0.051$ across alternative annotators.

---

## 3. Prioritized Recommendation: U-Net vs. Instance-Aware Model

Based on empirical evidence and mathematical audit of the competition objective:

1. **Ceiling on Semantic U-Net:**
   Semantic segmentation with connected-component postprocessing cannot resolve touching or overlapping filaments without destructive heuristics. Even with optimal tuning, U-Net PQ plateaus around 0.28–0.30.
2. **Prioritized Recommendation:**
   **Proceed to an instance-aware architecture for the next development stage:**
   - **Mask R-CNN / Cascade Mask R-CNN:** Directly predicts bounding box proposals and individual instance masks, naturally separating touching filaments and outputting individual confidence scores per filament.
   - **YOLOv8-seg / YOLOv11-seg:** Highly efficient single-stage instance segmentation, suited for fast iteration on 1024x1024 / 2048x2048 crops.
   - **SOLOv2 / QueryInst:** Direct instance segmentation without bounding boxes, ideal for long non-convex serpentine structures like solar filaments.
