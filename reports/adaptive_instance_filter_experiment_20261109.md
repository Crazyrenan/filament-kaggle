# Adaptive instance-filter experiment — negative result

**Decision:** Retain the existing confidence-0.97 U-Net v2 submission. Kaggle submissions `58150471` and `58152503` scored **0.31** (unfiltered original v2) and **0.33** (confidence 0.97) respectively. No third CSV was generated or uploaded.

## Protocol and provenance

- Model: unchanged original U-Net ResNet34 v2 [checkpoint](../checkpoints/unet_resnet34_v2/best_model.pth), SHA-256 `f9422022d65c0de79c61020b9ce5d81082c46b8919ddb5b1a6880624b02018c4`.
- Predictions: the original full-resolution probability cache, with 512-pixel tiles and 128-pixel overlap. Baseline extraction remains threshold 0.80, minimum area 500, and mean component probability at least 0.97.
- [Tuning-only frozen artifact](adaptive_filter_tuning_frozen_20261109T052841Z.json): five predefined rules on 71 tuning images, selected by Global PQ (SQ tie break). It records checkpoint, cache, candidate-source, baseline-submission, and partition hashes before internal scoring.
- [Single internal-check comparison](adaptive_filter_internal_once_20261109T052939Z.json): only the unchanged 0.97 baseline and the tuning-selected rule were scored on the 71 internal-check images. The [decision manifest](adaptive_filter_decision_20261109.json) pins both artifacts and their hashes.
- Evaluator: local implementation of the organizer's public notebook v6, scoring each annotation set. Hidden Kaggle scorer parity and independent-holdout validity remain unverified.

## Error analysis

The [component-level audit](confidence_loss_audit_20261109.json) confirms that the confidence-0.97 filter removed **185 false-positive events** and lost **56 matched events** on the internal check. Each event is counted per annotation set. These internal labels were used only for diagnostic reporting; rule thresholds were based on tuning data.

| Tuning category rejected by 0.97 | Events | Median mean confidence | Median q25 confidence | Median area | Median elongation | Median matched IoU |
|---|---:|---:|---:|---:|---:|---:|
| Matched filament | 48 | 0.9640 | 0.9493 | 1,171 px | 4.93 | 0.6184 |
| False detection | 233 | 0.9583 | 0.9389 | 864 px | 3.81 | — |

Matched components usually have higher area, elongation, and lower-tail confidence, but the distributions overlap. Among all matched tuning components, mean probability has Spearman correlation **0.409** with matched IoU; the 10th-percentile probability has **0.432**, while the 90th percentile is nearly saturated and correlates only **0.137**. The [feature plots](figures/confidence_loss_features_20261109.png) show the overlap. Bounding-box aspect, extent, and compactness are recorded in the audit JSON.

For rejected matched versus rejected FP events on tuning, median 10th-percentile confidence is **0.886 vs 0.878**, median 90th-percentile confidence **0.998 vs 0.996**, bounding-box aspect **1.69 vs 1.59**, box extent **0.287 vs 0.355**, and compactness **0.226 vs 0.307**. The lower extent and compactness are consistent with thinner, less blob-like components, but neither descriptor separates the classes cleanly.

## Tuning selection

Four conservative rescue rules were predefined from the tuning medians. A component already passing mean confidence 0.97 is retained. The selected rule also rescues a rejected component when **area ≥1,000 pixels, covariance-axis elongation ≥4.5, and 25th-percentile probability ≥0.94**.

| Rule | Tuning Global PQ | TP | FP | FN |
|---|---:|---:|---:|---:|
| Confidence 0.97 baseline | 0.378708 | 417 | 250 | 384 |
| Area + elongation | 0.381602 | 438 | 286 | 363 |
| Area + q25 confidence | 0.379345 | 432 | 281 | 369 |
| **Area + elongation + q25** | **0.385482** | **437** | **268** | **364** |
| Area + compactness + q25 | 0.384180 | 436 | 271 | 365 |

[Representative tuning gain](figures/adaptive_filter_tuning_gain_20261109.png) and [tuning loss](figures/adaptive_filter_tuning_loss_20261109.png) show both outcomes. Their colored overlays display annotation set 0; title PQ changes aggregate all sets for the image.

## Frozen internal-check result

| Configuration | Global PQ | SQ | RQ | TP | FP | FN | CPU post-processing |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Confidence 0.97 baseline** | **0.409903** | **0.673345** | **0.608757** | 431 | 240 | 314 | 0.109 s/image |
| Frozen adaptive rescue | 0.406587 | 0.670915 | 0.606019 | 443 | 274 | 302 | 0.107 s/image |

The candidate gained 12 TP but added 34 FP; **Global PQ changed by −0.003316**. A paired 5,000-replicate bootstrap over 70 capture-date groups gives a 95% interval of **−0.008828 to +0.001951**. The interval includes zero and the point estimate is negative. CPU timing differences of a few milliseconds are noise at this measurement scale. Cached model inference is 0.212 s/image with 608 MiB peak GPU memory; post-processing uses the CPU and does not change model VRAM.

The tuning-selected rule did not generalize to the existing internal check. Selecting a different rule using its internal outcome would violate the frozen protocol. The original 0.33 Kaggle submission and its provenance remain the best available official artifact. No checkpoint, existing submission, probability cache, or historical manifest was overwritten.

## Final verification

All **74 regression tests passed** with the project's `.conda` interpreter. `git diff --check` passed. SHA-256 checks passed for all seven recorded historical checkpoints, all three existing complete submission CSVs and their provenance, the frozen adaptive-filter source, and the tuning-to-internal artifact link. The output directory still has exactly three complete submission CSVs; the adaptive rule produced none.
