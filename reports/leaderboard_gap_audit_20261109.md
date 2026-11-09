# Leaderboard gap audit and provisional confidence-filter candidate

**Status (2026-11-09):** Kaggle submission ref `58150471` used `submission_provisional_unet_resnet34_v2_20261104T094726Z.csv` and received public score **0.31**. The same configuration scores **0.392699** on the existing 71-image internal check. No confirmed CSV-format, orientation, or preprocessing defect explains the gap. The hidden scorer's GT construction and the labels of the 180 test images are unavailable locally.

## Submission and evaluator audit

- [Submitted CSV audit](submitted_csv_audit_20261109_v2.json): SHA-256 matches its provenance; 1,577 rows, valid image stems, consecutive per-image instance suffixes, compressed COCO RLE counts that decode and re-encode exactly as 2048×2048 masks. All 180 test images were processed; 178 have at least one instance. The original v2 checkpoint hash also matches provenance.
- [Submission-path validation replay](submission_validation_replay_20261109.json): the exact `generate_submission` path on 71 internal-check images gives PQ **0.3926985825857285**, SQ **0.668071**, RQ **0.587809**, TP **487**, FP **425**, FN **258**. This matches the cached benchmark to floating-point precision. Fresh model predictions matched the submitted decoded masks pixel for pixel for the first and last test images with predictions; this specifically checks for transpose/flip errors in those examples.
- The archived [organizer self-evaluation notebook v6](reference/organizer_self_evaluation_v6.ipynb) counts every GT/prediction IoU pair strictly above 0.5, then aggregates IoU sums and TP/FP/FN over annotator-image entries. The local `organizer_pq_image` follows this public algorithm and passed synthetic parity tests. The hidden Kaggle evaluator and its GT RLE production are still unverified.
- [Evaluator sensitivity](evaluator_sensitivity_20261109.json): on internal check, pycocotools polygons with all sets score **0.392699**; first set only **0.393857**; OpenCV-rasterized polygons with all sets **0.411304**. These alternatives do not reproduce a decline to 0.31 and do not establish hidden-scorer behavior. No nonexclusive IoU matches occurred in this subset.
- The v2 reference YAML says `normalization: minmax`, but the actual training dataset and submission generator both convert grayscale `uint8` to float and divide by 255 exactly once. This is stale configuration documentation, not an observed runtime mismatch.

## Error and image-distribution findings

- [Tuning error audit](instance_error_audit_tuning_20261104T122811Z.json): 71 images, 106 annotation sets, **483 FP**, **336 FN**, 35 fragmentation events and 38 merge events under explicit 10%-overlap heuristics. False detections are the larger measured error count; the heuristic events are descriptive, not official scorer categories.
- [Component features](component_feature_audit_20261109.json): matched predictions have median mean probability **0.980** and area **1,512 px**; FP predictions have **0.971** and **1,027 px**. Median normalized radial distance is **0.518** for matched versus **0.611** for FP. [Feature plots](figures/component_features_20261109.png) and [baseline error examples](figures/instance_error_fp_20261104T122811Z.png) show representative cases.
- [Unlabeled image-distribution audit](image_distribution_audit_20261109.json): train/validation/test solar-disk median intensities are **126/126/126**, 5th-to-95th percentile contrasts **102/101/102**, and approximate disk radii **1017/1023/1019 px**. Test prediction median area is **1,304 px**, versus **1,370 px** on validation. [Distribution plots](figures/image_distribution_20261109.png) show no large shift in these measured features. Test labels were not used.
- The seven internal-check images sharing a capture date with training score **0.3365**, versus **0.4004** on the other 64. Date overlap is therefore not a supported explanation for an inflated internal score in this small comparison.
- Foreground miss rates near tile starts were 21.61%, versus 21.31% away from them. This does not support a tile-blending change at present.

**Gap assessment:** The approximately **0.083** difference between public and local PQ remains unexplained by a confirmed engineering defect. Local validation reuse, differences in hidden test annotations, or hidden evaluator details remain possible. Similar unlabeled image statistics do not rule out a label-distribution shift. The public leaderboard score is rounded, so the precise difference is not known.

## Targeted improvement and frozen evaluation

One low-risk change was selected from tuning-only evidence: discard a baseline connected component when its **mean foreground probability is below 0.97**. The checkpoint, image normalization, tiling, threshold 0.80, and area minimum 500 remain unchanged. No training was run. [Frozen tuning artifact](component_filter_frozen_20261109T043725Z.json) records all nine tested configurations and the source hash before the [single internal-check run](component_filter_internal_20261109T044117Z.json).

| Protocol | Global PQ | SQ | RQ | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Tuning baseline | 0.351797 | 0.661605 | 0.531732 | 465 | 483 | 336 |
| Tuning confidence 0.97 | 0.378708 | 0.666599 | 0.568120 | 417 | 250 | 384 |
| Internal baseline | 0.392699 | 0.668071 | 0.587809 | 487 | 425 | 258 |
| **Internal frozen confidence 0.97** | **0.409903** | **0.673345** | **0.608757** | **431** | **240** | **314** |

The internal-check PQ gain is **+0.017205**. A paired 5,000-replicate bootstrap by capture date gives a 95% interval of **+0.003248 to +0.029362**. This is uncertainty conditional on the existing validation sample, not independent verification or an official leaderboard estimate. Cached model inference is **0.212 s/image** with **608 MiB peak GPU memory**; measured CPU component extraction/filtering is approximately **0.112 s/image** for both baseline and filtered configurations. Timing is not a controlled production benchmark. [Tuning gain](figures/confidence_filter_tuning_gain_20261109.png) and [tuning loss](figures/confidence_filter_tuning_loss_20261109.png) illustrate the trade-off.

## New local artifact and next decision

The [new provisional submission](../outputs/submission_provisional_v2_confidence_0p97_20261109T044350Z.csv) has **1,134 rows** from all **180** test images; four images have no retained instance. Its [provenance](../outputs/submission_provisional_v2_confidence_0p97_20261109T044350Z.provenance.json) pins the checkpoint, frozen tuning artifact, evaluator caveats, source hashes, and CSV hash. It was **not uploaded**. The original official submission and historical checkpoints remain intact.

An independent full readback verified the exact CSV schema and test-image identifiers, round-trip decoding/re-encoding of every compressed COCO RLE mask at 2048×2048, and the CSV, checkpoint, source, frozen-artifact, and evaluation-report SHA-256 hashes. All **72 regression tests passed**; `git diff --check` passed; the seven historical checkpoint hashes and original official-submission CSV hash remain unchanged.

Kaggle's competition listing on 2026-11-09 still gives the submission cutoff as **2026-11-15 06:00 UTC = 13:00 WIB**. A manual leaderboard test of the new CSV could determine whether this local gain transfers; local evidence alone cannot resolve the public-score gap.
