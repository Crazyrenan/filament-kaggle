# Instance post-processing audit and frozen comparison

Status: **no defensible improvement; original U-Net ResNet34 v2 remains selected.**

## Protocol

- Same original v2 checkpoint and 142 cached full-resolution probability maps as the provisional benchmark. No model training or inference was repeated.
- The 71 tuning images selected a candidate using Global PQ. All seven configurations were frozen in the tuning artifact before one pass over the separate 71-image internal-check partition.
- The organizer-public-notebook-compatible local evaluator scores every annotation set. Hidden Kaggle evaluator parity and an independent holdout remain unverified.
- Cached model inference latency is 0.212 s/image and peak inference VRAM is 608.2 MiB. Candidate post-processing is CPU-only; the table reports its measured time separately.

## Tuning-only error audit

- Across 71 physical images and 106 annotation sets: TP 465, FP 483, FN 336.
- Fragmentation heuristic: 35 GT-set events with at least two predictions each covering at least 10% of the GT area.
- Merge heuristic: 38 prediction-set events with overlap of at least 10% of the prediction area with two or more GT instances.
- GT foreground miss rate within 16 px of tile starts: 21.61%; elsewhere: 21.31%. This small difference is descriptive, not evidence that blending caused errors. An overlap/blending ablation was therefore not run.
- Representative images: [FP](figures/instance_error_fp_20261104T122811Z.png), [FN](figures/instance_error_fn_20261104T122811Z.png), [fragmentation](figures/instance_error_fragmented_gt_heuristic_20261104T122811Z.png), [merge](figures/instance_error_merged_predictions_heuristic_20261104T122811Z.png). The colored overlay depicts GT set 0; the title count sums all annotation sets.

## Frozen candidate results

| Candidate | Tuning PQ | Internal PQ | Δ vs baseline | SQ | RQ | TP | FP | FN | CPU postprocess ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.351797 | 0.392699 | +0.000000 | 0.668071 | 0.587809 | 487 | 425 | 258 | 22.2 |
| morph_open3 | 0.351045 | 0.391610 | -0.001088 | 0.668160 | 0.586103 | 485 | 425 | 260 | 14.2 |
| morph_close3 | 0.352725 | 0.391050 | -0.001648 | 0.668179 | 0.585248 | 484 | 425 | 261 | 14.2 |
| shape_1p5 | 0.349986 | 0.391403 | -0.001296 | 0.667591 | 0.586291 | 479 | 410 | 266 | 13.1 |
| shape_2 | 0.352552 | 0.383176 | -0.009523 | 0.664650 | 0.576508 | 454 | 376 | 291 | 13.0 |
| split_erode3 | 0.353001 | 0.393733 | +0.001035 | 0.667897 | 0.589512 | 489 | 425 | 256 | 13.3 |
| split_erode5 | 0.353729 | 0.389078 | -0.003620 | 0.667957 | 0.582490 | 489 | 445 | 256 | 13.6 |

## Decision

- Tuning selected **split_erode5** (PQ 0.353729); its frozen internal-check PQ was 0.389078, versus 0.392699 baseline.
- Split-erode3 had a small positive internal-check difference but was not the tuning-selected candidate. Choosing it now would use internal-check results for optimization.
- Timing is measured in candidate order without a dedicated warmup; differences of a few milliseconds are approximate. All candidates reuse the same model inference and its measured VRAM.
- No new submission CSV was generated. The prior locally validated 180-image submission and its provenance remain the selected artifacts.

## Provenance

- [Tuning error audit](instance_error_audit_tuning_20261104T122811Z.json)
- [Frozen tuning artifact](postprocessing_tuning_frozen_20261104T123106Z.json)
- [Single internal-check comparison](postprocessing_internal_once_20261104T123333Z.json)
- [Original multi-model benchmark](model_benchmark_20261104T094701Z.json)
