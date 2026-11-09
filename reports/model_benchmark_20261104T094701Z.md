# Provisional multi-architecture benchmark

Run: `model_benchmark_20261104T094701Z.json`

Selection is provisional. Original v2 previously used all 142 validation images; the internal-check partition is not independent. Pair counting was checked against the organizer's public notebook v6 on synthetic cases; hidden Kaggle scorer and GT conversion parity remain unverified.

| Experiment | Protocol | Images | Global PQ | SQ | RQ | TP | FP | FN | Inference s/image | Peak VRAM MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unet_resnet34_v2 | Fixed full 142 | 142 | 0.364811 | 0.661987 | 0.551085 | 952 | 957 | 594 | 0.212 | 608.2 |
| unet_resnet34_v2 | Fixed internal | 71 | 0.384230 | 0.666846 | 0.576190 | 484 | 451 | 261 | 0.212 | 608.2 |
| unet_resnet34_v2 | Tuned internal | 71 | 0.392699 | 0.668071 | 0.587809 | 487 | 425 | 258 | 0.212 | 608.2 |
| unet_resnet34_v2_retrain_02 | Fixed full 142 | 142 | 0.341178 | 0.652115 | 0.523187 | 880 | 938 | 666 | 0.207 | 608.2 |
| unet_resnet34_v2_retrain_02 | Fixed internal | 71 | 0.361374 | 0.654739 | 0.551936 | 449 | 433 | 296 | 0.207 | 608.2 |
| unet_resnet34_v2_retrain_02 | Tuned internal | 71 | 0.349502 | 0.652666 | 0.535499 | 445 | 472 | 300 | 0.207 | 608.2 |
| segformer_b2_01 | Fixed full 142 | 142 | 0.279637 | 0.636455 | 0.439366 | 846 | 1459 | 700 | 0.502 | 2304.5 |
| segformer_b2_01 | Fixed internal | 71 | 0.291126 | 0.639791 | 0.455032 | 425 | 698 | 320 | 0.502 | 2304.5 |
| segformer_b2_01 | Tuned internal | 71 | 0.296237 | 0.652622 | 0.453918 | 362 | 488 | 383 | 0.502 | 2304.5 |

## Selection

Provisional winner: **unet_resnet34_v2** by tuned internal-check Global PQ.

Post-processing was selected on tuning images only. The fixed full-142 results are historical comparisons and did not select the winner.

The earlier 0.386869 and 0.400488 original-v2 PQ references used a different local protocol and are not directly comparable to this benchmark.
