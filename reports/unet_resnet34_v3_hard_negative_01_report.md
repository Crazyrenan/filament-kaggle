# U-Net ResNet34 v3 hard-negative experiment

Status: completed, provisional local evaluation; **no new submission**. The official public leaderboard baseline remains original v2 plus mean component confidence >= 0.97 (score 0.33). The existing CSV and every historical checkpoint were preserved.

## Provenance and leakage controls

- Experiment: `unet_resnet34_v3_hard_negative_01`; seed 43; config `configs/unet_resnet34_v3_hard_negative_01.yaml`.
- Parent: `checkpoints/unet_resnet34_v2/best_model.pth`, SHA-256 `f9422022d65c0de79c61020b9ce5d81082c46b8919ddb5b1a6880624b02018c4`. V3 was initialized from these weights, then fine-tuned. No historical checkpoint was written.
- Mining: `reports/unet_resnet34_v3_hard_negative_01_mining.json`; 565 training images examined, 398 candidates in 264 images before the pretraining sample review. The configured outer-disk rule (normalized centroid radius >= 0.75) selected 231 candidates in 174 training images. Candidates required v2 probability >= 0.8, connected-component area >= 500, mean confidence >= 0.95, no overlap with a 16-pixel halo around the **union of all annotation alternatives**, and an annotation-empty 512 × 512 crop. The outer-disk restriction reduces the chance that an unannotated central filament is used as a negative, but cannot eliminate incomplete-label risk.
- The mining loop only indexed the fixed 565-image training split. The hard-negative sampler rejects any filename outside that split. The 71 tuning and 71 internal-check filenames, and 180 test filenames, were verified disjoint from mining filenames. No internal-check labels or test labels were used for mining, training, checkpoint selection, or post-processing tuning. Tuning annotations were used for early stopping and post-processing selection.
- Train annotation JSON SHA-256 `5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1`. Partition SHA-256 `51399b702658c0e057e47f6242d671f6509382d24203c60211bc3448d1776270`. The training checkpoint's `provenance` field records the exact mining, config, and relevant training-source hashes. Model, loss, and preprocessing source hashes are `e51eea635bbbf8550c439eabe7b977256d7a19913bc753771e9a71739f9a1dbc`, `8654091d58c39ed764bce4362ed844255ce9e4d0f1b9ed009fc68d42f41a37f5`, and `2bbe852f44409590c4eada97b3982fda960118e9a5a7422214fcdf057121f727` respectively.

## Training

Project interpreter: `.conda/python.exe`, Python 3.11.18, PyTorch 2.14.1+cu126, NVIDIA GeForce RTX 3060. Conservative AdamW learning rates: encoder 5e-6, decoder/head 2e-5; mixed precision, batch size 4, BCE/Dice loss, positive class weight 6, cosine schedule. Each epoch sampled about 30% mined background, 40% foreground-centered, and 30% random patches, with matching flips and 90-degree rotations for image and mask. Grayscale values were divided by 255 once.

Training stopped after epoch 6 under the configured patience of 3. Best checkpoint was epoch 3, tuning center-patch Dice 0.679426. Six epochs took 175.79 seconds total; 1016 hard-negative, 1356 foreground, and 1018 random patches were sampled. Peak allocated GPU memory was 1226.67 MiB; end-of-epoch `nvidia-smi` snapshots showed 28–44% utilization, which are snapshots rather than a training-run average. Best checkpoint: `checkpoints/unet_resnet34_v3_hard_negative_01/best_model.pth`, SHA-256 `302aa58783fce9c4c664ac42aa9ece988a2f67e27545ae87768cb4c12b572a23`. `latest.pth`, `history.json`, `metrics.json`, `config.yaml`, and `provenance.json` in the same isolated directory support recovery and audit.

## Evaluation

The v2 and v3 models used the same 512-pixel tiled inference with 128-pixel overlap and the same organizer-notebook-compatible local PQ implementation. Each physical image's alternative annotation sets were scored separately. Fixed post-processing was mask threshold 0.80, minimum component area 500, and mean component confidence >= 0.97. Four v3 mean-confidence cutoffs (0.95, 0.96, 0.97, 0.975) were scored on tuning only. The 0.96 configuration was frozen in `reports/unet_resnet34_v3_hard_negative_01_frozen_20261109T232907Z.json` before a single internal-check evaluation, recorded in `reports/unet_resnet34_v3_hard_negative_01_internal_20261109T233024Z.json`.

| Subset and model | Global PQ | SQ | RQ | TP | FP | FN | Inference s/image | Postprocess s/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tuning, original v2 fixed | 0.378708 | 0.666599 | 0.568120 | 417 | 250 | 384 | 0.2090 | 0.1597 |
| Tuning, v3 fixed | 0.338576 | 0.678779 | 0.498801 | 312 | 138 | 489 | 0.2018 | 0.1361 |
| Tuning, v3 selected 0.96 | 0.385047 | 0.666743 | 0.577503 | 421 | 236 | 380 | 0.2018 | 0.1149 |
| Internal check, original v2 fixed | 0.409903 | 0.673345 | 0.608757 | 431 | 240 | 314 | 0.2141 | 0.1280 |
| Internal check, v3 fixed | 0.377595 | 0.689225 | 0.547855 | 332 | 135 | 413 | 0.2062 | 0.1003 |
| Internal check, v3 selected 0.96 | 0.410757 | 0.673642 | 0.609756 | 425 | 224 | 320 | 0.2062 | 0.1001 |

V2 and v3 cached inference peak allocated VRAM were each 608.18 MiB.

The selected v3 internal-check PQ difference from the v2 baseline was **+0.000854**. Its paired capture-date bootstrap 95% interval (5000 replicates, seed 42) was **[-0.012175, +0.012684]**. TP fell from 431 to 425 while FP fell from 240 to 224. The interval crosses zero and the point difference is below the predeclared 0.005 gate. V3 therefore does not demonstrate a defensible gain; no new submission CSV was generated. The internal check is historically exposed for v2 and cannot establish independent or official-evaluator performance.

## Validation and decision

- Full regression suite: **77 passed** (`.conda/python.exe -m pytest -q`).
- All **7 historical checkpoint** sizes and SHA-256 hashes matched `reports/historical_checkpoint_hashes.json` after training and evaluation.
- Existing official-baseline CSV SHA-256 remained `7e56a491d467e17d8c91570f984b28dddcf46d39ac2131adb7a2f73d82e4d265`.
- No v3 submission was generated or uploaded. Retain `outputs/submission_provisional_v2_confidence_0p97_20261109T044350Z.csv` as the current best official submission.

The organizer-notebook-compatible local evaluator is a provisional proxy; parity with Kaggle's hidden evaluator is unverified. The v3 checkpoint and probability cache remain available for future diagnostics without modifying the v2 artifacts.
