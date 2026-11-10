# Controlled v2/v3 probability ensemble

Status: complete, **provisional local evaluation; no new submission**. The official public leaderboard baseline remains original U-Net ResNet34 v2 with score 0.33.

## Audit and protocol

- Checkpoints: original v2 SHA-256 `f9422022d65c0de79c61020b9ce5d81082c46b8919ddb5b1a6880624b02018c4`; hard-negative v3 SHA-256 `302aa58783fce9c4c664ac42aa9ece988a2f67e27545ae87768cb4c12b572a23`. Seven historical checkpoint size/hash pairs matched `reports/historical_checkpoint_hashes.json` before and after the experiment. Neither model was retrained or modified.
- Preflight artifact: `reports/ensemble_v2_v3_01_preflight.json`. Both caches contain all 142 validation images as finite float32 2048×2048 probability maps in [0, 1]. The index sets and index-to-filename mappings match. Their manifest validation fingerprints differ only because the saved index order differs. Both used grayscale `/255` once, 512-pixel Hann-blended tiles, 128-pixel overlap, and batch size 4. Five live two-model inferences matched both cached maps exactly (`max_abs_live_vs_cache_difference = 0`). Source and cache-manifest SHA-256 hashes are in the preflight and frozen artifacts.
- Partition: 71 tuning and 71 internal-check physical images, disjoint by index. The internal check was historically exposed to original-v2 model selection and is not independent. Local scoring follows `filament_seg.official_metric.organizer_pq_image` over all annotation alternatives per physical image; hidden Kaggle evaluator parity remains unverified.
- Search space was fixed before tuning: v2 probability weight {0.25, 0.50, 0.75}; mask threshold {0.75, 0.80}; minimum component area {300, 500}; mean component confidence {0.96, 0.97}. All 24 configurations were scored **only on tuning**. The selection rule was Global PQ, then SQ, fewer FP, and descending weight/confidence/area/threshold. The grid's encoded-RLE scorer was checked against direct organizer scoring in unit tests and on all 71 selected tuning images. The selected configuration was written to `reports/ensemble_v2_v3_01_frozen_20261110T001241Z.json` before the one internal-check evaluation.

## Frozen candidate and measured results

The tuning-selected ensemble is **75% v2 + 25% v3 probabilities**, mask threshold **0.75**, minimum area **300**, mean component confidence **0.96**. Results use Global PQ aggregation across all alternative annotation sets.

| 71-image subset | Global PQ | SQ | RQ | TP | FP | FN | CPU blend + postprocess s/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tuning, v2 confidence-0.97 baseline | 0.378708 | 0.666599 | 0.568120 | 417 | 250 | 384 | 0.1597* |
| Tuning, frozen ensemble | 0.393426 | 0.665970 | 0.590757 | 441 | 251 | 360 | 0.1169 |
| Internal check, v2 confidence-0.97 baseline | 0.409903 | 0.673345 | 0.608757 | 431 | 240 | 314 | 0.1124* |
| Internal check, frozen ensemble | 0.414304 | 0.669007 | 0.619282 | 440 | 236 | 305 | 0.1048 |

*Historical v2 timing measures post-processing only; ensemble timing includes blending and post-processing. These CPU timings were measured in separate runs.

With both checkpoints resident on the RTX 3060, five tuning-image live inferences took **0.4233 seconds per image** on average for both models together (individual observations: 0.5477, 0.3924, 0.3923, 0.3899, 0.3944 s). Peak allocated CUDA memory was **702.9 MiB**, peak reserved memory **934.0 MiB**. The first observation includes initial runtime overhead. These are measured values, not the sum of separately cached timings.

The single internal-check result is `reports/ensemble_v2_v3_01_internal_20261110T001321Z.json`. Global PQ difference from the v2 baseline was **+0.004401**. Paired capture-date bootstrap (5000 replicates, seed 42) gave a 95% difference interval of **[-0.005601, +0.013921]**. The preset submission rule required a gain >0.005, a positive interval lower bound, and at least 90% of baseline TP. The first two requirements failed. The candidate was not changed after observing internal results.

## Decision and verification

- **Retain** `outputs/submission_provisional_v2_confidence_0p97_20261109T044350Z.csv` as the current best official submission. Its SHA-256 remained `7e56a491d467e17d8c91570f984b28dddcf46d39ac2131adb7a2f73d82e4d265`.
- No ensemble CSV was generated or uploaded to Kaggle. Both checkpoints, all cached predictions, previous submissions, and provenance artifacts remain unchanged.
- Full regression suite passed, including ensemble mask-alignment and organizer-scoring parity tests. All seven historical checkpoint hashes matched after evaluation.

This result is provisional because the original v2 was previously evaluated on the internal-check images and hidden evaluator parity is unknown. The small local gain does not justify replacing the 0.33 official baseline.
