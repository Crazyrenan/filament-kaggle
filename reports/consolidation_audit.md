# Project consolidation audit — 2026-10-31

## Repository and branch audit

Initial state: `feature/unet-resnet34-v2` at `75602ea`, with a modified `notebooks/02_training.ipynb` containing execution-output changes. The worktree had no other untracked files. `main` and `origin/main` were at `93603c4`. One worktree was registered at this project path. Remote: `https://github.com/Crazyrenan/filament-kaggle.git`; remote `HEAD` still advertised `feature/project-initialization`. No tags existed.

Before integration, `codex/pre-consolidation-backup-20261031` was created at `75602ea`. The uncommitted notebook output was preserved in `stash@{0}` with message `pre-consolidation 02_training notebook execution output`; it was not replayed into the canonical notebook because it included stale cell outputs, not source edits. Checkpoint files were recorded by SHA-256 below before any source changes.

| Branch | Initial head | Unique changes relative to `main` | Integration/conflicts | Resolution/final disposition |
|---|---|---|---|---|
| `main`, `origin/main` | `93603c4` | Stage 3.2/3.3 baseline, dataset, instance/RLE infrastructure | Base | Fast-forwarded locally to v2 branch; retained as canonical |
| `feature/pipeline-audit-improvements`, remote counterpart | `140e836` | v1 audit, full-resolution inference/evaluation, validation reports/figures, submission generator | Ancestor of v2; no conflict | Included through v2 fast-forward; branch retained |
| `feature/unet-resnet34-v2` | `75602ea` | v2 config, trained workflow, smoke/resume tests, notebook implementation | Four commits ahead of main; fast-forward, no conflict | All commits integrated locally; branch retained |
| `feature/project-initialization`, remote counterpart | `f4cd912` | Initial repository and dataset normalization | Ancestor of main | Already integrated; retained |
| `codex/pre-consolidation-backup-20261031` | `75602ea` | Recovery reference before cleanup | No merge | Retained |

`git diff main...feature/pipeline-audit-improvements --stat` showed 20 changed paths; `git diff main...feature/unet-resnet34-v2 --stat` showed 22. The v2 branch contained the pipeline-audit commit, so a separate merge of the audit branch would duplicate work. Normal fast-forward integration from `93603c4` to `75602ea` preserved every unique commit. No force push, reset, branch deletion, or checkpoint relocation was used.

## Notebook inventory and disposition

| Original notebook | Purpose and unique value | Final disposition |
|---|---|---|
| `01_eda.ipynb` | Dataset inventory, class imbalance, crop bias, v1 qualitative plot | Retained and extended with COCO geometry, intensity, split, annotator agreement, and instance visualization; 9 code cells passed fresh execution |
| `02_training.ipynb` | v2 preparation, smoke/full training, resume, evaluation, comparison | Retained; explicit safe gates, root paths, checkpoint metadata/history; 12 code cells passed from root and `notebooks` |
| `03_submission.ipynb` | Test preview and CSV creation | Retained; trained v2 checkpoint required, complete 180-image generation opt-in, existing CSV protected; 6 code cells passed from both working directories |
| Five `02_training_verified*`/`03_submission_verified*` notebooks | Historical Stage 3.2/3.3 executed outputs, already summarized in `stage_3_3_verification.md` | Removed from the active notebook directory as tracked Git deletions; recoverable from `93603c4` and backup branch |

New `04_instance_error_analysis.ipynb` (3 code cells) and `05_postprocessing_optimization.ipynb` (3 code cells) passed fresh execution. `scripts/update_canonical_notebooks.py` was a stale generator for the old notebook contents; it was removed after its history was preserved. The 33-row, five-image CSV was renamed to `outputs/historical_partial_submission_5_images.csv` to prevent mistaking it for a full test submission; its content was preserved. The older `outputs/verification/02_training.executed.20261030_110102.ipynb` and research reports remain tracked historical artifacts outside the canonical notebook directory.

## EDA restoration and data integrity

Fresh EDA confirmed 707 physical train JPEGs, 180 test JPEGs, 1,154 COCO image records, 8,199 annotations, and 2048 × 2048 dimensions. Annotation-set multiplicities were 411/145/151 filenames with one/two/three sets. All annotation IDs and references were valid; the sampled structural polygon check found zero malformed polygons. COCO-area quantiles (0/10/50/90/100%) were 9/409.8/1228/4684.6/37739 pixels. Primary-set instance count had mean 7.37, maximum 26. In 50 primary-set masks, mean foreground fraction was 0.420% (range 0.025–1.101%). Eleven of the first 30 center crops were empty (36.7%). Thirty sampled images had mean grayscale intensity 82.30/255 and mean standard deviation 56.31. A seed-42 split reproduced 565/142 filenames with zero exact overlap; 13 capture dates occurred on both sides. Three sampled alternative-set foreground IoUs relative to set 0 included 0.416–0.589, confirming real annotator variation. These sampled figures are not population estimates.

The notebook retains the earlier v1 prediction plot and adds primary-set instance colors. Outstanding EDA: exhaustive file corruption and duplicate-pixel checks, full-dataset foreground/crop statistics, quantitative fragmentation/overlap and foreground/background contrast. Historical EDA in Git had only initial file-format scans, which are covered by the current `DatasetInspector` cell.

## Training, checkpoints, and submission safety

`project_path()` now anchors configured checkpoint paths to the checkout, while preserving absolute test paths. The v2 trainer writes unique smoke files under `smoke_runs/`; production resume reads only `latest.pth`; a completed 30/30 checkpoint reports `already_completed`; a fresh run refuses to overwrite existing weights; new checkpoints include purpose, experiment, model configuration and root-resolved checkpoint directory. Legacy checkpoints have no such metadata, so architecture compatibility is enforced by strict state-dict loading and their known paths/epochs. `configs/unet_resnet34_v2.yaml` now says `val_dice_patch`, matching actual best-checkpoint selection. No full training was launched.

Existing ignored checkpoint SHA-256 values remained identical after all notebook checks:

| File | Epoch/purpose | SHA-256 |
|---|---|---|
| `checkpoints/unet_resnet34/best_model.pth` | v1 best, epoch 11 | `E3E7218969AD35E880C2BD80C2CC53EE5496503F2B2FA65ABCBA9D1606E587CD` |
| `checkpoints/unet_resnet34/latest.pth` | v1 latest, epoch 18 | `F7329C3AE7CC53A4EE8C12702126CA7127C94D923D9D7BEED19BB00BA6F2E0BC` |
| `checkpoints/unet_resnet34_v2/best_model.pth` | v2 best, epoch 29 | `F9422022D65C0DE79C61020B9CE5D81082C46B8919DDB5B1A6880624B02018C4` |
| `checkpoints/unet_resnet34_v2/latest.pth` | v2 latest, epoch 30 | `670C9A0837F0F8248471A7F1FF1B104D8C1A2AFE7AC9BC7E1EF3CE5AF4D0D75B` |
| `checkpoints/unet_resnet34_v2/smoke_checkpoint.pth` | legacy smoke, epoch 0 | `6551611B469BF03DFF419DA87C316FE3B6836C93BCCE4E96A9168FBEB5A3482D` |
| `notebooks/checkpoints/unet_resnet34_v2/smoke_checkpoint.pth` | misplaced legacy smoke, epoch 0 | `C6CAE2F81A4D27E019E2620A3462D2EFA84B127DC4B16271236B97AFEA20AE03` |

The ignored v1 `epoch_001.pth` also remained unchanged (`EF1E73D4000C8D1CB578F811D031CC93C1E9E12F70B9255E1CE9007265AE4BA1`). The misplaced legacy smoke was **not** moved or deleted.

`03_submission.ipynb` loads the trained v2 best checkpoint independently, checks epoch/purpose, previews one test and one validation image, and defaults full CSV generation off. `generate_submission()` rejects duplicate test stems and existing output paths before inference; exclusive creation prevents an overwrite race. The prior five-image file remains historical. Structural RLE validation is distinct from Kaggle acceptance.

The required `filament_id,segmentation_rle` columns and compressed full-resolution COCO counts were established in the earlier [competition target audit](competition_target_audit.md), which links to the [official Kaggle specification](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates). A fresh web fetch during this consolidation returned no readable page text, so it did not add new official evidence or resolve scorer uncertainty.

## Evaluation and bounded post-processing results

`diagnostic_panoptic_quality(..., return_details=True)` now returns matched GT/prediction indices and IoUs plus unmatched indices, using the same assignment as summary PQ/SQ/RQ. The 04 notebook uses actual primary-set COCO instances and trained v2 weights. On **two selected validation images** at threshold 0.75/area 300: `20110322005814Mh.jpeg` had PQ 0.6416, TP/FP/FN 1/0/0; `20111011112054Uh.jpeg` had PQ 0.2032, TP/FP/FN 2/9/0. The second image's dominant observed error was nine false components (areas 317–1801 pixels). These two images do not explain historical full-validation 695/742/291 counts.

The 05 notebook ran inference **once** for each of four fixed validation images and reused the probability maps for three configurations. On the two-image tuning subset, diagnostic global PQ was 0.2617 (0.75/300), 0.2629 (0.70/300), and 0.3020 (0.75/500). The selected 0.75/500 setting scored 0.5460 on a disjoint two-image holdout (TP/FP/FN 12/3/3). These tiny, different subsets cannot establish improvement over historical full-validation v1 0.275885, v2 0.386869, or v2 0.400488 references. Those v2 numbers were supplied as historical references and have not been reproduced in this consolidation. No morphology, overlap, or full-validation sweep was run; the notebook documents controlled follow-up experiments rather than inventing their outcomes.

## Validation

Environment: `C:\Users\Vinny\Project\Kaggle\Solar Filament Segmentation\.conda\python.exe`, Python 3.11.18, torch 2.14.1+cu126, NVIDIA RTX 3060 selected by notebooks. Baseline before changes: **43 passed**. After changes: **47 passed** (including checkpoint preservation, completed resume, cwd independence, cached scoring, matching details, and submission overwrite protection). Fresh nbclient runs passed all canonical notebooks from root; `02_training` and `03_submission` also passed from `notebooks/`. Executed copies are outside this Git checkout at `C:\Users\Vinny\Documents\ChatGPT\filament\notebook_runs`, preserving canonical notebooks without new outputs. `git diff --check` passed. Notebook kernels emitted a Windows ZMQ event-loop warning; execution succeeded.

A direct `train_v2_model(..., resume=True)` call from the `notebooks/` working directory loaded the root `latest.pth` and returned `already_completed` with 30 history entries; it ran no epoch. All five canonical files passed `nbformat.validate` after updating their v4 minor version and cell IDs. Source and tests passed `compileall`.

## Remaining priorities and limits

1. Verify organizer scorer matching, empty-image conventions, multi-annotator aggregation, and official CSV acceptance. Local PQ remains diagnostic.
2. Recompute v2 full-validation PQ and error categories with a fixed checkpoint, annotation policy, tiling, threshold/area, and holdout protocol. Existing v2 historical numbers are unverified here.
3. Expand instance-level error coding (fragmentation, merging, limb, boundary) and compare annotator sets without leakage. The two-image notebook is a starting audit.
4. Test morphology, confidence/elongation filtering, and 0/64/128/256-pixel tiling overlap one family at a time; new tiling requires fresh inference. Track GPU memory and latency.
5. Test a temporal/grouped holdout and full-dataset quality scans before treating random-image validation as generalization evidence.
6. Review the final 180-image CSV and Kaggle acceptance before submission. The local schema check alone is insufficient.
