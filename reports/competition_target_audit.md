# Stage 3.1 — Competition target and evaluation audit

Audited 2026-10-31. Local inspection used `C:\Users\Vinny\Project\Kaggle\Solar Filament Segmentation\.conda\python.exe` (Python 3.11.18); `.conda\Scripts\python.exe` does not exist in this checkout. No training or dataset modification was performed.

## 1. Competition target — confirmed official requirement

The required prediction is **instance segmentation of individual solar filaments**: a separate binary mask for each predicted filament, rather than one foreground mask for the whole image or a multiclass map. The organizers say the competition focuses on segmentation masks, not the supplied bounding boxes, spine coordinates, or Left/Right/Unidentifiable/Ambiguous labels. [Official task and MAGFiLO description](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking), [official submission specification](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates).

The input at inference is the H-alpha image from the test directory. The organizers state that other ground-truth metadata may not be used to train the model. [Official evaluation rules](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates).

## 2. Evaluation metric — confirmed formula, implementation details pending

The **updated leaderboard metric is Panoptic Quality (PQ)**, replacing the earlier Dice criterion. For matched ground-truth/predicted segments, the organizers define

$$
\mathrm{PQ}=\frac{\sum_{(g,p)\in TP}\mathrm{IoU}(g,p)}{|TP|+0.5|FP|+0.5|FN|},
\qquad \mathrm{IoU}(g,p)=\frac{|g\cap p|}{|g\cup p|}.
$$

Unmatched predictions and missed ground-truth filaments contribute to the denominator, so fragmentation and merging can lower the score even when foreground-pixel overlap looks good. The competition also describes Dice/IoU distributions and one-to-many/many-to-one relations in its broader judging rubric; it assigns 70% to quantitative and 30% to qualitative review. [Official updated goal, evaluation, and leaderboard ranking](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking).

**Implementation uncertainty:** The public overview does not specify the exact pair-matching threshold, tie-breaking, aggregation across images/annotation sets, or the stated edge-case checks. The organizer points to a [self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) as the implementation, but its source was not accessible in this audit (the public Kaggle API pull returned HTTP 404). These details must be checked against that notebook or an accepted submission before claiming a bit-for-bit local PQ replica. An older sentence in the official submission section still mentions mean Dice on part of the test set; the dated August 7 update explicitly replaces Dice with PQ, so treat that sentence as a documentation inconsistency. [Official overview](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates), [organizer discussion confirming the self-evaluation notebook](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/727186).

The repository's [`evaluate.py`](../src/filament_seg/evaluate.py) and [`train.py`](../src/filament_seg/train.py) compute semantic IoU/Dice, not the official instance-aware PQ. They are useful auxiliary metrics but cannot select the final competition model alone.

## 3. Submission format — confirmed official requirement

Submit **one CSV for the entire test set**, with exact columns `filament_id,segmentation_rle` and **one row per predicted filament**. `filament_id` is the test image identifier plus a suffix unique within that image, for example `20150125172714Mh_1`; the suffix has no matching meaning. `segmentation_rle` contains **only compressed COCO RLE counts** for that filament's binary 2048 × 2048 mask, as produced by `pycocotools`; do not include the RLE size field or literal quote characters in the value. The official examples use the image identifier without the `.jpeg` extension. [Official “Submission File” section](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates).

The local test folder contains **180** `.jpeg` images and no local sample submission or competition scorer: [`data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images/`](../data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images/). The project [`mask_to_rle`](../src/filament_seg/inference.py#L126) returns space-delimited start/length pairs, **not** compressed COCO counts; it is incompatible with the official CSV as written. [`03_submission.ipynb`](../notebooks/03_submission.ipynb) imports this helper but does not yet construct a submission. No CSV was generated in this audit.

## 4. Annotation structure — confirmed local dataset finding

The training [MAGFiLO COCO JSON](../data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json) has top-level `info`, `licenses`, `categories`, `images`, and `annotations` fields. It contains **1,154 image records for 707 unique `images[].file_name` values** and **8,199 annotation records**; there are 707 physical training JPEGs. All image records report `height=width=2048`. `images[].id` identifies an annotation set and is distinct from the physical filename: one repeated file has IDs `050103-20111116063134Lh`, `050102-20111116063134Lh`, and `050101-20111116063134Lh`. Of the 707 filenames, 411 have one record, 145 have two, and 151 have three.

Each `annotations[]` entry references `image_id`, has `segmentation` as a list containing one polygon, and also has `id`, `category_id`, `bbox`, `area`, `iscrowd`, and `spine`. A polygon is a flat list of floating-point `(x,y)` pairs; the first polygon starts `[480.9382, 850.955, 480.307, 852.0535, ...]`. The four category IDs are `1=Left`, `2=Right`, `3=Unidentifiable`, and `4=Ambiguous`; annotation counts are 2,535, 2,590, 3,074, and 0 respectively. These categories are present in the data but are not the requested output classes. All 1,154 image IDs and 8,199 annotation IDs are unique, and every annotation `image_id` resolves to an image record. [JSON fields `categories`, `images`, `annotations`](../data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json).

## 5. Dataset pipeline compatibility — engineering assessment

[`SolarFilamentDataset`](../src/filament_seg/dataset.py#L279) correctly groups image records by filename and retains alternate sets in `images_by_filename`. Its `annotation_policy="first"` selects `images_by_filename[filename][0]["id"]`, then [`cv2.fillPoly`](../src/filament_seg/dataset.py#L329) unions that set's polygons into one binary mask. This is a reasonable **foreground-proposal training target**, but it discards filament identities and all alternate annotations in the returned mask. It cannot directly train or validate per-filament PQ. The returned `image_id` is a training annotation-set ID, not the test submission ID derived from a test filename. [Local dataset implementation](../src/filament_seg/dataset.py#L300).

[`configs/baseline.yaml`](../configs/baseline.yaml) sets one output class and BCE + Dice loss, which fits a binary foreground baseline. It also sets `model.in_channels: 3` while the dataset returns a one-channel image; this must be reconciled before model execution. [`inference.py`](../src/filament_seg/inference.py) has connected-component/watershed separation, but those methods and the submission encoder have not been validated against official PQ or format.

## 6. Data leakage and split — local findings and recommendation

[`02_training.ipynb`](../notebooks/02_training.ipynb) permutes the **707 unique filenames** with seed 42 and takes 565 train / 142 validation indices. Recomputing that split found **zero exact filename overlap**, so alternate annotation sets for one physical image cannot cross this split. The physical train and test folders also have **zero exact filename overlap**. These are confirmed local checks, not proof of no visual or temporal similarity.

Filename date prefixes show **13 validation images sharing a capture date with a training image** (13 shared dates); **33 test images share a capture date with a training image** (31 shared dates). This could make a random-image holdout optimistic for temporal generalization. For model selection, compare with a grouped holdout by capture date or wider time window, while keeping all annotation sets for a physical image together. Pixel-level duplicate/near-duplicate checks and observing-station grouping were not performed. [Notebook split code](../notebooks/02_training.ipynb), [JSON `images[].file_name` and `images[].date_captured`](../data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json).

## 7. Recommended model output and loss — engineering recommendation, not official mandate

A one-channel U-Net foreground logit map with BCE-with-logits + soft Dice is a valid **Stage 3.2 baseline component**, provided its input channel count matches the grayscale dataset. It is not the final competition output. Preserve each annotation set and filament instance for evaluation; add a non-training instance-separation stage, score separated masks with official-protocol PQ, and encode each final 2048 × 2048 instance with `pycocotools` compressed counts. Avoid using category, spine, bbox, or other ground-truth metadata as additional training targets until the organizer's restriction is clarified. [Official task and submission requirements](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates), [local config](../configs/baseline.yaml), [local dataset](../src/filament_seg/dataset.py#L318).

**Full-resolution inference is required for submission output** because the official mask dimensions are fixed at 2048 × 2048. Training on 512 × 512 crops is compatible if inference reconstructs native-resolution, per-instance masks without seams or coordinate changes. The choice of tiled versus full-image model execution is an engineering decision, not an explicit organizer mandate. [Official submission specification](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates), [local training transforms](../src/filament_seg/preprocessing.py).

## 8. Open questions and Stage 3.2 gate

1. Retrieve and inspect the organizer's [self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) for exact instance matching, threshold, multi-annotator aggregation, empty-image handling, and edge-case checks. The overview confirms PQ and its formula, but not these implementation details.
2. Resolve the official page's obsolete mean-Dice leaderboard sentence against the August 7 PQ update; confirm by comparing a known submission with the scorer. [Official overview](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates).
3. Clarify whether the organizer's ban on other ground-truth metadata rules out auxiliary supervision from `spine`, `bbox`, or `category_id`. [Official evaluation rules](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates).
4. Verify the exact behavior expected when a test image has no predicted filaments. The official page says one row per predicted filament but does not explicitly prescribe a zero-instance row. [Official submission specification](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates).

**Decision:** The official target, updated PQ metric/formula, and submission schema are verified, so a U-Net foreground baseline can proceed as **Stage 3.2**. Do not call a binary-mask output competition-ready or use local semantic Dice as its final selection metric. Exact scorer parity and a valid compressed-RLE submission remain separate gates before competitive evaluation.
