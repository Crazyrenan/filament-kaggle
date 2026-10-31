"""Quantitative evaluation metrics for semantic and instance segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from filament_seg.utils import get_logger

logger = get_logger(__name__)


def compute_iou(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Compute Intersection over Union (Jaccard Index) for binary masks.

    Args:
        pred_mask: Boolean or binary array representing predicted mask.
        target_mask: Boolean or binary array representing ground-truth mask.
        eps: Small epsilon to prevent division by zero when both masks are empty.

    Returns:
        IoU score as float in [0.0, 1.0].
    """
    pred_bool = pred_mask.astype(bool)
    target_bool = target_mask.astype(bool)

    intersection = np.logical_and(pred_bool, target_bool).sum()
    union = np.logical_or(pred_bool, target_bool).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    if intersection == 0:
        return 0.0

    return float(intersection / union)


def compute_dice(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Compute Dice Coefficient (F1-score) for binary masks.

    Args:
        pred_mask: Boolean or binary array representing predicted mask.
        target_mask: Boolean or binary array representing ground-truth mask.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Dice score as float in [0.0, 1.0].
    """
    pred_bool = pred_mask.astype(bool)
    target_bool = target_mask.astype(bool)

    intersection = np.logical_and(pred_bool, target_bool).sum()
    total_area = pred_bool.sum() + target_bool.sum()

    if total_area == 0:
        return 1.0 if intersection == 0 else 0.0

    if intersection == 0:
        return 0.0

    return float((2.0 * intersection) / total_area)


def compute_precision_recall(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    eps: float = 1e-7,
) -> tuple[float, float]:
    """Compute pixel-level Precision and Recall.

    Args:
        pred_mask: Predicted binary mask.
        target_mask: Ground-truth binary mask.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Tuple of (precision, recall).
    """
    pred_bool = pred_mask.astype(bool)
    target_bool = target_mask.astype(bool)

    tp = np.logical_and(pred_bool, target_bool).sum()
    fp = np.logical_and(pred_bool, ~target_bool).sum()
    fn = np.logical_and(~pred_bool, target_bool).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return precision, recall


def evaluate_semantic_predictions(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute diagnostic foreground metrics, not competition Panoptic Quality.

    Probabilities are foreground at ``>= threshold``. Dice and IoU are 1.0 when
    both binary masks are empty; precision and recall are 0.0 in that case.
    Reports both macro-averaged (mean per image) and micro-averaged (global) metrics.

    Args:
        predictions: List of probability arrays (values in [0.0, 1.0]).
        targets: List of binary ground-truth arrays.
        threshold: Decision threshold for binarizing probabilities.

    Returns:
        Dictionary containing mean and global IoU, Dice, Precision, and Recall.
    """
    ious: list[float] = []
    dices: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    total_intersection = 0
    total_pred_pixels = 0
    total_target_pixels = 0
    total_union = 0

    for pred, target in zip(predictions, targets):
        bin_pred = (pred >= threshold).astype(np.uint8)
        ious.append(compute_iou(bin_pred, target))
        dices.append(compute_dice(bin_pred, target))
        p, r = compute_precision_recall(bin_pred, target)
        precisions.append(p)
        recalls.append(r)

        pred_bool = bin_pred.astype(bool)
        target_bool = target.astype(bool)
        inter = np.logical_and(pred_bool, target_bool).sum()
        total_intersection += int(inter)
        total_pred_pixels += int(pred_bool.sum())
        total_target_pixels += int(target_bool.sum())
        total_union += int(np.logical_or(pred_bool, target_bool).sum())

    micro_dice = (
        float(2.0 * total_intersection / (total_pred_pixels + total_target_pixels))
        if (total_pred_pixels + total_target_pixels) > 0
        else 1.0
    )
    micro_iou = (
        float(total_intersection / total_union)
        if total_union > 0
        else 1.0
    )

    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "micro_dice": micro_dice,
        "micro_iou": micro_iou,
    }


def diagnostic_panoptic_quality(
    ground_truth: list[np.ndarray],
    predictions: list[np.ndarray],
    iou_threshold: float = 0.5,
    inclusive: bool = False,
    return_details: bool = False,
) -> dict[str, Any]:
    """Match individual binary masks and compute diagnostic PQ, SQ, and RQ.

    Matching uses linear sum assignment maximizing valid IoU matches, with strict
    IoU > 0.5 (or >= 0.5 if inclusive=True). Empty prediction masks are filtered
    gracefully without crashing.

    Args:
        ground_truth: List of 2D binary ground-truth instance masks.
        predictions: List of 2D binary predicted instance masks.
        iou_threshold: Minimum IoU required for an eligible match. Default 0.5.
        inclusive: If True, uses IoU >= threshold. Default is False (> threshold).
        return_details: Include matched GT/prediction indices, IoUs, and unmatched indices.

    Returns:
        Dictionary with pq, sq, rq, tp, fp, fn, and optional matching details.
    """
    from scipy.optimize import linear_sum_assignment

    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")

    # Validate and filter ground truth instances
    valid_gt: list[np.ndarray] = []
    gt_indices: list[int] = []
    shape = None
    for original_index, mask in enumerate(ground_truth):
        arr = np.asarray(mask)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError("ground truth instances must be nonempty 2D arrays")
        if shape is None:
            shape = arr.shape
        if arr.shape != shape:
            raise ValueError(f"instances must share shape {shape}, got {arr.shape}")
        if arr.any():
            valid_gt.append(arr.astype(bool, copy=False))
            gt_indices.append(original_index)

    # Validate and filter prediction instances (gracefully ignore all-zero prediction masks)
    valid_preds: list[np.ndarray] = []
    pred_indices: list[int] = []
    for original_index, mask in enumerate(predictions):
        arr = np.asarray(mask)
        if arr.ndim != 2 or arr.size == 0:
            continue
        if shape is not None and arr.shape != shape:
            raise ValueError(f"predictions must share shape {shape}, got {arr.shape}")
        if shape is None:
            shape = arr.shape
        if arr.any():
            valid_preds.append(arr.astype(bool, copy=False))
            pred_indices.append(original_index)

    n_gt = len(valid_gt)
    n_pred = len(valid_preds)

    # Both empty: perfect diagnosis
    if n_gt == 0 and n_pred == 0:
        result = {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0}
        return {**result, "matches": [], "unmatched_gt": [], "unmatched_pred": []} if return_details else result

    # Ground truth empty, but predictions made: all false positives
    if n_gt == 0 and n_pred > 0:
        result = {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": n_pred, "fn": 0}
        return {**result, "matches": [], "unmatched_gt": [], "unmatched_pred": pred_indices} if return_details else result

    # Predictions empty, but ground truth exists: all false negatives
    if n_gt > 0 and n_pred == 0:
        result = {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": n_gt}
        return {**result, "matches": [], "unmatched_gt": gt_indices, "unmatched_pred": []} if return_details else result

    # Compute pairwise IoU matrix
    ious = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i, truth in enumerate(valid_gt):
        for j, pred in enumerate(valid_preds):
            overlap = np.count_nonzero(truth & pred)
            if overlap:
                union = np.count_nonzero(truth | pred)
                ious[i, j] = overlap / union

    eligible = ious >= iou_threshold if inclusive else ious > iou_threshold
    scores = np.zeros((n_gt + n_pred, n_gt + n_pred), dtype=np.float64)
    scores[:n_gt, :n_pred] = np.where(eligible, 1.0 + ious, -1.0)
    rows, columns = linear_sum_assignment(scores, maximize=True)

    matched_pairs = [
        (row, column) for row, column in zip(rows, columns)
        if row < n_gt and column < n_pred and eligible[row, column]
    ]
    matched_iou = sum(ious[row, column] for row, column in matched_pairs)
    tp = len(matched_pairs)
    fp = n_pred - tp
    fn = n_gt - tp
    denominator = tp + 0.5 * fp + 0.5 * fn

    result = {
        "pq": float(matched_iou / denominator) if denominator > 0 else 1.0,
        "sq": float(matched_iou / tp) if tp > 0 else 0.0,
        "rq": float(tp / denominator) if denominator > 0 else 1.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }
    if not return_details:
        return result
    matched_gt = {row for row, _ in matched_pairs}
    matched_pred = {column for _, column in matched_pairs}
    return {
        **result,
        "matches": [
            {"gt_index": gt_indices[row], "pred_index": pred_indices[column], "iou": float(ious[row, column])}
            for row, column in matched_pairs
        ],
        "unmatched_gt": [gt_indices[row] for row in range(n_gt) if row not in matched_gt],
        "unmatched_pred": [pred_indices[column] for column in range(n_pred) if column not in matched_pred],
    }


def evaluate_full_resolution_dataset(
    model: Any,
    dataset: Any,
    sample_indices: list[int],
    device: Any = "cuda",
    threshold: float = 0.6,
    min_area: int = 200,
    use_tiled: bool = True,
    tile_size: int = 512,
    overlap: int = 128,
    annotation_set_index: int = 0,
    output_csv_path: str | Path | None = None,
    output_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate full-resolution 2048x2048 validation set without data leakage.

    Computes semantic Dice and IoU as well as instance-level Panoptic Quality.
    Handles empty ground-truth and empty-prediction cases explicitly.
    Produces per-image metrics and global aggregates.

    Args:
        model: Trained PyTorch segmentation model.
        dataset: SolarFilamentDataset instance.
        sample_indices: Indices of validation images.
        device: Execution device.
        threshold: Probability binarization threshold.
        min_area: Minimum component pixel area.
        use_tiled: Whether to use tiled inference.
        tile_size: Tile size for inference.
        overlap: Overlap for tiled inference.
        annotation_set_index: Which annotation set to evaluate against (default 0).
        output_csv_path: Optional path to save per-image CSV.
        output_json_path: Optional path to save aggregate and detailed JSON.

    Returns:
        Dictionary containing summary metrics and per-image records.
    """
    import json
    import pandas as pd
    import torch
    from filament_seg.inference import iter_probability_instances, predict_tiled

    dev = torch.device(device)
    model.eval()

    records: list[dict[str, Any]] = []
    total_inter = 0
    total_union = 0
    total_gt_pixels = 0
    total_pred_pixels = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_matched_iou = 0.0

    for i, idx in enumerate(sample_indices):
        sample = dataset[idx]
        filename = sample["filename"]
        image_tensor = sample["image"]
        gt_mask = sample["mask"].numpy().squeeze().astype(np.uint8)

        # Full-resolution ground truth instances
        gt_instances = [inst["mask"] for inst in dataset.iter_instances(idx, annotation_set_index=annotation_set_index)]

        # If dataset item was cropped by a transform (e.g. 512x512), recover full resolution
        if hasattr(dataset, "image_dir") and hasattr(dataset, "filenames"):
            expected_shape = (2048, 2048)
            if hasattr(dataset, "images_by_filename"):
                img_sets = dataset.images_by_filename.get(filename, [])
                if img_sets:
                    expected_shape = (img_sets[0].get("height", 2048), img_sets[0].get("width", 2048))
            if image_tensor.shape[-2:] != expected_shape:
                import cv2
                import torch
                raw_img = cv2.imread(str(dataset.image_dir / filename), cv2.IMREAD_GRAYSCALE)
                if raw_img is not None:
                    image_tensor = torch.as_tensor(raw_img.astype(np.float32) / 255.0, dtype=torch.float32).unsqueeze(0)
                raw_mask = np.zeros(expected_shape, dtype=np.uint8)
                for inst_m in gt_instances:
                    raw_mask = np.maximum(raw_mask, inst_m)
                gt_mask = raw_mask

        # Native resolution prediction
        if use_tiled:
            prob_map = predict_tiled(
                model=model,
                image=image_tensor,
                tile_size=tile_size,
                overlap=overlap,
                device=dev,
            )
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    logits = model(image_tensor.unsqueeze(0).to(dev))
                    probs = torch.sigmoid(logits)
                    prob_map = probs.squeeze().cpu().numpy().astype(np.float32)

        # Semantic metrics
        pred_bin = (prob_map >= threshold).astype(np.uint8)
        dice = compute_dice(pred_bin, gt_mask)
        iou = compute_iou(pred_bin, gt_mask)
        p, r = compute_precision_recall(pred_bin, gt_mask)

        # Instance segmentation & PQ
        pred_instances = list(iter_probability_instances(prob_map, threshold=threshold, min_area=min_area))
        pq_res = diagnostic_panoptic_quality(gt_instances, pred_instances)

        # Pixel accumulations
        pred_bool = pred_bin.astype(bool)
        gt_bool = gt_mask.astype(bool)
        inter = np.logical_and(pred_bool, gt_bool).sum()
        union = np.logical_or(pred_bool, gt_bool).sum()

        total_inter += int(inter)
        total_union += int(union)
        total_gt_pixels += int(gt_bool.sum())
        total_pred_pixels += int(pred_bool.sum())

        total_tp += int(pq_res["tp"])
        total_fp += int(pq_res["fp"])
        total_fn += int(pq_res["fn"])
        total_matched_iou += float(pq_res["sq"] * pq_res["tp"])

        records.append({
            "index": idx,
            "filename": filename,
            "gt_instances": len(gt_instances),
            "pred_instances": len(pred_instances),
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(p),
            "recall": float(r),
            "tp": int(pq_res["tp"]),
            "fp": int(pq_res["fp"]),
            "fn": int(pq_res["fn"]),
            "pq": float(pq_res["pq"]),
            "sq": float(pq_res["sq"]),
            "rq": float(pq_res["rq"]),
            "gt_pixels": int(gt_bool.sum()),
            "pred_pixels": int(pred_bool.sum()),
        })

    # Compute aggregate metrics
    df = pd.DataFrame(records)
    macro_dice = float(df["dice"].mean())
    macro_iou = float(df["iou"].mean())
    macro_precision = float(df["precision"].mean())
    macro_recall = float(df["recall"].mean())
    macro_pq = float(df["pq"].mean())
    macro_sq = float(df["sq"].mean())
    macro_rq = float(df["rq"].mean())

    micro_dice = float(2.0 * total_inter / (total_gt_pixels + total_pred_pixels)) if (total_gt_pixels + total_pred_pixels) > 0 else 1.0
    micro_iou = float(total_inter / total_union) if total_union > 0 else 1.0

    global_denominator = total_tp + 0.5 * total_fp + 0.5 * total_fn
    global_pq = float(total_matched_iou / global_denominator) if global_denominator > 0 else 1.0
    global_sq = float(total_matched_iou / total_tp) if total_tp > 0 else 0.0
    global_rq = float(total_tp / global_denominator) if global_denominator > 0 else 1.0

    summary = {
        "num_images": len(sample_indices),
        "threshold": threshold,
        "min_area": min_area,
        "use_tiled": use_tiled,
        "semantic_metrics": {
            "macro_dice": macro_dice,
            "macro_iou": macro_iou,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "micro_dice": micro_dice,
            "micro_iou": micro_iou,
        },
        "instance_metrics": {
            "macro_pq": macro_pq,
            "macro_sq": macro_sq,
            "macro_rq": macro_rq,
            "global_pq": global_pq,
            "global_sq": global_sq,
            "global_rq": global_rq,
            "total_gt_instances": int(df["gt_instances"].sum()),
            "total_pred_instances": int(df["pred_instances"].sum()),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
        },
    }

    if output_csv_path:
        csv_p = Path(output_csv_path)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_p, index=False)
        logger.info("Saved validation per-image metrics to %s", csv_p)

    if output_json_path:
        json_p = Path(output_json_path)
        json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved validation aggregate metrics to %s", json_p)

    return {"summary": summary, "per_image": records}


def compare_models(
    model_a: Any,
    model_b: Any,
    dataset: Any,
    sample_indices: list[int],
    device: Any = "cuda",
    threshold_a: float = 0.75,
    min_area_a: int = 300,
    threshold_b: float = 0.75,
    min_area_b: int = 300,
    model_a_name: str = "Baseline U-Net (v1)",
    model_b_name: str = "U-Net ResNet34 v2",
    tile_size: int = 512,
    overlap: int = 128,
) -> dict[str, Any]:
    """Compare two models on the exact same validation subset and coordinates.

    Supports fixed postprocessing comparison (isolating model architectural/loss improvements)
    as well as model-specific tuned postprocessing comparisons.

    Args:
        model_a: First model (e.g. baseline v1).
        model_b: Second model (e.g. EDA-informed v2).
        dataset: SolarFilamentDataset instance.
        sample_indices: Validation image indices to evaluate.
        device: Target execution device.
        threshold_a: Probability threshold for Model A.
        min_area_a: Min area for Model A.
        threshold_b: Probability threshold for Model B.
        min_area_b: Min area for Model B.
        model_a_name: Display name for Model A.
        model_b_name: Display name for Model B.
        tile_size: Tiled inference tile size.
        overlap: Tile overlap.

    Returns:
        Dictionary containing comparison summary table and per-model results.
    """
    import time
    import torch

    dev = torch.device(device)
    logger.info("Comparing %s vs %s on %d validation images...", model_a_name, model_b_name, len(sample_indices))

    # Evaluate Model A
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0_a = time.time()
    res_a = evaluate_full_resolution_dataset(
        model=model_a,
        dataset=dataset,
        sample_indices=sample_indices,
        device=dev,
        threshold=threshold_a,
        min_area=min_area_a,
        use_tiled=True,
        tile_size=tile_size,
        overlap=overlap,
    )
    time_a = time.time() - t0_a
    mem_a = torch.cuda.max_memory_allocated() / (1024**2) if dev.type == "cuda" else 0.0

    # Evaluate Model B
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0_b = time.time()
    res_b = evaluate_full_resolution_dataset(
        model=model_b,
        dataset=dataset,
        sample_indices=sample_indices,
        device=dev,
        threshold=threshold_b,
        min_area=min_area_b,
        use_tiled=True,
        tile_size=tile_size,
        overlap=overlap,
    )
    time_b = time.time() - t0_b
    mem_b = torch.cuda.max_memory_allocated() / (1024**2) if dev.type == "cuda" else 0.0

    sum_a = res_a["summary"]
    sum_b = res_b["summary"]

    comparison_data = [
        {
            "Metric": "Macro Dice",
            model_a_name: sum_a["semantic_metrics"]["macro_dice"],
            model_b_name: sum_b["semantic_metrics"]["macro_dice"],
            "Delta (v2 - v1)": sum_b["semantic_metrics"]["macro_dice"] - sum_a["semantic_metrics"]["macro_dice"],
        },
        {
            "Metric": "Micro (Global) Dice",
            model_a_name: sum_a["semantic_metrics"]["micro_dice"],
            model_b_name: sum_b["semantic_metrics"]["micro_dice"],
            "Delta (v2 - v1)": sum_b["semantic_metrics"]["micro_dice"] - sum_a["semantic_metrics"]["micro_dice"],
        },
        {
            "Metric": "Macro IoU",
            model_a_name: sum_a["semantic_metrics"]["macro_iou"],
            model_b_name: sum_b["semantic_metrics"]["macro_iou"],
            "Delta (v2 - v1)": sum_b["semantic_metrics"]["macro_iou"] - sum_a["semantic_metrics"]["macro_iou"],
        },
        {
            "Metric": "Pixel Precision",
            model_a_name: sum_a["semantic_metrics"]["macro_precision"],
            model_b_name: sum_b["semantic_metrics"]["macro_precision"],
            "Delta (v2 - v1)": sum_b["semantic_metrics"]["macro_precision"] - sum_a["semantic_metrics"]["macro_precision"],
        },
        {
            "Metric": "Pixel Recall",
            model_a_name: sum_a["semantic_metrics"]["macro_recall"],
            model_b_name: sum_b["semantic_metrics"]["macro_recall"],
            "Delta (v2 - v1)": sum_b["semantic_metrics"]["macro_recall"] - sum_a["semantic_metrics"]["macro_recall"],
        },
        {
            "Metric": "Global Panoptic Quality (PQ)",
            model_a_name: sum_a["instance_metrics"]["global_pq"],
            model_b_name: sum_b["instance_metrics"]["global_pq"],
            "Delta (v2 - v1)": sum_b["instance_metrics"]["global_pq"] - sum_a["instance_metrics"]["global_pq"],
        },
        {
            "Metric": "Macro Panoptic Quality (PQ)",
            model_a_name: sum_a["instance_metrics"]["macro_pq"],
            model_b_name: sum_b["instance_metrics"]["macro_pq"],
            "Delta (v2 - v1)": sum_b["instance_metrics"]["macro_pq"] - sum_a["instance_metrics"]["macro_pq"],
        },
        {
            "Metric": "True Positives (TP)",
            model_a_name: sum_a["instance_metrics"]["total_tp"],
            model_b_name: sum_b["instance_metrics"]["total_tp"],
            "Delta (v2 - v1)": sum_b["instance_metrics"]["total_tp"] - sum_a["instance_metrics"]["total_tp"],
        },
        {
            "Metric": "False Positives (FP)",
            model_a_name: sum_a["instance_metrics"]["total_fp"],
            model_b_name: sum_b["instance_metrics"]["total_fp"],
            "Delta (v2 - v1)": sum_b["instance_metrics"]["total_fp"] - sum_a["instance_metrics"]["total_fp"],
        },
        {
            "Metric": "False Negatives (FN)",
            model_a_name: sum_a["instance_metrics"]["total_fn"],
            model_b_name: sum_b["instance_metrics"]["total_fn"],
            "Delta (v2 - v1)": sum_b["instance_metrics"]["total_fn"] - sum_a["instance_metrics"]["total_fn"],
        },
        {
            "Metric": "Inference Time (s/image)",
            model_a_name: time_a / len(sample_indices),
            model_b_name: time_b / len(sample_indices),
            "Delta (v2 - v1)": (time_b - time_a) / len(sample_indices),
        },
        {
            "Metric": "Peak VRAM (MiB)",
            model_a_name: mem_a,
            model_b_name: mem_b,
            "Delta (v2 - v1)": mem_b - mem_a,
        },
    ]

    import pandas as pd
    df_comp = pd.DataFrame(comparison_data)

    return {
        "comparison_table": df_comp,
        "results_a": res_a,
        "results_b": res_b,
    }
