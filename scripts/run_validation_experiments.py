"""Run full-resolution validation experiments, threshold tuning, and postprocessing comparison."""

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

from filament_seg.dataset import SolarFilamentDataset
from filament_seg.evaluate import (
    compute_dice,
    compute_iou,
    compute_precision_recall,
    diagnostic_panoptic_quality,
)
from filament_seg.inference import (
    iter_probability_instances,
    predict_tiled,
    separate_instances_watershed,
)
from filament_seg.model import build_model
from filament_seg.train import load_checkpoint

def main():
    root = Path('.').resolve()
    train_dir = root / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    image_dir = train_dir / "train_images"
    json_path = train_dir / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    ckpt_path = root / "checkpoints/unet_resnet34/best_model.pth"
    
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset and deterministic split
    dataset = SolarFilamentDataset(image_dir, json_path)
    indices = np.random.default_rng(42).permutation(len(dataset))
    val_indices = indices[565:].tolist()
    assert len(val_indices) == 142
    
    # Load model
    model = build_model(pretrained=False).to(device)
    ckpt = load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()
    print(f"Loaded {ckpt_path.name} (Epoch: {ckpt.get('epoch')})")
    
    print(f"\n--- Precomputing Full-Resolution Tiled Probability Maps for {len(val_indices)} validation images ---")
    t0 = time.time()
    val_data = []
    
    for i, idx in enumerate(val_indices):
        sample = dataset[idx]
        filename = sample["filename"]
        img_tensor = sample["image"]
        gt_mask = sample["mask"].numpy().squeeze().astype(np.uint8)
        
        # Ground truth instances for all available sets for this image
        num_sets = len(dataset.images_by_filename[filename])
        all_gt_sets = []
        for set_idx in range(num_sets):
            instances = [inst["mask"] for inst in dataset.iter_instances(idx, annotation_set_index=set_idx)]
            all_gt_sets.append(instances)
            
        prob_map = predict_tiled(
            model=model,
            image=img_tensor,
            tile_size=512,
            overlap=128,
            device=device,
            batch_size=4,
        )
        
        val_data.append({
            "idx": idx,
            "filename": filename,
            "gt_mask": gt_mask,
            "all_gt_sets": all_gt_sets,
            "prob_map": prob_map,
        })
        if (i + 1) % 20 == 0 or (i + 1) == len(val_indices):
            print(f"[{i+1}/{len(val_indices)}] Processed {filename} ({time.time() - t0:.1f}s)")
            
    print(f"Precomputation completed in {time.time() - t0:.1f}s")
    
    # Experiment 1: Semantic Metrics across Thresholds
    print("\n--- Running Semantic Metric Threshold Sweep ---")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
    semantic_results = []
    
    for th in thresholds:
        dices, ious, precs, recs = [], [], [], []
        tot_inter, tot_gt, tot_pred = 0, 0, 0
        for item in val_data:
            bin_pred = (item["prob_map"] >= th).astype(np.uint8)
            gt = item["gt_mask"]
            d = compute_dice(bin_pred, gt)
            iou = compute_iou(bin_pred, gt)
            p, r = compute_precision_recall(bin_pred, gt)
            dices.append(d)
            ious.append(iou)
            precs.append(p)
            recs.append(r)
            
            inter = np.logical_and(bin_pred > 0, gt > 0).sum()
            tot_inter += inter
            tot_pred += (bin_pred > 0).sum()
            tot_gt += (gt > 0).sum()
            
        micro_dice = (2.0 * tot_inter) / (tot_pred + tot_gt) if (tot_pred + tot_gt) > 0 else 1.0
        
        semantic_results.append({
            "threshold": th,
            "macro_dice": float(np.mean(dices)),
            "macro_iou": float(np.mean(ious)),
            "macro_precision": float(np.mean(precs)),
            "macro_recall": float(np.mean(recs)),
            "micro_dice": float(micro_dice),
        })
        print(f"Thresh {th:.2f} | Macro Dice: {np.mean(dices):.4f} | Macro IoU: {np.mean(ious):.4f} | Prec: {np.mean(precs):.4f} | Recall: {np.mean(recs):.4f} | Micro Dice: {micro_dice:.4f}")
        
    # Experiment 2: Postprocessing & Panoptic Quality Grid Search
    print("\n--- Running Postprocessing & Panoptic Quality Grid Search ---")
    min_areas = [50, 100, 200, 300, 500]
    methods = ["connected_components", "watershed"]
    pq_grid_results = []
    
    for th in [0.4, 0.5, 0.6, 0.65, 0.7, 0.75]:
        for area in min_areas:
            # Connected components
            pqs, sqs, rqs = [], [], []
            tot_tp, tot_fp, tot_fn = 0, 0, 0
            tot_matched_iou = 0.0
            
            for item in val_data:
                prob = item["prob_map"]
                gt_insts = item["all_gt_sets"][0] # Primary annotator
                preds = list(iter_probability_instances(prob, threshold=th, min_area=area))
                res = diagnostic_panoptic_quality(gt_insts, preds)
                pqs.append(res["pq"])
                sqs.append(res["sq"])
                rqs.append(res["rq"])
                tot_tp += res["tp"]
                tot_fp += res["fp"]
                tot_fn += res["fn"]
                tot_matched_iou += res["sq"] * res["tp"]
                
            denom = tot_tp + 0.5 * tot_fp + 0.5 * tot_fn
            global_pq = tot_matched_iou / denom if denom > 0 else 1.0
            
            pq_grid_results.append({
                "method": "connected_components",
                "threshold": th,
                "min_area": area,
                "macro_pq": float(np.mean(pqs)),
                "macro_sq": float(np.mean(sqs)),
                "macro_rq": float(np.mean(rqs)),
                "global_pq": float(global_pq),
                "total_tp": tot_tp,
                "total_fp": tot_fp,
                "total_fn": tot_fn,
            })
            print(f"CC | Thresh {th:.2f}, Area {area:3d} | Macro PQ: {np.mean(pqs):.4f} | Global PQ: {global_pq:.4f} | TP: {tot_tp}, FP: {tot_fp}, FN: {tot_fn}")
            
    # Also evaluate Watershed on top thresholds with min_area=200
    for th in [0.5, 0.6, 0.65, 0.7]:
        pqs, sqs, rqs = [], [], []
        tot_tp, tot_fp, tot_fn = 0, 0, 0
        tot_matched_iou = 0.0
        
        for item in val_data:
            prob = item["prob_map"]
            gt_insts = item["all_gt_sets"][0]
            labeled_ws, count = separate_instances_watershed(
                prob,
                core_threshold=min(th + 0.15, 0.95),
                mask_threshold=th,
                min_area=200,
            )
            preds = [(labeled_ws == l).astype(np.uint8) for l in range(1, count + 1)] if count > 0 else []
            res = diagnostic_panoptic_quality(gt_insts, preds)
            pqs.append(res["pq"])
            sqs.append(res["sq"])
            rqs.append(res["rq"])
            tot_tp += res["tp"]
            tot_fp += res["fp"]
            tot_fn += res["fn"]
            tot_matched_iou += res["sq"] * res["tp"]
            
        denom = tot_tp + 0.5 * tot_fp + 0.5 * tot_fn
        global_pq = tot_matched_iou / denom if denom > 0 else 1.0
        
        pq_grid_results.append({
            "method": "watershed",
            "threshold": th,
            "min_area": 200,
            "macro_pq": float(np.mean(pqs)),
            "macro_sq": float(np.mean(sqs)),
            "macro_rq": float(np.mean(rqs)),
            "global_pq": float(global_pq),
            "total_tp": tot_tp,
            "total_fp": tot_fp,
            "total_fn": tot_fn,
        })
        print(f"Watershed | Thresh {th:.2f}, Area 200 | Macro PQ: {np.mean(pqs):.4f} | Global PQ: {global_pq:.4f} | TP: {tot_tp}, FP: {tot_fp}, FN: {tot_fn}")

    # Experiment 3: Multi-annotator evaluation on best configuration
    # Find best config
    df_pq = pd.DataFrame(pq_grid_results)
    best_row = df_pq.loc[df_pq["global_pq"].idxmax()]
    best_th = float(best_row["threshold"])
    best_area = int(best_row["min_area"])
    best_method = str(best_row["method"])
    print(f"\n>>> Best Postprocessing Configuration: Method={best_method}, Thresh={best_th}, MinArea={best_area} <<<")
    print(f">>> Best Global PQ: {best_row['global_pq']:.4f}, Macro PQ: {best_row['macro_pq']:.4f} <<<")
    
    # Per-image detailed metrics for best configuration
    detailed_records = []
    multi_annotator_records = []
    
    for item in val_data:
        prob = item["prob_map"]
        gt = item["gt_mask"]
        fn = item["filename"]
        idx = item["idx"]
        
        bin_pred = (prob >= best_th).astype(np.uint8)
        d = compute_dice(bin_pred, gt)
        iou = compute_iou(bin_pred, gt)
        p, r = compute_precision_recall(bin_pred, gt)
        
        preds = list(iter_probability_instances(prob, threshold=best_th, min_area=best_area))
        res_primary = diagnostic_panoptic_quality(item["all_gt_sets"][0], preds)
        
        # Evaluate against all alternative sets
        set_pqs = []
        for s_idx, s_insts in enumerate(item["all_gt_sets"]):
            s_res = diagnostic_panoptic_quality(s_insts, preds)
            set_pqs.append(s_res["pq"])
            
        detailed_records.append({
            "idx": idx,
            "filename": fn,
            "num_annotators": len(item["all_gt_sets"]),
            "dice": float(d),
            "iou": float(iou),
            "precision": float(p),
            "recall": float(r),
            "gt_instances_set0": len(item["all_gt_sets"][0]),
            "pred_instances": len(preds),
            "tp": res_primary["tp"],
            "fp": res_primary["fp"],
            "fn": res_primary["fn"],
            "pq_primary": float(res_primary["pq"]),
            "sq_primary": float(res_primary["sq"]),
            "rq_primary": float(res_primary["rq"]),
            "pq_multi_mean": float(np.mean(set_pqs)),
            "pq_multi_std": float(np.std(set_pqs)),
        })
        
    df_detailed = pd.DataFrame(detailed_records)
    csv_path = reports_dir / "validation_detailed_metrics.csv"
    df_detailed.to_csv(csv_path, index=False)
    print(f"Saved detailed per-image validation results to {csv_path}")
    
    # Save complete JSON summary
    summary = {
        "best_configuration": {
            "method": best_method,
            "threshold": best_th,
            "min_area": best_area,
            "global_pq": float(best_row["global_pq"]),
            "macro_pq": float(best_row["macro_pq"]),
            "total_tp": int(best_row["total_tp"]),
            "total_fp": int(best_row["total_fp"]),
            "total_fn": int(best_row["total_fn"]),
        },
        "semantic_threshold_sweep": semantic_results,
        "pq_grid_results": pq_grid_results,
        "multi_annotator_summary": {
            "mean_pq_primary": float(df_detailed["pq_primary"].mean()),
            "mean_pq_multi_annotator_avg": float(df_detailed["pq_multi_mean"].mean()),
            "multi_annotator_images_count": int((df_detailed["num_annotators"] > 1).sum()),
        }
    }
    
    json_path = reports_dir / "validation_experiments_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved experiment summary to {json_path}")

if __name__ == "__main__":
    main()
