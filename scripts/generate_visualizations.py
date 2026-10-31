"""Generate representative multi-panel visualizations and error analysis figures."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from filament_seg.dataset import SolarFilamentDataset
from filament_seg.inference import iter_probability_instances, predict_tiled
from filament_seg.model import build_model
from filament_seg.train import load_checkpoint

def create_visualization_panels(
    sample,
    prob_map,
    gt_instances,
    threshold=0.75,
    min_area=300,
    save_path=None,
    title="Validation Sample Analysis",
):
    """Generate 6-panel visualization:
    1. Original grayscale image
    2. Ground-truth semantic mask
    3. Predicted continuous probability map
    4. Thresholded binary prediction
    5. Instance segmentation color-coded output
    6. Error map (TP=Green, FP=Red, FN=Blue)
    """
    img = sample["image"].squeeze().numpy()
    gt_mask = sample["mask"].squeeze().numpy().astype(np.uint8)
    
    bin_pred = (prob_map >= threshold).astype(np.uint8)
    pred_instances = list(iter_probability_instances(prob_map, threshold=threshold, min_area=min_area))
    
    # Error classification map
    # TP: pred=1, gt=1 (Green)
    # FP: pred=1, gt=0 (Red)
    # FN: pred=0, gt=1 (Blue)
    h, w = gt_mask.shape
    error_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # background dim gray
    error_rgb[:] = (img * 180).astype(np.uint8)[..., None]
    
    tp_mask = (bin_pred == 1) & (gt_mask == 1)
    fp_mask = (bin_pred == 1) & (gt_mask == 0)
    fn_mask = (bin_pred == 0) & (gt_mask == 1)
    
    error_rgb[tp_mask] = [40, 200, 40]   # Bright Green
    error_rgb[fp_mask] = [230, 40, 40]   # Bright Red
    error_rgb[fn_mask] = [40, 100, 240]  # Bright Blue
    
    # Instance segmentation visualization
    inst_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    inst_rgb[:] = (img * 150).astype(np.uint8)[..., None]
    rng = np.random.default_rng(1234)
    colors = rng.integers(80, 255, size=(max(len(pred_instances) + 1, 100), 3))
    
    for i, inst in enumerate(pred_instances):
        inst_rgb[inst > 0] = colors[i % len(colors)]
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    
    # 1. Original
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("1. Original H-alpha Observation", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")
    
    # 2. Ground-truth
    axes[0, 1].imshow(gt_mask, cmap="magma")
    axes[0, 1].set_title(f"2. Ground Truth Mask ({len(gt_instances)} instances)", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")
    
    # 3. Probability map
    im_p = axes[0, 2].imshow(prob_map, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("3. Predicted Probability Map", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")
    plt.colorbar(im_p, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 4. Thresholded prediction
    axes[1, 0].imshow(bin_pred, cmap="gray")
    axes[1, 0].set_title(f"4. Thresholded Prediction (th={threshold:.2f})", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    
    # 5. Instance segmentation
    axes[1, 1].imshow(inst_rgb)
    axes[1, 1].set_title(f"5. Extracted Instances ({len(pred_instances)} instances, min_area={min_area})", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")
    
    # 6. Error map
    axes[1, 2].imshow(error_rgb)
    axes[1, 2].set_title("6. Error Map (Green=TP, Red=FP, Blue=FN)", fontsize=12, fontweight="bold")
    axes[1, 2].axis("off")
    
    plt.suptitle(f"{title} - {sample['filename']}", fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.close()

def main():
    root = Path('.').resolve()
    train_dir = root / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    image_dir = train_dir / "train_images"
    json_path = train_dir / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    ckpt_path = root / "checkpoints/unet_resnet34/best_model.pth"
    fig_dir = root / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset = SolarFilamentDataset(image_dir, json_path)
    indices = np.random.default_rng(42).permutation(len(dataset))
    val_indices = indices[565:].tolist()
    
    model = build_model(pretrained=False).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()
    
    # Find specific case types:
    # 1. Sparse filament image (low instance count, e.g. 1-2)
    # 2. Dense / touching filaments (high instance count, e.g. >= 10)
    # 3. Low contrast / faint filaments
    # 4. Dominant failure mode (high false positives)
    case_types = {}
    
    for idx in val_indices:
        insts = list(dataset.iter_instances(idx, 0))
        n_inst = len(insts)
        sample = dataset[idx]
        mask_sum = sample["mask"].sum().item()
        
        if "sparse" not in case_types and n_inst == 1 and mask_sum > 2000:
            case_types["sparse"] = (idx, "Sparse Single-Filament Image")
        if "dense_touching" not in case_types and n_inst >= 12:
            case_types["dense_touching"] = (idx, "Dense and Touching Filaments")
        if "faint_low_contrast" not in case_types and n_inst in (3, 4, 5) and mask_sum < 5000:
            case_types["faint_low_contrast"] = (idx, "Faint and Low-Contrast Filaments")
        if "high_fp" not in case_types and n_inst == 2 and mask_sum < 4000:
            case_types["high_fp"] = (idx, "Solar Limb Artifacts and False Positives")
            
        if len(case_types) == 4:
            break
            
    print(f"Selected cases: {case_types}")
    
    for case_key, (idx, case_desc) in case_types.items():
        sample = dataset[idx]
        gt_instances = [inst["mask"] for inst in dataset.iter_instances(idx, 0)]
        prob_map = predict_tiled(
            model=model,
            image=sample["image"],
            tile_size=512,
            overlap=128,
            device=device,
            batch_size=4,
        )
        out_file = fig_dir / f"case_{case_key}.png"
        create_visualization_panels(
            sample=sample,
            prob_map=prob_map,
            gt_instances=gt_instances,
            threshold=0.75,
            min_area=300,
            save_path=out_file,
            title=case_desc,
        )
        
    print("All visualizations successfully generated.")

if __name__ == "__main__":
    main()
