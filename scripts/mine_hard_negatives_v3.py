"""Mine conservative annotation-empty false detections from training images only."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filament_seg.benchmark import load_candidate, sha256_file
from filament_seg.dataset import SolarFilamentDataset
from filament_seg.inference import predict_tiled

EXPERIMENT = "unet_resnet34_v3_hard_negative_01"
V2 = ROOT / "checkpoints/unet_resnet34_v2/best_model.pth"
OUTPUT = ROOT / "reports" / f"{EXPERIMENT}_mining.json"


def train_split(dataset: SolarFilamentDataset) -> tuple[list[int], list[int]]:
    if len(dataset) != 707:
        raise ValueError("Expected 707 physical annotated images")
    indices = np.random.default_rng(42).permutation(len(dataset))
    train, validation = indices[:565].tolist(), indices[565:].tolist()
    if set(train) & set(validation) or len(train) != 565 or len(validation) != 142:
        raise AssertionError("Physical-image split is invalid")
    return train, validation


def all_annotation_union(dataset: SolarFilamentDataset, index: int) -> np.ndarray:
    """Rasterize every alternative training annotation set for exclusion only."""
    filename = dataset.filenames[index]
    union = np.zeros((2048, 2048), dtype=np.uint8)
    for image_record in dataset.images_by_filename[filename]:
        for annotation in dataset.annotations_by_image_id.get(image_record["id"], []):
            for polygon in annotation["segmentation"]:
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                if len(points) >= 3:
                    cv2.fillPoly(union, [np.rint(points).astype(np.int32)], 1)
    return union


def centered_crop(center_y: int, center_x: int, size: int = 512) -> tuple[int, int]:
    return (int(np.clip(center_y - size // 2, 0, 2048 - size)),
            int(np.clip(center_x - size // 2, 0, 2048 - size)))


def mine_image(probability: np.ndarray, annotation_union: np.ndarray) -> list[dict]:
    if probability.shape != (2048, 2048) or annotation_union.shape != probability.shape:
        raise ValueError("Expected full-resolution probability and annotation masks")
    binary = (probability >= .8).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = []
    # The exclusion halo avoids treating a slightly displaced true filament as background.
    halo = cv2.dilate(annotation_union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33)))
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        if area < 500:
            continue
        roi = labels[y:y + height, x:x + width] == component
        mean_confidence = float(probability[y:y + height, x:x + width][roi].mean())
        if mean_confidence < .95 or np.any(halo[y:y + height, x:x + width][roi]):
            continue
        center_x, center_y = [int(round(value)) for value in centers[component]]
        top, left = centered_crop(center_y, center_x)
        # Only annotation-empty 512x512 crops become hard-negative training patches.
        if np.any(annotation_union[top:top + 512, left:left + 512]):
            continue
        candidates.append({"top": top, "left": left, "component_area": area,
                           "mean_probability": mean_confidence,
                           "centroid_y": center_y, "centroid_x": center_x})
    candidates.sort(key=lambda item: (item["mean_probability"], item["component_area"]), reverse=True)
    return candidates[:3]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Mining artifact already exists: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local mining run")
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    annotation_path = base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    dataset = SolarFilamentDataset(base / "train_images", annotation_path)
    train, validation = train_split(dataset)
    val_names = {dataset.filenames[i] for i in validation}
    test_names = {p.name for p in (ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images").glob("*.jpeg")}
    if {dataset.filenames[i] for i in train} & (val_names | test_names):
        raise ValueError("Mining split overlaps validation or test filenames")
    model, metadata = load_candidate(V2, device="cuda")
    if metadata["checkpoint_sha256"] != "f9422022d65c0de79c61020b9ce5d81082c46b8919ddb5b1a6880624b02018c4":
        raise ValueError("Original v2 checkpoint hash changed")
    mined = {}
    torch.cuda.reset_peak_memory_stats()
    for position, index in enumerate(train, 1):
        filename = dataset.filenames[index]
        image = cv2.imread(str(dataset.image_dir / filename), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f"Invalid training image: {filename}")
        union = all_annotation_union(dataset, index)
        probability = predict_tiled(model, image.astype(np.float32) / 255.0,
                                    device="cuda", tile_size=512, overlap=128, batch_size=4)
        patches = mine_image(probability, union)
        if patches:
            mined[filename] = patches
        if position % 25 == 0 or position == len(train):
            print(f"Mined {position}/{len(train)} training images, "
                  f"candidate images={len(mined)}, patches={sum(map(len, mined.values()))}", flush=True)
    if not mined:
        raise RuntimeError("No conservative training-only hard negatives were found")
    payload = {
        "experiment_id": EXPERIMENT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(V2), "source_checkpoint_sha256": metadata["checkpoint_sha256"],
        "annotation_path": str(annotation_path), "annotation_sha256": sha256_file(annotation_path),
        "training_indices": train, "training_filenames": [dataset.filenames[i] for i in train],
        "validation_filenames_excluded": [dataset.filenames[i] for i in validation],
        "test_filename_overlap": 0, "validation_filename_overlap": 0,
        "strategy": "v2 512/128 tiled probabilities; threshold 0.8, component area >=500, mean confidence >=0.95; no overlap with 16-pixel dilated union of all training annotation sets; 512x512 crop contains zero annotated pixels; top three per image",
        "caveat": "Unannotated true filaments can still be mislabeled negative; all annotation sets and a halo reduce this risk but cannot eliminate it.",
        "normalization": "grayscale uint8 divided by 255 once",
        "training_images_examined": len(train), "images_with_candidates": len(mined),
        "patches": sum(map(len, mined.values())), "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "mining_source_sha256": sha256_file(Path(__file__)),
        "patches_by_filename": mined,
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"MINING_COMPLETE artifact={OUTPUT} patches={payload['patches']}", flush=True)


if __name__ == "__main__":
    main()
