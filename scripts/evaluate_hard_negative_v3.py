"""Cache v3 predictions, tune on 71 images, freeze, then score internal once."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from filament_seg.benchmark import cache_probability_maps, load_candidate, sha256_file
from filament_seg.component_filter import ComponentFilter
from filament_seg.dataset import SolarFilamentDataset
from run_component_filter_experiment import score
from generate_component_filter_submission import bootstrap_interval

EXPERIMENT = "unet_resnet34_v3_hard_negative_01"
V3 = ROOT / "checkpoints" / EXPERIMENT / "best_model.pth"
V2 = ROOT / "checkpoints/unet_resnet34_v2/best_model.pth"
PARTITION = ROOT / "reports/validation_partition_v1.json"
HISTORICAL = ROOT / "reports/historical_checkpoint_hashes.json"
V2_REPORT = ROOT / "reports/model_benchmark_20261104T094701Z.json"
FIXED = ComponentFilter("fixed_v2_protocol", min_mean_probability=.97)
GRID = [ComponentFilter(f"confidence_{str(value).replace('.', 'p')}", min_mean_probability=value)
        for value in (.95, .96, .97, .975)]


def context():
    partition = json.loads(PARTITION.read_text(encoding="utf-8"))
    base = ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train"
    dataset = SolarFilamentDataset(base / "train_images",
                                   base / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    report = json.loads(V2_REPORT.read_text(encoding="utf-8"))
    old = next(row for row in report["results"] if row["model"]["experiment_id"] == "unet_resnet34_v2")
    v2_cache = Path(old["cache"])
    v2_manifest = json.loads((v2_cache / "cache_manifest.json").read_text(encoding="utf-8"))
    expected = next(row["sha256"] for row in json.loads(HISTORICAL.read_text(encoding="utf-8"))
                    if Path(row["path"]).resolve() == V2.resolve())
    if sha256_file(V2) != expected or v2_manifest["checkpoint_sha256"] != expected:
        raise ValueError("Historical v2 checkpoint or cache hash changed")
    indices = partition["tuning_indices"] + partition["internal_check_indices"]
    if set(indices) != set(v2_manifest["indices"]) or len(indices) != 142:
        raise ValueError("Validation cache or partition mismatch")
    if not all((v2_cache / f"{index:04d}.npy").is_file() for index in indices):
        raise FileNotFoundError("Original v2 probability cache incomplete")
    return dataset, partition, v2_cache, v2_manifest


def v3_cache(dataset, partition):
    state = json.loads((V3.parent / "metrics.json").read_text(encoding="utf-8"))
    if state["status"] != "training_complete":
        raise ValueError("V3 training has not completed")
    model, metadata = load_candidate(V3, device="cuda")
    indices = partition["tuning_indices"] + partition["internal_check_indices"]
    folder, manifest = cache_probability_maps(model, metadata, dataset, indices,
        ROOT / "outputs/probability_cache", device="cuda", tile_size=512, overlap=128, batch_size=4)
    del model
    return folder, manifest


def timing(folder: Path, indices: list[int]) -> dict:
    rows = [json.loads((folder / f"{index:04d}.timing.json").read_text(encoding="utf-8")) for index in indices]
    return {"inference_seconds_per_image": float(np.mean([row["seconds"] for row in rows])),
            "peak_inference_vram_mib": float(max(row["peak_gpu_memory_mib"] for row in rows))}


def tune() -> Path:
    dataset, partition, v2_folder, v2_manifest = context()
    v3_folder, v3_manifest = v3_cache(dataset, partition)
    tuning = partition["tuning_indices"]
    baseline = score(tuning, FIXED, v2_folder, dataset)
    fixed = score(tuning, FIXED, v3_folder, dataset)
    candidates = [score(tuning, config, v3_folder, dataset) for config in GRID]
    selected = max(candidates, key=lambda row: (row["summary"]["global_pq"], row["summary"]["sq"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "reports" / f"{EXPERIMENT}_frozen_{stamp}.json"
    artifact = {"phase": "frozen_after_tuning", "experiment_id": EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "provisional",
        "v2_checkpoint_sha256": sha256_file(V2), "v3_checkpoint_sha256": sha256_file(V3),
        "partition_sha256": sha256_file(PARTITION),
        "v2_cache_manifest_sha256": sha256_file(v2_folder / "cache_manifest.json"),
        "v3_cache_manifest_sha256": sha256_file(v3_folder / "cache_manifest.json"),
        "evaluation_source_sha256": sha256_file(Path(__file__)),
        "metric_source_sha256": sha256_file(ROOT / "src/filament_seg/official_metric.py"),
        "filter_source_sha256": sha256_file(ROOT / "src/filament_seg/component_filter.py"),
        "v2_cache": str(v2_folder), "v3_cache": str(v3_folder),
        "tuning_indices": tuning, "internal_indices": partition["internal_check_indices"],
        "fixed_postprocessing": vars(FIXED), "v3_selected_postprocessing": selected["config"],
        "selection_rule": "highest tuning Global PQ, SQ tie break; four predeclared mean confidence cutoffs",
        "v2_fixed_tuning": baseline, "v3_fixed_tuning": fixed, "v3_tuning_candidates": candidates,
        "v2_inference": timing(v2_folder, tuning), "v3_inference": timing(v3_folder, tuning),
        "limitation": "Original v2 was historically evaluated on internal images; hidden evaluator parity unverified."}
    with destination.open("x", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2)
    print("FROZEN", destination, "v2_tuning", baseline["summary"]["global_pq"],
          "v3_fixed", fixed["summary"]["global_pq"],
          "v3_selected", selected["summary"]["global_pq"], selected["config"], flush=True)
    return destination


def internal(frozen_path: Path) -> Path:
    if list(ROOT.glob(f"reports/{EXPERIMENT}_internal_*.json")):
        raise FileExistsError("Internal-check result already exists; refuse a second evaluation")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    dataset, partition, v2_folder, _ = context()
    if frozen["phase"] != "frozen_after_tuning" or frozen["evaluation_source_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("Frozen evaluation source changed")
    checks = [(V2, "v2_checkpoint_sha256"), (V3, "v3_checkpoint_sha256"),
              (PARTITION, "partition_sha256"),
              (v2_folder / "cache_manifest.json", "v2_cache_manifest_sha256"),
              (Path(frozen["v3_cache"]) / "cache_manifest.json", "v3_cache_manifest_sha256"),
              (ROOT / "src/filament_seg/official_metric.py", "metric_source_sha256"),
              (ROOT / "src/filament_seg/component_filter.py", "filter_source_sha256")]
    if any(sha256_file(path) != frozen[key] for path, key in checks):
        raise ValueError("Checkpoint, partition, cache, scorer, or filter changed after freeze")
    if frozen["internal_indices"] != partition["internal_check_indices"]:
        raise ValueError("Internal indices changed")
    indices = frozen["internal_indices"]
    selected_config = ComponentFilter(**frozen["v3_selected_postprocessing"])
    baseline = score(indices, FIXED, v2_folder, dataset)
    fixed = score(indices, FIXED, Path(frozen["v3_cache"]), dataset)
    selected = fixed if selected_config.min_mean_probability == FIXED.min_mean_probability else score(
        indices, selected_config, Path(frozen["v3_cache"]), dataset)
    interval = bootstrap_interval(baseline["per_image"], selected["per_image"])
    delta = selected["summary"]["global_pq"] - baseline["summary"]["global_pq"]
    # Require both a useful point improvement and positive paired uncertainty bound.
    supported = delta > .005 and interval[0] > 0 and selected["summary"]["tp"] >= .9 * baseline["summary"]["tp"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "reports" / f"{EXPERIMENT}_internal_{stamp}.json"
    artifact = {"phase": "internal_scored_once", "status": "provisional", "experiment_id": EXPERIMENT,
        "frozen_path": str(frozen_path.resolve()), "frozen_sha256": sha256_file(frozen_path),
        "v2_fixed": baseline, "v3_fixed": fixed, "v3_tuned": selected,
        "delta_global_pq": delta, "paired_capture_date_bootstrap_95pct_delta_interval": interval,
        "bootstrap_seed": 42, "bootstrap_replicates": 5000,
        "v2_inference": timing(v2_folder, indices), "v3_inference": timing(Path(frozen["v3_cache"]), indices),
        "decision": "submission_supported" if supported else "retain_official_v2_0p33",
        "limitation": "Internal check is historically exposed for original v2 and is not independent; hidden evaluator parity unverified."}
    with destination.open("x", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2)
    print("INTERNAL", destination, "v2", baseline["summary"], "v3_fixed", fixed["summary"],
          "v3_tuned", selected["summary"], "CI", interval, "decision", artifact["decision"], flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("tune", "internal"))
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args()
    if args.phase == "tune":
        tune()
    elif args.frozen is None:
        parser.error("--frozen required for internal scoring")
    else:
        internal(args.frozen)
