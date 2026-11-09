"""Write a concise comparison report and enrich the experiment registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.experiment_tracking import _update_registry
from filament_seg.benchmark import sha256_file


def summary(report_path: Path) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    registry = ROOT / "reports/experiment_registry.json"
    lines = [
        "# Provisional multi-architecture benchmark", "",
        f"Run: `{report_path.name}`", "",
        "Selection is provisional. Original v2 previously used all 142 validation images; "
        "the internal-check partition is not independent. Pair counting was checked against "
        "the organizer's public notebook v6 on synthetic cases; hidden Kaggle scorer and GT "
        "conversion parity remain unverified.", "",
        "| Experiment | Protocol | Images | Global PQ | SQ | RQ | TP | FP | FN | Inference s/image | Peak VRAM MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["results"]:
        experiment = item["model"]["experiment_id"]
        latency = item["inference"]["seconds_per_image"]
        peak = item["inference"]["peak_gpu_memory_mib"]
        for label, key in (("Fixed full 142", "fixed_full_142"),
                           ("Fixed internal", "fixed_internal"),
                           ("Tuned internal", "tuned_internal")):
            score = item[key]["summary"]
            lines.append(
                f"| {experiment} | {label} | {score['num_images']} | {score['global_pq']:.6f} | "
                f"{score['sq']:.6f} | {score['rq']:.6f} | {score['tp']} | {score['fp']} | "
                f"{score['fn']} | {latency:.3f} | {peak:.1f} |"
            )
        chosen = item["selected_postprocessing"]
        tuned = item["tuned_internal"]["summary"]
        source_names = ["dataset.py", "preprocessing.py", "losses.py"]
        source_names += (["transformer.py", "transformer_train.py"]
                         if item["model"]["architecture"] == "SegFormer"
                         else ["model.py", "train.py"])
        training_source_hashes = None if experiment == "unet_resnet34_v2" else {
            f"src/filament_seg/{name}": sha256_file(ROOT / "src/filament_seg" / name)
            for name in source_names
        }
        effective_config = Path(item["model"]["checkpoint"]).parent / "config.yaml"
        _update_registry(registry, experiment, {
            "checkpoint_sha256": item["model"]["checkpoint_sha256"],
            "benchmark_report": str(report_path.resolve()),
            "validation_fingerprint": report["validation_fingerprint"],
            "fixed_full_142_global_pq_public_notebook": item["fixed_full_142"]["summary"]["global_pq"],
            "fixed_internal_global_pq_public_notebook": item["fixed_internal"]["summary"]["global_pq"],
            "tuned_internal_global_pq_public_notebook": tuned["global_pq"],
            "tuned_internal_sq": tuned["sq"], "tuned_internal_rq": tuned["rq"],
            "tuned_internal_tp": tuned["tp"], "tuned_internal_fp": tuned["fp"],
            "tuned_internal_fn": tuned["fn"],
            "selected_postprocessing": chosen,
            "inference_seconds_per_image": latency,
            "inference_peak_gpu_memory_mib": peak,
            "selection_status": "provisional" if experiment == report["ranking"][0] else "candidate",
            "training_source_sha256": training_source_hashes,
            "effective_config_sha256": sha256_file(effective_config) if effective_config.is_file() else None,
        })
    lines.extend(["", "## Selection", "",
                  f"Provisional winner: **{report['ranking'][0]}** by tuned internal-check Global PQ.",
                  "", "Post-processing was selected on tuning images only. The fixed full-142 results "
                  "are historical comparisons and did not select the winner.", "",
                  "The earlier 0.386869 and 0.400488 original-v2 PQ references used a different "
                  "local protocol and are not directly comparable to this benchmark.", ""])
    destination = report_path.with_suffix(".md")
    with destination.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Updated registry={registry}; summary={destination}", flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    summary(args.report)
