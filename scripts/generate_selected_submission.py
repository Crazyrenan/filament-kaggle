"""Generate a unique locally validated CSV from an explicit selection manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg.benchmark import code_revision, load_candidate, sha256_file
from filament_seg.submission import generate_submission, validate_submission_rows


def generate_from_manifest(manifest_path: str | Path) -> tuple[Path, Path]:
    source = Path(manifest_path).resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest["status"] not in {"provisional", "verified"}:
        raise ValueError("Selection manifest status must be provisional or verified")
    checkpoint = Path(manifest["checkpoint"])
    if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint SHA-256 mismatch")
    if code_revision(ROOT)["source_sha256"] != manifest["code_revision"]["source_sha256"]:
        raise ValueError("Inference/evaluator source changed since selection")
    if manifest["preprocessing"]["normalization"] != "divide_by_255":
        raise ValueError("Unsupported selected preprocessing")
    model, metadata = load_candidate(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    if metadata["architecture"] != manifest["architecture"] or metadata["experiment_id"] != manifest["experiment_id"]:
        raise ValueError("Selected model does not match manifest")
    images = sorted((ROOT / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images").glob("*.jpeg"))
    if len(images) != 180:
        raise ValueError(f"Expected 180 test images, found {len(images)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "outputs" / f"submission_{manifest['status']}_{manifest['experiment_id']}_{stamp}.csv"
    post = manifest["postprocessing"]
    tiling = manifest["tiling"]
    if post["instance_method"] != "connected_components":
        raise ValueError("Unsupported selected instance extraction method")
    frame = generate_submission(
        model, images, output_csv_path=output,
        device="cuda" if torch.cuda.is_available() else "cpu",
        threshold=float(post["threshold"]), min_area=int(post["min_area"]),
        tile_size=int(tiling["tile_size"]), overlap=int(tiling["overlap"]),
        batch_size=int(tiling["batch_size"]),
    )
    if frame.empty or validate_submission_rows(frame.to_dict("records"), images) != len(frame):
        raise ValueError("Selected submission has no valid instance rows")
    loaded = pd.read_csv(output, dtype=str)
    if list(loaded.columns) != ["filament_id", "segmentation_rle"] or len(loaded) != len(frame):
        raise ValueError("Saved submission differs from validated predictions")
    provenance = {
        "selection_status": manifest["status"], "selection_manifest": str(source),
        "selection_manifest_sha256": sha256_file(source),
        "experiment_id": manifest["experiment_id"], "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "postprocessing": post, "tiling": tiling, "preprocessing": manifest["preprocessing"],
        "evaluator": manifest["evaluator"], "source_revision": manifest["code_revision"],
        "csv": str(output), "csv_sha256": sha256_file(output),
        "test_images": len(images), "predicted_filaments": len(frame),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation": "local schema and compressed COCO RLE decode; not a Kaggle upload",
    }
    provenance_path = output.with_suffix(".provenance.json")
    with provenance_path.open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    print(f"PASS submission={output} rows={len(frame)} provenance={provenance_path}", flush=True)
    return output, provenance_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    generate_from_manifest(args.manifest)
