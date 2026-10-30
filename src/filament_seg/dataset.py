"""Dataset loading, inspection utilities, and target-mask construction."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from filament_seg.utils import get_logger

logger = get_logger(__name__)

SUPPORTED_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".fits",
    ".fts",
    ".npy",
}
SUPPORTED_ANNOTATION_EXTS = {
    ".csv",
    ".json",
    ".xml",
    ".txt",
    ".rle",
    ".png",
    ".tif",
    ".tiff",
    ".npy",
}


class DatasetInspector:
    """Exploratory utility to inspect uncharacterized astronomical datasets.

    Designed to fulfill Section 4 dataset investigation criteria:
    - Image file formats, dimensions, and channel counts.
    - Annotation file formats and structures.
    - Image-to-annotation correspondence.
    - Intensity distributions and potential class imbalance.
    """

    def __init__(self, data_dir: str | Path) -> None:
        """Initialize inspector with the root dataset directory.

        Args:
            data_dir: Path to dataset root directory.
        """
        self.data_dir = Path(data_dir)
        self.exists = self.data_dir.exists()

    def discover_files(self) -> dict[str, list[Path]]:
        """Scan directory and group files by extension category.

        Returns:
            Dictionary mapping category ('images', 'annotations', 'other') to paths.
        """
        if not self.exists:
            return {"images": [], "annotations": [], "other": []}

        all_files = [p for p in self.data_dir.rglob("*") if p.is_file()]
        images: list[Path] = []
        annotations: list[Path] = []
        other: list[Path] = []

        for file_path in all_files:
            ext = file_path.suffix.lower()
            if ext in SUPPORTED_IMAGE_EXTS and not file_path.name.startswith("."):
                images.append(file_path)
            elif ext in {".csv", ".json", ".xml", ".rle"}:
                annotations.append(file_path)
            else:
                other.append(file_path)

        return {"images": images, "annotations": annotations, "other": other}

    def inspect_images(self, sample_limit: int = 50) -> dict[str, Any]:
        """Analyze image dimensions, channels, formats, and pixel intensities.

        Args:
            sample_limit: Maximum number of images to sample for pixel statistics.

        Returns:
            Dictionary containing image metadata and distribution statistics.
        """
        if not self.exists:
            return {"status": "data_dir_not_found", "total_images": 0}

        files = self.discover_files()["images"]
        total_images = len(files)
        if total_images == 0:
            return {"total_images": 0, "formats": {}, "shapes": []}

        formats: Counter[str] = Counter()
        shapes: list[tuple[int, ...]] = []
        intensity_stats: list[dict[str, float]] = []

        # Try to import cv2 or PIL for dimension inspection
        sample_files = files[:sample_limit]
        for path in sample_files:
            formats[path.suffix.lower()] += 1
            shape_info, stats = self._read_image_info(path)
            if shape_info is not None:
                shapes.append(shape_info)
            if stats is not None:
                intensity_stats.append(stats)

        # Count formats across all files
        for path in files[sample_limit:]:
            formats[path.suffix.lower()] += 1

        summary: dict[str, Any] = {
            "total_images": total_images,
            "sampled_count": len(sample_files),
            "formats": dict(formats),
            "unique_shapes": sorted(list(set(shapes))),
        }

        if intensity_stats:
            means = [s["mean"] for s in intensity_stats]
            mins = [s["min"] for s in intensity_stats]
            maxs = [s["max"] for s in intensity_stats]
            stds = [s["std"] for s in intensity_stats]
            summary["intensity_summary"] = {
                "overall_min": float(min(mins)),
                "overall_max": float(max(maxs)),
                "average_mean": float(sum(means) / len(means)),
                "average_std": float(sum(stds) / len(stds)),
            }

        return summary

    def inspect_annotations(self) -> dict[str, Any]:
        """Examine annotation files and schema without making assumptions.

        Returns:
            Dictionary detailing discovered annotation files and their schemas.
        """
        if not self.exists:
            return {"status": "data_dir_not_found"}

        annotation_files = self.discover_files()["annotations"]
        results: dict[str, Any] = {
            "total_annotation_files": len(annotation_files),
            "files": [],
        }

        for path in annotation_files:
            file_meta: dict[str, Any] = {
                "filename": path.name,
                "relative_path": str(path.relative_to(self.data_dir)),
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            }

            if path.suffix.lower() == ".csv":
                try:
                    import pandas as pd

                    df = pd.read_csv(path, nrows=5)
                    file_meta["columns"] = list(df.columns)
                    file_meta["sample_rows"] = df.to_dict(orient="records")
                except Exception as err:
                    file_meta["read_error"] = str(err)
            elif path.suffix.lower() == ".json":
                try:
                    with path.open("r", encoding="utf-8") as json_file:
                        data = json.load(json_file)
                    if isinstance(data, dict):
                        file_meta["keys"] = list(data.keys())
                    elif isinstance(data, list):
                        file_meta["item_count"] = len(data)
                        if data and isinstance(data[0], dict):
                            file_meta["sample_keys"] = list(data[0].keys())
                except Exception as err:
                    file_meta["read_error"] = str(err)

            results["files"].append(file_meta)

        return results

    def generate_report(self) -> dict[str, Any]:
        """Generate a complete dataset inspection report.

        Returns:
            Structured dictionary answering Section 4 criteria.
        """
        return {
            "data_dir": str(self.data_dir.resolve()),
            "exists": self.exists,
            "images": self.inspect_images(),
            "annotations": self.inspect_annotations(),
        }

    def print_summary(self, report: dict[str, Any] | None = None) -> None:
        """Print a formatted terminal summary of the dataset inspection.

        Args:
            report: Optional pre-generated report dictionary.
        """
        rep = report or self.generate_report()
        print("\n" + "=" * 60)
        print("SOLAR FILAMENT DATASET INSPECTION REPORT")
        print("=" * 60)
        print(f"Dataset Path: {rep.get('data_dir')}")
        print(f"Path Exists:  {rep.get('exists')}")

        if not rep.get("exists"):
            print("Status: Dataset directory does not exist or has not been mounted yet.")
            print("=" * 60 + "\n")
            return

        imgs = rep.get("images", {})
        print(f"\n[1] Images:")
        print(f"  - Total Count:   {imgs.get('total_images', 0)}")
        print(f"  - Formats:       {imgs.get('formats', {})}")
        print(f"  - Unique Shapes: {imgs.get('unique_shapes', [])}")
        if "intensity_summary" in imgs:
            istats = imgs["intensity_summary"]
            print(f"  - Pixel Range:   [{istats['overall_min']}, {istats['overall_max']}]")
            print(f"  - Mean / Std:    {istats['average_mean']:.2f} / {istats['average_std']:.2f}")

        anns = rep.get("annotations", {})
        print(f"\n[2] Annotations:")
        print(f"  - Total Files:   {anns.get('total_annotation_files', 0)}")
        for item in anns.get("files", []):
            print(f"  - {item['filename']} ({item['suffix']}, {item['size_bytes']} bytes)")
            if "columns" in item:
                print(f"    Columns: {item['columns']}")
            if "keys" in item:
                print(f"    Keys:    {item['keys']}")

        print("=" * 60 + "\n")

    def _read_image_info(
        self, path: Path
    ) -> tuple[tuple[int, ...] | None, dict[str, float] | None]:
        """Safely read image shape and pixel statistics."""
        try:
            import cv2
            import numpy as np

            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                shape = tuple(img.shape)
                stats = {
                    "min": float(np.min(img)),
                    "max": float(np.max(img)),
                    "mean": float(np.mean(img)),
                    "std": float(np.std(img)),
                }
                return shape, stats
        except Exception:
            pass

        # Fallback to PIL
        try:
            import numpy as np
            from PIL import Image

            with Image.open(path) as img_pil:
                arr = np.array(img_pil)
                shape = tuple(arr.shape)
                stats = {
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
                return shape, stats
        except Exception:
            return None, None


class SolarFilamentDataset:
    """Load physical MAGFiLO images and their COCO polygon annotations."""

    def __init__(
        self,
        image_dir: str | Path,
        annotation_path: str | Path,
        transform: Any | None = None,
        annotation_policy: str = "first",
    ) -> None:
        if annotation_policy != "first":
            raise ValueError("annotation_policy must be 'first'")

        self.image_dir = Path(image_dir)
        self.annotation_path = Path(annotation_path)
        self.transform = transform
        self.annotation_policy = annotation_policy

        with self.annotation_path.open("r", encoding="utf-8") as annotation_file:
            coco = json.load(annotation_file)

        self.images_by_filename: dict[str, list[dict[str, Any]]] = {}
        for image in coco["images"]:
            self.images_by_filename.setdefault(image["file_name"], []).append(image)

        self.filenames = list(self.images_by_filename)
        self.annotations_by_image_id: dict[str, list[dict[str, Any]]] = {}
        for annotation in coco["annotations"]:
            self.annotations_by_image_id.setdefault(annotation["image_id"], []).append(
                annotation
            )

        missing = [name for name in self.filenames if not (self.image_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} annotated images; first: {missing[0]}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import cv2
        import numpy as np
        import torch

        filename = self.filenames[idx]
        image = cv2.imread(str(self.image_dir / filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Could not read image: {self.image_dir / filename}")

        mask = np.zeros(image.shape, dtype=np.uint8)
        image_id = self.images_by_filename[filename][0]["id"]
        for annotation in self.annotations_by_image_id.get(image_id, []):
            for polygon in annotation["segmentation"]:
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                if len(points) >= 3:
                    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        image = torch.as_tensor(np.asarray(image), dtype=torch.float32)
        mask = torch.as_tensor(np.asarray(mask), dtype=torch.float32)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return {"image": image, "mask": mask}
