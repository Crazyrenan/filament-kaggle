"""Unit tests for DatasetInspector utility."""

from pathlib import Path
import numpy as np
from PIL import Image

from filament_seg.dataset import DatasetInspector


def test_inspector_nonexistent_directory():
    inspector = DatasetInspector("nonexistent_directory_xyz")
    assert not inspector.exists
    report = inspector.generate_report()
    assert report["exists"] is False
    assert report["images"]["total_images"] == 0
    # Should not raise exception
    inspector.print_summary(report)


def test_inspector_with_mock_dataset(tmp_path: Path):
    # Create mock images and annotations
    images_dir = tmp_path / "train" / "images"
    images_dir.mkdir(parents=True)

    # Save a mock PNG
    mock_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(mock_array)
    img.save(images_dir / "sample_001.png")

    # Save a mock annotation CSV
    ann_file = tmp_path / "train_labels.csv"
    ann_file.write_text("image_id,filament_count\nsample_001,3\n", encoding="utf-8")

    inspector = DatasetInspector(tmp_path)
    assert inspector.exists

    files = inspector.discover_files()
    assert len(files["images"]) == 1
    assert len(files["annotations"]) == 1

    report = inspector.generate_report()
    assert report["images"]["total_images"] == 1
    assert ".png" in report["images"]["formats"]
    assert report["annotations"]["total_annotation_files"] == 1
    assert report["annotations"]["files"][0]["filename"] == "train_labels.csv"
