"""Unit tests for utility functions."""

from pathlib import Path
import pytest

from filament_seg.utils import get_device, get_logger, load_config, project_path, project_root, save_config, set_seed


def test_set_seed():
    set_seed(123)
    # Ensure it executes without raising exceptions


def test_load_and_save_config(tmp_path: Path):
    sample_config = {"project": "solar_filament", "epochs": 10, "nested": {"lr": 0.001}}
    config_file = tmp_path / "test_config.yaml"

    save_config(sample_config, config_file)
    assert config_file.exists()

    loaded = load_config(config_file)
    assert loaded == sample_config


def test_load_config_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config_file_path_12345.yaml")


def test_get_logger():
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"


def test_get_device():
    dev = get_device("cpu")
    assert "cpu" in str(dev)


def test_project_paths_independent_of_working_directory(monkeypatch, tmp_path):
    expected = project_root()
    monkeypatch.chdir(tmp_path)
    assert project_root() == expected
    assert project_path("checkpoints/unet_resnet34_v2/latest.pth") == expected / "checkpoints/unet_resnet34_v2/latest.pth"
    assert project_path(tmp_path / "model.pth") == tmp_path / "model.pth"
