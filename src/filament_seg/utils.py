"""Core utility functions for reproducibility, configuration, logging, and hardware detection."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import yaml


def get_logger(name: str = "filament_seg", level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a standard logger for the pipeline.

    Args:
        name: Name of the logger.
        level: Logging verbosity level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch (if available).

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration dictionary from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file is not valid YAML or is empty.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML file to contain a dictionary, got {type(data).__name__}")

    return data


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    """Save configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary to save.
        output_path: Path where the YAML file should be written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(config, file_handle, default_flow_style=False, sort_keys=False)


def get_device(preferred: str | None = None) -> Any:
    """Return available torch.device, with priority: CUDA -> MPS -> CPU.

    Args:
        preferred: Optional explicit preference ("cuda", "cpu", "mps").

    Returns:
        torch.device instance, or string fallback if PyTorch is not yet installed.
    """
    try:
        import torch

        if preferred:
            return torch.device(preferred)

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return preferred or "cpu"


def project_root() -> Path:
    """Return the source checkout root independently of the process working directory."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"Project root not found from {__file__}")
    return root


def project_path(path: str | Path) -> Path:
    """Resolve a configured path against this checkout, preserving absolute paths."""
    value = Path(path)
    return value if value.is_absolute() else project_root() / value
