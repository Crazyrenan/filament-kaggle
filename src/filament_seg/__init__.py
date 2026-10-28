"""Solar Filament Instance Segmentation package.

End-to-end deep learning pipeline for detecting and segmenting solar filaments
in astronomical imagery (Kaggle 2026).
"""

__version__ = "0.1.0"

from filament_seg.utils import get_device, get_logger, load_config, set_seed

__all__ = [
    "__version__",
    "set_seed",
    "load_config",
    "get_device",
    "get_logger",
]
