"""Smoke test to ensure the filament_seg package and all modules import correctly."""

import importlib


def test_package_import():
    pkg = importlib.import_module("filament_seg")
    assert hasattr(pkg, "__version__")
    assert pkg.__version__ == "0.1.0"


def test_submodules_import():
    submodules = [
        "filament_seg.utils",
        "filament_seg.dataset",
        "filament_seg.preprocessing",
        "filament_seg.model",
        "filament_seg.losses",
        "filament_seg.train",
        "filament_seg.evaluate",
        "filament_seg.inference",
    ]
    for module_name in submodules:
        mod = importlib.import_module(module_name)
        assert mod is not None
