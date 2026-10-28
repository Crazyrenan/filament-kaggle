# Solar Filament Instance Segmentation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end deep learning and scientific computer-vision pipeline for automated detection and instance segmentation of solar filaments in astronomical solar observations. Built for the **Kaggle Solar Filament Segmentation Challenge 2026**.

---

## 1. Scientific Context

Solar filaments are massive ribbons of dense, cool plasma suspended within the intensely hot solar corona by magnetic flux ropes. When observed against the solar disk in monochromatic spectral lines (most prominently Hydrogen-alpha, $\text{H}\alpha$ at $656.28\text{ nm}$), they appear as dark, elongated absorption structures due to scattering. When projected beyond the solar limb against the darkness of space, these identical structures appear bright and are termed **solar prominences**.

```text
       Solar Corona (~1,000,000 K)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        \    Helical Magnetic Flux Rope    /
         \     Cool Plasma (~10,000 K)    /
   ============================================  <- Solar Chromosphere
                     Solar Disk
```

### Why Automated Segmentation Matters

1. **Space Weather Forecasting:** Filaments represent unstable magnetic equilibria. Their eruption frequently triggers Coronal Mass Ejections (CMEs) that propagate through interplanetary space, driving geomagnetic storms on Earth that disrupt satellite communications, GPS navigation, and terrestrial power grids.
2. **Morphological Characterization:** Manual tracing of filament spine coordinates, chirality, and footprint areas across terabytes of continuous full-disk solar observatory surveys (e.g., SDO/AIA, GONG, NVST, CHASE) is infeasible.
3. **Instance Separation Challenge:** Solar filaments frequently overlap, bifurcate, or appear fragmented across complex chromospheric backgrounds with intense active-region fibrils and sunspots. Converting semantic masks into verified physical instances requires robust, boundary-aware computer vision.

---

## 2. Architecture & Repository Structure

The repository is structured as a modular, installable Python package (`filament_seg`) paired with lightweight experiment notebooks and configuration files:

```text
solar-filament-segmentation/
├── src/
│   └── filament_seg/
│       ├── __init__.py          # Package exports and version metadata
│       ├── dataset.py           # Dataset loading & uncharacterized dataset inspector
│       ├── preprocessing.py     # Image loading, normalization, and Albumentations transforms
│       ├── model.py             # SMP segmentation architecture initialization
│       ├── losses.py            # BCE, Soft Dice, and combined BCEDiceLoss
│       ├── train.py             # Modular training and validation loops with checkpointing
│       ├── evaluate.py          # Metric calculations: IoU, Dice, Precision, Recall
│       ├── inference.py         # Prediction pipeline, watershed instance separation, RLE
│       └── utils.py             # Deterministic seed setting, config YAML I/O, device detection
├── notebooks/
│   ├── 01_eda.ipynb             # Dataset investigation and exploratory analysis
│   ├── 02_training.ipynb        # Staged training runner (calls filament_seg)
│   └── 03_submission.ipynb      # Kaggle inference and submission generator
├── configs/
│   └── baseline.yaml            # Single source of truth for hyperparameters
├── tests/                       # Unit tests for preprocessing, losses, and metrics
├── outputs/                     # Local experiment logs, metrics, and checkpoints
├── environment.yml              # Conda environment definition (solar-filament, Python 3.11)
├── requirements.txt             # Pip-compatible dependency manifest
├── pyproject.toml               # PEP 517 / 621 package build configuration
├── .gitignore                   # Ignores large files, weights, and caches
└── README.md                    # Project overview and engineering documentation
```

---

## 3. Staged Development Methodology

To guarantee scientific rigor, reproducibility, and prevent data leakage:

```text
[Phase 1: Project Setup] ────> [Phase 2: Dataset Discovery] ────> [Phase 3: Stage A Baseline]
  • Package initialization       • Dimensions & formats           • Semantic U-Net (ResNet34)
  • Conda env & pyproject        • Intensity distributions        • BCE + Dice combined loss
  • Test suite verified          • Annotation schema              • Validation tracking
                                                                          │
                                                                          ▼
[Phase 5: Optimization]  <──── [Phase 4: Stage B Instances] <─────────────┘
  • Encoders (EffNet/ConvNeXt)   • Marker-controlled Watershed
  • Hyperparameter tuning        • Connected components
  • Resolution ablation          • False-positive filtering
```

- **Stage A (Baseline):** Semantic segmentation using U-Net (`segmentation-models-pytorch`) with ResNet-34 backbone and combined BCE + Dice loss.
- **Stage B (Instance Segmentation):** Transform semantic probability maps into separated physical instances using distance transforms, core seeding, and marker-controlled watershed segmentation.
- **Stage C (Optimization):** Systematic single-variable ablations (backbone encoders, multi-scale inputs, boundary-weighted loss, threshold sweeps).

---

## 4. Local Installation & Setup

### Prerequisites
- Anaconda or Miniconda
- Git

### 1. Create and Activate the Conda Environment

```bash
# Clone the repository
git clone https://github.com/your-username/solar-filament-segmentation.git
cd solar-filament-segmentation

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate solar-filament
```

### 2. Install Package in Editable Mode

Install `filament_seg` in editable mode (`-e`) so changes to `src/` are immediately reflected:

```bash
pip install -e .
```

### 3. Verify Installation & Run Unit Tests

```bash
pytest tests/ -v
```

---

## 5. Kaggle Integration Strategy

The local codebase serves as the single source of truth. Kaggle Notebooks are kept lightweight and consume `filament_seg` via one of two deployment paths:

1. **Online Mode (Internet Enabled):**
   ```bash
   !git clone https://github.com/your-username/solar-filament-segmentation.git
   %cd solar-filament-segmentation
   !pip install -e .
   ```
2. **Offline Mode (Competition Inference / Internet Disabled):**
   - Package `src/` into a versioned Kaggle Dataset:
     ```bash
     tar -czvf filament_seg_src.tar.gz -C src filament_seg
     ```
   - In the Kaggle notebook:
     ```python
     import sys
     sys.path.append("/kaggle/input/filament-seg-src")
     import filament_seg
     ```

---

## 6. Engineering Standards

- **Strict Type Annotations:** All module signatures utilize PEP 484 type hints.
- **Reproducibility:** Deterministic seeds enforced across `random`, `numpy`, and `torch` via `set_seed()`.
- **Portable Paths:** All file operations utilize `pathlib.Path` to support cross-platform execution (Windows local dev and Linux Kaggle workers).
- **Metric Decoupling:** Semantic quality (IoU, Dice) is evaluated independently from instance separation metrics.
