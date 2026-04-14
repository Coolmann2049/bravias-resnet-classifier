# DeepBravais

> **1D Residual Network for Powder X-Ray Diffraction Classification of the 14 Bravais Lattices**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.12+](https://img.shields.io/badge/tensorflow-2.12+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

**DeepBravais** classifies synthetic powder X-ray diffraction (PXRD) patterns into one of the **14 Bravais lattice types** using a custom **1D Residual Network (ResNet-1D)** implemented in TensorFlow/Keras.

Rather than relying on noisy experimental databases, the project employs a **Physics-Informed Augmentation** strategy: patterns are synthesised from first principles using `pymatgen`, broadened analytically (Scherrer/Pseudo-Voigt), and corrupted with Poisson photon-counting noise. This gives full control over data quality and distribution.

---

## The 14 Bravais Lattices

| ID | Symbol | System         | Centering |
|----|--------|----------------|-----------|
|  0 | aP     | Triclinic      | P         |
|  1 | mP     | Monoclinic     | P         |
|  2 | mC     | Monoclinic     | C         |
|  3 | oP     | Orthorhombic   | P         |
|  4 | oI     | Orthorhombic   | I         |
|  5 | oF     | Orthorhombic   | F         |
|  6 | oC     | Orthorhombic   | C         |
|  7 | tP     | Tetragonal     | P         |
|  8 | tI     | Tetragonal     | I         |
|  9 | hR     | Trigonal       | R         |
| 10 | hP     | Hexagonal      | P         |
| 11 | cP     | Cubic          | P         |
| 12 | cI     | Cubic          | I         |
| 13 | cF     | Cubic          | F         |

---

## Physics Background

### Why Q-Space Instead of 2θ-Space?

The standard output from a diffractometer is expressed in **2θ** (twice the Bragg angle). However, training a CNN on 2θ patterns has a critical flaw: when the lattice constant `a` changes, all peak positions shift *non-linearly* in 2θ-space (because Bragg's law involves a sine), which makes the patterns look completely different to a convolution kernel.

**Q-space (momentum transfer)** solves this. Q is defined as:

```
Q = 4π sin(θ) / λ     [Å⁻¹]
```

where `θ = 2θ/2` is the Bragg angle and `λ` is the X-ray wavelength. Substituting Bragg's law (`nλ = 2d sin θ`):

```
Q = 2πn / d_hkl
```

This means **Q-peak positions scale linearly with 1/d** (the reciprocal lattice vector length). When a lattice constant changes, all peaks shift *together by a uniform multiplicative factor*. A CNN's **translation invariance** in Q-space therefore maps directly to **scale invariance** in real space — exactly the inductive bias we need to generalise across different unit-cell sizes.

### Scherrer Broadening & Pseudo-Voigt Profiles

Real diffraction peaks are not delta functions. Finite crystallite size broadens them according to the **Scherrer equation**:

```
β = Kλ / (L cos θ)
```

where `β` is the FWHM, `L` is the crystallite size, and `K ≈ 0.94`. This broadening is simulated using a **Pseudo-Voigt profile**:

```
PV(Q) = η·L(Q) + (1-η)·G(Q)
```

a linear combination of a Lorentzian `L` (dominant for small crystallites) and a Gaussian `G` (dominant for large crystallites). The mixing parameter `η ∈ [0,1]` and the FWHM are randomised per sample for augmentation.

### Poisson Noise

X-ray photon detection is a counting process; noise obeys **Poisson statistics**:

```
I_noisy ~ Poisson(I_true × scale)
```

The photon-count scale is drawn uniformly from `[500, 5000]` per sample, simulating both low-flux laboratory diffractometers and high-flux synchrotron setups.

---

## Architecture: ResNet-1D

```
Input  (N, 1024, 1)
   │
   ▼
Stem Conv1D(64, k=7) → BN → ReLU → MaxPool(3, stride=2)
   │                                           ↓ (512, 64)
Stage 1: 2× ResBlock(64,  k=7, stride=1)     ↓ (512, 64)
Stage 2: 2× ResBlock(128, k=5, stride=2)     ↓ (256, 128)
Stage 3: 2× ResBlock(256, k=5, stride=2)     ↓ (128, 256)
Stage 4: 2× ResBlock(512, k=3, stride=2)     ↓ (64,  512)
   │
   ▼
BN → ReLU
   │
Flatten  ← ⚠️  NOT GlobalAveragePooling — see note below
   │
Dense(512) → BN → ReLU → Dropout(0.4)
Dense(128) → BN → ReLU → Dropout(0.4)
Dense(14,  softmax)
```

### Why Flatten, Not GlobalAveragePooling?

`GlobalAveragePooling1D` discards all positional information — it cannot tell *where* a peak (or its absence) occurs in Q-space. **Systematic absences** (e.g., body-centred lattices extinguish all `h+k+l=odd` reflections) are encoded as *missing peaks at specific Q-positions*, not as an overall intensity reduction. Using `Flatten` preserves these positional fingerprints and lets the dense layers learn rules like "no peak at Q ≈ 1.4 Å⁻¹ but a peak at Q ≈ 2.0 Å⁻¹ → body-centred".

### ResBlock Pre-Activation Design

Each residual block follows the **pre-activation** scheme of He et al. (2016):

```
BN → ReLU → Conv1D → [SpatialDropout] → BN → ReLU → Conv1D
     └──────────── projection shortcut (if dims change) ─────┘
```

This improves gradient flow, especially in deeper networks.

---

## Repository Structure

```
bravais-resnet-classifier/
├── data/
│   ├── processed/              ← final dataset (.npz / .h5) stored here
│   └── simxrd_shards/          ← cached SimXRD-4M .db shards (auto-created)
├── outputs/
│   ├── checkpoints/            ← best model weights (saved during training)
│   ├── plots/                  ← training curves + confusion matrix
│   └── deepbravais_final.keras ← full saved model for inference
├── data_generator.py           ← physics-informed PXRD synthesis (pymatgen)
├── data_loader.py              ← SimXRD-4M database importer (Hugging Face)
├── models.py                   ← ResNet-1D architecture
├── train.py                    ← training loop + evaluation
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `pymatgen` pulls in several scientific libraries. On slow connections, this may take a few minutes. On Kaggle/Colab, it is available as a pre-installed package.

### 2a. Generate the dataset (physics simulation — no internet required)

```bash
# Default: 50,000 samples (~2–5 min on a modern CPU)
python data_generator.py

# Custom settings
python data_generator.py --n_samples 100000 --output data/processed/large.npz
```

This creates `data/processed/dataset.npz` containing:
- `X`: float32 array of shape `(N, 1024)` — PXRD patterns in Q-space
- `y`: int32 array of shape `(N,)` — Bravais lattice label (0–13)
- `class_names`: string array of the 14 Bravais symbols

---

### 2b. Download the dataset from SimXRD-4M (real database — recommended for Kaggle)

As an alternative to on-the-fly simulation, you can pull patterns directly from
**SimXRD-4M** — a dataset of 4 million PXRD patterns derived from 119,569 real
crystal structures in the Materials Project, published at ICLR 2025 [5].

`data_loader.py` streams the dataset shard-by-shard from Hugging Face, maps
all 230 space groups to the 14 Bravais lattices using the canonical ITA table,
performs stratified sampling so every class is equally represented, and saves
the result in the same `.npz` format that `train.py` expects.

#### Install additional dependencies

```bash
pip install ase>=3.22.0 huggingface_hub>=0.20.0
# (or just: pip install -r requirements.txt)
```

#### Basic usage

```bash
# Download & balance: 5 000 samples × 14 classes = 70 000 total (default)
python data_loader.py

# Larger subset for better generalisation
python data_loader.py --n_per_class 10000 --output data/processed/simxrd_10k

# Save as HDF5 instead of NumPy (faster random-access reads during training)
python data_loader.py --n_per_class 5000 --format h5

# Quick smoke-test: only pull 2 shards, 200 samples/class
python data_loader.py --max_shards 2 --n_per_class 200

# Offline / Kaggle: shards already downloaded to a local folder
python data_loader.py --no_download --db_dir /kaggle/input/simxrd-shards
```

#### CLI reference

| Argument | Default | Description |
|---|---|---|
| `--n_per_class` | 5000 | Samples to collect per Bravais class |
| `--output` | `data/processed/simxrd_balanced` | Output path (no extension) |
| `--format` | `npz` | `npz` (NumPy compressed) or `h5` (HDF5) |
| `--db_dir` | `data/simxrd_shards` | Local cache for downloaded .db shards |
| `--max_shards` | -1 (all) | Limit shard downloads — useful for quick tests |
| `--no_download` | False | Use already-downloaded shards, skip network |
| `--seed` | 42 | Random seed for reproducibility |

#### Output file (same schema as `data_generator.py`)

- `X`: float32 `(N, 1024)` — PXRD patterns, Q-space, normalised \[0, 1\]
- `y`: int32 `(N,)` — Bravais label 0–13
- `class_names`: 14 Bravais symbols (`aP`, `mP`, … `cF`)
- `Q_axis`: the 1024-bin Q-axis shared with `data_generator.py`

#### How SimXRD-4M patterns are converted

SimXRD-4M stores patterns in **d–I format** (lattice-plane spacing vs.
intensity). `data_loader.py` converts these to the same Q-space grid used
by `data_generator.py`:

```
Q = 2π / d   [Å⁻¹]
```

Peaks are then binned onto the shared 1024-point Q-axis spanning
\[0.7, 6.0\] Å⁻¹, and normalised to \[0, 1\]. No additional broadening
or noise is applied — SimXRD-4M already simulates 33 physical conditions
(grain size, stress, thermal vibrations, instrumental zero-shift, etc.).


### 3. Train the model

```bash
# Default: 50 epochs, batch=256, Adam lr=1e-3
python src/train.py

# On a GPU (Kaggle T4 / Colab):
python src/train.py --epochs 80 --batch 512

# Lightweight run on CPU:
python src/train.py --model small --epochs 30 --batch 128
```

Training produces:
- `outputs/checkpoints/deepbravais_best_<timestamp>.weights.h5` — best checkpoint
- `outputs/plots/training_curves.png` — loss/accuracy vs. epoch
- `outputs/plots/confusion_matrix.png` — per-class confusion heatmap
- `outputs/deepbravais_final.keras` — full model for inference

### 4. Resume from checkpoint (time-limited environments)

```python
from src.models import build_resnet1d
model = build_resnet1d()
model.load_weights("outputs/checkpoints/deepbravais_best_<timestamp>.weights.h5")
```

---

## Expected Performance

On 50,000 samples with default hyperparameters and a GPU:

| Metric             | Value (approx.) |
|--------------------|-----------------|
| Test Accuracy      | ~92–96%         |
| Training time (GPU)| ~15–25 min      |
| Training time (CPU)| ~2–4 hr         |
| Model parameters   | ~18 M           |

The cubic lattices (cP/cI/cF) and orthorhombic lattices (oP/oI/oF/oC) are the hardest to distinguish — the confusion matrix typically shows the most off-diagonal mass there.

---

## Hyperparameter Reference

| Argument     | Default | Description                            |
|--------------|---------|----------------------------------------|
| `--n_samples`| 50000   | Total training samples to generate     |
| `--epochs`   | 50      | Training epochs                        |
| `--batch`    | 256     | Mini-batch size                        |
| `--lr`       | 1e-3    | Initial Adam learning rate             |
| `--dropout`  | 0.4     | Dropout in dense head                  |
| `--model`    | full    | `full` (~18M params) or `small` (~2M)  |

---

## Limitations & Future Work

- **Single-phase only.** Real samples often contain mixtures. Multi-phase decomposition is a natural extension.
- **Texture effects** (preferred crystallographic orientation) are not simulated. Adding March-Dollase texture would increase realism.
- **Temperature factors** (Debye-Waller) are not modelled; adding them would modulate peak intensities at high-Q.
- **Background scattering** (amorphous hump, air scatter) could be added as a polynomial baseline augmentation.
- **Experimental validation** against the ICDD PDF-2/PDF-4 database would confirm generalisation beyond synthetic data.

---

## References

1. He, K. et al. (2016). *Identity Mappings in Deep Residual Networks.* ECCV 2016.
   https://arxiv.org/abs/1603.05027

2. Warren, B.E. (1990). *X-Ray Diffraction.* Dover Publications.

3. Ong, S.P. et al. (2013). *Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis.*
   Computational Materials Science, 68, 314–319.
   https://doi.org/10.1016/j.commatsci.2012.10.028

4. Toby, B.H. & Von Dreele, R.B. (2013). *GSAS-II: the genesis of a modern open-source all purpose crystallography software package.*
   J. Appl. Cryst., 46, 544–549.
   https://doi.org/10.1107/S0021889813003531

5. Cao, B., Dong, S., Liang, J., Luo, D., & Lookman, T. (2024).
   *SimXRD-4M: Big Simulated X-ray Diffraction Data Accelerates the Crystalline Symmetry Classification.*
   ICLR 2025.
   https://arxiv.org/abs/2406.15469
   Dataset: https://huggingface.co/datasets/caobin/SimXRD
   Code: https://github.com/Bin-Cao/SimXRD

6. Larsen, A.H. et al. (2017). *The Atomic Simulation Environment — A Python library for working with atoms.*
   J. Phys.: Condens. Matter, 29, 273002.
   https://doi.org/10.1088/1361-648X/aa680e

---

## License

MIT © 2024 DeepBravais Contributors
