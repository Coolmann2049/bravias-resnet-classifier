# DeepBravais

> **1D Residual Network for Powder X-Ray Diffraction Classification of the 14 Bravais Lattices**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.12+](https://img.shields.io/badge/tensorflow-2.12+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Site](https://img.shields.io/badge/🌐%20Live%20Site-deepbravais.yashdeep--jha.site-2a6e3f?style=flat)](https://deepbravais.yashdeep-jha.site/)

> 🌐 **[deepbravais.yashdeep-jha.site](https://deepbravais.yashdeep-jha.site/)** — Interactive results, architecture walkthrough, and training report

---

## Overview

**DeepBravais** classifies synthetic powder X-ray diffraction (PXRD) patterns into one of the **14 Bravais lattice types** using a custom **1D Residual Network (ResNet-1D)** implemented in TensorFlow/Keras.

Rather than relying on noisy experimental databases, the project employs a **Physics-Informed Augmentation** strategy: patterns are synthesised from first principles, broadened analytically (Scherrer/Pseudo-Voigt), and corrupted with Poisson photon-counting noise. This gives full control over data quality and distribution. 

### Recent Updates & Optimizations
- **Modular Architecture**: The codebase has been refactored into a clean `src/` package (separating config, physics, models, trainer, dataset, and visualization logic). 
- **Numpy-Only Physics Engine**: A highly optimized vectorised physics engine (via the `--numpy` flag) completely bypasses `pymatgen` for peak calculation, resulting in a **20-50x speedup** for generating hundreds of thousands of samples natively in numpy.

---

## Performance & Results

The final `ConvNeXt1D_Small` architecture (~1.6M parameters) was trained on **500,000 synthetic samples** evenly distributed across all 14 Bravais lattices.

**Test Set Evaluation (75,000 held-out samples):**
*   **Test Loss**: 0.1228
*   **Test Accuracy**: **96.78%**
*   **Macro Average F1-Score**: 0.9676

The model successfully distinguishes critical systematic absences for challenging centering rules, achieving >90% precision and recall across all classes, and near-perfect scores (0.99+) for Cubic (cP, cI, cF) and Trigonal (hR) systems.

---

## Pre-trained Weights

You can immediately start inference or fine-tuning using our best checkpoint weights (from Epoch 20, achieving 96.78% test accuracy):

**📥 [Download `deepbravais_best.weights.h5` from Google Drive](https://drive.google.com/file/d/11ap4gaBPyd5HjaN1aH_g-tDWpwdeIc3P/view?usp=sharing)**

To load these weights into the architecture:
```python
from src.models import build_resnet1d_small
model = build_resnet1d_small()
model.load_weights("deepbravais_best.weights.h5")
```

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

### Why a 1D CNN for PXRD?

A PXRD pattern is an inherently **1D signal** — a function of the scattering angle (or Q). Treating it as a 1D sequence and applying a 1D CNN is a natural fit: local convolution kernels learn to detect peaks and their neighbourhoods, while residual connections allow gradients to flow through the depth required to capture multi-peak relationships.

This design is independently validated by **AlphaDiffract** [7] (Andrejevic et al., 2026), a concurrent framework for automated PXRD analysis that also applies a 1D deep convolutional network — a 1D ConvNeXt — to predict crystal systems, space groups, and lattice parameters directly from 1D diffraction patterns, trained on 31 million simulated patterns. The convergence of both approaches on the *1D-CNN-over-Q-space* paradigm provides strong evidence that this is the right inductive bias for this problem.

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

Each residual block follows the **pre-activation** scheme of He et al. (2016) [1]:

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
├── src/
│   ├── config.py               ← core configurations & lattice sets
│   ├── dataset.py              ← balanced sampling & I/O logic
│   ├── models.py               ← ResNet-1D & ConvNeXt1D implementations
│   ├── physics.py              ← simulation physics (d-spacing / Scherrer)
│   ├── trainer.py              ← learning rate schedules / callbacks
│   └── visualization.py        ← headless matplotlib utilities
├── data_generator.py           ← high-speed data generator CLI (--numpy flag)
├── data_loader.py              ← SimXRD-4M Hugging Face downloader
├── train.py                    ← training CLI
├── models.py                   ← backwards-compatibility shim
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `pymatgen` pulls in several scientific libraries. On slow connections, this may take a few minutes. If you use the fast `--numpy` backend, `pymatgen` is not required.

### 2a. Generate the dataset (physics simulation — no internet required)

```bash
# Our ultra-fast Numpy-only backend (~20-50x speedup vs pymatgen)
python data_generator.py --n_samples 50000 --numpy

# Custom settings
python data_generator.py --n_samples 500000 --numpy --output data/processed/large.npz
```

This creates `data/processed/dataset.npz` containing:
- `X`: float32 array of shape `(N, 1024)` — PXRD patterns in Q-space
- `y`: int32 array of shape `(N,)` — Bravais lattice label (0–13)
- `class_names`: string array of the 14 Bravais symbols

---

### 2b. Download the dataset from SimXRD-4M (real database — optional)

As an alternative to on-the-fly simulation, you can pull patterns directly from
**SimXRD-4M** — a dataset of 4 million PXRD patterns derived from 119,569 real
crystal structures in the Materials Project, published at ICLR 2025 [5].

`data_loader.py` streams the dataset shard-by-shard from Hugging Face (**AI4Spectro**), maps
all 230 space groups to the 14 Bravais lattices using the canonical ITA table,
performs stratified sampling so every class is equally represented, and saves
the result in the same `.npz` format that `train.py` expects.

#### Basic usage

```bash
# Download & balance: 5,000 samples × 14 classes = 70,000 total (default)
python data_loader.py

# Larger subset for better generalisation
python data_loader.py --n_per_class 10000 --output data/processed/simxrd_10k
```

> ⚠️ **Kaggle Disk Warning**: The full SimXRD-4M training dataset is very large (>74GB per part). If executing in constrained environments like Kaggle (~20GB limit), it is strongly recommended to use step 2a (`data_generator.py --numpy`) instead. 


### 3. Train the model

```bash
# Default: 50 epochs, batch=256, Adam lr=1e-3
python train.py
```

Training produces:
- `outputs/checkpoints/deepbravais_best_<timestamp>.weights.h5` — best checkpoint
- `outputs/plots/training_curves.png` — loss/accuracy vs. epoch
- `outputs/plots/confusion_matrix.png` — per-class confusion heatmap
- `outputs/deepbravais_final.keras` — full model for inference

---

## Hyperparameter Reference

| Argument     | Default | Description                            |
|--------------|---------|----------------------------------------|
| `--n_samples`| 50000   | Total training samples to generate     |
| `--epochs`   | 50      | Training epochs                        |
| `--batch`    | 256     | Mini-batch size                        |
| `--lr`       | 1e-3    | Initial Adam learning rate             |
| `--dropout`  | 0.4     | Dropout in dense head                  |
| `--model`    | full    | `full` (~18M params) or `small` (~1.6M)|
| `--numpy`    | False   | If passed to data_generator.py, accelerates simulation by 20-50x using native matrix multiplication. |

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

4. Toby, B.H. & Von Dreele, R.B. (2013). *GSAS-II: the genesis of a modern open-source all purpose crystallography software package.*

5. Cao, B., Dong, S., Liang, J., Luo, D., & Lookman, T. (2024).
   *SimXRD-4M: Big Simulated X-ray Diffraction Data Accelerates the Crystalline Symmetry Classification.*
   ICLR 2025.
   https://arxiv.org/abs/2406.15469
   Original Dataset repository: https://huggingface.co/AI4Spectro
   Code: https://github.com/Bin-Cao/SimXRD

6. Larsen, A.H. et al. (2017). *The Atomic Simulation Environment — A Python library for working with atoms.*

7. Andrejevic, N., Du, M., Sharma, H., Horwath, J.P., Luo, A., Yin, X., Prince, M., Toby, B.H., & Cherukara, M.J. (2026).
   *AlphaDiffract: Automated Crystallographic Analysis of Powder X-ray Diffraction Data.*
   arXiv:2603.23367.
   https://arxiv.org/abs/2603.23367

---

## License

MIT © 2026 DeepBravais Contributors
