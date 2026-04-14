# DeepBravais: Training & Results Report

This document outlines the hardware setup, data generation timings, training parameters, and final evaluation metrics for the DeepBravais 1D-ResNet model. 

You can view the full interactive run in the official Kaggle notebook:
**[DeepBravais PXRD Classification (Kaggle)](https://www.kaggle.com/code/coolmann2049/deepbravias-pxrd-classification)**

---

## 1. Environment & Hardware
*   **Platform**: Kaggle Environment
*   **GPU**: 1x Tesla P100-PCIE (16GB)
*   **CPU**: Kaggle Standard CPU environment (for dataset generation)

---

## 2. Dataset Generation (Numpy-Engine)

Instead of relying on the internet to stream real datasets, 500,000 synthetic X-Ray diffraction patterns were generated from first principles directly in the Kaggle environment. We used the custom High-Performance Vectorized Numpy engine (`--numpy` flag) to bypass pymatgen's overhead.

*   **Dataset Size**: 500,000 samples 
*   **Distribution**: Perfectly balanced (~35,714 samples per class across 14 classes)
*   **Generation Time**: **1 hour 58 minutes** (Pure CPU) 
*   **Throughput**: ~4,200 patterns formulated per minute.
*   **Output Size**: ~1.4 GB `.npz` compressed array

*Command used:*
```bash
python data_generator.py --n_samples 500000 --numpy
```

---

## 3. Training Configuration

We trained the `ConvNeXt1D_Small` architectural variant (~1.6 Million parameters). 

*   **Total Epochs Used**: 20
*   **Batch Size**: 256
*   **Initial Learning Rate**: 3e-4 (Decaying smoothly to zero via CosineDecay)
*   **Loss Function**: Sparse Categorical Crossentropy
*   **Data Split**: 350k Train / 75k Validation / 75k Test
*   **Total Training Time**: **1 hour 3 minutes** (using the Tesla P100 GPU)
*   **Throughput**: ~104 seconds per epoch.

*Command used:*
```bash
python train.py --model small --epochs 20 --batch 256 --lr 3e-4 --dropout 0.5
```

---

## 4. Final Evaluation & Results

At the end of Epoch 20, the model achieved exceptional classification metrics on the **75,000 held-out test patterns**.

*   **Test Loss**: 0.1228
*   **Test Accuracy**: **96.78%**
*   **Macro F1-Score**: 0.9676

### Per-Class Classification Report

| Symbol | Bravais Lattice Type | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **aP** | Triclinic P | 0.9448 | 0.9291 | 0.9369 | 5358 |
| **mP** | Monoclinic P | 0.9404 | 0.8719 | 0.9049 | 5357 |
| **mC** | Monoclinic C | 0.9525 | 0.9239 | 0.9379 | 5358 |
| **oP** | Orthorhombic P | 0.9476 | 0.9558 | 0.9517 | 5357 |
| **oI** | Orthorhombic I | 0.9925 | 0.9634 | 0.9777 | 5357 |
| **oF** | Orthorhombic F | 0.9928 | 0.9731 | 0.9828 | 5357 |
| **oC** | Orthorhombic C | 0.9353 | 0.9601 | 0.9475 | 5357 |
| **tP** | Tetragonal P | 0.9549 | 0.9951 | 0.9746 | 5357 |
| **tI** | Tetragonal I | 0.9419 | 0.9808 | 0.9610 | 5357 |
| **hR** | Trigonal R | **0.9968** | **0.9996** | **0.9982** | 5357 |
| **hP** | Hexagonal P | 0.9679 | 0.9976 | 0.9825 | 5357 |
| **cP** | Cubic P | **1.0000** | **1.0000** | **1.0000** | 5357 |
| **cI** | Cubic I | 0.9976 | **1.0000** | **0.9988** | 5357 |
| **cF** | Cubic F | 0.9855 | 0.9994 | 0.9924 | 5357 |
| | | | | | |
| **Overall** | **Macro Average** | **0.9679** | **0.9678** | **0.9676** | **75000** |

### Key Takeaways
1. **Cubic Systems Perfected**: The model learns to classify Cubic Lattices (cP, cI, cF) nearly perfectly across different peak spacing patterns with an F1 score reaching 1.0000 on Primitive Cubic (cP).
2. **Speed + Accuracy Convergence**: The `numpy` physical engine proves extremely viable by successfully rendering 500,000 highly precise patterns in just 2 hours on CPU. Combining this with 1 hour of fine-tuning yields a 96.8% accurate classifier in a relatively constrained compute environment.
