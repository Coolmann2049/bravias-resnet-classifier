"""
data_generator.py
=================
CLI entry point — generates synthetic PXRD patterns using pymatgen
and saves them as a .npz file ready for train.py.

Physics simulation logic lives in src/physics.py.
I/O helpers live in src/dataset.py.

Usage
-----
    python data_generator.py                          # 50k samples
    python data_generator.py --n_samples 500000       # 500k samples
    python data_generator.py --n_samples 500000 \
                             --output data/processed/dataset_500k.npz
"""

import argparse
import os
import random
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    BRAVAIS_LATTICES, CLASS_NAMES, N_CLASSES,
    WAVELENGTH_CU_KA,
)
from src.physics import simulate_pattern
from src.dataset import save_npz

# Guard pymatgen import — give a clear error message if missing
try:
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False


# ===========================================================================
# Dataset generation
# ===========================================================================

def generate_dataset(n_samples:   int   = 50_000,
                     output_path: str   = "data/processed/dataset.npz",
                     wavelength:  float = WAVELENGTH_CU_KA,
                     seed:        int   = 42):
    """
    Generate ``n_samples`` physics-informed PXRD patterns distributed
    evenly across the 14 Bravais lattices, and save as a compressed .npz.

    Parameters
    ----------
    n_samples    : total number of samples to generate
    output_path  : output .npz path (extension added automatically)
    wavelength   : X-ray wavelength in Å (default Cu Kα = 1.5406 Å)
    seed         : reproducibility seed

    Returns
    -------
    X : (n_samples, 1024) float32
    y : (n_samples,)      int32
    """
    if not PYMATGEN_AVAILABLE:
        raise RuntimeError(
            "pymatgen is required. Install with:\n"
            "    pip install pymatgen"
        )

    np.random.seed(seed)
    random.seed(seed)

    calculator   = XRDCalculator(wavelength=wavelength)
    rng          = np.random.default_rng(seed)   # shared across all calls
    samples_each = n_samples // N_CLASSES
    remainder    = n_samples  % N_CLASSES

    X_list = []
    y_list = []

    print(f"[data_generator] Generating {n_samples:,} patterns "
          f"({samples_each}/class + {remainder} extra) …")

    for bl in tqdm(BRAVAIS_LATTICES, desc="Bravais class"):
        n = samples_each + (1 if bl["id"] < remainder else 0)
        for _ in range(n):
            pattern = simulate_pattern(bl, calculator, rng)
            X_list.append(pattern)
            y_list.append(bl["id"])

    X    = np.stack(X_list, axis=0)        # (N, 1024) float32
    y    = np.array(y_list, dtype=np.int32)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Strip .npz extension if caller added it (save_npz appends it)
    if output_path.endswith(".npz"):
        output_path = output_path[:-4]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_npz(X, y, output_path)

    print(f"[data_generator] Done ✓  X={X.shape}  y={y.shape}")
    return X, y


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic PXRD patterns for DeepBravais training "
            "using pymatgen XRDCalculator."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_samples", type=int,   default=50_000,
                        help="Total number of patterns to generate")
    parser.add_argument("--output",    type=str,
                        default="data/processed/dataset.npz",
                        help="Output .npz file path")
    parser.add_argument("--wavelength", type=float, default=WAVELENGTH_CU_KA,
                        help="X-ray wavelength in Å (default: Cu Kα 1.5406)")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Global random seed")

    args = parser.parse_args()
    generate_dataset(
        n_samples   = args.n_samples,
        output_path = args.output,
        wavelength  = args.wavelength,
        seed        = args.seed,
    )
