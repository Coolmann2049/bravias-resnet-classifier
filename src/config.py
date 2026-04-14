"""
src/config.py
=============
Single source of truth for every constant, hyperparameter, and
dataset definition used across the DeepBravais project.

Importing convention
--------------------
    from src.config import Q_MIN, Q_MAX, N_BINS, DEFAULT_LR, ...
"""
from typing import Dict, List

# ===========================================================================
# Physical / instrument constants
# ===========================================================================
WAVELENGTH_CU_KA  = 1.5406        # Å  — Cu Kα radiation
TWO_THETA_MIN     = 5.0           # degrees  (detector lower limit)
TWO_THETA_MAX     = 90.0          # degrees
Q_MIN             = 0.7           # Å⁻¹  (~2θ ≈ 5°,  Cu Kα)
Q_MAX             = 6.0           # Å⁻¹  (~2θ ≈ 90°, Cu Kα)
N_BINS            = 1024          # Q-axis bins

# Scherrer broadening range (FWHM in Å⁻¹)
FWHM_Q_MIN        = 0.02
FWHM_Q_MAX        = 0.20

# Poisson photon-count scale
POISSON_SCALE_MIN = 500
POISSON_SCALE_MAX = 5000

# Lattice constant sampling range (Å)
A_MIN, A_MAX      = 3.0, 8.0


# ===========================================================================
# 14 Bravais Lattice definitions
# ===========================================================================
N_CLASSES: int = 14

CLASS_NAMES: List[str] = [
    "aP",   #  0  Triclinic     P
    "mP",   #  1  Monoclinic    P
    "mC",   #  2  Monoclinic    C
    "oP",   #  3  Orthorhombic  P
    "oI",   #  4  Orthorhombic  I
    "oF",   #  5  Orthorhombic  F
    "oC",   #  6  Orthorhombic  C
    "tP",   #  7  Tetragonal    P
    "tI",   #  8  Tetragonal    I
    "hR",   #  9  Trigonal      R
    "hP",   # 10  Hexagonal     P
    "cP",   # 11  Cubic         P
    "cI",   # 12  Cubic         I
    "cF",   # 13  Cubic         F
]

BRAVAIS_LATTICES: List[Dict] = [
    {"id":  0, "name": "aP", "system": "triclinic",    "centering": "P"},
    {"id":  1, "name": "mP", "system": "monoclinic",   "centering": "P"},
    {"id":  2, "name": "mC", "system": "monoclinic",   "centering": "C"},
    {"id":  3, "name": "oP", "system": "orthorhombic", "centering": "P"},
    {"id":  4, "name": "oI", "system": "orthorhombic", "centering": "I"},
    {"id":  5, "name": "oF", "system": "orthorhombic", "centering": "F"},
    {"id":  6, "name": "oC", "system": "orthorhombic", "centering": "C"},
    {"id":  7, "name": "tP", "system": "tetragonal",   "centering": "P"},
    {"id":  8, "name": "tI", "system": "tetragonal",   "centering": "I"},
    {"id":  9, "name": "hR", "system": "trigonal",     "centering": "R"},
    {"id": 10, "name": "hP", "system": "hexagonal",    "centering": "P"},
    {"id": 11, "name": "cP", "system": "cubic",        "centering": "P"},
    {"id": 12, "name": "cI", "system": "cubic",        "centering": "I"},
    {"id": 13, "name": "cF", "system": "cubic",        "centering": "F"},
]

# Lowest-symmetry space group representative for each Bravais type.
# Chosen to maximise systematic-absence visibility without adding
# extra glide/screw extinction conditions.
REPRESENTATIVE_SPACEGROUP: Dict[str, int] = {
    "aP":  2,   # P-1
    "mP":  3,   # P2
    "mC":  5,   # C2
    "oP": 16,   # P222
    "oI": 23,   # I222
    "oF": 22,   # F222
    "oC": 20,   # C222
    "tP": 75,   # P4
    "tI": 79,   # I4
    "hR":146,   # R3
    "hP":143,   # P3
    "cP":195,   # P23
    "cI":197,   # I23
    "cF":196,   # F23
}


# ===========================================================================
# Training defaults  (overridable via CLI)
# ===========================================================================
DEFAULT_DATA_PATH  = "data/processed/dataset.npz"
DEFAULT_CKPT_DIR   = "outputs/checkpoints"
DEFAULT_PLOT_DIR   = "outputs/plots"
DEFAULT_EPOCHS     = 50
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR         = 1e-3
DEFAULT_DROPOUT    = 0.0    # ConvNeXt uses stochastic depth, not head dropout
DEFAULT_MODEL_TYPE = "full" # "full" | "small"
VAL_RATIO          = 0.15
TEST_RATIO         = 0.15
RANDOM_STATE       = 42
