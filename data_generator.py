"""
data_generator.py
=================
Physics-Informed PXRD Data Generator for DeepBravais.

Strategy
--------
We do NOT scrape noisy experimental databases. Instead, we:
  1. Use pymatgen's XRDCalculator to obtain ideal peak positions (2θ) for
     synthetic crystal structures belonging to each of the 14 Bravais lattices.
  2. Convert those positions from 2θ-space to Q-space (momentum transfer), which
     gives linear peak spacing under lattice-constant changes — exactly what a
     1D-CNN's translation invariance is designed to exploit.
  3. Project the Q-peaks onto a fixed 1024-bin histogram.
  4. Broaden each peak with a Pseudo-Voigt profile (simulates finite crystallite
     size via the Scherrer equation).
  5. Add Poisson noise to mimic photon-counting statistics.

Q-space conversion
------------------
  Q = 4π sin(θ) / λ   [Å⁻¹]

where θ = (2θ)/2 and λ is the X-ray wavelength (Cu Kα, λ = 1.5406 Å).

Run this script directly to generate a dataset:
    python src/data_generator.py --n_samples 50000 --output data/processed/dataset.npz
"""

import argparse
import os
import random
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional pymatgen import guard — gives a clear message if not installed
# ---------------------------------------------------------------------------
try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("[WARNING] pymatgen not installed. "
          "Install with:  pip install pymatgen")

# ---------------------------------------------------------------------------
# Physical / instrument constants
# ---------------------------------------------------------------------------
WAVELENGTH_CU_KA = 1.5406          # Å  — Cu Kα radiation
TWO_THETA_MIN    = 5.0             # degrees  (detector lower limit)
TWO_THETA_MAX    = 90.0            # degrees
Q_MIN            = 0.7             # Å⁻¹  — corresponds to ~2θ ≈ 5° (Cu Kα)
Q_MAX            = 6.0             # Å⁻¹  — corresponds to ~2θ ≈ 90° (Cu Kα)
N_BINS           = 1024            # length of the output 1D pattern

# Scherrer broadening range (FWHM in Q-space, Å⁻¹)
# Typical crystallite sizes 20–200 nm translate to very narrow peaks;
# we deliberately widen the range for augmentation diversity.
FWHM_Q_MIN       = 0.02            # Å⁻¹ — large crystallites, sharp peaks
FWHM_Q_MAX       = 0.20            # Å⁻¹ — small crystallites, broad peaks

# Poisson noise scale — controls photon-count level
POISSON_SCALE_MIN = 500            # low-flux measurement
POISSON_SCALE_MAX = 5000           # high-flux measurement

# Lattice constant sampling range (Å)
A_MIN, A_MAX = 3.0, 8.0

# ---------------------------------------------------------------------------
# 14 Bravais Lattice definitions
# ---------------------------------------------------------------------------
# Each entry is a dict with:
#   'name'    : human-readable label
#   'crystal_system' : used for building the pymatgen Lattice
#   'centering'      : P, I, F, A, B, C, R  (for structure factor rules)
#
# We pair each Bravais lattice with a placeholder structure using a single
# atom at the origin; the PXRD pattern is dominated by the lattice geometry
# and systematic absences from the centering, not the atomic form factor of
# that single atom.

BRAVAIS_LATTICES: List[Dict] = [
    # ── Triclinic ────────────────────────────────────────────────────────────
    {"id": 0,  "name": "aP",  "system": "triclinic",    "centering": "P"},
    # ── Monoclinic ───────────────────────────────────────────────────────────
    {"id": 1,  "name": "mP",  "system": "monoclinic",   "centering": "P"},
    {"id": 2,  "name": "mC",  "system": "monoclinic",   "centering": "C"},
    # ── Orthorhombic ─────────────────────────────────────────────────────────
    {"id": 3,  "name": "oP",  "system": "orthorhombic", "centering": "P"},
    {"id": 4,  "name": "oI",  "system": "orthorhombic", "centering": "I"},
    {"id": 5,  "name": "oF",  "system": "orthorhombic", "centering": "F"},
    {"id": 6,  "name": "oC",  "system": "orthorhombic", "centering": "C"},
    # ── Tetragonal ───────────────────────────────────────────────────────────
    {"id": 7,  "name": "tP",  "system": "tetragonal",   "centering": "P"},
    {"id": 8,  "name": "tI",  "system": "tetragonal",   "centering": "I"},
    # ── Rhombohedral ─────────────────────────────────────────────────────────
    {"id": 9,  "name": "hR",  "system": "trigonal",     "centering": "R"},
    # ── Hexagonal ────────────────────────────────────────────────────────────
    {"id": 10, "name": "hP",  "system": "hexagonal",    "centering": "P"},
    # ── Cubic ────────────────────────────────────────────────────────────────
    {"id": 11, "name": "cP",  "system": "cubic",        "centering": "P"},
    {"id": 12, "name": "cI",  "system": "cubic",        "centering": "I"},
    {"id": 13, "name": "cF",  "system": "cubic",        "centering": "F"},
]

N_CLASSES    = len(BRAVAIS_LATTICES)          # 14
CLASS_NAMES  = [bl["name"] for bl in BRAVAIS_LATTICES]

# ---------------------------------------------------------------------------
# Space-group representatives for each Bravais lattice
# (lowest symmetric representative, chosen to maximise systematic absence
#  visibility without adding extra glide/screw conditions)
# ---------------------------------------------------------------------------
REPRESENTATIVE_SPACEGROUP: Dict[str, int] = {
    "aP": 2,    # P-1
    "mP": 3,    # P2
    "mC": 5,    # C2
    "oP": 16,   # P222
    "oI": 23,   # I222
    "oF": 22,   # F222
    "oC": 21,   # C222
    "tP": 75,   # P4
    "tI": 79,   # I4
    "hR": 146,  # R3
    "hP": 168,  # P6
    "cP": 195,  # P23
    "cI": 197,  # I23
    "cF": 196,  # F23
}


# ---------------------------------------------------------------------------
# Helper: build a random pymatgen Lattice for a given crystal system
# ---------------------------------------------------------------------------
def _random_lattice(system: str, centering: str) -> "Lattice":
    """
    Build a random pymatgen Lattice consistent with the crystal system's
    metric tensor constraints, sampling a, b, c in [A_MIN, A_MAX].
    """
    rng = np.random.default_rng()

    def rnd_a():
        return rng.uniform(A_MIN, A_MAX)

    def rnd_angle(lo=75.0, hi=105.0):
        return rng.uniform(lo, hi)

    if system == "cubic":
        a = rnd_a()
        return Lattice.cubic(a)

    elif system == "tetragonal":
        a = rnd_a()
        c = rnd_a()
        return Lattice.tetragonal(a, c)

    elif system == "orthorhombic":
        a, b, c = rnd_a(), rnd_a(), rnd_a()
        # Avoid degenerate near-tetragonal / near-cubic cases
        while abs(a - b) < 0.3:
            b = rnd_a()
        while abs(b - c) < 0.3:
            c = rnd_a()
        return Lattice.orthorhombic(a, b, c)

    elif system == "hexagonal":
        a = rnd_a()
        c = rnd_a()
        return Lattice(np.array([
            [a, 0, 0],
            [-a / 2, a * np.sqrt(3) / 2, 0],
            [0, 0, c]
        ]))

    elif system == "trigonal":
        # Rhombohedral setting: a=b=c, α=β=γ ≠ 90°
        a   = rnd_a()
        ang = rng.uniform(55.0, 75.0)   # rhombohedral angle
        return Lattice.from_parameters(a, a, a, ang, ang, ang)

    elif system == "monoclinic":
        a, b, c = rnd_a(), rnd_a(), rnd_a()
        beta = rnd_angle(90.0, 120.0)   # unique axis b; α=γ=90°
        return Lattice.monoclinic(a, b, c, beta)

    elif system == "triclinic":
        a, b, c = rnd_a(), rnd_a(), rnd_a()
        al = rnd_angle(70.0, 110.0)
        be = rnd_angle(70.0, 110.0)
        ga = rnd_angle(70.0, 110.0)
        return Lattice.from_parameters(a, b, c, al, be, ga)

    else:
        raise ValueError(f"Unknown crystal system: {system}")


# ---------------------------------------------------------------------------
# Helper: build a minimal Structure for XRDCalculator
# ---------------------------------------------------------------------------
def _make_structure(bravais: Dict) -> "Structure":
    """
    Place a single Cu atom at the origin of a random lattice matching the
    specified Bravais type.  The XRD pattern is then determined by
    (a) the metric tensor (d-spacings) and
    (b) the systematic absences from the centering.
    """
    lattice = _random_lattice(bravais["system"], bravais["centering"])
    # Single Cu atom at fractional coordinates (0, 0, 0)
    structure = Structure(lattice, ["Cu"], [[0, 0, 0]])
    return structure


# ---------------------------------------------------------------------------
# Q-space conversion utilities
# ---------------------------------------------------------------------------
def two_theta_to_Q(two_theta_deg: np.ndarray,
                   wavelength: float = WAVELENGTH_CU_KA) -> np.ndarray:
    """
    Convert 2θ angles (degrees) to momentum transfer Q (Å⁻¹).

    Q = 4π sin(θ) / λ
      = 4π sin(2θ/2) / λ
    """
    theta_rad = np.radians(two_theta_deg / 2.0)
    return (4.0 * np.pi * np.sin(theta_rad)) / wavelength


def Q_to_two_theta(Q: np.ndarray,
                   wavelength: float = WAVELENGTH_CU_KA) -> np.ndarray:
    """Inverse of two_theta_to_Q — for verification / visualisation."""
    sin_theta = Q * wavelength / (4.0 * np.pi)
    # Clip to valid range for arcsin
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    return 2.0 * np.degrees(np.arcsin(sin_theta))


# Q-axis bin centres (linearly spaced — the key advantage of Q-space)
Q_AXIS = np.linspace(Q_MIN, Q_MAX, N_BINS)   # shape (1024,)
DQ     = Q_AXIS[1] - Q_AXIS[0]               # bin width ≈ 0.00518 Å⁻¹


# ---------------------------------------------------------------------------
# Peak profile: Pseudo-Voigt
# ---------------------------------------------------------------------------
def pseudo_voigt(Q: np.ndarray,
                 Q0: float,
                 fwhm: float,
                 eta: float = 0.5) -> np.ndarray:
    """
    Pseudo-Voigt profile:
        PV(Q) = η · L(Q) + (1 - η) · G(Q)

    where L is a Lorentzian and G is a Gaussian, both normalised to unit
    area.  η=0 → pure Gaussian, η=1 → pure Lorentzian.

    Parameters
    ----------
    Q    : 1D array of Q values (the axis)
    Q0   : peak centre (Å⁻¹)
    fwhm : full-width at half maximum (Å⁻¹)
    eta  : Lorentzian mixing parameter ∈ [0, 1]
    """
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0

    # Gaussian component (unit area)
    gauss = np.exp(-0.5 * ((Q - Q0) / sigma) ** 2)
    gauss /= (sigma * np.sqrt(2.0 * np.pi))

    # Lorentzian component (unit area)
    lorentz = (1.0 / (np.pi * gamma)) / (1.0 + ((Q - Q0) / gamma) ** 2)

    return eta * lorentz + (1.0 - eta) * gauss


# ---------------------------------------------------------------------------
# Core: simulate one PXRD pattern
# ---------------------------------------------------------------------------
def simulate_pattern(bravais: Dict,
                     calculator: "XRDCalculator",
                     fwhm_q_range: Tuple[float, float] = (FWHM_Q_MIN, FWHM_Q_MAX),
                     poisson_scale_range: Tuple[float, float] = (POISSON_SCALE_MIN,
                                                                  POISSON_SCALE_MAX)
                     ) -> np.ndarray:
    """
    Generate one augmented PXRD pattern for a random instance of `bravais`.

    Returns
    -------
    pattern : np.ndarray, shape (N_BINS,), dtype float32
        Normalised diffraction pattern in Q-space [0, 1].
    """
    rng = np.random.default_rng()

    # ── 1. Build a random structure ────────────────────────────────────────
    structure = _make_structure(bravais)

    # ── 2. Calculate XRD peaks (2θ, intensity) ─────────────────────────────
    try:
        pattern = calculator.get_pattern(structure,
                                         two_theta_range=(TWO_THETA_MIN,
                                                          TWO_THETA_MAX))
    except Exception:
        # Fallback: return zeros if pymatgen fails (rare edge cases)
        return np.zeros(N_BINS, dtype=np.float32)

    two_theta_peaks = np.array(pattern.x)   # degrees
    intensities     = np.array(pattern.y)   # arbitrary units, max-normalised

    if len(two_theta_peaks) == 0:
        return np.zeros(N_BINS, dtype=np.float32)

    # ── 3. Convert 2θ → Q ──────────────────────────────────────────────────
    Q_peaks = two_theta_to_Q(two_theta_peaks)

    # Keep only peaks within our Q window
    mask     = (Q_peaks >= Q_MIN) & (Q_peaks <= Q_MAX)
    Q_peaks  = Q_peaks[mask]
    intensities = intensities[mask]

    if len(Q_peaks) == 0:
        return np.zeros(N_BINS, dtype=np.float32)

    # ── 4. Randomise broadening parameters ─────────────────────────────────
    fwhm_q = rng.uniform(*fwhm_q_range)
    eta    = rng.uniform(0.1, 0.9)          # Lorentzian/Gaussian mix

    # ── 5. Project peaks onto Q-axis with Pseudo-Voigt broadening ──────────
    signal = np.zeros(N_BINS, dtype=np.float64)
    for Q0, I0 in zip(Q_peaks, intensities):
        signal += I0 * pseudo_voigt(Q_AXIS, Q0, fwhm_q, eta)

    # ── 6. Add Poisson noise ────────────────────────────────────────────────
    # Scale to a photon-count level, apply Poisson noise, re-scale back
    scale         = rng.uniform(*poisson_scale_range)
    signal_scaled = signal * (scale / (signal.max() + 1e-8))
    signal_noisy  = rng.poisson(np.maximum(signal_scaled, 0)).astype(np.float64)

    # ── 7. Normalise to [0, 1] ─────────────────────────────────────────────
    max_val = signal_noisy.max()
    if max_val > 0:
        signal_noisy /= max_val

    return signal_noisy.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_dataset(n_samples: int = 50_000,
                     output_path: str = "data/processed/dataset.npz",
                     wavelength: float = WAVELENGTH_CU_KA,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate `n_samples` physics-informed PXRD patterns distributed evenly
    across the 14 Bravais lattices, and save them as a compressed .npz file.

    Parameters
    ----------
    n_samples   : total number of samples to generate
    output_path : where to save the .npz file
    wavelength  : X-ray wavelength in Å (default: Cu Kα = 1.5406 Å)
    seed        : global random seed for reproducibility

    Returns
    -------
    X : np.ndarray, shape (n_samples, N_BINS), float32
    y : np.ndarray, shape (n_samples,), int32
    """
    if not PYMATGEN_AVAILABLE:
        raise RuntimeError("pymatgen is required. Install with: pip install pymatgen")

    np.random.seed(seed)
    random.seed(seed)

    calculator   = XRDCalculator(wavelength=wavelength)
    samples_each = n_samples // N_CLASSES
    remainder    = n_samples  % N_CLASSES

    X_list: List[np.ndarray] = []
    y_list: List[int]        = []

    print(f"[DeepBravais] Generating {n_samples:,} patterns "
          f"({samples_each} per class + {remainder} extra) ...")

    for bl in tqdm(BRAVAIS_LATTICES, desc="Bravais class"):
        n = samples_each + (1 if bl["id"] < remainder else 0)
        for _ in range(n):
            pattern = simulate_pattern(bl, calculator)
            X_list.append(pattern)
            y_list.append(bl["id"])

    X = np.stack(X_list, axis=0)      # (n_samples, 1024)
    y = np.array(y_list, dtype=np.int32)

    # Shuffle dataset
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y,
                        class_names=np.array(CLASS_NAMES))
    print(f"[DeepBravais] Saved {len(X):,} samples → {output_path}")
    print(f"              X shape : {X.shape}  dtype : {X.dtype}")
    print(f"              y shape : {y.shape}  classes : {N_CLASSES}")

    return X, y


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate physics-informed PXRD dataset for DeepBravais."
    )
    parser.add_argument("--n_samples",  type=int,   default=50_000,
                        help="Total number of synthetic patterns (default: 50000)")
    parser.add_argument("--output",     type=str,
                        default="data/processed/dataset.npz",
                        help="Output .npz file path")
    parser.add_argument("--wavelength", type=float, default=WAVELENGTH_CU_KA,
                        help=f"X-ray wavelength in Å (default: {WAVELENGTH_CU_KA})")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.n_samples,
        output_path=args.output,
        wavelength=args.wavelength,
        seed=args.seed,
    )
