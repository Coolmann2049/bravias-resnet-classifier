"""
src/dataset.py
==============
Data-handling utilities shared between the pymatgen and SimXRD pipelines.

Responsibilities
----------------
- Mapping 230 space groups → 14 Bravais labels  (SG_TO_BRAVAIS)
- d-spacing → Q conversion for SimXRD patterns   (d_to_Q)
- Pattern pre-processing onto the shared Q grid   (preprocess_pattern)
- Stratified sampling from SimXRD .db shards      (collect_balanced_samples)
- Train / val / test splitting                     (split_dataset)
- Loading .npz datasets                           (load_npz)
- Saving datasets as .npz or .h5                  (save_npz, save_h5)
"""
from __future__ import annotations

import collections
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    N_BINS, N_CLASSES, CLASS_NAMES,
    Q_MIN, Q_MAX,
    VAL_RATIO, TEST_RATIO, RANDOM_STATE,
)

# Fixed Q-axis (shared with src/physics.py but kept local to avoid a circular import)
Q_AXIS: np.ndarray = np.linspace(Q_MIN, Q_MAX, N_BINS, dtype=np.float32)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    from ase.db import connect as ase_connect
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


# ===========================================================================
# SECTION 1 — Space-Group → Bravais-Lattice look-up table
# ===========================================================================

def _build_sg_to_bravais() -> Dict[int, int]:
    """
    Return {space_group_number (1-230): bravais_label (0-13)}.

    Source: International Tables for Crystallography, Vol. A.
    All 230 space groups are assigned; an assertion guards completeness.
    """
    sg2b: Dict[int, int] = {}

    # Triclinic (1-2): always aP
    for sg in range(1, 3):
        sg2b[sg] = 0

    # Monoclinic (3-15): P or C
    mono_C = {5, 8, 9, 12, 15}
    for sg in range(3, 16):
        sg2b[sg] = 2 if sg in mono_C else 1

    # Orthorhombic (16-74): F / I / C / P
    ortho_F = {22, 42, 43, 69, 70}
    ortho_I = {23, 24, 44, 45, 46, 71, 72, 73, 74}
    ortho_C = {20, 21, 35, 36, 37, 38, 39, 40, 41, 63, 64, 65, 66, 67, 68}
    for sg in range(16, 75):
        if   sg in ortho_F: sg2b[sg] = 5
        elif sg in ortho_I: sg2b[sg] = 4
        elif sg in ortho_C: sg2b[sg] = 6
        else:               sg2b[sg] = 3

    # Tetragonal (75-142): I or P
    tetra_I = {79, 80, 82, 87, 88, 97, 98, 107, 108, 109, 110,
               119, 120, 121, 122, 139, 140, 141, 142}
    for sg in range(75, 143):
        sg2b[sg] = 8 if sg in tetra_I else 7

    # Trigonal (143-167): R or hP
    trig_R = {146, 148, 155, 160, 161, 166, 167}
    for sg in range(143, 168):
        sg2b[sg] = 9 if sg in trig_R else 10

    # Hexagonal (168-194): always hP
    for sg in range(168, 195):
        sg2b[sg] = 10

    # Cubic (195-230): F / I / P
    cubic_F = {196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228}
    cubic_I = {197, 199, 204, 206, 211, 214, 217, 220, 229, 230}
    for sg in range(195, 231):
        if   sg in cubic_F: sg2b[sg] = 13
        elif sg in cubic_I: sg2b[sg] = 12
        else:               sg2b[sg] = 11

    assert len(sg2b) == 230, f"BUG: {len(sg2b)} SGs assigned, expected 230"
    return sg2b


SG_TO_BRAVAIS: Dict[int, int] = _build_sg_to_bravais()


# ===========================================================================
# SECTION 2 — Pattern pre-processing (SimXRD d-I → Q-space histogram)
# ===========================================================================

def d_to_Q(d_array: np.ndarray) -> np.ndarray:
    """Convert d-spacing (Å) to momentum transfer Q (Å⁻¹):  Q = 2π / d."""
    d_safe = np.where(d_array > 0, d_array, np.inf)
    return (2.0 * np.pi) / d_safe


def preprocess_pattern(d_spacings: List[float],
                        intensities: List[float]) -> Optional[np.ndarray]:
    """
    Convert a SimXRD d-I peak list to a 1024-bin Q-space histogram
    normalised to [0, 1].

    Returns None if the pattern is empty or all-zero after processing.
    """
    d_arr = np.asarray(d_spacings,  dtype=np.float64)
    I_arr = np.asarray(intensities, dtype=np.float64)

    if len(d_arr) == 0 or len(I_arr) == 0:
        return None

    Q_vals = d_to_Q(d_arr)
    mask   = (Q_vals >= Q_MIN) & (Q_vals <= Q_MAX) & np.isfinite(Q_vals)
    Q_vals = Q_vals[mask]
    I_arr  = I_arr[mask]

    if len(Q_vals) == 0:
        return None

    bin_idx = np.searchsorted(Q_AXIS, Q_vals).clip(0, N_BINS - 1)
    signal  = np.zeros(N_BINS, dtype=np.float32)
    np.add.at(signal, bin_idx, I_arr.astype(np.float32))

    max_val = signal.max()
    if max_val <= 0:
        return None
    signal /= max_val
    return signal


# ===========================================================================
# SECTION 3 — SimXRD stratified sampling
# ===========================================================================

def collect_balanced_samples(
    db_paths:    List[str],
    n_per_class: int  = 5_000,
    verbose:     bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stream SimXRD ASE .db shards and collect ``n_per_class`` samples
    for each of the 14 Bravais lattices.

    Parameters
    ----------
    db_paths    : local paths to .db shard files
    n_per_class : target samples per class
    verbose     : print per-class report at the end

    Returns
    -------
    X : (N, 1024) float32 — normalised Q-space patterns
    y : (N,)      int32   — Bravais label 0-13
    """
    if not ASE_AVAILABLE:
        raise RuntimeError("ase not installed. Run: pip install ase")

    from tqdm import tqdm

    buckets: Dict[int, List[np.ndarray]] = {i: [] for i in range(N_CLASSES)}
    counts   = collections.Counter({i: 0 for i in range(N_CLASSES)})
    needed   = n_per_class * N_CLASSES
    collected = 0

    outer = tqdm(db_paths, desc="Shards", unit="shard")
    for db_path in outer:
        if collected >= needed:
            break
        if not os.path.isfile(str(db_path)):
            warnings.warn(f"Shard not found: {db_path}")
            continue
        try:
            db = ase_connect(str(db_path))
        except Exception as exc:
            warnings.warn(f"Cannot open {db_path}: {exc}")
            continue

        n_rows = None
        try:
            n_rows = db.count()
        except Exception:
            pass

        inner = tqdm(db.select(), total=n_rows,
                     desc=f"  └─ {Path(db_path).name}",
                     unit="row", leave=False)
        for row in inner:
            if collected >= needed:
                break
            try:
                tager = row.tager
                if isinstance(tager, str):
                    tager = eval(tager)
                sg = int(tager[0])
            except Exception:
                continue

            bv = SG_TO_BRAVAIS.get(sg)
            if bv is None or counts[bv] >= n_per_class:
                continue

            try:
                ld = row.latt_dis
                it = row.intensity
                if isinstance(ld, str): ld = eval(ld)
                if isinstance(it, str): it = eval(it)
            except Exception:
                continue

            pat = preprocess_pattern(ld, it)
            if pat is None:
                continue

            buckets[bv].append(pat)
            counts[bv] += 1
            collected  += 1
            inner.set_postfix(collected=collected, need=needed)

        inner.close()
        outer.set_postfix(
            collected=collected,
            classes_done=sum(1 for v in counts.values() if v >= n_per_class),
        )

    if verbose:
        print("\n" + "─" * 60)
        print(f"{'Bravais':>10}  {'Label':>5}  {'Collected':>10}  {'Target':>8}")
        print("─" * 60)
        incomplete = []
        for bv in range(N_CLASSES):
            n    = counts[bv]
            flag = "✓" if n >= n_per_class else "✗ INCOMPLETE"
            print(f"{CLASS_NAMES[bv]:>10}  {bv:>5}  {n:>10}  {n_per_class:>8}  {flag}")
            if n < n_per_class:
                incomplete.append(CLASS_NAMES[bv])
        print("─" * 60)
        if incomplete:
            warnings.warn(
                f"Incomplete classes: {incomplete}. "
                "Increase --max_shards or reduce --n_per_class.",
                RuntimeWarning,
            )

    X_list, y_list = [], []
    for bv in range(N_CLASSES):
        samps = buckets[bv][:n_per_class]
        X_list.extend(samps)
        y_list.extend([bv] * len(samps))

    X   = np.stack(X_list, axis=0).astype(np.float32)
    y   = np.array(y_list,       dtype=np.int32)
    perm = np.random.default_rng(seed=RANDOM_STATE).permutation(len(X))
    return X[perm], y[perm]


# ===========================================================================
# SECTION 4 — Train / val / test split
# ===========================================================================

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio:  float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed:       int   = RANDOM_STATE,
) -> Tuple[np.ndarray, ...]:
    """
    Stratified split into (X_train, X_val, X_test, y_train, y_val, y_test).

    Parameters
    ----------
    X, y       : full dataset arrays
    val_ratio  : fraction for validation (default 0.15)
    test_ratio : fraction for test       (default 0.15)
    seed       : random state for reproducibility

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    tmp_ratio  = val_ratio + test_ratio
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size    = tmp_ratio,
        stratify     = y,
        random_state = seed,
    )
    rel_test = test_ratio / tmp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size    = rel_test,
        stratify     = y_tmp,
        random_state = seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ===========================================================================
# SECTION 5 — Disk I/O
# ===========================================================================

def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an .npz dataset produced by data_generator.py or data_loader.py.

    Returns
    -------
    X : (N, N_BINS) float32
    y : (N,)        int32
    """
    data = np.load(path, allow_pickle=False)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int32)

    # Clip & per-sample normalise (guard against rare negative artefacts)
    X = np.clip(X, 0.0, None)
    row_max = X.max(axis=1, keepdims=True) + 1e-8
    X = X / row_max

    # Add channel dimension if needed
    if X.ndim == 2:
        X = X[:, :, np.newaxis]           # (N, 1024, 1)

    print(f"[dataset] Loaded {path}  →  X={X.shape}  y={y.shape}")
    return X, y


def save_npz(X: np.ndarray, y: np.ndarray, path: str) -> None:
    """Save dataset as a compressed NumPy .npz file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        X           = X,
        y           = y,
        class_names = np.array(CLASS_NAMES),
        Q_axis      = Q_AXIS,
        source      = np.array("DeepBravais"),
    )
    size_mb = os.path.getsize(path + ".npz") / 1e6
    print(f"[dataset] Saved → {path}.npz  ({size_mb:.1f} MB)")


def save_h5(X: np.ndarray, y: np.ndarray, path: str) -> None:
    """Save dataset as HDF5 (faster random access, useful for very large sets)."""
    if not H5PY_AVAILABLE:
        raise RuntimeError("h5py not installed. Run: pip install h5py")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("X",           data=X,                   compression="gzip")
        f.create_dataset("y",           data=y,                   compression="gzip")
        f.create_dataset("Q_axis",      data=Q_AXIS)
        f.create_dataset("class_names", data=np.array(CLASS_NAMES, dtype="S"))
        f.attrs["n_bins"] = N_BINS
        f.attrs["Q_min"]  = Q_MIN
        f.attrs["Q_max"]  = Q_MAX
    size_mb = os.path.getsize(path) / 1e6
    print(f"[dataset] Saved → {path}  ({size_mb:.1f} MB)")
