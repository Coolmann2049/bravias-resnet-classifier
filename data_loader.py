"""
data_loader.py
==============
CLI entry point — downloads SimXRD-4M shards from Hugging Face (AI4Spectro),
applies the SG→Bravais mapping, and saves a balanced .npz / .h5 dataset
ready for train.py.

All heavy logic lives in src/ submodules:
  - src/config.py    — CLASS_NAMES, N_CLASSES
  - src/dataset.py   — SG_TO_BRAVAIS, preprocess_pattern,
                       collect_balanced_samples, save_npz, save_h5

Dataset references (updated from caobin/SimXRD which is now 404)
-----------------------------------------------------------------
  https://huggingface.co/AI4Spectro
   - Training Part 1 : AI4Spectro/ILtrainV1_P1  (74.1 GB, 5 split-gz parts)
   - Training Part 2 : AI4Spectro/ILtrainV1_P2
   - Validation      : AI4Spectro/ILvalV1
   - Test            : AI4Spectro/ILtestV1

⚠  Kaggle disk warning
  Each training part is ~74 GB compressed. Kaggle notebooks only have
  ~20 GB disk, so full download is NOT possible on Kaggle.
  On Kaggle → use data_generator.py with --n_samples 500000 instead.

Usage
-----
    # Default: 5000 samples per class (70k total), auto-download
    python data_loader.py

    # Custom
    python data_loader.py --n_per_class 10000 --max_shards 8

    # Use already-downloaded .db files (no re-download)
    python data_loader.py --no_download --db_dir /path/to/shards
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config  import N_CLASSES, CLASS_NAMES
from src.dataset import collect_balanced_samples, save_npz, save_h5

# Optional HF hub import
try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    warnings.warn("huggingface_hub not installed. Run: pip install huggingface_hub",
                  ImportWarning)


# ===========================================================================
# Hugging Face helpers
# ===========================================================================

HF_REPO_ID   = "AI4Spectro/ILtrainV1_P1"   # Part 1 of training data
HF_REPO_ID_P2 = "AI4Spectro/ILtrainV1_P2"  # Part 2
HF_REPO_TYPE = "dataset"


def _list_db_files(repo_id: str = HF_REPO_ID) -> List[str]:
    if not HF_HUB_AVAILABLE:
        raise RuntimeError("huggingface_hub not installed.")
    return sorted(f for f in list_repo_files(repo_id, repo_type=HF_REPO_TYPE)
                  if f.endswith(".db") or f.endswith(".db.gz"))


def _download_shard(repo_file: str, local_dir: str,
                    repo_id: str = HF_REPO_ID) -> str:
    if not HF_HUB_AVAILABLE:
        raise RuntimeError("huggingface_hub not installed.")
    return hf_hub_download(
        repo_id   = repo_id,
        filename  = repo_file,
        repo_type = HF_REPO_TYPE,
        local_dir = local_dir,
    )


# ===========================================================================
# Main pipeline
# ===========================================================================

def run(n_per_class:  int  = 5_000,
        output:       str  = "data/processed/simxrd_balanced",
        fmt:          str  = "npz",
        db_dir:       str  = "data/simxrd_shards",
        no_download:  bool = False,
        max_shards:   int  = -1) -> None:
    """
    Full pipeline:
      1. (Optional) download .db shards from Hugging Face AI4Spectro.
      2. Stream shards → collect balanced Bravais samples.
      3. Save to disk as .npz or .h5.
    """
    print("=" * 60)
    print("  DeepBravais — SimXRD-4M Data Loader")
    print("=" * 60)
    print(f"  Target : {n_per_class} samples × {N_CLASSES} classes "
          f"= {n_per_class * N_CLASSES:,} total")
    print(f"  Format : {fmt.upper()}")
    print(f"  Output : {output}.{fmt}")
    print("=" * 60 + "\n")

    os.makedirs(db_dir, exist_ok=True)
    db_paths: List[str] = []

    # ── Step 1: discover / download shards ─────────────────────────────
    if no_download:
        db_paths = sorted(str(p) for p in Path(db_dir).rglob("*.db"))
        if not db_paths:
            sys.exit(f"[data_loader] No .db files found in {db_dir!r}. "
                     "Remove --no_download or pre-populate the directory.")
        print(f"[data_loader] Found {len(db_paths)} local shard(s).")
    else:
        print("[data_loader] Querying Hugging Face for shard list …")
        remote_files = _list_db_files()
        if max_shards > 0:
            remote_files = remote_files[:max_shards]
        print(f"[data_loader] Found {len(remote_files)} shard(s). "
              "Downloading …\n")
        for rf in tqdm(remote_files, desc="Downloading shards", unit="shard"):
            db_paths.append(_download_shard(rf, local_dir=db_dir))
        print(f"\n[data_loader] {len(db_paths)} shard(s) ready in {db_dir!r}.\n")

    if not db_paths:
        sys.exit("[data_loader] No .db files available. Aborting.")

    # ── Step 2: collect balanced samples ───────────────────────────────
    print("[data_loader] Streaming shards for stratified sampling …\n")
    X, y = collect_balanced_samples(db_paths, n_per_class=n_per_class)
    print(f"\n[data_loader] Final dataset: X={X.shape}  y={y.shape}")

    # ── Step 3: save ───────────────────────────────────────────────────
    if fmt == "h5":
        save_h5(X, y, output + ".h5")
    else:
        save_npz(X, y, output)

    print("\n[data_loader] Done ✓")
    print(f"  Load with:  data = np.load('{output}.{fmt}')")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download the SimXRD-4M dataset from Hugging Face (AI4Spectro) "
            "and produce a balanced PXRD dataset for DeepBravais training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_per_class", type=int,  default=5_000,
                        help="Samples per Bravais class")
    parser.add_argument("--output",      type=str,
                        default="data/processed/simxrd_balanced",
                        help="Output path (without extension)")
    parser.add_argument("--format",      dest="fmt",
                        choices=["npz", "h5"], default="npz")
    parser.add_argument("--db_dir",      type=str,
                        default="data/simxrd_shards",
                        help="Local directory for .db shard cache")
    parser.add_argument("--no_download", action="store_true",
                        help="Use already-downloaded files in --db_dir")
    parser.add_argument("--max_shards",  type=int,  default=-1,
                        help="Max shards to download (-1 = all)")
    parser.add_argument("--seed",        type=int,  default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)

    run(
        n_per_class = args.n_per_class,
        output      = args.output,
        fmt         = args.fmt,
        db_dir      = args.db_dir,
        no_download = args.no_download,
        max_shards  = args.max_shards,
    )
