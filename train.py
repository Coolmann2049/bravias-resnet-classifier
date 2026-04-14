"""
train.py
========
CLI entry point for DeepBravais training.

All logic lives in the src/ package:
  - src/config.py        — constants & defaults
  - src/dataset.py       — load_npz, split_dataset
  - src/models.py        — build_resnet1d, build_resnet1d_small
  - src/trainer.py       — compile_model, build_callbacks
  - src/visualization.py — plot_training_curves, plot_confusion_matrix

Usage
-----
    # Default run
    python train.py

    # Custom
    python train.py --data data/processed/dataset.npz \
                    --epochs 120 --batch 512 --lr 3e-4 --model full
"""

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras

# Ensure project root is on the path (needed in Kaggle / Colab)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    DEFAULT_DATA_PATH, DEFAULT_CKPT_DIR, DEFAULT_PLOT_DIR,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR,
    DEFAULT_DROPOUT, DEFAULT_MODEL_TYPE,
    N_CLASSES, CLASS_NAMES,
)
from src.dataset      import load_npz, split_dataset
from src.models       import build_resnet1d, build_resnet1d_small
from src.trainer      import compile_model, build_callbacks
from src.visualization import plot_training_curves, plot_confusion_matrix


# ===========================================================================
# Core training function
# ===========================================================================

def train(data_path:     str   = DEFAULT_DATA_PATH,
          ckpt_dir:      str   = DEFAULT_CKPT_DIR,
          plot_dir:      str   = DEFAULT_PLOT_DIR,
          epochs:        int   = DEFAULT_EPOCHS,
          batch_size:    int   = DEFAULT_BATCH_SIZE,
          learning_rate: float = DEFAULT_LR,
          dropout:       float = DEFAULT_DROPOUT,
          model_type:    str   = DEFAULT_MODEL_TYPE) -> None:
    """End-to-end training routine."""

    # ── GPU / mixed-precision setup ──────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[DeepBravais] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("[DeepBravais] Mixed precision (float16) enabled.")
    else:
        print("[DeepBravais] No GPU detected — training on CPU.")

    # ── 1. Load & split data ─────────────────────────────────────────────
    X, y = load_npz(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print(f"[DeepBravais] Split → train: {len(X_train):,}  "
          f"val: {len(X_val):,}  test: {len(X_test):,}")

    # ── 2. Build model ───────────────────────────────────────────────────
    builder = build_resnet1d_small if model_type == "small" else build_resnet1d
    model   = builder(dropout_rate=dropout)

    model = compile_model(
        model,
        learning_rate   = learning_rate,
        warmup_epochs   = 5,
        total_epochs    = epochs,
        steps_per_epoch = max(len(X_train) // batch_size, 1),
    )
    model.summary(line_length=100)
    print(f"[DeepBravais] Total parameters: {model.count_params():,}")

    # ── 3. Callbacks ─────────────────────────────────────────────────────
    callbacks, _ = build_callbacks(ckpt_dir)

    # ── 4. Class weights (handle minor class imbalance) ──────────────────
    n_samples    = len(y_train)
    counts       = np.bincount(y_train, minlength=N_CLASSES)
    class_weight = {i: n_samples / (N_CLASSES * max(c, 1))
                    for i, c in enumerate(counts)}

    # ── 5. Train ─────────────────────────────────────────────────────────
    print(f"\n[DeepBravais] Starting training  "
          f"epochs={epochs}  batch={batch_size}  lr={learning_rate}")
    print("=" * 70)

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = epochs,
        batch_size      = batch_size,
        class_weight    = class_weight,
        callbacks       = callbacks,
        verbose         = 1,
    )

    # ── 6. Evaluate ──────────────────────────────────────────────────────
    print("\n[DeepBravais] Evaluating on test set …")
    test_loss, test_acc = model.evaluate(X_test, y_test,
                                         batch_size=batch_size, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")

    y_pred = np.argmax(
        model.predict(X_test, batch_size=batch_size, verbose=0),
        axis=1,
    )
    print("\n[DeepBravais] Per-class report:")
    print(classification_report(y_test, y_pred,
                                 target_names=CLASS_NAMES, digits=4))

    # ── 7. Save plots & model ────────────────────────────────────────────
    plot_training_curves(history, save_dir=plot_dir)
    plot_confusion_matrix(y_test, y_pred,
                          class_names=CLASS_NAMES, save_dir=plot_dir)

    os.makedirs("outputs", exist_ok=True)
    save_path = os.path.join("outputs", "deepbravais_final.keras")
    model.save(save_path)
    print(f"[DeepBravais] Final model saved → {save_path}")
    print("[DeepBravais] Training complete. ✓")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DeepBravais ConvNeXt-1D on synthetic PXRD patterns."
    )
    parser.add_argument("--data",    type=str,   default=DEFAULT_DATA_PATH,
                        help="Path to the .npz dataset")
    parser.add_argument("--epochs",  type=int,   default=DEFAULT_EPOCHS,
                        help="Training epochs (default: 50)")
    parser.add_argument("--batch",   type=int,   default=DEFAULT_BATCH_SIZE,
                        help="Batch size (default: 256)")
    parser.add_argument("--lr",      type=float, default=DEFAULT_LR,
                        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT,
                        help="Head dropout rate (default: 0.0)")
    parser.add_argument("--model",   type=str,   default=DEFAULT_MODEL_TYPE,
                        choices=["full", "small"],
                        help="Model variant: 'full' (~7M) or 'small' (~1.5M)")
    parser.add_argument("--ckpt_dir", type=str,  default=DEFAULT_CKPT_DIR)
    parser.add_argument("--plot_dir", type=str,  default=DEFAULT_PLOT_DIR)

    args = parser.parse_args()
    train(
        data_path     = args.data,
        ckpt_dir      = args.ckpt_dir,
        plot_dir      = args.plot_dir,
        epochs        = args.epochs,
        batch_size    = args.batch,
        learning_rate = args.lr,
        dropout       = args.dropout,
        model_type    = args.model,
    )
