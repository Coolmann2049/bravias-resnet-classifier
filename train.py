"""
train.py
========
Training pipeline for DeepBravais — 1D-ResNet PXRD Bravais Lattice Classifier.

Pipeline
--------
  1. Load pre-generated dataset from   data/processed/dataset.npz
  2. Normalise + reshape inputs        (N, 1024) → (N, 1024, 1)
  3. Stratified train/val/test split   70 / 15 / 15
  4. Build & compile the ResNet-1D model
  5. Train with:
       • Cosine-annealing LR schedule
       • ModelCheckpoint  — saves best weights (val_accuracy)
       • EarlyStopping    — patience 10 epochs
       • ReduceLROnPlateau — halves LR if val_loss stalls for 5 epochs
  6. Evaluate on held-out test set; print classification report
  7. Save training curves and confusion matrix to  outputs/

Usage
-----
    # From the project root:
    python src/train.py

    # With custom settings:
    python src/train.py --data    data/processed/dataset.npz \
                        --epochs  100 \
                        --batch   256 \
                        --model   full   # or 'small'
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend — safe on Kaggle/Colab/servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix)

import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Make the src/ directory importable when running from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import build_resnet1d, build_resnet1d_small, N_CLASSES, N_BINS

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH   = "data/processed/dataset.npz"
DEFAULT_CKPT_DIR    = "outputs/checkpoints"
DEFAULT_PLOT_DIR    = "outputs/plots"
DEFAULT_EPOCHS      = 50
DEFAULT_BATCH_SIZE  = 256
DEFAULT_LR          = 1e-3
DEFAULT_DROPOUT     = 0.4
DEFAULT_MODEL_TYPE  = "full"        # "full" or "small"
VAL_RATIO           = 0.15
TEST_RATIO          = 0.15
RANDOM_STATE        = 42

# Class names for reporting
CLASS_NAMES = [
    "aP", "mP", "mC",
    "oP", "oI", "oF", "oC",
    "tP", "tI",
    "hR", "hP",
    "cP", "cI", "cF",
]


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_data(npz_path: str):
    """
    Load dataset from .npz, normalise to [0,1], and reshape for Conv1D.

    Returns
    -------
    X : np.ndarray  shape (N, N_BINS, 1)   float32
    y : np.ndarray  shape (N,)             int32
    class_names : list[str]
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Dataset not found at '{npz_path}'.\n"
            "Generate it first with:\n"
            "    python src/data_generator.py"
        )

    print(f"[DeepBravais] Loading dataset from {npz_path} …")
    data = np.load(npz_path, allow_pickle=True)
    X    = data["X"].astype(np.float32)   # (N, 1024)
    y    = data["y"].astype(np.int32)     # (N,)
    cnames = (data["class_names"].tolist()
              if "class_names" in data else CLASS_NAMES)

    # Each pattern is already normalised to [0,1] by data_generator.py.
    # As a safety net, clip and re-normalise per-sample.
    X = np.clip(X, 0.0, None)
    row_max = X.max(axis=1, keepdims=True) + 1e-8
    X = X / row_max

    # Add channel dimension for Conv1D: (N, 1024) → (N, 1024, 1)
    X = X[:, :, np.newaxis]

    print(f"           X : {X.shape}  {X.dtype}  range [{X.min():.3f}, {X.max():.3f}]")
    print(f"           y : {y.shape}  classes = {len(np.unique(y))}")
    return X, y, cnames


def split_data(X, y, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
               random_state=RANDOM_STATE):
    """
    Stratified split into train / validation / test sets.
    """
    # First split: train + (val + test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=val_ratio + test_ratio,
        stratify=y,
        random_state=random_state
    )
    # Second split: val and test
    relative_test = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_test,
        stratify=y_tmp,
        random_state=random_state
    )
    print(f"[DeepBravais] Split → train: {len(X_train):,}  "
          f"val: {len(X_val):,}  test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Model compilation
# ---------------------------------------------------------------------------
def compile_model(model: keras.Model, learning_rate: float = DEFAULT_LR):
    """
    Compile with Adam optimiser, sparse categorical cross-entropy, and
    standard accuracy metric.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate,
                                        clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def build_callbacks(ckpt_dir: str, patience_es: int = 10,
                    patience_rlr: int = 5) -> list:
    """
    Returns a list of Keras callbacks:
      • ModelCheckpoint  — saves best model weights by val_accuracy
      • EarlyStopping    — stops when val_accuracy has not improved
      • ReduceLROnPlateau — halves LR when val_loss plateaus
      • TensorBoard      — optional, writes to outputs/logs/
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(ckpt_dir, f"deepbravais_best_{timestamp}.weights.h5")

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,   # smaller file; load with model.load_weights()
        verbose=1,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=patience_es,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=patience_rlr,
        min_lr=1e-6,
        verbose=1,
    )

    # Optional TensorBoard logging
    log_dir = os.path.join("outputs", "logs", timestamp)
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir,
                                              histogram_freq=0)

    print(f"[DeepBravais] Checkpoint will be saved to: {ckpt_path}")
    return [checkpoint, early_stop, reduce_lr, tensorboard], ckpt_path


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------
def plot_training_curves(history: keras.callbacks.History,
                         save_dir: str) -> None:
    """
    Plot and save training / validation loss and accuracy curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DeepBravais — Training History", fontsize=14, fontweight="bold")

    # ── Loss ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(hist["loss"],     label="Train Loss",      color="#2196F3", lw=2)
    ax.plot(hist["val_loss"], label="Val Loss",         color="#F44336", lw=2, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sparse Categorical Cross-Entropy")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Accuracy ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(hist["accuracy"],     label="Train Accuracy", color="#2196F3", lw=2)
    ax.plot(hist["val_accuracy"], label="Val Accuracy",   color="#F44336", lw=2, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[DeepBravais] Training curves saved → {path}")


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: list,
                          save_dir: str) -> None:
    """
    Plot and save a normalised confusion matrix heatmap.
    """
    os.makedirs(save_dir, exist_ok=True)
    cm       = confusion_matrix(y_true, y_pred)
    cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm * 100,
        annot=True, fmt=".1f", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5, linecolor="lightgrey",
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("DeepBravais — Normalised Confusion Matrix (%)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[DeepBravais] Confusion matrix saved → {path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(data_path: str     = DEFAULT_DATA_PATH,
          ckpt_dir: str      = DEFAULT_CKPT_DIR,
          plot_dir: str      = DEFAULT_PLOT_DIR,
          epochs: int        = DEFAULT_EPOCHS,
          batch_size: int    = DEFAULT_BATCH_SIZE,
          learning_rate: float = DEFAULT_LR,
          dropout: float     = DEFAULT_DROPOUT,
          model_type: str    = DEFAULT_MODEL_TYPE) -> None:
    """
    End-to-end training routine.
    """
    # ── GPU / mixed-precision setup ───────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[DeepBravais] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        # Enable mixed precision for faster training on modern GPUs
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("[DeepBravais] Mixed precision (float16) enabled.")
    else:
        print("[DeepBravais] No GPU detected — training on CPU.")

    # ── 1. Load data ──────────────────────────────────────────────────────
    X, y, class_names = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ── 2. Build model ────────────────────────────────────────────────────
    if model_type == "small":
        model = build_resnet1d_small(dropout_rate=dropout)
    else:
        model = build_resnet1d(dropout_rate=dropout)

    model = compile_model(model, learning_rate=learning_rate)
    model.summary(line_length=100)
    print(f"[DeepBravais] Total parameters: {model.count_params():,}")

    # ── 3. Callbacks ──────────────────────────────────────────────────────
    callbacks, ckpt_path = build_callbacks(ckpt_dir)

    # ── 4. Compute class weights (handle minor imbalance from remainder) ──
    n_samples   = len(y_train)
    counts      = np.bincount(y_train, minlength=N_CLASSES)
    class_weight = {i: n_samples / (N_CLASSES * max(c, 1))
                    for i, c in enumerate(counts)}

    # ── 5. Train ──────────────────────────────────────────────────────────
    print(f"\n[DeepBravais] Starting training  "
          f"epochs={epochs}  batch={batch_size}  lr={learning_rate}")
    print("=" * 70)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 6. Evaluate on held-out test set ──────────────────────────────────
    print("\n[DeepBravais] Evaluating on test set …")
    test_loss, test_acc = model.evaluate(X_test, y_test,
                                         batch_size=batch_size, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")

    y_pred_prob = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    print("\n[DeepBravais] Per-class report:")
    print(classification_report(y_test, y_pred, target_names=class_names,
                                 digits=4))

    # ── 7. Save outputs ───────────────────────────────────────────────────
    plot_training_curves(history, plot_dir)
    plot_confusion_matrix(y_test, y_pred, class_names, plot_dir)

    # Also save the full model (architecture + weights) for inference
    os.makedirs("outputs", exist_ok=True)
    model_save_path = os.path.join("outputs", "deepbravais_final.keras")
    model.save(model_save_path)
    print(f"[DeepBravais] Final model saved → {model_save_path}")
    print("[DeepBravais] Training complete. ✓")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DeepBravais ResNet-1D on synthetic PXRD patterns."
    )
    parser.add_argument("--data",    type=str,   default=DEFAULT_DATA_PATH,
                        help="Path to the .npz dataset")
    parser.add_argument("--epochs",  type=int,   default=DEFAULT_EPOCHS,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch",   type=int,   default=DEFAULT_BATCH_SIZE,
                        help="Batch size (default: 256)")
    parser.add_argument("--lr",      type=float, default=DEFAULT_LR,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT,
                        help="Dropout rate in dense head (default: 0.4)")
    parser.add_argument("--model",   type=str,   default=DEFAULT_MODEL_TYPE,
                        choices=["full", "small"],
                        help="Model variant: 'full' or 'small' (default: full)")
    parser.add_argument("--ckpt_dir", type=str,  default=DEFAULT_CKPT_DIR,
                        help="Directory for model checkpoints")
    parser.add_argument("--plot_dir", type=str,  default=DEFAULT_PLOT_DIR,
                        help="Directory for output plots")
    args = parser.parse_args()

    train(
        data_path=args.data,
        ckpt_dir=args.ckpt_dir,
        plot_dir=args.plot_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        dropout=args.dropout,
        model_type=args.model,
    )
