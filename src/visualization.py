"""
src/visualization.py
====================
Plotting utilities for the DeepBravais training pipeline.

All functions save to disk AND return the matplotlib Figure so callers
can embed them in notebooks or run headlessly (no display required).

Public API
----------
    plot_training_curves(history, save_dir, show)
    plot_confusion_matrix(y_true, y_pred, class_names, save_dir, show)
"""

import os
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Lazy matplotlib import — allows the module to be imported in headless
# environments (Kaggle, CI) without raising a DisplayError.
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ===========================================================================
# Training-curve plot
# ===========================================================================

def plot_training_curves(history,
                         save_dir: str = "outputs/plots",
                         show: bool = False,
                         filename: str = "training_curves.png"):
    """
    Plot loss and accuracy curves from a Keras History object.

    Parameters
    ----------
    history  : keras History returned by model.fit()
    save_dir : directory to write the PNG  (created if absent)
    show     : if True, call plt.show() — set False for headless runs
    filename : output filename

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DeepBravais — Training History", fontsize=14, fontweight="bold")

    epochs = range(1, len(history.history["loss"]) + 1)

    # ── Loss ─────────────────────────────────────────────────────────────
    ax_loss.plot(epochs, history.history["loss"],
                 color="#4C72B0", linewidth=2, label="Train Loss")
    if "val_loss" in history.history:
        ax_loss.plot(epochs, history.history["val_loss"],
                     color="#DD8452", linewidth=2, linestyle="--", label="Val Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Sparse Categorical Cross-Entropy")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    # ── Accuracy ─────────────────────────────────────────────────────────
    ax_acc.plot(epochs, history.history["accuracy"],
                color="#4C72B0", linewidth=2, label="Train Accuracy")
    if "val_accuracy" in history.history:
        ax_acc.plot(epochs, history.history["val_accuracy"],
                    color="#DD8452", linewidth=2, linestyle="--", label="Val Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[visualization] Saved training curves → {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return fig


# ===========================================================================
# Confusion-matrix plot
# ===========================================================================

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          save_dir: str = "outputs/plots",
                          show: bool = False,
                          filename: str = "confusion_matrix.png",
                          normalise: bool = True):
    """
    Plot a (optionally normalised) confusion matrix as a heatmap.

    Parameters
    ----------
    y_true       : ground-truth integer labels  (N,)
    y_pred       : predicted integer labels     (N,)
    class_names  : list of 14 class-name strings; defaults to integers
    save_dir     : directory to write the PNG
    show         : call plt.show() if True
    filename     : output filename
    normalise    : if True, show row-normalised percentages (default True)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        cm_plot  = np.where(row_sums > 0, cm / row_sums * 100, 0.0)
        fmt, cbar_label = ".1f", "Recall (%)"
    else:
        cm_plot, fmt, cbar_label = cm.astype(float), ".0f", "Count"

    n = cm.shape[0]
    tick_labels = class_names if class_names else [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    title = "DeepBravais — Normalised Confusion Matrix (%)" if normalise \
            else "DeepBravais — Confusion Matrix"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Bravais Lattice", fontsize=11)
    ax.set_ylabel("True Bravais Lattice",      fontsize=11)
    plt.tight_layout()

    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[visualization] Saved confusion matrix → {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return fig
