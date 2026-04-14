"""
models.py
=========
1D Residual Network (ResNet-1D) for Bravais Lattice Classification.

Architecture Overview
---------------------
Input  : (batch, 1024, 1)  — PXRD pattern in Q-space

Stem   : Conv1D(64, 7) → BN → ReLU → MaxPool(3)   output: (batch, 340, 64)

Stage 1: 2 × ResBlock(64,  kernel=7)               output: (batch, 340, 64)
Stage 2: 2 × ResBlock(128, kernel=5, stride=2)     output: (batch, 170, 128)
Stage 3: 2 × ResBlock(256, kernel=5, stride=2)     output: (batch, 85,  256)
Stage 4: 2 × ResBlock(512, kernel=3, stride=2)     output: (batch, 43,  512)

Head   : Flatten → Dense(512) → BN → ReLU → Dropout(0.4)
         → Dense(128) → BN → ReLU → Dropout(0.4)
         → Dense(14, softmax)

Design Rationale
----------------
• We use **Flatten** (not GlobalAveragePooling1D) before the dense layers.
  This preserves absolute positional information in the flattened feature
  map, allowing the network to learn *where* peaks are absent (systematic
  absences / extinctions) — a crucial crystallographic fingerprint that
  encodes the centering type (P, I, F, C …).

• Large kernels in early stages (7, 5) capture the broad envelope of
  diffraction patterns; smaller kernels (3) in deeper stages refine
  fine-grained peak relationships.

• BatchNormalization after every convolution stabilises training.

• Dropout (0.4) in the dense head prevents co-adaptation on the large
  Flatten output.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_BINS    = 1024    # input length (Q-axis)
N_CLASSES = 14      # 14 Bravais lattices


# ---------------------------------------------------------------------------
# Building block: 1D Residual Block
# ---------------------------------------------------------------------------
def residual_block(x: tf.Tensor,
                   filters: int,
                   kernel_size: int = 3,
                   stride: int = 1,
                   dropout_rate: float = 0.0) -> tf.Tensor:
    """
    Pre-activation Residual Block for 1D sequences.

    Structure:
        ┌────────────────────────────────────────────────────┐
        │  BN → ReLU → Conv1D(filters, k, stride) → Dropout │
        │  BN → ReLU → Conv1D(filters, k, 1)                │
        └──────── (shortcut projection if needed) ───────────┘

    Pre-activation (BN before Conv) follows He et al. (2016) "Identity
    Mappings in Deep Residual Networks" and generally trains better on
    small datasets.

    Parameters
    ----------
    x           : input tensor
    filters     : number of output channels
    kernel_size : convolutional kernel width
    stride      : stride for the first conv (use 2 for downsampling)
    dropout_rate: spatial dropout rate between the two convolutions
    """
    shortcut = x

    # ── First convolution ─────────────────────────────────────────────────
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Projection shortcut: match dims when filters or stride change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters, kernel_size=1, strides=stride,
            padding="same", use_bias=False
        )(layers.Activation("relu")(layers.BatchNormalization()(shortcut)))

    x = layers.Conv1D(
        filters, kernel_size=kernel_size, strides=stride,
        padding="same", use_bias=False
    )(x)

    if dropout_rate > 0.0:
        # SpatialDropout1D drops entire feature maps — better for conv layers
        x = layers.SpatialDropout1D(dropout_rate)(x)

    # ── Second convolution ────────────────────────────────────────────────
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(
        filters, kernel_size=kernel_size, strides=1,
        padding="same", use_bias=False
    )(x)

    # ── Merge ─────────────────────────────────────────────────────────────
    x = layers.Add()([x, shortcut])
    return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
def build_resnet1d(input_length: int = N_BINS,
                   n_classes: int = N_CLASSES,
                   dropout_rate: float = 0.4) -> keras.Model:
    """
    Build the DeepBravais 1D-ResNet model.

    Parameters
    ----------
    input_length : length of the input PXRD pattern (default: 1024)
    n_classes    : number of output classes (default: 14)
    dropout_rate : dropout rate in the dense head (default: 0.4)

    Returns
    -------
    model : keras.Model (uncompiled)
    """
    inputs = keras.Input(shape=(input_length, 1), name="pxrd_input")

    # ── Stem ──────────────────────────────────────────────────────────────
    # Wide initial receptive field to capture low-Q envelope features
    x = layers.Conv1D(64, kernel_size=7, strides=1,
                      padding="same", use_bias=False,
                      name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2,
                            padding="same", name="stem_pool")(x)
    # shape: (batch, 512, 64)

    # ── Stage 1: 64 filters, no downsampling ──────────────────────────────
    for i in range(2):
        x = residual_block(x, filters=64, kernel_size=7,
                           stride=1, dropout_rate=0.0)
    # shape: (batch, 512, 64)

    # ── Stage 2: 128 filters, stride-2 downsampling ───────────────────────
    x = residual_block(x, filters=128, kernel_size=5,
                       stride=2, dropout_rate=0.15)
    x = residual_block(x, filters=128, kernel_size=5,
                       stride=1, dropout_rate=0.0)
    # shape: (batch, 256, 128)

    # ── Stage 3: 256 filters, stride-2 downsampling ───────────────────────
    x = residual_block(x, filters=256, kernel_size=5,
                       stride=2, dropout_rate=0.15)
    x = residual_block(x, filters=256, kernel_size=5,
                       stride=1, dropout_rate=0.0)
    # shape: (batch, 128, 256)

    # ── Stage 4: 512 filters, stride-2 downsampling ───────────────────────
    x = residual_block(x, filters=512, kernel_size=3,
                       stride=2, dropout_rate=0.15)
    x = residual_block(x, filters=512, kernel_size=3,
                       stride=1, dropout_rate=0.0)
    # shape: (batch, 64, 512)

    # ── Final BN + ReLU (close the pre-activation chain) ──────────────────
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # ── Classification Head ───────────────────────────────────────────────
    # IMPORTANT: Flatten — NOT GlobalAveragePooling1D.
    # Preserves positional (Q-position) information so that the dense layers
    # can detect systematic absences (missing peaks at specific Q values).
    x = layers.Flatten(name="flatten")(x)
    # shape: (batch, 64 * 512) = (batch, 32768)

    # L2 regularisation on dense layers to combat overfitting from the
    # large Flatten output (32,768 features → 16.8M weights in fc1 alone).
    _l2 = regularizers.l2(1e-4)

    x = layers.Dense(512, use_bias=False, kernel_regularizer=_l2,
                     name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dropout(dropout_rate, name="fc1_drop")(x)

    x = layers.Dense(128, use_bias=False, kernel_regularizer=_l2,
                     name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Activation("relu", name="fc2_relu")(x)
    x = layers.Dropout(dropout_rate, name="fc2_drop")(x)

    outputs = layers.Dense(n_classes, activation="softmax",
                           name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="DeepBravais_ResNet1D")
    return model


# ---------------------------------------------------------------------------
# Compact variant: lighter model for quick experiments / CPU runs
# ---------------------------------------------------------------------------
def build_resnet1d_small(input_length: int = N_BINS,
                         n_classes: int = N_CLASSES,
                         dropout_rate: float = 0.4) -> keras.Model:
    """
    A lighter DeepBravais variant (~3× fewer parameters).
    Useful for prototyping or CPU-only environments.
    """
    inputs = keras.Input(shape=(input_length, 1), name="pxrd_input")

    x = layers.Conv1D(32, 7, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(3, strides=2, padding="same")(x)

    for _ in range(2):
        x = residual_block(x, filters=32,  kernel_size=7)
    x = residual_block(x, filters=64,  kernel_size=5, stride=2)
    x = residual_block(x, filters=64,  kernel_size=5)
    x = residual_block(x, filters=128, kernel_size=3, stride=2)
    x = residual_block(x, filters=128, kernel_size=3)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(n_classes, activation="softmax",
                           name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs,
                       name="DeepBravais_ResNet1D_Small")


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = build_resnet1d()
    model.summary(line_length=100)
    print(f"\nTotal parameters: {model.count_params():,}")

    # Verify output shape
    import numpy as np
    dummy = np.random.rand(4, N_BINS, 1).astype(np.float32)
    out   = model(dummy, training=False)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}   (should be (4, 14))")
    assert out.shape == (4, N_CLASSES), "Output shape mismatch!"
    print("✓ Shape check passed.")
