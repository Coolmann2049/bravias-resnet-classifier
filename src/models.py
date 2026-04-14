"""
src/models.py
=============
ConvNeXt-1D architecture for Bravais Lattice classification from PXRD.

Inspired by AlphaDiffract (Andrejevic et al., 2026).  Large depthwise
kernels (up to k=31) provide the receptive field needed to detect
systematic absences across the full Q range [0.7, 6.0] Å⁻¹.

Public API
----------
    build_resnet1d(input_length, n_classes, dropout_rate) → keras.Model
    build_resnet1d_small(input_length, n_classes, dropout_rate) → keras.Model
    StochasticDepth  (layer class, exposed for serialisation)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import N_BINS, N_CLASSES


# ===========================================================================
# Stochastic Depth  (DropPath)
# ===========================================================================

class StochasticDepth(layers.Layer):
    """
    Randomly drops the entire residual branch during training.

    At inference the branch is always kept but scaled by (1 - drop_rate)
    to preserve expected activation magnitude.

    Reference: Huang et al. (2016) "Deep Networks with Stochastic Depth".
    """
    def __init__(self, drop_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def call(self, x, training=None):
        if not training:
            return x
        batch_size    = tf.shape(x)[0]
        keep_prob     = 1.0 - self.drop_rate
        random_tensor = tf.random.uniform((batch_size, 1, 1)) + keep_prob
        random_tensor = tf.floor(random_tensor)
        random_tensor = tf.cast(random_tensor,             dtype=x.dtype)
        keep_prob_t   = tf.cast(tf.constant(keep_prob),   dtype=x.dtype)
        return (x / keep_prob_t) * random_tensor

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_rate": self.drop_rate})
        return cfg


# ===========================================================================
# Building blocks
# ===========================================================================

def convnext_block(x: tf.Tensor,
                   filters: int,
                   kernel_size: int = 7,
                   drop_path_rate: float = 0.0) -> tf.Tensor:
    """
    ConvNeXt-1D inverted-bottleneck block.

    Structure
    ---------
    Input
      ↓  DepthwiseConv1D(kernel_size)     — large kernel, O(k·C) params
      ↓  LayerNorm
      ↓  Dense(filters × 4)              — pointwise expand
      ↓  GELU
      ↓  Dense(filters)                  — pointwise contract
      ↓  [StochasticDepth]
      ↓  Add(residual)
    """
    residual = x

    x = layers.DepthwiseConv1D(kernel_size=kernel_size,
                               padding="same", use_bias=True)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(filters * 4)(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dense(filters)(x)

    if drop_path_rate > 0.0:
        x = StochasticDepth(drop_path_rate)(x)

    return layers.Add()([x, residual])


def _downsample(x: tf.Tensor, filters: int, stride: int = 2) -> tf.Tensor:
    """LayerNorm → stride-2 pointwise Conv1D (transition between stages)."""
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Conv1D(filters, kernel_size=2, strides=stride,
                      padding="same", use_bias=False)(x)
    return x


# ===========================================================================
# Full model  (~7M parameters)
# ===========================================================================

def build_resnet1d(input_length: int = N_BINS,
                   n_classes: int   = N_CLASSES,
                   dropout_rate: float = 0.0) -> keras.Model:
    """
    Build the DeepBravais ConvNeXt-1D model (~7M parameters).

    The function name 'build_resnet1d' is preserved for drop-in
    compatibility with train.py and any user notebooks.

    Architecture summary
    --------------------
    Stem   : Conv1D(96, k=7, stride=2) + LayerNorm  → (batch, 512, 96)
    Stage 1: 3 × ConvNeXtBlock(96,  k=31)            → (batch, 512, 96)
    Stage 2: 3 × ConvNeXtBlock(192, k=15)  + DS      → (batch, 256, 192)
    Stage 3: 9 × ConvNeXtBlock(384, k=7)   + DS      → (batch, 128, 384)
    Stage 4: 3 × ConvNeXtBlock(768, k=7)   + DS      → (batch,  64, 768)
    Head   : GAP → LayerNorm → Dense(n_classes, softmax)

    Parameters
    ----------
    input_length : Q-axis bins (default 1024)
    n_classes    : output classes (default 14)
    dropout_rate : optional head dropout (default 0; SD regularises instead)

    Returns
    -------
    model : keras.Model  (uncompiled)
    """
    # (channels, n_blocks, kernel_size, stochastic-depth rate)
    stage_cfg = [
        ( 96, 3, 31, 0.00),
        (192, 3, 15, 0.05),
        (384, 9,  7, 0.10),
        (768, 3,  7, 0.15),
    ]

    inputs = keras.Input(shape=(input_length, 1), name="pxrd_input")

    x = layers.Conv1D(96, kernel_size=7, strides=2,
                      padding="same", use_bias=False,
                      name="stem_conv")(inputs)
    x = layers.LayerNormalization(epsilon=1e-6, name="stem_ln")(x)

    for s_idx, (filters, n_blocks, kernel, dp_rate) in enumerate(stage_cfg):
        if s_idx > 0:
            x = _downsample(x, filters)
        for _ in range(n_blocks):
            x = convnext_block(x, filters=filters,
                               kernel_size=kernel,
                               drop_path_rate=dp_rate)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="head_ln")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(n_classes, activation="softmax",
                           name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs,
                       name="DeepBravais_ConvNeXt1D")


# ===========================================================================
# Compact variant (~1.5M parameters)
# ===========================================================================

def build_resnet1d_small(input_length: int = N_BINS,
                          n_classes: int   = N_CLASSES,
                          dropout_rate: float = 0.0) -> keras.Model:
    """
    Compact ConvNeXt-1D (~1.5M parameters) for rapid prototyping / CPU runs.
    """
    stage_cfg = [
        ( 32, 2, 31, 0.00),
        ( 64, 2, 15, 0.05),
        (128, 3,  7, 0.10),
        (256, 2,  7, 0.10),
    ]

    inputs = keras.Input(shape=(input_length, 1), name="pxrd_input")

    x = layers.Conv1D(32, kernel_size=7, strides=2,
                      padding="same", use_bias=False,
                      name="stem_conv")(inputs)
    x = layers.LayerNormalization(epsilon=1e-6, name="stem_ln")(x)

    for s_idx, (filters, n_blocks, kernel, dp_rate) in enumerate(stage_cfg):
        if s_idx > 0:
            x = _downsample(x, filters)
        for _ in range(n_blocks):
            x = convnext_block(x, filters=filters,
                               kernel_size=kernel,
                               drop_path_rate=dp_rate)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="head_ln")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(n_classes, activation="softmax",
                           name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs,
                       name="DeepBravais_ConvNeXt1D_Small")


# ===========================================================================
# Quick sanity check
# ===========================================================================
if __name__ == "__main__":
    import numpy as np

    print("=== Full model (ConvNeXt-1D) ===")
    model = build_resnet1d()
    model.summary(line_length=100)
    print(f"\nTotal parameters: {model.count_params():,}")

    dummy = np.random.rand(4, N_BINS, 1).astype(np.float32)
    out   = model(dummy, training=False)
    assert out.shape == (4, N_CLASSES), f"Shape mismatch: {out.shape}"
    print(f"Input  : {dummy.shape}  →  Output : {out.shape}  ✓")

    print("\n=== Small model (ConvNeXt-1D) ===")
    small = build_resnet1d_small()
    print(f"Total parameters: {small.count_params():,}")
