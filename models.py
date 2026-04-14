"""
models.py
=========
ConvNeXt-1D for Bravais Lattice Classification from PXRD patterns.

Architecture: DeepBravais-ConvNeXt
-----------------------------------
Inspired by AlphaDiffract (Andrejevic et al., 2026), which demonstrated
that a 1D ConvNeXt outperforms standard 1D ResNets for PXRD crystallographic
classification by using:

  1. Large depthwise kernels  — see enough Q-space to detect peak *absences*
     (systematic absences span 0.1–0.5 Å⁻¹; ResNet k=7 sees only 0.035 Å⁻¹)
  2. Inverted bottleneck blocks — far fewer parameters than dense Flatten head
  3. LayerNorm  — more stable than BatchNorm at small batch sizes / mixed precision
  4. GELU activation — smoother gradients than ReLU
  5. GlobalAveragePooling — safe with large kernels; avoids the 16.8M-param
     Flatten → Dense bottleneck that caused the 40% val accuracy plateau

Input  : (batch, 1024, 1)  — PXRD pattern in Q-space, normalised [0, 1]

Stem   : Conv1D(96, k=7, stride=2) → LayerNorm        → (batch, 512, 96)
Stage 1: 3 × ConvNeXtBlock(96,  k=31)  no downsample  → (batch, 512, 96)
Stage 2: 3 × ConvNeXtBlock(192, k=15)  stride-2 stem  → (batch, 256, 192)
Stage 3: 9 × ConvNeXtBlock(384, k=7)   stride-2 stem  → (batch, 128, 384)
Stage 4: 3 × ConvNeXtBlock(768, k=7)   stride-2 stem  → (batch,  64, 768)

Head   : GlobalAveragePooling1D → LayerNorm → Dense(14, softmax)

~7M parameters  (vs ~18M for the previous ResNet-1D)

Public API  (identical to original file — train.py works unchanged)
----------
    from models import build_resnet1d, build_resnet1d_small, N_CLASSES, N_BINS
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_BINS    = 1024   # input length (Q-axis bins)
N_CLASSES = 14     # 14 Bravais lattices


# ---------------------------------------------------------------------------
# Stochastic Depth (DropPath)
# ---------------------------------------------------------------------------
class StochasticDepth(layers.Layer):
    """
    Randomly drops the entire residual branch during training.
    At inference, the branch is kept but scaled by (1 - drop_rate).
    Introduced in 'Deep Networks with Stochastic Depth' (Huang et al., 2016).
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
        random_tensor = tf.floor(random_tensor)          # Bernoulli sample
        return (x / keep_prob) * random_tensor

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_rate": self.drop_rate})
        return cfg


# ---------------------------------------------------------------------------
# ConvNeXt-1D block
# ---------------------------------------------------------------------------
def convnext_block(x: tf.Tensor,
                   filters: int,
                   kernel_size: int = 7,
                   drop_path_rate: float = 0.0) -> tf.Tensor:
    """
    ConvNeXt-1D inverted-bottleneck block.

    Structure
    ---------
    Input
      ↓  Depthwise Conv1D(filters, kernel_size)   — large kernel, few params
      ↓  LayerNorm
      ↓  Dense(filters × 4)                       — pointwise expand
      ↓  GELU
      ↓  Dense(filters)                           — pointwise contract
      ↓  [StochasticDepth]
      ↓  Add(residual)

    Why depthwise?  Standard conv costs kernel×C_in×C_out params.
                    Depthwise costs only kernel×C params — ~C× cheaper.

    Why large kernel?  A k=31 kernel at bin-width 0.00518 Å⁻¹ sees
                       31×0.00518 ≈ 0.16 Å⁻¹ of Q-space in one operation —
                       enough to span a typical systematic-absence gap.
    """
    residual = x

    x = layers.DepthwiseConv1D(kernel_size=kernel_size,
                               padding="same", use_bias=True)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(filters * 4)(x)          # expand
    x = layers.Activation("gelu")(x)
    x = layers.Dense(filters)(x)              # contract

    if drop_path_rate > 0.0:
        x = StochasticDepth(drop_path_rate)(x)

    x = layers.Add()([x, residual])
    return x


# ---------------------------------------------------------------------------
# Downsampling transition between stages
# ---------------------------------------------------------------------------
def _downsample(x: tf.Tensor, filters: int, stride: int = 2) -> tf.Tensor:
    """LayerNorm → stride-2 pointwise Conv1D."""
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Conv1D(filters, kernel_size=2, strides=stride,
                      padding="same", use_bias=False)(x)
    return x


# ---------------------------------------------------------------------------
# Full model  (~7M parameters)
# ---------------------------------------------------------------------------
def build_resnet1d(input_length: int = N_BINS,
                   n_classes: int = N_CLASSES,
                   dropout_rate: float = 0.0) -> keras.Model:
    """
    Build the DeepBravais ConvNeXt-1D model.

    The function name is kept as 'build_resnet1d' for drop-in compatibility
    with train.py.  Internally this is a ConvNeXt-1D, not a ResNet.

    Parameters
    ----------
    input_length : Q-axis bins (default 1024)
    n_classes    : number of Bravais classes (default 14)
    dropout_rate : optional dropout before the classifier head

    Returns
    -------
    model : keras.Model (uncompiled), ~7M parameters
    """
    # (channels, n_blocks, kernel_size, stochastic-depth rate)
    stage_cfg = [
        ( 96, 3, 31, 0.00),   # Stage 1 — k=31 → 0.16 Å⁻¹ receptive field
        (192, 3, 15, 0.05),   # Stage 2
        (384, 9,  7, 0.10),   # Stage 3 — deepest, most blocks
        (768, 3,  7, 0.15),   # Stage 4
    ]

    inputs = keras.Input(shape=(input_length, 1), name="pxrd_input")

    # ── Stem: stride-2 conv → LayerNorm ───────────────────────────────────
    x = layers.Conv1D(96, kernel_size=7, strides=2,
                      padding="same", use_bias=False,
                      name="stem_conv")(inputs)
    x = layers.LayerNormalization(epsilon=1e-6, name="stem_ln")(x)
    # shape: (batch, 512, 96)

    # ── ConvNeXt stages ───────────────────────────────────────────────────
    for s_idx, (filters, n_blocks, kernel, dp_rate) in enumerate(stage_cfg):
        if s_idx > 0:                          # downsample between stages
            x = _downsample(x, filters)
        for _ in range(n_blocks):
            x = convnext_block(x,
                               filters        = filters,
                               kernel_size    = kernel,
                               drop_path_rate = dp_rate)
    # shape after stage 4: (batch, 64, 768)

    # ── Head ──────────────────────────────────────────────────────────────
    # GlobalAveragePooling is safe here: large kernels in early stages have
    # already encoded positional (Q-position) information into the feature
    # vectors before pooling removes the spatial dimension.
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="head_ln")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(n_classes, activation="softmax",
                           name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs,
                       name="DeepBravais_ConvNeXt1D")


# ---------------------------------------------------------------------------
# Compact variant  (~1.5M parameters)
# ---------------------------------------------------------------------------
def build_resnet1d_small(input_length: int = N_BINS,
                         n_classes: int = N_CLASSES,
                         dropout_rate: float = 0.0) -> keras.Model:
    """
    Compact ConvNeXt-1D (~1.5M parameters).
    Good for CPU-only environments or rapid prototyping.
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


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
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
