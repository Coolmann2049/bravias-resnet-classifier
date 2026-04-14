"""
src/trainer.py
==============
Training-loop utilities: model compilation and Keras callbacks.

Public API
----------
    compile_model(model, learning_rate, warmup_epochs, total_epochs,
                  steps_per_epoch) → compiled keras.Model
    build_callbacks(ckpt_dir, patience_es, patience_rlr) → (callbacks, ckpt_path)
"""

import os
from datetime import datetime

from tensorflow import keras

from src.config import DEFAULT_LR, DEFAULT_EPOCHS


# ===========================================================================
# Model compilation
# ===========================================================================

def compile_model(model: keras.Model,
                  learning_rate:   float = DEFAULT_LR,
                  warmup_epochs:   int   = 5,
                  total_epochs:    int   = DEFAULT_EPOCHS,
                  steps_per_epoch: int   = 1) -> keras.Model:
    """
    Compile with Adam + cosine-decay schedule with linear warmup.

    Warmup prevents the val-loss spike caused by a high LR hitting an
    un-trained network at step 0.  Cosine decay smoothly anneals the LR
    to near-zero over the remaining epochs.

    Parameters
    ----------
    model           : uncompiled keras.Model
    learning_rate   : peak LR (reached after warmup)
    warmup_epochs   : epochs over which LR linearly ramps up
    total_epochs    : total training epochs (for decay schedule length)
    steps_per_epoch : training steps per epoch (= len(X_train)//batch_size)

    Returns
    -------
    model : same model, now compiled in-place
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate = learning_rate,
        decay_steps           = max(total_steps - warmup_steps, 1),
        alpha                 = 1e-6,
        warmup_target         = learning_rate,
        warmup_steps          = warmup_steps,
    )

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule,
                                          clipnorm=1.0),
        loss      = keras.losses.SparseCategoricalCrossentropy(),
        metrics   = ["accuracy"],
    )
    return model


# ===========================================================================
# Callbacks
# ===========================================================================

def build_callbacks(ckpt_dir:     str = "outputs/checkpoints",
                    patience_es:  int = 15,
                    patience_rlr: int = 7) -> tuple:
    """
    Build and return standard Keras callbacks for DeepBravais training.

    Callbacks
    ---------
    ModelCheckpoint  — saves best model weights (by val_accuracy)
    EarlyStopping    — stops when val_accuracy has not improved for
                       ``patience_es`` epochs; restores best weights
    ReduceLROnPlateau — halves LR if val_loss stalls for ``patience_rlr``
                        epochs (works alongside cosine schedule as a fallback)
    TensorBoard      — writes logs to outputs/logs/<timestamp>/

    Parameters
    ----------
    ckpt_dir     : directory for checkpoint .h5 files
    patience_es  : EarlyStopping patience (default 15)
    patience_rlr : ReduceLROnPlateau patience (default 7)

    Returns
    -------
    callbacks : list[keras.callbacks.Callback]
    ckpt_path : str  — path to the best-weights file
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(ckpt_dir,
                             f"deepbravais_best_{timestamp}.weights.h5")

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath         = ckpt_path,
        monitor          = "val_accuracy",
        mode             = "max",
        save_best_only   = True,
        save_weights_only= True,
        verbose          = 1,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor              = "val_accuracy",
        mode                 = "max",
        patience             = patience_es,
        restore_best_weights = True,
        verbose              = 1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = patience_rlr,
        min_lr   = 1e-7,
        verbose  = 1,
    )

    log_dir    = os.path.join("outputs", "logs", timestamp)
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir,
                                              histogram_freq=0)

    print(f"[trainer] Checkpoint will be saved to: {ckpt_path}")
    return [checkpoint, early_stop, reduce_lr, tensorboard], ckpt_path
