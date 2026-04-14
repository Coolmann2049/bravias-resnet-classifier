"""
models.py  —  backward-compatibility shim
==========================================
The model architecture now lives in src/models.py.
This file re-exports everything so that notebooks and scripts that do

    from models import build_resnet1d, N_CLASSES

continue to work without modification.
"""
from src.models import (           # noqa: F401  (re-export)
    StochasticDepth,
    convnext_block,
    build_resnet1d,
    build_resnet1d_small,
)
from src.config import N_BINS, N_CLASSES  # noqa: F401
