"""
Machine learning models for the Collective Encoder framework.

This module provides high-level model wrappers (inheriting from PyTorch Lightning's
`LightningModule`) that coordinate networks, loss functions, metrics, and optimizers.
It includes deep learning models (VAEs, AEs, Graph nets) and classical
statistical methods (PCA, TICA, ICA).
"""
from .base import CEModelBase
from .pca_model import PCAModel, PCAEncoder
from .ica_model import ICAModel, ICAEncoder
from .tica_model import TICAModel, TICAEncoder
from .resolver import get_model, get_net

__all__ = [
    "CEModelBase",
    "PCAModel",
    "PCAEncoder",
    "ICAModel",
    "ICAEncoder",
    "TICAModel",
    "TICAEncoder",
    "get_model",
    "get_net",
]
