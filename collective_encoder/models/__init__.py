"""
Machine learning models for the Collective Encoder framework.

This module provides high-level model wrappers (inheriting from PyTorch Lightning's
`LightningModule`) that coordinate networks, loss functions, metrics, and optimizers.
It includes deep learning models (VAEs, AEs, Graph nets) and classical
statistical methods (PCA, TICA, ICA).
"""


from collective_encoder.common.registry import define_kind, register, lazy_exports
from .base import CEModelBase

define_kind("models", CEModelBase)

_BUILTINS = {
    "AE": ("collective_encoder.models.neural_nets.ae_net", "AE"),
    "sAE": ("collective_encoder.models.neural_nets.ae_net", "sAE"),
    "VAE": ("collective_encoder.models.neural_nets.vae_net", "VAE"),
    "sVAE": ("collective_encoder.models.neural_nets.vae_net", "sVAE"),
    "DVAE": ("collective_encoder.models.neural_nets.dvae_net", "DVAE"),
    "sDVAE": ("collective_encoder.models.neural_nets.dvae_net", "sDVAE"),
    "EDVAE": ("collective_encoder.models.neural_nets.edvae_net", "EDVAE"),
    "BGE": ("collective_encoder.models.neural_nets.bge", "BondGraphEncoderDecoder"),
    "BGE_V2": ("collective_encoder.models.neural_nets.bge_v2", "BondGraphEncoderDecoderV2"),
    "PCA": ("collective_encoder.models.pca_model", "PCAModel"),
    "PCAEncoder": ("collective_encoder.models.pca_model", "PCAModel"),
    "ICA": ("collective_encoder.models.ica_model", "ICAModel"),
    "ICAEncoder": ("collective_encoder.models.ica_model", "ICAModel"),
    "TICA": ("collective_encoder.models.tica_model", "TICAModel"),
    "TICAEncoder": ("collective_encoder.models.tica_model", "TICAModel"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("models", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = ["CEModelBase"] + [t for m, t in _BUILTINS.values()]
