"""
Loss functions for the Collective Encoder framework.

This module provides implementations of various losses, including reconstruction
losses (MSE, NLL), physics-informed constraints (steric, bond deviations), and
Kullback-Leibler divergences for variational formulations.
"""


from collective_encoder.common.registry import define_kind, register, lazy_exports
from .base import CELossBase

define_kind("losses", CELossBase)

_BUILTINS = {
    "CELossMSE": ("collective_encoder.losses.mse", "CELossMSE"),
    "CELossNLL": ("collective_encoder.losses.nll", "CELossNLL"),
    "CELossLatent": ("collective_encoder.losses.latent", "CELossLatent"),
    "CELossMSEDict": ("collective_encoder.losses.mse_dict", "CELossMSEDict"),
    "CELossSteric": ("collective_encoder.losses.steric", "CELossSteric"),
    "CELossBondDeviation": ("collective_encoder.losses.bond_deviation", "CELossBondDeviation"),
    "CELossKLDFlow": ("collective_encoder.losses.kld_flow", "CELossKLDFlow"),
    "CELossKLDGaussianMixture": ("collective_encoder.losses.kld_gaussian_mixture", "CELossKLDGaussianMixture"),
    "CELossKLDUniformGaussian": ("collective_encoder.losses.kld_uniform_gaussian", "CELossKLDUniformGaussian"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("losses", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = ["CELossBase"] + [t for m, t in _BUILTINS.values()]