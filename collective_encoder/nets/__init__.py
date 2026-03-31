"""Neural network architectures for molecular dynamics modeling."""

from collective_encoder.nets.vae_net import VAE
from collective_encoder.nets.ae_net import AE
from collective_encoder.nets.dvae_net import DVAE
from collective_encoder.nets.ae_base import AEBase

__all__ = [
    "VAE",
    "AE",
    "DVAE",
    "AEBase",
]