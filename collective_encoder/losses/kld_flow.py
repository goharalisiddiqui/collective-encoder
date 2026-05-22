from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from .kld_uniform_gaussian import CELossKLDUniformGaussian

EPSILON = 1e-7


class CELossKLDFlow(CELossKLDUniformGaussian):
    _IDENTIFIER = "CELossKLDFlow"
    _REQUIRED_ARGS = CELossKLDUniformGaussian._REQUIRED_ARGS + ['n_components']
    
    def kld(self, mu, logvar):
        """
        KLD between the Gaussian latent distribution and a Normalizing flow

        """
        
        return kld
