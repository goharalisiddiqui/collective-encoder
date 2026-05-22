from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from .kld_uniform_gaussian import CELossKLDUniformGaussian

EPSILON = 1e-7


class CELossKLDGaussianMixture(CELossKLDUniformGaussian):
    _IDENTIFIER = "CELossKLDGaussianMixture"
    _REQUIRED_ARGS = CELossKLDUniformGaussian._REQUIRED_ARGS + ['n_components']
    
    def kld(self, mu, logvar):
        """
        KLD between the Gaussian latent distribution and a Gaussian Mixture prior.  
        The Gaussian Mixture is assumed to have 'n_components' components, each with mean 0 and variance 1, and equal mixing coefficients.  
        The KLD is computed as the difference between the log-probability of the latent under the Gaussian Mixture and the log-probability of the latent under the Gaussian distribution defined by (mu, logvar).

        """
        
        n_components = self.n_components
        # Compute log-probability of the latent under the Gaussian distribution defined by (mu, logvar)
        log_qz_x = -0.5 * torch.sum(logvar + torch.pow(mu, 2) + torch.exp(logvar), axis=1) - 0.5 * mu.shape[1] * torch.log(torch.tensor(2 * torch.pi))
        
        # Compute log-probability of the latent under the Gaussian Mixture prior
        log_pz = -0.5 * torch.sum(torch.pow(mu, 2) + torch.exp(logvar), axis=1) - 0.5 * mu.shape[1] * torch.log(torch.tensor(2 * torch.pi)) - torch.log(torch.tensor(n_components, dtype=torch.float))
        
        kld = log_qz_x - log_pz
        return kld
