from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from .base import CELossBase
from .kld_schedulers.resolver import KLDResolver

EPSILON = 1e-7


class CELossKLDUniformGaussian(CELossBase):
    _IDENTIFIER = "CELossKLDUniformGaussian"
    _OPTIONAL_ARGS = {
        'mu_name': 'mu_latent',
        'logvar_name': 'logvar_latent',
        "kld_max_type": 'Fixed',
        "kld_max_scheduler_args": None,
    }
    
    def __init__(self, 
                args: Dict[str, Any] = None, 
                **kwargs) -> None:
        super().__init__(args, **kwargs)
        
        self.kld_scheduler = KLDResolver(self.kld_max_type, 
                                        self.kld_max_scheduler_args,
                                        **kwargs)
    
    def on_validation_epoch_end(self, plmodule):
        self.kld_scheduler.on_validation_epoch_end(plmodule)
    
    def kld(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> torch.Tensor:
        """
        KLD between the Gaussian latent distribution and a Normalizing flow (neural spline flows with a RealNVP structure) prior.

        """
        mu = meta[self.mu_name]
        logvar = meta[self.logvar_name]
        # KLD between univariate gaussian to Standard, explanation here:
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # Second Gaussian is zero mean and variance of 1, the prior on z
        # sum for all the latent variables
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)
        return kld

    def forward(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        loss_kld = self.kld(inp, latent, output, labels, meta)
        loss_reg = torch.mean(loss_kld, dim=0)
        
        meta = {"kld" : loss_reg}
        kld_max = self.kld_scheduler.get_kld_max(self)
        if loss_reg <= kld_max:
            loss_reg *= 0.0
        
        return loss_reg, meta
