from typing import Any, Dict, Tuple

import torch
from torch.nn import functional as F

from .base import CEMetricBase


class CEMetricKLD(CEMetricBase):
    _IDENTIFIER = "CEMetricKLD"
    _OPTIONAL_ARGS = {
        'prior': 'uniform_gaussian',
        'reduction': 'mean',
        'mu_name': 'mu_latent',
        'logvar_name': 'logvar_latent',
    }
    
    def calculate(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        if self.prior != 'uniform_gaussian':
            raise NotImplementedError(f"Prior '{self.prior}' is not implemented for KLD metric. Only 'uniform_gaussian' is supported.")
        if self.reduction not in ['mean', 'sum']:
            raise ValueError(f"Reduction method '{self.reduction}' is not supported. Use 'mean' or 'sum'.")
        
        try:
            mu = meta[self.mu_name]
            logvar = meta[self.logvar_name]
        except KeyError as e:
            raise KeyError(f"Missing required keys in meta dictionary for KLD calculation: {e}. "
                           f"Expected keys: '{self.mu_name}' and '{self.logvar_name}'.")
            
        mu = meta[self.mu_name]
        logvar = meta[self.logvar_name]
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)
        kld = kld.mean() if self.reduction == 'mean' else kld.sum()

        return kld, {}
