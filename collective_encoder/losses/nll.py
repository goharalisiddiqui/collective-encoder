from typing import Any, Dict, Tuple

import torch
from torch.distributions.normal import Normal

from .base import CELossBase

EPSILON = 1e-7

class CELossNLL(CELossBase):
    _IDENTIFIER = "CELossNLL"
    _OPTIONAL_ARGS = {
        'mu_name': 'mu_x',
        'logvar_name': 'logvar_x',
    }
    
    def __init__(self, 
                args: Dict[str, Any] = None, 
                **kwargs) -> None:
        super().__init__(self, args, **kwargs)

    def forward(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        mu_x = meta[self.mu_name]
        logvar_x = meta[self.logvar_name]

        logvar_x = torch.clamp(logvar_x, min=-4.0, max=4.0) # Clamp log-variance to prevent numerical instability in exp/log operations
        sd = torch.exp(0.5 * logvar_x) + EPSILON
        p_x = Normal(mu_x, sd)
        loss_rec = -torch.sum(p_x.log_prob(inp), dim=1)

        loss_rec = torch.mean(loss_rec)

        return loss_rec, {}
