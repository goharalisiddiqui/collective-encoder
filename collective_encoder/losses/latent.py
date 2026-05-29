from typing import Dict, Tuple

import torch

from .base import CELossBase


class CELossLatent(CELossBase):
    _IDENTIFIER = "CELossLatent"
    
    def forward(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encourage sequential latent points to be close and equidistant."""
        loss_latent = torch.tensor(0.0, device=latent.device)
        if latent.size(0) > 1:
            batch_dist = torch.norm(latent[1:] - latent[:-1], dim=1)
            loss_latent = torch.mean(batch_dist)
        if latent.size(0) > 2:
            loss_latent = loss_latent + torch.var(batch_dist)
        return loss_latent, {}
