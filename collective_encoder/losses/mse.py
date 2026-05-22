from typing import Any, Dict, Tuple

import torch
from torch.distributions.normal import Normal

from .base import CELossBase

EPSILON = 1e-7

class CELossMSE(CELossBase):
    _IDENTIFIER = "CELossMSE"
    _OPTIONAL_ARGS = {
        'reduction': 'mean',
    }
    
    def __init__(self, 
                args: Dict[str, Any] = None, 
                **kwargs) -> None:
        super().__init__(self, args, **kwargs)
        self.mse = torch.nn.MSELoss(reduction=self.reduction)

    def forward(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        loss = self.mse(output, inp)

        return loss, {}
