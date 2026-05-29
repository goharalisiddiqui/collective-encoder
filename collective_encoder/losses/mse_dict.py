from typing import Any, Dict, Tuple

import torch
from torch.distributions.normal import Normal

from .base import CELossBase


class CELossMSEDict(CELossBase):
    _IDENTIFIER = "CELossMSE"
    _REQUIRED_ARGS = ['keys']
    _OPTIONAL_ARGS = {
        'weights': None,
        'reduction': 'mean',
        'accumulation': 'sum',
    }
    
    def __init__(self, 
                args: Dict[str, Any] = None, 
                **kwargs) -> None:
        super().__init__(args, **kwargs)
        if self.weights == None:
            self.weights = [1.0] * len(self.keys)
        
        if len(self.keys) != len (self.weights):
            self.raise_error(f"Keys {self.keys} and weights {self.weights} must be of same length.")
        
        self.loss_fn = torch.nn.MSELoss(reduction=self.reduction)

    def forward(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        losses = {}
        for key, weight in zip(self.keys, self.weights):
            losses[key] = self.loss_fn(output[key], labels[key]) * weight

        if self.accumulation == 'sum':
            loss = sum(losses.values())
        elif self.accumulation == 'mean':
            loss = sum(losses.values())/len(losses)
        else:
            self.raise_error(f"Accumulation type '{self.accumulation}' is not recognized.")
            
        return loss, losses

