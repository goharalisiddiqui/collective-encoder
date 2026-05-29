from typing import Dict, Tuple

import torch
from torch.nn import functional as F

from .base import CEMetricBase


class CEMetricMAEDict(CEMetricBase):
    _IDENTIFIER = "CEMetricMAEDict"
    _REQUIRED_ARGS = ['keys']
    _OPTIONAL_ARGS = {
        'reduction': 'mean',
        'accumulation': 'sum',
    }
    
    def forward(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        mae = {}
        for key in self.keys:
            mae[key] = (
                F.l1_loss(output[key], labels[key], reduction=self.reduction)
                if labels[key].numel() > 0
                else torch.tensor(0.0, device=output[key].device)
            )
        
        if self.accumulation == 'sum':
            mae_acc = sum(mae.values())
        elif self.accumulation == 'mean':
            mae_acc = sum(mae.values())/len(mae)
        else:
            self.raise_error(f"Accumulation type '{self.accumulation}' is not recognized.")
            
        return mae_acc, mae

