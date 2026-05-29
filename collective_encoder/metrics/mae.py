from typing import Any, Dict, Tuple

import torch
from torch.nn import functional as F

from .base import CEMetricBase


class CEMetricMAE(CEMetricBase):
    _IDENTIFIER = "CEMetricMAE"
    _OPTIONAL_ARGS = {
        'reduction': 'mean',
    }
    
    def __init__(self, 
                args: Dict[str, Any] = None, 
                **kwargs) -> None:
        super().__init__(args, **kwargs)

    def calculate(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        mae = F.l1_loss(inp, output, reduction=self.reduction)

        return mae, {}
