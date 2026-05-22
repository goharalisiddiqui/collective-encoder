from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from collective_encoder.common.module import CEModule


class CELossBase(torch.nn.Module, CEModule, ABC):
    def __init__(self, 
                args: Dict[str, Any] = None, 
                **kwargs) -> None:
        
        torch.nn.Module.__init__(self)
        CEModule.__init__(self, args=args, **kwargs)

    @abstractmethod
    def forward(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass
