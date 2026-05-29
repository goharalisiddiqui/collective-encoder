from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from collective_encoder.common.module import CEModule


class CEMetricBase(CEModule, ABC):
    def __init__(self, 
                args: Dict[str, Any] = None, 
                **kwargs) -> None:
        
        CEModule.__init__(self, args=args, **kwargs)
        
    def __call__(self, *args, **kwds):
        return self.calculate(*args, **kwds)

    @abstractmethod
    def calculate(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass
