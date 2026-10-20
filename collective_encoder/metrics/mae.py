from typing import Any, Dict, Tuple

import torch
from torch.nn import functional as F

from .base import CEMetricBase


class CEMetricMAE(CEMetricBase):
    """
    Standard Mean Absolute Error (MAE) metric.

    Computes the L1 distance between the network output and the input.
    Often used to report a human-readable average error.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary. Optional keys:
        - ``reduction`` (str, default 'mean'): Reduction method ('mean' or 'sum').
    kwargs : dict
        Additional keyword arguments.
    """
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
        """
        Compute the MAE metric.
        
        Parameters
        ----------
        inp : torch.Tensor
            Target input values.
        latent : torch.Tensor
            Latent representation (unused).
        output : torch.Tensor
            Predicted output values.
        labels : torch.Tensor
            Ground truth labels (unused).
        meta : dict
            Metadata (unused).

        Returns
        -------
        tuple
            (mae_value, empty_dict).
        """
        mae = F.l1_loss(inp, output, reduction=self.reduction)

        return mae, {}
