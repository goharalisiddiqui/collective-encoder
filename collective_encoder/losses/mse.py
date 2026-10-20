from typing import Any, Dict, Tuple

import torch
from torch.distributions.normal import Normal

from .base import CELossBase

EPSILON = 1e-7

class CELossMSE(CELossBase):
    """
    Standard Mean Squared Error (MSE) reconstruction loss.

    Computes the MSE between the network output and the original input.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary. Optional keys:
        - ``reduction`` (str, default 'mean'): Specifies the reduction to apply to the output.
    kwargs : dict
        Additional keyword arguments.
    """
    _IDENTIFIER = "CELossMSE"
    _OPTIONAL_ARGS = {
        'reduction': 'mean',
    }
    
    def __init__(self, 
                args: Dict[str, Any] = None, 
                **kwargs) -> None:
        super().__init__(args, **kwargs)
        self.mse = torch.nn.MSELoss(reduction=self.reduction)

    def forward(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the MSE loss.
        
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
            (mse_loss, empty_dict).
        """
        loss = self.mse(output, inp)

        return loss, {}
