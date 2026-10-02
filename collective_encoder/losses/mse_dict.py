from typing import Any, Dict, Tuple

import torch
from torch.distributions.normal import Normal

from .base import CELossBase


class CELossMSEDict(CELossBase):
    """
    MSE loss tailored for dictionary-based outputs/labels.

    Computes MSE across multiple named components and aggregates them.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary. Required keys:
        - ``keys`` (list of str): Keys identifying the components in output/labels dictionaries.
        Optional keys:
        - ``weights`` (list of float, optional): Weighting factors for each key.
        - ``reduction`` (str, default 'mean'): Specifies the reduction for the MSE.
        - ``accumulation`` (str, default 'sum'): Aggregation method ('sum' or 'mean').
    kwargs : dict
        Additional keyword arguments.
    """
    _IDENTIFIER = "CELossMSEDict"
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
                output: dict,
                labels: dict,
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the dictionary MSE loss.
        
        Parameters
        ----------
        inp : torch.Tensor
            Original input (unused).
        latent : torch.Tensor
            Latent encoding (unused).
        output : dict
            Dictionary of predicted tensors.
        labels : dict
            Dictionary of target tensors.
        meta : dict
            Metadata (unused).

        Returns
        -------
        tuple
            (aggregated_loss, dict_of_individual_losses).
        """
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

