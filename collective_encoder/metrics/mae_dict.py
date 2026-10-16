from typing import Dict, Tuple

import torch
from torch.nn import functional as F

from .base import CEMetricBase


class CEMetricMAEDict(CEMetricBase):
    """
    MAE metric tailored for dictionary-based outputs/labels.

    Computes MAE across multiple named components and aggregates them.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary. Required keys:
        - ``keys`` (list of str): Keys identifying the components in output/labels.
        Optional keys:
        - ``reduction`` (str, default 'mean'): Specifies the reduction for individual MAEs.
        - ``accumulation`` (str, default 'sum'): Aggregation method ('sum' or 'mean').
    kwargs : dict
        Additional keyword arguments.
    """
    _IDENTIFIER = "CEMetricMAEDict"
    _REQUIRED_ARGS = ['keys']
    _OPTIONAL_ARGS = {
        'reduction': 'mean',
        'accumulation': 'sum',
    }
    
    def calculate(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: dict,
                labels: dict,
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the dictionary MAE metric.
        
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
            (aggregated_mae, dict_of_individual_maes).
        """
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

