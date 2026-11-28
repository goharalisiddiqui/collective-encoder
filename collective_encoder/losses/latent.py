from typing import Dict, Tuple

import torch

from .base import CELossBase


class CELossLatent(CELossBase):
    """
    Latent space constraint loss for sequential data.

    Encourages sequential points in the latent space to be close to each other
    (minimizes mean step distance) and spaced equidistantly (minimizes variance
    of step distances).

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary (unused).
    kwargs : dict
        Additional keyword arguments.
    """
    _IDENTIFIER = "CELossLatent"
    
    def forward(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the sequential latent loss.

        Parameters
        ----------
        inp : torch.Tensor
            Network input (unused).
        latent : torch.Tensor
            Latent space encodings of shape ``(batch_size, latent_dim)``.
        output : torch.Tensor
            Network output (unused).
        labels : torch.Tensor
            Ground truth labels (unused).
        meta : dict
            Metadata dictionary (unused).

        Returns
        -------
        tuple
            (latent_loss, empty_dict).
        """
        loss_latent = torch.tensor(0.0, device=latent.device)
        if latent.size(0) > 1:
            batch_dist = torch.norm(latent[1:] - latent[:-1], dim=1)
            loss_latent = torch.mean(batch_dist)
        if latent.size(0) > 2:
            loss_latent = loss_latent + torch.var(batch_dist)
        return loss_latent, {}
