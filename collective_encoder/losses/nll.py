from typing import Any, Dict, Tuple

import torch
from torch.distributions.normal import Normal

from .base import CELossBase

EPSILON = 1e-7

class CELossNLL(CELossBase):
    """
    Negative Log-Likelihood (NLL) reconstruction loss.

    Assumes the network outputs parameters (mu, logvar) of a normal distribution
    and computes the negative log-probability of the input data under that distribution.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary. Optional keys:
        - ``mu_name`` (str, default 'mu_x'): Key in `meta` for the distribution mean.
        - ``logvar_name`` (str, default 'logvar_x'): Key in `meta` for the distribution log variance.
    kwargs : dict
        Additional keyword arguments.
    """
    _IDENTIFIER = "CELossNLL"
    _OPTIONAL_ARGS = {
        'mu_name': 'mu_x',
        'logvar_name': 'logvar_x',
    }
    
    def forward(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the NLL loss.
        
        Parameters
        ----------
        inp : torch.Tensor
            Target input values (the observed data).
        latent : torch.Tensor
            Latent encoding (unused).
        output : torch.Tensor
            Deterministic reconstruction (unused, distribution parameters in `meta`).
        labels : torch.Tensor
            Ground truth labels (unused).
        meta : dict
            Metadata containing the ``mu`` and ``logvar`` predictions.

        Returns
        -------
        tuple
            (nll_loss, empty_dict).
        """
        mu_x = meta[self.mu_name]
        logvar_x = meta[self.logvar_name]

        logvar_x = torch.clamp(logvar_x, min=-4.0, max=4.0) # Clamp log-variance to prevent numerical instability in exp/log operations
        sd = torch.exp(0.5 * logvar_x) + EPSILON
        p_x = Normal(mu_x, sd)
        loss_rec = -torch.sum(p_x.log_prob(inp), dim=1)

        loss_rec = torch.mean(loss_rec)

        return loss_rec, {}
