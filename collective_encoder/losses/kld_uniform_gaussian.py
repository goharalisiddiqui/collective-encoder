from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from .base import CELossBase
from .kld_schedulers.resolver import KLDResolver

EPSILON = 1e-7


class CELossKLDUniformGaussian(CELossBase):
    """
    Standard KLD loss between a Gaussian encoder and a standard uniform Gaussian prior.

    Includes a capacity scheduler (beta-VAE type) to constrain the KLD penalty.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary. Optional keys:
        - ``mu_name`` (str, default 'mu_latent'): Key for latent mean in `meta`.
        - ``logvar_name`` (str, default 'logvar_latent'): Key for latent logvar in `meta`.
        - ``kld_max_type`` (str, default 'Fixed'): Type of KLD scheduling.
        - ``kld_max_scheduler_args`` (dict, optional): Arguments for the scheduler.
    kwargs : dict
        Additional keyword arguments.
    """
    _IDENTIFIER = "CELossKLDUniformGaussian"
    _OPTIONAL_ARGS = {
        'mu_name': 'mu_latent',
        'logvar_name': 'logvar_latent',
        "kld_max_type": 'Fixed',
        "kld_max_scheduler_args": None,
    }
    
    def __init__(self, 
                args: Dict[str, Any] = None, 
                **kwargs) -> None:
        super().__init__(args, **kwargs)
        
        self.kld_scheduler = KLDResolver(self.kld_max_type, 
                                        self.kld_max_scheduler_args,
                                        **kwargs)
    
    def on_validation_epoch_end(self, plmodule):
        """
        Hook called at the end of a validation epoch to update the KLD capacity scheduler.

        Parameters
        ----------
        plmodule : pytorch_lightning.LightningModule
            The parent Lightning module.
        """
        self.kld_scheduler.on_validation_epoch_end(plmodule)
    
    def kld(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> torch.Tensor:
        """
        Compute the analytical KLD between the encoded Gaussian and a standard N(0, I) prior.

        Parameters
        ----------
        inp : torch.Tensor
            Network input (unused).
        latent : torch.Tensor
            Sampled latent variables (unused here, purely analytical).
        output : torch.Tensor
            Network output (unused).
        labels : torch.Tensor
            Ground truth labels (unused).
        meta : dict
            Metadata containing ``mu_latent`` and ``logvar_latent``.

        Returns
        -------
        torch.Tensor
            A 1D tensor of KLD values for each batch sample.
        """
        mu = meta[self.mu_name]
        logvar = meta[self.logvar_name]
        # KLD between univariate gaussian to Standard, explanation here:
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # Second Gaussian is zero mean and variance of 1, the prior on z
        # sum for all the latent variables
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)
        return kld

    def forward(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the total regularized KLD loss.
        
        Applies a constraint via the KLD capacity scheduler. If the mean KLD
        is below the scheduler's max value, the loss penalty is zeroed out.

        Parameters
        ----------
        inp : torch.Tensor
            Network input.
        latent : torch.Tensor
            Latent representation.
        output : torch.Tensor
            Network output.
        labels : torch.Tensor
            Ground truth labels.
        meta : dict
            Metadata dictionary.

        Returns
        -------
        tuple
            (regularized_kld_loss, metric_dict).
        """
        loss_kld = self.kld(inp, latent, output, labels, meta)
        loss_reg = torch.mean(loss_kld, dim=0)
        
        meta = {"kld" : loss_reg}
        kld_max = self.kld_scheduler.get_kld_max(self)
        if loss_reg <= kld_max:
            loss_reg *= 0.0
        
        return loss_reg, meta
