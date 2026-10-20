from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from collective_encoder.common.module import CEModule


class CELossBase(torch.nn.Module, CEModule, ABC):
    """
    Abstract base class for all loss functions in the Collective Encoder framework.

    Inherits from `torch.nn.Module` for standard PyTorch gradients and
    from `CEModule` for configuration handling.

    Parameters
    ----------
    args : dict, optional
        Dictionary of configuration options.
    kwargs : dict
        Additional keyword arguments forwarded to the parent modules.
    """

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
        """
        Compute the loss.

        Parameters
        ----------
        inp : torch.Tensor
            The original input data provided to the network.
        latent : torch.Tensor
            The latent space representation produced by the encoder.
        output : torch.Tensor
            The reconstruction produced by the decoder.
        labels : torch.Tensor
            Ground truth targets or labels from the dataset.
        meta : dict of str to torch.Tensor
            Additional metadata or auxiliary outputs from the network.

        Returns
        -------
        tuple
            A tuple containing:
            - The scalar total loss tensor for backpropagation.
            - A dictionary of individual loss components/metrics for logging.
        """
        pass
