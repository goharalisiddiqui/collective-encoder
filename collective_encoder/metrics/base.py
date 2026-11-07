from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from collective_encoder.common.module import CEModule


class CEMetricBase(CEModule, ABC):
    """
    Abstract base class for all evaluation metrics.

    Metrics are used for logging and evaluation, but not for backpropagation.
    """
        
    def __call__(self, *args, **kwds):
        """
        Alias for `calculate`.
        """
        return self.calculate(*args, **kwds)

    @abstractmethod
    def calculate(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the evaluation metric.

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
            - The scalar metric tensor.
            - A dictionary of individual sub-metrics for logging.
        """
        pass
