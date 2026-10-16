import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from collective_encoder.models.base import CEModelBase
from collective_encoder.models.modules.fast_ica import FastICAModule

_log = logging.getLogger(__name__)


class ICAModel(CEModelBase):
    """
    FastICA Dimensionality Reduction Model inheriting from :class:`CEModelBase`.

    Performs linear independent component extraction via FastICA (Hyvärinen 1999).
    Fitted analytically on training data without requiring PyTorch Lightning training loops.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary. Required keys:
        - ``latent_dim`` (int): Number of independent components.
        - ``datapoint_shape`` (tuple): Shape of the input data.
        - ``dataset_type`` (str): Identifier for the dataset type.
        Optional keys:
        - ``fun`` (str, default 'logcosh'): The functional form of the G function.
        - ``max_iter`` (int, default 200): Maximum number of iterations for FastICA.
        - ``tol`` (float, default 1e-4): Tolerance for convergence.
        - ``whiten`` (bool, default True): Whether to apply whitening before ICA.
        - ``random_state`` (int, default 42): Seed for random number generator.
    kwargs : dict
        Additional arguments forwarded to `CEModelBase`.
    """

    _IDENTIFIER = "ICA"
    _COMPATIBLE_DATASETS = ["DEFAULT", "DISTANCES", "SOAP", "SOAP_PS", "POSITIONS"]
    _REQUIRED_ARGS = ["latent_dim", "datapoint_shape", "dataset_type"]
    _OPTIONAL_ARGS = CEModelBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "fun": "logcosh",
        "max_iter": 200,
        "tol": 1e-4,
        "whiten": True,
        "random_state": 42,
    })

    @staticmethod
    def extract_args_from_datamodule(datamodule, args: dict) -> dict:
        """
        Extract dynamically needed configuration arguments from the datamodule.

        Parameters
        ----------
        datamodule : pytorch_lightning.LightningDataModule
            The datamodule containing dataset shape info.
        args : dict
            The base configuration dict to update.

        Returns
        -------
        dict
            The updated configuration dict.
        """
        args["datapoint_shape"] = datamodule.get_datapoint_shape()
        args["dataset_type"] = datamodule.dataset_type
        return args

    def __init__(self, args: Dict[str, Any] = None, **kwargs) -> None:
        super().__init__(args=args, **kwargs)

        assert self.dataset_type in self._COMPATIBLE_DATASETS, (
            f"Dataset type '{self.dataset_type}' is not compatible with ICAModel. "
            f"Compatible types: {self._COMPATIBLE_DATASETS}"
        )

        self.input_dim = int(self.datapoint_shape[0])
        self.latent_dim = int(self.latent_dim)

        self.encoder_net = FastICAModule(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            fun=self.fun,
            max_iter=self.max_iter,
            tol=self.tol,
            whiten=self.whiten,
            random_state=self.random_state,
        )

    def get_norm_len(self) -> int:
        """
        Get the required size for normalization buffers.

        Returns
        -------
        int
            Length of the data dimension to be normalized.
        """
        return self.datapoint_shape[0]

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor using stored Mean and Range buffers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        if self.Mean.numel() != np.prod(x.shape[1:]):
            self.raise_error(
                f"Mean and Range buffers must have the same number of elements as input. "
                f"Got Mean shape: {self.Mean.shape}, Range shape: {self.Range.shape}, input shape: {x.shape}"
            )
        mean_expanded = self.Mean.view(1, *(x.shape[1:])).expand(x.shape)
        range_expanded = self.Range.view(1, *(x.shape[1:])).expand(x.shape)
        return (x - mean_expanded) / range_expanded

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize input tensor using stored Mean and Range buffers.

        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor.

        Returns
        -------
        torch.Tensor
            Denormalized tensor.
        """
        if self.Mean.numel() != np.prod(x.shape[1:]):
            self.raise_error(
                f"Mean and Range buffers must have the same number of elements as input. "
                f"Got Mean shape: {self.Mean.shape}, Range shape: {self.Range.shape}, input shape: {x.shape}"
            )
        mean_expanded = self.Mean.view(1, *(x.shape[1:])).expand(x.shape)
        range_expanded = self.Range.view(1, *(x.shape[1:])).expand(x.shape)
        return x * range_expanded + mean_expanded

    def fit(self, datamodule=None, X: Optional[torch.Tensor] = None) -> "ICAModel":
        """
        Fits FastICA analytically on training data.

        Parameters
        ----------
        datamodule : LightningDataModule, optional
            Datamodule providing train_dataloader().
        X : torch.Tensor, optional
            Direct input tensor (if datamodule is not used).

        Returns
        -------
        ICAModel
            The fitted model.
        """
        if X is None:
            if datamodule is None:
                raise ValueError("Either datamodule or X must be provided to fit ICAModel.")
            batches = []
            for batch in datamodule.train_dataloader():
                data, _ = self._batch_split(batch)
                batches.append(data.detach().cpu() if isinstance(data, torch.Tensor) else torch.tensor(data))
            X = torch.cat(batches, dim=0)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        else:
            X = X.to(dtype=torch.float32)

        if self.normIn:
            if not self.normSet:
                self.set_norm(datamodule)
            X = self.normalize(X)

        self.encoder_net.fit(X)
        self.log_msg("ICAModel fitted successfully.")
        return self

    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        z = self.encoder_net(x)
        return z, {}

    def decoder(self, z: torch.Tensor) -> Tuple[Optional[torch.Tensor], dict]:
        x_rec = self.encoder_net.inverse(z)
        return x_rec, {}

    def forward(self, data: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor, dict]:
        """
        Compute and return the forward pass output for a given batch.

        Parameters
        ----------
        data : torch.Tensor
            Batch of input data.

        Returns
        -------
        tuple
            (output_tensor, latent_tensor, metadata_dict).
        """
        data_norm = self.normalize(data)
        z, meta_enc = self.encoder(data_norm)
        meta = {
            "latent": z,
            "mu_latent": z,
            "z_sample": z,
        }
        meta.update(meta_enc)
        return None, z, meta


# Aliases
ICAEncoder = ICAModel
