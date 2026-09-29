import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from collective_encoder.models.base import CEModelBase
from collective_encoder.models.modules.tica import TICAModule

_log = logging.getLogger(__name__)


class TICAModel(CEModelBase):
    """Time-lagged Independent Component Analysis (TICA) Model inheriting from :class:`CEModelBase`.

    Performs linear dimensionality reduction maximizing autocorrelation at a specified lag time tau.
    Fitted analytically on sequential training data without requiring PyTorch Lightning training loops.
    """

    _IDENTIFIER = "TICA"
    _COMPATIBLE_DATASETS = ["DEFAULT", "DISTANCES", "SOAP", "SOAP_PS", "POSITIONS"]
    _REQUIRED_ARGS = ["latent_dim", "datapoint_shape", "dataset_type"]
    _OPTIONAL_ARGS = CEModelBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "lag": 1,
        "epsilon": 1e-6,
        "center": True,
    })

    @staticmethod
    def extract_args_from_datamodule(datamodule, args: dict) -> dict:
        args["datapoint_shape"] = datamodule.get_datapoint_shape()
        args["dataset_type"] = datamodule.dataset_type
        return args

    def __init__(self, args: Dict[str, Any] = None, **kwargs) -> None:
        super().__init__(args=args, **kwargs)

        assert self.dataset_type in self._COMPATIBLE_DATASETS, (
            f"Dataset type '{self.dataset_type}' is not compatible with TICAModel. "
            f"Compatible types: {self._COMPATIBLE_DATASETS}"
        )

        self.input_dim = int(self.datapoint_shape[0])
        self.latent_dim = int(self.latent_dim)
        self.lag = int(getattr(self, "lag", 1))
        self.epsilon = float(getattr(self, "epsilon", 1e-6))
        self.center = bool(getattr(self, "center", True))

        self.encoder_net = TICAModule(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            lag=self.lag,
            epsilon=self.epsilon,
            center=self.center,
        )

    def get_norm_len(self) -> int:
        return self.datapoint_shape[0]

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.Mean.numel() != np.prod(x.shape[1:]):
            self.raise_error(
                f"Mean and Range buffers must have the same number of elements as input. "
                f"Got Mean shape: {self.Mean.shape}, Range shape: {self.Range.shape}, input shape: {x.shape}"
            )
        mean_expanded = self.Mean.view(1, *(x.shape[1:])).expand(x.shape)
        range_expanded = self.Range.view(1, *(x.shape[1:])).expand(x.shape)
        return (x - mean_expanded) / range_expanded

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.Mean.numel() != np.prod(x.shape[1:]):
            self.raise_error(
                f"Mean and Range buffers must have the same number of elements as input. "
                f"Got Mean shape: {self.Mean.shape}, Range shape: {self.Range.shape}, input shape: {x.shape}"
            )
        mean_expanded = self.Mean.view(1, *(x.shape[1:])).expand(x.shape)
        range_expanded = self.Range.view(1, *(x.shape[1:])).expand(x.shape)
        return x * range_expanded + mean_expanded

    def fit(self, datamodule=None, X: Optional[torch.Tensor] = None) -> "TICAModel":
        """Fits TICA analytically on training trajectory data."""
        if X is None:
            if datamodule is None:
                raise ValueError("Either datamodule or X must be provided to fit TICAModel.")
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
        self.log_msg(
            f"TICAModel fitted successfully with lag={self.lag}. Top eigenvalues: "
            f"{[round(float(v), 4) for v in self.encoder_net.eigenvalues[:min(5, self.latent_dim)]]}"
        )
        return self

    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        z = self.encoder_net(x)
        return z, {}

    def decoder(self, z: torch.Tensor) -> Tuple[Optional[torch.Tensor], dict]:
        x_rec = self.encoder_net.inverse(z)
        return x_rec, {}

    def forward(self, data: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor, dict]:
        data_norm = self.normalize(data)
        z, meta_enc = self.encoder(data_norm)
        meta = {
            "latent": z,
            "mu_latent": z,
            "z_sample": z,
            "eigenvalues": self.encoder_net.eigenvalues,
            "timescales": self.encoder_net.timescales,
        }
        meta.update(meta_enc)
        return None, z, meta


# Aliases
TICAEncoder = TICAModel
