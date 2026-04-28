import numpy as np
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from collective_encoder.nets.dvae_net import DVAE


class _EmbedProxy:
    """Lightweight wrapper that reports the post-embedding datapoint shape to VAE.__init__.

    VAE.__init__ calls ``datamodule.get_datapoint_shape()`` to prepend the input
    dimension to the layer list.  For "flatten" embedding the true input to the
    network is ``prod(raw_shape)``, not ``raw_shape[0]``.  This proxy overrides
    that single method while delegating every other attribute access to the real
    datamodule so the rest of VAE.__init__ works unchanged.
    """

    def __init__(self, datamodule, embedded_shape: Tuple[int, ...]) -> None:
        self._datamodule = datamodule
        self._embedded_shape = embedded_shape

    def get_datapoint_shape(self) -> Tuple[int, ...]:
        return self._embedded_shape

    def __getattr__(self, name: str):
        return getattr(self._datamodule, name)


class EDVAE(DVAE):
    def __init__(self,
                 datamodule,
                 network: List[int],
                 embedding_type: str = "flatten",
                 normIn: Optional[bool] = False,
                 lrate: float = 0.01,
                 weight_decay: float = 1e-7,
                 scheduler: bool = True,
                 scheduler_args: dict = None,
                 outname: str = './EDVAE_untitled/EDVAE_',
                 test_plotter: str = "LDplotter",
                 export_latent: bool = False,
                 beta: float = 1.0,
                 batch_norm: bool = True,
                 C_reg: Optional[Tuple[float, int, int]] = None,
                 C_auto: bool = False,
                 use_steric_loss: bool = False,
                 use_bond_deviation_loss: bool = False,
                 ):
        self.save_hyperparameters(ignore=['datamodule'])

        # Store the raw shape before the proxy changes what super() sees
        raw_shape = datamodule.get_datapoint_shape()
        self._raw_datapoint_shape = raw_shape

        # Build a proxy that reports the embedded input dimension to VAE.__init__
        if embedding_type == "flatten":
            embedded_length = int(np.prod(raw_shape))
            proxy = _EmbedProxy(datamodule, (embedded_length,))
        else:
            raise ValueError(f"Unknown embedding_type: '{embedding_type}'. Supported: 'flatten'")

        super().__init__(
            datamodule=proxy,
            network=network,
            normIn=normIn,
            lrate=lrate,
            weight_decay=weight_decay,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
            outname=outname,
            test_plotter=test_plotter,
            export_latent=export_latent,
            beta=beta,
            batch_norm=batch_norm,
            C_reg=C_reg,
            C_auto=C_auto,
            use_steric_loss=use_steric_loss,
            use_bond_deviation_loss=use_bond_deviation_loss,
        )

    def init_network(self) -> None:
        self.log_msg(f"[Initializing {type(self).__name__} Module] hidden layers: {self.network}, "
                  f"embedding: {self.hparams.embedding_type}")

        raw_shape = self._raw_datapoint_shape

        if self.hparams.embedding_type == "flatten":
            self.embedding = nn.Flatten()
            embedded_length = int(np.prod(raw_shape))
            self.log_msg(f"  {raw_shape} --> {embedded_length} (flatten embedding)")

        super().init_network()

        if self.hparams.embedding_type == "flatten":
            self.deembedding = nn.Unflatten(1, raw_shape)
            self.log_msg(f"  {embedded_length} --> {raw_shape} (unflatten deembedding)")

    def set_norm(self) -> None:
        """Override to flatten Mean/Range buffers for multi-dim input."""
        super().set_norm()
        if self.hparams.embedding_type == "flatten":
            self.Mean = self.Mean.flatten()
            self.Range = self.Range.flatten()

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x_out = super().forward(x)

        if self.metaD:
            return x_out

        x_out, meta = x_out
        x_out = self.deembedding(x_out)
        return x_out, meta

    def get_latent(self, data_x: torch.Tensor):
        data_x = self.embedding(data_x)
        data_x = self.normalize(data_x)
        latent_mu, latent_logvar = self.encoder_net(data_x)
        return latent_mu.detach().cpu().numpy(), latent_logvar.detach().cpu().numpy()

    def decode_latent(self, latent: torch.Tensor, keeptensor: bool = False):
        pred, _ = self.decoder(latent)
        pred = self.denormalize(pred)
        pred = self.deembedding(pred)
        if not keeptensor:
            pred = pred.detach().cpu().numpy()
        return pred
