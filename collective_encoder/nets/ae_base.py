from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, Dict

import torch
import torch.nn.functional as F

from collective_encoder.nets.base import CENetBase


class MetatomicModelAE(torch.nn.Module):
    def __init__(self,
                 encoder: torch.nn.Module,
                 normIn: bool = False,
                 dmean: torch.Tensor = torch.zeros(1),
                 drange: torch.Tensor = torch.ones(1),
                 ):
        super().__init__()
        self.encoder = encoder

        self.register_buffer('normIn', torch.tensor(normIn, dtype=torch.bool))
        self.register_buffer('Mean', dmean)
        self.register_buffer('Range', drange)

    def normalize(self, x: torch.Tensor):
        if not self.normIn:
            return x
        mean_expanded = self.Mean.view(1, -1).expand_as(x)
        range_expanded = self.Range.view(1, -1).expand_as(x)
        return (x - mean_expanded) / range_expanded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        latent, _ = self.encoder(x)
        mean, logvar = latent
        return mean


class AEBase(CENetBase, ABC):
    _REQUIRED_ARGS = ['dim_data', 'dim_latent']
    
    """Base class for dense-tensor autoencoder architectures (VAE, AE, DVAE, EDVAE).

    Handles dense-tensor normalization, reparametrization, the generic training
    step, and metric/plotter dispatch.  Graph-based networks use
    ``BondGraphEncoderDecoder`` instead.

    Subclasses must implement ``encoder(x)`` and ``decoder(z)``; they call
    ``save_hyperparameters()`` before ``super().__init__()``.
    """

    def __init__(self,
                 args: Dict[str, Any] = None,
                 **kwargs
                 ):
        super().__init__(args=args, **kwargs)
        self.metatomic_model_cls = MetatomicModelAE

    # ------------------------------------------------------------------
    # Dense-tensor normalization
    # ------------------------------------------------------------------
    
    def get_norm_len(self) -> int:
        return self.dim_data

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean_expanded = self.Mean.view(1, -1).expand_as(x)
        range_expanded = self.Range.view(1, -1).expand_as(x)
        return (x - mean_expanded) / range_expanded

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean_expanded = self.Mean.view(1, -1).expand_as(x)
        range_expanded = self.Range.view(1, -1).expand_as(x)
        return x * range_expanded + mean_expanded

    # ------------------------------------------------------------------
    # Reparametrization
    # ------------------------------------------------------------------

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reparametrize_multivariate(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        dist = torch.distributions.MultivariateNormal(torch.zeros(mu.shape[1]), torch.eye(mu.shape[1]))
        samples = dist.rsample(mu.shape[:-1]).to(mu.device)
        return mu + samples * std

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def latent_to_decoder_input(self, latent) -> Tuple:
        return latent, {}

    def get_metad_output(self,
                         latent: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                         meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        return latent

    def extra_training_step(self, data, latent, result, meta, losses):
        return losses

    # ------------------------------------------------------------------
    # test_step is overridden here: it handles labels, plotter, export
    # ------------------------------------------------------------------

    def test_step(self, test_batch, batch_idx) -> torch.Tensor:
        data = test_batch[0]
        labels = test_batch[1] if len(test_batch) > 1 else None
        latent, pred, meta = self(data)
        batch_size = self.trainer.datamodule.hparams.test_batch_size if self.trainer and self.trainer.datamodule else None

        for metric_name, metric_func in self.test_metrics.items():
            metric, metric_meta = metric_func(data, latent, pred, meta)
            self.log(f"test_{metric_name}", metric, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
            for key, value in metric_meta.items():
                if isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                    self.log(f"test_{key}", value, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
            meta.update(metric_meta)

        if self.hparams.test_plotter is not None:
            self._plot_test(data, latent, pred, labels, meta)
        if self.hparams.export_latent:
            self.export_latent(latent, labels)
        return meta['mae'] if 'mae' in meta else torch.tensor(0.0)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    

    def get_latent(self, data_x: torch.Tensor) -> torch.Tensor:
        data_x = data_x.float()
        data_x = self.normalize(data_x)
        latent, meta_latent = self.encoder(data_x)
        return latent

    def get_latent_names(self) -> str:
        return "latent"

    def export_latent(self, latent: torch.Tensor, labels: torch.Tensor = None) -> None:
        raise NotImplementedError("export_latent() not implemented for this model")

    def get_metatomic_model(self):
        raise NotImplementedError("get_metatomic_model() not implemented for this model")
