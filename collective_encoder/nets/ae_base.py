from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict

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
    """Base class for dense-tensor autoencoder architectures (VAE, AE, DVAE, EDVAE).

    Handles dense-tensor normalization, reparametrization, the generic training
    step, and metric/plotter dispatch.  Graph-based networks use
    ``BondGraphEncoderDecoder`` instead.

    Subclasses must implement ``encoder(x)`` and ``decoder(z)``; they call
    ``save_hyperparameters()`` before ``super().__init__()``.
    """

    def __init__(self,
                 dim_data: int,
                 dim_latent: int,
                 normIn: bool = False,
                 lrate: float = 0.01,
                 weight_decay: float = 1e-7,
                 scheduler: bool = False,
                 scheduler_args: dict = None,
                 outname: str = './untitled/untitled_',
                 test_plotter: str = None,
                 export_latent: bool = False,
                 ):
        super().__init__()

        self.dim_data = dim_data
        self.dim_latent = dim_latent

        self.losses = {
            "rec_loss": self.loss_mse,
        }
        self.test_metrics = {
            "mae": self.metric_mae,
        }
        self.val_metrics = {
            "mae": self.metric_mae,
        }

        self.metaD = False
        self.register_buffer('normIn', torch.tensor(normIn, dtype=torch.bool))
        self.register_buffer('normSet', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('Mean', torch.zeros(dim_data))
        self.register_buffer('Range', torch.ones(dim_data))
        self.metatomic_model_cls = MetatomicModelAE

    # ------------------------------------------------------------------
    # Optimizer hook — uses get_train_parameters() instead of all params
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.get_train_parameters(),
            lr=self.hparams.lrate,
            weight_decay=self.hparams.weight_decay,
        )

    # ------------------------------------------------------------------
    # Dense-tensor normalization
    # ------------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normIn:
            return x
        elif not self.normSet:
            self.set_norm()
        mean_expanded = self.Mean.view(1, -1).expand_as(x)
        range_expanded = self.Range.view(1, -1).expand_as(x)
        return (x - mean_expanded) / range_expanded

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normIn:
            return x
        if not self.normSet:
            self.set_norm()
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

    def get_train_parameters(self):
        return self.parameters()

    @abstractmethod
    def encoder(self, x: torch.Tensor):
        raise NotImplementedError("Subclass must implement encoder()")

    @abstractmethod
    def decoder(self, z: torch.Tensor):
        raise NotImplementedError("Subclass must implement decoder()")

    def latent_to_decoder_input(self, latent) -> Tuple:
        return latent, {}

    def get_metad_output(self,
                         latent: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                         meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        return latent

    def extra_training_step(self, data, latent, result, meta, losses):
        return losses

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict]]:
        meta = {}
        x = self.normalize(x)
        latent, meta_latent = self.encoder(x)

        if self.metaD:
            meta.update(meta_latent)
            return self.get_metad_output(latent, meta)

        latent, meta_sample = self.latent_to_decoder_input(latent)

        pred, meta_dec = self.decoder(latent)
        pred = self.denormalize(pred)

        meta.update(meta_latent)
        meta.update(meta_dec)
        meta.update(meta_sample)

        return latent, pred, meta

    # ------------------------------------------------------------------
    # Losses and metrics
    # ------------------------------------------------------------------

    def loss_mse(self, x, latent, pred, meta):
        loss = F.mse_loss(pred, x, reduction='none')
        loss = torch.mean(loss)
        mae = F.l1_loss(pred, x, reduction='none')
        mae = torch.mean(mae)
        return loss, {"mae": mae.item()}

    def aggregate_losses(self, losses: dict) -> torch.Tensor:
        return torch.sum(torch.stack(list(losses.values())))

    def metric_mae(self, x, latent, pred, meta):
        mae = F.l1_loss(pred, x, reduction='none')
        mae = torch.mean(mae)
        return mae, {}

    # ------------------------------------------------------------------
    # Generic step (used by training_step and validation_step from CENetBase)
    # ------------------------------------------------------------------

    def step(self, batch, stage: str) -> torch.Tensor:
        data = batch[0]
        latent, pred, meta = self(data)
        batch_size = self.trainer.datamodule.hparams.batch_size if self.trainer and self.trainer.datamodule else None

        losses = {}
        for loss_name, loss_func in self.losses.items():
            loss, loss_meta = loss_func(data, latent, pred, meta)
            self.log(f"{stage}_{loss_name}", loss.detach(), prog_bar=(stage == "train"),
                     on_step=(stage == "train"), on_epoch=True, batch_size=batch_size)
            losses[loss_name] = loss
            meta.update(loss_meta)
        losses = self.extra_training_step(data, latent, pred, meta, losses)

        if stage == "val":
            for metric_name, metric_func in self.val_metrics.items():
                metric, metric_meta = metric_func(data, latent, pred, meta)
                self.log(f"val_{metric_name}", metric.detach(),
                         prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
                for key, value in metric_meta.items():
                    if isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                        self.log(f"val_{key}", value,
                                 prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
                meta.update(metric_meta)

        loss = self.aggregate_losses(losses)
        self.log(f"{stage}_loss", loss.detach(), prog_bar=(stage == "train"),
                 on_step=(stage == "train"), on_epoch=True, batch_size=batch_size)
        for key, value in meta.items():
            if isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                self.log(f"{stage}_{key}", value,
                         prog_bar=False, on_step=(stage == "train"), on_epoch=True, batch_size=batch_size)
        return loss

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
            self.plotter(data, latent, pred, labels, meta)
        if self.hparams.export_latent:
            self.export_latent(latent, labels)
        return meta['mae'] if 'mae' in meta else torch.tensor(0.0)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def plotter(self, data, latent, pred, labels, meta) -> None:
        if self.hparams.test_plotter is None:
            return
        if self.hparams.test_plotter == "LDplotter":
            from collective_encoder.plotters.latent_space_plotter import LDplotter
            labels_names = self.trainer.datamodule.label_list
            assert len(labels_names) == labels.shape[1], \
                f"Labels names and labels do not match. {len(labels_names)} != {labels.shape[1]}"
            labels_dict = {labels_names[i]: labels[:, i] for i in range(labels.shape[1])}
            LDplotter(data, latent, pred, labels_dict, meta,
                      logger=self.logger.experiment, outstem=self.hparams.outname)
        else:
            raise ValueError(f"Unknown plotter: {self.hparams.test_plotter}")

    def plot_extra(self, data_x, data_y, latents) -> None:
        return

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
