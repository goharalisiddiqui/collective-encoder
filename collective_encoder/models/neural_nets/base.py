from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch_geometric.data import Data

from collective_encoder.models.base import CEModelBase
from collective_encoder.losses.mse import CELossMSE
from collective_encoder.metrics.mae import CEMetricMAE


class CENetBase(CEModelBase, pl.LightningModule, ABC):
    """Shared PyTorch Lightning + CEModelBase base for all neural network models.

    Provides unified optimizer configuration, LR-scheduler setup, training-start
    logging, gradient-based training/validation/testing steps, and loss aggregation.

    Subclasses must implement:
        - ``get_norm_len() -> int``
        - ``_normalize(x)``
        - ``_denormalize(x)``

    Subclasses may override:
        - ``_build_optimizer()``
        - ``_get_scheduler_args()``
        - ``_validate_norm_sizes(Mean, Range)``
        - ``print_hparams()``
    """

    _OPTIONAL_ARGS = CEModelBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "lrate": 1e-3,
        "weight_decay": 0.0,
        "scheduler": False,
        "scheduler_args": None,
    })

    def __init__(self, args: Dict[str, Any] = None, **kwargs) -> None:
        pl.LightningModule.__init__(self)
        CEModelBase.__init__(self, args=args, **kwargs)

        self.losses = {
            "loss": CELossMSE({}, **kwargs),
        }
        self.metrics = {
            "mae": CEMetricMAE({}, **kwargs),
        }

    # ------------------------------------------------------------------
    # Optimizer and Scheduler Configuration
    # ------------------------------------------------------------------

    def _get_train_params(self):
        prams_from_losses = []
        for loss in self.losses.values():
            prams_from_losses.extend(loss.parameters())
        return list(self.parameters()) + prams_from_losses

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self._get_train_params(),
            lr=self.lrate,
            weight_decay=self.weight_decay,
        )

    def _get_scheduler_args(self) -> dict:
        defaults: dict = {
            "mode": "min",
            "factor": 0.8,
            "patience": 3,
            "min_lr": 1e-10,
            "cooldown": 10,
        }
        defaults.update(self.scheduler_args or {})
        return defaults

    def configure_optimizers(self):
        optimizer = self._build_optimizer()
        if not self.scheduler:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self._get_scheduler_args()
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Training Lifecycle
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        self.log_msg("==================================")
        self.log_msg(f"Starting training {type(self).__name__} module")
        self.log_msg("==================================")
        self.log_msg("[Optimization Settings]")
        self.log_msg(f"  Learning rate      = {self.lrate}")
        self.log_msg(f"  l2 regularization  = {self.weight_decay}")
        self.log_msg(f"  norm_in            = {self.normIn}")
        self.log_msg(f"  export_latent      = {self.export_latent}")
        self.log_msg(f"  output_directory   = {self.output_directory}")
        if self.scheduler:
            extra = self.scheduler_args or {}
            self.log_msg(f"  LR scheduler       = Enabled {extra}")
        else:
            self.log_msg("  LR scheduler       = Disabled")
        self.log_msg("[Hyperparameters]")
        self.print_hparams()
        self.log_msg("==================================")

    def print_hparams(self) -> None:
        pass

    # ------------------------------------------------------------------
    # PyTorch Lightning Step Hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self.forward(batch)

    def extra_training_step(self, inp, latent, output, labels, meta, losses):
        return losses

    def _step(self, batch, stage: str) -> torch.Tensor:
        data, labels = self._batch_split(batch)
        output, latent, meta = self(data)
        data = self.normalize(data)
        batch_size = self.trainer.datamodule.batch_size \
            if self.trainer and self.trainer.datamodule else None

        with torch.no_grad():
            metrics = self.metrics if stage in ["train", "val"] else self.test_metrics
            metrics = self._multiple_calculate(data, latent, output, labels, meta, 
                                        metrics, stage, batch_size)

        if stage == "test":
            if len(self.test_plotters) > 0:
                self._plot_test_batch(data, latent, output, labels, meta)
            return metrics.get("mae", torch.tensor(0.0, device=self.device))

        losses = self._multiple_calculate(data, latent, output, labels, meta, 
                                        self.losses, stage, batch_size)
        losses = self.extra_training_step(data, latent, output, labels, meta, losses)
        loss = self.aggregate_losses(losses)
        self.log(f"{stage}_loss", loss.detach(), prog_bar=(stage == "train"),
                on_step=(stage == "train"), on_epoch=True, batch_size=batch_size)
        return loss

    def on_test_start(self):
        self._plot_test_start()

    def on_test_end(self):
        self._plot_test_finish()

    def aggregate_losses(self, losses: dict) -> torch.Tensor:
        return torch.sum(torch.stack(list(losses.values())))
