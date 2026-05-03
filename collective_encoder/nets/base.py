from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple, Union

import torch
from torch_geometric.data import Data
from torch.nn import functional as F

import pytorch_lightning as pl

from collective_encoder.common.module import CEModule


class CENetBase(pl.LightningModule, CEModule, ABC):
    _OPTIONAL_ARGS = {
        'lrate': 1e-3,
        'weight_decay': 0.0,
        'normIn': False,
        'scheduler': False,
        'output_directory': './ce_net_output/untitled_',
        'scheduler_args': None,
        'test_plotter': None,
        'export_latent': False,
        }

    """Shared PyTorch Lightning + CEModule base for all collective encoder networks.

    Provides unified optimizer configuration, LR-scheduler setup, training-start
    logging, and normalization-state management.

    Subclasses **must** implement:
        - ``forward(batch)``
        - ``step(batch, stage: str) -> torch.Tensor``

    Subclasses **may** override:
        - ``_build_optimizer()`` — change optimizer class or parameter source
        - ``_get_scheduler_args()`` — change ReduceLROnPlateau defaults
        - ``_validate_norm_sizes(Mean, Range)`` — assert buffer dimension constraints
        - ``print_hparams()`` — log architecture-specific hyperparameters

    The following hparams keys are consumed by this base class and must be
    saved via ``save_hyperparameters()`` in the concrete subclass ``__init__``:
        ``lrate``, ``weight_decay``, ``scheduler``, ``scheduler_args``
    """

    def __init__(self, args: Dict[str, Any] = None, **kwargs) -> None:
        pl.LightningModule.__init__(self)
        CEModule.__init__(self, args=args, **kwargs)

        self.register_buffer('normIn', torch.tensor(self.normIn, dtype=torch.bool))
        self.register_buffer('normSet', torch.tensor(False, dtype=torch.bool))
        
        self.losses = {
            "loss": self.loss,
        }
        self.metrics = {
            "mae": self.metric_mae,
        }  
        self.test_metrics = {
            "mae": self.metric_mae,
        }
        
    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    
    @abstractmethod
    def get_norm_len(self) -> int:
        """Return the expected length of the Mean and Range normalization buffers."""
        pass

    def set_norm(self) -> None:
        """Load feature statistics from the trainer's datamodule scaler.

        Safe to call before the trainer is attached: if no trainer is present
        yet the call is silently ignored and ``normalize()`` will retry on the
        next forward pass.
        """
        trainer = getattr(self, 'trainer', None)
        if trainer is None:
            return
        if not trainer.datamodule:
            self.raise_error("Trainer has no datamodule attached; \
                             cannot compute normalization.", RuntimeError)
        self.register_buffer('Mean', torch.zeros(self.get_norm_len()))
        self.register_buffer('Range', torch.ones(self.get_norm_len()))
        with torch.no_grad():
            dm = trainer.datamodule
            Mean = torch.tensor(dm.get_scaler_mean(), device=self.device)
            Range = torch.tensor(dm.get_scaler_scale(), device=self.device)
            Range = Range.clone()
            Range[Range == 0.0] = 1.0
            self._validate_norm_sizes(Mean, Range)
            self.log_msg("Setting normalization for inputs.")
            self.Mean = Mean
            self.Range = Range
            self.normSet = torch.tensor(True, dtype=torch.bool)
    
    def normalize(self, x: Union[torch.Tensor, Data]) -> Union[torch.Tensor, Data]:
        """Normalize input data using the stored Mean and Range buffers.

        If normalization buffers are not yet set, this method will attempt to
        set them by calling ``set_norm()``. If normalization is still not set
        after that call, an error is raised.
        """
        if not self.normIn:
            return x
        if not self.normSet:
            self.set_norm()
        return self._normalize(x)
    
    def denormalize(self, x: Union[torch.Tensor, Data]) -> Union[torch.Tensor, Data]:
        """Denormalize input data using the stored Mean and Range buffers.

        If normalization buffers are not yet set, this method will attempt to
        set them by calling ``set_norm()``. If normalization is still not set
        after that call, an error is raised.
        """
        if not self.normIn:
            return x
        if not self.normSet:
            self.set_norm()
        return self._denormalize(x)
    
    @abstractmethod
    def _normalize(self, x: Union[torch.Tensor, Data]) -> Union[torch.Tensor, Data]:
        """Normalize input data using the stored Mean and Range buffers."""
        pass

    @abstractmethod
    def _denormalize(self, x: Union[torch.Tensor, Data]) -> Union[torch.Tensor, Data]:
        """Denormalize input data using the stored Mean and Range buffers."""
        pass

    def _validate_norm_sizes(self, Mean: torch.Tensor, Range: torch.Tensor) -> None:
        """Hook for subclasses to assert normalization buffer dimensions."""
        pass

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------

    def _get_train_params(self):
        """Return an iterable of parameters to optimize over.

        Override to exclude certain parameters from optimization, or to set
        different optimization hyperparameters for different parameter subsets.
        The default is ``self.parameters()`` with no per-parameter options.
        """
        return self.parameters()
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Return the optimizer instance.

        Override to change the optimizer class or the parameter iterable.
        The default uses ``torch.optim.Adam`` over all ``self.parameters()``.
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lrate,
            weight_decay=self.hparams.weight_decay,
        )

    def _get_scheduler_args(self) -> dict:
        """Return ``ReduceLROnPlateau`` kwargs, merging defaults with hparam overrides.

        Override in subclasses to change defaults; the user-supplied
        ``scheduler_args`` dict (from hparams) is always applied on top.
        """
        defaults: dict = {
            "mode": "min",
            "factor": 0.8,
            "patience": 3,
            "min_lr": 1e-10,
            "cooldown": 10,
        }
        defaults.update(self.hparams.scheduler_args or {})
        return defaults

    def configure_optimizers(self):
        optimizer = self._build_optimizer()
        if not self.hparams.scheduler:
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
    # Training lifecycle
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        self.log_msg("==================================")
        self.log_msg(f"Starting training {type(self).__name__} module")
        self.log_msg("==================================")
        self.log_msg("[Optimization Settings]")
        self.log_msg(f"  Learning rate      = {self.hparams.lrate}")
        self.log_msg(f"  l2 regularization  = {self.hparams.weight_decay}")
        if self.hparams.scheduler:
            extra = self.hparams.scheduler_args or {}
            self.log_msg(f"  LR scheduler       = Enabled {extra}")
        else:
            self.log_msg("  LR scheduler       = Disabled")
        self.log_msg("[Hyperparameters]")
        self.print_hparams()
        self.log_msg("==================================")

    def print_hparams(self) -> None:
        """Override to log architecture-specific hyperparameters at training start."""
        pass

    # ------------------------------------------------------------------
    # Step delegation — satisfies the PyTorch Lightning contract
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self.forward(batch)
    
    def _plot_test(self, inp, latent, output, labels, meta) -> None:
        if self.test_plotter is None:
            return
        from collective_encoder.testplotters.resolver import get_testplotter
        plotter_cls = get_testplotter(self.test_plotter)
        plotter = plotter_cls(self.output_directory, logger=self.logger)
        plotter.plot(inp, latent, output, labels, meta)
    
    def _multiple_calculate(self, 
                            data: Union[torch.Tensor, Data],
                            latent: torch.Tensor,
                            pred: torch.Tensor,
                            meta: Dict[str, Any],
                            funcs: Dict[str, Callable],
                            stage: str,
                            batch_size: int = None,
                            ) -> dict:
        results = {}
        for name, func in funcs.items():
            result, result_meta = func(data, latent, pred, meta)
            results[name] = result
            if isinstance(result, (int, float)) or (isinstance(result, torch.Tensor) and result.numel() == 1):
                self.log(f"{stage}_{name}", result.detach(), prog_bar=(stage == "train"),
                         on_step=(stage == "train"), on_epoch=True, batch_size=batch_size)
            for key, value in result_meta.items():
                if isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                    self.log(f"{stage}_{key}_{name}", value,
                             prog_bar=False, on_step=(stage == "train"), on_epoch=True, batch_size=batch_size)
            meta.update(result_meta)
        return results

    def _step(self, batch, stage: str) -> torch.Tensor:
        data = batch[0] if isinstance(batch, (list, tuple)) else batch
        latent, pred, meta = self(data)
        batch_size = self.trainer.datamodule.hparams.batch_size \
            if self.trainer and self.trainer.datamodule else None

        with torch.no_grad():
            metrics = self.metrics if stage in ["train", "val"] else self.test_metrics
            metrics = self._multiple_calculate(data, latent, pred, meta, 
                                           metrics, stage, batch_size)
    
        if stage == "test":
            self._plot_test(data, latent, pred, batch[1] if len(batch) > 1 else None, meta)
            return metrics.get("mae", torch.tensor(0.0))

        losses = self._multiple_calculate(data, latent, pred, meta, 
                                          self.losses, stage, batch_size)
        losses = self.extra_training_step(data, latent, pred, meta, losses)
        loss = self.aggregate_losses(losses)
        self.log(f"{stage}_loss", loss.detach(), prog_bar=(stage == "train"),
                 on_step=(stage == "train"), on_epoch=True, batch_size=batch_size)
        return loss
    
    # ------------------------------------------------------------------
    # Encoder and decoder delegation
    # ------------------------------------------------------------------
    
    def _encode(self, data):
        data = self.normalize(data)
        z, meta = self.encoder(data)
        return z, meta

    def _decode(self, z: torch.Tensor):
        out, meta = self.decoder(z)
        out = self.denormalize(out)
        return out, meta
    
    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        z = self.encoder_net(x)
        return z, {}

    def decoder(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x_out = self.decoder_net(z)
        return x_out, {}

    def latent_to_decoder_input(self, latent) -> Tuple:
        return latent, {}

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute and return the forward pass output for a given batch."""
        meta = {}

        latent, meta_latent = self._encode(data)
        latent, meta_sample = self.latent_to_decoder_input(latent)
        pred, meta_dec = self._decode(latent)
        
        meta.update(meta_latent)
        meta.update(meta_dec)
        meta.update(meta_sample)

        return pred, latent, meta
    
    # ------------------------------------------------------------------
    # Losses and metrics
    # ------------------------------------------------------------------

    def loss(self, inp, latent, output, labels, meta):
        loss = F.mse_loss(output, labels, reduction='none')
        loss = torch.mean(loss)
        return loss

    def metric_mae(self, inp, latent, output, labels, meta):
        mae = F.l1_loss(output, labels, reduction='none')
        mae = torch.mean(mae)
        return mae, {}

    def aggregate_losses(self, losses: dict) -> torch.Tensor:
        return torch.sum(torch.stack(list(losses.values())))


    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_latent(self, data):
        return self.encode(data)

    def get_decoded(self, latent):
        return self.decode(latent)
    


