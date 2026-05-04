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
        
        normIn = self.normIn
        delattr(self, 'normIn') # We need to delete this to create it as a buffer.
        self.register_buffer('normIn', torch.tensor(normIn, dtype=torch.bool))
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
        self.test_plotters = {}
        
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
            lr=self.lrate,
            weight_decay=self.weight_decay,
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
    # Training lifecycle
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
        self.log_msg(f"  ouptut_directory     = {self.output_directory}")
        if self.scheduler:
            extra = self.scheduler_args or {}
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
        from collective_encoder.testplotters.resolver import get_testplotter
        
        label_names = self.trainer.datamodule.get_label_names()
        if isinstance(labels, torch.Tensor):
            if len(label_names) != labels.shape[1]:
                self.raise_error(f"Number of label names ({len(label_names)}) "
                                 f"does not match number of label columns ({labels.shape[1]}).")
            labels_dict = {name: labels[:, i] for i, name in enumerate(label_names)}
        elif isinstance(labels, dict):
            labels_dict = labels
        else:
            self.raise_error(f"Unexpected labels type: {type(labels)}")
        for name, args in self.test_plotters.items():
            try:
                plotter_cls = get_testplotter(name)
                plotter_args = args
                plotter_args['run_directory'] = self.output_directory
                plotter_args['logger'] = self.logger
                plotter = plotter_cls(plotter_args, **self.get_run_args())
                plotter.plot(inp, latent, output, labels_dict, meta)
            except Exception as e:
                self.log_exception(f"Test plotting failed", e)
        
    def _multiple_calculate(self, 
                            inp: Union[torch.Tensor, Data],
                            latent: torch.Tensor,
                            output: torch.Tensor,
                            labels: torch.Tensor,
                            meta: Dict[str, Any],
                            funcs: Dict[str, Callable],
                            stage: str,
                            batch_size: int = None,
                            ) -> dict:
        results = {}
        for name, func in funcs.items():
            result, result_meta = func(inp, latent, output, labels, meta)
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
    
    def _batch_split(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            self.raise_error(f"Expected batch to be a tuple or list of (data, labels), got "
                             f"{type(batch)} with length {len(batch) if isinstance(batch, (tuple, list)) else 'N/A'}")
        data, labels = batch
        return data, labels
    
    def extra_training_step(self, inp, latent, output, labels, meta, losses):
        return losses

    def _step(self, batch, stage: str) -> torch.Tensor:
        data, labels = self._batch_split(batch)
        output, latent, meta = self(data)
        batch_size = self.trainer.datamodule.batch_size \
            if self.trainer and self.trainer.datamodule else None

        with torch.no_grad():
            metrics = self.metrics if stage in ["train", "val"] else self.test_metrics
            metrics = self._multiple_calculate(data, latent, output, labels, meta, 
                                           metrics, stage, batch_size)
    
        if stage == "test":
            self._plot_test(data, latent, output, labels, meta)
            return metrics.get("mae", torch.tensor(0.0))

        losses = self._multiple_calculate(data, latent, output, labels, meta, 
                                          self.losses, stage, batch_size)
        losses = self.extra_training_step(data, latent, output, labels, meta, losses)
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
        output, meta_dec = self._decode(latent)
        
        meta.update(meta_latent)
        meta.update(meta_dec)
        meta.update(meta_sample)

        return output, latent, meta
    
    # ------------------------------------------------------------------
    # Losses and metrics
    # ------------------------------------------------------------------

    def loss(self, 
             inp: Union[torch.Tensor, Data],
             latent: torch.Tensor, 
             output: torch.Tensor, 
             labels: torch.Tensor, 
             meta: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        loss = F.mse_loss(inp, output, reduction='none')
        loss = torch.mean(loss)
        return loss, {}

    def metric_mae(self, 
                   inp: Union[torch.Tensor, Data],
                   latent: torch.Tensor,
                   output: torch.Tensor,
                   labels: torch.Tensor,
                   meta: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        mae = F.l1_loss(inp, output, reduction='none')
        mae = torch.mean(mae)
        return mae, {}

    def aggregate_losses(self, losses: dict) -> torch.Tensor:
        return torch.sum(torch.stack(list(losses.values())))
    
    # ------------------------------------------------------------------
    # Test plotting
    # ------------------------------------------------------------------

    def add_test_plotter(self, plotter_name: str, plotter_args: None) -> None:
        if plotter_name in self.test_plotters.keys():
            self.raise_error(f"Test plotter '{plotter_name}' already exists. Cannot add duplicate plotter.")
        self.test_plotters[plotter_name] = plotter_args or {}
    
    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_latent(self, data: torch.Tensor) -> torch.Tensor:
        return self.encode(data)

    def get_decoded(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode(latent)
    


