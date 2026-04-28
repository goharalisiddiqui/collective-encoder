import torch
import pytorch_lightning as pl

from collective_encoder.common.module import CEModule


class CENetBase(pl.LightningModule, CEModule):
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

    def __init__(self) -> None:
        pl.LightningModule.__init__(self)
        CEModule.__init__(self, verbose=True)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

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

    def _validate_norm_sizes(self, Mean: torch.Tensor, Range: torch.Tensor) -> None:
        """Hook for subclasses to assert normalization buffer dimensions."""
        pass

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------

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
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self.step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self.forward(batch)
