from abc import ABC, abstractmethod
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from collective_encoder.common.module import CEModule
from collective_encoder.metrics.mae import CEMetricMAE


class CEModelBase(nn.Module, CEModule, ABC):
    """Root base class for all representation learning models in collective_encoder.

    Inherits from :class:`torch.nn.Module`, :class:`CEModule`, and :class:`abc.ABC`.
    Encapsulates core model capabilities independent of training paradigm:
        - Normalization buffer management (Mean, Range, normIn, normSet)
        - Data encoding and decoding interfaces (forward, encoder, decoder)
        - Test plotter lifecycle management and evaluation dispatching
        - Checkpoint serialization and deserialization

    Subclasses must implement:
        - ``get_norm_len() -> int``
        - ``_normalize(x)``
        - ``_denormalize(x)``
    """

    _IDENTIFIER: str = None
    _REQUIRED_ARGS: List[str] = []
    _OPTIONAL_ARGS: Dict[str, Any] = {
        "normIn": False,
        "export_latent": False,
        "output_directory": "./ce_net_output/untitled_",
    }

    @staticmethod
    def extract_args_from_datamodule(datamodule, args: dict) -> dict:
        """Extract dataset-specific properties into args dict before instantiation."""
        return args

    def __init__(self, args: Dict[str, Any] = None, **kwargs) -> None:
        nn.Module.__init__(self)
        CEModule.__init__(self, args=args, **kwargs)

        normIn = getattr(self, "normIn", False)
        if hasattr(self, "normIn"):
            delattr(self, "normIn")  # Delete attribute to register as PyTorch persistent buffer
        self.register_buffer("normIn", torch.tensor(normIn, dtype=torch.bool))
        self.register_buffer("normSet", torch.tensor(False, dtype=torch.bool))

        # Defined here so checkpoints can be loaded seamlessly
        norm_len = self.get_norm_len()
        self.register_buffer("Mean", torch.zeros(norm_len))
        self.register_buffer("Range", torch.ones(norm_len))

        self.test_metrics = {
            "mae": CEMetricMAE({}, **kwargs),
        }
        self.test_plotters = []

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @abstractmethod
    def get_norm_len(self) -> int:
        """Return the expected length of the Mean and Range normalization buffers."""
        pass

    def set_norm(self, datamodule=None) -> None:
        """Load feature statistics from the datamodule scaler.

        Args:
            datamodule: Optional datamodule reference. If None, retrieves from attached trainer.
        """
        dm = datamodule
        if dm is None:
            if hasattr(self, "trainer") and self.trainer and hasattr(self.trainer, "datamodule") and self.trainer.datamodule:
                dm = self.trainer.datamodule
            elif hasattr(self, "datamodule") and self.datamodule:
                dm = self.datamodule

        if dm is None:
            self.raise_error(
                "Datamodule not available; cannot compute normalization. "
                "Ensure datamodule is passed to set_norm() or attached to trainer.",
                RuntimeError,
            )

        with torch.no_grad():
            Mean = dm.get_scaler_mean()
            Range = dm.get_scaler_scale()
            self._validate_norm_sizes(Mean, Range)
            device = next(self.parameters()).device if list(self.parameters()) else self.Mean.device
            Mean = torch.tensor(Mean, device=device, dtype=torch.float32)
            Range = torch.tensor(Range, device=device, dtype=torch.float32)
            Range = Range.clone()
            Range[Range == 0.0] = 1.0
            self.Mean = Mean
            self.Range = Range
            self.normSet = torch.tensor(True, dtype=torch.bool)
            self.log_msg("Normalization buffers set from datamodule scaler.")

    def normalize(self, x: Union[torch.Tensor, Data]) -> Union[torch.Tensor, Data]:
        """Normalize input data using the stored Mean and Range buffers."""
        if not self.normIn:
            return x
        if not self.normSet:
            self.set_norm()
        return self._normalize(x)

    def denormalize(self, x: Union[torch.Tensor, Data]) -> Union[torch.Tensor, Data]:
        """Denormalize input data using the stored Mean and Range buffers."""
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
    # Forward Pass & Encoding / Decoding
    # ------------------------------------------------------------------

    def _encode(self, data):
        data = self.normalize(data)
        z, meta = self.encoder(data)
        return z, meta

    def _decode(self, z: torch.Tensor):
        out, meta = self.decoder(z)
        if out is not None:
            out = self.denormalize(out)
        return out, meta

    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        z = self.encoder_net(x)
        return z, {}

    def decoder(self, z: torch.Tensor) -> Tuple[Optional[torch.Tensor], dict]:
        if hasattr(self, "decoder_net") and self.decoder_net is not None:
            x_out = self.decoder_net(z)
            return x_out, {}
        return None, {}

    def latent_to_decoder_input(self, latent) -> Tuple:
        return latent, {}

    def forward(self, data) -> Tuple[Optional[torch.Tensor], torch.Tensor, dict]:
        """Compute and return the forward pass output for a given batch."""
        meta = {}

        latent, meta_latent = self._encode(data)
        latent, meta_sample = self.latent_to_decoder_input(latent)
        output, meta_dec = self._decode(latent)

        meta.update(meta_latent)
        meta.update(meta_dec)
        meta.update(meta_sample)

        return output, latent, meta

    def get_latent(self, data: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.normalize(data))[0]

    def get_decoded(self, latent: torch.Tensor) -> torch.Tensor:
        out = self.decoder(latent)[0]
        return self.denormalize(out) if out is not None else None

    # ------------------------------------------------------------------
    # Test Plotters & Metric Calculations
    # ------------------------------------------------------------------

    def add_test_plotter(self, plotter_name: str, plotter_args: dict = None) -> None:
        self.test_plotters.append((plotter_name, plotter_args))

    def _plot_test_start(self) -> None:
        from collective_encoder.testplotters.resolver import get_testplotter

        initialized_plotters = []
        for name, args in self.test_plotters:
            try:
                plotter_cls = get_testplotter(name)
                plotter_args = (args or {}).copy()
                if hasattr(self, "logger"):
                    plotter_args["logger"] = self.logger
                plotter = plotter_cls(plotter_args, **self.get_run_args())
                initialized_plotters.append(plotter)
            except Exception as e:
                self.log_exception(f"Test plotter '{name}' failed to initialize: {e}", RuntimeError)
        self.test_plotters = initialized_plotters

    def _plot_test_batch(self, inp, latent, output, labels, meta, datamodule=None) -> None:
        dm = datamodule
        if dm is None:
            if hasattr(self, "trainer") and self.trainer and hasattr(self.trainer, "datamodule") and self.trainer.datamodule:
                dm = self.trainer.datamodule
            elif hasattr(self, "datamodule") and self.datamodule:
                dm = self.datamodule

        label_names = dm.get_label_names() if dm else []
        if isinstance(labels, torch.Tensor):
            if len(label_names) != labels.shape[1]:
                self.raise_error(
                    f"Number of label names ({len(label_names)}) "
                    f"does not match number of label columns ({labels.shape[1]})."
                )
            labels_dict = {name: labels[:, i] for i, name in enumerate(label_names)}
        elif isinstance(labels, dict):
            labels_dict = labels
        else:
            self.raise_error(f"Unexpected labels type: {type(labels)}")

        for plotter in self.test_plotters:
            plotter.add_batch(inp, latent, output, labels_dict, meta)

    def _plot_test_finish(self) -> None:
        for plotter in self.test_plotters:
            plotter.finish()

    def get_test_plotter_metrics(self) -> Dict[str, float]:
        """Collects all scalar metrics recorded by initialized test plotters."""
        metrics: Dict[str, float] = {}
        for plotter in getattr(self, "test_plotters", []):
            if hasattr(plotter, "get_metrics"):
                metrics.update(plotter.get_metrics())
        return metrics

    def _multiple_calculate(
        self,
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
            if hasattr(self, "log"):
                if isinstance(result, (int, float)) or (isinstance(result, torch.Tensor) and result.numel() == 1):
                    self.log(f"{stage}_{name}", result.detach(), prog_bar=(stage == "train"),
                             on_step=(stage == "train"), on_epoch=True, batch_size=batch_size)
                for key, value in result_meta.items():
                    if isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                        self.log(f"{stage}_{name}_{key}", value,
                                 prog_bar=False, on_step=(stage == "train"), on_epoch=True, batch_size=batch_size)
            meta.update(result_meta)
        return results

    def _batch_split(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            self.raise_error(
                f"Expected batch to be a tuple or list of (data, labels), got "
                f"{type(batch)} with length {len(batch) if isinstance(batch, (tuple, list)) else 'N/A'}"
            )
        data, labels = batch
        return data, labels

    # ------------------------------------------------------------------
    # Universal Testing Loop for Non-Lightning / Analytical Evaluation
    # ------------------------------------------------------------------

    def test(self, datamodule) -> None:
        """Evaluate model on datamodule test set and execute all test plotters."""
        self.datamodule = datamodule
        if self.normIn and not self.normSet:
            self.set_norm(datamodule)

        self._plot_test_start()
        self.eval()
        device = next(self.parameters()).device if list(self.parameters()) else self.Mean.device
        with torch.no_grad():
            for batch in datamodule.test_dataloader():
                data, labels = self._batch_split(batch)
                if isinstance(data, torch.Tensor):
                    data = data.to(device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                output, latent, meta = self(data)
                norm_data = self.normalize(data)
                if len(self.test_plotters) > 0:
                    self._plot_test_batch(norm_data, latent, output, labels, meta, datamodule=datamodule)
        self._plot_test_finish()

    # ------------------------------------------------------------------
    # Checkpoint Serialization
    # ------------------------------------------------------------------

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save model state dict and hyperparameters to checkpoint file."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hyper_parameters": {"args": self.args},
            },
            checkpoint_path,
        )
        self.log_msg(f"Saved checkpoint to: {checkpoint_path}")

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, args: dict = None, **kwargs) -> "CEModelBase":
        """Load model instance from checkpoint file."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_args = ckpt.get("hyper_parameters", {}).get("args", {})
        if args is not None:
            saved_args.update(args)
        model = cls(args=saved_args, **kwargs)
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip potential lightning module prefix
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        return model
