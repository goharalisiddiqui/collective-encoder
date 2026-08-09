from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, Dict

import numpy as np

import torch
import torch.nn.functional as F

from collective_encoder.nets.base import CENetBase


class AEBase(CENetBase, ABC):
    _REQUIRED_ARGS = ['encoder_network', 'decoder_network', 
                      'datapoint_shape', 'dataset_type']
    _OPTIONAL_ARGS = CENetBase._OPTIONAL_ARGS
    _OPTIONAL_ARGS.update({
        'batch_norm': False,  # Whether to use batch normalization in the encoder/decoder
        'activation': 'relu',  # Activation function
        'activation_args': None,  # Arguments for the activation function
    })
    _COMPATIBLE_DATASETS = []
    
    """Base class for dense-tensor autoencoder architectures (VAE, AE, DVAE, EDVAE).

    Handles dense-tensor normalization, reparametrization, the generic training
    step, and metric/plotter dispatch.  Graph-based networks use
    ``BondGraphEncoderDecoder`` instead.

    Subclasses must implement ``encoder(x)`` and ``decoder(z)``; they call
    ``save_hyperparameters()`` before ``super().__init__()``.
    """
    @staticmethod
    def extract_args_from_datamodule(datamodule, args) -> dict:
        args['datapoint_shape'] = datamodule.get_datapoint_shape()
        args['dataset_type'] = datamodule.dataset_type
        return args

    def __init__(self,
                args: Dict[str, Any] = None,
                **kwargs
                ):
        super().__init__(args=args, **kwargs)
        self.metatomic_model_cls = MetatomicModelAE
        
        assert self.dataset_type in self._COMPATIBLE_DATASETS, (
            f"Dataset type '{self.dataset_type}' is not compatible with AE. "
            f"Compatible types: {self._COMPATIBLE_DATASETS}"
        )
        
        encoder_nodes = [int(x) for x in self.encoder_network]
        encoder_nodes.insert(0, self.datapoint_shape[0])
        
        self.latent_dim = encoder_nodes[-1]
        
        decoder_nodes = [int(x) for x in self.decoder_network]
        decoder_nodes.insert(0, self.latent_dim)
        decoder_nodes.append(self.datapoint_shape[0])

        self.encoder_network = encoder_nodes
        self.decoder_network = decoder_nodes
        self.init_network()
    
    @abstractmethod
    def init_network(self):
        """Initialize the encoder and decoder networks. Called by __init__() after
        processing hyperparameters. Subclasses must implement this method."""
        pass

    # ------------------------------------------------------------------
    # Dense-tensor normalization
    # ------------------------------------------------------------------
    
    def get_norm_len(self) -> int:
        return self.datapoint_shape[0]

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.Mean.numel() != np.prod(x.shape[1:]):
            self.raise_error(f"Mean and Range buffers must have the same number" 
                             f" of elements as the input features. Got Mean shape:"
                             f" {self.Mean.shape}, Range shape: {self.Range.shape},"
                             f" input shape: {x.shape}")
        mean_expanded = self.Mean.view(1, *(x.shape[1:])).expand(x.shape)
        range_expanded = self.Range.view(1, *(x.shape[1:])).expand(x.shape)
        return (x - mean_expanded) / range_expanded

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.Mean.numel() != np.prod(x.shape[1:]):
            self.raise_error(f"Mean and Range buffers must have the same number" 
                             f" of elements as the input features. Got Mean shape:"
                             f" {self.Mean.shape}, Range shape: {self.Range.shape},"
                             f" input shape: {x.shape}")
        mean_expanded = self.Mean.view(1, *(x.shape[1:])).expand(x.shape)
        range_expanded = self.Range.view(1, *(x.shape[1:])).expand(x.shape)
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

    def get_metad_output(self,
                        latent: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                        meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        return latent

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_latent_names(self) -> str:
        return "latent"

    def export_latent(self, latent: torch.Tensor, labels: torch.Tensor = None) -> None:
        raise NotImplementedError("export_latent() not implemented for this model")

    def get_metatomic_model(self):
        raise NotImplementedError("get_metatomic_model() not implemented for this model")


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
        mean_expanded = self.Mean.view(1, *(x.shape[1:])).expand(x.shape)
        range_expanded = self.Range.view(1, *(x.shape[1:])).expand(x.shape)
        return (x - mean_expanded) / range_expanded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        latent, _ = self.encoder(x)
        mean, logvar = latent
        return mean