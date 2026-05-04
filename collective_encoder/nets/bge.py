from typing import Any, List, Optional, Dict, Tuple, Union

import numpy as np

import torch
from torch import nn, Tensor

from collective_encoder.nets.base import CENetBase
from .modules.graph_encoder import BondGraphEncoder
from .modules.graph_decoder import BondGraphDecoder

from collective_encoder.utils import check_dict_contains_keys


class BondGraphEncoderDecoder(CENetBase):
    _IDENTIFIER = "BGE"
    _COMPATIBLE_DATASETS = ["GRAPH"]
    _REQUIRED_ARGS = ['encoder_args', 'decoder_args']
    _OPTIONAL_ARGS = CENetBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'loss_fn': nn.MSELoss(),
        'loss_weights': [1.0, 1.0, 1.0, 1.0],
        'out_labels': ['bond_dist', 'angle', 'dihedral_cos', 'dihedral_sin'],
        'loss_latent_weight': 0.0,
    })

    """PyTorch Lightning module for bond-graph autoencoder (BGE).

    Encodes molecular graphs to a latent space and decodes back to
    bond/angle/torsion geometry.

    Args:
        encoder_args: Keyword arguments for :class:`BondGraphEncoder`.
        decoder_args: Keyword arguments for :class:`BondGraphDecoder`.
            Must include ``template_khop``.
        datamodule: Optional data module used to initialize the decoder at
            construction time.  When ``None`` the decoder is initialized lazily
            on the first call to :meth:`decode`.
        lrate: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        normIn: Enable input normalization.
        scheduler: Enable ``ReduceLROnPlateau`` scheduler.
        scheduler_args: Overrides for the scheduler defaults.
        loss_fn: Reconstruction loss (default: MSE).
        loss_weights: Per-output loss weights (length 4).
        loss_latent_weight: Weight for the latent regularization loss.
        out_labels: Names for the four output channels.
        outname: Output stem for saved files.
    """
    
    def __init__(
        self,
        datamodule = None,
        args: Dict[str, Any] = None,
        **kwargs
    ):
        self.save_hyperparameters(ignore=["datamodule"])
        super().__init__(args=args, **kwargs)
        
        # Some check
        assert len(self.out_labels) == 4, "out_labels must be a list of 4 strings"
        assert len(self.loss_weights) == 4, "loss_weights must be None or a list of 4 floats"
        check_dict_contains_keys(self.encoder_args, required_keys=['latent_dim'])
        check_dict_contains_keys(self.decoder_args, required_keys=['template_khop'])
        

        self.latent_dim = self.encoder_args['latent_dim']

        if datamodule is not None:
            self._init_encoder(datamodule)
            self._init_decoder(datamodule)
        else: # Initialize decoder lazily on first decode call (requires trainer/datamodule access)
            self.encoder_net = None
            self.decoder_net = None

        
        
        self.losses = {
            'encdec': self.loss_encdec,
        }
        if self.loss_latent_weight > 0.0:
            self.losses['latent'] = self.loss_latent
        
        self.metrics = {
            'mae': self.metric_encdec_mae
        }
        self.test_metrics = self.metrics.copy()

    def get_norm_len(self):
        return self.encoder_args['node_feat'] + self.encoder_args['edge_feat']

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def print_hparams(self):
        super().print_hparams()
        self.log_msg(f"  loss_fn: {self.loss_fn}")
        self.log_msg(f"  loss_weights: {self.loss_weights}")
        self.log_msg(f"  loss_latent_weight: {self.loss_latent_weight}")
        self.log_msg(f"  out_labels: {self.out_labels}")
        self.ce_log_dict("encoder_args", self.encoder_args, 2)
        self.ce_log_dict("decoder_args", self.decoder_args, 2)

    # ------------------------------------------------------------------
    # Optimizer and scheduler hooks
    # ------------------------------------------------------------------

    def _get_scheduler_args(self) -> dict:
        defaults: dict = {
            "mode": "min",
            "factor": 0.7,
            "patience": 10,
            "min_lr": 1e-9,
            "cooldown": 10,
        }
        defaults.update(self.scheduler_args or {})
        return defaults

    # ------------------------------------------------------------------
    # Normalization size validation hook
    # ------------------------------------------------------------------

    def _validate_norm_sizes(self, Mean: torch.Tensor, Range: torch.Tensor) -> None:
        expected = self.encoder_args['node_feat'] + self.encoder_args['edge_feat']
        assert Mean.size(0) == expected, \
            f"Mean size {Mean.size(0)} does not match expected {expected}"
        assert Range.size(0) == expected, \
            f"Range size {Range.size(0)} does not match expected {expected}"

    # ------------------------------------------------------------------
    # Decoder initialization
    # ------------------------------------------------------------------
    
    def _init_encoder(self, datamodule) -> None:
        datasetobject = datamodule.get_train_dataset()
        self.encoder_args.update(
            node_feat=datasetobject.get_num_node_features(),
            edge_feat=datasetobject.get_num_edge_features(),
        )
        self.encoder_net = BondGraphEncoder(**self.encoder_args)

    def _init_decoder(self, datamodule) -> None:
        datasetobject = datamodule.get_train_dataset()
        self.template_khop = self.decoder_args['template_khop']
        template_data = datasetobject.get_template_graph(k=self.template_khop)
        bond_index, angle_index, torsion_index = datasetobject.get_label_indices()
        gnn_dec_kwargs = {
            "template_data": template_data,
            "label_indices": (bond_index, angle_index, torsion_index),
            "latent_dim": self.latent_dim,
        }
        gnn_dec_args = self.decoder_args.copy()
        gnn_dec_args.pop('template_khop', None)
        self.decoder_net = BondGraphDecoder(**gnn_dec_kwargs, **gnn_dec_args)

    # ------------------------------------------------------------------
    # Graph-specific normalization (cannot be shared: operates on PyG Data)
    # ------------------------------------------------------------------

    def _normalize(self, data):
        """Normalize node and edge attributes of a PyG Data object in-place.

        Concatenated ``Mean``/``Range`` buffers are split by node/edge dimension.
        Skips if already normalized (tracked via ``data._normalized``).
        """
        if not self.normIn:
            return data
        if not self.normSet:
            self.set_norm()

        if getattr(data, '_normalized', False):
            return data

        node_dim = self.encoder_args['node_feat']
        edge_dim = self.encoder_args['edge_feat']
        mean_node = self.Mean[:node_dim]
        range_node = self.Range[:node_dim]
        mean_edge = self.Mean[node_dim:node_dim + edge_dim]
        range_edge = self.Range[node_dim:node_dim + edge_dim]

        if hasattr(data, 'x') and data.x is not None:
            if data.x.size(-1) != node_dim:
                raise ValueError(f"Node feature dim mismatch: data.x={data.x.size(-1)} expected={node_dim}")
            if data.x.dtype != mean_node.dtype:
                mean_node = mean_node.to(data.x.dtype)
                range_node = range_node.to(data.x.dtype)
            data.x = (data.x - mean_node.view(1, -1)) / range_node.view(1, -1)

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.size(-1) < edge_dim:
                raise ValueError(f"Edge feature dim mismatch: edge_attr={data.edge_attr.size(-1)} expected>={edge_dim}")
            ea = data.edge_attr
            if ea.dtype != mean_edge.dtype:
                mean_edge = mean_edge.to(ea.dtype)
                range_edge = range_edge.to(ea.dtype)
            head = (ea[:, :edge_dim] - mean_edge.view(1, -1)) / range_edge.view(1, -1)
            data.edge_attr = torch.cat([head, ea[:, edge_dim:]], dim=-1) if ea.size(-1) > edge_dim else head

        setattr(data, '_normalized', True)
        return data

    def _denormalize(self, data):
        """Inverse of :meth:`normalize` for a Data object."""
        if not self.normIn:
            return data
        if not self.normSet:
            self.set_norm()

        if not getattr(data, '_normalized', False):
            return data

        node_dim = self.encoder_args['node_feat']
        edge_dim = self.encoder_args['edge_feat']
        mean_node = self.Mean[:node_dim]
        range_node = self.Range[:node_dim]
        mean_edge = self.Mean[node_dim:node_dim + edge_dim]
        range_edge = self.Range[node_dim:node_dim + edge_dim]

        if hasattr(data, 'x') and data.x is not None:
            if data.x.dtype != mean_node.dtype:
                mean_node = mean_node.to(data.x.dtype)
                range_node = range_node.to(data.x.dtype)
            data.x = data.x * range_node.view(1, -1) + mean_node.view(1, -1)

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            ea = data.edge_attr
            if ea.size(-1) >= edge_dim:
                if ea.dtype != mean_edge.dtype:
                    mean_edge = mean_edge.to(ea.dtype)
                    range_edge = range_edge.to(ea.dtype)
                head = ea[:, :edge_dim] * range_edge.view(1, -1) + mean_edge.view(1, -1)
                data.edge_attr = torch.cat([head, ea[:, edge_dim:]], dim=-1) if ea.size(-1) > edge_dim else head

        setattr(data, '_normalized', False)
        return data

    # ------------------------------------------------------------------
    # Encode / decode / forward
    # ------------------------------------------------------------------

    def _batch_split(self, batch):
        return batch, self.extract_labels(batch)

    def decoder(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # We lazily initialize the encoder/decoder if it wasn't initialized at construction time.
        if self.decoder_net is None or self.encoder_net is None:
            try:
                self.trainer
            except RuntimeError:
                self.raise_error("Encoder/Decoder not initialized and trainer not found.", error_type=RuntimeError)
            datamodule = getattr(self.trainer, 'datamodule', None)
            if datamodule is None:
                self.raise_error("Encoder/Decoder not initialized and trainer datamodule not found.", error_type=RuntimeError)
            if self.encoder_net is None:
                self._init_encoder(datamodule)
            if self.decoder_net is None:
                self._init_decoder(datamodule)
        return super().decoder(z)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def extract_labels(self, batch) -> dict:
        """Extract per-graph target tensors from a PyG batch."""
        num_graphs = batch.batch.max().item() + 1
        out_labels = self.out_labels
        return {
            out_labels[0]: batch.y_bonds.view(num_graphs, -1),
            out_labels[1]: batch.y_angles.view(num_graphs, -1),
            out_labels[2]: batch.y_torsions_cos.view(num_graphs, -1),
            out_labels[3]: batch.y_torsions_sin.view(num_graphs, -1),
        }

    def loss_encdec(self, input, latent, output, labels, meta) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        for out_label, weight in zip(self.out_labels, self.loss_weights):
            losses[out_label] = self.loss_fn(output[out_label], labels[out_label]) * weight

        return sum(losses.values()), losses

    def metric_encdec_mae(self, input, latent, output, labels, meta) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mae = {}
        for out_label in self.out_labels:
            mae[out_label] = (
                torch.abs(output[out_label] - labels[out_label]).mean()
                if labels[out_label].numel() > 0
                else torch.tensor(0.0, device=output[out_label].device)
            )
        recon_mae = sum(mae.values()) / len(mae)
        return recon_mae, mae
    
    def loss_latent(self, input, latent, output, labels, meta) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encourage sequential latent points to be close and equidistant."""
        loss_latent = torch.tensor(0.0, device=latent.device)
        if latent.size(0) > 1:
            batch_dist = torch.norm(latent[1:] - latent[:-1], dim=1)
            loss_latent = torch.mean(batch_dist)
        if latent.size(0) > 2:
            loss_latent = loss_latent + torch.var(batch_dist)
        return loss_latent, {}
    
    def aggregate_losses(self, losses):
        loss = losses['encdec']
        if self.loss_latent_weight > 0.0:
            loss = loss + self.loss_latent_weight * losses['latent']
        return loss

__all__ = ["BondGraphEncoderDecoder"]
