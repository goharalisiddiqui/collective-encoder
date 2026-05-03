from typing import List, Optional, Dict, Union

import numpy as np

import torch
from torch import nn, Tensor

from collective_encoder.nets.base import CENetBase
from .modules.graph_encoder import BondGraphEncoder
from .modules.graph_decoder import BondGraphDecoder


class BondGraphEncoderDecoder(CENetBase):
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
        encoder_args: Dict[str, Union[int, float]],
        decoder_args: Dict[str, Union[int, float]],
        datamodule=None,
        lrate: float = 1e-4,
        weight_decay: float = 0.0,
        normIn: bool = False,
        scheduler: bool = False,
        scheduler_args: Optional[Dict] = None,
        loss_fn: Optional[nn.Module] = None,
        loss_weights: Optional[List[float]] = None,
        loss_latent_weight: float = 0.0,
        out_labels: Optional[List[str]] = None,
        outname: str = './BGE_untitled/BGE_',
    ):
        self.save_hyperparameters(ignore=["datamodule"])
        super().__init__()

        if out_labels is None:
            out_labels = ['bond_dist', 'angle', 'dihedral_cos', 'dihedral_sin']
        assert len(out_labels) == 4, "out_labels must be a list of 4 strings"

        assert loss_weights is None or len(loss_weights) == 4, \
            "loss_weights must be None or a list of 4 floats"
        if loss_weights is None:
            loss_weights = [1.0, 1.0, 1.0, 1.0]
        self.loss_weights = loss_weights
        self.latent_dim = encoder_args['latent_dim']

        if datamodule is not None:
            self._init_decoder(datamodule)
        else: # Initialize decoder lazily on first decode call (requires trainer/datamodule access)
            self.gnn_dec = None

        self.gnn_enc = BondGraphEncoder(**encoder_args)
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.register_buffer('normIn', torch.tensor(normIn, dtype=torch.bool))
        self.register_buffer('normSet', torch.tensor(False, dtype=torch.bool))
        num_norm = encoder_args['node_feat'] + encoder_args['edge_feat']
        self.register_buffer('Mean', torch.zeros(num_norm))
        self.register_buffer('Range', torch.ones(num_norm))

    # ------------------------------------------------------------------
    # Optimizer and scheduler hooks
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lrate,
            weight_decay=self.hparams.weight_decay,
        )

    def _get_scheduler_args(self) -> dict:
        defaults: dict = {
            "mode": "min",
            "factor": 0.7,
            "patience": 10,
            "min_lr": 1e-9,
            "cooldown": 10,
        }
        defaults.update(self.hparams.scheduler_args or {})
        return defaults

    # ------------------------------------------------------------------
    # Normalization size validation hook
    # ------------------------------------------------------------------

    def _validate_norm_sizes(self, Mean: torch.Tensor, Range: torch.Tensor) -> None:
        expected = self.hparams.encoder_args['node_feat'] + self.hparams.encoder_args['edge_feat']
        assert Mean.size(0) == expected, \
            f"Mean size {Mean.size(0)} does not match expected {expected}"
        assert Range.size(0) == expected, \
            f"Range size {Range.size(0)} does not match expected {expected}"

    # ------------------------------------------------------------------
    # Decoder initialization
    # ------------------------------------------------------------------

    def _init_decoder(self, datamodule) -> None:
        datasetobject = datamodule.get_train_dataset()
        self.template_khop = self.hparams.decoder_args['template_khop']
        template_data = datasetobject.get_template_graph(k=self.template_khop)
        bond_index, angle_index, torsion_index = datasetobject.get_label_indices()
        gnn_dec_kwargs = {
            "template_data": template_data,
            "label_indices": (bond_index, angle_index, torsion_index),
            "latent_dim": self.latent_dim,
        }
        gnn_dec_args = self.hparams.decoder_args.copy()
        gnn_dec_args.pop('template_khop', None)
        self.gnn_dec = BondGraphDecoder(**gnn_dec_kwargs, **gnn_dec_args)

    # ------------------------------------------------------------------
    # Graph-specific normalization (cannot be shared: operates on PyG Data)
    # ------------------------------------------------------------------

    def normalize(self, data):
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

        node_dim = self.hparams.encoder_args['node_feat']
        edge_dim = self.hparams.encoder_args['edge_feat']
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

    def denormalize(self, data):
        """Inverse of :meth:`normalize` for a Data object."""
        if not self.normIn:
            return data
        if not self.normSet:
            self.set_norm()

        if not getattr(data, '_normalized', False):
            return data

        node_dim = self.hparams.encoder_args['node_feat']
        edge_dim = self.hparams.encoder_args['edge_feat']
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

    def encode(self, data):
        data = self.normalize(data)
        return self.gnn_enc(data)

    def decode(self, latent):
        if self.gnn_dec is None:
            try:
                self.trainer
            except RuntimeError:
                raise RuntimeError("Decoder not initialized and trainer not found.")
            datamodule = getattr(self.trainer, 'datamodule', None)
            if datamodule is None:
                raise RuntimeError("Decoder not initialized and trainer datamodule not found.")
            self._init_decoder(datamodule)
        return self.gnn_dec(latent)

    def forward(self, data):
        latent = self.encode(data)
        pred = self.decode(latent)
        return pred, latent

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def extract_labels(self, batch) -> dict:
        """Extract per-graph target tensors from a PyG batch."""
        num_graphs = batch.batch.max().item() + 1
        out_labels = self.hparams.out_labels
        return {
            out_labels[0]: batch.y_bonds.view(num_graphs, -1),
            out_labels[1]: batch.y_angles.view(num_graphs, -1),
            out_labels[2]: batch.y_torsions_cos.view(num_graphs, -1),
            out_labels[3]: batch.y_torsions_sin.view(num_graphs, -1),
        }

    def loss_encdec(self, pred: dict, labels: dict, stage: str, batch_size=None) -> torch.Tensor:
        losses = {}
        for out_label, weight in zip(self.hparams.out_labels, self.loss_weights):
            losses[out_label] = self.loss_fn(pred[out_label], labels[out_label]) * weight
            self.log(f"{stage}_recon_{out_label}_loss", losses[out_label],
                     prog_bar=False, on_epoch=True, batch_size=batch_size)

        self.log(f"{stage}_recon_loss", sum(losses.values()),
                 prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True, batch_size=batch_size)

        with torch.no_grad():
            mae = {}
            for out_label in self.hparams.out_labels:
                mae[out_label] = (
                    torch.abs(pred[out_label] - labels[out_label]).mean()
                    if labels[out_label].numel() > 0
                    else torch.tensor(0.0, device=pred[out_label].device)
                )
                self.log(f"{stage}_recon_{out_label}_mae", mae[out_label],
                         prog_bar=False, on_epoch=True, batch_size=batch_size)
            self.log(f"{stage}_recon_mae", sum(mae.values()) / len(mae),
                     prog_bar=(stage != "train"), on_epoch=True, batch_size=batch_size)

        return sum(losses.values())

    def loss_latent(self, latent: torch.Tensor, stage: str) -> torch.Tensor:
        """Encourage sequential latent points to be close and equidistant."""
        loss_latent = torch.tensor(0.0, device=latent.device)
        if latent.size(0) > 1:
            batch_dist = torch.norm(latent[1:] - latent[:-1], dim=1)
            loss_latent = torch.mean(batch_dist)
        if latent.size(0) > 2:
            loss_latent = loss_latent + torch.var(batch_dist)
        self.log(f"{stage}_loss_latent", loss_latent,
                 prog_bar=False, on_step=(stage == "train"), on_epoch=True)
        return loss_latent

    def step(self, batch, stage: str) -> torch.Tensor:
        pred, latent = self.forward(batch)
        labels = self.extract_labels(batch)
        batch_size = self.trainer.datamodule.hparams.batch_size if self.trainer and self.trainer.datamodule else None

        loss = self.loss_encdec(pred, labels, stage, batch_size=batch_size)

        if self.hparams.loss_latent_weight > 0.0:
            loss = loss + self.hparams.loss_latent_weight * self.loss_latent(latent, stage)

        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"),
                 on_step=(stage == "train"), on_epoch=True, batch_size=batch_size)
        return loss

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_latent(self, data):
        return self.encode(data)

    def get_decoded(self, latent):
        return self.decode(latent)


__all__ = ["BondGraphEncoderDecoder"]
