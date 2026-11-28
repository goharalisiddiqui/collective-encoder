from typing import Any, Dict

import torch
from torch import nn

from collective_encoder.models.neural_nets.base import CENetBase
from .modules.graph_encoder import BondGraphEncoder
from .modules.graph_decoder import BondGraphDecoder

from collective_encoder.losses.mse_dict import CELossMSEDict
from collective_encoder.losses.latent import CELossLatent
from collective_encoder.metrics.mae_dict import CEMetricMAEDict

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
            construction time. When ``None`` the decoder is initialized lazily
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
    @staticmethod
    def extract_args_from_datamodule(datamodule, args) -> dict:
        try:
            khops = args['decoder_args']['template_khop']
        except KeyError:
            raise ValueError("Missing 'template_khop' in decoder_args")
        datasetobject = datamodule.get_any_dataset()
        encoder_args = {
            'node_feat': datasetobject.num_node_features,
            'edge_feat': datasetobject.num_edge_features,
        }
        decoder_args = {
            'template_data': datasetobject.get_template_graph(k=khops),
            'label_indices': datasetobject.get_label_indices(),
        }

        species_templates = {}
        species_label_indices = {}
        for split_attr in ("train_data", "val_data", "test_data", "predict_data"):
            ds = getattr(datamodule, split_attr, None)
            if ds is not None and hasattr(ds, "get_species_template_graphs"):
                for sid, tmpl in ds.get_species_template_graphs(k=khops).items():
                    if sid not in species_templates:
                        species_templates[sid] = tmpl
                for sid, lidx in ds.get_species_label_indices().items():
                    if sid not in species_label_indices:
                        species_label_indices[sid] = lidx
        if not species_templates and hasattr(datasetobject, "get_species_template_graphs"):
            species_templates = datasetobject.get_species_template_graphs(k=khops)
            species_label_indices = datasetobject.get_species_label_indices()

        if len(species_templates) > 1:
            decoder_args["species_templates"] = species_templates
            decoder_args["species_label_indices"] = species_label_indices

        args['encoder_args'].update(encoder_args)
        args['decoder_args'].update(decoder_args)
        return args
    
    def __init__(
        self,
        args: Dict[str, Any] = None,
        **kwargs
    ):
        self.save_hyperparameters()
        super().__init__(args=args, **kwargs)
        
        # Some check
        assert len(self.out_labels) == 4, "out_labels must be a list of 4 strings"
        assert len(self.loss_weights) == 4, "loss_weights must be a list of 4 floats"
        check_dict_contains_keys(self.encoder_args, required_keys=['latent_dim'])
        
        self.template_khop = self.decoder_args.pop('template_khop', None)
        self.latent_dim = self.encoder_args['latent_dim']

        self._init_encoder()
        self._init_decoder()
        
        self.losses = {
            'encdec': CELossMSEDict({
                'keys': self.out_labels,
                'weights': self.loss_weights,
                }, **kwargs),
        }
        if self.loss_latent_weight > 0.0:
            self.losses['latent'] = CELossLatent({}, **kwargs)
        
        self.metrics = {
            'mae': CEMetricMAEDict({
                'keys': self.out_labels,
                }, **kwargs)
        }
        self.test_metrics = {
            'mae': CEMetricMAEDict({
                'keys': self.out_labels,
                }, **kwargs)
        }

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
    # Encoder/Decoder initialization
    # ------------------------------------------------------------------
    
    def _init_encoder(self) -> None:
        check_dict_contains_keys(self.encoder_args, 
                                    required_keys=['node_feat', 
                                                   'edge_feat', 
                                                   'latent_dim'])
        self.encoder_net = BondGraphEncoder(**self.encoder_args)

    def _init_decoder(self, datamodule=None) -> None:
        check_dict_contains_keys(self.decoder_args, 
                                    required_keys=['template_data', 
                                                   'label_indices'])
        self.decoder_net = BondGraphDecoder(**self.decoder_args,
                                            latent_dim=self.latent_dim)

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
    # Forward pass
    # ------------------------------------------------------------------

    def decoder(self, z: torch.Tensor, species_id: torch.Tensor = None):
        """Invoke the bond-graph decoder network (optionally passing ``species_id``)."""
        if hasattr(self, "decoder_net") and self.decoder_net is not None:
            x_out = self.decoder_net(z, species_id=species_id)
            return x_out, {}
        return None, {}

    def _decode(self, z: torch.Tensor, species_id: torch.Tensor = None):
        """Run the decoder network and denormalize the output."""
        out, meta = self.decoder(z, species_id=species_id)
        if out is not None:
            out = self.denormalize(out)
        return out, meta

    def get_decoded(self, latent: torch.Tensor, species_id: torch.Tensor = None):
        """Decode a latent tensor back into reconstructed geometry."""
        out = self.decoder(latent, species_id=species_id)[0]
        return self.denormalize(out) if out is not None else None

    def forward(self, data):
        """Encode and decode a batch of molecular bond graphs."""
        meta = {}
        species_id = getattr(data, "species_id", None)
        latent, meta_latent = self._encode(data)
        latent, meta_sample = self.latent_to_decoder_input(latent)
        output, meta_dec = self._decode(latent, species_id=species_id)

        meta.update(meta_latent)
        meta.update(meta_dec)
        meta.update(meta_sample)
        if species_id is not None:
            meta["species_id"] = species_id
        return output, latent, meta

    def _batch_split(self, batch):
        return batch, self.extract_labels(batch)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _slice_and_pad_attr(
        self,
        batch,
        attr_name: str,
        num_graphs: int,
        max_dim: int,
        species_counts: dict,
    ) -> torch.Tensor:
        """Slice a 1-D PyG batch target attribute per graph and pad to ``(num_graphs, max_dim)`` with ``NaN``."""
        raw = getattr(batch, attr_name)
        if not getattr(self.decoder_net, "is_multispecies", False) and raw.numel() == num_graphs * max_dim:
            return raw.view(num_graphs, max_dim)

        out = torch.full(
            (num_graphs, max_dim),
            float("nan"),
            dtype=raw.dtype,
            device=raw.device,
        )

        slice_dict = getattr(batch, "_slice_dict", None)
        if isinstance(slice_dict, dict) and attr_name in slice_dict:
            ptr = slice_dict[attr_name]
            for g in range(num_graphs):
                s_idx = int(ptr[g])
                e_idx = int(ptr[g + 1])
                length = min(e_idx - s_idx, max_dim)
                if length > 0:
                    out[g, :length] = raw[s_idx : s_idx + length]
            return out

        sids = getattr(batch, "species_id", None)
        if sids is not None and species_counts:
            offset = 0
            for g, sid_t in enumerate(sids.view(-1)):
                cnt = species_counts.get(int(sid_t.item()), 0)
                length = min(cnt, max_dim)
                if length > 0 and offset + cnt <= raw.numel():
                    out[g, :length] = raw[offset : offset + length]
                offset += cnt
            return out

        if num_graphs > 0 and raw.numel() % num_graphs == 0:
            per_g = raw.numel() // num_graphs
            out[:, : min(per_g, max_dim)] = raw.view(num_graphs, per_g)[:, :max_dim]
        return out

    def _align_dict_target(
        self,
        raw_tensor: torch.Tensor,
        num_graphs: int,
        max_dim: int,
        species_counts: dict,
        sids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Reshape a dictionary target tensor to ``(num_graphs, max_dim)`` and mask
        zero-padded columns beyond each species' true internal coordinate count
        with ``NaN`` when running in multi-species mode.
        """
        t = raw_tensor.view(num_graphs, -1)
        if not getattr(self.decoder_net, "is_multispecies", False):
            return t

        out = torch.full(
            (num_graphs, max_dim),
            float("nan"),
            dtype=t.dtype,
            device=t.device,
        )
        if sids is not None and species_counts:
            for g, sid_t in enumerate(sids.view(-1)):
                cnt = species_counts.get(int(sid_t.item()), t.shape[1])
                length = min(cnt, max_dim, t.shape[1])
                if length > 0:
                    out[g, :length] = t[g, :length]
            return out

        c = min(t.shape[1], max_dim)
        out[:, :c] = t[:, :c]
        return out

    def extract_labels(self, batch) -> dict:
        """Extract per-graph target tensors from a PyG batch."""
        batch_vec = getattr(batch, "batch", None)
        num_graphs = (int(batch_vec.max().item()) + 1) if batch_vec is not None and batch_vec.numel() > 0 else 1
        out_labels = self.out_labels
        dec = self.decoder_net
        sids = getattr(batch, "species_id", None)
        if sids is None and hasattr(batch, "y_species_id"):
            sids = getattr(batch, "y_species_id")

        bond_counts = (
            {sid: len(v) for sid, v in dec.species_bond_index.items()}
            if getattr(dec, "is_multispecies", False)
            else {}
        )
        angle_counts = (
            {sid: len(v) for sid, v in dec.species_angle_index.items()}
            if getattr(dec, "is_multispecies", False)
            else {}
        )
        torsion_counts = (
            {sid: len(v) for sid, v in dec.species_torsion_index.items()}
            if getattr(dec, "is_multispecies", False)
            else {}
        )

        cos_attr = "y_cos_torsions" if hasattr(batch, "y_cos_torsions") else "y_torsions_cos"
        sin_attr = "y_sin_torsions" if hasattr(batch, "y_sin_torsions") else "y_torsions_sin"

        key_map = [
            ("y_bonds", out_labels[0], getattr(dec, "max_bonds", 0), bond_counts),
            ("y_angles", out_labels[1], getattr(dec, "max_angles", 0), angle_counts),
            (cos_attr, out_labels[2], getattr(dec, "max_torsions", 0), torsion_counts),
            (sin_attr, out_labels[3], getattr(dec, "max_torsions", 0), torsion_counts),
        ]
        res = {}
        consumed_attrs = set()
        for attr_name, out_key, max_d, counts_map in key_map:
            if hasattr(batch, attr_name) and getattr(batch, attr_name) is not None:
                consumed_attrs.add(attr_name)
                t_src = getattr(batch, attr_name)
                if t_src.ndim >= 2:
                    eff_max = max_d if max_d > 0 else t_src.view(num_graphs, -1).shape[1]
                    res[out_key] = self._align_dict_target(
                        t_src, num_graphs, eff_max, counts_map, sids
                    )
                else:
                    eff_max = max_d if max_d > 0 else (t_src.numel() // max(1, num_graphs))
                    res[out_key] = self._slice_and_pad_attr(
                        batch, attr_name, num_graphs, eff_max, counts_map
                    )

        # Preserve any additional y_* attributes (e.g. y_phi_2, y_psi_2, y_species_id)
        for extra_k in batch.keys():
            if (
                extra_k not in consumed_attrs
                and extra_k.startswith("y_")
                and isinstance(batch[extra_k], torch.Tensor)
            ):
                res[extra_k] = batch[extra_k].view(num_graphs, -1)
                short_k = extra_k[2:]
                if short_k not in res:
                    res[short_k] = res[extra_k]

        if sids is not None and "species_id" not in res:
            res["species_id"] = sids.view(num_graphs, -1)
        return res

    def aggregate_losses(self, losses):
        loss = losses['encdec']
        if self.loss_latent_weight > 0.0:
            loss = loss + self.loss_latent_weight * losses['latent']
        return loss
    
__all__ = ["BondGraphEncoderDecoder"]

# ------------------------------------------------------------------
# For backward compatibility with old checkpoints
# ------------------------------------------------------------------
BondGraphNetEncoderDecoder = BondGraphEncoderDecoder 
BondGraphNetEncoder = BondGraphEncoder
BondGraphNetDecoder = BondGraphDecoder
from .modules.mp_modules import (
    ScalarFeatureEmbedding,
    AttentionMP,
)

__all__.extend(["BondGraphNetEncoderDecoder", "BondGraphNetEncoder", "BondGraphNetDecoder"])
