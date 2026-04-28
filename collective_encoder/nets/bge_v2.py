from typing import List, Optional, Dict, Union

import torch
from torch import nn

from collective_encoder.nets.base import CENetBase
from collective_encoder.nets.bge import BondGraphEncoderDecoder
from .modules.graph_encoder import BondGraphEncoderV2 as BondGraphEncoder
from .modules.graph_decoder import BondGraphDecoder


class BondGraphEncoderDecoderV2(BondGraphEncoderDecoder):
    """BGE v2 — uses :class:`BondGraphEncoderV2` and always requires a datamodule.

    Inherits all loss, step, normalize, and scheduler logic from
    :class:`BondGraphEncoderDecoder`.  Only the encoder and eager decoder
    initialization differ.
    """

    def __init__(
        self,
        datamodule,
        encoder_args: Dict[str, Union[int, float]],
        decoder_args: Dict[str, Union[int, float]],
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
        # Skip BondGraphEncoderDecoder.__init__; call CENetBase for dual-init.
        CENetBase.__init__(self)

        if out_labels is None:
            out_labels = ['bond_dist', 'angle', 'dihedral_cos', 'dihedral_sin']
        assert datamodule is not None, "datamodule must be provided"
        assert len(out_labels) == 4, "out_labels must be a list of 4 strings"

        assert loss_weights is None or len(loss_weights) == 4, \
            "loss_weights must be None or a list of 4 floats"
        if loss_weights is None:
            loss_weights = [1.0, 1.0, 1.0, 1.0]
        self.loss_weights = loss_weights
        self.latent_dim = encoder_args['latent_dim']

        datasetobject = datamodule.get_dataset()
        self.template_khop = decoder_args['template_khop']
        template_data = datasetobject.get_template_graph(k=self.template_khop)
        bond_index, angle_index, torsion_index = datasetobject.get_label_indices()

        gnn_dec_kwargs = {
            "template_data": template_data,
            "label_indices": (bond_index, angle_index, torsion_index),
            "latent_dim": self.latent_dim,
        }
        gnn_dec_args = decoder_args.copy()
        gnn_dec_args.pop('template_khop', None)

        self.gnn_enc = BondGraphEncoder(**encoder_args)
        self.gnn_dec = BondGraphDecoder(**gnn_dec_kwargs, **gnn_dec_args)
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.register_buffer('normIn', torch.tensor(normIn, dtype=torch.bool))
        self.register_buffer('normSet', torch.tensor(False, dtype=torch.bool))
        num_norm = encoder_args['node_feat'] + encoder_args['edge_feat']
        self.register_buffer('Mean', torch.zeros(num_norm))
        self.register_buffer('Range', torch.ones(num_norm))


__all__ = ["BondGraphEncoderDecoderV2"]
