import numpy as np
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from collective_encoder.nets.dvae_net import DVAE

class EDVAE(DVAE):
    _IDENTIFIER = "EDVAE"
    _COMPATIBLE_DATASETS = DVAE._COMPATIBLE_DATASETS + ['POSITIONS']
    _REQUIRED_ARGS = DVAE._REQUIRED_ARGS + ['embedding_type']
    _OPTIONAL_ARGS = DVAE._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'embedding_args': None,
    })
    
    def __init__(self,
                 datamodule,
                 args: Dict[str, Any] = None,
                 **kwargs
                 ):
        self.save_hyperparameters(ignore=['datamodule'])
        
        raw_shape = datamodule.get_datapoint_shape()
        self._raw_datapoint_shape = raw_shape
        
        super().__init__(datamodule=datamodule, args=args, **kwargs)
        
    def init_network(self) -> None:
        raw_shape = self._raw_datapoint_shape
        embedded_length = int(np.prod(raw_shape))
        
        self.network[0] = embedded_length

        if self.embedding_type == "flatten":
            self.embedding = nn.Flatten()
            embedded_length = int(np.prod(raw_shape))
            self.log_msg(f"  {raw_shape} --> {embedded_length} (flatten embedding)")

        super().init_network()

        if self.embedding_type == "flatten":
            self.deembedding = nn.Unflatten(1, raw_shape)
            self.log_msg(f"  {embedded_length} --> {raw_shape} (unflatten deembedding)")

    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x = self.embedding(x)
        z = self.encoder_net(x)
        return z, {}

    def decoder(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x_out = self.decoder_net(z)
        x_out = self.deembedding(x_out)
        return x_out, {}
