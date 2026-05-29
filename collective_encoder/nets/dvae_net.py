from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from collective_encoder.nets.modules.variational_nn import VariationalNN
from collective_encoder.nets.modules.simple_nn import SimpleNN

from collective_encoder.losses.mse import CELossMSE

from collective_encoder.nets.vae_net import VAE


class DVAE(VAE):
    _IDENTIFIER = "DVAE"

    def __init__(self, args = None, **kwargs):
        super().__init__(args, **kwargs)
        
        self.losses['rec_loss'] = CELossMSE({}, **kwargs)

    def init_network(self):
        self.encoder_net = VariationalNN(layers=self.network, batch_norm=self.batch_norm)
        self.decoder_net = SimpleNN(layers=self.network[::-1], batch_norm=self.batch_norm)

    def decoder(self, z):
        z = self.decoder_net(z)
        return z, {}
