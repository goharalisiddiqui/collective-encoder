from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from collective_encoder.nets.modules.variational_encoder import VariationalNN
from collective_encoder.nets.modules.simple_encoder import SimpleNN

from collective_encoder.nets.vae_net import VAE


class DVAE(VAE):
    _IDENTIFIER = "DVAE"

    def init_network(self):
        self.log_msg(f"[Initializing {type(self).__name__} Module] hidden layers: {self.network}")
        self.print_hparams()
        self.encoder_net = VariationalNN(layers=self.network, batch_norm=self.batch_norm)
        self.decoder_net = SimpleNN(layers=self.network[::-1], batch_norm=self.batch_norm)

    def decoder(self, z):
        z = self.decoder_net(z)
        return z, {}

    def recon_loss(self, inp, latent, output, labels, meta):
        loss_rec = F.mse_loss(output, inp, reduction='none')
        loss_rec = torch.mean(loss_rec, dim = 1)
        loss_rec = torch.mean(loss_rec)

        return loss_rec, {}


