from typing import Any, Dict, List, Optional, Tuple

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
        self.encoder_net = VariationalNN(layers=self.encoder_network, 
                                         batch_norm=self.batch_norm,
                                         activation=self.activation,
                                         activation_args=self.activation_args)
        self.decoder_net = SimpleNN(layers=self.decoder_network, 
                                    batch_norm=self.batch_norm,
                                    activation=self.activation,
                                    activation_args=self.activation_args)

    def decoder(self, z):
        z = self.decoder_net(z)
        return z, {}

# ------------------------------------------------------------------
# Symmetric Deterministic Variational Autoencoder (sDVAE) subclass with symmetric encoder and decoder architectures
# ------------------------------------------------------------------

class sDVAE(DVAE):
    _IDENTIFIER = "sDVAE"
    
    """
    Symmetric Deterministic Variational Autoencoder (sDVAE) with symmetric encoder and decoder architectures.
    The encoder and decoder architectures are determined by the provided network
    """

    def __init__(self,
                 args: Dict[str, Any] = None,
                 **kwargs
                 ):
        self.save_hyperparameters()
        network = args.pop('network', None)
        if network is None:
            raise ValueError("Argument 'network' is required for sVAE")
        args['encoder_network'] = network
        args['decoder_network'] = network[:-1][::-1]
        super().__init__(args=args, **kwargs)