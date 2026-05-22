from typing import Dict, Any

from collective_encoder.nets.vae_net import VAE


class sVAE(VAE):
    _IDENTIFIER = "sVAE"
    
    """
    Simple Variational Autoencoder (sVAE) with symmetric encoder and decoder architectures.
    The encoder and decoder architectures are determined by the provided network
    """

    def __init__(self,
                 network: list,
                 args: Dict[str, Any] = None,
                 **kwargs
                 ):
        self.save_hyperparameters()
        args['encoder_network'] = network
        args['decoder_network'] = network[:-1][::-1]
        super().__init__(args=args, **kwargs)
