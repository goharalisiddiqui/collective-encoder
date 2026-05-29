from typing import Dict, Any

from collective_encoder.nets.ae_net import AE


class sAE(AE):
    _IDENTIFIER = "sAE"
    
    """
    Simple Autoencoder (sAE) with symmetric encoder and decoder architectures.
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
