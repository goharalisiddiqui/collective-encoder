from typing import Dict, Any

from collective_encoder.models.neural_nets.ae_base import AEBase
from collective_encoder.models.neural_nets.modules.simple_nn import SimpleNN


class AE(AEBase):
    _IDENTIFIER = "AE"
    _COMPATIBLE_DATASETS = ["DEFAULT", "DISTANCES", "SOAP", "SOAP_PS"]

    def __init__(self,
                args: Dict[str, Any] = None,
                 **kwargs
                ):
        self.save_hyperparameters()
        super().__init__(args=args, **kwargs)

    def init_network(self) -> None:
        self.encoder_net = SimpleNN(layers=self.encoder_network, 
                                    batch_norm=self.batch_norm,
                                    activation=self.activation,
                                    activation_args=self.activation_args)
        self.decoder_net = SimpleNN(layers=self.decoder_network, 
                                    batch_norm=self.batch_norm, 
                                    activation=self.activation,
                                    activation_args=self.activation_args)

    def print_hparams(self):
        super().print_hparams()
        self.log_msg(f"  Network architecture: {self.encoder_network} -> {self.decoder_network}")

# ------------------------------------------------------------------
# Symmetric Autoencoder (sAE) subclass with symmetric encoder and decoder architectures
# ------------------------------------------------------------------

class sAE(AE):
    _IDENTIFIER = "sAE"
    
    """
    Symmetric Autoencoder (sAE) with symmetric encoder and decoder architectures.
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
