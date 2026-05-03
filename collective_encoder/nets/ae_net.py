from typing import Dict, Any

from collective_encoder.nets.ae_base import AEBase
from collective_encoder.nets.modules.simple_encoder import SimpleNN


class AE(AEBase):
    _IDENTIFIER = "AE"
    _COMPATIBLE_DATASETS = ["DEFAULT", "DISTANCES", "SOAP", "SOAP_PS"]

    def __init__(self,
                 datamodule,
                 args: Dict[str, Any] = None,
                 **kwargs
                 ):
        self.save_hyperparameters(ignore=['datamodule'])
        super().__init__(datamodule=datamodule, args=args, **kwargs)

    def init_network(self) -> None:
        self.encoder_net = SimpleNN(layers=self.network, batch_norm=self.batch_norm)
        self.decoder_net = SimpleNN(layers=self.network[::-1], batch_norm=self.batch_norm)
    
    def print_hparams(self):
        super().print_hparams()
        self.log_msg(f"  Network architecture: {self.network}")

