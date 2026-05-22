from typing import Dict, Any

from collective_encoder.nets.ae_base import AEBase
from collective_encoder.nets.modules.simple_nn import SimpleNN


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
        self.encoder_net = SimpleNN(layers=self.encoder_network, batch_norm=self.batch_norm)
        self.decoder_net = SimpleNN(layers=self.decoder_network, batch_norm=self.batch_norm)

    def print_hparams(self):
        super().print_hparams()
        self.log_msg(f"  Network architecture: {self.encoder_network} -> {self.decoder_network}")

