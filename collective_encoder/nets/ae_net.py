from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from collective_encoder.nets.ae_base import AEBase
from collective_encoder.nets.modules.simple_encoder import SimpleNN


class AE(AEBase):
    _COMPATIBLE_DATASETS = ["DEFAULT", "DISTANCES", "SOAP", "SOAP_PS"]

    def __init__(self,
                 datamodule,
                 network: List[int],
                 normIn: Optional[bool] = False,
                 lrate: float = 0.01,
                 weight_decay: float = 1e-7,
                 scheduler: bool = True,
                 scheduler_args: dict = None,
                 outname: str = './AE_untitled/AE_',
                 test_plotter: str = "LDplotter",
                 export_latent: bool = False,
                 batch_norm: bool = True,
                 ):
        self.save_hyperparameters(ignore=['datamodule'])

        assert len(network) >= 2, "Network must have at least 2 entries (hidden..., latent)"
        assert datamodule.hparams.dataset_type in self._COMPATIBLE_DATASETS, (
            f"Dataset type '{datamodule.hparams.dataset_type}' is not compatible with AE. "
            f"Compatible types: {self._COMPATIBLE_DATASETS}"
        )

        nodes = [int(x) for x in network]
        datapoint_shape = datamodule.get_datapoint_shape()
        nodes.insert(0, datapoint_shape[0])

        super().__init__(dim_data=nodes[0],
                         dim_latent=nodes[-1],
                         normIn=normIn,
                         lrate=lrate,
                         weight_decay=weight_decay,
                         scheduler=scheduler,
                         scheduler_args=scheduler_args,
                         outname=outname,
                         test_plotter=test_plotter,
                         export_latent=export_latent,
                         )

        self.network = nodes
        self.init_network()

    def init_network(self) -> None:
        self.log_msg(f"[Initializing {type(self).__name__} Module] hidden layers: {self.network}")
        self.encoder_net = SimpleNN(layers=self.network, batch_norm=self.hparams.batch_norm)
        self.decoder_net = SimpleNN(layers=self.network[::-1], batch_norm=self.hparams.batch_norm)

