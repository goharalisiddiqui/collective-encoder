from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from collective_encoder.nets.modules.variational_encoder import VariationalNN
from collective_encoder.nets.modules.simple_encoder import SimpleNN

from collective_encoder.nets.vae_net import VAE


class DVAE(VAE):
    def __init__(self,
                 datamodule,
                 network: List[int],
                 normIn: Optional[bool] = False,
                 lrate: float = 0.01,
                 weight_decay: float = 1e-7,
                 scheduler: bool = True,
                 scheduler_args: dict = None,
                 outname: str = './VAE_untitled/DVAE_',
                 test_plotter : str = "LDplotter",
                 export_latent : bool = False,
                 beta: float = 1.0,
                 batch_norm: bool = True,
                 C_reg: Optional[Tuple[float, int, int]] = None,
                 C_auto: bool = False,
                 D_reg: Optional[float] = None,
                 use_steric_loss = False,
                 use_bond_deviation_loss = False,
                 ):
        super().__init__(
            datamodule=datamodule,
            network=network,
            normIn=normIn,
            lrate=lrate,
            weight_decay=weight_decay,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
            outname=outname,
            test_plotter=test_plotter,
            export_latent=export_latent,
            beta=beta,
            batch_norm=batch_norm,
            C_reg=C_reg,
            C_auto=C_auto,
            D_reg=D_reg,
            use_steric_loss=use_steric_loss,
            use_bond_deviation_loss=use_bond_deviation_loss,
        )


    def init_network(self):
        self.log_msg(f"[Initializing {type(self).__name__} Module] hidden layers: {self.network}")
        self.print_hparams()
        self.encoder_net = VariationalNN(layers=self.network, batch_norm=self.hparams.batch_norm)
        self.decoder_net = SimpleNN(layers=self.network[::-1], batch_norm=self.hparams.batch_norm)

    def decoder(self, z):
        z = self.decoder_net(z)
        return z, {}

    def recon_loss(self, x, latent, pred, meta):
        loss_rec = F.mse_loss(pred, x, reduction='none')
        loss_rec = torch.mean(loss_rec, dim = 1)
        loss_rec = torch.mean(loss_rec)

        return loss_rec, {}


