import pandas as pd
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.encoders.variational_encoder import VariationalNN
from nets.encoders.simple_encoder import SimpleNN

from nets.vae_net import VAE

class DVAE(VAE):
    def __init__(self,
                 datamodule,
                 network: List[int],
                 normIn: Optional[bool] = False,
                 lrate: float = 0.01,
                 weight_decay: float = 1e-7,
                 scheduler: bool = True,
                 scheduler_args : dict = {},
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
            use_steric_loss=use_steric_loss,
            use_bond_deviation_loss=use_bond_deviation_loss,
        )

        self.losses = {
            "rec_loss": self.loss_mse,
        }

    def init_network(self):
        print(f"[Initializing {type(self).__name__} Module]")
        print("- hidden layers:", self.network)
        self.print_hparams()
        print("")
        print("========= NN =========")
        self.encoder_net = VariationalNN(layers=self.network, batch_norm=self.hparams.batch_norm)
        print("(Reparameterization Sampler)\n\n")
        self.decoder_net = SimpleNN(layers=self.network[::-1], batch_norm=self.hparams.batch_norm)
        print("======================")

    def decoder(self, z):
        z = self.decoder_net(z)
        return z, {}

    def recon_loss(self, tru_x, recon_x):
        loss_rec = F.mse_loss(recon_x, tru_x, reduction='none')
        loss_rec = torch.mean(loss_rec, dim = 1)

        return loss_rec, {}


