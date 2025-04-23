import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ae_base import AEBase

TORCH_PI = torch.acos(torch.zeros(1))*2
EPSILON = 1e-9

torch.manual_seed(0)
torch.set_printoptions(threshold=10_000)


class AE(AEBase):
    def __init__(self,
                 l: list,
                 lr : float = 0.01,
                 l2_reg : float = 1e-7,
                 batch_norm : bool = True,
                 lr_scheduler : bool = True,
                 plot_every : int = 0,
                 saveplotdata : bool = False,
                 outname : str = './AE_untitled/AE_'):
        super().__init__(dim_data = l[0],
                         dim_latent = l[-1],
                         lr = lr,
                         l2_reg = l2_reg,
                         lr_scheduler = lr_scheduler,
                         outname = outname,
                         plot_every = plot_every,
                         saveplotdata = saveplotdata)
        assert len(l) >= 3
        self.save_hyperparameters()
        #### Setting up the layers of the netwrok ####
        self.init_network()


    def init_network(self):
        print(f"[Initializing {type(self).__name__} Module]")
        print("- hidden layers:", self.hparams.l)
        print("")
        print("========= NN =========")
        self.init_encoder()
        print("")
        self.init_decoder()
        print("======================")

    def init_encoder(self):
        l = self.hparams.l
        batch_norm = self.hparams.batch_norm
        encoder_layers = []
        for i in range(len(l) - 2):
            print(l[i], " --> ", l[i + 1], end=" ")
            encoder_layers.append(nn.Linear(l[i], l[i + 1]))
            encoder_layers.append(nn.ReLU(True))
            print("(relu)")
            if batch_norm:
                encoder_layers.append(nn.BatchNorm1d(l[i + 1]))
                print("(batch_normalization layer)")
        print(l[-2], " --> ", l[-1], end=" ")
        encoder_layers.append(nn.Linear(l[-2], l[-1]))
        self.encoder = nn.Sequential(*encoder_layers)

    def init_decoder(self):
        l = self.hparams.l
        batch_norm = self.hparams.batch_norm
        decoder_layers = []
        a = len(l) - 1
        for i in range(len(l) - 2):
            print(l[a - i], " --> ", l[a - i - 1], end=" ")
            decoder_layers.append(nn.Linear(l[a- i], l[a - i - 1]))
            decoder_layers.append(nn.ReLU(True))
            print("(relu)")
            if batch_norm:
                decoder_layers.append(nn.BatchNorm1d(l[a - i - 1]))
                print("(batch_normalization layer)")
        print(l[1], " --> ", l[0], end=" ")
        decoder_layers.append(nn.Linear(l[1], l[0]))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        x = self.normalize(x)
        z = self.encode(x)
        if self.metaD:
            return z, z # Because our plumed module expects a tuple
        x_out = self.decode(z)
        x_out = self.denormalize(x_out)

        return x_out, {"z" : z}

    def mae_loss(self, recon_x, tru_x):
        loss_mae = F.l1_loss(recon_x, tru_x, reduction='none')
        loss_mae = torch.mean(loss_mae, dim = 1)
        loss_mae = torch.mean(loss_mae, dim = 0)

        return loss_mae

    def loss(self, recon_x, tru_x, **kwargs):

        loss = F.mse_loss(recon_x, tru_x)

        if loss.isnan().any().detach().cpu().numpy():
            print("loss contains nan. Can't continue")
            exit()

        loss_mae = self.mae_loss(recon_x, tru_x)

        return {'loss' : loss,
                    'mae_loss' : loss_mae}

    def get_latent(self, data_x):
        latent = self.encode(data_x)
        return latent.detach().cpu().numpy()






