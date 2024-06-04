import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.distributions.normal import Normal
torch.manual_seed(0)
TORCH_PI = torch.acos(torch.zeros(1))*2


import pandas as pd
pd.set_option('display.max_columns', None)



import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from statistics import mean as list_mean

from scipy.stats import multivariate_normal

from nets.vae_net import VAE


class VAE_mse(VAE):
    def __init__(self,
                 l:list,
                 lr : float = 0.01,
                 l2_reg : float = 1e-7,
                 beta : float = 1.0,
                 batch_norm : bool = True,
                 lr_scheduler : bool = False,
                 plot_every : int = 0,
                 outname : str = './VAE_mse_untitled/VAE_mse_'):
        super().__init__(l, lr, l2_reg, beta, batch_norm, lr_scheduler, plot_every, outname)

        self.decoder_output = nn.Linear(l[1], l[0])
        print(l[1], " --> ", l[0], end=" ")
        print("(feature space)")
        print("======================")

    def decode(self, z):
        z = self.decoder_hidden(z)
        x_out = self.decoder_output(z)
        return x_out

    def forward(self, x):
        mu_latent, logvar_latent = self.encode(x)
        if mu_latent.isnan().any() or logvar_latent.isnan().any():
            print("Nan in encoder network (Gradient diminished or exploded). Can't continue")
            exit()

        if self.metaD:
            return mu_latent, logvar_latent

        z = self.reparametrize_multivariate(mu_latent, logvar_latent)

        x_out = self.decode(z)

        return x_out, {"mu_latent" : mu_latent, "logvar_latent" : logvar_latent, "z_sample" : z}

    def recon_loss(self, tru_x, recon_x):
        loss_rec = F.mse_loss(recon_x, tru_x, reduction='none')
        loss_rec = torch.mean(loss_rec, dim = 1)

        return loss_rec

    def loss(self, recon_x, tru_x, **kwargs):
        mu_latent = kwargs["mu_latent"]
        logvar_latent = kwargs["logvar_latent"]
        z_sample = kwargs["z_sample"]

        loss_rec = self.recon_loss(tru_x, recon_x)
        loss_reg = self.reg_loss(z_sample, mu_latent, logvar_latent)


        loss_rec = torch.mean(loss_rec) # Mean of batch
        loss_reg = torch.mean(loss_reg) # Mean of batch
        loss = loss_rec + self.hparams.beta * loss_reg

        if loss.isnan().any().detach().cpu().numpy():
            print("loss contains nan. Can't continue")
            exit()

        loss_mae = self.mae_loss(recon_x, tru_x)

        return {'loss' : loss, 'mae_loss' : loss_mae, 'rec_loss' : loss_rec, 'reg_loss' : loss_reg}



