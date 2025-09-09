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
from nets.vae_net import VAE_args as DVAE_args



class DVAE(VAE):
    def __init__(self,
                 l: list,
                 lr : float = 0.01,
                 l2_reg : float = 1e-7,
                 beta : float = 1.0,
                 batch_norm : bool = True,
                 lr_scheduler : bool = True,
                 plot_every : int = 0,
                 C_max : float = 0.0,
                 C_start : int = 0,
                 C_end : int = 0,
                 C_auto : bool = False,
                 D : float = np.inf,
                 atomic_numbers = None,
                 bond_indices = None,
                 use_steric_loss = False,
                 use_bond_deviation_loss = False,
                 saveplotdata : bool = False,
                 outname : str = './DVAE_untitled/DVAE_',
                 ):
        super().__init__(l = l,
                         lr = lr,
                         l2_reg = l2_reg,
                         beta = beta,
                         batch_norm = batch_norm,
                         lr_scheduler = lr_scheduler,
                         plot_every = plot_every,
                         C_max = C_max,
                         C_start = C_start,
                         C_end = C_end,
                         C_auto = C_auto,
                         D = D,
                         atomic_numbers = atomic_numbers,
                         bond_indices = bond_indices,
                         use_steric_loss = use_steric_loss,
                         use_bond_deviation_loss = use_bond_deviation_loss,
                         saveplotdata = saveplotdata,
                         outname = outname,
                         )

    def init_decoder_output(self):
        l = self.hparams.l
        self.decoder_output = nn.Linear(l[1], l[0])
        print(l[1], " --> ", l[0], end=" ")
        print("(feature space)")

    def decode(self, z):
        z = self.decoder_hidden(z)
        x_out = self.decoder_output(z)
        return x_out

    def forward(self, x):
        x = self.normalize(x)
        mu_latent, logvar_latent = self.encode(x)
        if self.metaD:
            return mu_latent, logvar_latent
        if mu_latent.isnan().any() or logvar_latent.isnan().any():
            print("Nan in encoder network (Gradient diminished or exploded). Can't continue")
            exit()
        z = self.reparametrize_multivariate(mu_latent, logvar_latent)
        x_out = self.decode(z)
        x_out = self.denormalize(x_out)

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
        loss_reg, meta_reg = self.reg_loss(z_sample, mu_latent, logvar_latent)


        loss_rec = torch.mean(loss_rec) # Mean of batch
        loss_reg = torch.mean(loss_reg) # Mean of batch
        loss = loss_rec + self.hparams.beta * loss_reg

        # if loss.isnan().any().detach().cpu().numpy():
        #     print("loss contains nan. Can't continue")
        #     exit()
        if self.hparams.use_bond_deviation_loss:
            loss_bond = self.bond_deviation_loss(recon_x)
            loss += loss_bond
        
        if self.hparams.use_steric_loss:
            loss_steric = self.steric_loss(recon_x)
            loss += loss_steric * 100

        loss_mae = self.mae_loss(recon_x, tru_x)
        loss_mae = torch.mean(loss_mae)
        
        return_val  = {'loss': loss,
                'mae_loss': loss_mae,
                'rec_loss': loss_rec,
                'reg_loss': loss_reg,
                "current_C": meta_reg["current_C"],
                "kld": meta_reg["kld"]}
        
        if self.hparams.use_bond_deviation_loss:
            return_val["bond_deviation_loss"] = loss_bond
        
        if self.hparams.use_steric_loss:
            return_val["steric_loss"] = loss_steric
        

        return return_val


