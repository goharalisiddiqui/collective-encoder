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

from nets.ae_base import AEBase


class VAE(AEBase):
    def __init__(self,
                 l:list,
                 lr : float = 0.01,
                 l2_reg : float = 1e-7,
                 beta : float = 1.0,
                 n_samples : int = 1,
                 lr_scheduler : bool = False,
                 outname : str = './LITcollVAE_untitled/LITcollVAE_'):
        super().__init__(l[0], l[-1], lr, l2_reg, lr_scheduler, outname)
        assert len(l) >= 3

        #### Setting up the layers of the netwrok ####
        print(f"[Initializing {type(self).__name__} Module]")
        print("- hidden layers:", l)
        print("")
        print("========= NN =========")
        encoder_layers = []
        for i in range(len(l) - 2):
            print(l[i], " --> ", l[i + 1], end=" ")
            encoder_layers.append(nn.Linear(l[i], l[i + 1]))
            encoder_layers.append(nn.ReLU(True))
            print("(relu)")
            encoder_layers.append(nn.BatchNorm1d(l[i + 1]))
            print("(batch_normalization layer)")
        self.encoder_hidden = nn.Sequential(*encoder_layers)
        self.encoder_mu = nn.Linear(l[-2], l[-1])
        print(l[-2], " --> ", l[-1], end=" ")
        print("(mu for latent space)")
        self.encoder_logvar = nn.Linear(l[-2], l[-1])
        print( "  ", " \--> ", l[-1], end=" ")
        print("(logvar for latent space)\n\n")

        print("(Reparameterization Sampler)\n\n")

        decoder_layers = []
        a = len(l) - 1
        for i in range(len(l) - 2):
            print(l[a - i], " --> ", l[a - i - 1], end=" ")
            decoder_layers.append(nn.Linear(l[a- i], l[a - i - 1]))
            decoder_layers.append(nn.ReLU(True))
            print("(relu)")
            decoder_layers.append(nn.BatchNorm1d(l[a - i - 1]))
            print("(batch_normalization layer)")
        self.decoder_hidden = nn.Sequential(*decoder_layers)

        if type(self).__name__ == "VAE":
            self.decoder_mu = nn.Linear(l[1], l[0])
            print(l[1], " --> ", l[0], end=" ")
            print("(mu for feature space)")
            self.decoder_logvar = nn.Linear(l[1], l[0])
            print( "  ", " \--> ", l[0], end=" ")
            print("(logvar for feature space)\n\n")
            print("======================")


        self.save_hyperparameters()

    def print_hparams(self):
        print("- Beta \t=", self.hparams.beta)

    def encode(self, x):
        x = self.encoder_hidden(x)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.decoder_hidden(z)
        mu_x = self.decoder_mu(z)
        logvar_x = self.decoder_logvar(z)
        return mu_x, logvar_x

    def forward(self, x):
        mu_latent, logvar_latent = self.encode(x)
        if mu_latent.isnan().any() or logvar_latent.isnan().any():
            print("Nan in encoder network (Gradient diminished or exploded). Can't continue")
            exit()
        if self.metaD:
            return mu_latent, logvar_latent

        z = self.reparametrize(mu_latent, logvar_latent)

        mu_x, logvar_x = self.decode(z) # q(x|z)
        if mu_x.isnan().any() or logvar_x.isnan().any():
            print("Nan in decoder network (Gradient diminished or exploded). Can't continue")
            exit()
        x_out = self.reparametrize(mu_x, logvar_x)

        return x_out, {"mu_latent" : mu_latent, "logvar_latent" : logvar_latent,
                    "mu_x" : mu_x, "logvar_x" : logvar_x, "z_sample" : z}



    def kld(self, mu, logvar):
        # KLD between univariate gaussian to Standard, explanation here:
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # Second Gaussian is zero mean and variance of 1, the prior on z
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1) # sum for all the latent variables

        return kld

    def recon_loss(self, tru_x, mu_x, logvar_x):

        p_x = Normal(mu_x, torch.exp(logvar_x))
        loss_rec = -torch.mean(p_x.log_prob(tru_x), axis=1)

        return loss_rec

    def reg_loss(self, z_sample, mu_latent, logvar_latent):

        loss_kld = self.kld(mu_latent, logvar_latent)

        cov = torch.cov(z_sample.T)
        loss_cov = (cov * (1 - torch.eye(z_sample.shape[1]))).flatten().mean() * 0.5
        loss_cov = torch.abs(loss_cov)

        loss_reg = loss_kld + torch.abs(loss_cov)

        return loss_reg



    def loss(self, recon_x, tru_x, **kwargs):
        mu_latent = kwargs["mu_latent"]
        logvar_latent = kwargs["logvar_latent"]
        mu_x = kwargs["mu_x"]
        logvar_x = kwargs["logvar_x"]
        z_sample = kwargs["z_sample"]

        loss_rec = self.recon_loss(tru_x, mu_x, logvar_x)
        loss_reg = self.reg_loss(z_sample, mu_latent, logvar_latent)

        loss_rec = torch.mean(loss_rec) # Mean of batch
        loss_reg = torch.mean(loss_reg) # Mean of batch
        loss = loss_rec + self.hparams.beta * loss_reg

        if loss.isnan().any().detach().cpu().numpy():
            print("loss contains nan. Can't continue")
            exit()

        return {'loss' : loss, 'rec_loss' : loss_rec, 'reg_loss' : loss_reg}







