import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.distributions.normal import Normal
torch.manual_seed(0)
TORCH_PI = torch.acos(torch.zeros(1))*2
import argparse


import pandas as pd
pd.set_option('display.max_columns', None)


from ae_base import AEBase

EPSILON = 1e-9

torch.set_printoptions(threshold=10_000)

def vae_parse_args():
    desc = "VAE NN for enhanced sampling MD"
    parser = argparse.ArgumentParser(description=desc)


    parser.add_argument('--cmax', type=float, default = 0.0, dest='C_max', help='Maximum C for information control in Beta-VAE')
    parser.add_argument('--cstart', type=int, default = 0, dest='C_start', help='Epoch where C start to increase for information control in Beta-VAE')
    parser.add_argument('--cend', type=int, default = 0, dest='C_end', help='Epoch where C stops to increase for information control in Beta-VAE')
    parser.add_argument('--beta', required=True, type=float, help='beta for the beta-VAE')

    args, _ = parser.parse_known_args()

    return args


VAE_args = vae_parse_args()

class VAE(AEBase):
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
                 saveplotdata : bool = False,
                 outname : str = './LITcollVAE_untitled/LITcollVAE_'):
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
        print("(Reparameterization Sampler)\n\n")
        self.init_decoder()
        print("======================")

    def init_encoder(self):
        self.init_encoder_layers()
        self.init_encoder_output()

    def init_decoder(self):
        self.init_decoder_layers()
        self.init_decoder_output()

    def init_encoder_layers(self):
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
        self.encoder_hidden = nn.Sequential(*encoder_layers)

    def init_decoder_layers(self):
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
        self.decoder_hidden = nn.Sequential(*decoder_layers)

    def init_encoder_output(self):
        l = self.hparams.l
        self.encoder_mu = nn.Linear(l[-2], l[-1])
        print(l[-2], " --> ", l[-1], end=" ")
        print("(mu for latent space)")
        self.encoder_logvar = nn.Linear(l[-2], l[-1])
        print( "  ", " \--> ", l[-1], end=" ")
        print("(logvar for latent space)\n\n")

    def init_decoder_output(self):
        l = self.hparams.l
        self.decoder_mu = nn.Linear(l[1], l[0])
        print(l[1], " --> ", l[0], end=" ")
        print("(mu for feature space)")
        self.decoder_logvar = nn.Linear(l[1], l[0])
        print( "  ", " \--> ", l[0], end=" ")
        print("(logvar for feature space)\n\n")
        print("======================")

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
        x = self.normalize(x)
        mu_latent, logvar_latent = self.encode(x)
        if self.metaD:
            return mu_latent, logvar_latent
        if mu_latent.isnan().any() or logvar_latent.isnan().any():
            print("Nan in encoder network (Gradient diminished or exploded). Can't continue")
            exit()

        z = self.reparametrize_multivariate(mu_latent, logvar_latent)

        mu_x, logvar_x = self.decode(z) # q(x|z)
        if mu_x.isnan().any() or logvar_x.isnan().any():
            print("Nan in decoder network (Gradient diminished or exploded). Can't continue")
            exit()
        x_out = self.reparametrize_multivariate(mu_x, logvar_x)

        return x_out, {"mu_latent" : mu_latent, "logvar_latent" : logvar_latent,
                    "mu_x" : mu_x, "logvar_x" : logvar_x, "z_sample" : z}



    def kld(self, mu, logvar):
        # KLD between univariate gaussian to Standard, explanation here:
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # Second Gaussian is zero mean and variance of 1, the prior on z
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1) # sum for all the latent variables

        return kld

    def recon_loss(self, tru_x, mu_x, logvar_x):

        sd = torch.exp(0.5 * logvar_x) + EPSILON
        p_x = Normal(mu_x, sd)
        loss_rec = -torch.mean(p_x.log_prob(tru_x), axis=1)


        # if (loss_rec < 0.0).any().detach().cpu().numpy():
        #     var = 0.5 * torch.exp(logvar_x)
        #     print("\n loss_rec= ", loss_rec)
        #     print("\n var= ", var)
        #     print("\n tru_x= ", tru_x)
        #     print("\n log_prob= ", p_x.log_prob(tru_x))
        #     exit()

        return loss_rec

    def reg_loss(self, z_sample, mu_latent, logvar_latent):
        loss_kld = self.kld(mu_latent, logvar_latent)
        loss_kld = torch.mean(loss_kld, dim = 0)

        C = 0.0
        if self.hparams.C_max != 0.0:
            c_start, c_end, cmax = self.hparams.C_start, self.hparams.C_end, self.hparams.C_max
            if self.current_epoch >= c_start and self.current_epoch <= c_end:
                C = cmax * (self.current_epoch - c_start) / (c_end - c_start)
            elif self.current_epoch > c_end:
                C = cmax
        loss_reg = torch.abs(loss_kld - C)

        return loss_reg, {"current_C" : C, "kld" : loss_kld}

    def mae_loss(self, recon_x, tru_x):
        tru_x = self.denormalize(tru_x)
        recon_x = self.denormalize(recon_x)

        loss_mae = F.l1_loss(recon_x, tru_x, reduction='none')
        loss_mae = torch.mean(loss_mae, dim = 1)
        loss_mae = torch.mean(loss_mae, dim = 0)

        return loss_mae

    def loss(self, recon_x, tru_x, **kwargs):
        mu_latent = kwargs["mu_latent"]
        logvar_latent = kwargs["logvar_latent"]
        mu_x = kwargs["mu_x"]
        logvar_x = kwargs["logvar_x"]
        z_sample = kwargs["z_sample"]

        loss_rec = self.recon_loss(tru_x, mu_x, logvar_x)
        loss_reg, meta_reg = self.reg_loss(z_sample, mu_latent, logvar_latent)

        loss_rec = torch.mean(loss_rec) # Mean of batch
        loss_reg = torch.mean(loss_reg) # Mean of batch
        loss = loss_rec + self.hparams.beta * loss_reg

        if loss.isnan().any().detach().cpu().numpy():
            print("loss contains nan. Can't continue")
            exit()

        loss_mae = self.mae_loss(recon_x, tru_x)

        return {'loss' : loss,
                    'mae_loss' : loss_mae,
                    'rec_loss' : loss_rec,
                    'reg_loss' : loss_reg,
                    "current_C" : meta_reg["current_C"],
                    "kld" : meta_reg["kld"]}

    def plot_avg_sigma(self, latent_logvar):
        # This implements any extra printing or plotting in child class
        ld_mean = np.mean(np.exp(0.5 * latent_logvar), axis = 0)
        print("========= Avg. Sigma per LD =========")
        for i in range(len(ld_mean)):
            print(f"LD {i} : {ld_mean[i]}")
        print("=====================================")

    def plot_extra(self, data_x, data_y, latents):
        latent_logvar = latents[1]
        self.plot_latent(latents, data_y, self.plot_sd, "latent_pdf")
        self.plot_avg_sigma(latent_logvar)

    def get_latent(self, data_x):
        latent_mu, latent_logvar = self.encode(data_x)
        return latent_mu.detach().cpu().numpy(), latent_logvar.detach().cpu().numpy()

    def get_latent_names(self):
        return "mu_latent", "logvar_latent"

    def plot_sd(fig, ax, latents, train_y, i, yaxis, label, scalarMap):
        latent_mu, latent_sd = latents
        ax.errorbar(latent_mu[:, i], latent_mu[:, yaxis],xerr=latent_sd[:,i],yerr=latent_sd[:,yaxis], ecolor=scalarMap.to_rgba(train_y) if train_y is not None else None, alpha=0.1, ls='none')






