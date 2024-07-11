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
from nets.vae_net import VAE_args as VAEC_args


class VAEC(VAE):
    def __init__(self,
                 l : list,
                 lw : list,
                 lr : float = 0.01,
                 l2_reg : float = 1e-7,
                 beta : float = 1.0,
                 batch_norm : bool = True,
                 lr_scheduler : bool = False,
                 plot_every : int = 0,
                 C_max : float = 0.0,
                 C_start : int = 0,
                 C_end : int = 0,
                 outname : str = './VAEC_untitled/VAEC_'):
        self.save_hyperparameters()
        assert len(lw) == 3
        self.n_solv = np.prod(lw)
        self.n_lin = l[0] -  self.n_solv
        assert self.n_lin >= 0, "Number of linear input nodes must be greater than or equal to zero"
        self.n_hin = l[1] - l[1]//2
        if self.n_lin == 0:
            self.n_hin = l[1]
        super().__init__(
            l = l,
            lr = lr,
            l2_reg = l2_reg,
            beta = beta,
            batch_norm = batch_norm,
            lr_scheduler = lr_scheduler,
            plot_every = plot_every,
            C_max = C_max,
            C_start = C_start,
            C_end = C_end,
            outname = outname)

    def init_network(self):
        l = self.hparams.l
        lw = self.hparams.lw
        n_hin = self.n_hin
        print(f"[Initializing {type(self).__name__} Module]")
        print("- hidden layers:", l)
        print("- solvation convolution layers:", lw)
        print("")
        print("========= Input Conv NN =========")
        self.init_input_conv()
        print("=============================")
        print("")
        print("========= Input Linear NN =========")
        self.init_input_lin()
        print("=============================")
        print("")
        print("========= Encoder-Decoder NN =========")
        self.init_encoder()
        print("(Reparameterization Sampler)\n\n")
        self.init_decoder_layers()
        print("======================")
        print("========= Output Linear NN =========")
        self.init_output_lin()
        print("=============================")
        print("========= Output Conv NN =========")
        self.init_output_conv()
        print("=============================")
        print("========= Output  NN =========")
        self.init_decoder_output()
        print("=============================")

    def init_input_conv(self):
        l = self.hparams.l
        lw = self.hparams.lw
        n_hin = self.n_hin
        batch_norm = self.hparams.batch_norm
        conv_input_layers = []
        print(self.n_solv, " -unflatten-> ", lw)
        conv_input_layers.append(nn.Unflatten(1, (1,-1)))
        conv_input_layers.append(nn.Unflatten(2, (lw[0], lw[1], lw[2])))
        print(lw, " -conv-> ", lw, end=" ")
        conv_input_layers.append(nn.Conv3d(1, 1, (lw[0], lw[1], lw[2]), stride = 1, padding = 'same'))
        print("(relu)")
        conv_input_layers.append(nn.ReLU(True))
        print(lw, " -flatten-> ", self.n_solv)
        conv_input_layers.append(nn.Flatten())
        if batch_norm:
            print("(batch_normalization layer)")
            conv_input_layers.append(nn.BatchNorm1d(self.n_solv))
        print(self.n_solv, " --> ", n_hin)
        conv_input_layers.append(nn.Linear(self.n_solv, n_hin))
        print("(relu)")
        conv_input_layers.append(nn.ReLU(True))
        self.conv_input = nn.Sequential(*conv_input_layers)

    def init_input_lin(self):
        batch_norm = self.hparams.batch_norm
        l = self.hparams.l
        if self.n_lin == 0:
            print("No linear input layer")
            self.lin_input = nn.Identity()
        else:
            lin_input_layers = []
            print(l[0] -  self.n_solv, " --> ", l[1]//2, end=" ")
            lin_input_layers.append(nn.Linear(l[0] -  self.n_solv, l[1]//2))
            print("(relu)")
            lin_input_layers.append(nn.ReLU(True))
            if batch_norm:
                print("(batch_normalization layer)")
                lin_input_layers.append(nn.BatchNorm1d(l[1]//2))
            self.lin_input = nn.Sequential(*lin_input_layers)

    def init_encoder_layers(self):
        l = self.hparams.l
        batch_norm = self.hparams.batch_norm
        encoder_layers = []
        for i in range(1, len(l) - 2):
            print(l[i], " --> ", l[i + 1], end=" ")
            encoder_layers.append(nn.Linear(l[i], l[i + 1]))
            encoder_layers.append(nn.ReLU(True))
            print("(relu)")
            if batch_norm:
                encoder_layers.append(nn.BatchNorm1d(l[i + 1]))
                print("(batch_normalization layer)")
        self.encoder_hidden = nn.Sequential(*encoder_layers)

    def init_output_lin(self):
        l = self.hparams.l
        if self.n_lin == 0:
            print("No linear output layer")
            self.lin_output = nn.Identity()
        else:
            lin_output_layers = []
            lin_output_layers.append(nn.Linear(l[1], self.n_lin))
            print(l[1], " --> ", self.n_lin, end=" ")
            lin_output_layers.append(nn.ReLU(True))
            print("(relu)")
            self.lin_output = nn.Sequential(*lin_output_layers)
            print("\n\n")

    def init_output_conv(self):
        batch_norm = self.hparams.batch_norm
        l = self.hparams.l
        lw = self.hparams.lw
        n_hin = self.n_hin
        conv_output_layers = []
        print(l[1], " --> ", self.n_solv)
        conv_output_layers.append(nn.Linear(l[1], self.n_solv))
        print("(relu)")
        conv_output_layers.append(nn.ReLU(True))
        if batch_norm:
            conv_output_layers.append(nn.BatchNorm1d(self.n_solv))
            print("(batch_normalization layer)")
        print(self.n_solv, " -unflatten-> ", lw)
        conv_output_layers.append(nn.Unflatten(1, (1,-1)))
        conv_output_layers.append(nn.Unflatten(2, (lw[0], lw[1], lw[2])))
        print(lw, " -conv-> ", lw)
        conv_output_layers.append(nn.Conv3d(1, 1, (lw[0], lw[1], lw[2]), stride = 1, padding = 'same'))
        print(lw, " -flatten-> ", self.n_solv, end=" ")
        conv_output_layers.append(nn.Flatten())
        print("(relu)")
        conv_output_layers.append(nn.ReLU(True))
        self.conv_output = nn.Sequential(*conv_output_layers)

    def init_decoder_output(self):
        l = self.hparams.l
        self.decoder_mu = nn.Linear(l[0], l[0])
        print(l[0], " --> ", l[0], end=" ")
        print("(mu for feature space)")
        self.decoder_logvar = nn.Linear(l[0], l[0])
        print( "  ", " \--> ", l[0], end=" ")
        print("(logvar for feature space)\n\n")
        print("======================")

    def encode(self, x):
        x = self.normalize(x)
        if self.n_lin == 0:
            x = self.conv_input(x)
        else:
            x_lin = self.lin_input(x[:,:x.size()[1] - self.n_solv])
            x_conv = self.conv_input(x[:,0:self.n_solv])
            x = torch.cat((x_lin, x_conv), dim=1)
        x = self.encoder_hidden(x)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.decoder_hidden(z)
        if self.n_lin == 0:
            x_h =  self.conv_output(z)
        else:
            z_lin = self.lin_output(z)
            z_conv = self.conv_output(z)
            x_h = torch.cat((z_lin, z_conv), dim=1)
        x_mu = self.decoder_mu(x_h)
        x_logvar = self.decoder_mu(x_h)
        return x_mu, x_logvar




