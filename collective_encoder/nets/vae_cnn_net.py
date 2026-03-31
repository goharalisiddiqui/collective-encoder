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

from collective_encoder.nets.vae_net import VAE
from collective_encoder.nets.vae_net import VAE_args
import argparse

def vaec_parse_args():
    desc = "VAE NN for enhanced sampling MD"
    parser = argparse.ArgumentParser(description=desc)


    parser.add_argument('--solvation', required=True, dest='lw', nargs='+', type=str, default= None, help='Grid size of the solvation grid')

    args, _ = parser.parse_known_args()

    return args


VAEC_args = vaec_parse_args()
VAEC_args = argparse.Namespace(**vars(VAE_args), **vars(VAEC_args))


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
                 saveplotdata : bool = False,
                 outname : str = './VAEC_untitled/VAEC_'):
        self.save_hyperparameters()
        lw = [int(i) for i in lw]
        assert len(lw) == 2 or len(lw) == 3, f"[{type(self).__name__} Module]: Number of solvation grid dimensions must be 2 or 3"
        self.solvation_dimensions = len(lw)
        self.n_solv = np.prod(lw)
        self.n_lin = l[0] -  self.n_solv
        assert l[0] == self.n_solv, f"[{type(self).__name__} Module]: Number of grid points and NN input does not match"
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
            saveplotdata = saveplotdata,
            outname = outname)

    def get_lw_shape(self):
        if self.solvation_dimensions == 2:
            return (self.hparams.lw[0], self.hparams.lw[1])
        elif self.solvation_dimensions == 3:
            return (self.hparams.lw[0], self.hparams.lw[1], self.hparams.lw[2])
    def get_conv_mod(self):
        if self.solvation_dimensions == 2:
            return nn.Conv2d
        elif self.solvation_dimensions == 3:
            return nn.Conv3d

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
        print("========= Encoder-Decoder NN =========")
        self.init_encoder()
        print("(Reparameterization Sampler)\n\n")
        self.init_decoder_layers()
        print("======================")
        print("")
        print("========= Output Conv NN =========")
        self.init_output_conv()
        print("=============================")
        print("")
        print("========= Output  NN =========")
        self.init_decoder_output()
        print("=============================")

    def init_input_conv(self):
        lw = self.hparams.lw
        lw_shape = self.get_lw_shape()
        conv_mod = self.get_conv_mod()
        n_hin = self.n_hin
        batch_norm = self.hparams.batch_norm
        conv_input_layers = []
        print(self.n_solv, " -unflatten-> ", lw)
        conv_input_layers.append(nn.Unflatten(1, (1,-1)))
        conv_input_layers.append(nn.Unflatten(2, lw_shape))
        print(lw, " -conv (1-10)-> ", lw, end=" ")
        conv_input_layers.append(conv_mod(1, 10, lw_shape, stride = 1, padding = 'same'))
        print("(relu)")
        conv_input_layers.append(nn.ReLU(True))
        print(lw, " -conv (10-20)-> ", lw, end=" ")
        conv_input_layers.append(conv_mod(10, 20, lw_shape, stride = 1, padding = 'same'))
        print("(relu)")
        conv_input_layers.append(nn.ReLU(True))
        print(lw, " -conv (20-10)-> ", lw, end=" ")
        conv_input_layers.append(conv_mod(20, 10, lw_shape, stride = 1, padding = 'same'))
        print("(relu)")
        conv_input_layers.append(nn.ReLU(True))
        print(lw, " -conv (10-1)-> ", lw, end=" ")
        conv_input_layers.append(conv_mod(10, 1, lw_shape, stride = 1, padding = 'same'))
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

    def init_output_conv(self):
        batch_norm = self.hparams.batch_norm
        lw_shape = self.get_lw_shape()
        conv_mod = self.get_conv_mod()
        l = self.hparams.l
        lw = self.hparams.lw
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
        conv_output_layers.append(nn.Unflatten(2, lw_shape))
        print(lw, " -conv (1 - 10)-> ", lw)
        conv_output_layers.append(conv_mod(1, 10, lw_shape, stride = 1, padding = 'same'))
        print("(relu)")
        conv_output_layers.append(nn.ReLU(True))
        print(lw, " -conv (10 - 20)-> ", lw)
        conv_output_layers.append(conv_mod(10, 20, lw_shape, stride = 1, padding = 'same'))
        print("(relu)")
        conv_output_layers.append(nn.ReLU(True))
        print(lw, " -conv (20 - 10)-> ", lw)
        conv_output_layers.append(conv_mod(20, 10, lw_shape, stride = 1, padding = 'same'))
        print("(relu)")
        conv_output_layers.append(nn.ReLU(True))
        print(lw, " -conv (10 - 1)-> ", lw)
        conv_output_layers.append(conv_mod(10, 1, lw_shape, stride = 1, padding = 'same'))
        print("(relu)")
        conv_output_layers.append(nn.ReLU(True))
        print(lw, " -flatten-> ", self.n_solv, end=" ")
        conv_output_layers.append(nn.Flatten())
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
        x = self.conv_input(x)
        x = self.encoder_hidden(x)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.decoder_hidden(z)
        x_h =  self.conv_output(z)
        x_mu = self.decoder_mu(x_h)
        x_logvar = self.decoder_mu(x_h)
        return x_mu, x_logvar




