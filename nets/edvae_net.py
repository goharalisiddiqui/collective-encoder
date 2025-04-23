import argparse

import numpy as np

import pandas as pd
pd.set_option('display.max_columns', None)

import torch
import torch.nn as nn
torch.manual_seed(0)
TORCH_PI = torch.acos(torch.zeros(1))*2

from nets.dvae_net import DVAE
from nets.dvae_net import DVAE_args

def edvae_parse_args():
    desc = "Embedded Deterministic VAE Module"
    parser = argparse.ArgumentParser(description=desc)


    parser.add_argument('--emb_type', dest= "embedding_type", required=True, type=str, help='Type of embedding for Embedded Deterministic VAE Module')

    args, _ = parser.parse_known_args()

    res_args = DVAE_args()

    return argparse.Namespace(**vars(args), **vars(res_args))

EDVAE_args = edvae_parse_args

class EDVAE(DVAE):
    def __init__(self,
                 l: list,
                 embedding_type : str,
                 datapoint_shape : tuple,
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
                 outname : str = './EDVAE_untitled/EDVAE_'):

        self.save_hyperparameters()

        super().__init__(l,
                         lr,
                         l2_reg,
                         beta,
                         batch_norm,
                         lr_scheduler,
                         plot_every,
                         C_max,
                         C_start,
                         C_end,
                         saveplotdata,
                         outname = outname)

        self.Mean = torch.zeros(self.hparams.l[0])
        self.Range = torch.ones(self.hparams.l[0])

    def init_network(self):
        print(f"[Initializing {type(self).__name__} Module]")
        print("- Hidden layers:", self.hparams.l)
        print("- Embedding type:", self.hparams.embedding_type)
        print("")
        print("========= NN =========")
        self.init_embedding()
        self.init_encoder()
        print("")
        self.init_decoder()
        self.init_deembedding()
        print("======================")

    def set_norm(self, Mean: torch.Tensor, Range: torch.Tensor):
        Range[Range == 0.0] = 1.0
        print(f"[{type(self).__name__}] Setting normalization for inputs.")
        self.normIn = True
        if self.hparams.embedding_type == "flatten":
            self.Mean = Mean.flatten()
            self.Range = Range.flatten()
    
    def init_embedding(self):
        l = self.hparams.l
        datapoint_shape = self.hparams.datapoint_shape

        if self.hparams.embedding_type == "flatten":
            self.embedding = nn.Flatten()
            self.embedded_length = np.prod(datapoint_shape)
            # print("(Flatten layer)")
            print(datapoint_shape, " --> ", self.embedded_length," (embedding)")

        self.hparams.l[0] = self.embedded_length

    def init_deembedding(self):
        l = self.hparams.l
        datapoint_shape = self.hparams.datapoint_shape

        if self.hparams.embedding_type == "flatten":
            self.deembedding = nn.Unflatten(1, datapoint_shape)
            # print("(Unflatten layer)")
            print(l[0], " --> ", datapoint_shape," (deembedding)")

    def forward(self, x):
        x = self.embedding(x)
        x_out = super().forward(x)

        if self.metaD:
            return x_out

        x_out, meta = x_out
        x_out = self.deembedding(x_out)

        return x_out, meta

    def get_latent(self, data_x):
        data_x = self.embedding(data_x)
        data_x = self.normalize(data_x)
        latent_mu, latent_logvar = self.encode(data_x)
        return latent_mu.detach().cpu().numpy(), latent_logvar.detach().cpu().numpy()

    def decode_latent(self, latent):
        x_out = self.decode(latent)
        x_out = self.denormalize(x_out)
        x_out = self.deembedding(x_out)
        return x_out.detach().cpu().numpy()
    
    def print_labels_latent_correlations(self, latent, labels = None):
        pass
    
    def plot_extra(self, data_x, data_y, latents):
        latent_logvar = latents[1]
        self.plot_latent(latents, data_y, self.plot_sd, "latent_pdf")


