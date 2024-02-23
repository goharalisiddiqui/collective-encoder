import os
import itertools
import sys
import time
import numpy as np




import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable
torch.manual_seed(0)
TORCH_PI = torch.acos(torch.zeros(1))*2


import pandas as pd
pd.set_option('display.max_columns', None) 



import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from statistics import mean as list_mean

from scipy.stats import multivariate_normal

from gmvae_utils import qy_map, qz_map, pz_map, px_map, log_normal, gaussian_sample, progbar






class LITcollGMVAE(pl.LightningModule):
    def __init__(self,
                 outname : str = './LITcollGMVAE_untitled/LITcollGMVAE_',
                 k : int = 10,
                 n_x : int = 784, 
                 n_z : int = 64,
                 qy_dims : list = [16,16],
                 qz_dims : list = [16,16],
                 pz_dims : list = [16,16],
                 px_dims : list = [16,16],
                 r_nent : int = 1, # 0.5 was good.
                 batch_size : int =1000, 
                 lr : float = 0.00001):
        """Build a GMM VAE model.
        
        Args:
            k (int): Number of mixture components.
            n_x (int): Number of observable dimensions.
            n_z (int): Number of hidden dimensions.
            qy_dims (iterable of int): Iterable of hidden dimensions in qy subgraph.
            qz_dims (iterable of int): Iterable of hidden dimensions in qz subgraph.
            pz_dims (iterable of int): Iterable of hidden dimensions in pz subgraph.
            px_dims (iterable of int): Iterable of hidden dimensions in px subgraph.
            r_nent (float): A constant for weighting negative entropy term in the loss.
            batch_size (int): Number of samples in each batch.
            lr (float): Learning rate.
        """
        super().__init__()
        

        
        # NN(Q_y)
        self.qy_logit, self.qy_ytransform = qy_map(n_x, k, qy_dims)

        # NN(Q_z)
        self.qz_ytransform, self.qz_hlayers, self.qz_zmtransform, self.qz_zvtransform = [[None] * k for i in range(4)]
        for i in range(self.k):
            self.qz_ytransform[i], self.qz_hlayers[i], self.qz_zmtransform[i], self.qz_zvtransform[i] = qz_map(n_x, k, n_z, qz_dims)

        # NN(P_z)
        self.pz_hlayers, self.pz_zmtransform, self.pz_zvtransform = [[None] * k for i in range(3)]
        for i in range(self.k):
            self.pz_hlayers[i], self.pz_zmtransform[i], self.pz_zvtransform[i] = pz_map(k, n_z, pz_dims)
        
        # NN(P_x)  
        self.px_hlayers, self.px_xmtransform, self.px_xvtransform = [[None] * k for i in range(3)]
        for i in range(self.k):
            self.px_hlayers[i], self.px_xmtransform[i], self.px_xvtransform[i] = px_map(k, n_z, px_dims)
            


        self.save_hyperparameters()




    def probabilistic_cluster_assignments(self, x_data):
        pyx = self.qy_logit(x_data)
        pyx_soft = self.qy_ytransform(pyx)
        
        return pyx, pyx_soft        
        
        
    def encode(self, x_data):
        with torch.no_grad():
            k_oh = F.one_hot(torch.arange(0, self.hparams.k) % self.hparams.k)
        z, zm, zv = [[None] * self.hparams.k for i in range(3)]
        for i in range(self.hparams.k):
            with torch.no_grad():
                k_expand = k_oh[i].unsqueeze(0).expand(x_data.size[0], self.hparams.k)
            z_input = torch.cat((x_data, self.qz_ytransform[i](k_expand)), dim = 1)
            z_hidden = self.qz_hlayers[i](z_input)
            zm = self.qz_zmtransform[i](z_hidden)
            zv = self.qz_zvtransform[i](z_hidden)
            z = gaussian_sample(zm, zv)
        return z, zm, zv
    
    

    def decode(self, x_data):
        with torch.no_grad():
            k_oh = F.one_hot(torch.arange(0, self.hparams.k) % self.hparams.k)
        x, xm, xv = [[None] * self.hparams.k for i in range(3)]
        for i in range(self.hparams.k):
            with torch.no_grad():
                k_expand = k_oh[i].unsqueeze(0).expand(x_data.size[0], self.hparams.k)
            z_hidden = self.pz_hlayers[i](k_expand)
            zm = self.pz_zmtransform[i](z_hidden)
            zv = self.pz_zvtransform[i](z_hidden)
            z = gaussian_sample(zm, zv)
            x_hidden = self.px_hlayers[i](z)
            xm = self.px_xmtransform[i](x_hidden)
            xv = self.px_xvtransform[i](x_hidden)
        return xm, xv, zm, zv
    
    
    def forward(self, x_data):
        pyx, pyx_soft = self.probabilistic_cluster_assignments(x_data)
        z, zm, zv = self.encode(x_data)
        xm, xv, zm_prior, zv_prior = self.decode(x_data)

        loss = self.gmvae_loss(pyx, pyx_soft, zm, zv, z, zm_prior, zv_prior, xm, xv, x_data)
        
        return loss



    def labeled_loss(self, k, x, xm, xv, z, zm, zv, zm_prior, zv_prior):
        """Variational loss for the mixture VAE given for each given q(y=i|x, z), hence the
            name labeled_loss."""
        return -log_normal(x, xm, xv) + log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior) - np.log(1/k) 




    def gmvae_loss(self, pyx, pyx_soft, zm, zv, z, zm_prior, zv_prior, xm, xv, x):
        
        # self.nent = -tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.qy, logits=self.qy_logit) #YB: _v2
        nent = -F.cross_entropy(pyx, pyx_soft) # Why is this negative?
        losses = [None] * self.hparams.k
        for i in range(self.hparams.k):
            losses[i] = self.labeled_loss(self.hparams.k, x, xm[i], xv[i],
                                        z[i], zm[i], zv[i],
                                        zm_prior[i], zv_prior[i])
        loss = torch.cat(tuple([nent*self.hparams.r_nent] + [pyx_soft[:, i] * losses[i] for i in range(self.hparams.k)]), dim = 1).sum(dim = 1)
        
        loss = torch.mean(loss, dim=0) # Batch Mean
        
        return loss




