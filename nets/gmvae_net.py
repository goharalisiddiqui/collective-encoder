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

from nets.gmvae_utils import qy_map, qz_map, pz_map, px_map, log_normal, gaussian_sample




# DOI 10.1088/2632-2153/ab80b7


class GMVAE(pl.LightningModule):
    def __init__(self, 
                 n_x : int, 
                 n_z : int,
                 k : int = 2,
                 qy_dims : list = [16,16],
                 qz_dims : list = [16,16],
                 pz_dims : list = [16,16],
                 px_dims : list = [16,16],
                 r_nent : int = 1, # 0.5 was good.
                 lr : float = 0.01, 
                 l2_reg : float = 1e-7,
                 n_samples : int = 1,
                 outname : str = './GMVAE_untitled/GMVAE_'):
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
        
        #### Setting up the layers of the netwrok ####
        print("[Initializing LITcollGMVAE Module]")
        print("")
        print("========= NN =========")
        print("TODO: Add a printout of the network layers")
        # NN(Q_y)
        self.qy_logit, self.qy_ytransform = qy_map(n_x, k, qy_dims)

        # NN(Q_z)
        self.qz_ytransform, self.qz_hlayers, self.qz_zmtransform, self.qz_zvtransform = [[None] * k for i in range(4)]
        for i in range(k):
            self.qz_ytransform[i], self.qz_hlayers[i], self.qz_zmtransform[i], self.qz_zvtransform[i] = qz_map(n_x, k, n_z, qz_dims)

        # NN(P_z)
        self.pz_hlayers, self.pz_zmtransform, self.pz_zvtransform = [[None] * k for i in range(3)]
        for i in range(k):
            self.pz_hlayers[i], self.pz_zmtransform[i], self.pz_zvtransform[i] = pz_map(k, n_z, pz_dims)
        
        # NN(P_x)  
        self.px_hlayers, self.px_xmtransform, self.px_xvtransform = [[None] * k for i in range(3)]
        for i in range(k):
            self.px_hlayers[i], self.px_xmtransform[i], self.px_xvtransform[i] = px_map(n_z, n_x, px_dims)
            
        self.nent_loss = nn.CrossEntropyLoss()
        print("======================")
        # Model meta info
        self.normIn = False
        self.metaD = False
        
        self.step_loss_list = []
        self.train_loss_list = []
        
        #TODO: Save and log individual losses
        
        self.val_loss_list = []
        self.print_loss = 1
        
        # self.register_buffer('train_loss_list', [])
        self.register_buffer('Mean', torch.zeros(n_x))
        self.register_buffer('Range', torch.ones(n_x))
        self.save_hyperparameters()
        
    def set_norm(self, Mean: torch.Tensor, Range: torch.Tensor):
        Range[Range == 0.0] = 1.0
        self.normIn = True
        self.Mean = Mean
        self.Range = Range

    def normalize(self, x: Variable):
        batch_size = x.size(0)
        x_size = x.size(1)
        
        Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

        return x.sub(Mean).div(Range)
    
    def denormalize(self, x: Variable):
        batch_size = x.size(0)
        x_size = x.size(1)

        Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

        return x.mul(Range).add(Mean)


    def probabilistic_cluster_assignments(self, x_data):
        pyx = self.qy_logit(x_data)
        pyx_soft = self.qy_ytransform(pyx)
        
        return pyx, pyx_soft        
        
        
    def encode(self, x_data):
        z, zm, zv = [[None] * self.hparams.k for i in range(3)]
        for i in range(self.hparams.k):
            with torch.no_grad():
                k_oh = torch.zeros(self.hparams.k)
                k_oh[i] = 1
                k_expand = k_oh.unsqueeze(0).expand(x_data.size(0), self.hparams.k)
            z_input = torch.cat((x_data, self.qz_ytransform[i](k_expand)), dim = 1)
            z_hidden = self.qz_hlayers[i](z_input)
            zm[i] = self.qz_zmtransform[i](z_hidden)
            zv[i] = self.qz_zvtransform[i](z_hidden)
            z[i] = gaussian_sample(zm[i], zv[i])
        return z, zm, zv
    
    

    def decode(self, x_data):
        xm, xv, zm, zv = [[None] * self.hparams.k for i in range(4)]
        for i in range(self.hparams.k):
            with torch.no_grad():
                k_oh = torch.zeros(self.hparams.k)
                k_oh[i] = 1
                k_expand = k_oh.unsqueeze(0).expand(x_data.size(0), self.hparams.k)
            z_hidden = self.pz_hlayers[i](k_expand)
            zm[i] = self.pz_zmtransform[i](z_hidden)
            zv[i] = self.pz_zvtransform[i](z_hidden)
            z = gaussian_sample(zm[i], zv[i])
            x_hidden = self.px_hlayers[i](z)
            xm[i] = self.px_xmtransform[i](x_hidden)
            xv[i] = self.px_xvtransform[i](x_hidden)
        return xm, xv, zm, zv
    
    
    def forward(self, x_data):
        x_data = self.normalize(x_data)
        pyx, pyx_soft = self.probabilistic_cluster_assignments(x_data)
        z, zm, zv = self.encode(x_data)
        if self.metaD:
            return torch.cat(tuple([pyx_soft[:, i] * z[i] for i in range(self.hparams.k)]), dim = 1).sum(dim = 1).mean(dim=0)
        xm, xv, zm_prior, zv_prior = self.decode(x_data)

        # print("\nSize pyx = ", pyx.shape)
        # print("Size pyx_soft = ", pyx_soft.shape)
        # print("Size zm = ", zm.shape)
        # print("Size zv = ", zv.shape)
        # print("Size z = ", z.shape)
        # print("Size zm_prior = ", zm_prior.shape)
        # print("Size zv_prior = ", zv_prior.shape)
        # print("Size xm = ", xm.shape)
        # print("Size xv = ", xv.shape)
        # exit()
        
        
        return pyx, pyx_soft, zm, zv, z, zm_prior, zv_prior, xm, xv, x_data



    def labeled_loss(self, k, x, xm, xv, z, zm, zv, zm_prior, zv_prior):
        """Variational loss for the mixture VAE given for each given q(y=i|x, z), hence the
            name labeled_loss."""
            
        # print("\nSize zm = ", zm.shape)
        # print("Size zv = ", zv.shape)
        # print("Size z = ", z.shape)
        # print("Size zm_prior = ", zm_prior.shape)
        # print("Size zv_prior = ", zv_prior.shape)
        # print("Size xm = ", xm.shape)
        # print("Size xv = ", xv.shape)
        # print("Size x = ", x.shape)
        
        # Sum over the dimensions
        log_P_y = torch.ones(x.size(0),1) * np.log(1/k)
        log_P_x_z = log_normal(x, xm, xv).sum(dim=1, keepdim=True)
        log_Q_z_xy =  log_normal(z, zm, zv).sum(dim=1, keepdim=True)
        log_P_z_y = log_normal(z, zm_prior, zv_prior).sum(dim=1, keepdim=True)
        
        # print("\nSize log_P_y = ", log_P_y.shape)
        # print("Size log_P_x_z = ", log_P_x_z.shape)
        # print("Size log_Q_z_xy = ", log_Q_z_xy.shape)
        # print("Size log_P_z_y = ", log_P_z_y.shape)
        # exit()
        
        return -log_P_x_z + log_Q_z_xy - log_P_z_y - log_P_y




    def gmvae_loss(self, pyx, pyx_soft, zm, zv, z, zm_prior, zv_prior, xm, xv, x):
        
        nent = -self.nent_loss(pyx, pyx_soft) # Why is this negative?
        # print("\nSize nent = ", nent.shape)
        losses = [None] * self.hparams.k
        for i in range(self.hparams.k):
            losses[i] = self.labeled_loss(self.hparams.k, x, xm[i], xv[i],
                                        z[i], zm[i], zv[i],
                                        zm_prior[i], zv_prior[i])
        loss = torch.cat(tuple([pyx_soft[:, i] * losses[i] for i in range(self.hparams.k)]), dim = 1).sum(dim = 1)
        
        loss = nent * self.hparams.r_nent + torch.mean(loss, dim=0) # Batch Mean
        
        return loss



    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay= self.hparams.l2_reg)
        if False:
            return optimizer
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.8, patience=10,
                                                       min_lr=1e-10,
                                                       cooldown = 30,
                                                       verbose =True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            }
        }
    
    def on_train_start(self):
        print("\n\n==================================")
        print("Starting training GMVAE module")
        print("==================================")
        print("[Optimization Settings]")
        print("- Learning rate \t=", self.hparams.lr)
        print("- l2 regularization \t=", self.hparams.l2_reg)
        print("[Hyperparameters]")
        print("- r_nent \t=", self.hparams.r_nent)
        print("==================================\n\n")
        
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch[0].float()
        pyx, pyx_soft, zm, zv, z, zm_prior, zv_prior, xm, xv, x_data = self(data)
        loss = self.gmvae_loss(pyx, pyx_soft, zm, zv, z, zm_prior, zv_prior, xm, xv, x_data)
        
        self.step_loss_list.append(loss.item())
        # self.step_rec_loss_list.append(rec_loss.item())
        # self.step_reg_loss_list.append(reg_loss.item())
            
        return loss

    def on_train_epoch_end(self):
        self.train_loss_list.append(list_mean(self.step_loss_list))
        self.step_loss_list.clear()  # free memory
        
        # self.train_rec_loss_list.append(list_mean(self.step_rec_loss_list))
        # self.step_rec_loss_list.clear()  # free memory
        
        # self.train_reg_loss_list.append(list_mean(self.step_reg_loss_list))
        # self.step_reg_loss_list.clear()  # free memory
    
    def validation_step(self, val_batch, batch_idx):
        data = val_batch[0].float()
        pyx, pyx_soft, zm, zv, z, zm_prior, zv_prior, xm, xv, x_data = self(data)
        loss = self.gmvae_loss(pyx, pyx_soft, zm, zv, z, zm_prior, zv_prior, xm, xv, x_data)
        
        self.val_loss_list.append(loss.item())
        self.log('val_loss', loss.item(), prog_bar=True)
        # self.log('val_rec_loss', rec_loss.item(), prog_bar=True)
        # self.log('val_reg_loss', reg_loss.item(), prog_bar=True)
        
        # self.log('rec_loss', self.get_rec_loss(), prog_bar=True)
        # self.log('reg_loss', self.get_reg_loss(), prog_bar=True)
        return loss
    
    # def get_reg_loss(self):
    #     if len(self.train_reg_loss_list) > 0:
    #         return self.train_reg_loss_list[-1]
    #     else:
    #         return 0.0
    
    # def get_rec_loss(self):
    #     if len(self.train_rec_loss_list) > 0:
    #         return self.train_rec_loss_list[-1]
    #     else:
    #         return 0.0


    def test_step(self, test_batch, batch_idx):
        train_x, train_y = test_batch[0].float(), test_batch[1].float()
        self.plot_all(train_x, train_y)
        
        
    
    # def get_fve(self, datamodule):
    #     dl = datamodule.test_dataloader()
    #     flag = self.training
    #     self.training = False
    #     with torch.no_grad():
    #         data = next(iter(dl))[0].float()
    #         output,_ = self(data)
    #         target = self.normalize(data)
    #         sub = torch.sub(target, output)
    #         ss_err = torch.sum(torch.pow(sub, 2), dim=0)
    #         meann = torch.mean(target, dim=0, keepdim=True)
    #         sub_meann = torch.sub(target, meann)
    #         ss_tot = torch.sum(torch.pow(sub_meann, 2), dim=0)
    #         fve = 1 - torch.div(ss_err, ss_tot)
    #         fve_mean = torch.mean(fve).detach().cpu().numpy() # This calculates FVE for each input dim and mean it
            
    #         ss_err = torch.sum(torch.pow(sub, 2))
    #         ss_tot = torch.sum(torch.pow(sub_meann, 2))
    #         fve_sum = 1 - torch.div(ss_err, ss_tot).detach().cpu().numpy() # This calculates one FVE by taking inner product of vectors instead of square
    #     print("\n\n=======================================")
    #     print("Fraction of Variation Explined (FVE)")
    #     print("=======================================")
    #     # print("FVE_mean = ", fve_mean)
    #     print("FVE_sum = ", fve_sum)
    #     print("=======================================\n\n")
    #     self.training = flag
    #     return fve_mean




    def plot_all(self, data_x, data_y):
        epoch = self.current_epoch
        n_hidden = self.hparams.n_z       
        n_labels = data_y.shape[-1]
        

        fig, axes = plt.subplots(1, 3, squeeze=True,figsize=(6 * 3, 6))
        
        self.plot_training(axes)
        plt.tight_layout()
        fig.savefig(f"{self.hparams.outname}{epoch}_training.png", dpi=150)
        plt.close()
        
        n_rows = n_hidden if n_hidden > 2 else 1
        fig, axes = plt.subplots(n_rows, n_labels, squeeze=False, figsize=(6 * n_labels, 6 * n_rows))
        
        
        
        _, latent_mu, latent_var = self.encode(data_x)
        # latent_mu, latent_var = latent_mu.cpu().detach().numpy(), latent_var.cpu().detach().numpy()
        train_y = data_y.cpu().detach().numpy()
        
        # data_df = pd.DataFrame(np.concatenate((latent_mu, train_y), axis=1), columns=["Latent Dimension %d"%i for i in range(latent_mu.shape[1])] + self.trainer.datamodule.hparams.label_list)
        # print("\n\n=======================================")
        # print("Correlation of latent space with labels")
        # print("=======================================")
        # print(data_df.corr())
        # print("=======================================\n\n")
        
        
        
        if False and latent_mu.shape[0] > 5000: # Limit plots to 5000 points
            index = np.random.choice(latent_mu.shape[0], 5000, replace=False)
            latent_mu = latent_mu[index]
            latent_logvar = latent_logvar[index]
            train_y = train_y[index]
            
        if False: # Ignore input points outside a certain range
            choices = train_y
            latent_mu = latent_mu[choices > 0]
            latent_logvar = latent_logvar[choices > 0]
            train_y = train_y[choices > 0]
            
            choices = train_y
            latent_mu = latent_mu[choices < 2.0]
            latent_logvar = latent_logvar[choices < 2.0]
            train_y = train_y[choices < 2.0]
        
        
        
        for i in range(0, axes.shape[0]):
            for j in range(n_labels):
                self.plot_latent(fig, axes[i][j], latent_mu, latent_var, train_y[:,j], i, j)
        
        plt.tight_layout()
        fig.savefig(f"{self.hparams.outname}{epoch}_latent_space.png", dpi=150)
        plt.close()
        

    def plot_training(self, ax):
        ax[0].set_title("Network Loss minimization")
        ax[0].set_yscale("log")
        ax[0].plot(
            np.asarray(self.train_loss_list),
            ".-",
            c="tab:green",
        )
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        
        # ax[1].set_title("Reconstruction Loss minimization")
        # ax[1].set_yscale("log")
        # ax[1].plot(
        #     np.asarray(self.train_rec_loss_list),
        #     ".-",
        #     c="tab:blue",
        # )
        # ax[1].set_xlabel("Epoch")
        # ax[1].set_ylabel("Reconstruction loss")
        
        # ax[2].set_title("Regularization Loss minimization")
        # ax[2].set_yscale("log")
        # ax[2].plot(
        #     np.asarray(self.train_reg_loss_list),
        #     ".-",
        #     c="tab:red",
        # )
        # ax[2].set_xlabel("Epoch")
        # ax[2].set_ylabel("Regularization loss")
        
        

    def plot_latent(self, fig, ax, latent_mu, latent_var, train_y, i, j):
        ax.set_title("collenVAE Latent-space-"+str(i))
        
        
        
        cm = plt.get_cmap('jet')
        cNorm = matplotlib.colors.Normalize(vmin=min(train_y), vmax=max(train_y))
        
        # print(f"min={min(train_y)}, max={max(train_y)}\n")
        
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
        yaxis = (i+1) if (i+1) < self.hparams.n_z else 0
        
        if False: ## To remove outliers
            for ind,point in enumerate(latent_mu):
                if (point[0] < -20) or (point[1] < -20):
                    print(f"\nOutlier point: {point[0]}.{point[1]} ind:{ind}, will not be plotted")
                    latent_mu = np.delete(latent_mu, [ind], axis=0)
                    latent_sd = np.delete(latent_sd, [ind], axis=0)
                    train_y = np.delete(train_y, [ind], axis=0)
        
        marker_shapes = ['o', 'x', 's', 'D', 'v', '^', '<', '>', 'p', 'h']
        if False:
            for i in range(self.hparams.k):
                latent_mu_i = latent_mu[i].cpu().detach().numpy()
                latent_sd_i = np.sqrt(latent_mu[i].cpu().detach().numpy())
                ax.errorbar(latent_mu_i[:, i], latent_mu_i[:, yaxis],xerr=latent_sd_i[:,i],yerr=latent_sd_i[:,yaxis], fmt='none', ecolor=scalarMap.to_rgba(train_y), alpha=0.1)
        else:
            for i in range(self.hparams.k):
                latent_mu_i = latent_mu[i].cpu().detach().numpy()
                ax.scatter(latent_mu_i[:, i], latent_mu_i[:, yaxis], c=scalarMap.to_rgba(train_y), label="Whole dataset", alpha=0.3, marker=marker_shapes[i])
        ax.set_xlabel("h_{}".format(i))
        ax.set_ylabel("h_{}".format(yaxis))

        scalarMap.set_array(train_y)
        cb = fig.colorbar(scalarMap, ax=ax)
        cb.set_label(self.trainer.datamodule.hparams.label_list[j])
        # ax.legend()
        
    
        
        
    def plot_latent_surface(self, fig, ax, train_x, train_y, i):
        ax.set_title("LITcollVAE Latent-population-"+str(i))
        
        latent_mu, latent_logvar = self.encode(train_x)
        latent_mu, latent_logvar = latent_mu.cpu().detach().numpy(), latent_logvar.cpu().detach().numpy()
        
        latent_sd = np.sqrt(np.exp(latent_logvar))
        
        yaxis = (i+1) if (i+1) < latent_mu.shape[1] else 0
        
        x, y = np.mgrid[-3:3:.01, -3:3:.01]
        pos = np.dstack((x, y))

        res = np.zeros((len(x),len(y)))
        for l in range(latent_mu.shape[0]):
            res += multivariate_normal(mean=latent_mu[l,:], cov=[[latent_sd[l,0], 0.0],[0.0, latent_sd[l,1]]]).pdf(pos)
        
        cs = ax.contourf(x,y, res, 20)
        ax.set_xlabel("h_{}".format(i))
        ax.set_ylabel("h_{}".format(yaxis))
        cbar = fig.colorbar(cs)
        fig.savefig(f"{self.hparams.outname}LatentShape.png", dpi=150)
