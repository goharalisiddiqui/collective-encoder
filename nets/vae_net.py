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


class LITcollVAE(pl.LightningModule):
    def __init__(self, 
                 l:list, 
                 lr : float = 0.01, 
                 l2_reg : float = 1e-7,
                 beta : float = 1.0,
                 loss_type : str = 'mse',
                 n_samples : int = 1,
                 outname : str = './LITcollVAE_untitled/LITcollVAE_'):
        super().__init__()
        assert len(l) >= 3
        
        #### Setting up the layers of the netwrok ####
        print("[Initializing LITcollVAE Module]")
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
        if loss_type == 'mse':
            self.decoder_output = nn.Linear(l[1], l[0])
            print(l[1], " --> ", l[0], end=" ")
            print("(feature space)")
            print("======================")
        else:
            self.decoder_mu = nn.Linear(l[1], l[0])
            print(l[1], " --> ", l[0], end=" ")
            print("(mu for feature space)")
            self.decoder_logvar = nn.Linear(l[1], l[0])
            print( "  ", " \--> ", l[0], end=" ")
            print("(logvar for feature space)\n\n")
            print("======================")

            
        # Model meta info
        self.normIn = False
        self.metaD = False
        
        self.step_loss_list = []
        self.train_loss_list = []
        
        self.step_reg_loss_list = []
        self.train_reg_loss_list = []
        
        self.step_rec_loss_list = []
        self.train_rec_loss_list = []
        
        self.val_loss_list = []
        self.print_loss = 1
        
        # self.register_buffer('train_loss_list', [])
        self.register_buffer('Mean', torch.zeros(l[0]))
        self.register_buffer('Range', torch.ones(l[0]))
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
        
    def encode(self, x):
        if self.normIn:
            x = self.normalize(x)
        x = self.encoder_hidden(x)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar): # Drawing a random sample from the normal distribution mu, logvar
        std = torch.exp(0.5*logvar) 
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = self.decoder_hidden(z)
        x_out = self.decoder_output(z)
        return x_out

    def forward(self, x):
        mu_latent, logvar_latent = self.encode(x) # p(z|x)
        if mu_latent.isnan().any().detach().cpu().numpy() or logvar_latent.isnan().any().detach().cpu().numpy():
            print("Nan in encoder network (Gradient diminished or exploded). Can't continue")
            exit()
        if self.metaD:
            return mu_latent, logvar_latent
        if self.training:
            z = self.reparametrize(mu_latent, logvar_latent)
        else:
            z = mu_latent
        if self.hparams.loss_type == 'mse':
            x_out = self.decode(z)
            return x_out, {"mu_latent" : mu_latent, "logvar_latent" : logvar_latent}
        else:
            mu_x, logvar_x = self.decode(z) # q(x|z)
            if mu_x.isnan().any().detach().cpu().numpy() or logvar_x.isnan().any().detach().cpu().numpy():
                print("Nan in decoder network (Gradient diminished or exploded). Can't continue")
                exit()
            x_out = mu_x
        
            return x_out, {"mu_latent" : mu_latent, "logvar_latent" : logvar_latent,
                        "mu_x" : mu_x, "logvar_x" : logvar_x}

        
        
    def kld(self, mu, logvar): 
        # KLD between two univariate gaussians, explanation here:
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # Second Gaussian is zero mean and variance of 1, the prior on z
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1) # sum for all the latent variables
        # return torch.nan_to_num(kld, nan=0.0)
        return kld

    def recon_loss(self, tru_x, mu_z, logvar_z):
        mu_z_n = mu_z.unsqueeze(0).expand(self.hparams.n_samples, -1, -1)
        logvar_z_n = logvar_z.unsqueeze(0).expand(self.hparams.n_samples, -1, -1)
        z = self.reparametrize(mu_z_n, logvar_z_n)
        
        loss_rec_n = torch.zeros(z.size(dim=0), z.size(dim=1), device=z.device)
        for i in range(self.hparams.n_samples):
            mu_x, logvar_x = self.decode(z[i])
            p_x = Normal(mu_x, torch.exp(logvar_x))
            p_x.log_prob(tru_x)
            loss_rec = -torch.mean(p_x.log_prob(tru_x), axis=1)
            loss_rec_n[i] = loss_rec
        loss_rec = torch.mean(loss_rec_n, dim=0)
        
        return loss_rec

    

    def vae_loss(self, recon_x, tru_x, **kwargs):
        mu_latent = kwargs["mu_latent"]
        logvar_latent = kwargs["logvar_latent"]
        
        if self.hparams.loss_type == 'elbo': # samples q(z|x) n_samples time and calculate a mean of log p(x|z)
            loss_rec = self.recon_loss(tru_x, mu_latent, logvar_latent)
        elif self.hparams.loss_type == 'mse':
            loss_rec = F.mse_loss(recon_x, tru_x, reduction='mean')
        else:
            print("Unrecognized loss_type used in VAE model")
            exit()
        loss_reg = self.kld(mu_latent, logvar_latent)
        
        KLD = self.hparams.beta * loss_reg
        
        loss = torch.mean(loss_rec + KLD, dim=0) # mean of batch
        
        if loss.isnan().any().detach().cpu().numpy():
            print("loss contains nan. Can't continue")
            exit()
            
        loss_rec = torch.mean(loss_rec, dim = 0)
        loss_reg = torch.mean(loss_reg, dim = 0)
        return loss, loss_rec, loss_reg

    
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
        print("Starting training LITcollVAESimple module")
        print("==================================")
        print("[Optimization Settings]")
        print("- Learning rate \t=", self.hparams.lr)
        print("- l2 regularization \t=", self.hparams.l2_reg)
        print("[Hyperparameters]")
        print("- Beta \t=", self.hparams.beta)
        print("==================================\n\n")
        
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch[0].float()
        result, meta = self(data)
        target = self.normalize(data)
        loss, rec_loss, reg_loss = self.vae_loss(result, target, **meta)
        
        
        self.step_loss_list.append(loss.item())
        self.step_rec_loss_list.append(rec_loss.item())
        self.step_reg_loss_list.append(reg_loss.item())
            
        return loss

    def on_train_epoch_end(self):
        self.train_loss_list.append(list_mean(self.step_loss_list))
        self.step_loss_list.clear()  # free memory
        
        self.train_rec_loss_list.append(list_mean(self.step_rec_loss_list))
        self.step_rec_loss_list.clear()  # free memory
        
        self.train_reg_loss_list.append(list_mean(self.step_reg_loss_list))
        self.step_reg_loss_list.clear()  # free memory
    
    def validation_step(self, val_batch, batch_idx):
        data = val_batch[0].float()
        result, meta = self(data)
        target = self.normalize(data)
        
        loss, rec_loss, reg_loss = self.vae_loss(result, target, **meta)
        self.val_loss_list.append(loss)
        self.log('val_loss', loss.item(), prog_bar=True)
        self.log('val_rec_loss', rec_loss.item(), prog_bar=True)
        self.log('val_reg_loss', reg_loss.item(), prog_bar=True)
        
        self.log('rec_loss', self.get_rec_loss(), prog_bar=True)
        self.log('reg_loss', self.get_reg_loss(), prog_bar=True)
        return loss
    
    def get_reg_loss(self):
        if len(self.train_reg_loss_list) > 0:
            return self.train_reg_loss_list[-1]
        else:
            return 0.0
    
    def get_rec_loss(self):
        if len(self.train_rec_loss_list) > 0:
            return self.train_rec_loss_list[-1]
        else:
            return 0.0


    def test_step(self, test_batch, batch_idx):
        train_x, train_y = test_batch[0].float(), test_batch[1].float()
        self.plot_all(train_x, train_y)
        
        
    
    def get_fve(self, datamodule):
        dl = datamodule.test_dataloader()
        flag = self.training
        self.training = False
        with torch.no_grad():
            data = next(iter(dl))[0].float()
            output,_ = self(data)
            target = self.normalize(data)
            sub = torch.sub(target, output)
            ss_err = torch.sum(torch.pow(sub, 2), dim=0)
            meann = torch.mean(target, dim=0, keepdim=True)
            sub_meann = torch.sub(target, meann)
            ss_tot = torch.sum(torch.pow(sub_meann, 2), dim=0)
            fve = 1 - torch.div(ss_err, ss_tot)
            fve_mean = torch.mean(fve).detach().cpu().numpy() # This calculates FVE for each input dim and mean it
            
            ss_err = torch.sum(torch.pow(sub, 2))
            ss_tot = torch.sum(torch.pow(sub_meann, 2))
            fve_sum = 1 - torch.div(ss_err, ss_tot).detach().cpu().numpy() # This calculates one FVE by taking inner product of vectors instead of square
        print("\n\n=======================================")
        print("Fraction of Variation Explined (FVE)")
        print("=======================================")
        # print("FVE_mean = ", fve_mean)
        print("FVE_sum = ", fve_sum)
        print("=======================================\n\n")
        self.training = flag
        return fve_mean




    def plot_all(self, data_x, data_y):
        epoch = self.current_epoch
        n_hidden = self.hparams.l[-1]        
        n_labels = data_y.shape[-1]
        

        fig, axes = plt.subplots(1, 3, squeeze=True,figsize=(6 * 3, 6))
        
        self.plot_training(axes)
        plt.tight_layout()
        fig.savefig(f"{self.hparams.outname}{epoch}_training.png", dpi=150)
        plt.close()
        
        n_rows = n_hidden if n_hidden > 2 else 1
        fig, axes = plt.subplots(n_rows, n_labels, squeeze=False, figsize=(6 * n_labels, 6 * n_rows))
        
        
        
        latent_mu, latent_logvar = self.encode(data_x)
        latent_mu, latent_logvar = latent_mu.cpu().detach().numpy(), latent_logvar.cpu().detach().numpy()
        train_y = data_y.cpu().detach().numpy()
        
        data_df = pd.DataFrame(np.concatenate((latent_mu, train_y), axis=1), columns=["Latent Dimension %d"%i for i in range(latent_mu.shape[1])] + self.trainer.datamodule.hparams.label_list)
        print("\n\n=======================================")
        print("Correlation of latent space with labels")
        print("=======================================")
        print(data_df.corr())
        print("=======================================\n\n")
        
        
        
        if True and latent_mu.shape[0] > 5000: # Limit plots to 5000 points
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
        
        
        latent_sd = np.sqrt(np.exp(latent_logvar))
        
        for i in range(0, axes.shape[0]):
            for j in range(n_labels):
                self.plot_latent(fig, axes[i][j], latent_mu, latent_logvar, train_y[:,j], i, j)
        
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
        
        ax[1].set_title("Reconstruction Loss minimization")
        ax[1].set_yscale("log")
        ax[1].plot(
            np.asarray(self.train_rec_loss_list),
            ".-",
            c="tab:blue",
        )
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Reconstruction loss")
        
        ax[2].set_title("Regularization Loss minimization")
        ax[2].set_yscale("log")
        ax[2].plot(
            np.asarray(self.train_reg_loss_list),
            ".-",
            c="tab:red",
        )
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("Regularization loss")
        
        

    def plot_latent(self, fig, ax, latent_mu, latent_logvar, train_y, i, j):
        ax.set_title("collenVAE Latent-space-"+str(i))
        
        
        
        cm = plt.get_cmap('jet')
        cNorm = matplotlib.colors.Normalize(vmin=min(train_y), vmax=max(train_y))
        
        # print(f"min={min(train_y)}, max={max(train_y)}\n")
        
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
        yaxis = (i+1) if (i+1) < latent_mu.shape[1] else 0
        
        if False: ## To remove outliers
            for ind,point in enumerate(latent_mu):
                if (point[0] < -20) or (point[1] < -20):
                    print(f"\nOutlier point: {point[0]}.{point[1]} ind:{ind}, will not be plotted")
                    latent_mu = np.delete(latent_mu, [ind], axis=0)
                    latent_sd = np.delete(latent_sd, [ind], axis=0)
                    train_y = np.delete(train_y, [ind], axis=0)
        # print(latent_logvar)
                
        if False:
            ax.errorbar(latent_mu[:, i], latent_mu[:, yaxis],xerr=latent_sd[:,i],yerr=latent_sd[:,yaxis], fmt='none', ecolor=scalarMap.to_rgba(train_y), alpha=0.1)
        else:
            ax.scatter(latent_mu[:, i], latent_mu[:, yaxis], c=scalarMap.to_rgba(train_y), label="Whole dataset", alpha=0.3)
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
