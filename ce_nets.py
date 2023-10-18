import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable
torch.manual_seed(0)
TORCH_PI = torch.acos(torch.zeros(1))*2




import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from statistics import mean as list_mean

from scipy.stats import multivariate_normal



class LITcollAE(pl.LightningModule):
    def __init__(self, l:list, lr : float = 0.01, l2_reg : float = 1e-7, 
                 outname : str = './LITcollAE_untitled/LITcollAE_'):
        super().__init__()
        assert len(l) >= 2
        print("[Initializing LITcollAE Module]")
        print("- hidden layers:", l)
        print("")
        print("========= NN =========")
        modules = []
        for i in range(len(l) - 1):
            print(l[i], " --> ", l[i + 1], end=" ")
            if i < len(l) - 2:
                modules.append(nn.Linear(l[i], l[i + 1]))
                modules.append(nn.ReLU(True))
                modules.append(nn.BatchNorm1d(l[i + 1]))
                print("(relu)")
            else:
                modules.append(nn.Linear(l[i], l[i + 1]))
                print("")
        modules.append(nn.Sigmoid())
        print("(sigmoid)")
        self.encoder = nn.Sequential(*modules)
        modules = []
        a = len(l) - 1
        for i in range(len(l) - 1):
            print(l[a - i], " --> ", l[a - i - 1], end=" ")
            if i < len(l) - 2:
                modules.append(nn.Linear(l[a - i], l[a - i - 1]))
                modules.append(nn.ReLU(True))
                modules.append(nn.BatchNorm1d(l[a - i - 1]))
                print("(relu)")
            else:
                modules.append(nn.Linear(l[a - i], l[a - i - 1]))
                print("")
        self.decoder = nn.Sequential(*modules)
        print("======================")
        
        # Model meta info
        self.normIn = False
        self.metaD = False
        self.save_hyperparameters()
        
        self.train_loss_list = []
        self.val_loss_list = []
        self.print_loss = 1
        
        self.register_buffer('Mean', torch.zeros(l[0]))
        self.register_buffer('Range', torch.ones(l[0]))
        
    def set_norm(self, Mean: torch.Tensor, Range: torch.Tensor):
        self.normIn = True
        self.Mean = Mean
        self.Range = Range
        
    def normalize(self, x: Variable):
        batch_size = x.size(0)
        x_size = x.size(1)

        # print(f"\n\nmean shape={self.Mean.shape}\n\n")
        # print(f"\n\nmean shape={x.shape}\n\n")
        
        Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

        return x.sub(Mean).div(Range)
    
    def denormalize(self, x: Variable):
        batch_size = x.size(0)
        x_size = x.size(1)

        Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

        return x.mul(Range).add(Mean)

    def encode(self, x: Variable) -> (Variable):
        if self.normIn:
            x = self.normalize(x)
        z = self.encoder(x)
        return z

    def decode(self, x: Variable) -> (Variable):
        z = self.decoder(x)
        return z

    def forward(self, x: Variable) -> (Variable):
        z = self.encode(x)
        if self.metaD:
            return z
        y = self.decode(z)
        
        return y

    def loss_fn(self, output, target):
        return F.mse_loss(output, target)
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                    factor=0.7, patience=10,
    #                                                    min_lr=0.0000001)
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "monitor": "val_error",
    #             "frequency": 1,
    #         }
    #     }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay= self.hparams.l2_reg)
        return optimizer
    
    def on_train_start(self):
        print("\n\n==================================")
        print("Starting training LITcollAE module")
        print("==================================")
        print("[Optimization Settings]")
        print("- Learning rate \t=", self.hparams.lr)
        print("- l2 regularization \t=", self.hparams.l2_reg)
        print("==================================\n\n")
        
        

        # print("[{:>3}/{:>3}] {:>10}".format("ep", "tot", "train_loss",  "val_loss"))
    
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch[0].float()
        result = self(data)
        target = self.normalize(data)
        
        loss = self.loss_fn(result, target)
        self.train_loss_list.append(loss.item())
        # print(f"\nbatch_id={batch_idx}, loss_list_size={len(self.train_loss_list)}")
        # self.log('train_loss', loss.item(), prog_bar=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        data = val_batch[0].float()
        result = self(data)
        target = self.normalize(data)
        loss = self.loss_fn(result, target)
        self.val_loss_list.append(loss)
        self.log('val_loss', loss.item(), prog_bar=True)
        return loss

    # def on_validation_epoch_end(self):
    #     if (len(train_loss_list)) % self.print_loss == 0:
    #         print("[{:3d}/{:3d}] {:10.3f} {:10.3f}".format(len(self.train_loss_list),self.max_epochs, train_loss_list[-1], val_loss_list[-1]))
        

    def test_step(self, test_batch, batch_idx):
        epoch = self.current_epoch
        n_hidden = self.hparams.l[-1]

        
        train_x, train_y = test_batch[0].float(), test_batch[1].float()
        n_labels = train_y.shape[-1]
        
        
        
        fig, axes = plt.subplots(n_hidden if n_hidden > 2 else 1, n_labels + 1, squeeze=False,figsize=(13, 5))
        
        self.plot_training(axes[0][0])
        for i in range(0, axes.shape[0]):
            for j in range(n_labels):
                self.plot_latent(fig, axes[i][j+1], train_x, train_y[:,j], i, j)
        
        plt.tight_layout()
        fig.savefig(f"{self.hparams.outname}{epoch}_training.png", dpi=150)
        plt.close()
    
        return None


    def plot_training(self, ax):
        ax.set_title("Network Loss minimization")
        ax.set_yscale("log")
        ax.plot(
            np.asarray(self.train_loss_list),
            ".-",
            c="tab:green",
            label="loss",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("loss")
        ax.legend()

    def plot_latent(self, fig, ax, train_x, train_y, i, j):
        ax.set_title("LITcollAE Latent-space-"+str(i))
        
        latent_train = self.encode(train_x).cpu().detach().numpy()
        
        cm = plt.get_cmap('jet')
        cNorm = matplotlib.colors.Normalize(vmin=min(train_y), vmax=max(train_y))
        
        # print(f"min={min(train_y)}, max={max(train_y)}\n")
        
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
        yaxis = (i+1) if (i+1) < latent_train.shape[1] else 0
        ax.scatter(latent_train[:, i], latent_train[:, yaxis], c=scalarMap.to_rgba(train_y), label="Whole dataset", alpha=0.3)
        
        ax.set_xlabel("h_{}".format(i))
        ax.set_ylabel("h_{}".format(yaxis))

        scalarMap.set_array(train_y)
        cb = fig.colorbar(scalarMap, ax=ax)
        cb.set_label(self.trainer.datamodule.hparams.label_list[j])
        ax.legend()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        



class LITcollVAE(pl.LightningModule):
    def __init__(self, l:list, lr : float = 0.01, l2_reg : float = 1e-7, 
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
        self.val_loss_list = []
        self.print_loss = 1
        
        # self.register_buffer('train_loss_list', [])
        self.register_buffer('Mean', torch.zeros(l[0]))
        self.register_buffer('Range', torch.ones(l[0]))
        self.save_hyperparameters()
        
    def set_norm(self, Mean: torch.Tensor, Range: torch.Tensor):
        self.normIn = True
        self.Mean = Mean
        self.Range = Range

    def normalize(self, x: Variable):
        batch_size = x.size(0)
        x_size = x.size(1)

        # print(f"\n\nmean shape={self.Mean.shape}\n\n")
        # print(f"\n\nmean shape={x.shape}\n\n")
        
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
        m, l = self.encode_(x)
        if self.training:
            return self.reparametrize(m, l)
        else:
            return m
        
    def encode_(self, x):
        if self.normIn:
            x = self.normalize(x)
        x = self.encoder_hidden(x)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar): # Drawing a random sample from the normal distribution mu, logvar
        std = torch.exp(0.5*logvar) ### Why 0.5? maybe std = standard deviation
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, x):
        # x = x.to(device)
        x = self.decoder_hidden(x)
        mu = self.decoder_mu(x)
        logvar = self.decoder_logvar(x)
        return mu, logvar

    def forward(self, x):
        mu_latent, logvar_latent = self.encode_(x) # p(z|x)
        if self.training:
            z = self.reparametrize(mu_latent, logvar_latent)
        else:
            z = mu_latent

        if self.metaD:
            return mu_latent
        mu_x, logvar_x = self.decode(z) # q(x|z)
        if self.training:
            x_out = self.reparametrize(mu_x, logvar_x)
        else:
            x_out = mu_x
        
        return x_out, {"mu_latent" : mu_latent, "logvar_latent" : logvar_latent,
                       "mu_x" : mu_x, "logvar_x" : logvar_x}
        
        
    def kld(self, mu, logvar): 
        # KLD between two univariate gaussians, explanation here:
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # Second Gaussian is zero mean and variance of 1, the prior on z
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1) # sum for all the latent variables
        return kld


    def plain_mse(self, recon_x, tru_x, **kwargs):
        loss_rec = F.mse_loss(recon_x, tru_x, reduction='mean')
        return loss_rec

    def ortho_loss(self, recon_x, tru_x, **kwargs):
        latent = kwargs["latent"]

    def recon_loss_data(self, tru_x, mu_x, logvar_x):
        ## Basically only calculates log p(x|z) for one value of z taken from q(z|x) in the forward function of the model instead of calculating an expectation value
        # Sum on all the input distances
        loss_rec = -torch.mean(
            (-0.5 * torch.log(2 * TORCH_PI.to(mu_x.device)))
            + (-0.5 * logvar_x)
            + ((-0.5 / (0.0005 + torch.exp(logvar_x)))
                        * (tru_x - mu_x) ** 2.0),
            axis=1
        )
        return loss_rec
    

    def vae_loss(self, recon_x, tru_x, beta=1, **kwargs):
        # full vae loss for modeling a distributive latent space
        # AND a distributive reconstruction
        # correct formulation here:
        mu_latent = kwargs["mu_latent"]
        logvar_latent = kwargs["logvar_latent"]
        mu_x = kwargs["mu_x"]
        logvar_x = kwargs["logvar_x"]

        loss_rec = self.recon_loss_data(tru_x, mu_x, logvar_x)
        KLD = beta * self.kld(mu_latent, logvar_latent)
        # print(KLD)
        # print(loss_rec)
        loss = torch.mean(loss_rec + KLD, dim=0) # mean of batch
        return loss

    def naive_vae_loss(self, recon_x, tru_x, beta=50, **kwargs):
        mu_latent = kwargs["mu_latent"]
        logvar_latent = kwargs["logvar_latent"]
        mu_x = kwargs["mu_x"]
        logvar_x = kwargs["logvar_x"]

        loss_rec = F.mse_loss(mu_x, tru_x, reduction='mean')

        KLD = beta * self.kld(mu_latent, logvar_latent)
        loss = torch.mean(loss_rec + KLD, dim=0)
        return loss

    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                    factor=0.7, patience=10,
    #                                                    min_lr=0.0000001)
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "monitor": "val_error",
    #             "frequency": 1,
    #         }
    #     }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay= self.hparams.l2_reg)
        return optimizer
    
    def on_train_start(self):
        print("\n\n==================================")
        print("Starting training LITcollVAE module")
        print("==================================")
        print("[Optimization Settings]")
        print("- Learning rate \t=", self.hparams.lr)
        print("- l2 regularization \t=", self.hparams.l2_reg)
        print("==================================\n\n")
        
        

        # print("[{:>3}/{:>3}] {:>10}".format("ep", "tot", "train_loss",  "val_loss"))
    
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch[0].float()
        result, meta = self(data)
        target = self.normalize(data)
        
        loss = self.naive_vae_loss(result, target, **meta)
        
        self.step_loss_list.append(loss.item())
            
        # print(f"\nbatch_id={batch_idx}, loss_list_size={len(self.train_loss_list)}")
        # self.log('train_loss', loss.item(), prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.train_loss_list.append(list_mean(self.step_loss_list))
        self.step_loss_list.clear()  # free memory
    
    def validation_step(self, val_batch, batch_idx):
        data = val_batch[0].float()
        result, meta = self(data)
        target = self.normalize(data)
        
        loss = self.naive_vae_loss(result, target, **meta)
        self.val_loss_list.append(loss)
        self.log('val_loss', loss.item(), prog_bar=True)
        return loss

    # def on_validation_epoch_end(self):
    #     if (len(train_loss_list)) % self.print_loss == 0:
    #         print("[{:3d}/{:3d}] {:10.3f} {:10.3f}".format(len(self.train_loss_list),self.max_epochs, train_loss_list[-1], val_loss_list[-1]))
        

    def test_step(self, test_batch, batch_idx):
        epoch = self.current_epoch
        n_hidden = self.hparams.l[-1]

        
        train_x, train_y = test_batch[0].float(), test_batch[1].float()
        n_labels = train_y.shape[-1]
        
        
        
        fig, axes = plt.subplots(n_hidden if n_hidden > 2 else 1, n_labels + 1, squeeze=False,figsize=(13, 5))
        
        self.plot_training(axes[0][0])
        for i in range(0, axes.shape[0]):
            for j in range(n_labels):
                self.plot_latent(fig, axes[i][j+1], train_x, train_y[:,j], i, j)
        
        plt.tight_layout()
        fig.savefig(f"{self.hparams.outname}{epoch}_training.png", dpi=150)
        plt.close()
        
        # fig, axes = plt.subplots(1,1, squeeze=False,figsize=(13, 13))
        # self.plot_latent_surface(fig, axes[0][0], train_x, train_y, 0)
    
        return None


    def plot_training(self, ax):
        ax.set_title("Network Loss minimization")
        ax.set_yscale("log")
        ax.plot(
            np.asarray(self.train_loss_list),
            ".-",
            c="tab:green",
            label="loss",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("loss")
        ax.legend()
        
        

    def plot_latent(self, fig, ax, train_x, train_y, i, j):
        ax.set_title("LITcollVAE Latent-space-"+str(i))
        
        latent_mu, latent_logvar = self.encode_(train_x)
        latent_mu, latent_logvar = latent_mu.cpu().detach().numpy(), latent_logvar.cpu().detach().numpy()
        train_y = train_y.cpu().detach().numpy()
        
        if latent_mu.shape[0] > 5000:
            index = np.random.choice(latent_mu.shape[0], 5000, replace=False)
            latent_mu = latent_mu[index]
            latent_logvar = latent_logvar[index]
            train_y = train_y[index]
        
        
        latent_sd = np.sqrt(np.exp(latent_logvar))
        
        cm = plt.get_cmap('jet')
        cNorm = matplotlib.colors.Normalize(vmin=min(train_y), vmax=max(train_y))
        
        # print(f"min={min(train_y)}, max={max(train_y)}\n")
        
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
        yaxis = (i+1) if (i+1) < latent_mu.shape[1] else 0
        
        if True: ## To remove outliers
            for ind,point in enumerate(latent_mu):
                if (point[0] < -20) or (point[1] < -20):
                    print(f"Outlier point: {point[0]}.{point[1]} ind:{ind}, will not be plotted")
                    latent_mu = np.delete(latent_mu, [ind], axis=0)
                    latent_sd = np.delete(latent_sd, [ind], axis=0)
                    train_y = torch.cat([train_y[:ind], train_y[ind+1:]])
        # print(latent_logvar)
                
        if False:
            ax.errorbar(latent_mu[:, i], latent_mu[:, yaxis],xerr=latent_sd[:,i],yerr=latent_sd[:,yaxis], fmt='none', ecolor=scalarMap.to_rgba(train_y), alpha=0.3)
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
        
        latent_mu, latent_logvar = self.encode_(train_x)
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
