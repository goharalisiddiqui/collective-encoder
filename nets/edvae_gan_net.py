from nets.dvae_net import DVAE_args
from nets.dvae_net import DVAE
from scipy.stats import multivariate_normal
from statistics import mean as list_mean
from matplotlib import rc
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.distributions.normal import Normal
torch.manual_seed(0)
TORCH_PI = torch.acos(torch.zeros(1))*2


pd.set_option('display.max_columns', None)


def edvaegan_parse_args():
    desc = "Embedded Deterministic VAE Module"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--emb_type', dest="embedding_type", required=True,
                        type=str, help='Type of embedding for Embedded Deterministic VAE Module')

    args, _ = parser.parse_known_args()

    res_args = DVAE_args()

    return argparse.Namespace(**vars(args), **vars(res_args))


EDVAEGAN_args = edvaegan_parse_args


class EDVAEGAN(DVAE):
    def __init__(self,
                 l: list,
                 embedding_type: str,
                 datapoint_shape: tuple,
                 lr: float = 0.01,
                 l2_reg: float = 1e-7,
                 beta: float = 1.0,
                 batch_norm: bool = True,
                 lr_scheduler: bool = True,
                 plot_every: int = 0,
                 C_max: float = 0.0,
                 C_start: int = 0,
                 C_end: int = 0,
                 saveplotdata: bool = False,
                 outname: str = './EDVAE_untitled/EDVAE_'):

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
                         outname=outname)

        self.Mean = torch.zeros(self.hparams.l[0])
        self.Range = torch.ones(self.hparams.l[0])
        
        self.discriminator = nn.Sequential(
            nn.Linear(self.embedded_length, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.gan_loss_func = torch.nn.BCELoss()
        self.automatic_optimization = False



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

    def init_embedding(self):
        l = self.hparams.l
        datapoint_shape = self.hparams.datapoint_shape

        if self.hparams.embedding_type == "flatten":
            self.embedding = nn.Flatten()
            self.embedded_length = np.prod(datapoint_shape)
            print("(Flatten layer)")
            print(datapoint_shape, " --> ", self.embedded_length, " (embedding)")

        self.hparams.l[0] = self.embedded_length

    def init_deembedding(self):
        l = self.hparams.l
        datapoint_shape = self.hparams.datapoint_shape

        if self.hparams.embedding_type == "flatten":
            self.deembedding = nn.Unflatten(1, datapoint_shape)
            print("(Unflatten layer)")
            print(l[0], " --> ", datapoint_shape, " (deembedding)")

    def gan_loss(self, x, x_out):
        # Adversarial ground truths
        valid = Variable(torch.Tensor(x.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
        fake = Variable(torch.Tensor(x.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)
        guess_real = self.discriminator(x)
        guess_pred = self.discriminator(x_out)
        gan_gen_loss = self.gan_loss_func(guess_pred, valid)

        real_loss = self.gan_loss_func(guess_real, valid)
        fake_loss = self.gan_loss_func(self.discriminator(x_out.detach()), fake) # Detach to avoid backpropagation through the autoencoder
        gan_dis_loss = (real_loss + fake_loss) / 2

        return gan_gen_loss, gan_dis_loss
    
    def loss(self, recon_x, tru_x, **kwargs):
        losses = super().loss(recon_x, tru_x, **kwargs)
        
        gan_gen_loss, gan_dis_loss = self.gan_loss(kwargs['gan_feature_real'], kwargs['gan_feature_pred'])
        losses['loss'] = losses['loss'] + gan_gen_loss * 5.0
        losses['gan_gen_loss'] = gan_gen_loss
        losses['gan_dis_loss'] = gan_dis_loss
        
        return losses

    def forward(self, x):
        x = self.embedding(x)
        x_out = super().forward(x)

        if self.metaD:
            return x_out

        x_out, meta = x_out
        
        # GAN stuff
        meta['gan_feature_real'] = x
        meta['gan_feature_pred'] = x_out
        
        x_out = self.deembedding(x_out)
        

        return x_out, meta
    
    def get_train_parameters(self):
        modules_list = nn.ModuleList()
        for module in self.modules():
            if module not in self.discriminator and module is not self.discriminator:
                modules_list.append(module)
        modules_list = modules_list[1:]
        return modules_list.parameters()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.get_train_parameters(), lr=self.hparams.lr, weight_decay= self.hparams.l2_reg)
        optimizer_descriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, weight_decay= self.hparams.l2_reg)

        if False:
            return [optimizer, optimizer_descriminator]

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.8, patience=10,
                                                       min_lr=1e-10,
                                                       cooldown = 30,
                                                       verbose =True)

        scheduler_discriminator = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_descriminator, mode='min',
                                                       factor=0.8, patience=10,
                                                       min_lr=1e-10,
                                                       cooldown = 30,
                                                       verbose =True)
        return (
            {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
            },
            {
            "optimizer": optimizer_descriminator,
            "lr_scheduler": {
                "scheduler": scheduler_discriminator,
                "monitor": "val_gan_dis_loss",
                "frequency": 1,
            },
            }
        )
    
    def extra_training_step(self, data, result, meta, losses):
        optimizer_g, optimizer_d = self.optimizers()
        # train generator
        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()
        self.manual_backward(losses['loss'])
        self.clip_gradients(optimizer_g, gradient_clip_val=0.5, gradient_clip_algorithm="norm") # clip gradients
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)
        # train discriminator
        self.toggle_optimizer(optimizer_d)
        d_loss = losses["gan_dis_loss"]
        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        self.clip_gradients(optimizer_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm") # clip gradients
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)
        
        return losses
        
    def get_latent(self, data_x):
        data_x = self.normalize(data_x)
        data_x = self.embedding(data_x)
        latent_mu, latent_logvar = self.encode(data_x)
        return latent_mu.detach().cpu().numpy(), latent_logvar.detach().cpu().numpy()

    def decode_latent(self, latent, keeptensor=False):
        latent = self.decode(latent)
        latent = self.deembedding(latent)
        x_out = self.denormalize(latent)
        if not keeptensor:
            x_out = x_out.detach().cpu().numpy()
        return x_out

        