from abc import ABC
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.distributions.normal import Normal
torch.manual_seed(0)
TORCH_PI = torch.acos(torch.zeros(1))*2
import pytorch_lightning.loggers as pl_loggers


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from statistics import mean as list_mean

from scipy.stats import multivariate_normal

class AEBase(pl.LightningModule, ABC):
    def __init__(self,
                 dim_data : int,
                 dim_latent : int,
                 normIn : bool = False,
                 lrate : float = 0.01,
                 weight_decay : float = 1e-7,
                 scheduler : bool = False,
                 scheduler_args : dict = {},
                 outname : str = './untitled/untitled_',
                 test_plotter : str = None,
                 ):
        super().__init__()


        # Model meta info
        self.dim_data = dim_data
        self.dim_latent = dim_latent

        self.losses = {
            "rec_loss": self.loss_mse,
        }
        self.test_metrics = {
            "mae": self.metric_mae,
        }

        self.metaD = False
        self.register_buffer('normIn', torch.tensor(normIn, dtype=torch.bool))
        self.register_buffer('normSet', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('Mean', torch.zeros(dim_data))
        self.register_buffer('Range', torch.ones(dim_data))
        self.save_hyperparameters()

    def set_norm(self, Mean: torch.Tensor, Range: torch.Tensor):
        if not self.trainer.datamodule:
            raise RuntimeError("Trainer datamodule not found; cannot compute normalization.")
        with torch.no_grad():
            Mean = torch.tensor(self.trainer.datamodule.get_scaler_mean(), device=self.device)
            Range = torch.tensor(self.trainer.datamodule.get_scaler_scale(), device=self.device)
            Range = Range.clone()
            Range[Range == 0.0] = 1.0
            print(f"\n[{type(self).__name__}] Setting normalization for inputs.")
            self.Mean = Mean
            self.Range = Range
            self.normSet = torch.tensor(True, dtype=torch.bool)

    def normalize(self, x: Variable):
        if not self.normIn:
            return x
        if not self.normSet:
            self.set_norm()
        batch_size = x.size(0)
        x_size = x.size()[1:]

        Mean = self.Mean.unsqueeze(0).expand(batch_size, *x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, *x_size)
        
        return x.sub(Mean).div(Range)

    def denormalize(self, x: Variable):
        if not self.normIn:
            return x
        if not self.normSet:
            self.set_norm()

        batch_size = x.size(0)
        x_size = x.size()[1:]

        Mean = self.Mean.unsqueeze(0).expand(batch_size, *x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, *x_size)

        return x.mul(Range).add(Mean)

    def reparametrize(self, mu, logvar): # Drawing a random sample from the normal distribution mu, logvar
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def reparametrize_multivariate(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        dist = torch.distributions.MultivariateNormal(torch.zeros(mu.shape[1]), torch.eye(mu.shape[1]))
        samples = dist.rsample(mu.shape[:-1]).to(mu.device)
        return mu + samples*std

    def get_train_parameters(self):
        return self.parameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.get_train_parameters(), lr=self.hparams.lrate, weight_decay= self.hparams.weight_decay)
        if self.hparams.scheduler == False:
            return optimizer
        scheduler_args = {
            "mode": "min",
            "factor": 0.8,
            "patience": 3,
            "min_lr": 1e-10,
            "cooldown": 10,
        }
        scheduler_args.update(self.hparams.scheduler_args)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)
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
        print(f"Starting training {type(self).__name__} module")
        print("==================================")
        print("[Optimization Settings]")
        print("- Learning rate \t=", self.hparams.lrate)
        print("- l2 regularization \t=", self.hparams.weight_decay)
        if self.hparams.scheduler:
            print("- Learning rate scheduler \t= Enabled")
            if len(self.hparams.scheduler_args.keys()) > 0:
                print("  ", self.hparams.scheduler_args)
        else:
            print("- Learning rate scheduler \t= Disabled")
        print("[Hyperparameters]")
        self.print_hparams()
        print("==================================\n\n")

    def print_hparams(self):
        # This prints the hparams in child models
        return
    
    def extra_training_step(self, data, latent, result, meta, losses):
        # This implements any extra training in child class
        return losses

    def training_step(self, train_batch, batch_idx):
        self.step(train_batch, "train")
    
    def validation_step(self, val_batch, batch_idx):
        self.step(val_batch, "val")

    def encoder(self, x):
        raise NotImplementedError("This function must be implemented in child class")
    
    def decoder(self, z):
        raise NotImplementedError("This function must be implemented in child class")

    def latent_to_decoder_input(self, latent):
        return latent, {}
    
    def get_metad_output(self, latent, meta):
        return latent

    def forward(self, x):
        meta = {}
        x = self.normalize(x)
        latent, meta_latent = self.encoder(x)

        latent, meta_sample = self.latent_to_decoder_input(latent)

        if self.metaD:
            meta.update(meta_latent)
            return self.get_metad_output(latent, meta)

        pred, meta_dec = self.decoder(latent)
        meta.update(meta_latent)
        meta.update(meta_dec)
        meta.update(meta_sample)
        pred = self.denormalize(pred)
        return latent, pred, meta

    def loss_mse(self, x, latent, pred, meta):
        loss = F.mse_loss(pred, x, reduction='none')
        loss = torch.mean(loss)

        mae = F.l1_loss(pred, x, reduction='none')
        mae = torch.mean(mae)
        return loss, { "mae": mae.item() }

    def aggregate_losses(self, losses):
        return torch.sum(losses)

    def step(self, batch, stage: str):
        data = batch[0]
        latent, pred, meta = self(data)
        batch_size = self.trainer.datamodule.hparams.batch_size if self.trainer and self.trainer.datamodule else None
        
        losses = {}
        for loss_name, loss_func in self.losses.items():
            # print(f"\n\nComputing {loss_name}...\n\n")
            loss, loss_meta = loss_func(data, latent, pred, meta)
            self.log(f"{stage}_{loss_name}", loss, prog_bar=(stage=="train"), 
                    on_step=(stage=="train"), on_epoch=True, batch_size=data.shape[0])
            losses[loss_name] = loss
            meta.update(loss_meta)
        losses = self.extra_training_step(data, latent, pred, meta, losses)
        loss = torch.stack(list(losses.values()))

        loss = self.aggregate_losses(loss)

        self.log(f"{stage}_loss", loss, prog_bar=(stage=="train"), 
                 on_step=(stage=="train"), on_epoch=True, batch_size=batch_size)
        for key, value in meta.items():
            if isinstance(value, (int, float)):
                self.log(f"{stage}_{key}", value, prog_bar=False, on_step=(stage=="train"), on_epoch=True, batch_size=batch_size)
        return loss

    def metric_mae(self, x, latent, prec, meta):
        mae = F.l1_loss(prec, x, reduction='none')
        mae = torch.mean(mae)
        return mae, {}

    def plotter(self, data, latent, pred, meta):
        if self.hparams.test_plotter is None:
            return
        if self.hparams.test_plotter == "LDplotter":
            from plotters.latent_space_plotter import LDplotter
            LDplotter(data, latent, pred, meta, logger=self.logger.experiment, outstem=self.hparams.outname)
        else:
            raise ValueError(f"Unknown plotter: {self.hparams.test_plotter}")
        return

    def test_step(self, test_batch, batch_idx):
        data = test_batch[0]
        latent, pred, meta = self(data)
        batch_size = self.trainer.datamodule.hparams.test_batch_size if self.trainer and self.trainer.datamodule else None

        for metric_name, metric_func in self.test_metrics.items():
            metric, metric_meta = metric_func(data, latent, pred, meta)
            self.log(f"test_{metric_name}", metric, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
            for key, value in metric_meta.items():
                if isinstance(value, (int, float)):
                    self.log(f"test_{key}", value, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
            meta.update(metric_meta)
        if self.hparams.test_plotter is not None:
            self.plotter(data, latent, pred, meta)
        return

    def plot_extra(self, data_x, data_y, latents):
        # This implements any extra printing or plotting in child class
       return

    def get_latent(self, data_x):
        data_x = data_x.float()
        data_x = self.normalize(data_x)
        latent, meta_latent = self.encoder(data_x)
        return latent

    def get_latent_names(self):
        return "latent"