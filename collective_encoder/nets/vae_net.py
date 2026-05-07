import numpy as np
from typing import Any, List, Optional, Tuple, Union, Dict
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import pytorch_lightning as pl

from ase.data import covalent_radii

from gslibs.validation.input import check_mutually_exclusive

from collective_encoder.common.module import CEModule
from collective_encoder.nets.ae_base import AEBase
from collective_encoder.nets.modules.variational_encoder import VariationalNN

from metatomic.torch import ModelOutput

EPSILON = 1e-7

# ------------------------------------------------------------------
# KLD_max Schedulers
# ------------------------------------------------------------------

class KLDSchedulerBase(CEModule, ABC):
    _IDENTIFIER = ""
    _REQUIRED_ARGS = []

    def __init__(self, 
                 args, 
                 **kwargs):
        super().__init__(args=args, **kwargs)
    
    @abstractmethod
    def get_kld_max(self, plmodule: pl.LightningModule) -> float:
        raise NotImplementedError("get_kld_max must be implemented by subclasses")

    def on_validation_epoch_end(self, plmodule):
        """Optional hook that can be implemented by subclasses to update internal state at the end of each validation epoch."""
        pass

class KLDFixedScheduler(KLDSchedulerBase):
    _IDENTIFIER = "KLDFixedScheduler"
    _OPTIONAL_ARGS = {
        "value": 0.0,  # Fixed value for kld_max throughout training
    }

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__(args=args, **kwargs)
    
    def get_kld_max(self, plmodule):
        return self.value
    
class KLDLinearScheduler(KLDSchedulerBase):
    _IDENTIFIER = "KLDLinearScheduler"
    _REQUIRED_ARGS = ['start_value', 'end_value', 'start_epoch', 'end_epoch']

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__(args=args, **kwargs)
        if self.start_epoch >= self.end_epoch:
            self.raise_error("start_epoch must be less than end_epoch")
    
    def get_kld_max(self, plmodule):
        epoch = plmodule.current_epoch

        if epoch < self.start_epoch:
            return self.start_value
        elif epoch > self.end_epoch:
            return self.end_value
        else:
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            return self.start_value + progress * (self.end_value - self.start_value)

class KLDAutoScheduler(KLDSchedulerBase):
    _IDENTIFIER = "KLDAutoScheduler"
    _OPTIONAL_ARGS = {
        "kld_initial": 0.0,
        "kld_max":1.0,
        "increase_factor": 0.1,
        "monitor_metric": "val_rec_loss",
    }
    """
    Automatically sets kld_max according to the value of the monitored metric at validation epoch end (e.g. validation reconstruction loss).
    If the monitored metric is improving (decreasing), keeps the current kld_max value to keep improving with same regularization.
    Otherwise, gradually relax the regularization by the specified factor, allowing the model to focus more on reconstruction.
    
    This allows for a dynamic balance between reconstruction and regularization during training, potentially leading to better convergence and performance.
    
    """
    
    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__(args=args, **kwargs)
        self.value = self.kld_initial
        self.prev_metric = float('inf')
        
        if self.kld_initial < 0 or self.kld_max <= 0:
            self.raise_error("kld_initial must be >= 0 and kld_max must be > 0")
        if self.increase_factor <= 0:
            self.raise_error("increase_factor must be > 0")

    def get_kld_max(self, plmodule):
        return self.value
    
    def on_validation_epoch_end(self, plmodule):
        monitor_metric = self.monitor_metric
    
        metric_value = plmodule.trainer.callback_metrics.get(monitor_metric)
        if metric_value is None:
            self.raise_error(f"Monitor metric '{monitor_metric}' not found in callback metrics. "
                             f" Found metrics: {list(plmodule.trainer.callback_metrics.keys())}. ")
        if metric_value < self.prev_metric:
            self.prev_metric = metric_value
        else:
            self.value = min(self.kld_max, self.value * (1 + self.increase_factor) 
                             if self.value > 0 else 
                             self.kld_initial + self.increase_factor 
                                                * (self.kld_max - self.value))
            if self.value != self.kld_max:
                self.log_info(f"Validation metric '{monitor_metric}' did not improve "
                              f"(current: {metric_value:.4f}, best: {self.prev_metric:.4f}). "
                              f"Increasing kld_max to {self.value:.4f} for next epoch.")

def KLDResolver(kld_max_type, kld_max_scheduler_args, run_args):
    __REGISTRY__ = {
        'Fixed': KLDFixedScheduler,
        'Linear': KLDLinearScheduler,
        'Auto': KLDAutoScheduler,
    }
    if kld_max_type not in __REGISTRY__:
        raise ValueError(f"Invalid kld_max_type: {kld_max_type}. "
                         f"Must be one of {list(__REGISTRY__.keys())}")
    return __REGISTRY__[kld_max_type](args=kld_max_scheduler_args, **run_args)
        
# ------------------------------------------------------------------
# Metatomic Interface
# ------------------------------------------------------------------

class MetatomicModelVAE(torch.nn.Module):
    def __init__(self, 
                 encoder: torch.nn.Module,
                 normIn: bool = False,
                 dmean: torch.Tensor = torch.zeros(1), 
                 drange: torch.Tensor = torch.ones(1),
                 ):
        super().__init__()
        self.encoder = encoder

        self.register_buffer('normIn', torch.tensor(normIn, dtype=torch.bool))
        self.register_buffer('Mean', dmean)
        self.register_buffer('Range', drange)
    
    def get_metatomic_outputs(self):
        return {"features": ModelOutput(quantity="", unit="none", per_atom=False),}

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if self.normIn:
            # TorchScript-compatible broadcasting
            # Reshape Mean and Range to match x dimensions for broadcasting
            mean_expanded = self.Mean.view(1, -1).expand_as(x)
            range_expanded = self.Range.view(1, -1).expand_as(x)
            
            x = (x - mean_expanded) / range_expanded
        latent = self.encoder(x)
        mean, logvar = latent
        return mean

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class VAE(AEBase):
    _IDENTIFIER = "VAE"
    _COMPATIBLE_DATASETS = ["DEFAULT", "DISTANCES", "SOAP", "SOAP_PS"]
    _OPTIONAL_ARGS = AEBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "beta": 1.0,  # Weight for the KL divergence term in the loss function
        "kld_max_type": 'Fixed',
        "kld_max_scheduler_args": None,
        "use_bond_deviation_loss": False,  # Whether to include a bond deviation loss based on bonded atom pairs
        "use_steric_loss": False,
        "use_bond_deviation_loss": False,
    })

    def __init__(self,
                 datamodule,
                 args: Dict[str, Any] = None,
                 **kwargs
                 ):
        self.save_hyperparameters(ignore=['datamodule'])
        super().__init__(datamodule=datamodule, args=args, **kwargs)

        self.kld_scheduler = KLDResolver(self.kld_max_type, 
                                         self.kld_max_scheduler_args,
                                         kwargs)
        
        self.losses = {
            "rec_loss": self.recon_loss,
            "reg_loss": self.reg_loss,
        }
        if self.use_bond_deviation_loss:
            self.bond_indices = datamodule.get_bond_indices()
            self.atomic_numbers = datamodule.get_atns()
            self.losses["bond_deviation_loss"] = self.bond_deviation_loss
        if self.use_steric_loss:
            self.atomic_numbers = datamodule.get_atns()
            self.losses["steric_loss"] = self.steric_loss
        if self.use_bond_deviation_loss or self.use_steric_loss:
            cov_radii = [covalent_radii[el] for el in self.atomic_numbers]
            cov_radii = torch.tensor(cov_radii).float()
            cov_radii = cov_radii.unsqueeze(0)
            cd_t = cov_radii.transpose(0, 1)
            cov_mat = cov_radii.unsqueeze(0) + cd_t.unsqueeze(1)
            self.cov_mat = cov_mat.squeeze(1)

        self.metatomic_model_cls = MetatomicModelVAE
    
    def get_metatomic_model(self):
        model = self.metatomic_model_cls(
            encoder=self.encoder_net,
            normIn=self.normIn,
            dmean=self.Mean,
            drange=self.Range,
        )
        return model

    def get_metad_output(self, latent: Tuple[torch.Tensor, torch.Tensor], meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        # For metaD we use only use the mean of the latent distribution
        mean, logvar = latent
        return mean

    def aggregate_losses(self, losses):
        loss = losses['rec_loss'] + self.beta * losses['reg_loss']
        return loss

    def print_hparams(self):
        super().print_hparams()
        self.ce_log_dict("VAE hparams:", self.args)

    def init_network(self):
        self.encoder_net = VariationalNN(layers=self.network, 
                                         batch_norm=self.batch_norm)
        self.decoder_net = VariationalNN(layers=self.network[::-1], 
                                         batch_norm=self.batch_norm)

    def encoder(self, x):
        mu, logvar = self.encoder_net(x)
        return (mu, logvar), {}

    def decoder(self, z):
        mu_x, logvar_x = self.decoder_net(z)
        x_out = self.reparametrize_multivariate(mu_x, logvar_x)
        return x_out, {"mu_x": mu_x, "logvar_x": logvar_x}

    def latent_to_decoder_input(self, latent: Tuple[torch.Tensor, torch.Tensor]):
        mu_latent, logvar_latent = latent
        z = self.reparametrize_multivariate(mu_latent, logvar_latent)
        return z, {"mu_latent": mu_latent, "logvar_latent": logvar_latent, "z_sample": z}

    # @register_kl(MultivariateNormal, MultivariateNormal)
    def kld(self, mu, logvar):
        # KLD between univariate gaussian to Standard, explanation here:
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # Second Gaussian is zero mean and variance of 1, the prior on z
        # sum for all the latent variables
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)


        return kld
    
    def recon_loss(self, inp, latent, output, labels, meta):
        """
        Gaussian log-likelihood reconstruction loss. 
        Assumes the decoder outputs mean and log-variance of a Gaussian distribution over the input space.
        Computes the negative log-likelihood of the true input under this distribution."""
        
        x_norm = self.normalize(inp)
        mu_x = meta["mu_x"]
        logvar_x = meta["logvar_x"]

        logvar_x = torch.clamp(logvar_x, min=-4.0, max=4.0) # Clamp log-variance to prevent numerical instability in exp/log operations
        sd = torch.exp(0.5 * logvar_x) + EPSILON
        p_x = Normal(mu_x, sd)
        loss_rec = -torch.sum(p_x.log_prob(x_norm), dim=1)

        loss_rec = torch.mean(loss_rec)

        return loss_rec, {}

    def reg_loss(self, inp, latent, output, labels, meta):
        mu_latent = meta["mu_latent"]
        logvar_latent = meta["logvar_latent"]

        loss_kld = self.kld(mu_latent, logvar_latent)
        loss_reg = torch.mean(loss_kld, dim=0)
        
        meta = {"kld" : loss_reg}
        kld_max = self.kld_scheduler.get_kld_max(self)
        if loss_reg <= kld_max:
            loss_reg *= 0.0
        
        return loss_reg, meta
    
    def on_validation_epoch_end(self):
        self.kld_scheduler.on_validation_epoch_end(self)
    
    def bond_deviation_loss(self, inp, latent, output, labels, meta):
        bonded_indices = self.bond_indices
        coordinates = inp.view(inp.shape[0], -1, 3)
        n_atoms = coordinates.shape[-2]
        flattened_instances = coordinates.reshape(-1, n_atoms, 3)
        n_instances = flattened_instances.shape[0]
        flattened_coordinates = flattened_instances.reshape(-1, 3)

        for bond in bonded_indices:
            if bond[0] >= n_atoms or bond[1] >= n_atoms:
                raise ValueError(f"Invalid bond indices: {bond} for {n_atoms} atoms")

        mask1 = torch.zeros(len(bonded_indices), device=x.device)
        mask2 = torch.zeros(len(bonded_indices), device=x.device)
        cov_distances = torch.zeros(len(bonded_indices), device=x.device)

        for ind, (i, j) in enumerate(bonded_indices):
            mask1[ind] = i
            mask2[ind] = j
            cov_distances[ind] = self.cov_mat[i, j]

        mask1 = mask1.repeat(n_instances)
        mask2 = mask2.repeat(n_instances)
        cov_distances = cov_distances.repeat(n_instances)
        set1 = flattened_coordinates[mask1.long()]
        set2 = flattened_coordinates[mask2.long()]

        dist = F.pairwise_distance(set1, set2)
        deviation = (dist - cov_distances) ** 2
        return deviation.mean(), {}
    
    def steric_loss(self, inp, latent, output, labels, meta):
        coordinates = inp.view(inp.shape[0], -1, 3)
        n_atoms = coordinates.shape[-2]
        flattened_instances = coordinates.reshape(-1, n_atoms, 3)
        n_instances = flattened_instances.shape[0]
        flattened_coordinates = flattened_instances.reshape(-1, 3)

        n_pairs = n_atoms * (n_atoms - 1)
        mask1 = torch.zeros(n_pairs, device=inp.device)
        mask2 = torch.zeros(n_pairs, device=inp.device)
        cov_distances = torch.zeros(n_pairs, device=inp.device)

        ind = 0
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                mask1[ind] = i
                mask2[ind] = j
                cov_distances[ind] = self.cov_mat[i, j]
                ind += 1

        mask1 = mask1.repeat(n_instances)
        mask2 = mask2.repeat(n_instances)
        cov_distances = cov_distances.repeat(n_instances)
        set1 = flattened_coordinates[mask1.long()]
        set2 = flattened_coordinates[mask2.long()]

        dist = F.pairwise_distance(set1, set2)
        steric_mask = torch.where(dist > 0.5 * cov_distances, torch.zeros_like(dist), torch.ones_like(dist))
        strain = ((dist - cov_distances) ** 2) * steric_mask
        return strain.mean(), {}

    def plot_avg_sigma(self, latent_logvar):
        ld_mean = np.mean(np.exp(0.5 * latent_logvar), axis=0)
        lines = ["Avg. Sigma per LD:"] + [f"  LD {i}: {ld_mean[i]}" for i in range(len(ld_mean))]
        self.log_msg("\n".join(lines))

    def get_latent(self, data_x):
        data_x = self.normalize(data_x)
        (latent_mu, latent_logvar), _ = self.encoder(data_x)
        return latent_mu.detach().cpu().numpy(), latent_logvar.detach().cpu().numpy()

    def get_latent_mean(self, data_x):
        return self.get_latent(data_x)[0]

    def get_latent_names(self):
        return "mu_latent", "logvar_latent"
