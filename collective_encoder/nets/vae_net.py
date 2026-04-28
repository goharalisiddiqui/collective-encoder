import numpy as np
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

from ase.data import covalent_radii

from collective_encoder.nets.ae_base import AEBase
from collective_encoder.nets.modules.variational_encoder import VariationalNN

from metatomic.torch import ModelOutput

EPSILON = 1e-7

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

class VAE(AEBase):
    _COMPATIBLE_DATASETS = ["DEFAULT", "DISTANCES", "SOAP", "SOAP_PS"]

    def __init__(self,
                 datamodule,
                 network: List[int],
                 normIn: Optional[bool] = False,
                 lrate: float = 0.01,
                 weight_decay: float = 1e-7,
                 scheduler: bool = True,
                 scheduler_args: dict = None,
                 outname: str = './VAE_untitled/VAE_',
                 test_plotter : str = "LDplotter",
                 export_latent : bool = False,
                 beta: float = 1.0,
                 batch_norm: bool = True,
                 C_reg: Optional[Tuple[float, int, int]] = None,
                 C_auto: bool = False,
                 D_reg: Optional[float] = None,
                 use_steric_loss = False,
                 use_bond_deviation_loss = False,
                 ):
        self.save_hyperparameters(ignore=['datamodule'])
        # Checks
        assert len(network) >= 3, "Network must have at least 2 layers (input, hidden, output)"
        assert not all([a != None for a in [C_reg, D_reg]]), "C_reg and D_reg are incompatible, choose one of them"
        if C_reg is not None:
            assert len(C_reg) == 3, "C_reg must be a tuple of (C_value, start_epoch, end_epoch)"
            assert C_reg[0] >= 0.0, "C_value must be non-negative"
            assert C_reg[1] >= 0 and C_reg[2] >= 0, "start_epoch and end_epoch must be non-negative"
            assert not all([C_reg[0] != 0.0, C_auto == True]), "C_reg and C_auto are incompatible, choose one of them"
            assert C_reg[1] <= C_reg[2], f"Start epoch {C_reg[1]} must be less than or equal to end epoch {C_reg[2]} in C regulariser."

        assert datamodule.hparams.dataset_type in self._COMPATIBLE_DATASETS, \
            f"Datamodule {datamodule.hparams.dataset_type} is not compatible with {type(self).__name__}." \
            f"Compatible datamodules are: {self._COMPATIBLE_DATASETS}"

        nodes = [int(x) for x in network]
        datapoint_shape = datamodule.get_datapoint_shape()
        nodes.insert(0, datapoint_shape[0])
        super().__init__(dim_data=nodes[0],
                         dim_latent=nodes[-1],
                         normIn=normIn,
                         lrate=lrate,
                         weight_decay=weight_decay,
                         scheduler=scheduler,
                         scheduler_args=scheduler_args,
                         outname=outname,
                         test_plotter=test_plotter,
                         export_latent=export_latent,
                         )
        self.metatomic_model_cls = MetatomicModelVAE

        #### Setting up the layers of the network ####
        self.network = nodes
        self.init_network()

        self.losses = {
            "rec_loss": self.recon_loss,
            "reg_loss": self.reg_loss,
        }

        self.C_default = 1e-6

        if use_bond_deviation_loss:
            self.bond_indices = datamodule.get_bond_indices()
            self.atomic_numbers = datamodule.get_atns()
            self.losses["bond_deviation_loss"] = self.bond_deviation_loss
        if use_steric_loss:
            self.atomic_numbers = datamodule.get_atns()
            self.losses["steric_loss"] = self.steric_loss

        if use_bond_deviation_loss or use_steric_loss:
            cov_radii = [covalent_radii[el] for el in self.atomic_numbers]
            cov_radii = torch.tensor(cov_radii).float()
            cov_radii = cov_radii.unsqueeze(0)
            cd_t = cov_radii.transpose(0, 1)
            cov_mat = cov_radii.unsqueeze(0) + cd_t.unsqueeze(1)
            self.cov_mat = cov_mat.squeeze(1)

    
    def get_metatomic_model(self):
        model = self.metatomic_model_cls(
            encoder=self.encoder_net,
            normIn=self.hparams.normIn,
            dmean=self.Mean,
            drange=self.Range,
        )
        return model

    def get_metad_output(self, latent: Tuple[torch.Tensor, torch.Tensor], meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        # For metaD we use only use the mean of the latent distribution
        mean, logvar = latent
        return mean

    def aggregate_losses(self, losses):
        loss = losses['rec_loss'] + self.hparams.beta * losses['reg_loss']
        return loss

    def print_hparams(self):
        super().print_hparams()
        hparams: dict = {"beta": self.hparams.beta}
        if self.hparams.C_reg is not None:
            hparams["C_reg"] = {
                "value": self.hparams.C_reg[0],
                "start": self.hparams.C_reg[1],
                "end":   self.hparams.C_reg[2],
            }
        if self.hparams.C_auto:
            hparams["C_auto"] = True
        if self.hparams.D_reg is not None:
            hparams["D_reg"] = self.hparams.D_reg
        self.log_msg_dict("VAE hparams:", hparams)

    def init_network(self):
        self.log_msg(f"[Initializing {type(self).__name__} Module] hidden layers: {self.network}")
        self.print_hparams()
        self.encoder_net = VariationalNN(layers=self.network, batch_norm=self.hparams.batch_norm)
        self.decoder_net = VariationalNN(layers=self.network[::-1], batch_norm=self.hparams.batch_norm)

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
    
    def recon_loss(self, x, latent, pred, meta):
        # Compute NLL in normalized space so mu_x, logvar_x, and x are on the same scale.
        x_norm = self.normalize(x)
        mu_x = meta["mu_x"]
        logvar_x = meta["logvar_x"]

        sd = torch.exp(0.5 * logvar_x) + EPSILON
        p_x = Normal(mu_x, sd)
        loss_rec = -torch.mean(p_x.log_prob(x_norm), axis=1)

        loss_rec = torch.mean(loss_rec)

        return loss_rec, {}

    def reg_loss(self, x, latent, pred, meta):
        z_sample = meta["z_sample"]
        mu_latent = meta["mu_latent"]
        logvar_latent = meta["logvar_latent"]

        loss_kld = self.kld(mu_latent, logvar_latent)
        loss_reg = torch.mean(loss_kld, dim=0)
        
        meta = {"kld" : loss_reg}
        if self.hparams.C_reg is not None:
            C = self.C_default
            cmax, c_start, c_end = self.hparams.C_reg[0], self.hparams.C_reg[1], self.hparams.C_reg[2]
            if self.current_epoch >= c_start and self.current_epoch <= c_end:
                C = cmax * (self.current_epoch - c_start) / (c_end - c_start)
            elif self.current_epoch > c_end:
                C = cmax
            loss_reg = torch.abs(loss_kld - C)
            meta["C"] = C

        if self.hparams.D_reg is not None:
            if torch.mean(loss_reg) < self.hparams.D_reg:
                loss_reg *= 0.0
        
        return loss_reg, meta
    
    def bond_deviation_loss(self, x, latent, pred, meta):
        bonded_indices = self.bond_indices
        coordinates = x.view(x.shape[0], -1, 3)
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
    
    def steric_loss(self, x, latent, pred, meta):
        coordinates = x.view(x.shape[0], -1, 3)
        n_atoms = coordinates.shape[-2]
        flattened_instances = coordinates.reshape(-1, n_atoms, 3)
        n_instances = flattened_instances.shape[0]
        flattened_coordinates = flattened_instances.reshape(-1, 3)

        n_pairs = n_atoms * (n_atoms - 1)
        mask1 = torch.zeros(n_pairs, device=x.device)
        mask2 = torch.zeros(n_pairs, device=x.device)
        cov_distances = torch.zeros(n_pairs, device=x.device)

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
    
    def on_validation_epoch_end(self):
        if self.hparams.C_auto:
            val_rec_loss = self.trainer.callback_metrics.get("val_rec_loss")
            val_kld = self.trainer.callback_metrics.get("val_kld")
            if val_rec_loss is not None:
                prev = getattr(self, "_prev_val_rec_loss", None)
                if prev is not None and val_rec_loss.item() > prev:
                    self.C_default = val_kld.item() if val_kld is not None else self.C_default
                    self.log_msg(f"C_Scheduler: Setting C_default to {self.C_default} based on validation loss increase.")
                else:
                    self.C_default = 1e-6
                    self.log_msg("C_Scheduler: Resetting C_default to 1e-6 based on validation loss decrease.")
                self._prev_val_rec_loss = val_rec_loss.item()
        return super().on_validation_epoch_end()

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
