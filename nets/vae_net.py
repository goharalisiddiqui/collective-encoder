from nets.ae_base import AEBase
import pandas as pd
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import register_kl
torch.manual_seed(0)
TORCH_PI = torch.acos(torch.zeros(1))*2
import argparse
from warnings import warn

from ase.data import covalent_radii


pd.set_option('display.max_columns', None)


EPSILON = 1e-9

torch.set_printoptions(threshold=10_000)


def vae_parse_args():
    desc = "VAE NN for enhanced sampling MD"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--beta', required=True, type=float,
                        help='beta for the beta-VAE')

    parser.add_argument('--cauto', action='store_true', dest='C_auto', 
                        help='Activate automatic C scheduler for information control in Beta-VAE')
    
    parser.add_argument('--cmax', type=float, default=0.0, dest='C_max',
                        help='Maximum C for information control in Beta-VAE')
    parser.add_argument('--cstart', type=int, default=0, dest='C_start',
                        help='Epoch where C start to increase for information control in Beta-VAE')
    parser.add_argument('--cend', type=int, default=0, dest='C_end',
                        help='Epoch where C stops to increase for information control in Beta-VAE')

    parser.add_argument('--dvalue', dest='D', default=0.0, type=float,
                        help='D value for the beta-VAE')

    args, _ = parser.parse_known_args()

    return args


VAE_args = vae_parse_args


class VAE(AEBase):
    def __init__(self,
                 l: list,
                 lr: float = 0.01,
                 l2_reg: float = 1e-7,
                 beta: float = 1.0,
                 batch_norm: bool = True,
                 lr_scheduler: bool = True,
                 plot_every: int = 0,
                 C_max: float = 0.0,
                 C_start: int = 0,
                 C_end: int = 0,
                 C_auto: bool = False,
                 D: float = 0.0,
                 atomic_numbers = None,
                 bond_indices = None,
                 use_steric_loss = False,
                 use_bond_deviation_loss = False,
                 saveplotdata: bool = False,
                 outname: str = './VAE_untitled/VAE_',
                 ):
        super().__init__(dim_data=l[0],
                         dim_latent=l[-1],
                         lr=lr,
                         l2_reg=l2_reg,
                         lr_scheduler=lr_scheduler,
                         outname=outname,
                         plot_every=plot_every,
                         saveplotdata=saveplotdata)
        assert len(l) >= 3
        self.save_hyperparameters()
        assert any([a == 0.0 for a in [C_max, D]]), "Atleast one of C_max and D should be zero" 
        
        #### Setting up the layers of the netwrok ####
        self.init_network()

        # Arguments Checks
        assert not all([C_max != 0.0, C_auto == True]), "C_max and C_auto are incompatible, choose one of them"
        assert C_start <= C_end, "C_start must be less than or equal to C_end"

        self.C_default = 1e-6
        if use_bond_deviation_loss:
            if bond_indices is None or atomic_numbers is None:
                raise ValueError(
                    "Bond indices and atomic numbers must be provided for bond deviation loss")
        if use_steric_loss:
            if atomic_numbers is None:
                raise ValueError(
                    "Atomic numbers must be provided for steric loss")
        if atomic_numbers is not None:
            cov_radii = [covalent_radii[el] for el in atomic_numbers]
            cov_radii = torch.tensor(cov_radii).float()
            cov_radii = cov_radii.unsqueeze(0)
            cd_t = cov_radii.transpose(0, 1)
            cov_mat = cov_radii.unsqueeze(0) + cd_t.unsqueeze(1)
            self.cov_mat = cov_mat.squeeze(1)
            # print(self.cov_mat)
            # exit()

    def init_network(self):
        print(f"[Initializing {type(self).__name__} Module]")
        print("- hidden layers:", self.hparams.l)
        print("")
        print("========= NN =========")
        self.init_encoder()
        print("(Reparameterization Sampler)\n\n")
        self.init_decoder()
        print("======================")

    def init_encoder(self):
        self.init_encoder_layers()
        self.init_encoder_output()

    def init_decoder(self):
        self.init_decoder_layers()
        self.init_decoder_output()

    def init_encoder_layers(self):
        l = self.hparams.l
        batch_norm = self.hparams.batch_norm
        encoder_layers = []
        for i in range(len(l) - 2):
            print(l[i], " --> ", l[i + 1], end=" ")
            encoder_layers.append(nn.Linear(l[i], l[i + 1]))
            encoder_layers.append(nn.ReLU(True))
            print("(relu)")
            if batch_norm:
                encoder_layers.append(nn.BatchNorm1d(l[i + 1]))
                print("(batch_normalization layer)")
        self.encoder_hidden = nn.Sequential(*encoder_layers)

    def init_decoder_layers(self):
        l = self.hparams.l
        batch_norm = self.hparams.batch_norm
        decoder_layers = []
        a = len(l) - 1
        for i in range(len(l) - 2):
            print(l[a - i], " --> ", l[a - i - 1], end=" ")
            decoder_layers.append(nn.Linear(l[a - i], l[a - i - 1]))
            decoder_layers.append(nn.ReLU(True))
            print("(relu)")
            if batch_norm:
                decoder_layers.append(nn.BatchNorm1d(l[a - i - 1]))
                print("(batch_normalization layer)")
        self.decoder_hidden = nn.Sequential(*decoder_layers)

    def init_encoder_output(self):
        l = self.hparams.l
        self.encoder_mu = nn.Linear(l[-2], l[-1])
        print(l[-2], " --> ", l[-1], end=" ")
        print("(mu for latent space)")
        self.encoder_logvar = nn.Linear(l[-2], l[-1])
        print("  ", " \--> ", l[-1], end=" ")
        print("(logvar for latent space)\n\n")

    def init_decoder_output(self):
        l = self.hparams.l
        self.decoder_mu = nn.Linear(l[1], l[0])
        print(l[1], " --> ", l[0], end=" ")
        print("(mu for feature space)")
        self.decoder_logvar = nn.Linear(l[1], l[0])
        print("  ", " \--> ", l[0], end=" ")
        print("(logvar for feature space)\n\n")
        print("======================")

    def print_hparams(self):
        print("- Beta \t=", self.hparams.beta)

    def encode(self, x):
        x = self.encoder_hidden(x)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.decoder_hidden(z)
        mu_x = self.decoder_mu(z)
        logvar_x = self.decoder_logvar(z)
        return mu_x, logvar_x

    def forward(self, x):
        x = self.normalize(x)
        mu_latent, logvar_latent = self.encode(x)
        if self.metaD:
            return mu_latent, logvar_latent
        if mu_latent.isnan().any() or logvar_latent.isnan().any():
            warn("Nan in encoder network (Gradient diminished or exploded)")
            # exit()

        z = self.reparametrize_multivariate(mu_latent, logvar_latent)

        mu_x, logvar_x = self.decode(z)  # q(x|z)
        if mu_x.isnan().any() or logvar_x.isnan().any():
            warn("Nan in decoder network (Gradient diminished or exploded)")
            # exit()
        x_out = self.reparametrize_multivariate(mu_x, logvar_x)
        x_out = self.denormalize(x_out)
        mu_x = self.denormalize(mu_x)

        return x_out, {"mu_latent" : mu_latent, "logvar_latent" : logvar_latent,
                    "mu_x" : mu_x, "logvar_x" : logvar_x, "z_sample" : z}


    # @register_kl(MultivariateNormal, MultivariateNormal)
    def kld(self, mu, logvar):
        # KLD between univariate gaussian to Standard, explanation here:
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # Second Gaussian is zero mean and variance of 1, the prior on z
        # sum for all the latent variables
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)

        # q = MultivariateNormal(mu, torch.diag_embed(torch.exp(logvar)))
        # p = MultivariateNormal(torch.zeros_like(mu), torch.eye(mu.size(1)))
        # kld = torch.distributions.kl.kl_divergence(q, p)
        # kld = torch.sum(kld, axis=1) # sum for all the latent variables

        

        return kld

    def recon_loss(self, tru_x, mu_x, logvar_x):

        sd = torch.exp(0.5 * logvar_x) + EPSILON
        p_x = Normal(mu_x, sd)
        loss_rec = -torch.mean(p_x.log_prob(tru_x), axis=1)

        # if (loss_rec < 0.0).any().detach().cpu().numpy():
        #     var = 0.5 * torch.exp(logvar_x)
        #     print("\n loss_rec= ", loss_rec)
        #     print("\n var= ", var)
        #     print("\n tru_x= ", tru_x)
        #     print("\n log_prob= ", p_x.log_prob(tru_x))
        #     exit()

        return loss_rec

    def reg_loss(self, z_sample, mu_latent, logvar_latent):
        loss_kld = self.kld(mu_latent, logvar_latent)
        loss_kld = torch.mean(loss_kld, dim=0)

        C = self.C_default
        if self.hparams.C_max != 0.0:
            c_start, c_end, cmax = self.hparams.C_start, self.hparams.C_end, self.hparams.C_max
            if self.current_epoch >= c_start and self.current_epoch <= c_end:
                C = cmax * (self.current_epoch - c_start) / (c_end - c_start)
            elif self.current_epoch > c_end:
                C = cmax
        loss_reg = torch.abs(loss_kld - C)

        if torch.mean(loss_reg) < self.hparams.D:
            loss_reg *= 0.0

        return loss_reg, {"current_C" : C, "kld" : loss_kld}

    def mae_loss(self, recon_x, tru_x):
        loss_mae = F.l1_loss(recon_x, tru_x, reduction='none')
        loss_mae = torch.mean(loss_mae, dim=1)
        loss_mae = torch.mean(loss_mae, dim=0)

        return loss_mae
    
    def bond_deviation_loss(self, recon_x):
        bonded_indices = self.hparams.bond_indices
        
       
        coordinates = recon_x.view(recon_x.shape[0], -1, 3)
        # print(f"Coordinates shape: {coordinates.shape}")
        # print(f"Coordinates: {coordinates}")
        
        # Reshape the coordinates to get the flattened coordinates to be used in the pairwise distance
        n_atoms = coordinates.shape[-2]
        flattened_instances = coordinates.reshape(-1, n_atoms, 3)
        n_instances = flattened_instances.shape[0]
        flattened_coordinates = flattened_instances.reshape(-1, 3)
        # print(f"natoms: {n_atoms}")
        # print(f"flattened_instances: {flattened_instances.shape}")
        # print(f"n_instances: {n_instances}")
        # print(f"flattened_coordinates: {flattened_coordinates.shape}")
        
        # print(f"Flattened Coordinates: {flattened_coordinates.shape}")
        # print(f"Number of Instances: {n_instances}")
        # print(f"Number of Atoms: {n_atoms}")
        for bond in bonded_indices:
            if bond[0] >= n_atoms or bond[1] >= n_atoms:
                raise ValueError(
                    f"Invalid bond indices: {bond} for {n_atoms} atoms")
        

        # Mask the non-bonded atoms
        mask1 = torch.zeros(len(bonded_indices), device=recon_x.device)
        mask2 = torch.zeros(len(bonded_indices), device=recon_x.device)
        cov_distances = torch.zeros(
            len(bonded_indices), device=recon_x.device)

        for ind, (i, j) in enumerate(bonded_indices):
            mask1[ind] = i
            mask2[ind] = j
            cov_distances[ind] = self.cov_mat[i, j]
        # print(f"cov_distances shape: {cov_distances.shape}")
        # print(f"cov_distances: {cov_distances}")
        mask1 = mask1.repeat(n_instances)
        mask2 = mask2.repeat(n_instances)
        cov_distances = cov_distances.repeat(n_instances)
        set1 = flattened_coordinates[mask1.long()]
        set2 = flattened_coordinates[mask2.long()]

        # print(f"Set1: {set1.shape}")
        # print(f"Set2: {set2.shape}")

        # Calculate the pairwise distance between the bonded atoms
        dist = F.pairwise_distance(set1, set2)
        # print(f"Distance shape: {dist.shape}")
        # exit()


        deviation = (dist - cov_distances) ** 2
        # print(f"Deviation: {deviation[:len(bonded_indices)]}")
        # print(f"cov_distances: {cov_distances[:len(bonded_indices)]}")
        # exit()
        
        # print(f"\nDeviation: {deviation.mean()}")
        return deviation.mean()
    
    def steric_loss(self, recon_x: torch.Tensor):
        coordinates = recon_x.view(recon_x.shape[0], -1, 3)
        # print(f"Coordinates: {coordinates}")
        
        # Reshape the coordinates to get the flattened coordinates to be used in the pairwise distance
        n_atoms = coordinates.shape[-2]
        flattened_instances = coordinates.reshape(-1, n_atoms, 3)
        n_instances = flattened_instances.shape[0]
        flattened_coordinates = flattened_instances.reshape(-1, 3)
        # print(f"natoms: {n_atoms}")
        # print(f"flattened_instances: {flattened_instances.shape}")
        # print(f"n_instances: {n_instances}")
        # print(f"flattened_coordinates: {flattened_coordinates.shape}")
        
        # print(f"Flattened Coordinates: {flattened_coordinates.shape}")
        # print(f"Number of Instances: {n_instances}")
        # print(f"Number of Atoms: {n_atoms}")

        n_pairs = n_atoms * (n_atoms - 1)
        mask1 = torch.zeros(n_pairs, device=recon_x.device)
        mask2 = torch.zeros(n_pairs, device=recon_x.device)
        cov_distances = torch.zeros(
            n_pairs, device=recon_x.device)

        # for ind, (i, j) in enumerate(bonded_indices):
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

        # print(f"Set1: {set1.shape}")
        # print(f"Set2: {set2.shape}")

        # Calculate the pairwise distance between the bonded atoms
        dist = F.pairwise_distance(set1, set2)
        
        # print(f"Distance shape: {dist.shape}")
        # print(f"Distance: {dist[:10]}")
        # print(f"cov_distances shape: {cov_distances.shape}")
        # print(f"cov_distances: {cov_distances[:10]}")
        # exit()

        steric_mask = torch.where(dist > 0.5 * cov_distances, torch.zeros_like(dist), torch.ones_like(dist))
        
        strain = ((dist - cov_distances) ** 2) * steric_mask
        # print(f"Deviation: {deviation[:len(bonded_indices)]}")
        # print(f"cov_distances: {cov_distances[:len(bonded_indices)]}")
        # exit()
        
        # print(f"\nDeviation: {deviation.mean()}")
        return strain.mean()

    def loss(self, recon_x, tru_x, **kwargs):
        mu_latent = kwargs["mu_latent"]
        logvar_latent = kwargs["logvar_latent"]
        mu_x = kwargs["mu_x"]
        logvar_x = kwargs["logvar_x"]
        z_sample = kwargs["z_sample"]

        loss_rec = self.recon_loss(tru_x, mu_x, logvar_x)
        loss_reg, meta_reg = self.reg_loss(z_sample, mu_latent, logvar_latent)

        loss_rec = torch.mean(loss_rec)  # Mean of batch
        loss_reg = torch.mean(loss_reg)  # Mean of batch
        loss = loss_rec + self.hparams.beta * loss_reg
        
        if self.hparams.use_bond_deviation_loss:
            loss_bond = self.bond_deviation_loss(recon_x)
            loss += loss_bond
        
        if self.hparams.use_steric_loss:
            loss_steric = self.steric_loss(recon_x)
            loss += loss_steric * 100

        loss_mae = self.mae_loss(recon_x, tru_x)
        
        return_val  = {'loss': loss,
                'mae_loss': loss_mae,
                'rec_loss': loss_rec,
                'reg_loss': loss_reg,
                "current_C": meta_reg["current_C"],
                "kld": meta_reg["kld"]}
        
        if self.hparams.use_bond_deviation_loss:
            return_val["bond_deviation_loss"] = loss_bond
        
        if self.hparams.use_steric_loss:
            return_val["steric_loss"] = loss_steric
            
        return return_val
    
    def on_validation_epoch_end(self):
        if self.hparams.C_auto \
            and self.losses.get("val_rec_loss") is not None \
            and len(self.losses.get("val_rec_loss")) > 1:
            if self.losses.get("val_rec_loss")[-1] > self.losses.get("val_rec_loss")[-2]:
                self.C_default = self.losses.get("val_kld")[-1]
                print(f"\nC_Scheduler: Setting C_default to {self.C_default} based on validation loss increase.")
            else:
                self.C_default = 1e-6
                print("\nC_Scheduler: Resetting C_default to 1e-6 based on validation loss decrease.")
        return super().on_validation_epoch_end()

    def plot_avg_sigma(self, latent_logvar):
        # This implements any extra printing or plotting in child class
        ld_mean = np.mean(np.exp(0.5 * latent_logvar), axis=0)
        print("========= Avg. Sigma per LD =========")
        for i in range(len(ld_mean)):
            print(f"LD {i} : {ld_mean[i]}")
        print("=====================================")

    def plot_extra(self, data_x, data_y, latents):
        latent_logvar = latents[1]
        self.plot_latent(latents, data_y, self.plot_sd, "latent_pdf")
        self.plot_avg_sigma(latent_logvar)

    def get_latent(self, data_x):
        data_x = self.normalize(data_x)
        latent_mu, latent_logvar = self.encode(data_x)
        return latent_mu.detach().cpu().numpy(), latent_logvar.detach().cpu().numpy()

    def get_latent_mean(self, data_x):
        return self.get_latent(data_x)[0]

    def get_latent_names(self):
        return "mu_latent", "logvar_latent"

    def plot_sd(self, fig, ax, latents, train_y, i, yaxis, label, scalarMap=None):
        latent_mu, latent_logvar = latents
        latent_sd = np.exp(0.5 * latent_logvar)
        ax.errorbar(latent_mu[:, i], latent_mu[:, yaxis], xerr=latent_sd[:, i], yerr=latent_sd[:, yaxis],
                    ecolor=scalarMap.to_rgba(train_y) if train_y is not None else None, alpha=0.1, ls='none')
