##################################
# Loss functions
##################################

import torch
import torch.nn.functional as F

dtype = torch.float32
device = "cpu"

TORCH_PI = torch.acos(torch.zeros(1))*2

def kld(mu, logvar): 
    # KLD between two univariate gaussians, explanation here:
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # Second Gaussian is zero mean and variance of 1, the prior on z
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1) # sum for all the latent variables
    return kld


def plain_mse(recon_x, tru_x, **kwargs):
    loss_rec = F.mse_loss(recon_x, tru_x, reduction='mean')
    return loss_rec

def ortho_loss(recon_x, tru_x, **kwargs):
    latent = kwargs["latent"]

def recon_loss_data(tru_x, mu_x, logvar_x):
    ## Basically only calculates log p(x|z) for one value of z taken from q(z|x) in the forward function of the model instead of calculating an expectation value
    # Sum on all the input distances
    loss_rec = -torch.sum(
        (-0.5 * torch.log(TORCH_PI.to(mu_x.device)))
        + (-0.5 * logvar_x)
        + ((-0.5 / torch.exp(logvar_x))
           			* (tru_x - mu_x) ** 2.0),
        axis=1
    )
    return loss_rec



def vae_loss(recon_x, tru_x, beta=1, **kwargs):
    # full vae loss for modeling a distributive latent space
    # AND a distributive reconstruction
    # correct formulation here:
    mu_latent = kwargs["mu_latent"]
    logvar_latent = kwargs["logvar_latent"]
    mu_x = kwargs["mu_x"]
    logvar_x = kwargs["logvar_x"]

    loss_rec = recon_loss_data(tru_x, mu_x, logvar_x)
    KLD = beta * kld(mu_latent, logvar_latent)
    # print(KLD)
    # print(loss_rec)
    loss = torch.mean(loss_rec + KLD, dim=0) # mean of batch
    return loss

def naive_vae_loss(recon_x, tru_x, beta=1, **kwargs):
    mu_latent = kwargs["mu_latent"]
    logvar_latent = kwargs["logvar_latent"]
    mu_x = kwargs["mu_x"]
    logvar_x = kwargs["logvar_x"]

    loss_rec = F.mse_loss(mu_x, tru_x, reduction='mean')

    KLD = beta * kld(mu_latent, logvar_latent)
    loss = torch.mean(loss_rec + KLD, dim=0)
    return loss


    

