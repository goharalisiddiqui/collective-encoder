from typing import Any, Dict

import numpy as np

import torch
from torch import nn

from .kld_uniform_gaussian import CELossKLDUniformGaussian

try:
    from nflows.flows.base import Flow, CompositeTransform
    from nflows.transforms.coupling import AdditiveCouplingTransform
    from nflows.transforms.permutations import RandomPermutation
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms.splines import rational_quadratic_spline
except ImportError:
    raise ImportError(
        "Please install nflows: pip install nflows\n"
    )

EPSILON = 1e-7

# Flow prior used in https://doi.org/10.1063/5.0105120
class CELossKLDFlow(CELossKLDUniformGaussian):
    """
    KLD loss against a learned normalizing flow prior.

    Instead of a fixed standard normal prior, this computes the KL divergence
    using a Neural Spline Flow (NSF) / RealNVP prior, as used in data-driven
    collective variable models. Requires the `nflows` library.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary. Required keys:
        - ``latent_dim`` (int): Dimensionality of the latent space.
        Optional keys:
        - ``num_blocks`` (int, default 4): Number of flow blocks.
        - ``hidden_units`` (int, default 20): Hidden units in the flow network.
        - ``spline_knots`` (int, default 31): Knots for the rational quadratic spline.
        - ``domain`` (tuple, default (-10.0, 10.0)): Domain boundaries for splines.
        Includes all base arguments from `CELossKLDUniformGaussian`.
    kwargs : dict
        Additional keyword arguments.
    """
    _IDENTIFIER = "CELossKLDFlow"
    _REQUIRED_ARGS = CELossKLDUniformGaussian._REQUIRED_ARGS + ['latent_dim']
    _OPTIONAL_ARGS = CELossKLDUniformGaussian._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "num_blocks": 4,
        "hidden_units": 20,
        "spline_knots": 31,
        "domain": (-10.0, 10.0),
    })
    
    def __init__(self,
                args: Dict[str, Any] = None,
                **kwargs):
        super().__init__(args=args, **kwargs)
        self.create_flow()

    def kld(self, 
            inp: torch.Tensor, 
            latent: torch.Tensor, 
            output: torch.Tensor, 
            labels: torch.Tensor, 
            meta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
        """
        Compute KL divergence between Gaussian encoder and NSF RealNVP prior.
        
        Mathematical formula:
            KL(q(z|x) || p(z)) = E[log q(z|x) - log p(z)]
            
            where:
            - q(z|x) = N(z; mu, exp(logvar))  [Gaussian encoder]
            - p(z) implicit from NSF: z' = f^{-1}(z) where z' ~ N(0, I)
            - log p(z) = log N(f^{-1}(z); 0, I) + log|det J_{f^{-1}}(z)|

        Parameters
        ----------
        inp : torch.Tensor
            Network input (unused).
        latent : torch.Tensor
            Sampled latent variables.
        output : torch.Tensor
            Network output (unused).
        labels : torch.Tensor
            Ground truth labels (unused).
        meta : dict
            Metadata dictionary containing ``mu_x`` and ``logvar_x``.

        Returns
        -------
        torch.Tensor
            Scalar KL divergence tensor.
        """
        mu = meta[self.mu_name]
        logvar = meta[self.logvar_name]
        
        # Step 1: Sample z from encoder q(z|x) = N(mu, exp(logvar))
        z = latent
        
        # Step 2: Compute log q(z|x) - log probability under encoder
        # For Gaussian: log N(z; mu, sigma^2) = -0.5 * sum(log(2*pi*sigma^2) + (z-mu)^2/sigma^2)
        variance = torch.exp(logvar)
        log_q = -0.5 * torch.sum(
            np.log(2 * np.pi) +          # log(2*pi)
            logvar +                      # log(sigma^2)
            (z - mu) ** 2 / variance,     # (z-mu)^2/sigma^2
            dim=1
        )
        # Shape: [batch_size]
        
        # Step 3: Transform z through flow: z' = f^{-1}(z)
        # The flow returns the transformed sample and log determinant jacobian
        z_transformed, ldj = self.flow.forward(z)
        # z_transformed shape: [batch_size, latent_dim]
        # ldj shape: [batch_size]
        
        # Step 4: Compute log p(z) using change of variables formula
        # Since z' = f^{-1}(z) and z' ~ N(0, I):
        # log p(z) = log N(z'; 0, I) + ldj
        
        # Log probability of transformed sample under standard normal N(0, I)
        log_p_z_transformed = -0.5 * torch.sum(
            z_transformed ** 2,
            dim=1
        ) - 0.5 * self.latent_dim * np.log(2 * np.pi)
        # Shape: [batch_size]
        
        # Apply change of variables
        log_p_z = log_p_z_transformed + ldj
        # Shape: [batch_size]
        
        # Step 5: KL divergence = E[log q(z|x) - log p(z)]
        kld = torch.mean(log_q - log_p_z)
        
        return kld 

    
    def create_flow(self):
        """
        Build NSF RealNVP flow using nflows library.
        
        Paper specification:
        - 4 RealNVP blocks
        - Each transforms half dimensions (alternating)
        - Rational quadratic spline
        - Domain: [-10, 10], 31 knot locations
        - Neural network: 1 hidden layer, 20 units, ReLU
        
        Note: Requires nflows library (`pip install nflows`).
        """
        
        bijectors = []
        num_blocks = self.num_blocks
        hidden_units = self.hidden_units
        spline_knots = self.spline_knots
        domain = self.domain
        latent_dim = self.latent_dim
        
        for i in range(num_blocks):
            # Determine mask for this block (alternating dimensions)
            if i % 2 == 0:
                mask = torch.cat([
                    torch.ones(latent_dim // 2),
                    torch.zeros(latent_dim - latent_dim // 2)
                ])
            else:
                mask = torch.cat([
                    torch.zeros(latent_dim // 2),
                    torch.ones(latent_dim - latent_dim // 2)
                ])
            
            num_transformed = int(mask.sum())
            num_identity = latent_dim - num_transformed
            
            def transform_net_create_fn(num_identity, num_transformed):
                main_nn = nn.Sequential(
                    nn.Linear(num_identity, hidden_units),
                    nn.ReLU(),
                )
                widths 
            # RealNVP with rational quadratic spline
            transform = AdditiveCouplingTransform(
                mask=mask,
                transform_net_create_fn=transform_net_create_fn,
                unconditional_transform=rational_quadratic_spline(
                    num_bins=spline_knots - 1,
                    tails='linear',
                    domain_min=domain[0],
                    domain_max=domain[1]
                )
            )
            
            bijectors.append(transform)
            
            # Add permutation between blocks
            if i < num_blocks - 1:
                bijectors.append(RandomPermutation(latent_dim))
        
        # Compose into flow
        
        flow = Flow(
            transform=bijectors[0] if len(bijectors) == 1 else CompositeTransform(bijectors),
            base_dist=StandardNormal(shape=[latent_dim])
        )
        
        return flow
