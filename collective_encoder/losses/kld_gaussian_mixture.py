from typing import Dict
import torch

from .kld_uniform_gaussian import CELossKLDUniformGaussian


class CELossKLDGaussianMixture(CELossKLDUniformGaussian):
    _IDENTIFIER = "CELossKLDGaussianMixture"
    _REQUIRED_ARGS = CELossKLDUniformGaussian._REQUIRED_ARGS + ['n_components']
    
    def kld(self, 
            inp: torch.Tensor, 
            latent: torch.Tensor, 
            output: torch.Tensor, 
            labels: torch.Tensor, 
            meta: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
        """
        KLD between the Gaussian latent distribution and a Gaussian Mixture prior.  
        The Gaussian Mixture is assumed to have 'n_components' components, each with mean 0 and variance 1, and equal mixing coefficients.  
        The KLD is computed as the difference between the log-probability of the latent under the Gaussian Mixture and the log-probability of the latent under the Gaussian distribution defined by (mu, logvar).
        
        Mathematical formula:
            KL(q(z|x) || p(z)) = E_{z~q(z|x)}[log q(z|x) - log p(z)]
            
            where:
            - q(z|x) = N(z; mu, exp(logvar))  [Gaussian encoder]
            - p(z) = (1/n_components) * sum_{k=1}^{n_components} N(z; 0, I)  [Gaussian Mixture prior]
            - log p(z) = log((1/n_components) * sum_{k=1}^{n_components} N(z; 0, I)) = logsumexp(-0.5 * sum(z^2 + log(2*pi)), dim=0) - log(n_components)
        """
        n_components = self.n_components
        mu = meta[self.mu_name]
        logvar = meta[self.logvar_name]
        z = latent
        # Compute log-probability of the latent under the Gaussian distribution defined by (mu, logvar)
        log_qz_x = -0.5 * torch.sum(torch.pow(z - mu, 2) / torch.exp(logvar) + logvar + torch.log(torch.tensor(2 * torch.pi)), axis=1)
        
        # Compute log-probability of the latent under the Gaussian Mixture prior with n_components components, each with variance 1 and mean 0
        log_pz = torch.logsumexp(-0.5 * torch.sum(z**2 + torch.log(torch.tensor(2 * torch.pi)), axis=1) - torch.log(torch.tensor(n_components)), dim=0)
        
        kld = log_qz_x - log_pz
        return kld
