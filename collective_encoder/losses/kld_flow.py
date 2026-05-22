import torch

from .kld_uniform_gaussian import CELossKLDUniformGaussian

EPSILON = 1e-7


class CELossKLDFlow(CELossKLDUniformGaussian):
    _IDENTIFIER = "CELossKLDFlow"
    
    def kld(self, mu, logvar):
        """
        KLD between the Gaussian latent distribution and a Normalizing flow

        """
        
        return kld
