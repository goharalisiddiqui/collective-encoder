import torch

from .kld_uniform_gaussian import CELossKLDUniformGaussian

EPSILON = 1e-7

# Flow prior used in https://doi.org/10.1063/5.0105120
class CELossKLDFlow(CELossKLDUniformGaussian):
    _IDENTIFIER = "CELossKLDFlow"
    
    def kld(self, mu, logvar):
        """
        KLD between the Gaussian latent distribution and a Normalizing flow (neural spline flows with a RealNVP structure) prior.

        """
        
        
        
        return kld
