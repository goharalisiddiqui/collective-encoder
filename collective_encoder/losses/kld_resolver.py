from collective_encoder.common.registry import get_module

_MAP: dict = {
    "gaussian":          "CELossKLDUniformGaussian",
    "gaussian_mixture":  "CELossKLDGaussianMixture",
    "nflow":             "CELossKLDFlow"
}

def create_kld_loss(name: str, args: dict, **kwargs):
    """
    Factory function to instantiate a KLD loss.

    Parameters
    ----------
    name : str
        The registered string identifier.
    args : dict
        Configuration dictionary to pass to the loss class.
    kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    CELossBase
        An instantiated KLD loss module.
    """
    
    kld_cls = get_module("losses", _MAP[name])
    return kld_cls(args=args, **kwargs)