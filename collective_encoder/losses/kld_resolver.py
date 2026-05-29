import importlib

_REGISTRY: dict = {
    "gaussian":         ("collective_encoder.losses.kld_uniform_gaussian",      "CELossKLDUniformGaussian"),
    "gaussian_mixture": ("collective_encoder.losses.kld_gaussian_mixture",      "CELossKLDGaussianMixture"),
    "nflow":             ("collective_encoder.losses.kld_flow",                  "CELossKLDFlow")
}


def get_kld_cls(name: str):
    """Return the KLD loss class for *name*.

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown KLD loss name: '{name}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[name]
    return getattr(importlib.import_module(module_path), class_name)

def create_kld_loss(name: str, args: dict, **kwargs):
    kld_cls = get_kld_cls(name)
    return kld_cls(args=args, **kwargs)