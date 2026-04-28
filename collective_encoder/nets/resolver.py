import importlib

_REGISTRY: dict = {
    "VAE":    ("collective_encoder.nets.vae_net",   "VAE"),
    "AE":     ("collective_encoder.nets.ae_net",    "AE"),
    "DVAE":   ("collective_encoder.nets.dvae_net",  "DVAE"),
    "EDVAE":  ("collective_encoder.nets.edvae_net", "EDVAE"),
    "BGE":    ("collective_encoder.nets.bge",    "BondGraphEncoderDecoder"),
    "BGE_V2": ("collective_encoder.nets.bge_v2", "BondGraphEncoderDecoderV2"),
}


def get_net(model_name: str):
    """Return the neural network class for *model_name*.

    Raises:
        ValueError: If *model_name* is not registered.
    """
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[model_name]
    return getattr(importlib.import_module(module_path), class_name)
