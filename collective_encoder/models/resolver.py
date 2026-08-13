import importlib
from typing import Type

from collective_encoder.models.base import CEModelBase

_MODEL_REGISTRY = {
    "AE": ("collective_encoder.models.neural_nets.ae_net", "AE"),
    "sAE": ("collective_encoder.models.neural_nets.ae_net", "sAE"),
    "VAE": ("collective_encoder.models.neural_nets.vae_net", "VAE"),
    "sVAE": ("collective_encoder.models.neural_nets.vae_net", "sVAE"),
    "DVAE": ("collective_encoder.models.neural_nets.dvae_net", "DVAE"),
    "sDVAE": ("collective_encoder.models.neural_nets.dvae_net", "sDVAE"),
    "EDVAE": ("collective_encoder.models.neural_nets.edvae_net", "EDVAE"),
    "BGE": ("collective_encoder.models.neural_nets.bge", "BondGraphEncoderDecoder"),
    "BGE_V2": ("collective_encoder.models.neural_nets.bge_v2", "BondGraphEncoderDecoderV2"),
    "PCA": ("collective_encoder.models.pca_model", "PCAModel"),
    "PCAEncoder": ("collective_encoder.models.pca_model", "PCAModel"),
    "ICA": ("collective_encoder.models.ica_model", "ICAModel"),
    "ICAEncoder": ("collective_encoder.models.ica_model", "ICAModel"),
}


def get_model(model_name: str) -> Type[CEModelBase]:
    """Retrieve model class by name from the registry."""
    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )
    module_path, class_name = _MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# Backward-compatible alias for existing code referencing get_net
get_net = get_model
