import importlib

_REGISTRY: dict = {
    "LDplotter": ("collective_encoder.testplotters.latent_space_plotter", "LDplotter"),
}

def get_testplotter(model_name: str):
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
