import importlib

_PROCESSOR_REGISTRY = {
    "Chunk": ("collective_encoder.datareaders.processors.chunk", "ChunkProcessor"),
    "CoarseGrain": ("collective_encoder.datareaders.processors.coarse_grain", "CoarseGrainProcessor"),
    "Filter": ("collective_encoder.datareaders.processors.filter", "FilterProcessor"),
}

def get_processor(processor_type: str):
    """
    Return the processor class mapped to a given string identifier.

    Parameters
    ----------
    processor_type : str
        The registered string identifier for the target processor.

    Returns
    -------
    type
        The corresponding processor class.

    Raises
    ------
    ValueError
        If `processor_type` is not found in the registry.
    """
    if processor_type not in _PROCESSOR_REGISTRY:
        raise ValueError(
            f"Unknown processor type: '{processor_type}'. "
            f"Available: {sorted(_PROCESSOR_REGISTRY)}"
        )
    module_path, class_name = _PROCESSOR_REGISTRY[processor_type]
    return getattr(importlib.import_module(module_path), class_name)
