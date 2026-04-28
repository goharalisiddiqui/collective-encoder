import importlib

# Maps the canonical identifier (must match _IDENTIFIER on the class) to its
# (module_path, class_name).  Adding a new reader = one line here.
_REGISTRY: dict = {
    "XTC":           ("collective_encoder.datareaders.xtc",           "XTCReader"),
    "XTC_CHUNKS":    ("collective_encoder.datareaders.xtc_chunks",    "XTCChunksReader"),
    "XTC_CHUNKS_CG": ("collective_encoder.datareaders.xtc_chunks_cg", "XTCChunksCGReader"),
    "PLUMED_OUTPUT": ("collective_encoder.datareaders.plumed_output",  "PlumedOutputReader"),
}


def get_datareader(datareader_type: str):
    """Return the datareader class for *datareader_type*.

    Raises:
        ValueError: If *datareader_type* is not registered.
    """
    if datareader_type not in _REGISTRY:
        raise ValueError(
            f"Unknown datareader type: '{datareader_type}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[datareader_type]
    return getattr(importlib.import_module(module_path), class_name)
