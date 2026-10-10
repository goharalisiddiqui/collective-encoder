import importlib

# Maps the canonical identifier (must match _IDENTIFIER on the class) to its
# (module_path, class_name).  Adding a new reader = one line here.
_REGISTRY: dict = {
    "COMPRESSED_TRAJECTORY": ("collective_encoder.datareaders.compressed_trajectory", "CompressedTrajectoryReader"),
    "XTC":                   ("collective_encoder.datareaders.compressed_trajectory", "CompressedTrajectoryReader"), # For backward compatibility
    "PLUMED_OUTPUT":         ("collective_encoder.datareaders.plumed_output",         "PlumedOutputReader"),
    "MD17":                  ("collective_encoder.datareaders.datasets.md17",         "MD17Reader"),
}


def get_datareader(datareader_type: str):
    """
    Return the datareader class mapped to a given string identifier.

    Parameters
    ----------
    datareader_type : str
        The registered string identifier for the target datareader.

    Returns
    -------
    type
        The corresponding datareader class.

    Raises
    ------
    ValueError
        If `datareader_type` is not found in the registry.
    """
    if datareader_type not in _REGISTRY:
        raise ValueError(
            f"Unknown datareader type: '{datareader_type}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[datareader_type]
    return getattr(importlib.import_module(module_path), class_name)
