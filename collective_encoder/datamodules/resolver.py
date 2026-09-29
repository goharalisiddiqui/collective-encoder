import importlib
from typing import List

_REGISTRY: dict = {
    "XTC":         ("collective_encoder.datamodules.coordinates", "CoordinatesDataModule"),
    "COORDINATES": ("collective_encoder.datamodules.coordinates", "CoordinatesDataModule"),
    "COLVAR":      ("collective_encoder.datamodules.colvars",     "ColvarsDataModule"),
}
    

def get_datamodule(datamodule_name: str):
    """
    Return the datamodule class mapped to a given string identifier.

    Parameters
    ----------
    datamodule_name : str
        The registered string identifier for the target datamodule
        (e.g., ``"COORDINATES"``, ``"COLVAR"``).

    Returns
    -------
    type
        The corresponding PyTorch Lightning DataModule class.

    Raises
    ------
    ValueError
        If `datamodule_name` is not found in the registry.
    """
    if datamodule_name not in _REGISTRY:
        raise ValueError(
            f"Unknown datamodule name: '{datamodule_name}'. "
            f"Available: {sorted(set(_REGISTRY))}"
        )
    module_path, class_name = _REGISTRY[datamodule_name]
    datamodule = getattr(importlib.import_module(module_path), class_name)
    return datamodule


def get_compatible_datareaders(dataloader_name: str) -> List[str]:
    """
    Return the list of compatible datareader identifiers for *dataloader_name*.

    Parameters
    ----------
    dataloader_name : str
        The registered string identifier for the datamodule.

    Returns
    -------
    list of str
        The list of compatible datareader identifiers.
    """
    dataloader_class = get_datamodule(dataloader_name)
    return dataloader_class.get_compatible_datareaders()


def get_compatible_datasets(dataloader_name: str) -> List[str]:
    """
    Return the list of compatible dataset identifiers for *dataloader_name*.

    Parameters
    ----------
    dataloader_name : str
        The registered string identifier for the datamodule.

    Returns
    -------
    list of str
        The list of compatible dataset identifiers.
    """
    dataloader_class = get_datamodule(dataloader_name)
    return dataloader_class.get_compatible_datasets()
