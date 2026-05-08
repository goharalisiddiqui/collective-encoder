import importlib
from typing import List

_REGISTRY: dict = {
    "XTC":         ("collective_encoder.datamodules.coordinates", "CoordinatesDataModule"),
    "COORDINATES": ("collective_encoder.datamodules.coordinates", "CoordinatesDataModule"),
    "COLVAR":      ("collective_encoder.datamodules.colvars",     "ColvarsDataModule"),
}
    

def get_datamodule(datamodule_name: str):
    """Return the datamodule class for *datamodule_name*.

    Args:
        datamodule_name: Identifier string (e.g. ``"COORDINATES"``, ``"COLVAR"``).

    Raises:
        ValueError: If *datamodule_name* is not registered.
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
    """Return the list of compatible datareader identifiers for *dataloader_name*."""
    dataloader_class = get_datamodule(dataloader_name)
    return dataloader_class.get_compatible_datareaders()


def get_compatible_datasets(dataloader_name: str) -> List[str]:
    """Return the list of compatible dataset identifiers for *dataloader_name*."""
    dataloader_class = get_datamodule(dataloader_name)
    return dataloader_class.get_compatible_datasets()
