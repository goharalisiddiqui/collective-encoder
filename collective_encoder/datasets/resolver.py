from typing import Dict
from collective_encoder.common.registry import get_module

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset_cls_dl(dataset_type: str, dataset_args: Dict, datareader):
    """
    Return ``(dataset_class, dataset_args, dataloader_class)`` for a given dataset type.

    Parameters
    ----------
    dataset_type : str
        String identifier for the dataset (e.g. ``"GRAPH"``, ``"SOAP"``).
    dataset_args : dict
        Original configuration dictionary for the dataset.
    datareader : BaseDataReader
        The instantiated datareader object, used to provide necessary dynamic
        properties like bond indices or atom selections.

    Returns
    -------
    tuple
        A tuple containing:
        - `dataset_class`: The dataset class to instantiate.
        - `dataset_args`: The configuration dict, possibly augmented with datareader properties.
        - `dataloader_class`: The appropriate PyTorch DataLoader class for this dataset type.

    Raises
    ------
    ValueError
        If `dataset_type` is not found in the registry.
    """
    dataset_cls = get_module("datasets", dataset_type)
    
    return dataset_cls._prepare_dataset(dataset_args, datareader)
