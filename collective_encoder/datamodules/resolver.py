from typing import Dict, List
from gslibs.utils.common import get_init_args
from collective_encoder.datareaders.resolver import get_datareader


def get_datamodule(datamodule_name: str, dataset_args: Dict[str, any]):
    """
    Get the dataloader class and required arguments for a given dataloader name.
    This function maps a dataloader name to its corresponding dataloader class
    and retrieves the required arguments for initializing that dataloader.

    Parameters:
    dataloader_name (str): The name of the dataloader to retrieve.
    dataset_args (Dict[str, any]): A dictionary of arguments to be passed to the dataloader.
                                  This dictionary will be updated with the required arguments for
                                  the specified dataloader.

    Returns:
    main_dl (type): The dataloader class corresponding to the specified dataloader name.
    dataset_args (Dict[str, any]): The updated dictionary of arguments including the required
                                  arguments for the dataloader.
    """

    if datamodule_name == "XTC" or datamodule_name == "COORDINATES":
        from collective_encoder.datamodules.coordinates import CoordinatesDataModule as datamodule
        dataset_args.update({
            "datareader_type": "XTC",
        })
    # elif datamodule_name == "COLVAR":
    #     from collective_encoder.datamodules.colvar import ColvarDataloader as main_dl
    #     # COLVAR dataloaders don't use datareaders in the same way
    #     pass
    # elif datamodule_name == "MD17":
    #     from collective_encoder.datamodules.md17 import MD17Dataloader as main_dl
    #     # MD17 dataloaders handle their own data loading
    #     pass
    else:
        raise ValueError(f"Unknown datamodule name: {datamodule_name}")

    # Only process datareader args if the dataloader uses a datareader
    if "datareader_type" in dataset_args:
        datareader_args_names = get_init_args(
            get_datareader(dataset_args["datareader_type"]))
        datareader_args = dataset_args.get("datareader_args", {})
        for arg in datareader_args_names:
            if arg in dataset_args:
                datareader_args[arg] = dataset_args.pop(arg)
        dataset_args["datareader_args"] = datareader_args

    return datamodule, dataset_args


def get_compatible_datareaders(dataloader_name: str) -> List[str]:
    """
    Get a list of compatible datareader types for a given dataloader.

    Parameters:
    dataloader_name (str): The name of the dataloader.

    Returns:
    List[str]: A list of compatible datareader type names.
    """
    dataloader_class, _ = get_dataloader(dataloader_name, {})
    return dataloader_class.get_compatible_datareaders()


def get_compatible_datasets(dataloader_name: str) -> List[str]:
    """
    Get a list of compatible dataset types for a given dataloader.

    Parameters:
    dataloader_name (str): The name of the dataloader.

    Returns:
    List[str]: A list of compatible dataset type names.
    """
    dataloader_class, _ = get_dataloader(dataloader_name, {})
    return dataloader_class.get_compatible_datasets()
        