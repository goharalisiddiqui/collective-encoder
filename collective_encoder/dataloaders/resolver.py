from typing import Dict
from collective_encoder.utils import get_init_args
from collective_encoder.datareaders.resolver import get_datareader


def get_dataloader(dataloader_name: str, dataset_args: Dict[str, any]):
    '''
    Get the dataloader class and required arguments for a given dataloader name.
    This function maps a dataloader name to its corresponding dataloader class 
    and retrieves the required arguments for initializing that dataloader.
    
    Parameters:
    dataloader_name (str): The name of the dataloader to retrieve.
    dataset_args (Dict[str, any]): A dictionary of arguments to be passed to the dataloader. This dictionary will be updated with the required arguments for 
    the specified dataloader.

    Returns:
    main_dl (type): The dataloader class corresponding to the specified dataloader name.
    dataset_args (Dict[str, any]): The updated dictionary of arguments including the required arguments for the
    '''
    if dataloader_name == "XTC":
        from collective_encoder.dataloaders.default import DefaultDatamodule as main_dl
        dataset_args.update({
            "datareader_type": "XTC",
        })
    # elif dataloader_name == "COLVAR":
    #     from collective_encoder.dataloaders.colvar_dataloader import ColvarDataset as main_dl
    else:
        raise ValueError(f"Unknown dataloader name: {dataloader_name}")
    
    datareader_args_names = get_init_args(
        get_datareader(dataset_args["datareader_type"]))
    datareader_args = dataset_args.get("datareader_args", {})
    for arg in datareader_args_names:
        if arg in dataset_args:
            datareader_args[arg] = dataset_args.pop(arg)
    dataset_args["datareader_args"] = datareader_args
    return main_dl, dataset_args
        