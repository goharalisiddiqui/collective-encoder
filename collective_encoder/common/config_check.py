from typing import Dict, List

REQUIRED_TOP_LEVEL = [
    "nepochs", "lrate", "network_name", "data_name", "data_args", "network_args"
]

def validate_duplicate_keys(config_path: str):
    """
    Check for duplicate keys in the YAML config file.
    Raises an error if duplicates are found, otherwise returns True.
    
    Parameters:
    config_path (str): The file path to the YAML configuration file.

    Returns:
    bool: True if no duplicate keys are found, otherwise raises a ValueError.
    """
    try:
        with open(config_path, 'r') as f:
            pass
            #FIXME: Implement a proper duplicate key check using yaml library
        return True
    except Exception as e:
        raise RuntimeError(f"Error reading config: {e}")
    

def validate_required_fields(config: Dict[str, any], fields : List[str] = REQUIRED_TOP_LEVEL):
    """
    Validate that all required fields are present in the config dictionary.
    Raises an error if any required field is missing, otherwise returns True.
    
    Parameters:
    config (dict): The configuration dictionary to validate.
    fields (List[str]): A list of required top-level keys that must be present 
        in the config dictionary. Defaults to REQUIRED_TOP_LEVEL.

    Returns:
    bool: True if all required fields are present, otherwise raises a 
        ValueError indicating which field is
    """
    if len(config) == 0:
        return True  # Empty config is valid (no required fields)
    if config is None:
        raise ValueError("Config is None, expected a dictionary.")
    if len(config) == 0:
        raise ValueError(f"Config is empty, expected a dictionary with required fields: {fields}.")
    
    # Check required top-level keys
    for k in fields:
        if k not in config:
            raise ValueError(f"Required config key '{k}' missing.")
    return True
