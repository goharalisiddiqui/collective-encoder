from typing import Any, Dict, List

import yaml

REQUIRED_TOP_LEVEL = [
    "nepochs", "lrate", "network_type", "network_args", "datamodule_type", "datamodule_args"
]


class _DuplicateKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node):
    loader.flatten_mapping(node)
    pairs = loader.construct_pairs(node)
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        dupes = {k for k in keys if keys.count(k) > 1}
        raise ValueError(f"Duplicate keys in config: {dupes}")
    return dict(pairs)


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def validate_duplicate_keys(config_path: str) -> bool:
    """
    Check for duplicate keys in the YAML config file.
    Raises ValueError if duplicates are found, otherwise returns True.
    """
    try:
        with open(config_path, 'r') as f:
            yaml.load(f, Loader=_DuplicateKeyLoader)
        return True
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error reading config: {e}")


def validate_required_fields(config: Dict[str, Any], fields: List[str] = REQUIRED_TOP_LEVEL) -> bool:
    """
    Validate that all required fields are present in the config dictionary.
    Raises ValueError if any required field is missing, otherwise returns True.
    """
    if len(fields) == 0:
        return True
    if config is None:
        raise ValueError("Config is None, expected a dictionary.")
    if len(config) == 0:
        raise ValueError(f"Config is empty, expected a dictionary with required fields: {fields}.")

    for k in fields:
        if k not in config:
            raise ValueError(f"Required config key '{k}' missing.")
    return True
