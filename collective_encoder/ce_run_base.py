import logging
_log = logging.getLogger(__name__)
import os
import yaml

import argparse
import warnings

import numpy as np

import torch

from gslibs.utils.common import recursive_update

from collective_encoder.common.config_check import (
    validate_duplicate_keys, 
)
from gslibs.utils.filesystem import create_rundir, output_to_file

from collective_encoder.utils import check_dict_contains_keys
from collective_encoder.datamodules.resolver import get_datamodule
from collective_encoder.nets.resolver import get_net

warnings.filterwarnings("ignore", ".*does not have many workers.*")
torch.set_default_dtype(torch.float64)
_COMMON_REQUIRED_KEYS = ['outpath', 'outfolder', 'nexp', 'overwrite', 'output_to_file']
_OVERRIDABLE_DMOD_ARGS = ['batch_size', 'val_batch_size', 
                         'num_workers', 'test_batch_size']

##################################
# Arguments
##################################
def parse_args():
    desc = "Prepare datamodule for training a collective encoder model based on the provided configuration."
    parser = argparse.ArgumentParser(description=desc)

    # Run Settings
    parser.add_argument('--config', required=True, type=str,
                        help='')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with small data and epochs')
    
    args = parser.parse_args()

    return args

def get_required_keys(settings: dict) -> list:
    """Get the required keys for the module."""
    req_keys = settings.get('required_keys', [])
    req_keys = list(set(req_keys + _COMMON_REQUIRED_KEYS))
    return req_keys

def get_default_config_path(settings: dict) -> str:
    """Get the default config path for the module."""
    return os.path.join(os.path.dirname(__file__), 
                                       'configs', 
                                       settings.get('module'), 
                                       'defaults.yaml')

def get_debug_config_path(settings: dict) -> str:
    """Get the debug config path for the module."""
    return os.path.join(os.path.dirname(__file__), 
                                       'configs', 
                                       settings.get('module'), 
                                       'debug.yaml')
def prepare(settings: dict):
    """Prepare the module."""
    if 'module' not in settings:
        raise ValueError("Module name must be specified in settings.")
    
    required_keys = get_required_keys(settings)
    default_config_path = get_default_config_path(settings)
    debug_config_path = get_debug_config_path(settings)
    
    args = parse_args()
    config_path = args.config
    debug = args.debug
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    validate_duplicate_keys(config_path)
    config = yaml.safe_load(open(default_config_path, 'r'))
    recursive_update(config, yaml.safe_load(open(config_path, 'r')))
    if debug or config.get('debug', False):
        # Load debug config and override values
        recursive_update(config, yaml.safe_load(open(debug_config_path, 'r')))
        torch.manual_seed(0)
        np.random.seed(0)
        print("Running in debug mode.")
    check_dict_contains_keys(config, required_keys=required_keys)
    
    
    
    
    # ##################################
    # # Config validation
    # ##################################
    # _KNOWN_CONFIG_KEYS = {
    #     'debug', 'outpath', 'outfolder', 'overwrite', 'nexp', 'output_to_file',
    #     'save_checkpoint', 'save_serial_model', 'nepochs', 'lrate', 'weight_decay',
    #     'nogpu', 'export_latent', 'wandb', 'wandb_project', 'wandb_entity',
    #     'scheduler', 'scheduler_args', 'normIn', 'network_type', 'network_args',
    #     'datamodule_type', 'datamodule_args', 'data_analyser', 'data_args',
    #     'load_model', 'output_traj', 'save_metatomic', 'early_stopping',
    #     'early_stopping_args', 'verbose', 'metatomic_metadata', 'test_plotter_type', 
    #     'test_plotter_args',
    # }
    # for key in config:
    #     if key not in _KNOWN_CONFIG_KEYS:
    #         _log.warning("Unknown config key '%s' — will be ignored", key)

    ##################################
    # Output directory
    ##################################
    run_dir = create_rundir(config['outpath'], 
                        config['outfolder'], 
                        config['nexp'], 
                        overwrite=config['overwrite'])

    ##################################
    # Output to file
    ##################################
    if config['output_to_file']:
        output_to_file(run_dir, filename="out.txt")
    
    ##################################
    # Meta args used in all modules
    ##################################
    logging_level = config.get('verbose', 'INFO')
    if logging_level is True:
        logging_level = 'INFO'
    if logging_level is False:
        logging_level = 'WARNING'
    if not hasattr(logging, logging_level.upper()):
        raise ValueError(f"Invalid logging level: {logging_level}. "
                         f"Valid levels: {logging._nameToLevel.keys()}")
    logging_level = getattr(logging, logging_level.upper(), logging.INFO)
    logging.basicConfig(filename=os.path.join(run_dir, "run.log"),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging_level)
    metargs = {
        'verbose': config.get('verbose', True),
        'root_logger_name': settings['module'],
        'run_dir': run_dir,
    }
    return config, metargs

def load_datamodule(config, metargs):
    if 'load_datamodule' in config:
        dmod_path = config['load_datamodule']
        _log.info("Loading datamodule from: " + dmod_path)
        dmod_ckpt = os.path.join(dmod_path, "datamodule.pth")
        
        # torch.serialization.add_safe_globals(torch.serialization.get_unsafe_globals_in_checkpoint(dmod_ckpt)) # !!! Very Unsafe, only do this if you trust the source of the checkpoint !!!
        dm = torch.load(dmod_ckpt, weights_only=False)
        dm_args = dm.get_args()
        dm_override_args = config.get('datamodule_args', {})
        for key, value in dm_override_args.items():
            if key not in _OVERRIDABLE_DMOD_ARGS:
                raise ValueError(f"Cannot override datamodule argument '{key}'. "
                                    f"Allowed keys: {_OVERRIDABLE_DMOD_ARGS}")
            _log.info(f"Overriding datamodule argument '{key}' with "
                        f"value: {value}, previous value: {getattr(dm, key, 'N/A')}")
            dm_args[key] = value
            setattr(dm, key, value)
    else:
        dm_type = config['datamodule_type']
        dm_args = config['datamodule_args']
        dm_cls = get_datamodule(dm_type)
        dm = dm_cls(args=dm_args, **metargs)
    
    return dm

def load_model(config, metargs, dm):
    nn_type = config['network_type']
    nn_cls = get_net(nn_type)
    nn_args = {
        'lrate': config['lrate'],
        'weight_decay': config['weight_decay'],
        'normIn': config['normIn'],
        'scheduler': config['scheduler'],
        'scheduler_args': config.get('scheduler_args', {}),
    }
    nn_args.update(config.get('network_args', {}))

    if 'load_network' in config:
        if len(config.get('network_args', {})) > 0:
            _log.warning("network_args will be ignored when loading a model.")
            config['network_args'] = {}

        ckpt_path = os.path.join(config['load_network'], "checkpoints")
        potential_ckpts = [a for a in os.listdir(ckpt_path) if a.endswith(".ckpt")]
        if len(potential_ckpts) == 0:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")
        for name in ['best', 'saved', 'last']:
            if name + ".ckpt" in potential_ckpts:
                nn_ckpt = os.path.join(ckpt_path, name + ".ckpt")
                break
        else:
            _log.warning("No 'best', 'saved', or 'last' checkpoint found. "
                            "Using the first available checkpoint.") 
            nn_ckpt = os.path.join(ckpt_path, potential_ckpts[0])
        _log.info("Loading network from: " + nn_ckpt)
        
        # torch.serialization.add_safe_globals(torch.serialization.get_unsafe_globals_in_checkpoint(dmod_ckpt)) # !!! Very Unsafe, only do this if you trust the source of the checkpoint !!!
        nn_args_saved = torch.load(nn_ckpt, weights_only=False, 
                    map_location='cpu')['hyper_parameters']['args']
        forbidden_overrides = ['normIn'] # Some arguments cannot be overridden when loading a model
        nn_args_saved.update({k: v for k, v in nn_args.items() if k not in forbidden_overrides})
        nn_args = nn_cls.extract_args_from_datamodule(dm, nn_args_saved)
        model = nn_cls.load_from_checkpoint(nn_ckpt,
                                            args=nn_args,
                                            **metargs)
    
    else:
        nn_args = nn_cls.extract_args_from_datamodule(dm, nn_args)
        model = nn_cls(args=nn_args, 
                       **metargs)
    
    return model




    
