import logging
_log = logging.getLogger(__name__)

import os
import yaml
import argparse
import warnings

import numpy as np

import torch
import pytorch_lightning as pl

from gslibs.utils.common import recursive_update
from gslibs.utils.common import get_required_init_args

from collective_encoder.utils import check_dict_contains_keys
from collective_encoder.nets.resolver import get_net
from collective_encoder.datamodules.resolver import get_datamodule
from collective_encoder.common.config_check import (
    validate_duplicate_keys, 
    validate_required_fields 
)
from collective_encoder.dataanalysers.resolver import get_dataanalyser
from gslibs.utils.filesystem import create_rundir, output_to_file

warnings.filterwarnings("ignore", ".*does not have many workers.*")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'tester', 'defaults.yaml')
DEBUG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'tester', 'debug.yaml')
OVERRIDABLE_DMOD_ARGS = ['batch_size', 'val_batch_size', 
                         'num_workers', 'test_batch_size']
torch.set_default_dtype(torch.float64)


##################################
# Arguments
##################################
def parse_args():
    desc = "Surrogate model to predict dynamics of molecular systems as time series data"
    parser = argparse.ArgumentParser(description=desc)

    # Run Settings
    parser.add_argument('--config', required=True, type=str,
                        help='')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with small data and epochs')
    
    args = parser.parse_args()

    return args

def test(config_path: str, debug: bool = False):
    """Test a collective encoder model based on the provided configuration."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    validate_duplicate_keys(config_path)
    config = yaml.safe_load(open(DEFAULT_CONFIG_PATH, 'r'))
    recursive_update(config, yaml.safe_load(open(config_path, 'r')))
    if debug or config.get('debug', False):
        # Load debug config and override values
        recursive_update(config, yaml.safe_load(open(DEBUG_CONFIG_PATH, 'r')))
        torch.manual_seed(0)
        np.random.seed(0)
        print("Running in debug mode.")
    check_dict_contains_keys(config, required_keys=[
        'outfolder', 'nexp', 'overwrite', 'output_to_file', 'test_plotter_type',
        'network_train_path', 'network_type'
    ])
    config['outpath'] = config.get('outpath', 
                            os.path.join(config['network_train_path'], 'test_runs'))
    if not os.path.isdir(config['network_train_path']):
        raise FileNotFoundError(f"Network training directory not found at {config['network_train_path']}")

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
    logging_level = getattr(logging, config.get('verbose', 'INFO').upper(), logging.INFO)
    logging.basicConfig(filename=os.path.join(run_dir, "run.log"),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging_level)
    metargs = {
        'verbose': config.get('verbose', True),
        'root_logger_name': __name__,
        'run_dir': run_dir,
    }

    ##################################
    # Creating Dataset
    ##################################
    if 'load_datamodule' in config:
        check_dict_contains_keys(config, required_keys=['load_datamodule'])
        dmod_path = config['load_datamodule']
        _log.info("Loading datamodule from: " + dmod_path)
        dmod_ckpt = os.path.join(dmod_path, "datamodule.pth")
        
        # torch.serialization.add_safe_globals(torch.serialization.get_unsafe_globals_in_checkpoint(dmod_ckpt)) # !!! Very Unsafe, only do this if you trust the source of the checkpoint !!!
        dm = torch.load(dmod_ckpt, weights_only=False)
        dm_args = dm.get_args()
        dm_override_args = config.get('datamodule_args', {})
        for key, value in dm_override_args.items():
            if key not in OVERRIDABLE_DMOD_ARGS:
                raise ValueError(f"Cannot override datamodule argument '{key}'. "
                                 f"Allowed keys: {OVERRIDABLE_DMOD_ARGS}")
            _log.info(f"Overriding datamodule argument '{key}' with "
                      f"value: {value}, previous value: {getattr(dm, key, 'N/A')}")
            dm_args[key] = value
            setattr(dm, key, value)
    else:
        check_dict_contains_keys(config, required_keys=[
            'datamodule_type', 'datamodule_args'
        ])
        dm_type = config['datamodule_type']
        dm_args = config['datamodule_args']
        dm_cls = get_datamodule(dm_type)
        validate_required_fields(dm_args, 
                                get_required_init_args(dm_cls))
        
        dataset_type = dm_args.get('dataset_type', None)
        if dataset_type not in nn_cls._COMPATIBLE_DATASETS:
            raise ValueError(f"Network '{nn_type}' is not compatible with dataset" \
                            f" '{dataset_type}'")
        dm = dm_cls(dm_args, **metargs)
        
    ##################################
    # Data analysis and visualization
    ##################################
    da = config.get('data_analyser_type', None)
    if da != None:
        analyser_cls = get_dataanalyser(da)    
        da_args = config.get('data_analyser_args', {})
        da_args['datamodule_args'] = dm_args
        da_args['output_dir'] = run_dir + "/data_analysis"
        analyser = analyser_cls(da_args,**metargs)
        analyser.write_data(dm.get_train_dataset(), label="train")
        analyser.write_data(dm.get_val_dataset(), label="val")

    ##################################
    # Setting up the NN
    ##################################
    nn_path = config['network_train_path']
    ckpt_path = os.path.join(nn_path, "checkpoints")
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
    
    nn_type = config['network_type']
    nn_cls = get_net(nn_type)
    
    # torch.serialization.add_safe_globals(torch.serialization.get_unsafe_globals_in_checkpoint(dmod_ckpt)) # !!! Very Unsafe, only do this if you trust the source of the checkpoint !!!
    model = nn_cls.load_from_checkpoint(nn_ckpt, 
                                        **metargs)

    ##################################
    # Training the NN
    ##################################
    trainargs = {"log_every_n_steps" : 1,
                 "default_root_dir" : run_dir}
    
    trainer = pl.Trainer(**trainargs)


    ##################################
    if config.get('test_plotter_type', False):
        model.add_test_plotter(config['test_plotter_type'], config.get('test_plotter_args', None))  
    trainer.test(model, datamodule=dm)


def main():
    """Main entry point for the collective encoder testing."""
    args = parse_args()
    test(args.config, args.debug)

if __name__ == "__main__":
    main()





