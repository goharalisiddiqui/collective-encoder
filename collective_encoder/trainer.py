import logging
import os
import wandb
import yaml
import shutil

import argparse
import warnings

_log = logging.getLogger(__name__)

import numpy as np

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from gslibs.utils.common import recursive_update
from gslibs.utils.common import get_required_init_args

from collective_encoder.nets.resolver import get_net
from collective_encoder.datamodules.resolver import get_datamodule
from collective_encoder.common.config_check import (
    validate_duplicate_keys, 
    validate_required_fields 
)
from collective_encoder.dataanalysers.resolver import get_dataanalyser
from gslibs.utils.filesystem import create_rundir, output_to_file

warnings.filterwarnings("ignore", ".*does not have many workers.*")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'trainer', 'defaults.yaml')
DEBUG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'trainer', 'debug.yaml')
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

def train(config_path: str, debug: bool = False):
    """Train a collective encoder model based on the provided configuration."""
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
    validate_required_fields(config)

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
    logging.basicConfig(filename=os.path.join(run_dir, "run.log"),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging.INFO)
    metargs = {
        'verbose': config.get('verbose', True),
        'root_logger_name': __name__,
        'run_dir': run_dir,
    }
    
    ##################################
    # Config validation
    ##################################
    _KNOWN_CONFIG_KEYS = {
        'debug', 'outpath', 'outfolder', 'overwrite', 'nexp', 'output_to_file',
        'save_checkpoint', 'save_serial_model', 'nepochs', 'lrate', 'weight_decay',
        'nogpu', 'export_latent', 'wandb', 'wandb_project', 'wandb_entity',
        'scheduler', 'scheduler_args', 'normIn', 'network_type', 'network_args',
        'datamodule_type', 'datamodule_args', 'data_analyser', 'data_args',
        'load_model', 'output_traj', 'save_metatomic', 'early_stopping',
        'early_stopping_args', 'verbose', 'metatomic_metadata', 'test_plotter_type', 
        'test_plotter_args',
    }
    for key in config:
        if key not in _KNOWN_CONFIG_KEYS:
            _log.warning("Unknown config key '%s' — will be ignored", key)

    ##################################
    # Creating Dataset
    ##################################
    if 'load_datamodule' in config:
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
    nn_type = config['network_type']
    nn_cls = get_net(nn_type)
    req_fields = get_required_init_args(nn_cls)
    validate_required_fields(config['network_args'], req_fields)
    nn_args = {
        'lrate': config['lrate'],
        'weight_decay': config['weight_decay'],
        'normIn': config['normIn'],
        'scheduler': config['scheduler'],
        'scheduler_args': config.get('scheduler_args', {}),
    }
    nn_args.update(config.get('network_args', {}))

    load_model = config.get('load_model', None)
    if load_model != None:
        checkpoint_file = load_model
        print(f"Loading model from {checkpoint_file}")
        model = nn_cls.load_from_checkpoint(checkpoint_file, 
                                            datamodule=dm,
                                            args=nn_args, **metargs)
    else:
        model = nn_cls(datamodule=dm, 
                       args=nn_args, 
                       **metargs)

    ##################################
    # Training the NN
    ##################################
    trainargs = {"max_epochs" : config['nepochs'],
                 "log_every_n_steps" : 1,
                 "default_root_dir" : run_dir}
    if not config.get('nogpu', False):
        trainargs["accelerator"] = 'auto'
        trainargs["devices"] = 'auto'
    if config.get('wandb', False):
        wandb_logger = WandbLogger(project=config['wandb_project'],
                                 entity=config['wandb_entity'],
                                 save_dir=run_dir,
                                 name=run_dir.strip(".").strip("/").replace("/", "_"),
                                 log_model=False,)
        # wandb_logger.watch(model, log_graph=True)
        trainargs["logger"] = wandb_logger

    callbacks = []
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    # Early stopping
    if config.get('early_stopping', True):
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_args', {}).get('patience', 100),
            min_delta=config.get('early_stopping_args', {}).get('min_delta', 1e-8),
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop_callback)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=run_dir + '/checkpoints',
        filename=config['network_type'] + '-{epoch:02d}-{val_loss:.6f}',
        save_top_k=1,
        mode='min',
    )
    callbacks.append(checkpoint_callback)

    trainargs["callbacks"] = callbacks
    # trainargs["num_sanity_val_steps"] = 0

    # trainargs["gradient_clip_val"] = 0.5
    # trainargs["gradient_clip_algorithm"] = "norm"

    trainer = pl.Trainer(**trainargs)

    if config['nepochs'] > 0:
        _log.info("Starting training for %d epochs...", config['nepochs'])
        trainer.fit(model, datamodule=dm)
        _log.info("Training completed.")
        if config.get('wandb', False):
            wandb.finish()

    if config['nepochs'] == 0 and load_model == None:
        _log.warning("Both nepochs and load_model are not set. Nothing to do.")

    # Save the best model checkpoint as best.ckpt
    best_checkpoint_path = checkpoint_callback.best_model_path
    if best_checkpoint_path != "":
        shutil.copy(best_checkpoint_path, os.path.dirname(best_checkpoint_path) + "/best.ckpt")
    _log.info(f"Best model saved at: {best_checkpoint_path}")

    ##################################
    # Analysing a loaded model
    ##################################
    if config.get('output_traj', False):
        if not config['datamodule_type'] in ["COORDINATES", "XTC"]:
            raise ValueError(f"Unsupported data type: {config['datamodule_type']} for trajectory output")
        else:
            pred = model(dm.get_full_batch()[0])[0].detach().cpu().numpy()
            dm.output_trajectory(os.path.join(run_dir, "recon_trajectory.pdb"), pred)

    ##################################
    if config.get('test_plotter_type', False):
        model.add_test_plotter(config['test_plotter_type'], config.get('test_plotter_args', None))
    _log.info("Starting testing...")
    trainer.test(model, datamodule=dm)
    _log.info("Testing completed.")

    #####################################
    # Save metatomic model
    #####################################
    if config.get('save_metatomic', False):
        try:
            from collective_encoder.mtomic.wrapper import MetatomicCV
            from metatomic.torch import (
                AtomisticModel,
                ModelCapabilities,
                ModelMetadata,
                ModelOutput,
                System,
                ModelEvaluationOptions,
            )
            from metatensor.torch import Labels
        except ImportError:
            raise ImportError("metatomic is not installed. Please install it with `pip install metatomic`")

        dataprocessor = dm.get_dataset().get_metatomic_dataprocessor()
        # FIXME: Remove wandb hooks to avoid issues during serialization
        metamodel = model.get_metatomic_model()
        metatomic_model = MetatomicCV(dataprocessor, metamodel)
        metadata = ModelMetadata(
            name=config.get('metatomic_metadata', {}).get('name', 'unknown'),
            description=config.get('metatomic_metadata', {}).get('description', 'unknown'),
            authors=config.get('metatomic_metadata', {}).get('authors', []),
            references=config.get('metatomic_metadata', {}).get('references', {}),
        )
        capabilities = ModelCapabilities(
            outputs=metamodel.get_metatomic_outputs(),
            atomic_types=dataprocessor.get_atomic_types(),
            interaction_range=dataprocessor.get_interaction_range(),
            length_unit=dataprocessor.get_length_unit(),
            supported_devices=["cpu", "cuda"],
            dtype="float64",
        )
        metatomic_module = AtomisticModel(
            module=metatomic_model.eval(),
            metadata=metadata,
            capabilities=capabilities,
        )
        ##################################
        # Sanity check of the model
        ##################################
        fake_systems = dm.get_dataset().get_fake_systems()
        fake_options = ModelEvaluationOptions(
            length_unit=dataprocessor.get_length_unit(),
            outputs=metamodel.get_metatomic_outputs(),
            selected_atoms=None,
            # selected_atoms=Labels(
            #     ['system', 'atom'], torch.tensor([0, 512]).reshape(-1, 2)
            # ),
        )
            
        # Run inference
        try:
            print("Running sanity check of the metatomic model...")
            with torch.no_grad():
                output = metatomic_module(fake_systems, fake_options, False)
        except Exception as e:
            raise RuntimeError("metatomic model failed the sanity check: "+str(e))

        print("metatomic model passed the sanity check.\nSerializing the model...")
        metatomic_model_file = os.path.join(run_dir, "metatomic_model.pt")
        metatomic_extension_directory = os.path.join(run_dir, "metatomic_extensions")
        metatomic_module.save(metatomic_model_file,
                            collect_extensions=metatomic_extension_directory)

        print(f"@@ metatomic model saved as: {metatomic_model_file}")

def main():
    """Main entry point for the collective encoder training."""
    args = parse_args()
    train(args.config, args.debug)

if __name__ == "__main__":
    main()





