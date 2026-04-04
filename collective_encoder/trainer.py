import os
import yaml
import shutil

import argparse
import warnings

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

warnings.filterwarnings("ignore", ".*does not have many workers.*")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'defaults.yaml')
DEBUG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'debug.yaml')
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

def train(config: str, debug: bool = False):
    """Train a collective encoder model based on the provided configuration."""
    args = parse_args()
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}")
    validate_duplicate_keys(args.config)
    config = yaml.safe_load(open(DEFAULT_CONFIG_PATH, 'r'))
    recursive_update(config, yaml.safe_load(open(args.config, 'r')))
    if args.debug or config.get('debug', False):
        # Load debug config and override values
        recursive_update(config, yaml.safe_load(open(DEBUG_CONFIG_PATH, 'r')))
        torch.manual_seed(0)
        np.random.seed(0)
        print("Running in debug mode.")
    validate_required_fields(config)

    ##################################
    # Importing Lightning Modules
    ##################################
    nntype = config['network_name']
    datatype = config['data_name']
    
    main_nn = get_net(nntype)
    req_fields = get_required_init_args(main_nn)
    req_fields.remove('datamodule')
    validate_required_fields(config['network_args'], req_fields)

    data_args = config['data_args']
    main_dl, data_args = get_datamodule(datatype, data_args)
    validate_required_fields(data_args, 
                             get_required_init_args(main_dl))
    
    dataset_type = data_args.get('dataset_type', None)
    if dataset_type not in main_nn._COMPATIBLE_DATASETS:
        raise ValueError(f"Network '{nntype}' is not compatible with dataset" \
                         f" '{dataset_type}'")

    ##################################
    # Output directory
    ##################################
    nexp = int(config['nexp'])
    run_stem = config['outpath'] + "/" + config['outfolder'] + "_"
    run_dir = run_stem + str(nexp)

    if not config['overwrite']:
        while True:
            run_dir = run_stem + str(nexp)
            if not os.path.isdir(run_dir):
                os.makedirs(run_dir)
                break
            nexp = nexp + 1
    else:
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir)

    if len(os.listdir(run_dir)) != 0:
        shutil.rmtree(run_dir, ignore_errors=True)
        os.mkdir(run_dir)
    output_stem = run_dir + "/"

    ##################################
    # Output to file
    ##################################
    if config['output_to_file']:
        import sys
        import subprocess
        print("Redirecting output to file "+output_stem+"out.txt")
        tee = subprocess.Popen(["tee", output_stem+"out.txt"], stdin=subprocess.PIPE)
        # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
        # of any child processes we spawn)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

    print("Using Pytorch", torch.__version__)

    ##################################
    # Creating Dataset
    ##################################

    colvardata = main_dl(**data_args)
    if config['data_name'] == 'MD17':
        colvardata.prepare_data()
        colvardata.setup(stage="fit")

    if config.get('data_analyser', False):
        if config['data_analyser'] == 'ala2':
            from collective_encoder.plotters.ala2 import Ala2DataAnalyser as DataAnalyser
        else:
            warnings.warn("Unknown data analyser type: "+config['data_analyser'])

        analyser = DataAnalyser(output_dir=run_dir+"/data_analysis", data_args=config['data_args'])
        analyser.write_data(colvardata.get_dataset())

    ##################################
    # Setting up the NN
    ##################################
    netargs = {
        'lrate': config['lrate'],
        'weight_decay': config['weight_decay'],
        'normIn': config['normIn'],
        'scheduler': config['scheduler'],
        'scheduler_args': config.get('scheduler_args', {}),
        'outname': output_stem,
        'datamodule': colvardata,
    }
    netargs.update(config.get('network_args', {}))

    load_model = config.get('load_model', None)
    if load_model != None:
        checkpoint_file = load_model
        print(f"Loading model from {checkpoint_file}")
        model = main_nn.load_from_checkpoint(checkpoint_file, **netargs)
    else:
        model = main_nn(**netargs)

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
        filename=config['network_name'] + '-{epoch:02d}-{val_loss:.6f}',
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
        trainer.fit(model, datamodule=colvardata)

    if config['nepochs'] == 0 and load_model == None:
        warnings.warn("Both nepochs and load_model are not set. Nothing to do.")

    # Save the best model checkpoint as best.ckpt
    best_checkpoint_path = checkpoint_callback.best_model_path
    if best_checkpoint_path != "":
        shutil.copy(best_checkpoint_path, os.path.dirname(best_checkpoint_path) + "/best.ckpt")
    print(f"@@ Best model saved at: {best_checkpoint_path}")

    ##################################
    # Analysing a loaded model
    ##################################
    # model.print_fve(colvardata)
    if config.get('output_traj', False):
        if not config['data_name'] in ["XTC"]:
            raise ValueError(f"Unsupported data type: {config['data_name']}")
        else:
            pred = model(colvardata.get_full_batch()[0])[0].detach().cpu().numpy()
            colvardata.output_trajectory(f"{output_stem}recon_trajectory.pdb", pred)


    #####################################
    # Output latent space of the dataset
    #####################################
    if config.get('export_latent', False):
        model.export_latent(next(iter(colvardata.test_dataloader())))

    ##################################
    trainer.test(model, datamodule=colvardata)

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

        dataprocessor = colvardata.get_dataset().get_metatomic_dataprocessor()
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
        fake_systems = colvardata.get_fake_systems()
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





