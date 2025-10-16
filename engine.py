import sys
import os
import yaml

sys.path.append(os.path.dirname(os.getcwd() + '/nets/'))
sys.path.append(os.path.dirname(os.getcwd() + '/dataloaders/'))

import argparse
import warnings

import numpy as np

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

from utils import parse_vars

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
np.random.seed(0)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
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


args = parse_args()

assert os.path.isfile(args.config), "Config file not found!"
config = yaml.safe_load(open(args.config, 'r'))
if args.debug:
    config['debug'] = True

if config['debug']:
    config['nepochs'] = 2
    config['nexp'] = 0
    config['outpath'] = "train_runs"
    config['outfolder'] = "debug"
    config['overwrite'] = True
    config['wandb'] = False
    config['output_to_file'] = True
    config['nogpu'] = True
    config['export_latent'] = False

    config['data_args']['dataset_size'] = 50
    config['data_args']['sequential'] = True
    config['data_args']['train_size'] = 40
    config['data_args']['batch_size'] = 4
    config['data_args']['validation_size'] = 10
    config['data_args']['val_batch_size'] = 4
    config['data_args']['test_full_dataset'] = True
    # config['data_args']['dataset_type'] = "DEFAULT"
    config['data_args']['num_workers'] = 1
    print("Running in debug mode")

##################################
# Importing Lightning Modules
##################################
nntype = config['network_name']
if nntype == "AE":
    from nets.ae_net import AE as main_nn
elif nntype == "VAE":
    from nets.vae_net import VAE as main_nn
elif nntype == "DVAE":
    from nets.dvae_net import DVAE as main_nn
elif nntype == "EDVAE":
    from nets.edvae_net import EDVAE as main_nn
elif nntype == "EDVAEGAN":
    from nets.edvae_gan_net import EDVAEGAN as main_nn
elif nntype == "GMVAE":
    from ce_nets import GMVAE as main_nn
elif nntype == "VAEGAN" or nntype == "VAEGAN_mse":
    from ce_nets import VAEGAN as main_nn
elif nntype == "VAECGAN" or nntype == "VAECGAN_mse":
    from ce_nets import VAECGAN as main_nn
elif nntype == "VAEC_mse":
    from ce_nets import VAEC_mse as main_nn
elif nntype == "VAEC":
    from nets.vae_cnn_net import VAEC as main_nn
elif nntype == "GRAPH_ENCODER":
    from nets.gnn_encoder import BondGraphNetEncoderDecoder as main_nn
else:
    raise ValueError("Unknown network type: "+nntype)

datatype = config['data_name']
if datatype == 'KMC':
    from dataloaders.kmc_dataloader import KmcDataset as main_dl
elif datatype == 'COLVAR':
    from dataloaders.colvar_dataloader import ColvarDataset as main_dl
elif datatype == 'MD17':
    from dataloaders.md17_dataloader import MD17Data as main_dl
elif datatype == 'XTC':
    from dataloaders.xtc_dataloader import XtcDataset as main_dl
elif datatype == 'XYZ':
    from dataloaders.xyz_dataloader import XyzLoader as main_dl

##################################
# Output directory
##################################
nexp = config['nexp']
run_stem = config['outpath'] + "/" + config['outfolder'] + "_"
run_dir = run_stem + str(nexp)

if not config['overwrite']:
    while True:
        run_dir = run_dir + str(nexp)
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir)
            break
        nexp = nexp + 1
else:
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

if len(os.listdir(run_dir)) != 0:
    import shutil
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
    
datamodargs = config['data_args']
colvardata = main_dl(**datamodargs)
if config['data_name'] == 'MD17':
    colvardata.prepare_data()
    colvardata.setup(stage="fit")

##################################
# Setting up the NN
##################################
netargs = config['network_args']
netargs['datamodule'] = colvardata
netargs['outname'] = output_stem

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
if not config['nogpu']:
    trainargs["accelerator"] = 'auto'
    trainargs["devices"] = 'auto'
if config['wandb']:
    wandb_logger = WandbLogger(project=config['wandb_project'],
                             entity=config['wandb_entity'],
                             save_dir=run_dir,
                             name=run_dir.strip(".").strip("/").replace("/", "_"),
                             log_model=False)
    trainargs["logger"] = wandb_logger

callbacks = []
# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='epoch')
callbacks.append(lr_monitor)
# Early stopping
if 'early_stopping' in config and config['early_stopping']:
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)

trainargs["callbacks"] = callbacks

# trainargs["gradient_clip_val"] = 0.5
# trainargs["gradient_clip_algorithm"] = "norm"

trainer = pl.Trainer(**trainargs)

if config['nepochs'] > 0:
    trainer.fit(model, datamodule=colvardata)

if config['nepochs'] == 0 and load_model == None:
    warnings.warn("Both nepochs and load_model are not set. Nothing to do.")

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
# Serializing and checkpointing the model
##################################
if config['save_serial_model']:
    modelpath = config.get('save_serial_model_path', '.')
    model.export_serial_model(os.path.join(run_dir, modelpath))

if config['save_checkpoint'] and config['nepochs'] > 0:
    trainer.save_checkpoint(f"{output_stem}checkpoint")
    print(f"@@ checkpoint saved as: {output_stem}checkpoint")

##################################
trainer.test(model, datamodule=colvardata)

#####################################
# Save metatomic model
#####################################
if config.get('save_metatomic', False):
    try:
        from mtomic.wrapper import MetatomicCV
        from metatomic.torch import (
            AtomisticModel,
            ModelCapabilities,
            ModelMetadata,
            ModelOutput,
            System,
            ModelEvaluationOptions,
        )
    except ImportError:
        raise ImportError("metatomic is not installed. Please install it with `pip install metatomic`")

    dataprocessor = colvardata.get_dataset().get_metatomic_dataprocessor()
    metamodel = model.get_metatomic_model()
    metatomic_model = MetatomicCV(dataprocessor, metamodel)
    metadata = ModelMetadata(
        name=config.get('metatomic_metadata', {}).get('name', 'unknown'),
        description=config.get('metatomic_metadata', {}).get('description', 'unknown'),
        authors=config.get('metatomic_metadata', {}).get('authors', []),
        references=config.get('metatomic_metadata', {}).get('references', {}),
    )
    capabilities = ModelCapabilities(
        outputs={"features": ModelOutput(quantity="", unit="none", per_atom=False),},
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
    # Sanity checks of the model
    at_types = colvardata.get_atns()
    fake_systems = [
        System(
            types=torch.tensor(at_types, dtype=torch.long),
            positions=torch.randn(len(at_types), 3, dtype=torch.float64),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),),
        System(
            types=torch.tensor(at_types, dtype=torch.long),
            positions=torch.randn(len(at_types), 3, dtype=torch.float64),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),)
        ]
    fake_options = ModelEvaluationOptions(
        length_unit="nanometer",
        outputs={"features": ModelOutput(quantity="", unit="none", per_atom=False),},
        selected_atoms=None,
    )
        
    # Run inference
    try:
        with torch.no_grad():
            output = metatomic_module(fake_systems, fake_options, False)
    except Exception as e:
        raise RuntimeError("metatomic model failed the sanity check: "+str(e))


    metatomic_model_file = os.path.join(run_dir, "metatomic_model.pt")
    metatomic_module.save(metatomic_model_file)

    print(f"@@ metatomic model saved as: {metatomic_model_file}")





