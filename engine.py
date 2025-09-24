import sys
import os

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
    desc = "Autoencoder neural network for enhanced sampling MD"
    parser = argparse.ArgumentParser(description=desc)

    ## Run types
    parser.add_argument('--datatype', required=True, type=str,
                        help='Input file for training', 
                        choices=['COLVAR', 'KMC', 'MD17', 'XTC', 'XYZ'])
    parser.add_argument('--networktype', required=True, type=str,
                        help='Type of the model', 
                        choices=['AE', 'VAE', 'DVAE', 'EDVAE', 'EDVAEGAN', 
                                 'GMVAE', 'VAEGAN', 'VAECGAN', 'VAECGAN', 
                                 'VAEC_mse', 'VAEC', 'GRAPH_ENCODER'])
    parser.add_argument('--debug', action="store_true", 
                        help='Run in debug mode')
    
    # Run parameters
    parser.add_argument('--nepochs', type=int, required=True, 
                        help='Number of epochs to run')
    parser.add_argument('--nogpu', action="store_true", 
                        help='Do not use gpu acceleration even if available')
    parser.add_argument('--export_latent', action="store_true", 
                        help='Export latent space values on the data')

    # Output Settings
    parser.add_argument('--outpath', required=True, type=str, 
                        help='Output folder for saving the training output')
    parser.add_argument('--outfolder', type=str, default='ce_training', 
                        help='Stem of the folder name to save the output')
    parser.add_argument('--nexp', required=False, default=1, type=int, 
                        help='Experiment number for output names')
    parser.add_argument('--tblogger', action="store_true", 
                        help='Log to Tensorboard logger')
    parser.add_argument('--overwrite', action="store_true", 
                        help='Overwrite output folder')
    parser.add_argument('--output_to_file', action="store_true", 
                        help='Also store output in a file')
    parser.add_argument('--output_traj', action="store_true", 
                        help='Print trajectory of the training data')

    # Logging
    parser.add_argument('--wandb', action="store_true", 
                        help='Log to WandB logger')
    parser.add_argument('--wandb_project', type=str, 
                        default='Collective_encoder', help='WandB project name')

    # Save and/or Load Model
    parser.add_argument('--save_checkpoint', action="store_true", 
                        help='Save Checkpoint')
    parser.add_argument('--save_serial_model', action="store_true", 
                        help='Save Model')
    parser.add_argument('--save_serial_model_path', default=".", type=str, 
                        help='Output folder for saving the model')
    parser.add_argument('--load_model', default=None, type=str, 
                        help='Load model from checkpoint')


    args, _ = parser.parse_known_args()

    return args

args = parse_args()


if args.debug:
    args.nepochs = 2
    args.nexp = 0
    args.outfolder = "debug"
    args.overwrite = True
    args.wandb = False
    print("Running in debug mode")

overwrite = args.overwrite
export_latent = args.export_latent
odir = args.outpath + "/" + args.outfolder + "_"
nntype = args.networktype
nexp = args.nexp
output_to_file = args.output_to_file
output_to_terminal = True
num_epochs = args.nepochs
train = True if num_epochs > 0 else False

##################################
# Importing Lightning Modules
##################################
if nntype == "AE":
    from nets.ae_net import AE as main_nn
    nn_nested_args = argparse.Namespace
elif nntype == "VAE":
    from nets.vae_net import VAE as main_nn
    from nets.vae_net import VAE_args as nn_nested_args
elif nntype == "DVAE":
    from nets.dvae_net import DVAE as main_nn
    from nets.dvae_net import DVAE_args as nn_nested_args
elif nntype == "EDVAE":
    from nets.edvae_net import EDVAE as main_nn
    from nets.edvae_net import EDVAE_args as nn_nested_args
elif nntype == "EDVAEGAN":
    from nets.edvae_gan_net import EDVAEGAN as main_nn
    from nets.edvae_gan_net import EDVAEGAN_args as nn_nested_args
elif nntype == "GMVAE":
    from ce_nets import GMVAE as main_nn
    nn_nested_args = {}
elif nntype == "VAEGAN" or nntype == "VAEGAN_mse":
    from ce_nets import VAEGAN as main_nn
    nn_nested_args = {}
elif nntype == "VAECGAN" or nntype == "VAECGAN_mse":
    from ce_nets import VAECGAN as main_nn
    nn_nested_args = {}
elif nntype == "VAEC_mse":
    from ce_nets import VAEC_mse as main_nn
    nn_nested_args = {}
elif nntype == "VAEC":
    from nets.vae_cnn_net import VAEC as main_nn
    from nets.vae_cnn_net import VAEC_args as nn_nested_args
elif nntype == "GRAPH_ENCODER":
    from nets.gnn_encoder import BondGraphNetEncoderDecoder as main_nn
    from nets.gnn_encoder import BGNE_args as nn_nested_args
else:
    raise ValueError("Unknown network type: "+nntype)

if args.datatype == 'KMC':
    from dataloaders.kmc_dataloader import KmcDataset as main_dl
    from dataloaders.kmc_dataloader import KMC_args as data_nested_args
elif args.datatype == 'COLVAR':
    from dataloaders.colvar_dataloader import ColvarDataset as main_dl
    from dataloaders.colvar_dataloader import COLVAR_args as data_nested_args
elif args.datatype == 'MD17':
    from dataloaders.md17_dataloader import MD17Data as main_dl
    from dataloaders.md17_dataloader import MD17_args as data_nested_args
elif args.datatype == 'XTC':
    from dataloaders.xtc_dataloader import XtcDataset as main_dl
    from dataloaders.xtc_dataloader import XTC_args as data_nested_args
elif args.datatype == 'XYZ':
    from dataloaders.xyz_dataloader import XyzLoader as main_dl
    from dataloaders.xyz_dataloader import XYZ_args as data_nested_args

##################################
# Output directory
##################################
odir_name = odir+str(nexp)

if not overwrite:
    while True:
        odir_name = odir+str(nexp)
        if not os.path.isdir(odir_name):
            os.makedirs(odir_name)
            break
        nexp = nexp + 1
else:
    if not os.path.isdir(odir_name):
        os.makedirs(odir_name)

if len(os.listdir(odir_name)) != 0:
    import shutil
    shutil.rmtree(odir_name, ignore_errors=True)
    os.mkdir(odir_name)

##################################
# Output to file
##################################
if output_to_file:
    import sys
    import subprocess
    print("Redirecting output to file "+odir_name+"/out.txt")
    tee = subprocess.Popen(["tee", odir_name+"/out.txt"], stdin=subprocess.PIPE)
    # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
    # of any child processes we spawn)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

print("Using Pytorch", torch.__version__)

##################################
# Creating Dataset
##################################
outname = odir_name+"/"+nntype+"_"

datamodargs = {}
data_nested_args = data_nested_args()
datamodargs = datamodargs | vars(data_nested_args)

if args.debug:
    datamodargs['dataset_size'] = 50
    datamodargs['sequential'] = True
    datamodargs['batch_size'] = 4
    datamodargs['val_batch_size'] = 4


colvardata = main_dl(**datamodargs)
if args.datatype == 'MD17':
    colvardata.prepare_data()
    colvardata.setup(stage="fit")

##################################
# Setting up the NN
##################################
netargs = {}
netargs['datamodule'] = colvardata
netargs['outname'] = outname

nn_nested_args = nn_nested_args()
netargs = netargs | vars(nn_nested_args)

if args.load_model != None and args.load_model != "false":
    checkpoint_file = args.load_model
    print(f"Loading model from {checkpoint_file}")
    model = main_nn.load_from_checkpoint(checkpoint_file, **netargs)
else:
    model = main_nn(**netargs)

##################################
# Training the NN
##################################
trainargs = {"max_epochs" : num_epochs,
             "log_every_n_steps" : 1,
             "default_root_dir" : odir_name}
if not args.nogpu:
    trainargs["accelerator"] = 'auto'
    trainargs["devices"] = 'auto'
if args.wandb:
    wandb_logger = WandbLogger(project=args.wandb_project,
                             name=odir_name.strip(".").strip("/").replace("/", "_"),
                             log_model=True)
    trainargs["logger"] = wandb_logger
if args.tblogger:
    tblogger = TensorBoardLogger(version=odir_name, save_dir=args.outpath)
    trainargs["logger"] = tblogger

lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainargs["callbacks"] = [lr_monitor]

# trainargs["gradient_clip_val"] = 0.5
# trainargs["gradient_clip_algorithm"] = "norm"

trainer = pl.Trainer(**trainargs)

if train:
    # trainer.tune(model, datamodule=colvardata) # To auto find the lr
    trainer.fit(model, datamodule=colvardata)

if not train and args.load_model == None:
    print("WARNING! Model neither loaded nor trained!")

##################################
# Analysing a loaded model
##################################
# model.print_fve(colvardata)
if args.output_traj:
    colvardata.output_trajectory(f"{odir_name}/data_trajectory.pdb")
    if not args.datatype in ["XTC"]:
        raise ValueError(f"Unsupported data type: {args.datatype}")
    else:
        pred = model(colvardata.get_full_batch()[0])[0].detach().cpu().numpy()
        colvardata.output_trajectory(f"{odir_name}/recon_trajectory.pdb", pred)

#####################################
# Output latent space of the dataset
#####################################
if export_latent:
    model.export_latent(next(iter(colvardata.test_dataloader())))

##################################
# Serializing and checkpointing the model
##################################
if args.save_serial_model:
    modelpath = args.save_serial_model_path if args.save_serial_model_path != None else '.'
    model.export_serial_model(f'{odir_name}/{modelpath}')

if args.save_checkpoint and train:
    trainer.save_checkpoint(f"{outname}checkpoint")
    print(f"@@ checkpoint saved as: {outname}checkpoint")

##################################
trainer.test(model, datamodule=colvardata)






