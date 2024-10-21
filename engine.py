#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import torch
import os
import argparse


from timeit import default_timer as timer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

torch.manual_seed(0)
np.random.seed(0)
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

sys.path.append(os.path.dirname(os.getcwd() + '/nets/'))
sys.path.append(os.path.dirname(os.getcwd() + '/dataloaders/'))


##################################
# Arguments
##################################
def parse_args():
    desc = "Autoencoder neural network for enhanced sampling MD"
    parser = argparse.ArgumentParser(description=desc)

    ## Type of Data
    parser.add_argument('--runtype', type=str, default='COLVAR', help='Input file for training', choices=['COLVAR', 'KMC','MD17'])

    # Output Settings
    parser.add_argument('--outpath', required=True, type=str, help='Output folder for saving the training output')
    parser.add_argument('--outfolder', type=str, default='ce_training', help='Stem of the folder name to save the output')
    parser.add_argument('--nexp', required=False, default=1, type=int, help='Experiment number for output names')
    parser.add_argument('--wand', action="store_true", help='Log to WandB logger')
    parser.add_argument('--tblogger', action="store_true", help='Log to Tensorboard logger')
    parser.add_argument('--overwrite', action="store_true", help='Overwrite output folder')
    parser.add_argument('--output_to_file', action="store_true", help='Also store output in a file')

    # Save and/or Load Model
    parser.add_argument('--save_checkpoint', action="store_true", help='Save Checkpoint')
    parser.add_argument('--save_serial_model', action="store_true", help='Save Model')
    parser.add_argument('--save_serial_model_path', default=None, type=str, help='Output folder for saving the model')
    parser.add_argument('--load_model', default=None, type=str, help='Load model from checkpoint')

    # Run parameters
    parser.add_argument('--nogpu', action="store_true", help='Do not use gpu acceleration')
    parser.add_argument('--normalize', action="store_true", help='Normalize input or not')
    parser.add_argument('--plot_every', type=int, default=0, help='Number of epochs to run')

    parser.add_argument('--lrate', type=float, default=1e-4, help='Learning rate for the training')
    parser.add_argument('--l2norm', type=float, default=1e-3, help='Weights regularization for the training')
    parser.add_argument('--nobatchnorm', action="store_false", help='Disable batch normalization in the network')

    parser.add_argument('--network', type=str, default= '500,100,10,2', help='Architecture of the Autoencoder')
    parser.add_argument('--networktype', type=str, default='VAE_mse', help='Type of the Autoencoder, AE, VAE_mse, VAE_elbo')
    parser.add_argument('--nepochs', type=int, help='Number of epochs to run')

    parser.add_argument('--export_latent', action="store_true", help='Export latent space values on the data')
    parser.add_argument('--no_plotdata', action="store_true", help='Do not export plot data as numpy objects')


    args, _ = parser.parse_known_args()

    return args


args = parse_args()



overwrite = args.overwrite
export_latent = args.export_latent
odir = args.outpath + "/" + args.outfolder + "_"
nntype = args.networktype
nexp = args.nexp
# Input directory and columns
ignore_list = ["#!", "FIELDS", "time"]
# Input standarization
standardize_inputs = args.normalize # Normalize inputs to range -1 to 1 (no normalization for values below 1e-6)
# Output file
output_to_file = args.output_to_file
output_to_terminal = True
# Load pre-trained model
# Train model
hidden_nodes = args.network # NN hidden layers
num_epochs = args.nepochs
plot_every = args.plot_every
train = True if num_epochs > 0 else False
# Optimization
lrate = args.lrate  # Learning rate
l2_reg = args.l2norm  # Regularization of network weights






##################################
# Importing Lightning Modules
##################################
#import AE_nn
if nntype == "AE":
    from nets.ae_net import AE as main_nn
    nn_nested_args = argparse.Namespace()
elif nntype == "VAE_mse":
    from ce_nets import VAE_mse as main_nn
    nn_nested_args = argparse.Namespace()
elif nntype == "VAE":
    from nets.vae_net import VAE as main_nn
    from nets.vae_net import VAE_args as nn_nested_args
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
else:
    raise ValueError("Unknown network type")

if args.runtype == 'KMC':
    from dataloaders.kmc_dataloader import KmcDataset as main_dl
    from dataloaders.kmc_dataloader import KMC_args as data_nested_args
elif args.runtype == 'COLVAR':
    from dataloaders.colvar_dataloader import ColvarDataset as main_dl
    from dataloaders.colvar_dataloader import COLVAR_args as data_nested_args
elif args.runtype == 'MD17':
    from dataloaders.md17_dataloader import MD17Data as main_dl
    from dataloaders.md17_dataloader import MD17_args as data_nested_args




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
if args.runtype in ["COLVAR", "KMC"]:
    datamodargs['train_prop'] = 0.8
    datamodargs['batch_prop'] = 0.1
    datamodargs['standardize_inputs'] = False # We dont standardize the inputs here, we do it in the model otherwise it does not work with plumed

data_nested_args = data_nested_args()
datamodargs = datamodargs | vars(data_nested_args)


colvardata = main_dl(**datamodargs)
if args.runtype == 'MD17':
    colvardata.prepare_data()

colvardata.setup(stage="fit")


##################################
# Setting up the NN
##################################

nodes = [int(x) for x in hidden_nodes.split(",")]
nodes.insert(0, colvardata.num_inputs)

netargs = {}
netargs['lr'] = lrate
netargs['l2_reg'] = l2_reg
netargs['outname'] = outname
netargs['batch_norm'] = args.nobatchnorm
netargs['plot_every'] = args.plot_every
netargs['saveplotdata'] = not args.no_plotdata

if nntype != "GMVAE":
    netargs['l'] = nodes
else:
    netargs['n_x'] = nodes[0]
    netargs['n_z'] = nodes[-1]

nn_nested_args = nn_nested_args()
netargs = netargs | vars(nn_nested_args)


if args.load_model != None:
    checkpoint_file = args.load_model
    print(f"Loading model from {checkpoint_file}")
    model = main_nn.load_from_checkpoint(checkpoint_file, **netargs)
else:
    model = main_nn(**netargs)
if standardize_inputs:
    model.set_norm(torch.Tensor(colvardata.get_scaler_mean(), device=model.device),
                    torch.Tensor(colvardata.get_scaler_scale(), device=model.device))

##################################
# Training the NN
##################################
trainargs = {"max_epochs" : num_epochs,
             "log_every_n_steps" : 1,
             "default_root_dir" : odir_name}
if not args.nogpu:
    trainargs["accelerator"] = 'auto'
    trainargs["devices"] = 'auto'
if args.wand:
    wandb_logger = WandbLogger(log_model="all")
    trainargs["logger"] = wandb_logger
if args.tblogger:
    tblogger = TensorBoardLogger(version=odir_name, save_dir=args.outpath)
    trainargs["logger"] = tblogger

trainargs["gradient_clip_val"] = 0.5
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



#####################################
# Output latent space of the dataset
#####################################

if export_latent:
    model.export_latent(next(iter(colvardata.test_dataloader())))




##################################
# Serializing and checkpointing the model
##################################
if args.save_serial_model:
    modelpath = args.modelpath if args.save_serial_model_path != None else '.'
    model.export_serial_model(f'{odir_name}/{modelpath}')

if args.save_checkpoint:
    trainer.save_checkpoint(f"{outname}checkpoint")
    print(f"@@ checkpoint saved as: {outname}checkpoint")


##################################
trainer.test(model, datamodule=colvardata)






