#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import torch
import os
import argparse


from timeit import default_timer as timer
import pytorch_lightning as pl

torch.manual_seed(0)
np.random.seed(0)


##################################
# Arguments
##################################
def parse_args():
    desc = "Autoencoder neural network for enhanced sampling MD"
    parser = argparse.ArgumentParser(description=desc)



    parser.add_argument('--inputfile', type=str, help='Input file for training')
    parser.add_argument('--outpath', required=True, type=str, help='Output folder for saving the training output')
    parser.add_argument('--outfolder', type=str, default='ce_training', help='Stem of the folder name to save the output')
    parser.add_argument('--nexp', required=False, default=1, type=int, help='Experiment number for output names')


    parser.add_argument('--modelpath', type=str, help='Output folder for saving the model')
    parser.add_argument('--save_model', action="store_true", help='Save Model')

    parser.add_argument('--save_checkpoint', action="store_true", help='Save Checkpoint')

    parser.add_argument('--load_model', action="store_true", help='Load Model')
    parser.add_argument('--modelfile', type=str, help='From where to load the model')


    parser.add_argument('--labels', nargs='+', help='Labels to ignore in the input files. Used for visualisation')
    parser.add_argument('--gpu', action="store_true", help='Use gpu acceleration')
    parser.add_argument('--overwrite', action="store_true", help='Overwrite output folder')
    parser.add_argument('--normalize', action="store_true", help='Normalize input or not')
    parser.add_argument('--output_to_file', action="store_true", help='Also store output in a file')

    parser.add_argument('--beta', type=float, default=1.0, help='beta for the beta-VAE')
    parser.add_argument('--lrate', type=float, default=1e-4, help='Learning rate for the training')
    parser.add_argument('--l2norm', type=float, default=1e-3, help='Weights regularization for the training')


    parser.add_argument('--network', type=str, default= '500,100,10,2', help='Architecture of the Autoencoder')
    parser.add_argument('--solvation', type=str, default= '0', help='Grid size of the solvation grid')
    parser.add_argument('--networktype', type=str, default='VAE_mse', help='Type of the Autoencoder, AE, VAE_mse, VAE_elbo')
    parser.add_argument('--nepochs', type=int, help='Number of epochs to run')

    args = parser.parse_args()

    return args


args = parse_args()

start = timer()

overwrite = args.overwrite
odir = args.outpath + "/" + args.outfolder + "_"
nntype = args.networktype
nexp = args.nexp
solvation = args.solvation
# Input directory and columns
ignore_list = ["#!", "FIELDS", "time"]
label_list = args.labels
# Input standarization
standardize_inputs = args.normalize # Normalize inputs to range -1 to 1 (no normalization for values below 1e-6)
# Output file
output_to_file = args.output_to_file
output_to_terminal = True
# Load pre-trained model
load_state = args.load_model
state_file = args.modelfile
# Train model
hidden_nodes = args.network # NN hidden layers
num_epochs = args.nepochs
train = True if num_epochs > 0 else False
# Optimization
lrate = args.lrate  # Learning rate
l2_reg = args.l2norm  # Regularization of network weights
# Hyperparameters
beta = args.beta
# Save model
save_model = args.save_model
save_checkpoint = args.save_checkpoint






##################################
# Importing Lightning Modules
##################################
#import AE_nn
if nntype == "AE":
    from ce_nets import LITcollAE as main_nn
elif nntype == "VAE_mse" or nntype == "VAE":
    from ce_nets import LITcollVAE as main_nn
elif nntype == "GMVAE":
    from ce_nets import GMVAE as main_nn
elif nntype == "VAEGAN" or nntype == "VAEGAN_mse":
    from ce_nets import VAEGAN as main_nn
elif nntype == "VAECGAN" or nntype == "VAECGAN_mse":
    assert solvation != '0', "Solvation grid size not provided"
    from ce_nets import VAECGAN as main_nn
else:
    raise ValueError("Unknown network type")
from ce_dataloaders import LITColvarData as main_dl




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
        os.mkdir(odir_name)

if len(os. listdir(odir_name)) != 0:
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
infile = args.inputfile

outname = odir_name+"/"+nntype+"_"


colvardata = main_dl(infile, train_prop=0.8, batch_prop=0.1,
                     label_list=label_list, standardize_inputs=False) # We dont standardize the inputs here, we do it in the model otherwise it does not work with plumed

colvardata.setup(stage="")




##################################
# Setting up the NN
##################################

nodes = [int(x) for x in hidden_nodes.split(",")]
nodes.insert(0, colvardata.num_inputs)

solv = [int(x) for x in solvation.split(",")]

netargs = {'lr': lrate, 'l2_reg' :l2_reg, 'outname': outname}

if nntype == "AE":
    netargs['l'] = nodes
elif nntype == "VAE_mse" or nntype == "VAEGAN_mse":
    netargs['l'] = nodes
    netargs['loss_type'] = "mse"
    netargs['beta'] = beta
elif nntype == "VAE" or nntype == "VAEGAN":
    netargs['l'] = nodes
    netargs['loss_type'] = "elbo"
    netargs['beta'] = beta
elif nntype == "VAECGAN":
    netargs['l'] = nodes
    netargs['lw'] = solv
    netargs['loss_type'] = "elbo"
    netargs['beta'] = beta
elif nntype == "VAECGAN_mse":
    netargs['l'] = nodes
    netargs['lw'] = solv
    netargs['loss_type'] = "mse"
    netargs['beta'] = beta
elif nntype == "GMVAE":
    netargs['n_x'] = nodes[0]
    netargs['n_z'] = nodes[-1]
else:
    raise ValueError("Unknown network type")

if load_state:
    model = main_nn.load_from_checkpoint(state_file, **netargs)
else:
    model = main_nn(**netargs)
if standardize_inputs:
    model.set_norm(torch.Tensor(colvardata.get_scaler_mean(), device=model.device),
                    torch.Tensor(colvardata.get_scaler_scale(), device=model.device))

##################################
# Training the NN
##################################
if args.gpu and torch.cuda.is_available():
    print("GPU enabled")
    trainer = pl.Trainer(max_epochs=num_epochs, log_every_n_steps=1, default_root_dir=odir_name, accelerator='gpu', devices=1)
else:
    print("NO GPU")
    trainer = pl.Trainer(max_epochs=num_epochs, log_every_n_steps=1, default_root_dir=odir_name)
if train:
    start = timer()
    # trainer.tune(model, datamodule=colvardata) # To auto find the lr
    trainer.fit(model, datamodule=colvardata)

    end = timer()
    elapsed = end - start
    print(f"\nTook {elapsed} s; {colvardata.num_inputs - len(label_list)} CVs, {len(colvardata.all_dataset)} frames")

if not train and not load_state:
    print("WARNING! Model neither loaded nor trained!")



##################################
# Analysing a loaded model
##################################
if nntype != "AE" and nntype != "GMVAE":
    model.get_fve(colvardata)





##################################
# Serializing and checkpointing the model
##################################
if save_model:
    print("[Exporting the model]")

    fake_loader = torch.utils.data.DataLoader(colvardata.all_dataset, batch_size=1, shuffle=False)
    fake_input = next(iter(fake_loader))[0].float()

    if args.modelpath == None:
        modelpath = odir_name
    else:
        modelpath = args.modelpath
    if not os.path.isdir(modelpath):
            os.makedirs(modelpath)

    model.metaD = True
    # model.to_torchscript(file_path=f"{modelpath}/encoder.pt", method='trace', example_inputs=fake_input, strict=False)
    torch.jit.save(model.to_torchscript(method='trace', example_inputs=fake_input, strict=False), f"{odir_name}/{modelpath}/encoder.pt")

    print(f"@@ model exported as: {odir_name}/{modelpath}/encoder.pt")

if save_checkpoint:
    trainer.save_checkpoint(f"{outname}checkpoint")
    print(f"@@ checkpoint saved as: {outname}checkpoint")


##################################
trainer.test(model, datamodule=colvardata)



