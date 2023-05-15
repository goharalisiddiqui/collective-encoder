#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import os
import argparse

from timeit import default_timer as timer
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from matplotlib import rc
import pytorch_lightning as pl


torch.manual_seed(0)
np.random.seed(0)


##################################
# Arguments used for online training during metad
##################################
def parse_args():
    desc = "Autoencoder neural network for enhanced sampling MD"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--online', action='store_true', help='Is the script being used as part of online training during metaD.')
    
    parser.add_argument('--inputfile', type=str, help='Input file for online training')
    parser.add_argument('--outpath', type=str, help='Output folder for saving the training output')
    parser.add_argument('--modelpath', type=str, help='Output folder for saving the model')
    
    args = parser.parse_args()
    if args.online:
        if (args.inputfile is None or args.outpath is None or args.modelpath is None):
            parser.error("--online requires --inputfile, --outpath and --modelpath")
    
    return args
    
    
args = parse_args()

start = timer()






if (not args.online):
    ##################################
    # Run Settings
    ##################################
    # Output directory
    overwrite = True
    odir = "run"
    nntype = "AE"
    nexp = 1
    # Input directory and columns
    data_folder = "20221201_COLLECTIVE_ENCODER_TRAINING_DATA"
    ignore_list = ["#!", "FIELDS", "time"]
    # label_list = ["phi", "psi"]
    label_list = ["dist_Au-K1"]
    # Input standarization
    standardize_inputs = True # Normalize inputs to range -1 to 1 (no normalization for values below 1e-6)
    # Output file
    output_to_file = False
    # Load pre-trained model
    load_state = False
    state_file = ""
    # Train model
    hidden_nodes = "1000,500,100,10,2" # NN hidden layers
    train = True
    num_epochs = 10
    # Optimization
    lrate = 1e-2  # Learning rate
    l2_reg = 1e-7  # Regularization of network weights
    # Save model
    save_model = False
    save_checkpoint = False
    save_plumed_ANN = False


    ##################################
    # Some safety checks
    ##################################
    if odir == "train":
        overwrite = False
        output_to_file = True
        save_checkpoint = True
        train = True
else:
    overwrite = False
    odir = args.outpath + "train"
    nntype = "AE"
    nexp = 1
    # Input directory and columns
    data_folder = args.inputfile
    ignore_list = ["#!", "FIELDS", "time"]
    # label_list = ["phi", "psi"]
    label_list = ["phi", "psi"]
    # Input standarization
    standardize_inputs = True # Normalize inputs to range -1 to 1 (no normalization for values below 1e-6)
    # Output file
    output_to_file = False
    # Load pre-trained model
    load_state = False
    state_file = ""
    # Train model
    hidden_nodes = "1000,500,100,10,2" # NN hidden layers
    train = True
    num_epochs = 100
    # Optimization
    lrate = 1e-2  # Learning rate
    l2_reg = 1e-7  # Regularization of network weights
    # Save model
    save_model = True
    save_checkpoint = False
    save_plumed_ANN = False
    





##################################
# Importing Lightning Modules
##################################
#import AE_nn
from ce_nets import LITcollAE as main_nn
from ce_dataloaders import LITColvarData as main_dl




##################################
# Output directory
##################################

odir_name = odir+str(nexp)

if not overwrite:
    while True:
        odir_name = odir+str(nexp)
        if not os.path.isdir(odir_name):
            os.mkdir("./"+odir_name)
            break
        nexp = nexp + 1
else:
    if not os.path.isdir(odir_name):
        os.mkdir("./"+odir_name)

if len(os. listdir("./"+odir_name)) != 0:
    import shutil
    shutil.rmtree("./"+odir_name, ignore_errors=True)
    os.mkdir("./"+odir_name)



##################################
# Output to file
##################################
if output_to_file:
    import sys
    orig_stdout = sys.stdout
    f = open("./"+odir_name+"/out.txt", 'w')
    print("Redirecting output to file ./"+odir_name+"/out.txt")
    sys.stdout = f


print("Using Pytorch", torch.__version__)

##################################
# Creating Dataset
##################################
data_dir = os.environ.get('DATA_DIR')
if args.online:
    infile = args.inputfile
else:
    infile = f"{data_dir}/{data_folder}/INPUTS"

outname = "./"+odir_name+"/"+nntype+"_"


colvardata = main_dl(infile, train_prop=0.8, batch_prop=0.1, 
                     label_list=label_list, standardize_inputs=True)

colvardata.setup()




##################################
# Setting up the NN
##################################

nodes = [int(x) for x in hidden_nodes.split(",")]
nodes.insert(0, num_inputs)

model = main_nn(nodes, lr=lrate, l2_reg=l2_reg, outname=outname)






##################################
# Training the NN
##################################
if train:
    start = timer()
    
    
    trainer = pl.Trainer(max_epochs=num_epochs, log_every_n_steps=1, default_root_dir="./"+odir_name)
    trainer.fit(model, train_loader, valid_loader)
    
    
    end = timer()
    elapsed = end - start
    print(f"\nTook {elapsed} s; {alldata.shape[1]} CVs, {alldata.shape[0]} frames")

if not train and not load_state:
    print("WARNING! Model neither loaded nor trained!")
model.eval()






if output_to_file:
    sys.stdout = orig_stdout
    f.close()
