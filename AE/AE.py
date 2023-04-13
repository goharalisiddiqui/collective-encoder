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
    nexp = 2
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
    num_epochs = 1000
    # Optimization
    lrate = 1e-2  # Learning rate
    l2_reg = 1e-7  # Regularization of network weights
    # Save model
    save_model = True
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
# Importing Dataset
##################################
#import AE_nn as dnn
from AE_dataset import ColvarDataset as ColDataset

##################################
# Importing Neural Networks
##################################
#import AE_nn
from AE_nn import NN_AutoE as main_nn

##################################
# Importing Loss Functions
##################################
# import AE_loss
#from AE_loss import loss_trace as f_loss
#from AE_loss import loss_trace as f_loss
#from AE_loss import check_LDA_cholesky as f_loss_check

##################################
# Importing Plotting Functions
##################################
import AE_plot
from AE_plot import plot_results

##################################
# Importing Encoding Functions
##################################
import AE_encode
from AE_encode import encode_latent
#from AE_encode import encode_cv
#from AE_encode import encode_cv_all





##################################
# output directory
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
# output to file
##################################
if output_to_file:
    import sys
    orig_stdout = sys.stdout
    f = open("./"+odir_name+"/out.txt", 'w')
    print("Redirecting output to file ./"+odir_name+"/out.txt")
    sys.stdout = f




print("Using Pytorch", torch.__version__)

##################################
# Reading data from files
##################################
data_dir = os.environ.get('DATA_DIR')
if args.online:
    infile = args.inputfile
else:
    infile = f"{data_dir}/{data_folder}/INPUTS"

outname = "./"+odir_name+"/"+nntype+"_"

with open(infile) as f:
    first_line = f.readline().split()
    

assert all([x in first_line for x in label_list]) # Asserts that all the print labels are present in the data file

header = [x for x in first_line if x not in ignore_list and x not in label_list]
first_col = len(ignore_list) + len(label_list) - 2  # 0-indexed
header_string = ",".join(header)
num_inputs = len(header)
print(f"Loading data from file: {infile}")

col_range = range(first_col, first_col + num_inputs)

alldata = np.loadtxt(infile, usecols=col_range)
alllabel = np.loadtxt(infile, usecols=[a for a in range(1,len(label_list) + 1)], ndmin = 2)

print("[Imported data]")
print("- data.shape:", alldata.shape)
print("- label.shape:", alllabel.shape)
assert alldata.shape[0] == alllabel.shape[0]
##################################################################################











##################################
# Creating datasets
##################################

if standardize_inputs:
    print("[Standardize inputs]")
    print("- Calculating mean and range over the training set")
    Max = np.amax(alldata, axis=0)
    Min = np.amin(alldata, axis=0)
    Mean = (Max + Min) / 2.0
    Range = (Max - Min) / 2.0
    if np.sum(np.argwhere(Range < 1e-6)) > 0:
        print(
            "- [Warning] Skipping normalization where range of values is < 1e-6. Input(s):",
            np.argwhere(Range < 1e-6).reshape(-1),
        )
        Range[Range < 1e-6] = 1.0
else:
    Mean = 0.0
    Range = 1.0


## Shuffling data
p = np.random.permutation(len(alldata))
alldata, alllabel = alldata[p], alllabel[p]

## Dividing data into training and validation data
FRAMES = alldata.shape[0]
train_data = int(FRAMES * 0.8)

## Choosing batch size and creating dataloader for training
batch_tr = int(FRAMES * 0.1)
train_labels = ColDataset([alldata[:train_data], alllabel[:train_data]])
train_loader = DataLoader(train_labels, batch_size=batch_tr, shuffle=True, drop_last=True)

## Creating extra Dataloader that have only one batch for model evaluation
train_all_labels = ColDataset([alldata[:train_data], alllabel[:train_data]])
train_all_loader = DataLoader(train_all_labels, batch_size=train_data, drop_last=True)

# Creating dataloader for validation with only one batch
valid_data = int(FRAMES * 0.2)
batch_val = valid_data
valid_labels = ColDataset([alldata[train_data : train_data + valid_data], alllabel[train_data : train_data + valid_data]])
valid_loader = DataLoader(valid_labels, batch_size=batch_val, drop_last=True)

# Printing
print(f"Total frames: {FRAMES}, Train size: {train_data}, Batch size: {batch_tr}, Validation size: {valid_data}")

# Checking if any of the Dataloaders are empyty
assert len(train_loader) >= 1
assert len(train_all_loader) >= 1
assert len(valid_loader) >= 1
###################################################################################

















##################################
# Setting up the NN
##################################

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu" # dont use cuda since libtorch not compiled with it 
print(f"Using device = {device}\n")

nodes = [int(x) for x in hidden_nodes.split(",")]
nodes.insert(0, num_inputs)
n_hidden = nodes[-1]

print("[NN Architecture]")
print("- hidden layers:", nodes)
print("")
print("========= NN =========")
model = main_nn(nodes)
## For nomalizing the inputs
if standardize_inputs:
    model.set_norm(torch.tensor(Mean, dtype=dtype, device=device), torch.tensor(Range, dtype=dtype, device=device))
print("======================")

model.to(device)

if torch.cuda.is_available():
     print("using CUDA acceleration")
     print("========================")

# -- Optimization --
# lrate = 0.0005  # Learning rate
# l2_reg = 1e-5  # Regularization of network weights


## Setting variables for the module files
AE_encode.device = device
AE_plot.device = device
AE_encode.n_hidden = n_hidden
# AE_loss.device = device
# AE_loss.dtype = dtype
AE_plot.outname = outname


# OPTIMIZERS
opt = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=l2_reg)









##################################
# Loading state dict
##################################

init_epoch = 0
if load_state:
    if not os.path.exists(state_file):
        raise ValueError('Cannot find state file')
    print(f"Loading model and optimizer state from file {state_file}.")
    model.load_state_dict(torch.load(state_file)["model_state_dict"])
    opt.load_state_dict(torch.load(state_file)["optimizer_state_dict"])
    init_epoch = torch.load(state_file)["epoch"]







##################################################
# Resetting lr and weight_decay of the optimizer
##################################################
for g in opt.param_groups:
    g['lr'] = lrate
    g['weight_decay'] = l2_reg





##################################
# Optimizer state
##################################
print("")
print("[Optimization]")
print("- Learning rate \t=", opt.param_groups[0]['lr'])
print("- l2 regularization \t=", opt.param_groups[0]['weight_decay'])





##################################
# Training the NN
##################################
if train:
    start = timer()
    if torch.cuda.is_available():
        print("using CUDA acceleration")
        print("========================")
    model.to(device)
    epochs = []
    loss_list = []

    print_loss = 1
    plot_every = (int)(num_epochs/4)
    plot_validation = True

    ## Format output
    float_formatter = lambda x: "%.6f" % x
    np.set_printoptions(formatter={"float_kind": float_formatter})
    ## Print Header
    print("[{:>3}/{:>3}] {:>10}".format("ep", "tot", "loss"))
    loss_fn = nn.MSELoss()
    ## Training
    model.train()
    for epoch in range(num_epochs):
        ## Go through the batches and train the network
        for data in train_loader:
            # =================get data===================
            X, y = data[0].float().to(device), data[1].long().to(device)
            # =================forward====================
            H = model.forward(X)
            # =================Normalize truth===================
            X_n = model.normalize(X)
            # =================loss===================
            loss = loss_fn(H, X_n)
            # =================reg loss===================
            # =================backprop===================
            opt.zero_grad()
            loss.backward()
            opt.step()

            # =================Clear GPU mem===================
            del X;del y;del H;del X_n

        ## Save results
        epochs.append(epoch + init_epoch + 1)
        loss_list.append(loss)

        ## Print training status
        if (epoch + 1) % print_loss == 0:
            print("[{:3d}/{:3d}] {:10.3f}".format(init_epoch + epoch + 1,init_epoch + num_epochs, loss))
            # print("Cuda memory alloc=",torch.cuda.memory_allocated())

        if (epoch + 1) % plot_every == 0:
            plot_results(epochs, model, loss_list, train_all_loader, valid_loader, label_list, save = True)

    init_epoch += num_epochs # Just to restart training
    end = timer()
    elapsed = end - start
    print(f"\nTook {elapsed} s; {alldata.shape[1]} CVs, {alldata.shape[0]} frames")

if not train and not load_state:
    print("WARNING! Model neither loaded nor trained!")
model.eval()





####### MODEL SAVING #######

if save_model:
    print("[Exporting the model]")
    device = "cpu"
    model.to(device)
    fake_loader = DataLoader(train_labels, batch_size=1, shuffle=False)
    fake_input = next(iter(fake_loader))[0].float().to(device)

    model.metaD = True
    mod = torch.jit.trace(model, fake_input)

    if args.online:
        mod.save(f"{args.modelpath}/encoder.pt")
        print(f"@@ model exported as: {args.modelpath}/encoder.pt")
    mod.save(f"{outname}.pt")
    print(f"@@ model exported as: {outname}.pt")


if save_checkpoint:
    # == EXPORT CHECKPOINT ==
    # python binary
    model.to("cpu")
    torch.save(
        {
            "epoch": num_epochs + init_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        },
        f"{outname}checkpoint",
    )
    print(f"@@ checkpoint saved as: {outname}checkpoint")


if save_plumed_ANN:
    # == EXPORT MODEL AS PLUMED KEYWORD ANN ==
    n_l = len(model.state_dict())//4 + 1
    print(model.state_dict().keys())
    # print(",".join([str(num) for num in model.state_dict()["encoder.0.weight"].cpu().detach().numpy().flatten()]))
    with open(f"{outname}ann.dat", "w") as ofile:
        ofile.write("ANN ...\n")

        ofile.write("   ARG="+header_string+"\n")
        ofile.write(f"   NUM_LAYERS=%d"%n_l+"\n")
        ofile.write("   ACTIVATIONS="+",".join(["ReLU" for num in range(n_l - 2)])+"\n")
        ofile.write("   NUM_NODES="+",".join([str(num) for num in nodes])+"\n")

        for ind,key in enumerate(model.state_dict().keys()):
            if 'encoder' in key:
                if 'weight' in key:
                    ofile.write("   WEIGHTS"+str(ind//2)+"="+",".join([str(num) for num in model.state_dict()[key].cpu().detach().numpy().flatten()]))
                    ofile.write("\n")
                elif 'bias' in key:
                    ofile.write("   BIASES"+str(ind//2)+"="+",".join([str(num) for num in model.state_dict()[key].cpu().detach().numpy().flatten()]))
                    ofile.write("\n")
        ofile.write("...")

    print(f"@@ plumed keyword saved as: {outname}ann.dat")



if output_to_file:
    sys.stdout = orig_stdout
    f.close()
