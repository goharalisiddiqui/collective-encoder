##################################
# Loss functions
##################################

import torch

lambdA = 0
dtype = torch.float32
device = "cpu"

def loss_trace(H, label):
    #
    # N = Number of observations(frames)
    # d = Number of latent space variables
    N, d = H.shape
    #######################################################################
    #Calculating scatter matrices
    #######################################################################
    # Mean centered observations for entire population
    H_bar = H - torch.mean(H, 0, True)
    # Total scatter matrix (cov matrix over all observations)
    S_t = H_bar.t().matmul(H_bar) / (N - 1)
    # Define within scatter matrix and compute it
    S_w = torch.Tensor().new_zeros((d, d), device=device, dtype=dtype)
    S_w_inv = torch.Tensor().new_zeros((d, d), device=device, dtype=dtype)
    buf = torch.Tensor().new_zeros((d, d), device=device, dtype=dtype)
    # Loop over classes to compute means and covs
    for i in range(categ):
        # check which elements belong to class i
        H_i = H[torch.nonzero(label == i).view(-1)]   ####
        # compute mean centered obs of class i  
        H_i_bar = H_i - torch.mean(H_i, 0, True)
        # count number of elements
        N_i = H_i.shape[0]
        if N_i == 0:
            continue
        S_w += H_i_bar.t().matmul(H_i_bar) / ((N_i - 1) * categ)
    S_b = S_t - S_w
    S_w = S_w + lambdA * torch.diag(torch.Tensor().new_ones((d), device=device, dtype=dtype))

    rat = torch.trace(S_w)/torch.trace(S_b)


    loss = rat

    return loss, S_b, S_w

def loss_traceNoncov(H, label):
    #
    # N = Number of observations(frames)
    # d = Number of latent space variables
    N, d = H.shape
    #######################################################################
    #Calculating scatter matrices
    #######################################################################
    # Mean centered observations for entire population
    H_bar = H - torch.mean(H, 0, True)
    # Total scatter matrix (cov matrix over all observations)
    S_t = H_bar.t().matmul(H_bar) / (N - 1)
    # Define within scatter matrix and compute it
    S_w = torch.Tensor().new_zeros((d, d), device=device, dtype=dtype)
    S_w_inv = torch.Tensor().new_zeros((d, d), device=device, dtype=dtype)
    buf = torch.Tensor().new_zeros((d, d), device=device, dtype=dtype)
    # Loop over classes to compute means and covs
    for i in range(categ):
        # check which elements belong to class i
        H_i = H[torch.nonzero(label == i).view(-1)]   ####
        # compute mean centered obs of class i  
        H_i_bar = H_i - torch.mean(H_i, 0, True)
        # count number of elements
        N_i = H_i.shape[0]
        if N_i == 0:
            continue
        S_w += H_i_bar.t().matmul(H_i_bar) / ((N_i - 1) * categ)
    S_b = S_t - S_w
    S_w = S_w + lambdA * torch.diag(torch.Tensor().new_ones((d), device=device, dtype=dtype))

    tr_ratio = torch.trace(S_w)/torch.trace(S_b)
    
    off_d = S_t.clone().fill_diagonal_(0)
    cross_cov = off_d.abs().sum()


    loss = tr_ratio + cross_cov

    return loss, S_b, S_w

