import os
import itertools
import sys
import time
import numpy as np


import torch
import torch.nn as nn
torch.manual_seed(0)
TORCH_PI = torch.acos(torch.zeros(1))*2


from torch.distributions.normal import Normal


def gaussian_sample(mean, var):
    std = torch.sqrt(var) 
    eps = torch.randn_like(std)
    return mean + eps*std
		
def log_normal(x, mu, var, eps=1e-10):
    if eps > 0.0:
        var = torch.add(var, eps)
    p_x = Normal(mu, torch.sqrt(var)) 
    return p_x.log_prob(x)
		
# n_x is number input features
# n_z is dim of latent space
# k is number of clusters/categories in gaussian mix

n_h = 16
use_batch_norm = True

# vae subgraphs
def qy_map(n_x, k, hidden_dims=[16, 16]):
    """q(y|x) computation subgraph generator function.
    
    Args:
        x (Tensor): x tensor.
        k (int): Number of mixtures in the distribution.
        hidden_dims (iterable of int): Hidden layer dimensions as an iterable.
    """
    
    # Hidden layers.
    qy_layers = []
    qy_layers.append(nn.Linear(n_x, hidden_dims[0]))
    qy_layers.append(nn.ReLU(True))
    if use_batch_norm:
        qy_layers.append(nn.BatchNorm1d(hidden_dims[0]))
    for i in range(1, len(hidden_dims)):
        qy_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        qy_layers.append(nn.ReLU(True))
        if use_batch_norm:
            qy_layers.append(nn.BatchNorm1d(hidden_dims[i]))    
    qy_layers.append(nn.Linear(hidden_dims[-1], k))
    qy_layers.append(nn.ReLU(True))
    
    # Output layers.
    qy_logit = nn.Sequential(*qy_layers)
    qy_ytransform = nn.Softmax(dim=1)
    
    return qy_logit, qy_ytransform

def qz_map(n_x, k, n_z, hidden_dims=[16,16]):
    """q(z|x,y) computation subgraph generator function.
    """
    # Initial y transformation.
    qz_ytransform = nn.Linear(k, k)
    
    # Add hidden layers.
    qz_hlayers = []
    qz_hlayers.append(nn.Linear(n_x + k, hidden_dims[0]))
    qz_hlayers.append(nn.ReLU(True))
    if use_batch_norm:
        qz_hlayers.append(nn.BatchNorm1d(hidden_dims[0]))
    for i in range(1, len(hidden_dims)):
        qz_hlayers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        qz_hlayers.append(nn.ReLU(True))
        if use_batch_norm:
            qz_hlayers.append(nn.BatchNorm1d(hidden_dims[i]))
    qz_hlayers = nn.Sequential(*qz_hlayers)
    
    # Output layers.
    qz_zmtransform = nn.Linear(hidden_dims[-1], n_z)
    qz_zvtransform = nn.Sequential(nn.Linear(hidden_dims[-1], n_z), nn.Softplus())#+1e-5

    return qz_ytransform, qz_hlayers, qz_zmtransform, qz_zvtransform

def pz_map(k, n_z, hidden_dims=[16]):
    """p(z|y) is computed here."""
    # Hidden layers.
    pz_hlayers = []
    pz_hlayers.append(nn.Linear(k, hidden_dims[0]))
    pz_hlayers.append(nn.ReLU(True))
    if use_batch_norm:
        pz_hlayers.append(nn.BatchNorm1d(hidden_dims[0]))
    for i in range(1, len(hidden_dims)):
        pz_hlayers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        pz_hlayers.append(nn.ReLU(True))
        if use_batch_norm:
            pz_hlayers.append(nn.BatchNorm1d(hidden_dims[i]))
    pz_hlayers = nn.Sequential(*pz_hlayers)
    
    # Output layers.
    pz_zmtransform = nn.Linear(hidden_dims[-1], n_z)
    pz_zvtransform = nn.Sequential(nn.Linear(hidden_dims[-1], n_z), nn.Softplus())#+1e-5

    return pz_hlayers, pz_zmtransform, pz_zvtransform

def px_fixed_map(n_z, n_x):
    """p(x|z) is computed here."""
    px_logit = nn.Sequential(nn.Linear(n_z, n_h),nn.ReLU(True),nn.Linear(n_h, n_x))
    return px_logit

def px_map(n_z, n_x, hidden_dims=[16,16]):
    """p(x|z) is computed here."""
    # Hidden layers.
    px_hlayers = []
    px_hlayers.append(nn.Linear(n_z, hidden_dims[0]))
    px_hlayers.append(nn.ReLU(True))
    if use_batch_norm:
        px_hlayers.append(nn.BatchNorm1d(hidden_dims[0]))
    for i in range(1, len(hidden_dims)):
        px_hlayers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        px_hlayers.append(nn.ReLU(True))
        if use_batch_norm:
            px_hlayers.append(nn.BatchNorm1d(hidden_dims[i]))
    px_hlayers = nn.Sequential(*px_hlayers)
    
    # Output layers.
    px_xmtransform = nn.Linear(hidden_dims[-1], n_x)
    px_xvtransform = nn.Sequential(nn.Linear(hidden_dims[-1], n_x), nn.Softplus())#+1e-5

    return px_hlayers, px_xmtransform, px_xvtransform

def labeled_loss(k, x, xm, xv, z, zm, zv, zm_prior, zv_prior):
    """Variational loss for the mixture VAE given for each given q(y=i|x, z), hence the
        name labeled_loss."""
    return -log_normal(x, xm, xv) + log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior) - np.log(1/k) 









































# def progbar(i, iter_per_epoch, message='', bar_length=50, display=True):
#     j = (i % iter_per_epoch) + 1
#     end_epoch = j == iter_per_epoch
#     if display:
#         perc = int(100. * j / iter_per_epoch)
#         prog = ''.join(['='] * int(bar_length * perc / 100))
#         template = "\r[{:" + str(bar_length) + "s}] {:3d}%. {:s}"
#         string = template.format(prog, perc, message)
#         sys.stdout.write(string)
#         sys.stdout.flush()
#         if end_epoch:
#             sys.stdout.write('\r{:100s}\r'.format(''))
#             sys.stdout.flush()
#     return end_epoch, (i + 1)/iter_per_epoch

# def strip_consts(graph_def, max_const_size=32):
#     """Strip large constant values from graph_def."""
#     strip_def = tf.GraphDef()
#     for n0 in graph_def.node:
#         n = strip_def.node.add()
#         n.MergeFrom(n0)
#         if n.op == 'Const':
#             tensor = n.attr['value'].tensor
#             size = len(tensor.tensor_content)
#             if size > max_const_size:
#                 tensor.tensor_content = b"<stripped %d bytes>"%size
#     return strip_def

# def show_graph(graph_def, max_const_size=32):
#     """Visualize TensorFlow graph."""
#     if hasattr(graph_def, 'as_graph_def'):
#         graph_def = graph_def.as_graph_def()
#     strip_def = strip_consts(graph_def, max_const_size=max_const_size)
#     code = """
#         <script>
#           function load() {{
#             document.getElementById("{id}").pbtxt = {data};
#           }}
#         </script>
#         <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
#         <div style="height:600px">
#           <tf-graph-basic id="{id}"></tf-graph-basic>
#         </div>
#     """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

#     iframe = """
#         <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
#     """.format(code.replace('"', '&quot;'))
#     with open('graph.htm','w') as file:
#         file.write('<!DOCTYPE html> <html> <body> \n' +
#                    iframe +
#                    ' </body> </html>')
#     print('graph written')

# def show_default_graph():
#     show_graph(tf.get_default_graph().as_graph_def())







