
##################################
# Encoding functions
##################################

import numpy as np
import torch
from torch.autograd import Variable
n_hidden = 0

def encode_latent(loader, model, batch):
    """Compute the latent space over a dataloader"""
    s = np.empty((len(loader), batch, n_hidden))
    l = np.empty((len(loader), batch))
    d = np.empty((len(loader), batch))
    for i, data in enumerate(loader):
        x, lab, dis = data[0].float(), data[1].long(), data[2].float()
        x = Variable(x).to(device)
        cv = model.encode(x)
        s[i] = cv.detach().cpu().numpy()
        l[i] = lab
        d[i] = dis

    s = s.reshape(len(loader) * batch, n_hidden)
    s = s[0 : len(loader) * batch]

    l = l.reshape(len(loader) * batch)
    d = d.reshape(len(loader) * batch)
    

    sA = s[l == 0]
    sB = s[l == 1]
    dA = d[l == 0]
    dB = d[l == 1]
    

    return sA, sB, dA, dB