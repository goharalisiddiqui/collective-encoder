
##################################
# Define Networks
##################################

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

torch.manual_seed(0) ## don't know the utility of this yet

class NN_AutoE(nn.Module):
    def __init__(self, l):
        super(NN_AutoE, self).__init__()

        modules = []
        for i in range(len(l) - 1):
            print(l[i], " --> ", l[i + 1], end=" ")
            if i < len(l) - 2:
                modules.append(nn.Linear(l[i], l[i + 1]))
                modules.append(nn.ReLU(True))
                modules.append(nn.BatchNorm1d(l[i + 1]))
                print("(relu)")
            else:
                modules.append(nn.Linear(l[i], l[i + 1]))
                print("")
        modules.append(nn.Sigmoid())
        print("(sigmoid)")

        self.encoder = nn.Sequential(*modules)
        modules = []
        a = len(l) - 1
        for i in range(len(l) - 1):
            print(l[a - i], " --> ", l[a - i - 1], end=" ")
            if i < len(l) - 2:
                modules.append(nn.Linear(l[a - i], l[a - i - 1]))
                modules.append(nn.ReLU(True))
                modules.append(nn.BatchNorm1d(l[a - i - 1]))
                print("(relu)")
            else:
                modules.append(nn.Linear(l[a - i], l[a - i - 1]))
                print("")
        

        self.decoder = nn.Sequential(*modules)

        # normalize input
        self.normIn = False
        self.metaD = False

    def set_norm(self, Mean: torch.Tensor, Range: torch.Tensor):
        self.normIn = True
        self.register_buffer('Mean', Mean)
        self.register_buffer('Range', Range)

    def normalize(self, x: Variable):
        batch_size = x.size(0)
        x_size = x.size(1)

        Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

        return x.sub(Mean).div(Range)
    
    def denormalize(self, x: Variable):
        batch_size = x.size(0)
        x_size = x.size(1)

        Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

        return x.mul(Range).add(Mean)

    def encode(self, x: Variable) -> (Variable):
        if self.normIn:
            x = self.normalize(x)
        z = self.encoder(x)
        return z

    def decode(self, x: Variable) -> (Variable):
        z = self.decoder(x)
        return z

    def forward(self, x: Variable) -> (Variable):
        z = self.encode(x)
        if self.metaD:
            return z
        y = self.decode(z)
        
        return y
    
    # @torch.jit.export
    # def attr_to(self, device):
    #     self.Mean = self.Mean.to(device)
    #     self.Range = self.Range.to(device)


    # def _apply(self, fn):
    #     super(NN_AutoE, self)._apply(fn)
    #     self.Mean = fn(self.Mean)
    #     self.Range = fn(self.Range)
    #     return self
