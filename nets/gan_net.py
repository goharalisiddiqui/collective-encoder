import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable
torch.manual_seed(0)
TORCH_PI = torch.acos(torch.zeros(1))*2


import pandas as pd
pd.set_option('display.max_columns', None) 



import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from statistics import mean as list_mean

from scipy.stats import multivariate_normal


class LITcollGAN(pl.LightningModule):
    def __init__(self, 
                 outname : str = './LITcollGAN_untitled/LITcollGAN_'):
        super().__init__()
        
        self.save_hyperparameters()
    
    def forward(self, x):
        return 0