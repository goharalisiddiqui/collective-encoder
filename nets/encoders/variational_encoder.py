from typing import List

import torch
import torch.nn as nn

class VariationalNN(nn.Module):
    def __init__(self,
                 layers: List[int],
                 batch_norm: bool = False,
                 ):
        super().__init__()
        self.layers = layers
        self.batch_norm = batch_norm
        self.init_encoder()

    def init_encoder(self):
        self.init_encoder_layers()
        self.init_encoder_output()

    def init_encoder_layers(self):
        l = self.layers
        batch_norm = self.batch_norm
        encoder_layers = []
        for i in range(len(l) - 2):
            print(l[i], " --> ", l[i + 1], end=" ")
            encoder_layers.append(nn.Linear(l[i], l[i + 1]))
            encoder_layers.append(nn.ReLU(True))
            print("(relu)")
            if batch_norm:
                encoder_layers.append(nn.BatchNorm1d(l[i + 1]))
                print("(batch_normalization layer)")
        self.encoder_hidden = nn.Sequential(*encoder_layers)

    def init_encoder_output(self):
        l = self.layers
        self.encoder_mu = nn.Linear(l[-2], l[-1])
        print(l[-2], " --> ", l[-1], end=" ")
        print("(mu for latent space)")
        self.encoder_logvar = nn.Linear(l[-2], l[-1])
        print("  ", " \\--> ", l[-1], end=" ")
        print("(logvar for latent space)\n\n")

    def forward(self, x):
        x = self.encoder_hidden(x)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar
