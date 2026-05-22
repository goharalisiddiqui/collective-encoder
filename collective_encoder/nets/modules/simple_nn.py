import logging
from typing import List

import torch
import torch.nn as nn

_log = logging.getLogger(__name__)


class SimpleNN(nn.Module):
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
            _log.info("%s --> %s (relu)", l[i], l[i + 1])
            encoder_layers.append(nn.Linear(l[i], l[i + 1]))
            encoder_layers.append(nn.ReLU(True))
            if batch_norm:
                encoder_layers.append(nn.BatchNorm1d(l[i + 1]))
                _log.info("  (batch_normalization layer)")
        self.encoder_hidden = nn.Sequential(*encoder_layers)

    def init_encoder_output(self):
        l = self.layers
        self.encoder_output = nn.Linear(l[-2], l[-1])
        _log.info("%s --> %s (feature space)", l[-2], l[-1])

    def forward(self, x):
        x = self.encoder_hidden(x)
        output = self.encoder_output(x)
        return output
