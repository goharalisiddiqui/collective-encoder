from abc import ABC, abstractmethod

import pytorch_lightning as pl

from collective_encoder.common.module import CEModule


class KLDSchedulerBase(CEModule, ABC):
    _IDENTIFIER = ""
    _REQUIRED_ARGS = []

    def __init__(self, 
                args, 
                **kwargs):
        super().__init__(args=args, **kwargs)
    
    @abstractmethod
    def get_kld_max(self, plmodule: pl.LightningModule) -> float:
        raise NotImplementedError("get_kld_max must be implemented by subclasses")

    def on_validation_epoch_end(self, plmodule):
        """Optional hook that can be implemented by subclasses to update internal state at the end of each validation epoch."""
        pass