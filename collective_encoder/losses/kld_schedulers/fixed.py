import pytorch_lightning as pl

from .base import KLDSchedulerBase

class KLDFixedScheduler(KLDSchedulerBase):
    _IDENTIFIER = "KLDFixedScheduler"
    _OPTIONAL_ARGS = {
        "value": 0.0,  # Fixed value for kld_max throughout training
    }

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__(args=args, **kwargs)
    
    def get_kld_max(self, plmodule: pl.LightningModule) -> float:
        return self.value