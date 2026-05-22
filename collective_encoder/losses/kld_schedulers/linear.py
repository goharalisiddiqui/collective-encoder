import pytorch_lightning as pl

from .base import KLDSchedulerBase


class KLDLinearScheduler(KLDSchedulerBase):
    _IDENTIFIER = "KLDLinearScheduler"
    _REQUIRED_ARGS = ['start_value', 'end_value', 'start_epoch', 'end_epoch']

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__(args=args, **kwargs)
        if self.start_epoch >= self.end_epoch:
            self.raise_error("start_epoch must be less than end_epoch")
    
    def get_kld_max(self, plmodule: pl.LightningModule) -> float:
        epoch = plmodule.current_epoch

        if epoch < self.start_epoch:
            return self.start_value
        elif epoch > self.end_epoch:
            return self.end_value
        else:
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            return self.start_value + progress * (self.end_value - self.start_value)
