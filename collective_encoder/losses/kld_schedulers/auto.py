import pytorch_lightning as pl

from .base import KLDSchedulerBase


class KLDAutoScheduler(KLDSchedulerBase):
    _IDENTIFIER = "KLDAutoScheduler"
    _OPTIONAL_ARGS = {
        "kld_initial": 0.0,
        "kld_max":1.0,
        "increase_factor": 0.1,
        "monitor_metric": "val_rec_loss",
    }
    """
    Automatically sets kld_max according to the value of the monitored metric at validation epoch end (e.g. validation reconstruction loss).
    If the monitored metric is improving (decreasing), keeps the current kld_max value to keep improving with same regularization.
    Otherwise, gradually relax the regularization by the specified factor, allowing the model to focus more on reconstruction.
    
    This allows for a dynamic balance between reconstruction and regularization during training, potentially leading to better convergence and performance.
    
    """
    
    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__(args=args, **kwargs)
        self.value = self.kld_initial
        self.prev_metric = float('inf')
        
        if self.kld_initial < 0 or self.kld_max <= 0:
            self.raise_error("kld_initial must be >= 0 and kld_max must be > 0")
        if self.increase_factor <= 0:
            self.raise_error("increase_factor must be > 0")

    def get_kld_max(self, plmodule: pl.LightningModule) -> float:
        return self.value

    def on_validation_epoch_end(self, plmodule: pl.LightningModule):
        monitor_metric = self.monitor_metric
    
        metric_value = plmodule.trainer.callback_metrics.get(monitor_metric)
        if metric_value is None:
            self.raise_error(f"Monitor metric '{monitor_metric}' not found in callback metrics. "
                            f" Found metrics: {list(plmodule.trainer.callback_metrics.keys())}. ")
        if metric_value < self.prev_metric:
            self.prev_metric = metric_value
        else:
            self.value = min(self.kld_max, self.value * (1 + self.increase_factor) 
                            if self.value > 0 else 
                            self.kld_initial + self.increase_factor 
                                                * (self.kld_max - self.value))
            if self.value != self.kld_max:
                self.log_info(f"Validation metric '{monitor_metric}' did not improve "
                            f"(current: {metric_value:.4f}, best: {self.prev_metric:.4f}). "
                            f"Increasing kld_max to {self.value:.4f} for next epoch.")

