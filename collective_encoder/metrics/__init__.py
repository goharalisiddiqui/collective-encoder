"""
Evaluation metrics for the Collective Encoder framework.

This module provides purely observational metrics (like MAE and evaluation KLD)
that are tracked during validation or testing but do not contribute gradients
to the optimization process.
"""


from collective_encoder.common.registry import define_kind, register, lazy_exports
from .base import CEMetricBase

define_kind("metrics", CEMetricBase)

_BUILTINS = {
    "MAE":     ("collective_encoder.metrics.mae",    "CEMetricMAE"),
    "MAE_DICT": ("collective_encoder.metrics.mae_dict", "CEMetricMAEDict"),
    "KLD":     ("collective_encoder.metrics.kld",    "CEMetricKLD"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("metrics", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = ["CEMetricBase"] + [t for m, t in _BUILTINS.values()]
