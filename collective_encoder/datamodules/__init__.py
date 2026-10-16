"""
Data modules for various molecular dynamics data formats.

This package provides PyTorch Lightning DataModules that manage data loading,
splitting, and batching. They wrap underlying datasets and datareaders
and prepare DataLoaders for the PyTorch Lightning Trainer.
"""


from collective_encoder.common.registry import define_kind, register, lazy_exports
from .base import BaseDataModule

define_kind("datamodules", BaseDataModule)

_BUILTINS = {
    "XTC":  ("collective_encoder.datamodules.coordinates", "CoordinatesDataModule"),
    "COORDINATES": ("collective_encoder.datamodules.coordinates", "CoordinatesDataModule"),
    "COLVAR": ("collective_encoder.datamodules.colvars", "ColvarsDataModule"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("datamodules", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = ["BaseDataModule"] + [t for m, t in _BUILTINS.values()]
