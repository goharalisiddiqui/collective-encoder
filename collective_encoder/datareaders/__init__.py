"""
Data readers for processing raw molecular data files.

This module contains classes that read raw data files from disk
(e.g., molecular trajectories like `.xtc`, or tabular outputs like PLUMED files)
into memory in a format that datamodules and datalabelers can process.
"""


from collective_encoder.common.registry import define_kind, register, lazy_exports
from .base import BaseDataReader
import collective_encoder.datareaders.processors

define_kind("datareaders", BaseDataReader)

_BUILTINS = {
    "COMPRESSED_TRAJECTORY": ("collective_encoder.datareaders.compressed_trajectory", "CompressedTrajectoryReader"),
    "XTC":                   ("collective_encoder.datareaders.compressed_trajectory", "CompressedTrajectoryReader"), # For backward compatibility
    "PLUMED_OUTPUT":         ("collective_encoder.datareaders.plumed_output",         "PlumedOutputReader"),
    "MD17":                  ("collective_encoder.datareaders.datasets.md17",         "MD17Reader"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("datareaders", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = ["BaseDataReader"] + [t for m, t in _BUILTINS.values()]