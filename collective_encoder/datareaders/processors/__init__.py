"""
Processors for data.
"""

from collective_encoder.common.registry import define_kind, register, lazy_exports
from .base import BaseDataProcessor

define_kind("dataprocessors", BaseDataProcessor)

_BUILTINS = {
    "CHUNK": ("collective_encoder.datareaders.processors.chunk", "ChunkProcessor"),
    "COARSE_GRAIN": ("collective_encoder.datareaders.processors.coarse_grain", "CoarseGrainProcessor"),
    "FILTER": ("collective_encoder.datareaders.processors.filter", "FilterProcessor"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("dataprocessors", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = ["BaseDataProcessor"] + [t for m, t in _BUILTINS.values()]