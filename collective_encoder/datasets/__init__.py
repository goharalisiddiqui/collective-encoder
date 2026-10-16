"""
Dataset classes for different molecular feature representations.

In this framework, datasets act as the feature extraction layer.
They accept raw structural data (e.g. from datareaders) and convert it into
tensors or graph structures required by the downstream neural network architectures.
"""


from collective_encoder.common.registry import define_kind, register, lazy_exports
from .base import BaseDataset

define_kind("datasets", BaseDataset)

_BUILTINS = {
    "DEFAULT": ("collective_encoder.datasets.positions", "PositionsDataset"),
    "POSITIONS": ("collective_encoder.datasets.positions", "PositionsDataset"),
    "DISTANCES": ("collective_encoder.datasets.distances", "DistancesDataset"),
    "BONDGRAPH": ("collective_encoder.datasets.bondgraph", "BondGraphDataset"),
    "BONDGRAPH_LATENT": ("collective_encoder.datasets.bondgraph_latent", "BondGraphLatentDataset"),
    "SOAP": ("collective_encoder.datasets.soap", "SOAPDataset"),
    "SOAP_PS": ("collective_encoder.datasets.soap_ps", "SoapPowerSpectrumDataset"),
    "COLVAR": ("collective_encoder.datasets.colvar", "ColvarDataset"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("datasets", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = ["BaseDataset"] + [t for m, t in _BUILTINS.values()]