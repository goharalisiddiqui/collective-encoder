"""
Plotting utilities for data analysis and visualization.

This module provides various classes derived from `BaseDataAnalyser` to perform
exploratory data analysis. The analysers receive data from the datamodules,
compute relevant statistics or visualizations (e.g., histograms, graph structures,
dihedral angle distributions), and save the results to disk. Subclasses must
implement the `write_data` method defined in `base.py`.
"""


from collective_encoder.common.registry import define_kind, register, lazy_exports
from .base import BaseDataAnalyser

define_kind("dataanalysers", BaseDataAnalyser)

_BUILTINS = {
    "LABELS":     ("collective_encoder.dataanalysers.labels",    "LabelsAnalyser"),
    "DATAPOINTS":     ("collective_encoder.dataanalysers.datapoints",    "DatapointsAnalyser"),
    "GRAPH_DATAPOINTS": ("collective_encoder.dataanalysers.graphs", "GraphDatapointsAnalyser"),
    "LABELS_DIHEDRAL": ("collective_encoder.dataanalysers.dihedrals", "LabelsDihedralAnalyser"),
    "DATAPOINTS_DIHEDRAL": ("collective_encoder.dataanalysers.dihedrals", "DatapointsDihedralAnalyser"),
    "GRAPH_DATAPOINTS_DIHEDRAL": ("collective_encoder.dataanalysers.dihedrals", "GraphDatapointsDihedralAnalyser"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("dataanalysers", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = ["BaseDataAnalyser"] + [t for m, t in _BUILTINS.values()]