"""
Plotting utilities for data analysis and visualization during model testing.

Provides base classes and implementations for generating scatter plots,
correlation matrices, and evaluating disentanglement metrics.
"""


from collective_encoder.common.registry import define_kind, register, lazy_exports
from .base import BaseTestPlotter

define_kind("testplotters", BaseTestPlotter)

_BUILTINS = {
    "SimplePlotter": ("collective_encoder.testplotters.simple", "SimplePlotter"),
    "DisentanglementBetaMetric": ("collective_encoder.testplotters.disentanglement_metrics.beta", "DisentanglementBetaMetric"),
    "DisentanglementFactorMetric": ("collective_encoder.testplotters.disentanglement_metrics.factor", "DisentanglementFactorMetric"),
    "DisentanglementDCIMetric": ("collective_encoder.testplotters.disentanglement_metrics.dci", "DisentanglementDCIMetric"),
    "DisentanglementMIGMetric": ("collective_encoder.testplotters.disentanglement_metrics.mig", "DisentanglementMIGMetric"),
    "DisentanglementModularityMetric": ("collective_encoder.testplotters.disentanglement_metrics.modularity", "DisentanglementModularityMetric"),
    "DisentanglementSAPMetric": ("collective_encoder.testplotters.disentanglement_metrics.sap", "DisentanglementSAPMetric"),
    "LatentCorrelationsPlotter": ("collective_encoder.testplotters.latent_correlations", "LatentCorrelationsPlotter"),
    "DisentanglementPlotter": ("collective_encoder.testplotters.disentanglement", "DisentanglementPlotter"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("testplotters", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = ["BaseTestPlotter"] + [t for m, t in _BUILTINS.values()]
