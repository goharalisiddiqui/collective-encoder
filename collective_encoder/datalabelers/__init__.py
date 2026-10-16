"""
Label calculation modules for molecular features.

This module provides tools for creating target "labels" for molecular structures
and trajectories (e.g. Structure Factors, Steinhardt parameters, Dihedrals).
These labels are useful for supervision or enforcing physics-informed loss
calculations. Implementations inherit from `BaseLabeler`, `FrameLabeler`,
or `BatchLabeler`.
"""


from collective_encoder.common.registry import define_kind, register, lazy_exports
from .base import BaseLabeler

define_kind("datalabelers", BaseLabeler)

_BUILTINS = {
    "DUMMY":                        ("collective_encoder.datalabelers.dummy",            "DummyLabeler"),
    "COORDINATION":                 ("collective_encoder.datalabelers.coordination",     "CoordinationCountLabeler"),
    "DISTANCE":                     ("collective_encoder.datalabelers.distance",         "DistanceValueLabeler"),
    "DIHEDRAL":                     ("collective_encoder.datalabelers.dihedral",         "DihedralValueLabeler"),
    "COLUMN_SELECTOR":              ("collective_encoder.datalabelers.column_selector",  "ColumnSelectorLabeler"),
    "STRUCTURE_FACTOR":             ("collective_encoder.datalabelers.structure_factor", "StaticStructureFactorLabeler"),
    "DEBYE_STRUCTURE_FACTOR":       ("collective_encoder.datalabelers.debye",            "DebyeStructureFactorLabeler"),
    "STEINHARDT_ORDER_PARAMETER":   ("collective_encoder.datalabelers.steinhardt",       "SteinhardtOrderParameterLabeler"),
    "CONCAT":                       ("collective_encoder.datalabelers.concat",           "ConcatLabeler"),
    "FROM_EXTERNAL":                ("collective_encoder.datalabelers.from_external",    "FromExternalLabeler"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("datalabelers", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = ["BaseLabeler"] + [t for m, t in _BUILTINS.values()]