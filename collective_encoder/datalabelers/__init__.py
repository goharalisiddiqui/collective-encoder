"""
Label calculation modules for molecular features.

This module provides tools for creating target "labels" for molecular structures
and trajectories (e.g. Structure Factors, Steinhardt parameters, Dihedrals).
These labels are useful for supervision or enforcing physics-informed loss
calculations. Implementations inherit from `BaseLabeler`, `FrameLabeler`,
or `BatchLabeler`.
"""
from .base import BaseLabeler
from .structure_factor import StaticStructureFactorLabeler
from .debye import DebyeStructureFactorLabeler
from .steinhardt import SteinhardtOrderParameterLabeler
from .concat import ConcatLabeler
from .from_external import FromExternalLabeler
from .resolver import get_labeler

__all__ = [
    "BaseLabeler",
    "StaticStructureFactorLabeler",
    "DebyeStructureFactorLabeler",
    "SteinhardtOrderParameterLabeler",
    "ConcatLabeler",
    "FromExternalLabeler",
    "get_labeler",
]
