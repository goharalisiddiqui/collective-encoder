# Backward compatibility module
# This module maintains backward compatibility for existing code that imports DefaultDatamodule from default.py
# The actual implementation has been moved to coordinates.py

import warnings
from collective_encoder.datamodules.coordinates import CoordinatesDataloader

# Issue a deprecation warning
warnings.warn(
    "Importing DefaultDatamodule from 'dataloaders.default' is deprecated. "
    "Please import CoordinatesDataloader from 'dataloaders.coordinates' or "
    "DefaultDatamodule from 'dataloaders' instead.",
    DeprecationWarning,
    stacklevel=2
)

# For backward compatibility
DefaultDatamodule = CoordinatesDataloader

__all__ = ["DefaultDatamodule"]