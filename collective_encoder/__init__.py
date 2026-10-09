"""
Collective Encoder: A machine learning framework for molecular dynamics.

This package provides autoencoder-based architectures for creating surrogate models
that predict dynamics of molecular systems as time series data.
"""

import os

__version__ = "0.1.0"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs')