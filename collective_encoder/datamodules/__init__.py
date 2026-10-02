"""
Data modules for various molecular dynamics data formats.

This package provides PyTorch Lightning DataModules that manage data loading,
splitting, and batching. They wrap underlying datasets and datareaders
and prepare DataLoaders for the PyTorch Lightning Trainer.
"""

from collective_encoder.datamodules.coordinates import CoordinatesDataModule
from collective_encoder.datamodules.resolver import get_datamodule
# from collective_encoder.datamodules.colvar import ColvarDataloader
# from collective_encoder.datamodules.md17 import MD17Dataloader, MD17Data

__all__ = [
    "CoordinatesDataModule",
    "get_datamodule"
    # "ColvarDataloader",
    # "MD17Dataloader",
]   