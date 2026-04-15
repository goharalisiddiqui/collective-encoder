"""Data loaders for various molecular dynamics data formats."""

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