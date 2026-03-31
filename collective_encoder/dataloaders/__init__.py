"""Data loaders for various molecular dynamics data formats."""

from collective_encoder.dataloaders.default import DefaultDatamodule
from collective_encoder.dataloaders.resolver import get_dataloader

__all__ = [
    "DefaultDatamodule",
    "get_dataloader"
]