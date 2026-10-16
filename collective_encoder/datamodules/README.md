# Data Modules

This module provides PyTorch Lightning DataModules that manage data loading, splitting (train/val/test), and batching. They bridge the gap between underlying raw data formats (handled by `datareaders`), feature extractors (`datasets`), and the training loop.

## Interface

All datamodules must implement the PyTorch Lightning `LightningDataModule` interface and usually inherit from `BaseDataModule` found in `base.py`.

### `BaseDataModule`
An abstract base class inheriting from both `CEModule` and `LightningDataModule`.
It handles general datamodule functionality like defining train/val/test splits, batch sizes, and data loading mechanisms with multiple workers.
Subclasses are typically responsible for implementing the `setup()` method to instantiate the concrete dataset objects that will be fed to the DataLoaders.

## Available DataModules

- `CoordinatesDataModule`: Specifically meant for reading atomic coordinates/trajectories. It internally sets up data readers and passes them to PyTorch datasets (e.g. ones that compute bonds, angles, SOAP features).
- `ColvarDataModule`: Designed to read and manage collective variable data often output by sampling codes like PLUMED (e.g. tabular data in COLVAR format).
