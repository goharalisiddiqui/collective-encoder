# Data Modules

This module provides PyTorch Lightning `LightningDataModule` implementations that manage data loading, train/val/test/predict splitting, target normalization, data analysis, and batching. They bridge the gap between raw data readers (`datareaders`), feature extractors (`datasets`), and the PyTorch Lightning `Trainer`.

## Interface

All datamodules must inherit from `BaseDataModule` found in `base.py`.

### `BaseDataModule`

An abstract base class inheriting from both `CEModule` and `pytorch_lightning.LightningDataModule`.
It provides:
- Validation of split sizes (`train_size`, `validation_size`, `test_size`, `predict_size`) and batch sizes (`batch_size`, `val_batch_size`, `test_batch_size`, `predict_batch_size`).
- Compatibility checks against `_COMPATIBLE_DATAREADERS`, `_COMPATIBLE_DATASETS`, and `_COMPATIBLE_LABELERS`.
- Target scaler fitting and inverse transformations (`StandardScaler` or `MinMaxScaler`).
- Automatic execution of configured `data_analysers` on prepared splits.

## Registration & Resolution

DataModules are registered under the `"datamodules"` kind in [`__init__.py`](__init__.py) using `collective_encoder.common.registry`:
- **Kind definition**: `define_kind("datamodules", BaseDataModule)`
- **Resolution**: `get_module("datamodules", datamodule_type)` from `collective_encoder.common.registry` (there is no `resolver.py` in `datamodules/`).
- **Adding a new DataModule**: Subclass `BaseDataModule` and register it in `_BUILTINS` inside `datamodules/__init__.py` or from an external package via:
  ```python
  from collective_encoder.common.registry import register
  register("datamodules", "MY_DATAMODULE", "my_package.datamodules", "MyDataModule")
  ```

## Available DataModules

| Registry Key | Class | Description |
| :--- | :--- | :--- |
| `COORDINATES` (or `XTC`) | `CoordinatesDataModule` | Reads molecular trajectories (via `COMPRESSED_TRAJECTORY` or `MD17` readers), converts frames to `ase.Atoms`, and builds structural/graph datasets (`POSITIONS`, `DISTANCES`, `BONDGRAPH`, `SOAP`, etc.). |
| `COLVAR` | `ColvarsDataModule` | Reads collective variable tabular data (via `PLUMED_OUTPUT`) and wraps it in a `COLVAR` dataset. |
