# Datasets

This module contains PyTorch `Dataset` implementations. In the Collective Encoder framework, datasets serve as the feature extraction layer: they take raw structural or tabular data (e.g., `ase.Atoms` objects or NumPy arrays from a datareader) and convert them into tensors or graph objects required by downstream models.

## Interface

All datasets must inherit from `BaseDataset` found in `base.py` (alongside `torch.utils.data.Dataset` or PyTorch Geometric's `Dataset`).

### `BaseDataset`

Inherits from `CEModule` and defines the dataset interface:
- `__len__(self) -> int`: Returns the total number of samples in the dataset.
- `__getitem__(self, index: int) -> Any`: Returns the processed sample at `index` (typically a `(features, label)` tuple or a PyTorch Geometric `Data` object).
- `get_norm_data(self) -> np.ndarray`: Returns a 2-D array used by the datamodule to fit feature normalizers (`StandardScaler` / `MinMaxScaler`).
- `get_datapoint_shape(self) -> tuple`: Returns the shape of a single sample's features for dynamically sizing model layers.
- `_prepare_dataset(cls, dataset_args, datareader)` (classmethod): Hook that returns `(dataset_cls, dataset_args, dataloader_cls)` (e.g., returning PyTorch Geometric's `DataLoader` for graph datasets).

## Registration & Resolution

Datasets are registered under the `"datasets"` kind in [`__init__.py`](__init__.py) using `collective_encoder.common.registry`:
- **Kind definition**: `define_kind("datasets", BaseDataset)`
- **Resolution**: `get_module("datasets", dataset_type)` from `collective_encoder.common.registry` (or `get_dataset_cls_dl(dataset_type, dataset_args, datareader)` in `datasets/resolver.py`, which resolves the class via `get_module("datasets", ...)` and invokes `_prepare_dataset`).
- **Adding a new Dataset**: Subclass `BaseDataset` and register it in `_BUILTINS` inside `datasets/__init__.py` or externally via:
  ```python
  from collective_encoder.common.registry import register
  register("datasets", "MY_DATASET", "my_package.datasets", "MyDataset")
  ```

## Available Datasets

| Registry Key | Class | Description |
| :--- | :--- | :--- |
| `POSITIONS` (or `DEFAULT`) | `PositionsDataset` | Extracts flattened atomic Cartesian coordinates. |
| `DISTANCES` | `DistancesDataset` | Computes pairwise interatomic distances (full upper-triangular or selected pairs). |
| `BONDGRAPH` | `BondGraphDataset` | Converts molecular structures into PyTorch Geometric graphs with bond lengths, angles, and dihedrals. |
| `BONDGRAPH_LATENT` | `BondGraphLatentDataset` | Constructs bond graphs paired with pre-computed latent vectors. |
| `SOAP` | `SOAPDataset` | Computes Smooth Overlap of Atomic Positions (SOAP) descriptors. |
| `SOAP_PS` | `SoapPowerSpectrumDataset` | Computes SOAP power spectrum descriptors via `featomic`. |
| `COLVAR` | `ColvarDataset` | Wraps 2-D tabular collective variable arrays into `(data, label)` tensor pairs. |
