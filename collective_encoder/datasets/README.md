# Datasets

This module contains PyTorch `Dataset` implementations. In the Collective Encoder framework, "datasets" serve as the feature extraction layer. They take the raw structural data (e.g., ASE Atoms objects from the datareader) and convert them into the specific numeric representations (features) required by the neural network architectures.

## Interface

All datasets must inherit from `BaseCEDataset` found in `base.py`.

### `BaseCEDataset`
This class inherits from both `torch.utils.data.Dataset` and `CEModule`.
Subclasses are required to implement:
- `_prepare_data()`: This is where the core logic of converting raw structures into PyTorch tensors/geometric data happens. It is executed once during initialization (or on the fly if overridden).
- `get_datapoint_shape()`: Returns the shape of a single instance/sample in the dataset, crucial for dynamically sizing neural network layers.

## Available Datasets

- `PositionsDataset`: Returns raw (flattened) atomic coordinates.
- `DistancesDataset`: Computes pairwise distances between atoms (can be full dense matrix or sparse).
- `SOAPDataset` / `SOAP_PS_Dataset`: Computes Smooth Overlap of Atomic Positions (SOAP) descriptors.
- `BondGraphDataset`: Converts molecular structures into PyTorch Geometric graph representations based on bonding.
- `BondGraphLatentDataset`: A specialized version that constructs graphs where nodes can carry latent vectors (useful for hierarchical/composed networks).
- `ColvarDataset`: Wraps flat tabular collective variable data for direct use by networks.
