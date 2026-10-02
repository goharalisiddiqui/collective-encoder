# Data Labelers

This module provides tools for creating "labels" or auxiliary targets for atomic structures and trajectories. These labels can be used for downstream supervision, latent space disentanglement, or physics-informed loss calculations (e.g., Structure Factors, Steinhardt parameters, Dihedrals, Coordination numbers).

## Interface

All data labelers must implement the `BaseLabeler` interface found in `base.py`.

### `BaseLabeler`
The base class for calculating properties of the data. Often, subclasses will instead inherit from one of two specialized base classes depending on how the calculation is performed:
- `FrameLabeler`: For label computations that happen per-frame (i.e. snapshot by snapshot). The `calculate_frame()` method needs to be implemented.
- `BatchLabeler`: For label computations that operate efficiently across batched data in PyTorch tensors. The `calculate_batch()` method needs to be implemented.

**Key Concepts:**
- `calculate`: Returns calculated properties. For `FrameLabeler` it processes frames, and for `BatchLabeler` it processes tensors.
- Configurations like periodic boundary conditions (PBC), simulation box vectors, and neighbour lists can be specified or handled internally depending on the specific labeler implementation.

## Available Labelers

- `DummyLabeler`: A pass-through labeler, primarily for testing.
- `DistanceLabeler`: Computes pairwise distances between atoms.
- `DihedralLabeler`: Calculates dihedral angles for groups of four atoms.
- `CoordinationLabeler`: Calculates coordination numbers.
- `DebyeStructureFactorLabeler` & `StaticStructureFactorLabeler`: Calculate structure factors used in scattering physics.
- `SteinhardtOrderParameterLabeler`: Calculates rotational invariants (Steinhardt parameters) for identifying crystal structures.
- `ColumnSelectorLabeler`: Selects pre-existing columns or features from the data.
- `ConcatLabeler`: Concatenates the outputs of multiple independent labelers into a single label tensor.
