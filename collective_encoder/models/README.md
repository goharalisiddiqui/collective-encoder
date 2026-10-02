# Models

This module defines the high-level machine learning models within the Collective Encoder framework. These models wrap PyTorch modules, Scikit-learn estimators, or custom implementations and interface directly with the main PyTorch Lightning `Trainer`.

## Interface

All model classes inherit from PyTorch Lightning's `LightningModule` and the framework's `CEModuleBase` (defined in `base.py`).

### `CEModelBase`
The abstract base class for all models. It defines the standard lifecycle methods for PyTorch Lightning, such as `training_step`, `validation_step`, and `test_step`. It also manages the instantiation of underlying neural networks (`nets`), loss functions, metrics, and optimizers.

Subclasses are expected to:
- Instantiate their core architectures in `__init__`.
- Define how input data is processed and passed to the loss functions/metrics in the step methods.

## Available Models

### Deep Learning Models (Neural Nets)
Located within the `neural_nets/` subfolder.
- `AENet`: Standard Autoencoder.
- `VAENet`: Variational Autoencoder.
- `BGENet` / `BGEV2Net`: Bond Graph Autoencoders for molecular graph data.
- `DVAENet` / `EDVAENet`: Distributed VAEs and Equivariant Distributed VAEs.

### Classical/Statistical Models
- `PCAModel`: Principal Component Analysis.
- `ICAModel`: Independent Component Analysis.
- `TICAModel`: Time-lagged Independent Component Analysis.
