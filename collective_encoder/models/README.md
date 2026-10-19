# Models

This module defines the machine learning models in the Collective Encoder framework, encompassing both deep neural networks (autoencoders, variational autoencoders, graph autoencoders) and classical statistical dimensionality reduction models (PCA, ICA, TICA).

## Interface

All models inherit from `CEModelBase` (`base.py`). Deep learning models trained with PyTorch Lightning further inherit from `CENetBase` (`neural_nets/base.py`), which subclasses both `CEModelBase` and `pytorch_lightning.LightningModule`.

### `CEModelBase` & `CENetBase`

- **`CEModelBase`**: Manages input normalization buffers (`Mean`, `Range`, `normIn`, `normSet`), encoding/decoding interfaces (`encoder`, `decoder`, `forward`), test plotter lifecycle, and checkpoint serialization.
- **`CENetBase`**: Adds PyTorch Lightning lifecycle methods (`training_step`, `validation_step`, `test_step`, `configure_optimizers`), loss/metric aggregation, and learning rate scheduling.

## Registration & Resolution

Models are registered under the `"models"` kind in [`__init__.py`](__init__.py) using `collective_encoder.common.registry`:
- **Kind definition**: `define_kind("models", CEModelBase)`
- **Resolution**: `get_module("models", model_type)` from `collective_encoder.common.registry` (there is no `resolver.py` in `models/`).
- **Adding a new Model**: Subclass `CEModelBase` (or `CENetBase`) and register it in `_BUILTINS` inside `models/__init__.py` or from an external package (such as `surrogate_dynamics`) via:
  ```python
  from collective_encoder.common.registry import register
  register("models", "MY_MODEL", "my_package.models", "MyModel")
  ```

## Available Models

### Deep Learning Models (`models/neural_nets/`)

| Registry Key | Class | Description |
| :--- | :--- | :--- |
| `AE` / `sAE` | `AE` / `sAE` | Standard and supervised Feedforward Autoencoders. |
| `VAE` / `sVAE` | `VAE` / `sVAE` | Standard and supervised Variational Autoencoders. |
| `DVAE` / `sDVAE` | `DVAE` / `sDVAE` | Disentangled / Distributed Variational Autoencoders. |
| `EDVAE` | `EDVAE` | Ensemble / Equivariant Disentangled Variational Autoencoder. |
| `BGE` | `BondGraphEncoderDecoder` | Bond Graph Autoencoder operating on molecular graph representations. |
| `BGE_V2` | `BondGraphEncoderDecoderV2` | Enhanced Bond Graph Autoencoder (v2). |

### Classical / Statistical Models

| Registry Key | Class | Description |
| :--- | :--- | :--- |
| `PCA` (or `PCAEncoder`) | `PCAModel` | Principal Component Analysis. |
| `ICA` (or `ICAEncoder`) | `ICAModel` | Independent Component Analysis. |
| `TICA` (or `TICAEncoder`) | `TICAModel` | Time-lagged Independent Component Analysis. |
