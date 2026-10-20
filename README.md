# Collective Encoder

**A modular, scalable framework for building machine learning models that encode collective motions in atomistic systems.**

---

## Overview

Collective Encoder is an open-source toolkit designed to accelerate research in molecular simulation, data-driven physics, and surrogate modeling. By abstracting over various autoencoder architectures (VAE/AE/DVAE/EDVAE, etc.) and data types (GROMACS XTC, MD17, COLVAR, KMC, XYZ), it enables researchers to rapidly prototype and train models, explore latent spaces, and deploy metatomic-compatible surrogates for further simulations or enhanced sampling.

## Features

- 🔧 **Configurable via YAML:** All experiments are driven by a simple, readable config file—change architectures, datasets, and training parameters with ease.
- 🧬 **Multiple Model Architectures:** Plug-and-play architectures such as VAE, AE, DVAE, EDVAE, GMVAE, VAEGAN, and BGE.
- 🧑‍🔬 **Supports Major Molecular Data Formats:** Includes loaders for GROMACS trajectories, MD17, COLVAR, KMC, and XYZ—add more with minimal code.
- ⚡ **PyTorch Lightning Integration:** Lightning modules ensure reproducibility, effective training management, and compatibility with modern tools (e.g., Weights & Biases).
- 🏁 **SLURM/HPC Friendly:** Scripts and debug modes make it painless to run jobs in cluster environments.
- 📈 **Automatic Checkpointing & Logging:** Includes built-in logging, early stopping, and model versioning.
- 📤 **Metatomic Model Export:** Export trained models for subsequent use in production or workflows using the metatomic format.

## Quick Start

### Prerequisites

- Python 3.10
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)
- (Optional) [Weights & Biases](https://wandb.ai/)

### Installation

Clone the repository:

```bash
git clone https://github.com/goharalisiddiqui/collective-encoder.git --branch v0.1.0
pip install -e ./collective-encoder  # Install as editable package
```

Via pip:

```bash
pip install git+https://github.com/goharalisiddiqui/collective-encoder.git@v0.1.0
```
*If you use conda or venv, create/activate your environment beforehand.*

### Project Structure

The codebase is organized into modular subpackages for clarity and extensibility. Each subpackage contains:
- `base.py`: Defines the abstract base class and interface for that plugin kind (inheriting from `CEModule`).
- `__init__.py`: Declares the plugin kind via `define_kind()`, registers all built-in implementations in the centralized registry (`collective_encoder.common.registry`) via `register()`, and lazily exports classes via `lazy_exports()`.
- Concrete implementation modules (and subfolders for shared components).

The runner scripts (`trainer.py`, `prepare_dmod.py`, `tester.py`) and other modules resolve classes dynamically from string identifiers in the YAML configuration using `get_module(kind, name)` from `collective_encoder.common.registry`.

```text
collective-encoder/
├── collective_encoder/
│   ├── common/                        # Shared base class (CEModule), centralized registry, config checkers
│   ├── dataanalysers/                 # Exploratory dataset analysis and plotting tools
│   ├── datalabelers/                  # Label calculation modules for structures and trajectories
│   ├── datamodules/                   # Lightning DataModules for different data modalities (Coordinates, COLVAR)
│   ├── datareaders/                   # Data readers/parsers for raw formats (XTC, MD17, PLUMED) and processors
│   ├── datasets/                      # Feature extraction layers (positions, distances, bond graphs, SOAP, COLVAR)
│   ├── hyperparameter_optimization/   # Optuna integration for hyperparameter sweeps
│   ├── loggers/                       # Custom PyTorch Lightning loggers (CSVPlotLogger)
│   ├── losses/                        # Loss functions and KLD schedulers for training
│   ├── metrics/                       # Evaluation metrics for validation and testing
│   ├── models/                        # Deep learning (AE, VAE, DVAE, EDVAE, BGE) and classical (PCA, ICA, TICA) models
│   ├── testplotters/                  # Post-training evaluation, latent space visualization, and disentanglement metrics
│   ├── mtomic/                        # Metatomic wrapper/export utilities
│   ├── configs/                       # Default and debug YAML configuration templates
│   ├── ce_run_base.py                 # Shared runner setup and datamodule loading logic
│   ├── trainer.py                     # Main training orchestrator
│   ├── prepare_dmod.py                # Script to prepare and serialize datamodules
│   ├── tester.py                      # Script for model evaluation and testing
│   ├── utils.py                       # Common helper utilities
│   └── cli.py                         # Command-line interface for train, prepare, test, and optuna
├── examples/                          # Example systems with configs and SLURM scripts
└── README.md
```
Each subpackage is designed for modularity and extensibility. To add a new component (e.g., a new dataset type), subclass the corresponding base class (`BaseDataset`) and register it in `collective_encoder.common.registry` using `register("datasets", "MY_DATASET", "module.path", "MyDatasetClass")` (either in `datasets/__init__.py` for built-ins or from an external package/script).

*Note: For more detailed documentation on the design, interfaces, and registration for each subpackage, please refer to the `README.md` files located directly within their respective directories (e.g., `collective_encoder/datareaders/README.md`).*

## Development & Contributing

- The project is in early stages for any meaningful contribution, but feel free to fork and experiment. Contributions will be considered as the project matures.

## Support & Questions

Feel free to reach out to [goharalisiddiqui@gmail.com](mailto:goharalisiddiqui@gmail.com).

---

**Collective Encoder** — Empowering data-driven collective variables for molecular dynamics.
