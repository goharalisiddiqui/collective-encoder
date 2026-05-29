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

The codebase is organized into modular subpackages for clarity and extensibility. Each subpackage contains a file to define the main class and any necessary utilities. If there is shared code (e.g pytorch modules), it is placed in a subfolder with appropriate name. Each module folder also contains an `__init__.py` file, a `base.py` file for abstract base classes (it should define the main interface for that module), and a `resolver.py` file that provides a method to resolve the modules based on string identifiers supplied in the config file. The main training script (`trainer.py`) and other modules use these resolvers to instantiate the appropriate classes based on the configuration.

```text
collective-encoder/
├── collective_encoder/
│   ├── common/            # Shared utilities, config checkers, root module for common functionality (e.g. logging)
│   ├── datamodules/       # Datamodules classes for different data modalities (Coordinates, COLVAR files etc.)
│   ├── datareaders/       # Data readers/parsers for different file formats (XTC, XYZ, COLVAR)
│   ├── datasets/          # Feature extraction for distances, positions, SOAP, etc.
│   ├── nets/              # Neural network architectures
│   ├── losses/            # Loss functions for training
│   ├── metrics/           # Metrics for evaluating model performance
│   ├── testplotters/      # Tools for analyzing/visualizing the results (e.g. latent space plots, reconstructions)
│   ├── mtomic/            # Metatomic wrapper/export utilities
│   ├── configs/           # Default config templates
│   └── trainer.py         # Main training orchestrator
│   └── prepare_dmod.py    # Script to prepare datamodules and save them for later use in training
│   └── test.py            # Script for model evaluation and testing
│   └── utils.py           # Some common utilities
│   └── cli.py             # Command line interface for data preparation, training and testing
├── examples/              # Example systems with configs and SLURM scripts for quick run
└── README.md
```
Each submodule is designed for modularity and extensibility. For example, to add a new dataset type, create a new class in `datasets/` that inherits from the base class and register it in the resolver.

## Development & Contributing

- The project is in early stages for any meaningful contribution, but feel free to fork and experiment. Contributions will be considered as the project matures.

## Support & Questions

Feel free to reach out to [goharalisiddiqui@gmail.com](mailto:goharalisiddiqui@gmail.com).

---

**Collective Encoder** — Empowering data-driven collective variables for molecular dynamics.
