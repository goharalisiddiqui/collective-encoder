# Collective Encoder

**A modular, scalable framework for building surrogate machine learning models that predict the dynamics of molecular systems.**

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

- Python 3.12
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)
- (Optional) [Weights & Biases](https://wandb.ai/)

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/collective-encoder.git
cd collective-encoder
```

Set up your Python environment:

```bash
pip install -r requirements.txt  # Install dependencies
```
*If you use conda or venv, create/activate your environment beforehand.*

### Project Structure

```text
collective-encoder/
├── collective_encoder/
│   ├── common/            # Shared utilities, config checkers, modules
│   ├── dataloaders/       # Data loader classes for XTC, MD17, KMC, etc.
│   ├── datasets/          # Feature extraction for distances, positions, SOAP, etc.
│   ├── nets/              # Neural network architectures and encoders
│   ├── plotters/          # Tools for analyzing/visualizing latent spaces
│   ├── mtomic/            # Metatomic wrapper/export utilities
│   ├── config.py          # Config management
│   └── engine.py          # Main training orchestrator
├── examples/              # Example configs and SLURM scripts
├── CLAUDE.md              # Guidance for Claude Code users
├── README.md
└── requirements.txt
```

### Preparing Your Run

1. **Edit the configuration:**
   Copy and modify an example config (e.g., `examples/ala2_tutorial/ala2_config.yaml`). Set your architecture, data paths, training hyperparameters, etc.

2. **(Optional) Test with Debug Mode:**
   Before launching your full experiment, run:

   ```bash
   python engine.py --config path/to/config.yaml --debug
   ```

   This uses a smaller dataset and two epochs for rapid validation.

3. **Launch Full Training (Command Line):**

   ```bash
   python engine.py --config path/to/config.yaml
   ```

4. **Run on HPC with SLURM:**
   Use the provided example SLURM script:

   ```bash
   sbatch ./example_run.sh
   ```

5. **Check Results:**
   Outputs, logs, and checkpoints will appear in the `train_runs/` (or configured `outpath`) directory.

### Configuration File Explained

Key fields in your YAML config:

```yaml
network_name: VAE            # Architecture (e.g. VAE, AE, DVAE, ...)
data_name: XTC               # Data type (e.g. XTC, MD17, ...)
nepochs: 50                  # Training epochs
outpath: ./results           # Results directory
data_args:                   # Dataset/dataloader options
    tprfile: ./data/file.tpr
    xtcfile: ./data/file.xtc
    batch_size: 32
network_args:                # Architecture parameters
    network: [256, 128, 2]
    beta: 1.0
```
For full documentation, see `examples/ala2_tutorial/ala2_config.yaml`.

### YAML Config Checking (Best Practice)

Use the included utility to validate your config and catch typos or missing fields before a run:

```python
from collective_encoder.common.config_check import validate_required_fields, validate_duplicate_keys
validate_duplicate_keys('path/to/config.yaml')
import yaml
with open('path/to/config.yaml') as f:
    cfg = yaml.safe_load(f)
validate_required_fields(cfg)
```

## Development & Contributing

- Please open issues or pull requests for bug reports, improvements, or new features.
- Follow PEP8 and maintain docstrings. Tests for new features are highly recommended.
- When adding new datasets, inherit from the relevant base data loader and register in the config resolver.

## Citing This Work

If you use Collective Encoder in your research, please cite us (to be updated with DOI/preprint):

```
@software{collective_encoder_2026,
  author = {Your Name and Co-authors},
  title = {Collective Encoder: Surrogate Models for Molecular Dynamics},
  url = {https://github.com/your-username/collective-encoder},
  version = {0.1},
  year = {2026}
}
```

## Support & Questions

For setup or usage questions, open a GitHub issue or reach out to [your-email@institution.edu](mailto:your-email@institution.edu).

---

**Collective Encoder** — Empowering molecular ML for diverse data and deployable surrogates.
