# Collective Encoder Examples

This directory contains examples and tutorials for using the collective-encoder package.

## Available Examples

### 🧬 Alanine Dipeptide Tutorial (`ala2_tutorial/`)

**What it demonstrates:**
- Training a Variational Autoencoder (VAE) on molecular dynamics data
- Learning 2D representations of protein conformational dynamics
- Analyzing and visualizing the learned latent space

**Key files:**
- `README.md` - Comprehensive tutorial guide
- `ala2_config.yaml` - Example configuration file
- `data/ala2.tpr` & `data/ala2.xtc` - Sample MD trajectory data
- `run_training.py` - Python training script
- `analyze_results.py` - Analysis and visualization script
- `quickstart.sh` - One-click complete workflow

**Quick start:**
```bash
cd ala2_tutorial/
./quickstart.sh
```

**What you'll learn:**
- How to configure the package for molecular data
- Different ways to run training (CLI vs Python)
- How to analyze and interpret results
- Customization options for different use cases

## Adding More Examples

To add a new example:

1. Create a new directory: `examples/my_example/`
2. Include:
   - `README.md` - Tutorial documentation
   - `config.yaml` - Configuration file
   - `data/` - Sample data (if small)
   - Python scripts for training/analysis
3. Update this overview file

## Getting Help

- Read the main package documentation: `../../CLAUDE.md`
- Check the tutorial READMEs for specific guidance
- Examine configuration files for parameter explanations