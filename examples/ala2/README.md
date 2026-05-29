# Alanine Dipeptide (Ala2) Tutorial: Learning Conformational Dynamics

This tutorial demonstrates how to use the **collective-encoder** package to train a Variational Autoencoder (VAE) on molecular dynamics data, specifically for learning a low-dimensional representation of alanine dipeptide conformational dynamics.

## 🧬 What is Alanine Dipeptide?

Alanine dipeptide (Ala2) is a small peptide consisting of:
- **ACE-ALA-NME**: Acetyl-Alanine-N-Methylamide
- A classic system for studying protein folding dynamics
- Well-defined conformational states (α-helix, β-sheet regions)
- Fast conformational transitions suitable for ML analysis

## 🎯 Learning Objective

We'll train a neural network to:
1. **Compress** high-dimensional atomic coordinates into a 2D latent space
2. **Capture** the essential conformational dynamics
3. **Discover** the main conformational states automatically
4. **Visualize** the free energy landscape in reduced dimensions

## 📁 Tutorial Structure

```
ala2_tutorial/
├── data/
│   ├── ala2.tpr              # GROMACS topology file
│   └── ala2.xtc              # Molecular dynamics trajectory
├── config.yaml               # Training configuration
├── requirements.txt          # Python dependencies
├── run.sh                    # Main script to run the tutorial
├── run_cpu.sh                # Contains SLURM directives for CPU execution
├── run_gpu.sh                # Contains SLURM directives for GPU execution
└── README.md                 # This file
```

## 🚀 Quick Start

```bash
bash run.sh -c config.yaml
```

## 🔬 Understanding the Configuration

### **Data Selection**
```yaml
selection: "(resname ALA or resname ACE or resname NME) and not element H"
```
- Selects all heavy atoms from the peptide
- Excludes hydrogens (fast dynamics, less important for conformations)
- Excludes water molecules (focus on peptide internal dynamics)

### **Feature Representation**
```yaml
dataset_type: "DISTANCES"
dataset_args: # Defines two groups of atoms for distances calculation ()
    group1: "0:10"
    group2: "0:10"
```
- Uses **pairwise distances** between atom groups
- Translation and rotation invariant
- Captures internal geometry changes

### **Network Architecture**
```yaml
network: [128, 64, 2]  # 128 → 64 → 2D latent space
```
- **Input**: ~121 pairwise distances (11×11)
- **Hidden layers**: 128 → 64 neurons
- **Output**: 2D latent representation
- **Decoder**: Mirrors encoder to reconstruct input

## 📊 Expected Results

### **Training Output**
```
results/ala2/
├── checkpoints/
│   └── best.ckpt             # Best model checkpoint
├── out.txt                   # Raw output
├── lightning_logs            # PyTorch Lightning logs
├── LDplotter_plots           # Latent space visualizations
└── run.log                   # Training log with metrics
```

Happy learning! 🚀
