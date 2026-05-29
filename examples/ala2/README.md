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
├── out.txt                   # Training logs
└── lightning_logs/           # TensorBoard logs
```

### **What the Model Learns**
1. **Latent Space**: 2D representation of conformational states
2. **Free Energy Landscape**: Metastable states and transitions
3. **Collective Variables**: Automatic discovery of important coordinates

## 🎮 Interactive Analysis

After training, analyze your results:

```python
import torch
import matplotlib.pyplot as plt
from collective_encoder.nets import VAE

# Load trained model
model = VAE.load_from_checkpoint('results/ala2_vae/checkpoints/best.ckpt')

# Load data and encode to latent space
# latent_coords = model.encode(data)

# Plot 2D free energy landscape
# plt.scatter(latent_coords[:, 0], latent_coords[:, 1])
# plt.xlabel('Latent Dimension 1')
# plt.ylabel('Latent Dimension 2')
# plt.title('Ala2 Conformational Landscape')
```

## 🔧 Customization Options

### **Different Network Types**
```yaml
network_name: AE        # Standard Autoencoder
network_name: DVAE      # Dynamical VAE
network_name: EDVAE     # Enhanced Dynamical VAE
```

### **Different Features**
```yaml
dataset_type: "POSITIONS"   # Raw atomic positions
dataset_type: "SOAP"        # SOAP descriptors
```

### **Monitoring Specific Variables**
```yaml
label_dihedrals:
  - phi_2     # φ (phi) dihedral angle
  - psi_2     # ψ (psi) dihedral angle
```

## 📈 Performance Tips

### **Quick Testing**
- Use `--debug` flag for rapid iteration
- Start with small `dataset_size` (100-500 frames)
- Use `nogpu: True` for debugging on CPU

### **Production Training**
- Use full dataset (`dataset_size: 10000+`)
- Enable GPU acceleration (`nogpu: False`)
- Use Weights & Biases for experiment tracking (`wandb: True`)

### **Hyperparameter Tuning**
- **Learning rate**: Try 0.0001, 0.001, 0.01
- **Latent dimensions**: 2D for visualization, 3-10D for better reconstruction
- **Beta parameter**: 0.1 (disentangled), 1.0 (standard), 10.0 (compressed)

## 🎯 Expected Timeline

| Task | Debug Mode | Full Training |
|------|------------|---------------|
| Data Loading | ~10 seconds | ~30 seconds |
| Training | ~1 minute | ~5-15 minutes |
| Total Time | **~2 minutes** | **~20 minutes** |

## 🧠 Scientific Insights

This tutorial demonstrates key concepts:

1. **Dimensionality Reduction**: 121D distances → 2D latent space
2. **Unsupervised Learning**: No labels needed, discovers patterns automatically
3. **Collective Variables**: Learned coordinates capture slow conformational dynamics
4. **Free Energy Landscapes**: 2D visualization of thermodynamic states

## 🆘 Troubleshooting

### Common Issues:
- **CUDA errors**: Set `nogpu: True` in config
- **Memory issues**: Reduce `batch_size` and `dataset_size`
- **Convergence problems**: Try lower learning rate or different `beta`

### Getting Help:
```bash
# Check package installation
python -c "import collective_encoder; print('✓ Working')"

# Validate configuration
python -c "from collective_encoder.config import Config; Config('ala2_config.yaml').validate()"
```

Happy learning! 🚀