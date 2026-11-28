# Mtomic

This module contains utilities to export trained models into the [Metatomic](https://metatomic.org/) ecosystem, enabling deployment of machine learning collective variables back into molecular dynamics engines (like LAMMPS or GROMACS via PLUMED).

## Components

- **`CEWrapper`**: A PyTorch `nn.Module` wrapper that packages the trained Collective Encoder (along with any necessary preprocessing datasets, like SOAP calculators or graph constructors) into a single module compatible with `metatomic.torch`. This wrapped model can then be exported via TorchScript for production use.
