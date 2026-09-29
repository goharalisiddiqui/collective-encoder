# Test Plotters

This module provides tools for plotting and analyzing model outputs during the testing phase. The plotters are instantiated during the evaluation loop and dynamically aggregate data batch by batch, generating plots and calculating analytical metrics (like disentanglement scores or correlations) at the end.

## Interface

All plotters must inherit from `BaseTestPlotter` in `base.py`.

### `BaseTestPlotter`
Inherits from `CEModule`.
Subclasses must implement two key methods:
- `add_batch(self, inp, latent, output, labels, meta)`: Accumulates data during the evaluation loop.
- `finish(self)`: Executed once the loop finishes; performs the actual computations, saves plots, and computes the final scalar metrics.

The plotters should return computed metrics via `get_metrics()`.

## Available Plotters
- `SimplePlotter`: Generic 1D/2D scatter plots of latent variables versus labels.
- `LatentCorrelationsPlotter`: Computes the linear and cross-correlation matrices between latent components and dataset labels.
- `DisentanglementPlotter`: Evaluates formal disentanglement metrics (e.g., MIG, Beta VAE score, Factor VAE score, SAP, Modularity, DCI) that assess how cleanly latent variables capture the underlying generative factors (labels).
